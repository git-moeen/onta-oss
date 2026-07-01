"""Knowledge graph management — list, create, delete named graphs within a tenant.

All KGs share the tenant's ontology but have separate instance data.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cograph_client.api.deps import get_enrichment_job_store, get_neptune_client
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.enrichment.models import JobCategory, JobStatus
from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.ontology_queries import (
    get_type_attributes_query,
    type_uri,
)
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import (
    _escape_literal,
    kg_graph_uri,
    tenant_graph_uri,
)

router = APIRouter(prefix="/graphs/{tenant}/kgs")

OMNIX_ONTO = "https://cograph.tech/onto"
TYPE_URI_PREFIX = "https://cograph.tech/types/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
NAME_ATTRS = ("name", "title", "label", "headline")

# Predicate carrying a KG's precomputed triple count in the tenant metadata
# graph (next to kg_name/kg_description). Counting every triple in a KG graph
# is a full scan — seconds for a large KG — so `list_kgs` must NOT compute it
# live on each request (the Explorer's load was dominated by N serial scans).
# Instead the count is stored once and served as a tiny lookup inside the
# metadata query that already lists the KGs. It is (re)materialized lazily on
# read when absent and invalidated after ingest (see `invalidate_triple_count`,
# called from explore.recompute_kg_stats).
KG_TRIPLE_COUNT = f"{OMNIX_ONTO}/kg_triple_count"


def _kg_meta_uri(tenant_id: str, name: str) -> str:
    return f"https://cograph.tech/kgs/{tenant_id}/{name}"


async def _live_triple_count(
    client: "NeptuneClient", tenant_id: str, name: str
) -> int:
    """Full-scan COUNT(*) for one KG graph. Slow — fallback path only."""
    graph = kg_graph_uri(tenant_id, name)
    sparql = f"SELECT (COUNT(*) as ?c) FROM <{graph}> WHERE {{ ?s ?p ?o }}"
    try:
        _, rows = parse_sparql_results(await client.query(sparql))
        return int(rows[0].get("c", "0")) if rows else 0
    except Exception:
        return 0


async def _store_triple_count(
    client: "NeptuneClient", tenant_id: str, name: str, count: int
) -> None:
    """Persist a KG's triple count in the tenant metadata graph (best-effort)."""
    base = tenant_graph_uri(tenant_id)
    kg_uri = _kg_meta_uri(tenant_id, name)
    try:
        await client.update(
            f"WITH <{base}>\n"
            f"DELETE {{ <{kg_uri}> <{KG_TRIPLE_COUNT}> ?old }}\n"
            f"INSERT {{ <{kg_uri}> <{KG_TRIPLE_COUNT}> {int(count)} }}\n"
            f"WHERE {{ OPTIONAL {{ <{kg_uri}> <{KG_TRIPLE_COUNT}> ?old }} }}"
        )
    except Exception:
        pass


async def invalidate_triple_count(
    client: "NeptuneClient", tenant_id: str, name: str
) -> None:
    """Drop a KG's stored triple count so the next `list_kgs` recomputes it.

    Called after ingest (data changed → count stale). Best-effort: a failure
    just means the stale count lingers until the next write.
    """
    base = tenant_graph_uri(tenant_id)
    kg_uri = _kg_meta_uri(tenant_id, name)
    try:
        await client.update(
            f"WITH <{base}> DELETE WHERE {{ <{kg_uri}> <{KG_TRIPLE_COUNT}> ?old }}"
        )
    except Exception:
        pass

# Predicates the resolver attaches to every entity at ingest time.
# Always present, always 100%, drown out the actual columns the user
# cares about — hidden from /type usage by default, opt-in via
# ?include_system=true. Sourced from schema_resolver.py.
SYSTEM_PREDICATES: frozenset[str] = frozenset({
    "http://www.w3.org/2000/01/rdf-schema#label",
    "https://cograph.tech/onto/ingested_at",
    "https://cograph.tech/onto/source",
})


class KGCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    description: str = ""


class KGInfo(BaseModel):
    name: str
    description: str = ""
    triple_count: int = 0
    # Dashboard-summary stats, served from the durable per-KG stats store (no
    # Neptune scan on the hot path). Default to zeros/active for KGs whose row
    # isn't materialized yet — the next list lazily backfills it from the
    # precomputed stats graph (mirrors triple_count's lazy materialization).
    entity_count: int = 0
    edge_count: int = 0
    # "active" | "enriching" — derived live from the tenant's in-flight jobs.
    status: str = "active"
    stats_updated_at: Optional[str] = None


@router.get("", response_model=list[KGInfo])
async def list_kgs(
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
    job_store=Depends(get_enrichment_job_store),
):
    """List all knowledge graphs for a tenant, with dashboard-summary stats.

    Triple counts are read from the metadata graph (stored alongside the KG
    registration) in the SAME query that lists the KGs — no per-KG scan on the
    hot path. KGs with no stored count yet (legacy, or freshly invalidated
    after ingest) fall back to a live COUNT(*); those run in PARALLEL and are
    written back so the next read is again a single tiny lookup.

    Entity/edge counts come from the durable per-KG stats store (kept fresh by
    the shared write/refresh path) — a single relational read, no Neptune. Rows
    for KGs that predate the store are backfilled lazily from their existing
    precomputed stats graph the first time they're listed (the same lazy
    materialization pattern as triple counts). ``status`` is derived live from
    the tenant's in-flight enrichment jobs.
    """
    base = tenant_graph_uri(tenant.tenant_id)

    # One query: KG registrations + their stored triple counts.
    sparql = (
        f"SELECT ?name ?desc ?count FROM <{base}> WHERE {{"
        f"  ?kg <{OMNIX_ONTO}/kg_name> ?name ."
        f"  OPTIONAL {{ ?kg <{OMNIX_ONTO}/kg_description> ?desc }}"
        f"  OPTIONAL {{ ?kg <{KG_TRIPLE_COUNT}> ?count }}"
        f"}}"
    )
    raw = await client.query(sparql)
    _, bindings = parse_sparql_results(raw)

    # Preserve discovery order; dedupe defensively on name.
    entries: list[dict] = []
    seen: set[str] = set()
    for row in bindings:
        name = row.get("name", "")
        if not name or name in seen:
            continue
        seen.add(name)
        raw_count = row.get("count")
        count = (
            int(raw_count) if raw_count not in (None, "") and raw_count.isdigit() else None
        )
        entries.append({"name": name, "desc": row.get("desc", ""), "count": count})

    # Materialize any missing counts in parallel, then persist them.
    missing = [e for e in entries if e["count"] is None]
    if missing:
        counts = await asyncio.gather(
            *(_live_triple_count(client, tenant.tenant_id, e["name"]) for e in missing)
        )
        for e, c in zip(missing, counts):
            e["count"] = c
        await asyncio.gather(
            *(
                _store_triple_count(client, tenant.tenant_id, e["name"], e["count"])
                for e in missing
            ),
            return_exceptions=True,
        )

    stats_by_kg = await _kg_stats_for(client, tenant.tenant_id, [e["name"] for e in entries])
    enriching = await _enriching_kgs(job_store, tenant.tenant_id)

    out: list[KGInfo] = []
    for e in entries:
        s = stats_by_kg.get(e["name"])
        out.append(
            KGInfo(
                name=e["name"],
                description=e["desc"],
                triple_count=e["count"] or 0,
                entity_count=s.entity_count if s else 0,
                edge_count=s.edge_count if s else 0,
                status="enriching" if e["name"] in enriching else "active",
                stats_updated_at=s.updated_at.isoformat() if s else None,
            )
        )
    return out


async def _kg_stats_for(client: "NeptuneClient", tenant_id: str, kg_names: list[str]):
    """Return {kg_name: KgStats} from the durable store, backfilling misses.

    Steady state: one relational read for the whole tenant (no Neptune). KGs
    without a row yet are backfilled in parallel from their precomputed stats
    graph; a KG whose stats graph isn't materialized either gets a background
    recompute scheduled (which populates the store) and is served as zeros for
    now. Best-effort throughout — a store/Neptune hiccup degrades to zeros, it
    never fails the KG listing.
    """
    from cograph_client.api.routes.explore import backfill_kg_summary, schedule_recompute
    from cograph_client.graph.kg_stats_store import KgStats, get_kg_stats_store

    store = get_kg_stats_store()
    try:
        rows = await store.list_for_tenant(tenant_id)
    except Exception:  # noqa: BLE001 — degrade to no stats rather than 500
        rows = []
    by_kg: dict[str, KgStats] = {r.kg_name: r for r in rows}

    missing = [n for n in kg_names if n not in by_kg]
    if missing:
        backfilled = await asyncio.gather(
            *(backfill_kg_summary(client, tenant_id, n) for n in missing),
            return_exceptions=True,
        )
        for name, res in zip(missing, backfilled):
            if isinstance(res, KgStats):
                by_kg[name] = res
            elif not isinstance(res, Exception):
                # res is None: stats graph not materialized yet → schedule a
                # recompute so the store is populated for next time.
                try:
                    schedule_recompute(client, tenant_id, name)
                except Exception:  # noqa: BLE001
                    pass
    return by_kg


async def _enriching_kgs(job_store, tenant_id: str) -> set[str]:
    """KG names with an in-flight (queued/running) enrichment or discovery job."""
    try:
        jobs = await job_store.list_for_tenant(tenant_id)
    except Exception:  # noqa: BLE001
        return set()
    active = {JobStatus.queued, JobStatus.running}
    enriching = {JobCategory.enrichment, JobCategory.discovery}
    return {
        j.kg_name
        for j in jobs
        if j.status in active and j.category in enriching and j.kg_name
    }


@router.post("", response_model=KGInfo, status_code=201)
async def create_kg(
    body: KGCreate,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Create a new knowledge graph for a tenant.

    Idempotent-safe: guarded with ``FILTER NOT EXISTS`` so calling it twice
    never duplicates the registration triples and never clobbers an existing
    registration (or its ``kg_description``). This is the same registration the
    shared write path performs via ``ensure_kg_registered`` (ONTA-153) — here we
    additionally write the description, which only the explicit "New KG" flow
    supplies.

    On a re-POST of an existing KG the guarded INSERT no-ops; we then return the
    *existing* KGInfo (real description + triple count) rather than claiming an
    empty/zero KG, so the response never lies about a KG that may already hold
    real data.

    Safety: ``body.name`` is pattern-validated by ``KGCreate`` (``[a-zA-Z0-9_-]``)
    so it's URI-safe, but the free-text ``description`` and (defensively) the name
    are escaped via the canonical ``_escape_literal`` before going into a SPARQL
    literal — no statement-breakout on a ``"`` / ``\\`` / newline.
    """
    base = tenant_graph_uri(tenant.tenant_id)
    kg_uri = _kg_meta_uri(tenant.tenant_id, body.name)

    insert_lines = [
        f'    <{kg_uri}> <{OMNIX_ONTO}/kg_name> "{_escape_literal(body.name)}" .',
        f"    <{kg_uri}> <{KG_TRIPLE_COUNT}> 0 .",
    ]
    if body.description:
        insert_lines.append(
            f'    <{kg_uri}> <{OMNIX_ONTO}/kg_description> '
            f'"{_escape_literal(body.description)}" .'
        )
    insert_block = "\n".join(insert_lines)
    sparql = (
        f"WITH <{base}>\n"
        f"INSERT {{\n{insert_block}\n}}\n"
        f"WHERE {{\n"
        f"  FILTER NOT EXISTS {{ <{kg_uri}> <{OMNIX_ONTO}/kg_name> ?n }}\n"
        f"}}"
    )

    await client.update(sparql)

    # The INSERT is idempotent, so a re-POST no-ops it. Read the registration
    # back and report what's actually stored: a pre-existing KG keeps its real
    # description + triple count; a freshly-created one reads back as the values
    # we just wrote (description as given, count 0).
    read = (
        f"SELECT ?desc ?count FROM <{base}> WHERE {{\n"
        f"  <{kg_uri}> <{OMNIX_ONTO}/kg_name> ?n .\n"
        f"  OPTIONAL {{ <{kg_uri}> <{OMNIX_ONTO}/kg_description> ?desc }}\n"
        f"  OPTIONAL {{ <{kg_uri}> <{KG_TRIPLE_COUNT}> ?count }}\n"
        f"}}"
    )
    try:
        _, rows = parse_sparql_results(await client.query(read))
    except Exception:
        rows = []
    if rows:
        row = rows[0]
        raw_count = row.get("count")
        count = int(raw_count) if raw_count not in (None, "") and raw_count.isdigit() else 0
        return KGInfo(
            name=body.name,
            description=row.get("desc", "") or "",
            triple_count=count,
        )
    # Read-back failed (e.g. Neptune hiccup) — fall back to the values we wrote.
    return KGInfo(name=body.name, description=body.description, triple_count=0)


@router.delete("/{kg_name}")
async def delete_kg(
    kg_name: str,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Delete a knowledge graph and all its data."""
    base = tenant_graph_uri(tenant.tenant_id)
    graph = kg_graph_uri(tenant.tenant_id, kg_name)
    kg_uri = f"https://cograph.tech/kgs/{tenant.tenant_id}/{kg_name}"

    # Drop all triples in the KG graph
    await client.update(f"DROP SILENT GRAPH <{graph}>")

    # Drop the precomputed type-stats graph + in-memory summary cache. The stats
    # graph URI is derived from the KG name, so a KG recreated under the same
    # name would otherwise serve this deleted graph's stale counts.
    from cograph_client.api.routes.explore import drop_kg_stats
    await drop_kg_stats(client, tenant.tenant_id, kg_name)

    # Remove KG metadata
    await client.update(
        f"DELETE WHERE {{\n"
        f"  GRAPH <{base}> {{\n"
        f"    <{kg_uri}> ?p ?o .\n"
        f"  }}\n"
        f"}}"
    )

    # Purge stale examples from the example bank for this KG
    try:
        from cograph_client.nlp.example_bank import get_example_bank
        bank = get_example_bank()
        if bank and bank._examples:
            before = len(bank._examples)
            bank._examples = [e for e in bank._examples if e.kg_name != kg_name]
            removed = before - len(bank._examples)
            if removed > 0:
                bank.save()
                import structlog
                structlog.get_logger("cograph.kg").info(
                    "example_bank_purged", kg=kg_name, removed=removed,
                    remaining=len(bank._examples),
                )
    except Exception:
        pass  # Bank purge is best-effort, don't fail the delete

    # Clear this KG's rows from the spatio-temporal secondary index. Scoped to
    # (tenant_id, kg_name) so a sibling KG's geometry facts are untouched — the
    # whole reason the index carries a kg_name dimension. Best-effort: the
    # eventually-consistent derived index must never block the KG delete.
    try:
        from cograph_client.spatiotemporal.registry import get_spatiotemporal_index
        await get_spatiotemporal_index().clear(tenant.tenant_id, kg_name=kg_name)
    except Exception:
        pass  # Derived-index cleanup is best-effort, don't fail the delete

    return {"deleted": kg_name}


# ---------------------------------------------------------------------------
# Browsing: type counts and per-type attribute usage within a KG.
# Read-only convenience endpoints that power the shell's /types and /type
# commands. The ontology itself is tenant-global; what's per-KG is which
# types actually have instances and how often each attribute is populated.
# ---------------------------------------------------------------------------


class TypeCount(BaseModel):
    name: str
    entity_count: int
    # Spatio-temporal index markers, read from the precomputed stats graph
    # (recompute_kg_stats materializes them; absence = False). Spatial = the
    # type's instances carry geo:wktLiteral geometry; temporal = they carry
    # validity bounds or a complete start+end date pair.
    spatially_indexed: bool = False
    temporally_indexed: bool = False


class AttributeUsage(BaseModel):
    name: str
    datatype: str = "string"
    count: int


class RelationshipUsage(BaseModel):
    name: str
    target_type: str | None = None
    count: int


class EntitySample(BaseModel):
    uri: str
    label: str = ""


class TypeUsage(BaseModel):
    name: str
    description: str = ""
    parent_type: str | None = None
    entity_count: int
    attributes: list[AttributeUsage] = []
    relationships: list[RelationshipUsage] = []
    samples: list[EntitySample] = []


@router.get("/{kg_name}/type-counts", response_model=list[TypeCount])
async def list_type_counts(
    kg_name: str,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """List every type that has instances in this KG, sorted by entity count.

    Tenant-global ontology types with zero instances in this KG are not
    returned here — fetch them via /ontology/types if the caller needs the
    full schema.
    """
    graph = kg_graph_uri(tenant.tenant_id, kg_name)
    sparql = (
        f"SELECT ?type (COUNT(DISTINCT ?e) AS ?cnt) FROM <{graph}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> ?type .\n"
        f'  FILTER(STRSTARTS(STR(?type), "{TYPE_URI_PREFIX}"))\n'
        f"}} GROUP BY ?type ORDER BY DESC(?cnt)"
    )
    raw, index_flags = await asyncio.gather(
        client.query(sparql),
        _read_type_index_flags(client, tenant.tenant_id, kg_name),
    )
    _, bindings = parse_sparql_results(raw)
    out: list[TypeCount] = []
    for row in bindings:
        t = row.get("type", "")
        if not t.startswith(TYPE_URI_PREFIX):
            continue
        # Skip nested URIs like .../types/{Type}/attrs/{name} which aren't types
        leaf = t[len(TYPE_URI_PREFIX):]
        if "/" in leaf:
            continue
        try:
            count = int(row.get("cnt", "0"))
        except ValueError:
            count = 0
        spatial, temporal = index_flags.get(leaf, (False, False))
        out.append(TypeCount(
            name=leaf,
            entity_count=count,
            spatially_indexed=spatial,
            temporally_indexed=temporal,
        ))
    return out


async def _read_type_index_flags(
    client: NeptuneClient, tenant_id: str, kg_name: str
) -> dict[str, tuple[bool, bool]]:
    """Per-type (spatially_indexed, temporally_indexed) from the stats graph.

    The markers are materialized by ``recompute_kg_stats``; a KG whose stats
    were never recomputed (or whose types carry neither marker) simply yields
    no rows — every type then defaults to (False, False). Best-effort: the
    flags decorate the type list, so a stats-graph hiccup must not take down
    the endpoint that powers the Explorer rail.
    """
    # Local import: explore imports this module (locally) for the triple-count
    # invalidation hook, so a module-level import here would create a cycle.
    from cograph_client.api.routes.explore import (
        _STAT_SPATIAL,
        _STAT_TEMPORAL,
        _stats_graph_uri,
    )

    stats = _stats_graph_uri(tenant_id, kg_name)
    sparql = (
        f"SELECT ?type ?sp ?tp FROM <{stats}> WHERE {{\n"
        f"  {{ ?type <{_STAT_SPATIAL}> ?sp }} UNION {{ ?type <{_STAT_TEMPORAL}> ?tp }}\n"
        f"}}"
    )
    flags: dict[str, tuple[bool, bool]] = {}
    try:
        _, rows = parse_sparql_results(await client.query(sparql))
    except Exception:  # noqa: BLE001 — decoration only, never fail the list
        return flags
    for row in rows:
        t = row.get("type", "")
        if not t.startswith(TYPE_URI_PREFIX):
            continue
        leaf = t[len(TYPE_URI_PREFIX):]
        spatial, temporal = flags.get(leaf, (False, False))
        # Accept both boolean lexical forms ("true" and "1") — see _read_type_stats.
        if row.get("sp", "") in ("true", "1"):
            spatial = True
        if row.get("tp", "") in ("true", "1"):
            temporal = True
        flags[leaf] = (spatial, temporal)
    return flags


@router.get("/{kg_name}/types/{type_name}/usage", response_model=TypeUsage)
async def get_type_usage(
    kg_name: str,
    type_name: str,
    include_system: bool = False,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Per-type breakdown for one type in one KG.

    Combines the tenant-global ontology definition (attribute names,
    datatypes, parent type) with per-KG instance numbers (entity count,
    attribute usage, sample entities) so the caller doesn't have to make
    three round-trips and re-join the results client-side.
    """
    tenant_graph = tenant_graph_uri(tenant.tenant_id)
    kg_graph = kg_graph_uri(tenant.tenant_id, kg_name)
    t_uri = type_uri(type_name)

    # 1) Ontology definition for this type (tenant-global graph).
    onto_sparql = (
        f"SELECT ?label ?comment ?parent FROM <{tenant_graph}> WHERE {{\n"
        f"  <{t_uri}> <http://www.w3.org/2000/01/rdf-schema#label> ?label .\n"
        f"  OPTIONAL {{ <{t_uri}> <http://www.w3.org/2000/01/rdf-schema#comment> ?comment }}\n"
        f"  OPTIONAL {{ <{t_uri}> <http://www.w3.org/2000/01/rdf-schema#subClassOf> ?parent }}\n"
        f"}}"
    )
    _, onto_rows = parse_sparql_results(await client.query(onto_sparql))

    # Tenant-global attribute definitions for this type — gives us the
    # canonical name + datatype, which we'll join with per-KG usage counts.
    _, attr_def_rows = parse_sparql_results(
        await client.query(get_type_attributes_query(tenant_graph, type_name))
    )
    attr_def: dict[str, dict[str, str]] = {}
    for r in attr_def_rows:
        a_uri = r.get("attr", "")
        if not a_uri:
            continue
        attr_def[a_uri] = {
            "name": r.get("attrLabel", ""),
            "range": r.get("range", ""),
        }

    # 2) Entity count for this type within this KG.
    count_sparql = (
        f"SELECT (COUNT(DISTINCT ?e) AS ?n) FROM <{kg_graph}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> <{t_uri}>\n"
        f"}}"
    )
    _, count_rows = parse_sparql_results(await client.query(count_sparql))
    try:
        entity_count = int(count_rows[0].get("n", "0")) if count_rows else 0
    except ValueError:
        entity_count = 0

    if entity_count == 0 and not onto_rows:
        # Nothing in the ontology and nothing in the KG → 404 so the CLI can
        # tell the user "no such type" instead of silently returning zeros.
        raise HTTPException(
            status_code=404,
            detail=f"Type '{type_name}' not found in tenant ontology or KG '{kg_name}'",
        )

    # 3) Per-predicate usage in this KG. SAMPLE(?o) lets us classify
    # attribute (literal) vs. relationship (typed entity) without a second
    # round-trip per predicate.
    pred_sparql = (
        f"SELECT ?p (COUNT(DISTINCT ?e) AS ?cnt) (SAMPLE(?o) AS ?sample)\n"
        f"FROM <{kg_graph}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> <{t_uri}> .\n"
        f"  ?e ?p ?o .\n"
        f"  FILTER(?p != <{RDF_TYPE}>)\n"
        f"}} GROUP BY ?p ORDER BY DESC(?cnt)"
    )
    _, pred_rows = parse_sparql_results(await client.query(pred_sparql))

    attributes: list[AttributeUsage] = []
    relationships: list[RelationshipUsage] = []
    for r in pred_rows:
        p_uri = r.get("p", "")
        if not include_system and p_uri in SYSTEM_PREDICATES:
            continue
        try:
            cnt = int(r.get("cnt", "0"))
        except ValueError:
            cnt = 0
        sample = r.get("sample", "")
        defn = attr_def.get(p_uri, {})
        # Predicate name: prefer ontology label, fall back to URI tail.
        name = defn.get("name") or p_uri.rstrip("/").split("/")[-1]
        rng = defn.get("range", "")
        # Classify: object pointing into the entities/types namespace OR
        # ontology-declared range that's another type → relationship.
        is_rel = (
            sample.startswith("https://cograph.tech/entities/")
            or sample.startswith(TYPE_URI_PREFIX)
            or rng.startswith(TYPE_URI_PREFIX)
        )
        if is_rel:
            target: str | None = None
            if rng.startswith(TYPE_URI_PREFIX):
                target = rng[len(TYPE_URI_PREFIX):]
            elif sample.startswith("https://cograph.tech/entities/"):
                # Entity URIs are .../entities/{TypeName}/{slug}; pull the
                # type out so the CLI can render "industries → Industry"
                # even when the ontology hasn't declared a typed range.
                tail = sample[len("https://cograph.tech/entities/"):]
                head = tail.split("/", 1)[0]
                if head:
                    target = head
            relationships.append(
                RelationshipUsage(name=name, target_type=target, count=cnt)
            )
        else:
            attributes.append(
                AttributeUsage(
                    name=name,
                    datatype=_xsd_to_datatype(rng),
                    count=cnt,
                )
            )

    # 4) Up to 3 sample entities with a name-like label, picked by trying
    # the conventional label attributes in order. Cheap one-shot query.
    label_optionals = "\n".join(
        f'    OPTIONAL {{ ?e <{TYPE_URI_PREFIX}{type_name}/attrs/{a}> ?{a} }}'
        for a in NAME_ATTRS
    )
    label_vars = " ".join(f"?{a}" for a in NAME_ATTRS)
    sample_sparql = (
        f"SELECT ?e {label_vars} FROM <{kg_graph}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> <{t_uri}> .\n"
        f"{label_optionals}\n"
        f"}} LIMIT 3"
    )
    samples: list[EntitySample] = []
    try:
        _, sample_rows = parse_sparql_results(await client.query(sample_sparql))
        for r in sample_rows:
            uri = r.get("e", "")
            label = next((r[a] for a in NAME_ATTRS if r.get(a)), "")
            samples.append(EntitySample(uri=uri, label=label))
    except Exception:
        # Sample fetch is decorative; don't blow up the whole response if
        # the SPARQL chokes on something we didn't anticipate.
        samples = []

    onto_row = onto_rows[0] if onto_rows else {}
    parent = onto_row.get("parent", "")
    return TypeUsage(
        name=type_name,
        description=onto_row.get("comment", ""),
        parent_type=parent.rstrip("/").split("/")[-1] if parent else None,
        entity_count=entity_count,
        attributes=attributes,
        relationships=relationships,
        samples=samples,
    )


def _xsd_to_datatype(uri: str) -> str:
    if not uri:
        return "string"
    if uri.startswith(TYPE_URI_PREFIX):
        return uri[len(TYPE_URI_PREFIX):]
    last = uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]
    return {
        "string": "string",
        "integer": "integer",
        "float": "float",
        "boolean": "boolean",
        "dateTime": "datetime",
        "Resource": "uri",
    }.get(last, "string")
