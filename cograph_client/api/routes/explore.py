"""Explorer API — read-only endpoints that power the Cograph Explorer web app.

All data comes from existing Neptune graphs; no new infra required. These
endpoints add convenience (bundling, coverage %, search) on top of the raw
ontology + KG queries already used by the CLI.

Type summaries are served from a precomputed per-KG **stats graph** (one
integer triple set per type, written at ingest / via recompute) so a read is
a couple of tiny lookups instead of a full instance scan. If stats are missing
for a type (e.g. a KG ingested before this existed), the endpoint falls back
to a live scan so it always returns correct data.
"""

import asyncio
import hashlib
import time

import structlog
from fastapi import APIRouter, Depends, Query

from cograph_client.api.deps import get_neptune_client
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.ontology_queries import type_uri
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import kg_graph_uri, tenant_graph_uri
from cograph_client.resolver import drift_control

logger = structlog.stdlib.get_logger("cograph.explore")

router = APIRouter(prefix="/graphs/{tenant}/explore")

# Ontology core-slot marker (ADR 0003 §3 / Pass D) — written by
# ontology_queries.mark_core_slot as `<attr_uri> <onto/coreSlot> "true"`. A core
# slot is EXEMPT from the ADR 0004 drift floor (always declared), so the edge
# filter must know whether the upgraded predicate carries this marker.
_CORE_SLOT_PRED = "https://cograph.tech/onto/coreSlot"

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDF_PROPERTY = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
RDFS = "http://www.w3.org/2000/01/rdf-schema"
TYPE_URI_PREFIX = "https://cograph.tech/types/"
ENTITY_URI_PREFIX = "https://cograph.tech/entities/"
SYSTEM_PREDICATES: frozenset[str] = frozenset({
    f"{RDFS}#label",
    "https://cograph.tech/onto/ingested_at",
    "https://cograph.tech/onto/source",
})

# In-memory hot cache on top of the persistent stats graph. Read-heavy data
# that only changes on ingest; warmed on first read, busted on recompute.
_SUMMARY_TTL_SECONDS = 300.0
_summary_cache: dict[tuple[str, str, str], tuple[float, dict]] = {}

# --- Precomputed stats graph --------------------------------------------------
# Per (type, predicate): coverage count + entity-valued-object total, plus a
# per-type entity count. All integer literals → no string escaping needed.
_STATS_NS = "https://cograph.tech/stats/"
_STAT_FOR_TYPE = _STATS_NS + "forType"
_STAT_FOR_PRED = _STATS_NS + "forPred"
_STAT_CNT = _STATS_NS + "cnt"
_STAT_REL = _STATS_NS + "rel"
_STAT_TARGET = _STATS_NS + "targetType"
_STAT_ENTITY_COUNT = _STATS_NS + "entityCount"

# --- Primary-type attribution (COG-35, follow-up to ADR 0001 multi-typing) ----
# With multi-typing an instance can carry more than one asserted rdf:type (its
# `also_types` co-classifications, e.g. an entity asserted as both Employee and
# Guest). Grouping the stats scan by raw rdf:type would count such an instance
# once PER asserted type — double-counting it across the Explorer's per-type
# panels. ADR rule 5 says each instance is counted exactly once, under its
# "primary type" (the most-specific asserted type).
#
# This guard reproduces, in pure SPARQL over the KG graph alone, the choice made
# by resolver.er.types.primary_type for the data this system actually writes:
#
#   * Asserted co-types (`also_types`) are GENUINE INDEPENDENT classifications
#     (ADR rule 1) — siblings, never an asserted subtype + its ancestor (ancestors
#     are recovered via query-time subclass closure, never asserted). For equal-
#     depth siblings, primary_type tie-breaks to the LEXICOGRAPHICALLY SMALLEST
#     type name. Type URIs share the `…/types/` prefix, so URI string order equals
#     type-name order — the guard below picks the smallest-URI asserted type.
#
# An instance therefore contributes to ?type only when ?type is its smallest
# asserted type URI; the NOT EXISTS rejects every heavier co-type. For a single-
# typed instance the inner pattern can never bind a different `types/` type, so
# the NOT EXISTS is vacuously satisfied and behavior is byte-identical to before
# — which is the common case and must not regress.
#
# Caveat (documented, out of scope): this matches primary_type for INDEPENDENT
# co-types, the only multi-typing the resolver emits. It does NOT consult the
# subClassOf hierarchy (which lives in the ontology graph, not this KG scan), so
# if an asserted subtype + ancestor pair ever appeared it would attribute to the
# smaller URI rather than the deeper type. The resolver does not produce that
# shape today.
_PRIMARY_TYPE_GUARD = (
    f"  FILTER NOT EXISTS {{\n"
    f"    ?e <{RDF_TYPE}> ?type2 .\n"
    f'    FILTER(STRSTARTS(STR(?type2), "{TYPE_URI_PREFIX}") '
    f"&& STR(?type2) < STR(?type))\n"
    f"  }}\n"
)


def _target_from_entity_uri(obj: str) -> str | None:
    """Entity URIs are .../entities/{TargetType}/{id} → the target type leaf."""
    if not obj.startswith(ENTITY_URI_PREFIX):
        return None
    head = obj[len(ENTITY_URI_PREFIX):].split("/", 1)[0]
    return head or None


def _stats_graph_uri(tenant_id: str, kg_name: str) -> str:
    return kg_graph_uri(tenant_id, kg_name) + "/stats"


def _stat_node(type_uri_str: str, pred_uri: str) -> str:
    h = hashlib.md5(f"{type_uri_str}|{pred_uri}".encode()).hexdigest()
    return f"{_STATS_NS}n/{h}"


def _coverage(count: int, entity_count: int) -> float:
    if not entity_count:
        return 0.0
    return round(count / entity_count * 100, 1)


def _xsd_to_datatype(uri: str) -> str:
    if not uri:
        return "string"
    if uri.startswith(TYPE_URI_PREFIX):
        return uri[len(TYPE_URI_PREFIX):]
    last = uri.split("#")[-1] if "#" in uri else uri.split("/")[-1]
    return {"string": "string", "integer": "integer", "float": "float",
            "boolean": "boolean", "dateTime": "datetime", "Resource": "uri"}.get(last, "string")


def _assemble_summary(
    type_name: str,
    onto_row: dict,
    parent_type: str | None,
    entity_count: int,
    pred_records: list[dict],
    attr_defs: dict[str, dict[str, str]],
) -> dict:
    """Build the panel payload from per-predicate records (cnt + rel total).

    A predicate is a relationship if any of its objects are entities
    (``rel > 0``) or the ontology declares its range as a type. Target type
    and datatype come from the ontology definitions.
    """
    attributes = []
    relationships = []
    for r in pred_records:
        p_uri = r.get("p", "")
        if not p_uri or p_uri == RDF_TYPE or p_uri in SYSTEM_PREDICATES:
            continue
        cnt = r.get("cnt", 0)
        rel = r.get("rel", 0)
        defn = attr_defs.get(p_uri, {})
        name = defn.get("name") or p_uri.rstrip("/").split("/")[-1]
        rng = defn.get("range", "")
        cov = _coverage(cnt, entity_count)
        if rel > 0 or rng.startswith(TYPE_URI_PREFIX):
            # Prefer the ontology-declared range; fall back to the target type
            # captured from a sample object's entity URI.
            target = rng[len(TYPE_URI_PREFIX):] if rng.startswith(TYPE_URI_PREFIX) else None
            if not target:
                target = r.get("target") or None
            avg_degree = round(rel / entity_count, 2) if entity_count else 0.0
            relationships.append({
                "name": name,
                "predicate_uri": p_uri,
                "target_type": target,
                "count": cnt,
                "coverage_pct": cov,
                "avg_degree": avg_degree,
            })
        else:
            attributes.append({
                "name": name,
                "predicate_uri": p_uri,
                "datatype": _xsd_to_datatype(rng),
                "count": cnt,
                "coverage_pct": cov,
            })
    return {
        "name": type_name,
        "description": onto_row.get("comment", ""),
        "parent_type": parent_type,
        "entity_count": entity_count,
        "attributes": attributes,
        "relationships": relationships,
    }


async def _live_scan(client: NeptuneClient, kg_graph: str, t_uri: str) -> tuple[int, list[dict]]:
    """Fallback: one instance scan → (entity_count, per-predicate records).

    Used only when precomputed stats are absent for a type. The rdf:type row
    yields the entity count; ``rel`` is the entity-valued object total.
    """
    pred_sparql = (
        f"SELECT ?p (COUNT(DISTINCT ?e) AS ?cnt) (SAMPLE(?o) AS ?sample)\n"
        f'  (SUM(IF(STRSTARTS(STR(?o), "{ENTITY_URI_PREFIX}"), 1, 0)) AS ?rel)\n'
        f"FROM <{kg_graph}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> <{t_uri}> .\n"
        f"  ?e ?p ?o .\n"
        # Primary-type attribution: when ?e is multi-typed, count it only under
        # its smallest asserted type URI so the fallback matches the precomputed
        # stats (see _PRIMARY_TYPE_GUARD). Single-typed: vacuously satisfied.
        f"  FILTER NOT EXISTS {{\n"
        f"    ?e <{RDF_TYPE}> ?type2 .\n"
        f'    FILTER(STRSTARTS(STR(?type2), "{TYPE_URI_PREFIX}") '
        f'&& STR(?type2) < "{t_uri}")\n'
        f"  }}\n"
        f"}} GROUP BY ?p ORDER BY DESC(?cnt)"
    )
    _, rows = parse_sparql_results(await client.query(pred_sparql))
    entity_count = 0
    records: list[dict] = []
    for r in rows:
        p_uri = r.get("p", "")
        try:
            cnt = int(r.get("cnt", "0"))
        except ValueError:
            cnt = 0
        try:
            rel = int(r.get("rel", "0"))
        except ValueError:
            rel = 0
        if p_uri == RDF_TYPE:
            entity_count = cnt
            continue
        records.append({"p": p_uri, "cnt": cnt, "rel": rel,
                        "target": _target_from_entity_uri(r.get("sample", ""))})
    return entity_count, records


async def _read_type_stats(
    client: NeptuneClient, tenant_id: str, kg_name: str, t_uri: str
) -> tuple[int, list[dict]] | None:
    """Read precomputed stats for one type, or None if not materialized."""
    stats = _stats_graph_uri(tenant_id, kg_name)
    ec_q = f"SELECT ?ec FROM <{stats}> WHERE {{ <{t_uri}> <{_STAT_ENTITY_COUNT}> ?ec }}"
    pred_q = (
        f"SELECT ?pred ?cnt ?rel ?target FROM <{stats}> WHERE {{\n"
        f"  ?s <{_STAT_FOR_TYPE}> <{t_uri}> ; <{_STAT_FOR_PRED}> ?pred ; <{_STAT_CNT}> ?cnt .\n"
        f"  OPTIONAL {{ ?s <{_STAT_REL}> ?rel }}\n"
        f"  OPTIONAL {{ ?s <{_STAT_TARGET}> ?target }}\n"
        f"}}"
    )
    ec_raw, pred_raw = await asyncio.gather(client.query(ec_q), client.query(pred_q))
    _, ec_rows = parse_sparql_results(ec_raw)
    if not ec_rows:
        return None
    try:
        entity_count = int(ec_rows[0].get("ec", "0"))
    except ValueError:
        entity_count = 0
    _, pred_rows = parse_sparql_results(pred_raw)
    records: list[dict] = []
    for r in pred_rows:
        try:
            cnt = int(r.get("cnt", "0"))
        except ValueError:
            cnt = 0
        try:
            rel = int(r.get("rel", "0"))
        except ValueError:
            rel = 0
        target_uri = r.get("target", "")
        target = target_uri[len(TYPE_URI_PREFIX):] if target_uri.startswith(TYPE_URI_PREFIX) else None
        records.append({"p": r.get("pred", ""), "cnt": cnt, "rel": rel, "target": target})
    return entity_count, records


def _dedupe_undirected(pairs: list[tuple[str, str]]) -> list[dict]:
    """Collapse directed (src, tgt) type pairs into undirected edges.

    The overview graph is undirected, so A→B and B→A are one line. Sorting each
    pair keys both directions to the same bucket. Weight is constant for now —
    the overview encodes magnitude via node size, not edge weight.
    """
    by_pair: dict[tuple[str, str], dict] = {}
    for s, t in pairs:
        if not s or not t or s == t:
            continue
        a, c = sorted((s, t))
        by_pair[(a, c)] = {"source": a, "target": c, "weight": 70}
    return list(by_pair.values())


async def _read_edges_from_stats(
    client: NeptuneClient, tenant_id: str, kg_name: str
) -> list[tuple[str, str]] | None:
    """Type→type edges from the precomputed stats graph, or None if unmaterialized.

    Reads the SAME instance-derived ``targetType`` the per-type summary uses, so
    the overview and the detail view agree by construction. Returns None (not [])
    when no stat rows carry a target, so the caller can fall back to a live scan.
    """
    stats = _stats_graph_uri(tenant_id, kg_name)
    q = (
        f"SELECT DISTINCT ?src ?tgt FROM <{stats}> WHERE {{\n"
        f"  ?s <{_STAT_FOR_TYPE}> ?src ; <{_STAT_TARGET}> ?tgt .\n"
        f"}}"
    )
    _, rows = parse_sparql_results(await client.query(q))
    if not rows:
        return None
    out: list[tuple[str, str]] = []
    for r in rows:
        su, tu = r.get("src", ""), r.get("tgt", "")
        if su.startswith(TYPE_URI_PREFIX) and tu.startswith(TYPE_URI_PREFIX):
            out.append((su[len(TYPE_URI_PREFIX):], tu[len(TYPE_URI_PREFIX):]))
    return out


async def _read_edges_from_stats_drift(
    client: NeptuneClient, tenant_id: str, kg_name: str
) -> list[tuple[str, str]] | None:
    """ADR 0004 drift-gated variant of :func:`_read_edges_from_stats`.

    Same instance-derived ``targetType`` edges, but each is additionally tagged
    with its **support** (the ``_STAT_REL`` entity-valued-object total), the
    **source type's entity count** (``entityCount``), and whether the upgraded
    predicate is a **core slot** (the ``onto/coreSlot`` marker that
    ``mark_core_slot`` writes in the ontology graph, keyed by the predicate URI).

    An edge is kept only when ``drift_control.should_declare(support,
    source_count, is_core_slot)`` is True — i.e. it clears the coverage+count
    floor, or is a core slot (exempt). Below-floor edges (the
    ``ManufacturerPartNumber.issuedby -> Retailer`` 6%-coverage shape) are
    excluded from the overview. Returns None when no stat rows carry a target,
    so the caller can fall back to a live scan (which is unfiltered — a fresh KG
    without materialized stats predates drift control and must not regress).

    Only invoked when the ``OMNIX_DRIFT_CONTROL`` flag is ON; with the flag OFF
    the caller takes the unchanged :func:`_read_edges_from_stats` path.
    """
    stats = _stats_graph_uri(tenant_id, kg_name)
    onto = tenant_graph_uri(tenant_id)
    # Per targeted edge: source type, target type, support (rel), and the
    # predicate URI (so we can join the ontology core-slot marker). entityCount
    # for the source type lives on a separate stat triple, joined in by URI.
    q = (
        f"SELECT DISTINCT ?src ?tgt ?pred ?rel ?ec FROM <{stats}> WHERE {{\n"
        f"  ?s <{_STAT_FOR_TYPE}> ?src ; <{_STAT_TARGET}> ?tgt ; <{_STAT_FOR_PRED}> ?pred .\n"
        f"  OPTIONAL {{ ?s <{_STAT_REL}> ?rel }}\n"
        f"  OPTIONAL {{ ?src <{_STAT_ENTITY_COUNT}> ?ec }}\n"
        f"}}"
    )
    # Core-slot markers from the ontology graph, keyed by predicate (attr) URI.
    core_q = (
        f"SELECT DISTINCT ?attr FROM <{onto}> WHERE {{\n"
        f"  ?attr <{_CORE_SLOT_PRED}> ?v .\n"
        f"}}"
    )
    rows_raw, core_raw = await asyncio.gather(client.query(q), client.query(core_q))
    _, rows = parse_sparql_results(rows_raw)
    if not rows:
        return None
    _, core_rows = parse_sparql_results(core_raw)
    core_slots = {r.get("attr", "") for r in core_rows if r.get("attr")}

    out: list[tuple[str, str]] = []
    for r in rows:
        su, tu = r.get("src", ""), r.get("tgt", "")
        if not (su.startswith(TYPE_URI_PREFIX) and tu.startswith(TYPE_URI_PREFIX)):
            continue
        try:
            support = int(r.get("rel", "0") or "0")
        except ValueError:
            support = 0
        try:
            source_count = int(r.get("ec", "0") or "0")
        except ValueError:
            source_count = 0
        is_core = r.get("pred", "") in core_slots
        if not drift_control.should_declare(support, source_count, is_core):
            continue
        out.append((su[len(TYPE_URI_PREFIX):], tu[len(TYPE_URI_PREFIX):]))
    return out


async def _live_edge_scan(client: NeptuneClient, kg_graph: str) -> list[tuple[str, str]]:
    """Fallback: derive type→type edges straight from instance triples.

    Used when stats aren't materialized yet (KG ingested before stats existed).
    Target type comes from the object entity URI leaf, matching the summary.
    """
    q = (
        f"SELECT DISTINCT ?type ?o FROM <{kg_graph}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> ?type .\n"
        f"  ?e ?p ?o .\n"
        f'  FILTER(STRSTARTS(STR(?type), "{TYPE_URI_PREFIX}"))\n'
        f'  FILTER(STRSTARTS(STR(?o), "{ENTITY_URI_PREFIX}"))\n'
        f"}}"
    )
    _, rows = parse_sparql_results(await client.query(q))
    out: list[tuple[str, str]] = []
    for r in rows:
        tu = r.get("type", "")
        src = tu[len(TYPE_URI_PREFIX):] if tu.startswith(TYPE_URI_PREFIX) else ""
        if not src or "/" in src:  # skip nested URIs like .../attrs/x
            continue
        tgt = _target_from_entity_uri(r.get("o", ""))
        if tgt:
            out.append((src, tgt))
    return out


async def recompute_kg_stats(client: NeptuneClient, tenant_id: str, kg_name: str) -> dict:
    """Recompute the stats graph for a KG in one whole-KG scan.

    Run at ingest time (or via the recompute endpoint / backfill). Replaces the
    KG's stats graph atomically and busts the in-memory cache for its types.
    """
    kg = kg_graph_uri(tenant_id, kg_name)
    stats = _stats_graph_uri(tenant_id, kg_name)
    scan = (
        f"SELECT ?type ?p (COUNT(DISTINCT ?e) AS ?cnt) (SAMPLE(?o) AS ?sample)\n"
        f'  (SUM(IF(STRSTARTS(STR(?o), "{ENTITY_URI_PREFIX}"), 1, 0)) AS ?rel)\n'
        f"FROM <{kg}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> ?type .\n"
        f"  ?e ?p ?o .\n"
        f'  FILTER(STRSTARTS(STR(?type), "{TYPE_URI_PREFIX}"))\n'
        f"{_PRIMARY_TYPE_GUARD}"
        f"}} GROUP BY ?type ?p"
    )
    _, rows = parse_sparql_results(await client.query(scan))

    entity_counts: dict[str, int] = {}
    triples: list[str] = []
    # ADR 0004: raw type-level relationship declarations (type_uri, pred_uri,
    # support). Source counts are resolved AFTER the loop, once entity_counts is
    # fully populated (rows are grouped by type+pred, so a type's rdf:type count
    # row may come after its predicate rows). Only collected when the flag is ON.
    drift_enabled = drift_control.drift_control_enabled()
    rel_decls: list[tuple[str, str, int]] = []
    for r in rows:
        type_uri_str = r.get("type", "")
        leaf = type_uri_str[len(TYPE_URI_PREFIX):] if type_uri_str.startswith(TYPE_URI_PREFIX) else ""
        if not leaf or "/" in leaf:  # skip nested URIs like .../attrs/x
            continue
        p_uri = r.get("p", "")
        try:
            cnt = int(r.get("cnt", "0"))
        except ValueError:
            cnt = 0
        try:
            rel = int(r.get("rel", "0"))
        except ValueError:
            rel = 0
        if p_uri == RDF_TYPE:
            entity_counts[type_uri_str] = cnt
            continue
        node = _stat_node(type_uri_str, p_uri)
        stat = (
            f"<{node}> <{_STAT_FOR_TYPE}> <{type_uri_str}> ; "
            f"<{_STAT_FOR_PRED}> <{p_uri}> ; "
            f"<{_STAT_CNT}> {cnt} ; <{_STAT_REL}> {rel}"
        )
        target = _target_from_entity_uri(r.get("sample", ""))
        if target:
            stat += f" ; <{_STAT_TARGET}> <{TYPE_URI_PREFIX}{target}>"
        triples.append(stat + " .")
        # A type-level relationship is a predicate carrying entity-valued
        # objects (rel > 0). Those are the declarations the drift floor gates.
        if drift_enabled and rel > 0:
            rel_decls.append((type_uri_str, p_uri, rel))
    for type_uri_str, n in entity_counts.items():
        triples.append(f"<{type_uri_str}> <{_STAT_ENTITY_COUNT}> {n} .")

    if triples:
        body = "\n".join(triples)
        update = (
            f"DROP SILENT GRAPH <{stats}> ;\n"
            f"INSERT DATA {{ GRAPH <{stats}> {{\n{body}\n}} }}"
        )
    else:
        update = f"DROP SILENT GRAPH <{stats}>"
    await client.update(update)

    for key in [k for k in _summary_cache if k[0] == tenant_id and k[1] == kg_name]:
        _summary_cache.pop(key, None)

    result = {"types": len(entity_counts), "predicate_rows": len(triples) - len(entity_counts)}
    if drift_enabled:
        result["drift"] = await _build_drift_report(
            client, tenant_id, kg_name, rel_decls, entity_counts
        )
    return result


async def _build_drift_report(
    client: NeptuneClient,
    tenant_id: str,
    kg_name: str,
    rel_decls: list[tuple[str, str, int]],
    entity_counts: dict[str, int],
) -> dict:
    """Build + log the ADR 0004 drift report for a recompute pass (flag ON only).

    Resolves each raw relationship declaration into the ``{key, support,
    source_count, is_core_slot}`` shape ``drift_control.drift_report`` consumes:
    source_count is the source type's entity count; ``is_core_slot`` reads the
    ``onto/coreSlot`` marker (keyed by predicate URI) from the ontology graph.
    The report (effective floors + kept/quarantined split) is returned and
    logged so the drift dashboard / tenant changelog can read it.
    """
    onto = tenant_graph_uri(tenant_id)
    core_q = (
        f"SELECT DISTINCT ?attr FROM <{onto}> WHERE {{\n"
        f"  ?attr <{_CORE_SLOT_PRED}> ?v .\n"
        f"}}"
    )
    _, core_rows = parse_sparql_results(await client.query(core_q))
    core_slots = {r.get("attr", "") for r in core_rows if r.get("attr")}

    declarations: list[dict] = []
    for type_uri_str, pred_uri, support in rel_decls:
        type_leaf = type_uri_str[len(TYPE_URI_PREFIX):] if type_uri_str.startswith(TYPE_URI_PREFIX) else type_uri_str
        pred_leaf = pred_uri.rstrip("/").split("/")[-1]
        declarations.append({
            "key": f"{type_leaf}.{pred_leaf}",
            "support": support,
            "source_count": entity_counts.get(type_uri_str, 0),
            "is_core_slot": pred_uri in core_slots,
        })
    report = drift_control.drift_report(declarations)
    logger.info(
        "drift_report",
        tenant=tenant_id,
        kg=kg_name,
        floor_cov=report["floor_cov"],
        floor_count=report["floor_count"],
        kept=report["kept"],
        quarantined=report["quarantined"],
        quarantine=report["quarantine"],
    )
    return report


async def drop_kg_stats(client: NeptuneClient, tenant_id: str, kg_name: str) -> None:
    """Drop a KG's precomputed stats graph and evict its in-memory summaries.

    Called when a KG is deleted. The stats graph URI is derived from the KG
    name, so without this a KG later recreated under the same name would serve
    the deleted graph's stale counts until the next recompute lands.
    """
    stats = _stats_graph_uri(tenant_id, kg_name)
    await client.update(f"DROP SILENT GRAPH <{stats}>")
    for key in [k for k in _summary_cache if k[0] == tenant_id and k[1] == kg_name]:
        _summary_cache.pop(key, None)


# Background recompute: the whole-KG scan takes ~15s, longer than the ALB
# response timeout, so we never want a request to block on it. The Neptune
# client is an app-state singleton, so a fire-and-forget task is safe.
_bg_tasks: set = set()


async def _safe_recompute(client: NeptuneClient, tenant_id: str, kg_name: str) -> None:
    try:
        await recompute_kg_stats(client, tenant_id, kg_name)
    except Exception:
        pass  # best-effort; reads fall back to a live scan until it succeeds


def schedule_recompute(client: NeptuneClient, tenant_id: str, kg_name: str) -> None:
    """Fire-and-forget a stats recompute (used by the endpoint + ingest hook)."""
    task = asyncio.create_task(_safe_recompute(client, tenant_id, kg_name))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


@router.get("/kgs/{kg_name}/types/{type_name}/summary")
async def get_type_summary(
    kg_name: str,
    type_name: str,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Bundle all Explorer panel data for one type in one call.

    Serves from precomputed stats (fast); falls back to a live scan if stats
    for this type are not yet materialized. All percentages are relative to
    entity_count.
    """
    cache_key = (tenant.tenant_id, kg_name, type_name)
    cached = _summary_cache.get(cache_key)
    if cached is not None and (time.monotonic() - cached[0]) < _SUMMARY_TTL_SECONDS:
        return cached[1]

    onto_graph = tenant_graph_uri(tenant.tenant_id)
    kg_graph = kg_graph_uri(tenant.tenant_id, kg_name)
    t_uri = type_uri(type_name)

    onto_sparql = (
        f"SELECT ?label ?comment ?parent FROM <{onto_graph}> WHERE {{\n"
        f"  <{t_uri}> <{RDFS}#label> ?label .\n"
        f"  OPTIONAL {{ <{t_uri}> <{RDFS}#comment> ?comment }}\n"
        f"  OPTIONAL {{ <{t_uri}> <{RDFS}#subClassOf> ?parent }}\n"
        f"}}"
    )
    attr_def_sparql = (
        f"SELECT ?attr ?attrLabel ?range FROM <{onto_graph}> WHERE {{\n"
        f"  ?attr <{RDF_TYPE}> <{RDF_PROPERTY}> .\n"
        f"  ?attr <{RDFS}#domain> <{t_uri}> .\n"
        f"  ?attr <{RDFS}#label> ?attrLabel .\n"
        f"  OPTIONAL {{ ?attr <{RDFS}#range> ?range }}\n"
        f"}}"
    )

    # Ontology lookups (tiny) + precomputed stats, all concurrent.
    onto_raw, attr_def_raw, stats = await asyncio.gather(
        client.query(onto_sparql),
        client.query(attr_def_sparql),
        _read_type_stats(client, tenant.tenant_id, kg_name, t_uri),
    )

    _, onto_rows = parse_sparql_results(onto_raw)
    onto_row = onto_rows[0] if onto_rows else {}
    parent_uri = onto_row.get("parent", "")
    parent_type = parent_uri.rstrip("/").split("/")[-1] if parent_uri else None

    _, attr_def_rows = parse_sparql_results(attr_def_raw)
    attr_defs: dict[str, dict[str, str]] = {
        r["attr"]: {"name": r.get("attrLabel", ""), "range": r.get("range", "")}
        for r in attr_def_rows
        if r.get("attr")
    }

    if stats is not None:
        entity_count, pred_records = stats
    else:
        # Stats not materialized for this type — fall back to a live scan.
        entity_count, pred_records = await _live_scan(client, kg_graph, t_uri)

    result = _assemble_summary(type_name, onto_row, parent_type, entity_count, pred_records, attr_defs)
    _summary_cache[cache_key] = (time.monotonic(), result)
    return result


@router.get("/kgs/{kg_name}/type-edges")
async def get_type_edges(
    kg_name: str,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Undirected type→type edges for the Explorer overview graph.

    Derived from instance data (the precomputed stats graph, with a live-scan
    fallback) rather than the ontology's declared ``rdfs:range``. This keeps the
    overview consistent with the per-type detail view: a relationship that
    exists in the data but whose ontology range was never upgraded to a type
    URI (e.g. a predicate first seen as a primitive attribute) is now drawn in
    both places. Returns ``[{source, target, weight}]``.

    ADR 0004 (flag ``OMNIX_DRIFT_CONTROL``): when ON, the stats read also
    respects the support floor — a low-support drift edge (e.g.
    ``ManufacturerPartNumber.issuedby -> Retailer`` at 6% coverage) is excluded
    from the overview, while high-coverage and core-slot edges are kept. With
    the flag OFF the read is byte-identical to before (no filtering).
    """
    if drift_control.drift_control_enabled():
        edges = await _read_edges_from_stats_drift(client, tenant.tenant_id, kg_name)
    else:
        edges = await _read_edges_from_stats(client, tenant.tenant_id, kg_name)
    if edges is None:
        kg_graph = kg_graph_uri(tenant.tenant_id, kg_name)
        edges = await _live_edge_scan(client, kg_graph)
    return _dedupe_undirected(edges)


@router.post("/kgs/{kg_name}/recompute-stats")
async def recompute_stats(
    kg_name: str,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Schedule a recompute of the precomputed type-stats for a KG.

    Returns immediately; the ~15s whole-KG scan runs in the background so it
    never hits the ALB response timeout.
    """
    schedule_recompute(client, tenant.tenant_id, kg_name)
    return {"status": "scheduled", "kg": kg_name}


@router.post("/kgs/{kg_name}/er-rebuild")
async def er_rebuild(
    kg_name: str,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Second-pass entity resolution (MOE-22): collapse intra-batch fragments.

    Re-runs ER over the already-ingested KG so same-entity rows that couldn't
    see each other's index triples mid-batch now merge. Runs synchronously and
    returns per-type before/after counts (the merge volume is modest). Stale
    type-stats are recomputed in the background afterward so the Explorer
    reflects the new counts without blocking this response.
    """
    from cograph_client.resolver.er.rebuild import rebuild_kg

    instance_graph = kg_graph_uri(tenant.tenant_id, kg_name)
    report = await rebuild_kg(client, instance_graph)
    # Counts changed → refresh precomputed stats (best-effort, background).
    schedule_recompute(client, tenant.tenant_id, kg_name)
    return {"status": "complete", "kg": kg_name, **report}


@router.get("/search")
async def search_explorer(
    kg_name: str = Query(..., alias="kg"),
    q: str = Query(..., min_length=1),
    kind: str = Query("type", pattern="^(type|attr)$"),
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Search types or attributes by name substring.

    kind=type  — returns matching type names + their instance counts.
    kind=attr  — returns every type that has an attribute matching the query.
    """
    onto_graph = tenant_graph_uri(tenant.tenant_id)
    kg_graph = kg_graph_uri(tenant.tenant_id, kg_name)
    q_lower = q.lower()

    if kind == "type":
        sparql = (
            f"SELECT DISTINCT ?type ?label FROM <{onto_graph}> WHERE {{\n"
            f"  ?type <{RDF_TYPE}> <{RDFS}#Class> .\n"
            f"  ?type <{RDFS}#label> ?label .\n"
            f'  FILTER(CONTAINS(LCASE(STR(?label)), "{_esc(q_lower)}"))\n'
            f"}}"
        )
        _, rows = parse_sparql_results(await client.query(sparql))

        results = []
        for row in rows:
            type_name = row.get("label", "")
            if not type_name:
                continue
            t_uri = type_uri(type_name)
            count_sparql = (
                f"SELECT (COUNT(DISTINCT ?e) AS ?n) FROM <{kg_graph}> WHERE {{\n"
                f"  ?e <{RDF_TYPE}> <{t_uri}> .\n"
                # Primary-type attribution: a multi-typed instance is counted
                # only under its smallest asserted type URI (see
                # _PRIMARY_TYPE_GUARD). Single-typed: vacuously satisfied.
                f"  FILTER NOT EXISTS {{\n"
                f"    ?e <{RDF_TYPE}> ?type2 .\n"
                f'    FILTER(STRSTARTS(STR(?type2), "{TYPE_URI_PREFIX}") '
                f'&& STR(?type2) < "{t_uri}")\n'
                f"  }}\n"
                f"}}"
            )
            try:
                _, count_rows = parse_sparql_results(await client.query(count_sparql))
                entity_count = int(count_rows[0].get("n", "0")) if count_rows else 0
            except Exception:
                entity_count = 0
            results.append({"name": type_name, "entity_count": entity_count})
        return results

    # kind == "attr"
    sparql = (
        f"SELECT DISTINCT ?attrLabel ?type ?typeLabel FROM <{onto_graph}> WHERE {{\n"
        f"  ?attr <{RDF_TYPE}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> .\n"
        f"  ?attr <{RDFS}#label> ?attrLabel .\n"
        f"  ?attr <{RDFS}#domain> ?type .\n"
        f"  ?type <{RDFS}#label> ?typeLabel .\n"
        f'  FILTER(CONTAINS(LCASE(STR(?attrLabel)), "{_esc(q_lower)}"))\n'
        f"}}"
    )
    _, rows = parse_sparql_results(await client.query(sparql))
    return [
        {"attr_name": r.get("attrLabel", ""), "type_name": r.get("typeLabel", "")}
        for r in rows
        if r.get("attrLabel") and r.get("typeLabel")
    ]


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')
