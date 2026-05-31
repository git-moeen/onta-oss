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

from fastapi import APIRouter, Depends, Query

from cograph_client.api.deps import get_neptune_client
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.ontology_queries import type_uri
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import kg_graph_uri, tenant_graph_uri

router = APIRouter(prefix="/graphs/{tenant}/explore")

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
_STAT_ENTITY_COUNT = _STATS_NS + "entityCount"


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
            target = rng[len(TYPE_URI_PREFIX):] if rng.startswith(TYPE_URI_PREFIX) else None
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
        f"SELECT ?p (COUNT(DISTINCT ?e) AS ?cnt)\n"
        f'  (SUM(IF(STRSTARTS(STR(?o), "{ENTITY_URI_PREFIX}"), 1, 0)) AS ?rel)\n'
        f"FROM <{kg_graph}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> <{t_uri}> .\n"
        f"  ?e ?p ?o .\n"
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
        records.append({"p": p_uri, "cnt": cnt, "rel": rel})
    return entity_count, records


async def _read_type_stats(
    client: NeptuneClient, tenant_id: str, kg_name: str, t_uri: str
) -> tuple[int, list[dict]] | None:
    """Read precomputed stats for one type, or None if not materialized."""
    stats = _stats_graph_uri(tenant_id, kg_name)
    ec_q = f"SELECT ?ec FROM <{stats}> WHERE {{ <{t_uri}> <{_STAT_ENTITY_COUNT}> ?ec }}"
    pred_q = (
        f"SELECT ?pred ?cnt ?rel FROM <{stats}> WHERE {{\n"
        f"  ?s <{_STAT_FOR_TYPE}> <{t_uri}> ; <{_STAT_FOR_PRED}> ?pred ; <{_STAT_CNT}> ?cnt .\n"
        f"  OPTIONAL {{ ?s <{_STAT_REL}> ?rel }}\n"
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
        records.append({"p": r.get("pred", ""), "cnt": cnt, "rel": rel})
    return entity_count, records


async def recompute_kg_stats(client: NeptuneClient, tenant_id: str, kg_name: str) -> dict:
    """Recompute the stats graph for a KG in one whole-KG scan.

    Run at ingest time (or via the recompute endpoint / backfill). Replaces the
    KG's stats graph atomically and busts the in-memory cache for its types.
    """
    kg = kg_graph_uri(tenant_id, kg_name)
    stats = _stats_graph_uri(tenant_id, kg_name)
    scan = (
        f"SELECT ?type ?p (COUNT(DISTINCT ?e) AS ?cnt)\n"
        f'  (SUM(IF(STRSTARTS(STR(?o), "{ENTITY_URI_PREFIX}"), 1, 0)) AS ?rel)\n'
        f"FROM <{kg}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> ?type .\n"
        f"  ?e ?p ?o .\n"
        f'  FILTER(STRSTARTS(STR(?type), "{TYPE_URI_PREFIX}"))\n'
        f"}} GROUP BY ?type ?p"
    )
    _, rows = parse_sparql_results(await client.query(scan))

    entity_counts: dict[str, int] = {}
    triples: list[str] = []
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
        triples.append(
            f"<{node}> <{_STAT_FOR_TYPE}> <{type_uri_str}> ; "
            f"<{_STAT_FOR_PRED}> <{p_uri}> ; "
            f"<{_STAT_CNT}> {cnt} ; <{_STAT_REL}> {rel} ."
        )
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

    return {"types": len(entity_counts), "predicate_rows": len(triples) - len(entity_counts)}


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


@router.post("/kgs/{kg_name}/recompute-stats")
async def recompute_stats(
    kg_name: str,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Recompute the precomputed type-stats for a KG (called at ingest / backfill)."""
    return await recompute_kg_stats(client, tenant.tenant_id, kg_name)


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
                f"  ?e <{RDF_TYPE}> <{t_uri}>\n"
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
