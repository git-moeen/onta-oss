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
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, Query

from cograph_client.api.deps import get_neptune_client
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.ontology_queries import attr_uri, type_uri
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
# Instance relationship predicates are minted as `…/onto/<predName>` (see
# nlp/pipeline.py); the matching ontology attr/core-slot marker lives at
# `attr_uri(<sourceTypeLeaf>, <predName>)`. Strip this prefix to recover predName.
ONTO_PRED_PREFIX = "https://cograph.tech/onto/"
SYSTEM_PREDICATES: frozenset[str] = frozenset({
    f"{RDFS}#label",
    "https://cograph.tech/onto/ingested_at",
    "https://cograph.tech/onto/source",
})

# In-memory hot cache on top of the persistent stats graph. Read-heavy data
# warmed on first read, busted whenever the underlying counts change.
#
# The TTL is a staleness *backstop*, NOT the invalidation mechanism: every
# in-process mutation that changes a type's summary — ingest, ER rebuild, AND
# enrichment/dedupe apply — routes through `recompute_kg_stats` (via
# `schedule_recompute`), which explicitly evicts this cache for the affected KG
# (see below). So a short TTL bought nothing but extra Neptune round trips —
# every ~5 min an active Explorer session re-queried the stats graph for data
# that had not changed and would be evicted the moment it did. A longer backstop
# keeps tenant/KG switches served from memory across a working session while
# still self-healing if an external writer ever mutates the underlying graph
# without going through `recompute_kg_stats`.
_SUMMARY_TTL_SECONDS = 1800.0
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

# --- Drift history graph (COG-57) ---------------------------------------------
# The observe-only mode (ADR 0004 §7) computes the per-relationship coverage
# distribution on every recompute but only *logs* it (CloudWatch, 30-day
# retention) — so "collect enough data to set the floor from real data" was just
# log-scraping that ages out. COG-57 persists each recompute's distribution to a
# per-KG **drift-history named graph** instead: a durable, SPARQL-queryable store
# the floor can later be calibrated from (and the histogram built over).
#
# The graph is APPEND-only (one snapshot per recompute), never DROP+rewritten
# like the stats graph — the whole value is the distribution accumulating over
# time. Each snapshot node carries the run's effective floors + kept/quarantined
# totals; each relationship in the distribution is a point node linked back to
# its snapshot. Integers/decimals/booleans are typed literals so a downstream
# query can aggregate them numerically without parsing.
_DRIFT_NS = "https://cograph.tech/drift/"
_DRIFT_RECORDED_AT = _DRIFT_NS + "recordedAt"      # xsd:dateTime
_DRIFT_KG = _DRIFT_NS + "kg"                        # kg name (provenance)
_DRIFT_FLOOR_COV = _DRIFT_NS + "floorCov"          # xsd:decimal
_DRIFT_FLOOR_COUNT = _DRIFT_NS + "floorCount"      # xsd:integer
_DRIFT_KEPT = _DRIFT_NS + "kept"                   # xsd:integer (count)
_DRIFT_QUARANTINED = _DRIFT_NS + "quarantined"     # xsd:integer (count)
_DRIFT_POINT_OF = _DRIFT_NS + "pointOf"            # point -> snapshot node
_DRIFT_KEY = _DRIFT_NS + "key"                     # "<TypeLeaf>.<predLeaf>"
_DRIFT_COVERAGE = _DRIFT_NS + "coverage"           # xsd:decimal (percent)
_DRIFT_SUPPORT = _DRIFT_NS + "support"             # xsd:integer
_DRIFT_SOURCE_COUNT = _DRIFT_NS + "sourceCount"    # xsd:integer
_DRIFT_IS_CORE = _DRIFT_NS + "isCoreSlot"          # xsd:boolean
_DRIFT_POINT_KEPT = _DRIFT_NS + "pointKept"        # xsd:boolean (per-relationship)
_XSD = "http://www.w3.org/2001/XMLSchema#"

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


def _drift_history_graph_uri(tenant_id: str, kg_name: str) -> str:
    """Per-KG append-only graph holding the observe-only drift distribution (COG-57)."""
    return kg_graph_uri(tenant_id, kg_name) + "/drift-history"


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
        # ?pred is the INSTANCE predicate URI (…/onto/<pred>), not an attr URI.
        # core_slots holds ontology attr URIs, so derive the matching attr URI
        # from the source-type leaf + predicate leaf (same join the live scan does).
        src_leaf = su[len(TYPE_URI_PREFIX):]
        p_uri = r.get("pred", "")
        pred_leaf = (
            p_uri[len(ONTO_PRED_PREFIX):] if p_uri.startswith(ONTO_PRED_PREFIX)
            else p_uri.rstrip("/").split("/")[-1]
        )
        is_core = attr_uri(src_leaf, pred_leaf) in core_slots
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


async def _live_edge_scan_drift(
    client: NeptuneClient, kg_graph: str, tenant_id: str
) -> list[tuple[str, str]]:
    """ADR 0004 drift-gated variant of :func:`_live_edge_scan`.

    Used (flag ON only) when a KG has NO materialized stats graph — legacy KGs
    ingested before stats/drift control existed. Without a floor here the
    ``ManufacturerPartNumber.issuedby -> Retailer`` 6%-coverage drift shape still
    surfaces in the overview for un-materialized KGs (the production gap this
    fixes); the flag-OFF path keeps the unfiltered :func:`_live_edge_scan`.

    Derives type→type edges straight from instance triples, but applies the SAME
    support floor as :func:`_read_edges_from_stats_drift`. Per (source type,
    predicate, target type) it computes the support (``COUNT(DISTINCT`` source
    entity), and per source type the entity count; an edge is kept only when
    ``drift_control.should_declare(support, source_count, is_core)`` is True.
    ``is_core`` reads the ontology ``onto/coreSlot`` marker, keyed by
    ``attr_uri(srcLeaf, predLeaf)`` (the instance predicate ``…/onto/<predName>``
    maps to the ontology attr ``…/types/<srcLeaf>/attrs/<predName>``).
    """
    onto = tenant_graph_uri(tenant_id)
    # (1) Per (source type, predicate, target type) support = distinct source
    # entities carrying that entity-valued object. Target leaf derived in SPARQL
    # from the object entity URI (…/entities/<TargetType>/<id>).
    edge_q = (
        f"SELECT ?type ?p ?tgt (COUNT(DISTINCT ?e) AS ?support) FROM <{kg_graph}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> ?type .\n"
        f"  ?e ?p ?o .\n"
        f'  FILTER(STRSTARTS(STR(?type), "{TYPE_URI_PREFIX}"))\n'
        f'  FILTER(STRSTARTS(STR(?o), "{ENTITY_URI_PREFIX}"))\n'
        f'  BIND(REPLACE(STR(?o), "^.*/entities/([^/]+)/.*$", "$1") AS ?tgt)\n'
        f"}} GROUP BY ?type ?p ?tgt"
    )
    # (2) Per source type entity count (source_count for the coverage ratio).
    count_q = (
        f"SELECT ?type (COUNT(DISTINCT ?e) AS ?ec) FROM <{kg_graph}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> ?type .\n"
        f'  FILTER(STRSTARTS(STR(?type), "{TYPE_URI_PREFIX}"))\n'
        f"}} GROUP BY ?type"
    )
    # (3) Core-slot markers from the ontology graph, keyed by attr URI — SAME
    # query shape as _read_edges_from_stats_drift.
    core_q = (
        f"SELECT DISTINCT ?attr FROM <{onto}> WHERE {{\n"
        f"  ?attr <{_CORE_SLOT_PRED}> ?v .\n"
        f"}}"
    )
    edge_raw, count_raw, core_raw = await asyncio.gather(
        client.query(edge_q), client.query(count_q), client.query(core_q)
    )

    _, count_rows = parse_sparql_results(count_raw)
    source_counts: dict[str, int] = {}
    for r in count_rows:
        tu = r.get("type", "")
        if not tu.startswith(TYPE_URI_PREFIX):
            continue
        try:
            source_counts[tu] = int(r.get("ec", "0") or "0")
        except ValueError:
            source_counts[tu] = 0

    _, core_rows = parse_sparql_results(core_raw)
    core_slots = {r.get("attr", "") for r in core_rows if r.get("attr")}

    _, edge_rows = parse_sparql_results(edge_raw)
    out: list[tuple[str, str]] = []
    for r in edge_rows:
        tu = r.get("type", "")
        src = tu[len(TYPE_URI_PREFIX):] if tu.startswith(TYPE_URI_PREFIX) else ""
        if not src or "/" in src:  # skip nested URIs like .../attrs/x
            continue
        tgt = r.get("tgt", "")
        if not tgt or "/" in tgt:
            continue
        try:
            support = int(r.get("support", "0") or "0")
        except ValueError:
            support = 0
        source_count = source_counts.get(tu, 0)
        p_uri = r.get("p", "")
        pred_leaf = (
            p_uri[len(ONTO_PRED_PREFIX):] if p_uri.startswith(ONTO_PRED_PREFIX)
            else p_uri.rstrip("/").split("/")[-1]
        )
        is_core = attr_uri(src, pred_leaf) in core_slots
        if not drift_control.should_declare(support, source_count, is_core):
            continue
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

    # Ingest changed the data → the KG's stored triple count is stale. Drop it
    # so the next `list_kgs` recomputes (and re-stores) it once. Local import
    # avoids an import cycle between this module and knowledge_graphs.
    from cograph_client.api.routes.knowledge_graphs import invalidate_triple_count
    await invalidate_triple_count(client, tenant_id, kg_name)

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
            # pred_uri is …/onto/<pred>; core_slots holds ontology attr URIs, so
            # match on attr_uri(type_leaf, pred_leaf), not the raw predicate URI.
            "is_core_slot": attr_uri(type_leaf, pred_leaf) in core_slots,
        })
    report = drift_control.drift_report(declarations)
    logger.info(
        "drift_report",
        tenant=tenant_id,
        kg=kg_name,
        observe_only=drift_control.observe_only(),
        floor_cov=report["floor_cov"],
        floor_count=report["floor_count"],
        kept=report["kept"],
        quarantined=report["quarantined"],
        quarantine=report["quarantine"],
        # Full per-relationship coverage distribution — the observe-only signal
        # the floor should ultimately be set from (not the hand-tuned 20%).
        coverages=report["coverages"],
    )
    # COG-57: also persist the distribution to a durable, queryable store so it
    # survives past CloudWatch's 30-day log retention. Best-effort — a history
    # write must never fail a recompute (it is observability, not correctness).
    await _persist_drift_history(client, tenant_id, kg_name, report)
    return report


def _typed(value: object, xsd_type: str) -> str:
    """Render a typed literal: ``"<value>"^^<xsd:...>``."""
    return f'"{value}"^^<{_XSD}{xsd_type}>'


async def _persist_drift_history(
    client: NeptuneClient, tenant_id: str, kg_name: str, report: dict
) -> None:
    """Append one drift-report snapshot to the per-KG drift-history graph (COG-57).

    Writes the run's effective floors + kept/quarantined totals as a snapshot
    node, plus one point node per relationship in ``report["coverages"]`` (the
    full distribution, kept and quarantined alike). APPEND-only — never DROPs the
    graph — so the distribution accumulates across recomputes and tenants/KGs,
    which is the data ADR 0004 needs to set ``OMNIX_DRIFT_FLOOR_COV`` from a real
    histogram instead of the hand-calibrated 20%.

    Wrapped in try/except: persistence is observability, so a Neptune write
    failure here is logged and swallowed rather than failing the recompute.
    """
    hist = _drift_history_graph_uri(tenant_id, kg_name)
    snap = f"{_DRIFT_NS}snap/{uuid.uuid4().hex}"
    recorded_at = datetime.now(timezone.utc).isoformat()

    triples = [
        f"<{snap}> <{_DRIFT_RECORDED_AT}> {_typed(recorded_at, 'dateTime')} ; "
        f'<{_DRIFT_KG}> "{_esc(kg_name)}" ; '
        f"<{_DRIFT_FLOOR_COV}> {_typed(report['floor_cov'], 'decimal')} ; "
        f"<{_DRIFT_FLOOR_COUNT}> {_typed(report['floor_count'], 'integer')} ; "
        f"<{_DRIFT_KEPT}> {_typed(report['kept'], 'integer')} ; "
        f"<{_DRIFT_QUARANTINED}> {_typed(report['quarantined'], 'integer')} ."
    ]
    for c in report.get("coverages", []):
        pt = f"{snap}/p/{hashlib.md5(c['key'].encode()).hexdigest()}"
        triples.append(
            f"<{pt}> <{_DRIFT_POINT_OF}> <{snap}> ; "
            f'<{_DRIFT_KEY}> "{_esc(c["key"])}" ; '
            f"<{_DRIFT_COVERAGE}> {_typed(c['coverage'], 'decimal')} ; "
            f"<{_DRIFT_SUPPORT}> {_typed(c['support'], 'integer')} ; "
            f"<{_DRIFT_SOURCE_COUNT}> {_typed(c['source_count'], 'integer')} ; "
            f"<{_DRIFT_IS_CORE}> {_typed(str(bool(c['is_core_slot'])).lower(), 'boolean')} ; "
            f"<{_DRIFT_POINT_KEPT}> {_typed(str(bool(c['kept'])).lower(), 'boolean')} ."
        )

    body = "\n".join(triples)
    try:
        await client.update(f"INSERT DATA {{ GRAPH <{hist}> {{\n{body}\n}} }}")
    except Exception:
        logger.warning("drift_history_persist_failed", tenant=tenant_id, kg=kg_name, exc_info=True)


async def drop_kg_stats(client: NeptuneClient, tenant_id: str, kg_name: str) -> None:
    """Drop a KG's precomputed stats graph and evict its in-memory summaries.

    Called when a KG is deleted. The stats graph URI is derived from the KG
    name, so without this a KG later recreated under the same name would serve
    the deleted graph's stale counts until the next recompute lands.
    """
    stats = _stats_graph_uri(tenant_id, kg_name)
    hist = _drift_history_graph_uri(tenant_id, kg_name)
    # Drop the drift-history graph too (COG-57): its URI is derived from the KG
    # name, so a KG recreated under the same name would otherwise inherit the
    # deleted KG's distribution. Matches the stats-graph cleanup rationale above.
    await client.update(f"DROP SILENT GRAPH <{stats}> ; DROP SILENT GRAPH <{hist}>")
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
    # ACT only when enabled AND not observe-only. Observe-only collects the
    # coverage distribution (via the recompute drift report) without touching the
    # overview, so the floor can be set from real data before it filters anything.
    drift_on = drift_control.drift_control_enabled() and not drift_control.observe_only()
    if drift_on:
        edges = await _read_edges_from_stats_drift(client, tenant.tenant_id, kg_name)
    else:
        edges = await _read_edges_from_stats(client, tenant.tenant_id, kg_name)
    if edges is None:
        # No materialized stats graph (legacy KG). The live scan must honor the
        # drift floor too when ACTING, else below-floor drift edges leak into the
        # overview for un-materialized KGs. Observe-only / flag OFF: unchanged scan.
        kg_graph = kg_graph_uri(tenant.tenant_id, kg_name)
        if drift_on:
            edges = await _live_edge_scan_drift(client, kg_graph, tenant.tenant_id)
        else:
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


@router.get("/kgs/{kg_name}/drift-history")
async def get_drift_history(
    kg_name: str,
    limit: int = Query(100, ge=1, le=1000),
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Read the accumulated observe-only drift distribution for a KG (COG-57).

    Returns the persisted recompute snapshots (newest first), each with the run's
    effective floors, kept/quarantined totals, and the full per-relationship
    coverage distribution. This is the durable, queryable replacement for
    log-scraping CloudWatch — the data ADR 0004 sets ``OMNIX_DRIFT_FLOOR_COV``
    from. Raw distribution access only; histogram/floor analysis is done offline.
    """
    hist = _drift_history_graph_uri(tenant.tenant_id, kg_name)
    q = (
        f"SELECT ?snap ?recordedAt ?floorCov ?floorCount ?kept ?quarantined "
        f"?key ?coverage ?support ?sourceCount ?isCore ?pointKept\n"
        f"FROM <{hist}> WHERE {{\n"
        f"  ?snap <{_DRIFT_RECORDED_AT}> ?recordedAt ;\n"
        f"        <{_DRIFT_FLOOR_COV}> ?floorCov ;\n"
        f"        <{_DRIFT_FLOOR_COUNT}> ?floorCount ;\n"
        f"        <{_DRIFT_KEPT}> ?kept ;\n"
        f"        <{_DRIFT_QUARANTINED}> ?quarantined .\n"
        f"  OPTIONAL {{\n"
        f"    ?pt <{_DRIFT_POINT_OF}> ?snap ;\n"
        f"        <{_DRIFT_KEY}> ?key ;\n"
        f"        <{_DRIFT_COVERAGE}> ?coverage ;\n"
        f"        <{_DRIFT_SUPPORT}> ?support ;\n"
        f"        <{_DRIFT_SOURCE_COUNT}> ?sourceCount ;\n"
        f"        <{_DRIFT_IS_CORE}> ?isCore ;\n"
        f"        <{_DRIFT_POINT_KEPT}> ?pointKept .\n"
        f"  }}\n"
        f"}} ORDER BY DESC(?recordedAt)"
    )
    try:
        _, rows = parse_sparql_results(await client.query(q))
    except Exception:
        logger.warning("drift_history_read_failed", tenant=tenant.tenant_id, kg=kg_name, exc_info=True)
        return {"kg": kg_name, "snapshots": []}

    # Reassemble flat (snapshot × point) rows into nested snapshots, preserving
    # the recordedAt-desc order. A snapshot with no points (empty distribution)
    # still appears, with coverages == [].
    snapshots: dict[str, dict] = {}
    for r in rows:
        sid = r.get("snap", "")
        if not sid:
            continue
        snap = snapshots.get(sid)
        if snap is None:
            snap = {
                "recorded_at": r.get("recordedAt", ""),
                "floor_cov": _to_float(r.get("floorCov")),
                "floor_count": _to_int(r.get("floorCount")),
                "kept": _to_int(r.get("kept")),
                "quarantined": _to_int(r.get("quarantined")),
                "coverages": [],
            }
            snapshots[sid] = snap
        if r.get("key"):
            snap["coverages"].append({
                "key": r["key"],
                "coverage": _to_float(r.get("coverage")),
                "support": _to_int(r.get("support")),
                "source_count": _to_int(r.get("sourceCount")),
                "is_core_slot": r.get("isCore") == "true",
                "kept": r.get("pointKept") == "true",
            })
    return {"kg": kg_name, "snapshots": list(snapshots.values())[:limit]}


def _to_int(v: str | None) -> int:
    try:
        return int(v) if v not in (None, "") else 0
    except (TypeError, ValueError):
        return 0


def _to_float(v: str | None) -> float:
    try:
        return float(v) if v not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


@router.get("/kgs/{kg_name}/types/{type_name}/records")
async def get_type_records(
    kg_name: str,
    type_name: str,
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None),
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Paged entity instances for the Explorer Data table (COG-100).

    Returns one page of instances of ``type_name``, ordered deterministically
    by entity URI (``ORDER BY ?e``) with keyset pagination via ``cursor`` (the
    last entity URI from the previous page).  For each entity the endpoint
    fetches all attribute values, excluding ``rdf:type`` and
    ``SYSTEM_PREDICATES``.  Attribute predicates are resolved to display names
    via the ontology (same ``attr_def`` query shape as ``get_type_summary``).
    The row ``name`` is the declared ``attrs/name`` attribute value when present
    (ingest stores the human-readable name there; ``rdfs:label`` holds the
    opaque entity-id slug), else ``rdfs:label``, else the entity-URI leaf.

    Response shape::

        {
            "columns": ["name", "<attr1>", ...],
            "rows": [{"id": "<uri>", "name": "...", "<attr1>": "...", ...}],
            "total": <int>,
            "next_cursor": "<uri>" | null,
        }

    Never errors on an empty/missing type; returns the empty sentinel instead.
    """
    _EMPTY = {"columns": ["name"], "rows": [], "total": 0, "next_cursor": None}

    onto_graph = tenant_graph_uri(tenant.tenant_id)
    kg_graph = kg_graph_uri(tenant.tenant_id, kg_name)
    t_uri = type_uri(type_name)

    # --- (1) attribute display-name map from ontology (same as get_type_summary) ---
    attr_def_sparql = (
        f"SELECT ?attr ?attrLabel ?range FROM <{onto_graph}> WHERE {{\n"
        f"  ?attr <{RDF_TYPE}> <{RDF_PROPERTY}> .\n"
        f"  ?attr <{RDFS}#domain> <{t_uri}> .\n"
        f"  ?attr <{RDFS}#label> ?attrLabel .\n"
        f"  OPTIONAL {{ ?attr <{RDFS}#range> ?range }}\n"
        f"}}"
    )

    # --- (2) entity page: keyset pagination ordered by ?e URI ---
    cursor_filter = f'  FILTER(STR(?e) > "{_esc(cursor)}")\n' if cursor else ""
    entities_sparql = (
        f"SELECT DISTINCT ?e FROM <{kg_graph}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> <{t_uri}> .\n"
        f"{_PRIMARY_TYPE_GUARD}"
        f"{cursor_filter}"
        f"}} ORDER BY ?e LIMIT {limit}"
    )

    # --- (3) total count: try stats graph first, fall back to COUNT query ---
    stats_graph = _stats_graph_uri(tenant.tenant_id, kg_name)
    total_sparql = (
        f"SELECT ?ec FROM <{stats_graph}> WHERE {{\n"
        f"  <{t_uri}> <{_STAT_ENTITY_COUNT}> ?ec\n"
        f"}}"
    )

    attr_def_raw, entity_raw, total_raw = await asyncio.gather(
        client.query(attr_def_sparql),
        client.query(entities_sparql),
        client.query(total_sparql),
    )

    _, attr_def_rows = parse_sparql_results(attr_def_raw)
    # Column budget.  Ontology-DECLARED attributes are always shown (they are the
    # type's schema — including enriched attrs like ``company`` that may sit on
    # only a handful of entities), so they are exempt from this cap.  The cap
    # only bounds the *extra* non-declared predicates discovered on the page, so
    # one rogue entity with dozens of ad-hoc predicates can't blow up the table.
    # Raised from 12 → 24 so a wide-but-legitimate declared schema isn't crowded
    # out and there's still headroom for a few observed-but-undeclared columns.
    _MAX_COLS = 24
    # Map ONTO pred URI → label.  We also need the instance predicate URI which
    # is `…/onto/<predLeaf>`.  Build both directions.  ``declared_display`` is the
    # ordered list of declared-attribute display labels that ALWAYS become
    # columns (deduped, alphabetical for a stable order — coverage isn't carried
    # by the attr-def query, so we don't pay an extra round-trip to rank by it).
    attr_label_by_onto: dict[str, str] = {}  # onto attr URI → label
    attr_label_by_pred: dict[str, str] = {}  # onto pred URI → label (instance triples)
    declared_display: list[str] = []
    declared_display_set: set[str] = set()
    for r in attr_def_rows:
        a_uri = r.get("attr", "")
        label = r.get("attrLabel") or a_uri.rstrip("/").split("/")[-1]
        if not a_uri:
            continue
        attr_label_by_onto[a_uri] = label
        # instance predicate URI: …/onto/<leaf>  where leaf is the last segment of
        # the attr URI (attrs/<leaf> → <leaf>)
        pred_leaf = a_uri.rstrip("/").split("/")[-1]
        inst_pred = ONTO_PRED_PREFIX + pred_leaf
        attr_label_by_pred[inst_pred] = label
        # ``name`` is rendered from rdfs:label as the first column; never let a
        # declared attribute literally named "name" duplicate it.
        if label != "name" and label not in declared_display_set:
            declared_display_set.add(label)
            declared_display.append(label)
    declared_display.sort()

    _, entity_rows = parse_sparql_results(entity_raw)
    entity_uris = [r.get("e", "") for r in entity_rows if r.get("e")]
    if not entity_uris:
        # No instances on this page — still need a total
        _, total_rows = parse_sparql_results(total_raw)
        total = _to_int(total_rows[0].get("ec") if total_rows else None)
        if not total:
            # Fall back to a COUNT query if stats absent
            count_sparql = (
                f"SELECT (COUNT(DISTINCT ?e) AS ?n) FROM <{kg_graph}> WHERE {{\n"
                f"  ?e <{RDF_TYPE}> <{t_uri}> .\n"
                f"{_PRIMARY_TYPE_GUARD}"
                f"}}"
            )
            _, cnt_rows = parse_sparql_results(await client.query(count_sparql))
            total = _to_int(cnt_rows[0].get("n") if cnt_rows else None)
        return {**_EMPTY, "total": total}

    # --- (4) fetch attribute values for the page entities ---
    uri_values = " ".join(f"<{u}>" for u in entity_uris)
    values_sparql = (
        f"SELECT ?e ?p ?o FROM <{kg_graph}> WHERE {{\n"
        f"  VALUES ?e {{ {uri_values} }}\n"
        f"  ?e ?p ?o .\n"
        f'  FILTER(?p != <{RDF_TYPE}>)\n'
        f"}}"
    )

    # Total count and attribute values fetched concurrently
    values_raw, total_raw2 = await asyncio.gather(
        client.query(values_sparql),
        client.query(total_sparql),
    )

    _, values_rows = parse_sparql_results(values_raw)

    # Determine total
    _, total_rows2 = parse_sparql_results(total_raw2)
    total = _to_int(total_rows2[0].get("ec") if total_rows2 else None)
    if not total:
        count_sparql = (
            f"SELECT (COUNT(DISTINCT ?e) AS ?n) FROM <{kg_graph}> WHERE {{\n"
            f"  ?e <{RDF_TYPE}> <{t_uri}> .\n"
            f"{_PRIMARY_TYPE_GUARD}"
            f"}}"
        )
        _, cnt_rows = parse_sparql_results(await client.query(count_sparql))
        total = _to_int(cnt_rows[0].get("n") if cnt_rows else None)

    # --- (5) assemble rows ---
    # Collect per-entity: label + attribute values keyed by display name.
    # ``_name_attr`` captures the instance value of the declared "name" attribute
    # (``…/onto/name`` ← ``attrs/name``): these entities carry their real,
    # human-readable name THERE. ``rdfs:label`` holds the opaque entity-id slug
    # (ingest writes ``(entity_uri, rdfs:label, entity.id)``), so attrs/name is
    # the PREFERRED name source — rdfs:label is only the fallback below it. We
    # don't render attrs/name as a SEPARATE column (it would duplicate the first
    # "name" column); its value feeds the first column instead.
    LABEL_PRED = f"{RDFS}#label"
    entity_data: dict[str, dict] = {
        u: {"_label": None, "_name_attr": None, "_attrs": {}} for u in entity_uris
    }
    # Column order: declared attributes ALWAYS first (schema columns, not subject
    # to the frequency cap), then any extra non-declared predicates observed on
    # the page — bounded by _MAX_COLS so a stray entity can't inflate the table.
    col_display: list[str] = list(declared_display)
    col_set: set[str] = set(declared_display)
    extra_count = 0

    for r in values_rows:
        e_uri = r.get("e", "")
        p_uri = r.get("p", "")
        o_val = r.get("o", "")
        if not e_uri or e_uri not in entity_data:
            continue
        if p_uri == LABEL_PRED:
            entity_data[e_uri]["_label"] = o_val
            continue
        if p_uri in SYSTEM_PREDICATES:
            continue
        # Resolve display name: check attr_label_by_pred (instance pred) first,
        # then attr_label_by_onto (onto attr URI), then fall back to the URI leaf.
        display = (
            attr_label_by_pred.get(p_uri)
            or attr_label_by_onto.get(p_uri)
            or p_uri.rstrip("/").split("/")[-1]
        )
        # "name" is rendered in the first column; a declared/instance predicate
        # named "name" (e.g. …/onto/name ← attrs/name) must not become a SEPARATE
        # column. But its value is the entity's real, human-readable name —
        # capture it so the first column can PREFER it over the slug-shaped
        # rdfs:label.
        if display == "name":
            if entity_data[e_uri]["_name_attr"] is None:
                entity_data[e_uri]["_name_attr"] = o_val
            continue
        if display not in col_set and extra_count < _MAX_COLS:
            col_set.add(display)
            col_display.append(display)
            extra_count += 1
        entity_data[e_uri]["_attrs"][display] = o_val

    columns = ["name"] + col_display
    rows = []
    for u in entity_uris:
        d = entity_data[u]
        # Name precedence: the declared "name" attribute's value (attrs/name)
        # FIRST, else rdfs:label, else the URI slug. Ingest writes
        # `(entity_uri, rdfs:label, entity.id)` — i.e. rdfs:label IS the opaque
        # entity-id slug — while the human-readable name lives in attrs/name. So
        # attrs/name must win over rdfs:label, otherwise the row degrades to the
        # slug (e.g. "4akvVWgTcS") even when a real name is present.
        label = d["_name_attr"] or d["_label"] or u.rstrip("/").split("/")[-1]
        row: dict = {"id": u, "name": label}
        for col in col_display:
            # Declared columns with no value on this entity render blank.
            row[col] = d["_attrs"].get(col, "")
        rows.append(row)

    next_cursor = entity_uris[-1] if len(entity_uris) == limit else None

    return {
        "columns": columns,
        "rows": rows,
        "total": total,
        "next_cursor": next_cursor,
    }


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
