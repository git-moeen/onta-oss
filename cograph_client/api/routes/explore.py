"""Explorer API — read-only endpoints that power the Cograph Explorer web app.

All data comes from existing Neptune graphs; no new infra required. These
endpoints add convenience (bundling, coverage %, search) on top of the raw
ontology + KG queries already used by the CLI.
"""

import asyncio
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

# Type summaries are read-heavy and only change on ingest, but each cold build
# scans every instance triple of the type (seconds on a large KG). Cache the
# assembled payload per (tenant, kg, type) with a short TTL so repeat Explorer
# loads — and the per-bubble navigations in the web app — return instantly.
_SUMMARY_TTL_SECONDS = 300.0
_summary_cache: dict[tuple[str, str, str], tuple[float, dict]] = {}


@router.get("/kgs/{kg_name}/types/{type_name}/summary")
async def get_type_summary(
    kg_name: str,
    type_name: str,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Bundle all Explorer panel data for one type in one call.

    Returns instance count, attributes + coverage %, relationships + coverage %
    and avg multiplicity. All percentages are relative to entity_count.
    """
    cache_key = (tenant.tenant_id, kg_name, type_name)
    cached = _summary_cache.get(cache_key)
    if cached is not None and (time.monotonic() - cached[0]) < _SUMMARY_TTL_SECONDS:
        return cached[1]

    onto_graph = tenant_graph_uri(tenant.tenant_id)
    kg_graph = kg_graph_uri(tenant.tenant_id, kg_name)
    t_uri = type_uri(type_name)

    # 1) Ontology description for this type (small — ontology graph).
    onto_sparql = (
        f"SELECT ?label ?comment ?parent FROM <{onto_graph}> WHERE {{\n"
        f"  <{t_uri}> <{RDFS}#label> ?label .\n"
        f"  OPTIONAL {{ <{t_uri}> <{RDFS}#comment> ?comment }}\n"
        f"  OPTIONAL {{ <{t_uri}> <{RDFS}#subClassOf> ?parent }}\n"
        f"}}"
    )

    # 2) Ontology attribute definitions (name + datatype per predicate URI).
    attr_def_sparql = (
        f"SELECT ?attr ?attrLabel ?range FROM <{onto_graph}> WHERE {{\n"
        f"  ?attr <{RDF_TYPE}> <{RDF_PROPERTY}> .\n"
        f"  ?attr <{RDFS}#domain> <{t_uri}> .\n"
        f"  ?attr <{RDFS}#label> ?attrLabel .\n"
        f"  OPTIONAL {{ ?attr <{RDFS}#range> ?range }}\n"
        f"}}"
    )

    # 3) Per-predicate usage in ONE instance scan: distinct-subject count
    #    (coverage), a sample object (attr/rel classification + target type),
    #    and the entity-valued object total (relationship avg degree). The
    #    rdf:type row's count is the entity count, so this single query
    #    replaces the former separate count / predicate / degree queries.
    pred_sparql = (
        f"SELECT ?p (COUNT(DISTINCT ?e) AS ?cnt) (SAMPLE(?o) AS ?sample)\n"
        f'  (SUM(IF(STRSTARTS(STR(?o), "{ENTITY_URI_PREFIX}"), 1, 0)) AS ?relTotal)\n'
        f"FROM <{kg_graph}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> <{t_uri}> .\n"
        f"  ?e ?p ?o .\n"
        f"}} GROUP BY ?p ORDER BY DESC(?cnt)"
    )

    # Independent queries → run concurrently (wall time ≈ the slowest scan,
    # not the sum). Ontology lookups are tiny; the instance scan dominates.
    onto_raw, attr_def_raw, pred_raw = await asyncio.gather(
        client.query(onto_sparql),
        client.query(attr_def_sparql),
        client.query(pred_sparql),
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

    _, all_pred_rows = parse_sparql_results(pred_raw)
    # Entity count comes from the rdf:type row of the same scan.
    entity_count = 0
    pred_rows = []
    rel_degrees: dict[str, int] = {}
    for r in all_pred_rows:
        p_uri = r.get("p", "")
        try:
            cnt = int(r.get("cnt", "0"))
        except ValueError:
            cnt = 0
        if p_uri == RDF_TYPE:
            entity_count = cnt
            continue
        pred_rows.append(r)
        try:
            rel_degrees[p_uri] = int(r.get("relTotal", "0"))
        except (ValueError, KeyError):
            pass

    def _coverage(count: int) -> float:
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

    attributes = []
    relationships = []

    for r in pred_rows:
        p_uri = r.get("p", "")
        if not p_uri or p_uri in SYSTEM_PREDICATES:
            continue
        try:
            cnt = int(r.get("cnt", "0"))
        except ValueError:
            cnt = 0
        sample = r.get("sample", "")
        defn = attr_defs.get(p_uri, {})
        name = defn.get("name") or p_uri.rstrip("/").split("/")[-1]
        rng = defn.get("range", "")
        is_rel = (
            sample.startswith("https://cograph.tech/entities/")
            or rng.startswith(TYPE_URI_PREFIX)
        )
        cov = _coverage(cnt)
        if is_rel:
            target: str | None = None
            if rng.startswith(TYPE_URI_PREFIX):
                target = rng[len(TYPE_URI_PREFIX):]
            elif sample.startswith("https://cograph.tech/entities/"):
                tail = sample[len("https://cograph.tech/entities/"):]
                head = tail.split("/", 1)[0]
                if head:
                    target = head
            total_degree = rel_degrees.get(p_uri, 0)
            avg_degree = round(total_degree / entity_count, 2) if entity_count else 0.0
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

    result = {
        "name": type_name,
        "description": onto_row.get("comment", ""),
        "parent_type": parent_type,
        "entity_count": entity_count,
        "attributes": attributes,
        "relationships": relationships,
    }
    _summary_cache[cache_key] = (time.monotonic(), result)
    return result


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
