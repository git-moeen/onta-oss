"""Explorer API — read-only endpoints that power the Cograph Explorer web app.

All data comes from existing Neptune graphs; no new infra required. These
endpoints add convenience (bundling, coverage %, search) on top of the raw
ontology + KG queries already used by the CLI.
"""

from fastapi import APIRouter, Depends, Query

from cograph_client.api.deps import get_neptune_client
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.ontology_queries import type_uri
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import kg_graph_uri, tenant_graph_uri

router = APIRouter(prefix="/graphs/{tenant}/explore")

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS = "http://www.w3.org/2000/01/rdf-schema"
TYPE_URI_PREFIX = "https://cograph.tech/types/"
SYSTEM_PREDICATES: frozenset[str] = frozenset({
    f"{RDFS}#label",
    "https://cograph.tech/onto/ingested_at",
    "https://cograph.tech/onto/source",
})


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
    onto_graph = tenant_graph_uri(tenant.tenant_id)
    kg_graph = kg_graph_uri(tenant.tenant_id, kg_name)
    t_uri = type_uri(type_name)

    # 1) Ontology description for this type.
    onto_sparql = (
        f"SELECT ?label ?comment ?parent FROM <{onto_graph}> WHERE {{\n"
        f"  <{t_uri}> <{RDFS}#label> ?label .\n"
        f"  OPTIONAL {{ <{t_uri}> <{RDFS}#comment> ?comment }}\n"
        f"  OPTIONAL {{ <{t_uri}> <{RDFS}#subClassOf> ?parent }}\n"
        f"}}"
    )
    _, onto_rows = parse_sparql_results(await client.query(onto_sparql))
    onto_row = onto_rows[0] if onto_rows else {}
    parent_uri = onto_row.get("parent", "")
    parent_type = parent_uri.rstrip("/").split("/")[-1] if parent_uri else None

    # 2) Ontology attribute definitions (name + datatype per predicate URI).
    attr_def_sparql = (
        f"SELECT ?attr ?attrLabel ?range FROM <{onto_graph}> WHERE {{\n"
        f"  ?attr <{RDF_TYPE}> <http://www.w3.org/1999/02/22-rdf-syntax-ns#Property> .\n"
        f"  ?attr <{RDFS}#domain> <{t_uri}> .\n"
        f"  ?attr <{RDFS}#label> ?attrLabel .\n"
        f"  OPTIONAL {{ ?attr <{RDFS}#range> ?range }}\n"
        f"}}"
    )
    _, attr_def_rows = parse_sparql_results(await client.query(attr_def_sparql))
    attr_defs: dict[str, dict[str, str]] = {
        r["attr"]: {"name": r.get("attrLabel", ""), "range": r.get("range", "")}
        for r in attr_def_rows
        if r.get("attr")
    }

    # 3) Instance count.
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

    # 4) Per-predicate usage: how many distinct subjects have this predicate,
    #    plus a sample object so we can classify attr vs. relationship.
    pred_sparql = (
        f"SELECT ?p (COUNT(DISTINCT ?e) AS ?cnt) (SAMPLE(?o) AS ?sample)\n"
        f"FROM <{kg_graph}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> <{t_uri}> .\n"
        f"  ?e ?p ?o .\n"
        f"  FILTER(?p != <{RDF_TYPE}>)\n"
        f"}} GROUP BY ?p ORDER BY DESC(?cnt)"
    )
    _, pred_rows = parse_sparql_results(await client.query(pred_sparql))

    # 5) Per-relationship target counts for avg degree.
    #    We need COUNT(?o) (not DISTINCT ?e) to compute avg.
    rel_degree_sparql = (
        f"SELECT ?p (COUNT(?o) AS ?total) FROM <{kg_graph}> WHERE {{\n"
        f"  ?e <{RDF_TYPE}> <{t_uri}> .\n"
        f"  ?e ?p ?o .\n"
        f"  FILTER(?p != <{RDF_TYPE}>)\n"
        f"  FILTER(STRSTARTS(STR(?o), 'https://cograph.tech/entities/'))\n"
        f"}} GROUP BY ?p"
    )
    _, rel_degree_rows = parse_sparql_results(await client.query(rel_degree_sparql))
    rel_degrees: dict[str, int] = {}
    for r in rel_degree_rows:
        try:
            rel_degrees[r["p"]] = int(r.get("total", "0"))
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

    return {
        "name": type_name,
        "description": onto_row.get("comment", ""),
        "parent_type": parent_type,
        "entity_count": entity_count,
        "attributes": attributes,
        "relationships": relationships,
    }


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
