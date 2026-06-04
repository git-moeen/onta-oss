"""Knowledge graph management — list, create, delete named graphs within a tenant.

All KGs share the tenant's ontology but have separate instance data.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from cograph_client.api.deps import get_neptune_client
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.ontology_queries import (
    get_type_attributes_query,
    type_uri,
)
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import kg_graph_uri, tenant_graph_uri

router = APIRouter(prefix="/graphs/{tenant}/kgs")

OMNIX_ONTO = "https://cograph.tech/onto"
TYPE_URI_PREFIX = "https://cograph.tech/types/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
NAME_ATTRS = ("name", "title", "label", "headline")

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


@router.get("", response_model=list[KGInfo])
async def list_kgs(
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """List all knowledge graphs for a tenant."""
    base = tenant_graph_uri(tenant.tenant_id)

    # Query the metadata graph for KG registrations
    sparql = (
        f"SELECT ?name ?desc FROM <{base}> WHERE {{"
        f"  ?kg <{OMNIX_ONTO}/kg_name> ?name ."
        f"  OPTIONAL {{ ?kg <{OMNIX_ONTO}/kg_description> ?desc }}"
        f"}}"
    )
    raw = await client.query(sparql)
    _, bindings = parse_sparql_results(raw)

    kgs = []
    for row in bindings:
        name = row.get("name", "")
        if not name:
            continue

        # Get triple count for this KG
        graph = kg_graph_uri(tenant.tenant_id, name)
        count_sparql = f"SELECT (COUNT(*) as ?c) FROM <{graph}> WHERE {{ ?s ?p ?o }}"
        try:
            count_raw = await client.query(count_sparql)
            _, count_bindings = parse_sparql_results(count_raw)
            count = int(count_bindings[0].get("c", "0")) if count_bindings else 0
        except Exception:
            count = 0

        kgs.append(KGInfo(name=name, description=row.get("desc", ""), triple_count=count))

    return kgs


@router.post("", response_model=KGInfo, status_code=201)
async def create_kg(
    body: KGCreate,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Create a new knowledge graph for a tenant."""
    base = tenant_graph_uri(tenant.tenant_id)
    kg_uri = f"https://cograph.tech/kgs/{tenant.tenant_id}/{body.name}"

    sparql = (
        f"INSERT DATA {{\n"
        f"  GRAPH <{base}> {{\n"
        f'    <{kg_uri}> <{OMNIX_ONTO}/kg_name> "{body.name}" .\n'
    )
    if body.description:
        sparql += f'    <{kg_uri}> <{OMNIX_ONTO}/kg_description> "{body.description}" .\n'
    sparql += f"  }}\n}}"

    await client.update(sparql)
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
    raw = await client.query(sparql)
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
        out.append(TypeCount(name=leaf, entity_count=count))
    return out


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
