from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from cograph_client.api.deps import get_neptune_client
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.config import settings
from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.ontology_queries import (
    get_full_ontology_query,
    get_subtypes_query,
    get_type_attributes_query,
    get_type_detail_query,
    get_type_functions_query,
    insert_attribute,
    insert_subtype,
    insert_type,
    list_types_query,
    upsert_attribute,
)
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import tenant_graph_uri
from cograph_client.models.ontology import (
    AttributeAdd,
    AttributeDefinition,
    ResolutionResult,
    ResolvedChange,
    ResolveRequest,
    SubtypeAdd,
    TypeCreate,
    TypeResponse,
)
from cograph_client.nlp.pipeline import get_embedding_service
from cograph_client.resolver.ontology_resolver import OntologyResolver
from cograph_client.resolver.type_matcher import TypeMatcher
from cograph_client.resolver.verdict_cache import JsonVerdictCache

router = APIRouter(prefix="/graphs/{tenant}/ontology")

# Verdict cache lives alongside the app data (same path the ingest route uses);
# for ECS/Fargate this should be on an EFS mount or replaced with DynamoDB.
_VERDICT_CACHE_PATH = Path("/tmp/omnix-verdict-cache.json")


@router.post("/types", status_code=201)
async def create_type(
    body: TypeCreate,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    graph_uri = tenant_graph_uri(tenant.tenant_id)
    sparql = insert_type(graph_uri, body.name, body.description, body.parent_type)
    await client.update(sparql)

    for attr in body.attributes:
        attr_sparql = insert_attribute(graph_uri, body.name, attr.name, attr.description, attr.datatype)
        await client.update(attr_sparql)

    return {"created": body.name, "attributes": len(body.attributes)}


@router.get("/types", response_model=list[TypeResponse])
async def list_types(
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    graph_uri = tenant_graph_uri(tenant.tenant_id)
    raw = await client.query(list_types_query(graph_uri))
    _, bindings = parse_sparql_results(raw)

    types = {}
    for row in bindings:
        label = row.get("label", "")
        if label not in types:
            types[label] = TypeResponse(
                name=label,
                description=row.get("comment", ""),
                parent_type=_extract_name(row.get("parent")) if row.get("parent") else None,
            )
        elif row.get("parent"):
            types[label].parent_type = _extract_name(row["parent"])

    return list(types.values())


@router.get("/types/{type_name}", response_model=TypeResponse)
async def get_type(
    type_name: str,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    graph_uri = tenant_graph_uri(tenant.tenant_id)

    raw = await client.query(get_type_detail_query(graph_uri, type_name))
    _, bindings = parse_sparql_results(raw)
    if not bindings:
        raise HTTPException(status_code=404, detail=f"Type '{type_name}' not found")

    row = bindings[0]
    result = TypeResponse(
        name=row.get("label", type_name),
        description=row.get("comment", ""),
        parent_type=_extract_name(row.get("parent")) if row.get("parent") else None,
    )

    attr_raw = await client.query(get_type_attributes_query(graph_uri, type_name))
    _, attr_bindings = parse_sparql_results(attr_raw)
    result.attributes = [
        AttributeDefinition(
            name=r.get("attrLabel", ""),
            description=r.get("attrComment", ""),
            datatype=_xsd_to_datatype(r.get("range", "")),
        )
        for r in attr_bindings
    ]

    sub_raw = await client.query(get_subtypes_query(graph_uri, type_name))
    _, sub_bindings = parse_sparql_results(sub_raw)
    result.subtypes = [r.get("label", "") for r in sub_bindings]

    func_raw = await client.query(get_type_functions_query(graph_uri, type_name))
    _, func_bindings = parse_sparql_results(func_raw)
    result.functions = [r.get("name", "") for r in func_bindings]

    return result


@router.post("/types/{type_name}/attributes", status_code=201)
async def add_attributes(
    type_name: str,
    body: AttributeAdd,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    graph_uri = tenant_graph_uri(tenant.tenant_id)
    for attr in body.attributes:
        sparql = insert_attribute(graph_uri, type_name, attr.name, attr.description, attr.datatype)
        await client.update(sparql)
    return {"type": type_name, "attributes_added": len(body.attributes)}


@router.post("/types/{type_name}/subtypes", status_code=201)
async def add_subtype(
    type_name: str,
    body: SubtypeAdd,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    graph_uri = tenant_graph_uri(tenant.tenant_id)
    sparql = insert_subtype(graph_uri, type_name, body.subtype)
    await client.update(sparql)
    return {"parent": type_name, "subtype": body.subtype}


@router.get("/schema")
async def get_full_schema(
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Get the complete ontology schema. Used by the NL pipeline."""
    graph_uri = tenant_graph_uri(tenant.tenant_id)
    raw = await client.query(get_full_ontology_query(graph_uri))
    _, bindings = parse_sparql_results(raw)

    types = {}
    for row in bindings:
        type_label = row.get("typeLabel", "")
        if not type_label:
            continue
        if type_label not in types:
            types[type_label] = {"attributes": [], "functions": []}
        if row.get("attrLabel") and row["attrLabel"] not in [a["name"] for a in types[type_label]["attributes"]]:
            types[type_label]["attributes"].append({
                "name": row["attrLabel"],
                "datatype": _xsd_to_datatype(row.get("range", "")),
            })
        if row.get("funcName") and row["funcName"] not in types[type_label]["functions"]:
            types[type_label]["functions"].append(row["funcName"])

    return {"types": types}


# ── Natural-language ontology evolution (COG-80) ──────────────────────────────
#
# `resolve` takes a fuzzy ask, resolves it against the current ontology, AUTO-
# APPLIES the high-confidence changes, and returns the rest as proposals. `apply`
# commits a single proposal the agent chose to confirm. Both write through the
# atomic upsert builders so retries are idempotent.


def _build_resolver(graph_uri: str) -> OntologyResolver:
    """Assemble an :class:`OntologyResolver` from the shared app primitives.

    Degrades gracefully: if the embedding service can't initialise (no key /
    offline) the resolver still runs on the TypeMatcher cascade's other layers.
    """
    try:
        embedding_service = get_embedding_service()
    except Exception:  # pragma: no cover - defensive: embeddings are optional
        embedding_service = None

    matcher = TypeMatcher(
        openrouter_key=settings.openrouter_api_key,
        cache=JsonVerdictCache(_VERDICT_CACHE_PATH),
        embedding_service=embedding_service,
        graph_uri=graph_uri,
    )
    return OntologyResolver(
        openrouter_key=settings.openrouter_api_key,
        type_matcher=matcher,
        embedding_service=embedding_service,
    )


async def _apply_change(change: ResolvedChange, graph_uri: str, client: NeptuneClient) -> list[str]:
    """Translate one resolved change into atomic upsert SPARQL and run it.

    Shared by `/resolve` (for confident `applied` changes) and `/apply` (for a
    confirmed proposal). Type minting uses the non-destructive `insert_type`
    (only ever adds class+label, never clears an existing type's
    description/parent); the property itself goes through `upsert_attribute`,
    whose single-valued `rdfs:range`/`rdfs:comment` are replaced atomically.
    """
    sparqls: list[str] = []

    # A `create` change means the subject type is newly minted — ensure it
    # exists first (idempotent on an existing type, never clobbers it).
    if change.action == "create":
        sparqls.append(insert_type(graph_uri, change.subject_type))

    # A relationship's range points at another type; ensure that target type
    # exists before we point an object property at it.
    if change.kind == "relationship":
        sparqls.append(insert_type(graph_uri, change.datatype_or_target))

    # `reuse` is already satisfied, but the upsert is idempotent, so emitting it
    # keeps the property authoritative without risk.
    sparqls.append(
        upsert_attribute(graph_uri, change.subject_type, change.name, datatype=change.datatype_or_target)
    )

    for sparql in sparqls:
        await client.update(sparql)
    return sparqls


@router.post("/resolve", response_model=ResolutionResult)
async def resolve_ontology(
    body: ResolveRequest,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
) -> ResolutionResult:
    """Resolve a fuzzy NL ask into ontology changes; auto-apply the confident
    ones, return ambiguous/new-type ones as proposals for the caller to confirm
    via `POST .../ontology/apply`.

    `dry_run=True` (the interactive Explorer path) is plan-only: the resolver
    runs exactly as below but NOTHING is written to Neptune — every change (what
    would have auto-applied plus the proposals) is returned under `proposals`,
    with `applied` empty, so the UI can render one uniform reviewable list."""
    graph_uri = tenant_graph_uri(tenant.tenant_id)
    resolver = _build_resolver(graph_uri)
    result = await resolver.resolve(body.ask, graph_uri, client)

    if body.dry_run:
        # Plan-only: write nothing, fold the would-be-applied changes into the
        # proposals list so the caller reviews everything uniformly.
        return ResolutionResult(
            applied=[],
            proposals=result.applied + result.proposals,
            summary=result.summary,
            dry_run=True,
        )

    for change in result.applied:
        await _apply_change(change, graph_uri, client)

    return result


@router.post("/apply")
async def apply_ontology_change(
    body: ResolvedChange,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Commit a single proposal previously returned by `/resolve` (stateless —
    the caller passes the change object straight back). Idempotent."""
    graph_uri = tenant_graph_uri(tenant.tenant_id)
    operations = await _apply_change(body, graph_uri, client)
    return {
        "applied": body,
        "operations": len(operations),
        "summary": f"Applied {change_label(body)}",
    }


def change_label(change: ResolvedChange) -> str:
    target = f" → {change.datatype_or_target}" if change.kind == "relationship" else f" ({change.datatype_or_target})"
    return f"{change.action} {change.kind} '{change.name}'{target} on {change.subject_type}"


def _extract_name(uri: str | None) -> str | None:
    if not uri:
        return None
    return uri.rstrip("/").split("/")[-1]


TYPE_URI_PREFIX = "https://cograph.tech/types/"


def _xsd_to_datatype(xsd_uri: str) -> str:
    if not xsd_uri:
        return "string"
    # Check if it's a reference to another ontology type
    if xsd_uri.startswith(TYPE_URI_PREFIX):
        return xsd_uri[len(TYPE_URI_PREFIX):]
    mapping = {
        "string": "string",
        "integer": "integer",
        "float": "float",
        "boolean": "boolean",
        "dateTime": "datetime",
        "Resource": "uri",
    }
    last = xsd_uri.split("#")[-1] if "#" in xsd_uri else xsd_uri.split("/")[-1]
    return mapping.get(last, "string")
