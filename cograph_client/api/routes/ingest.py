"""POST /graphs/{tenant}/ingest — raw data ingestion with schema resolution."""

import json
from pathlib import Path

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError

from cograph_client.api.deps import get_neptune_client
from cograph_client.api.rate_limit import limiter
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.config import settings
from cograph_client.graph.client import NeptuneClient
from cograph_client.resolver.models import CSVRowsRequest, CSVSchemaMapping, CSVSchemaRequest, IngestRequest, IngestResult
from cograph_client.graph.queries import kg_graph_uri, tenant_graph_uri
from cograph_client.nlp.pipeline import NLQueryPipeline
from cograph_client.resolver.schema_resolver import SchemaResolver
from cograph_client.resolver.verdict_cache import JsonVerdictCache

router = APIRouter(prefix="/graphs/{tenant}")
_log = structlog.stdlib.get_logger("cograph.api.ingest")

# Verdict cache lives alongside the app data. For ECS/Fargate deployments,
# this should be on an EFS mount or replaced with DynamoDB.
_CACHE_PATH = Path("/tmp/omnix-verdict-cache.json")


def _get_verdict_cache() -> JsonVerdictCache:
    return JsonVerdictCache(_CACHE_PATH)


@router.post("/ingest", response_model=IngestResult)
@limiter.limit("10/minute")
async def ingest(
    request: Request,
    body: IngestRequest,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Ingest raw content into the knowledge graph.

    Runs LLM extraction, schema resolution (type matching, attribute
    resolution, validation), and inserts validated triples into Neptune.
    """
    cache = _get_verdict_cache()
    resolver = SchemaResolver(
        neptune=client,
        anthropic_key=settings.anthropic_api_key,
        verdict_cache=cache,
    )
    # Use KG-specific graph for instance data if specified
    instance_graph = kg_graph_uri(tenant.tenant_id, body.kg_name) if body.kg_name else None
    result = await resolver.ingest(
        content=body.content,
        tenant_id=tenant.tenant_id,
        content_type=body.content_type,
        source=body.source,
        instance_graph=instance_graph,
    )
    # Invalidate ontology cache so queries pick up new types/relationships
    graph_uri = tenant_graph_uri(tenant.tenant_id)
    NLQueryPipeline.invalidate_cache(graph_uri)
    # Re-embed all affected types (new types + types with new attributes)
    # so semantic retrieval never serves stale embeddings
    affected_types = set(result.types_created)
    for attr_added in result.attributes_added:
        type_name = attr_added.split(".")[0]
        affected_types.add(type_name)
    if affected_types:
        from cograph_client.nlp.pipeline import get_embedding_service
        svc = get_embedding_service()
        if svc:
            try:
                await svc.embed_types(graph_uri, list(affected_types), client)
            except Exception:
                pass  # non-blocking
    # Refresh precomputed Explorer type-stats for this KG (background, non-blocking).
    if body.kg_name:
        try:
            from cograph_client.api.routes.explore import schedule_recompute
            schedule_recompute(client, tenant.tenant_id, body.kg_name)
        except Exception:
            pass
    return result


@router.post("/ingest/csv/schema", response_model=CSVSchemaMapping)
@limiter.limit("10/minute")
async def infer_csv_schema(
    request: Request,
    body: CSVSchemaRequest,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Step 1: Infer column mapping from CSV headers + sample rows. Single LLM call."""
    import anthropic
    from cograph_client.resolver.csv_resolver import CSVResolver
    from cograph_client.resolver.schema_resolver import SchemaResolver

    graph_uri = tenant_graph_uri(tenant.tenant_id)
    cache = _get_verdict_cache()
    resolver = SchemaResolver(neptune=client, anthropic_key=settings.anthropic_api_key, verdict_cache=cache)
    existing_types, _ = await resolver._fetch_ontology(graph_uri)

    csv_resolver = CSVResolver(
        anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key),
        settings.openrouter_api_key,
    )
    try:
        return await csv_resolver.infer_schema(body.headers, body.sample_rows, existing_types, body.total_rows)
    except (ValidationError, KeyError, json.JSONDecodeError) as e:
        _log.warning("csv_schema_inference_failed", error=str(e))
        raise HTTPException(
            status_code=422,
            detail=(
                f"Schema inference failed after retry: {e}. "
                "The CSV's leading rows may be too sparse — ensure at least a "
                "few rows have most fields populated."
            ),
        )


@router.post("/ingest/csv/rows", response_model=IngestResult)
@limiter.limit("200/minute")
async def ingest_csv_rows(
    request: Request,
    body: CSVRowsRequest,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Step 2: Insert rows using a pre-inferred mapping. No LLM call."""
    from cograph_client.resolver.csv_resolver import CSVResolver
    from cograph_client.resolver.models import ExtractionResult

    graph_uri = tenant_graph_uri(tenant.tenant_id)
    instance_graph = kg_graph_uri(tenant.tenant_id, body.kg_name) if body.kg_name else graph_uri
    cache = _get_verdict_cache()
    resolver = SchemaResolver(neptune=client, anthropic_key=settings.anthropic_api_key, verdict_cache=cache)
    resolver._instance_graph = instance_graph
    existing_types, existing_attrs = await resolver._fetch_ontology(graph_uri)

    # Pre-register ontology attributes from the CSV mapping.
    # This ensures the ontology has all attributes even if the first batch
    # of rows has empty values for some columns. Without this, types can
    # end up with 0 ontology attributes when all columns are relationships.
    from cograph_client.graph.ontology_queries import insert_attribute, insert_type

    def _mget(obj, key, default=None):
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    # Group (type_name, owned columns) for ontology pre-registration. Multi-entity
    # mode registers each in-row entity's type with its owned columns; legacy mode
    # registers the single `entity_type` with all columns (unchanged behavior).
    all_columns = _mget(body.mapping, "columns", []) or []
    entities_spec = _mget(body.mapping, "entities")
    if entities_spec:
        groups = [
            (_mget(spec, "type_name"),
             [c for c in all_columns if _mget(c, "entity") == _mget(spec, "name")])
            for spec in entities_spec
        ]
    else:
        groups = [(_mget(body.mapping, "entity_type", ""), all_columns)]

    for type_name, cols in groups:
        if type_name and type_name not in existing_types:
            await client.update(insert_type(graph_uri, type_name, ""))
            existing_types[type_name] = ""
            existing_attrs[type_name] = {}
        for col in cols:
            col_role = _mget(col, "role", "")
            raw_name = _mget(col, "attribute_name") or _mget(col, "column_name", "")
            col_name = raw_name.lower().replace(" ", "_") if raw_name else ""
            col_datatype = _mget(col, "datatype", "string")
            col_target = _mget(col, "target_type")

            if not col_name or col_role == "type_id":
                continue

            type_attrs = existing_attrs.get(type_name, {})
            if col_name not in type_attrs:
                if col_role == "relationship" and col_target:
                    await client.update(insert_attribute(graph_uri, type_name, col_name, "", col_target))
                else:
                    await client.update(insert_attribute(graph_uri, type_name, col_name, "", col_datatype))

    entities, relationships = CSVResolver.apply_mapping(body.mapping, body.rows)

    extraction = ExtractionResult(entities=entities, relationships=relationships)
    result = IngestResult(entities_extracted=len(entities))
    entity_uri_map: dict[str, str] = {}
    entity_type_map: dict[str, str] = {}
    batch_id = ""

    result = await resolver._resolve_and_insert(
        extraction, graph_uri, existing_types, existing_attrs,
        body.source, result, entity_uri_map, entity_type_map, batch_id,
    )

    NLQueryPipeline.invalidate_cache(graph_uri)
    # Incrementally embed new/changed types
    affected_types = set(result.types_created)
    for attr_added in result.attributes_added:
        if "." in attr_added:
            affected_types.add(attr_added.split(".")[0])
    if affected_types:
        from cograph_client.nlp.pipeline import get_embedding_service
        svc = get_embedding_service()
        if svc:
            try:
                await svc.embed_types(graph_uri, list(affected_types), client)
            except Exception:
                pass  # non-blocking
    return result


@router.post("/embeddings/build")
@limiter.limit("5/minute")
async def build_embeddings(
    request: Request,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Trigger a full embedding build for all ontology types in this tenant."""
    from cograph_client.nlp.pipeline import get_embedding_service
    svc = get_embedding_service()
    if not svc:
        return {"status": "embeddings_not_configured"}
    graph_uri = tenant_graph_uri(tenant.tenant_id)
    count = await svc.build_from_ontology(graph_uri, client)
    return {"status": "ok", "types_embedded": count}
