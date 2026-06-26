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
from cograph_client.graph.kg_writer import refresh_after_write
from cograph_client.graph.queries import kg_graph_uri, tenant_graph_uri
from cograph_client.resolver.attribute_resolver import AttributeSchema
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
    # Single shared post-write housekeeping path (graph/kg_writer.py) — the SAME
    # refresh the enrichment writer runs: invalidate the NL-planning cache,
    # re-embed affected types (new types + types that gained an attribute), and
    # recompute the Explorer's type-stats. Keeps ingestion and enrichment from
    # drifting on WHAT gets refreshed after a write.
    affected_types = set(result.types_created)
    for attr_added in result.attributes_added:
        affected_types.add(attr_added.split(".")[0])
    await refresh_after_write(
        client,
        tenant_id=tenant.tenant_id,
        kg_name=body.kg_name,
        affected_types=affected_types,
    )
    return result


@router.post("/ingest/csv/schema", response_model=CSVSchemaMapping)
@limiter.limit("10/minute")
async def infer_csv_schema(
    request: Request,
    body: CSVSchemaRequest,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Step 1: Infer column mapping from CSV headers + sample rows.

    Default (``OMNIX_CSV_INFERENCE_V2`` unset/truthy) is the ADR 0003
    evidence-grounded pipeline: a deterministic column profile (Pass A) feeds
    a REASON LLM call (Pass B), an adversarial REFUTE LLM call (Pass C), and
    a conceptual COMPLETE LLM call (Pass D). The response is the same
    ``CSVSchemaMapping`` contract as before, extended with optional,
    backward-compatible fields: per-decision ``why``/``confidence`` (on
    entities/columns), ``key_strategy`` per entity, the refute pass's
    ``violations``, an ``inference_audit`` block, and the completion pass's
    ``ontology_extensions`` (dependent-entity promotions, core slots, dataset
    constants, rejected candidates). ``OMNIX_CSV_INFERENCE_V2=0`` falls back
    to the legacy single-LLM-call path.

    Confirm gate (COG-52, until COG-56's judge panel lands): promotions and
    low-confidence completions come back flagged ``held_for_review`` — the
    gate is CLIENT-SIDE. The Explorer asks the user to confirm/edit held
    items; whatever mapping the client then posts to ``/ingest/csv/rows`` is
    applied as-is.

    Latency budget: schema inference is up to 3 sequential LLM calls, once per
    CSV file — REASON + REFUTE + COMPLETE (each with at most one validation
    retry at temperature 0.3); the Pass A profile is milliseconds. Row
    ingestion (``/ingest/csv/rows``) stays LLM-free, so the cost does not
    scale with row count. Clients should send the full file (capped at a few
    thousand rows) as ``sample_rows`` — profile fidelity, and therefore
    mapping quality, depends on it.
    """
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
    """Step 2: Insert rows using a pre-inferred mapping. No LLM call.

    The mapping is applied AS POSTED — including ``ontology_extensions``
    items flagged ``held_for_review`` by ``/ingest/csv/schema``: the confirm
    gate is client-side (the Explorer asks the user, then posts the possibly
    edited mapping here). Promoted types and their core slots are
    pre-registered in the tenant ontology — including slots with zero data,
    marked with a ``coreSlot`` triple as declared enrichment targets
    (ADR 0003 §3).
    """
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

            if not col_name:
                continue
            # type_id columns are pre-registered too: the key value now also
            # lands on instances as a regular attribute (ADR 0003 §2 —
            # key-as-attribute), so the ontology must know it.

            type_attrs = existing_attrs.get(type_name, {})
            if col_name not in type_attrs:
                if col_role == "relationship" and col_target:
                    await client.update(insert_attribute(graph_uri, type_name, col_name, "", col_target))
                else:
                    await client.update(insert_attribute(graph_uri, type_name, col_name, "", col_datatype))

    # Pre-register ontology extensions (ADR 0003 Pass D, COG-52): promoted
    # types and EVERY core slot — including slots with zero data in this file.
    # An empty core slot is a declared enrichment target (§3): the coreSlot
    # marker triple is what lets enrichment later query "instances with empty
    # core slots" as its work queue. No server-side gating happens here — the
    # held_for_review confirm gate is client-side, so whatever (possibly
    # user-edited) extensions arrive in the posted mapping are written
    # (judge-panel gating lands with COG-56).
    from cograph_client.graph.ontology_queries import mark_core_slot

    extensions = _mget(body.mapping, "ontology_extensions")
    for ext in (_mget(extensions, "types", []) or []) if extensions else []:
        ext_type = _mget(ext, "type_name") or _mget(ext, "type", "")
        if not ext_type:
            continue
        promoted_from = _mget(ext, "promoted_from_attribute")
        if ext_type not in existing_types:
            desc = (
                f"Dependent entity promoted from attribute '{promoted_from}' (ADR 0003 Pass D)"
                if promoted_from else ""
            )
            await client.update(insert_type(graph_uri, ext_type, desc))
            existing_types[ext_type] = ""
            existing_attrs[ext_type] = {}
        ext_attrs = existing_attrs.setdefault(ext_type, {})
        for slot in _mget(ext, "core_slots", []) or []:
            raw_slot = _mget(slot, "name", "")
            slot_name = raw_slot.lower().replace(" ", "_") if raw_slot else ""
            if not slot_name:
                continue
            slot_kind = _mget(slot, "kind", "attribute")
            slot_target = _mget(slot, "target_type")
            slot_why = _mget(slot, "why") or ""
            # A relationship slot's target type exists in the ontology even
            # when this dataset has ZERO instances of it (e.g. the issuer of
            # a promoted identifier with no issuer column).
            if slot_kind == "relationship" and slot_target and slot_target not in existing_types:
                await client.update(insert_type(graph_uri, slot_target, ""))
                existing_types[slot_target] = ""
                existing_attrs[slot_target] = {}
            if slot_name not in ext_attrs:
                if slot_kind == "relationship" and slot_target:
                    await client.update(insert_attribute(graph_uri, ext_type, slot_name, slot_why, slot_target))
                    slot_datatype = slot_target
                else:
                    await client.update(insert_attribute(graph_uri, ext_type, slot_name, slot_why, "string"))
                    slot_datatype = "string"
                # Store a real AttributeSchema (not a bare marker string): this dict
                # is existing_attrs[ext_type], which flows into resolve_attribute()
                # during the insert pass. A str here triggers
                # `'str' object has no attribute 'datatype'` the moment any ingested
                # entity of ext_type has an attribute matching this slot name.
                ext_attrs[slot_name] = AttributeSchema(name=slot_name, datatype=slot_datatype)
            await client.update(mark_core_slot(graph_uri, ext_type, slot_name))

    # Judge-panel gating (ADR 0003 §5, COG-56): the tenant-layer writes above
    # already happened — the tenant uses the shape immediately (ADR 0002 §2
    # fallback rule). Mapping-shape decisions that are judge-panel material
    # (dependent-entity promotions, core slots, dataset constants, and
    # reason-pass decisions with confidence < 0.7) are now ENQUEUED as
    # governance proposals on a fire-and-forget background task — this request
    # never awaits the panel, so ingest latency is unchanged. With no premium
    # service registered the proposals just land in the OSS pending holder
    # (tenant-layer-only behavior); the premium judge service registers via
    # register_governance_panel to judge, promote approved shapes to
    # Global-Public, and align re-derivations to approved canonical shapes.
    from cograph_client.resolver.governance import enqueue_shape_proposals, mapping_shape_proposals

    try:
        enqueue_shape_proposals(mapping_shape_proposals(
            body.mapping,
            tenant.tenant_id,
            dataset_hint=body.source or body.kg_name or "",
            proposer_model=CSVResolver.EXTRACT_MODEL,
        ))
    except Exception:
        # Gating is best-effort by design: a seam failure degrades to
        # tenant-layer-only behavior, never to a failed ingest.
        _log.warning("shape_governance_enqueue_failed", tenant=tenant.tenant_id, exc_info=True)

    applied = CSVResolver.apply_mapping(body.mapping, body.rows)
    entities, relationships = applied.entities, applied.relationships

    # Row-conservation accounting (ADR 0003 §2): report rows_in vs dropped so
    # a mismatch is loud — rows only ever drop when ALL owned values are empty.
    if applied.rows_dropped:
        _log.warning(
            "csv_rows_dropped",
            tenant=tenant.tenant_id,
            source=body.source,
            kg_name=body.kg_name,
            rows_in=applied.rows_in,
            rows_dropped=applied.rows_dropped,
            drops_by_entity=applied.drops_by_entity,
        )

    extraction = ExtractionResult(entities=entities, relationships=relationships)
    result = IngestResult(
        entities_extracted=len(entities),
        rows_in=applied.rows_in,
        rows_dropped=applied.rows_dropped,
        drops_by_entity=applied.drops_by_entity,
    )
    entity_uri_map: dict[str, str] = {}
    entity_type_map: dict[str, str] = {}
    batch_id = ""

    result = await resolver._resolve_and_insert(
        extraction, graph_uri, existing_types, existing_attrs,
        body.source, result, entity_uri_map, entity_type_map, batch_id,
    )

    # Shared post-write housekeeping (graph/kg_writer.py) — same path as /ingest
    # and the enrichment writer: cache-invalidate, re-embed affected types, and
    # recompute stats. (CSV-rows previously skipped the stats recompute; routing
    # through the shared path gives it the same refresh as every other writer.)
    affected_types = set(result.types_created)
    for attr_added in result.attributes_added:
        if "." in attr_added:
            affected_types.add(attr_added.split(".")[0])
    await refresh_after_write(
        client,
        tenant_id=tenant.tenant_id,
        kg_name=body.kg_name,
        affected_types=affected_types,
    )
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
