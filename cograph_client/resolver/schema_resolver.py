"""Schema Resolver — deterministic layer between LLM extraction and Neptune.

Pipeline:
  Raw data → LLM extraction (non-deterministic) → Schema Resolver → Neptune

The resolver enforces ontology consistency: type matching, attribute resolution,
schema-on-write validation, and Option D coexistence for structure promotion.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

import os

import anthropic
import httpx
import structlog

from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.ontology_queries import (
    PRIMITIVE_TYPES,
    batch_entity_exists_query,
    entity_exists_query,
    get_full_ontology_query,
    insert_attribute,
    insert_subtype,
    insert_type,
    parent_map_query,
    set_object_property_range,
    type_uri,
    upsert_type,
    upsert_type_comment,
    attr_uri,
)
from cograph_client.graph.layers import LayerStack, type_name_from_uri
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.kg_writer import insert_facts
from cograph_client.graph.provenance import build_provenance_triples, provenance_graph_uri
from cograph_client.graph.queries import BATCH_PREDICATE, batched_insert_triples, delete_batch_query, insert_triples, tenant_graph_uri
from cograph_client.resolver.attribute_resolver import (
    AttributeSchema,
    check_promotion,
    resolve_attribute,
)
from cograph_client.resolver.models import (
    AttrAction,
    CSVSchemaMapping,
    ExtractionResult,
    ExtractedAttribute,
    ExtractedEntity,
    ExtractedRelationship,
    IngestResult,
    MatchVerdict,
    RejectedValue,
    ValidatedTriple,
    ValidationOutcome,
)
from cograph_client.resolver.llm_router import PRIMARY_MODEL, openrouter_chat
from cograph_client.resolver.predicate_normalizer import normalize_predicate
from cograph_client.resolver.type_matcher import TypeMatcher
from cograph_client.resolver.validator import validate_triple
from cograph_client.resolver.verdict_cache import JsonVerdictCache

logger = structlog.stdlib.get_logger("cograph.resolver")

EXTRACTION_SYSTEM = """\
You are a knowledge graph extraction engine. Given raw text and the current \
ontology, extract structured entities, their attributes, and relationships.

Rules:
- Each entity must have a type_name (PascalCase, singular noun, e.g. "Property" not "properties")
- Each entity must have an id (use the most natural identifier: name, address, etc.)
- Attributes have a name (snake_case), value (string), and datatype (string, integer, float, boolean, datetime, uri)
- Relationships connect two entities by their id with a predicate (snake_case)

Type placement:
You will be given the existing ontology types. For each entity you extract:
- Always pick the MOST SPECIFIC type the data justifies (HotelGuest over Guest \
over Person; Condo over Property) — granularity is recovered later, coarseness \
is not.
- If its type already exists in the ontology, use that exact type name and set \
same_as to that name.
- If its type is new but is a subtype of an existing type (is-a relationship), \
set parent_type to the EXISTING type name. Prefer connecting to the hierarchy \
over creating orphaned types. A Broker is a Person. A City is a Place. A Condo \
is a Property. But geographic containment is NOT a subtype: State is NOT a \
subtype of City, City is NOT a subtype of State. Use relationships for containment.
- parent_chain: list the FULL is-a lineage of type_name, most-specific first, up \
to the most general type — e.g. type_name "HotelGuest" -> parent_chain \
["Guest", "Person"]; "Condo" -> ["Property", "Asset"]. Include ancestors even if \
they are NOT yet in the ontology (they will be created). This closes a brand-new \
multi-level hierarchy in one shot. Omit or leave empty only for a top-level type.
- also_types: ONLY for genuine, independent multi-classification — when the entity \
truly IS two unrelated things at once (a hotel employee who is also a guest: \
type_name "Employee", also_types ["Guest"]). These are NOT ancestors. Leave empty \
in the common case.
- If its type is genuinely unrelated to anything in the ontology, leave same_as \
and parent_type null and parent_chain empty.

Entity-first principle:
When unsure whether a value should be a literal attribute or a separate entity \
with a relationship, ALWAYS prefer creating a separate entity. Entities can have \
attributes and relationships added later; literals are dead ends. Only use literal \
attributes for truly atomic values: numbers, dates, booleans, short enums, or \
identifiers.

Reify measurements:
When a value is a MEASUREMENT, METRIC, or other observation that can CHANGE OVER \
TIME or carries PROVENANCE (a score, rating, price, ranking, benchmark result), \
model it as its OWN entity (e.g. type_name "Score", "Rating", "Price") with \
attributes "value" and, when available, "timestamp"/"as_of" — plus relationships \
linking it to the thing measured and to the provider/publisher that produced it. \
Name that producer relationship "measured_by" / "reported_by" / "published_by" / \
"produced_by" (NEVER the bare predicate "source" — that collides with internal \
housekeeping). Reify INSTEAD of a bare scalar attribute on the parent: a bare \
number loses its history and its provenance the moment a newer reading arrives. \
Reify only genuine observations; do NOT reify a fixed intrinsic property (a \
person's birth_year, a product's sku).

Lift providers / organizations:
When records carry a recurring CATEGORICAL naming a provider, vendor, publisher, \
manufacturer, organization, or brand (a value that repeats across records and \
names a real-world actor), create an "Organization" entity per distinct value and \
relate to it (e.g. provided_by / published_by / made_by) instead of leaving it a \
string. Do NOT lift free-form descriptive text or a one-off label that names no \
actor.

Subtypes with a description:
When a measurement or entity is a SPECIALIZED KIND of a more general type (e.g. a \
"Humanness Index" is a kind of Score; a "Condo" is a kind of Property), emit it \
as a subtype via parent_chain AND set subtype_description to a brief sentence \
explaining what it is / what it measures. The description becomes the new type's \
definition in the ontology. Set subtype_description ONLY for a new specialized \
type you are minting — leave it null otherwise.

Respond with valid JSON only. No markdown."""

EXTRACTION_USER_TEMPLATE = """\
Existing ontology types:
{existing_types}

Extract entities, attributes, and relationships from this content:

---
{content}
---

Return JSON:
{{
  "entities": [
    {{
      "type_name": "MostSpecificTypeName",
      "id": "identifier",
      "same_as": "<existing type name if this is the same concept, else null>",
      "parent_type": "<existing type name if this is a subtype, else null>",
      "parent_chain": ["<immediate parent>", "<grandparent>", "..."],
      "also_types": ["<independent co-type, rare>"],
      "subtype_description": "<brief definition when minting a NEW specialized subtype, else null>",
      "attributes": [
        {{"name": "attr_name", "value": "attr_value", "datatype": "string"}}
      ]
    }}
  ],
  "relationships": [
    {{
      "source_id": "entity_id",
      "predicate": "relationship_name",
      "target_id": "entity_id"
    }}
  ]
}}"""


class SchemaResolver:
    # Primary extraction model, routed through OpenRouter with the configured
    # fallback. Defaults to the shared primary.
    EXTRACT_MODEL = os.environ.get("OMNIX_EXTRACT_MODEL", PRIMARY_MODEL)
    EXTRACT_PROVIDER = os.environ.get("OMNIX_EXTRACT_PROVIDER", "openrouter")
    # Anthropic-SDK offline fallback (used only when no OpenRouter key is set) —
    # must be a NATIVE Anthropic model id. Env-overridable.
    INFER_MODEL = os.environ.get("OMNIX_INFER_MODEL", "claude-opus-4-8")
    ONTOLOGY_REFRESH_INTERVAL = int(os.environ.get("OMNIX_ONTOLOGY_REFRESH_INTERVAL", "50"))
    # Output ceiling for one extraction call. Raised 4096 → 8192: the
    # reification/lift prompt makes each record emit MANY more entities +
    # relationships, so a chunk's JSON can blow past 4096 tokens, get truncated,
    # fail to parse, and silently drop the whole batch. Env-overridable.
    EXTRACT_MAX_TOKENS = int(os.environ.get("OMNIX_EXTRACT_MAX_TOKENS", "8192"))

    def __init__(
        self,
        neptune: NeptuneClient,
        anthropic_key: str,
        verdict_cache: JsonVerdictCache,
        embedding_service: object | None = None,
    ):
        self._neptune = neptune
        self._anthropic = anthropic.AsyncAnthropic(api_key=anthropic_key)
        self._embedding_service = embedding_service
        from cograph_client.config import settings
        self._openrouter_key = settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._type_matcher = TypeMatcher(self._openrouter_key, verdict_cache, embedding_service)
        # Cross-file entity resolution. Best-effort: failures never block ingest.
        from cograph_client.resolver.er import ERPipeline
        self._er = ERPipeline(neptune)
        self._er_enabled = os.environ.get("COGRAPH_ER_ENABLED", "1") != "0"
        # Per-fact provenance (ADR 0002 §4): statement-metadata nodes in the
        # companion provenance graph. Default OFF so default triple output and
        # Neptune call pattern stay byte-identical.
        self._provenance_enabled = os.environ.get("COGRAPH_PROVENANCE_ENABLED", "0") == "1"
        # Governance seam (ADR 0002 §2): when ON, a brand-new type is ALSO
        # proposed to an LLM judge panel; on majority approval it is written
        # to the Global-Public layer with governance provenance. The tenant
        # write stays today's behavior either way — governance never blocks
        # or gates ingest. Default OFF (matching COGRAPH_PROVENANCE_ENABLED).
        self._governance_enabled = os.environ.get("COGRAPH_GOVERNANCE_ENABLED", "0") == "1"
        if self._governance_enabled:
            from cograph_client.resolver.governance import GovernanceEngine, LLMJudgePanel
            self._governance = GovernanceEngine(neptune)
            self._judge_panel = LLMJudgePanel(self._openrouter_key)
        # Background governance tasks (COG-46): the judge panel + Public-layer
        # write are scheduled off the ingest path; references are retained
        # here so drain_governance() can await them deterministically.
        self._governance_tasks: list[asyncio.Task] = []
        # child->parent (type-name) map for subclass-chain walks. Built once per
        # ingest from parent_map_query and mutated in-place as new subtypes are
        # created so later entities in the same batch can climb the chain.
        self._parent_of: dict[str, str] = {}

    async def ingest(
        self,
        content: str,
        tenant_id: str,
        content_type: str = "text",
        source: str = "",
        instance_graph: str | None = None,
    ) -> IngestResult:
        """Full ingestion pipeline: extract → resolve → validate → insert.

        Args:
            instance_graph: If set, instance data goes into this graph while
                ontology updates go into the tenant's base graph. This enables
                multiple KGs sharing one ontology.
        """
        graph_uri = tenant_graph_uri(tenant_id)
        # Ontology always goes to the base tenant graph
        # Instance data goes to instance_graph if specified, otherwise base graph
        self._instance_graph = instance_graph or graph_uri
        # Set graph URI on type matcher so embedding pre-filter can find the right store
        self._type_matcher._graph_uri = graph_uri

        # Step 1: Fetch existing ontology (needed for extraction context)
        existing_types, existing_attrs = await self._fetch_ontology(graph_uri)
        # Build the child->parent subclass map once per ingest. Used to climb the
        # hierarchy for ER config selection and ancestor synthesis. Mutated
        # in-place as new subtypes are created during this ingest.
        self._parent_of = await self._fetch_parent_map(graph_uri)

        # CSV: use schema-inference pipeline (1 LLM call for schema, deterministic for rows)
        if content_type == "csv":
            return await self._ingest_csv(content, graph_uri, existing_types, existing_attrs, source)

        # Text/JSON: chunk and process
        from cograph_client.resolver.chunker import (
            chunk_text,
            chunk_json_array,
            json_array_len,
        )
        is_json = content_type in ("json", "jsonl")
        if is_json:
            chunks = chunk_json_array(content)
        else:
            chunks = chunk_text(content)

        # Row-conservation accounting for the JSON path (ADR 0003 §2): a chunk
        # whose extraction yields nothing (e.g. truncated output) must not vanish
        # silently. We count records IN and records DROPPED so the run can never
        # be presented as complete while a whole batch was lost.
        rows_in = 0
        rows_dropped = 0

        if len(chunks) <= 1:
            # Small content — single extraction. JSON STILL routes through the
            # truncation-recovery helper (FIX 1): even one chunk's reified output
            # (each row → Model + reified Score + Organization + relationships) can
            # exceed max_tokens and get truncated, and bare _extract would then
            # silently return ZERO entities for the whole pull. Recovery splits +
            # retries down to the floor so a single chunk can't vanish.
            if is_json:
                rows_in = json_array_len(content)
                extraction, dropped = await self._extract_json_chunk_with_recovery(
                    content, existing_types,
                )
                rows_dropped += dropped
            else:
                extraction = await self._extract(content, content_type, existing_types)
        else:
            # Multiple chunks — extract each, deduplicate entities
            merged_entities = []
            merged_relationships = []
            seen_ids: set[str] = set()
            for chunk in chunks:
                if is_json:
                    chunk_rows = json_array_len(chunk)
                    rows_in += chunk_rows
                    extraction, dropped = await self._extract_json_chunk_with_recovery(
                        chunk, existing_types,
                    )
                    rows_dropped += dropped
                else:
                    extraction = await self._extract(chunk, content_type, existing_types)
                for e in extraction.entities:
                    if e.id not in seen_ids:
                        merged_entities.append(e)
                        seen_ids.add(e.id)
                merged_relationships.extend(extraction.relationships)
            extraction = ExtractionResult(
                entities=merged_entities,
                relationships=merged_relationships,
                source_text=content[:500],
            )

        logger.info(
            "extraction_complete",
            entities=len(extraction.entities),
            relationships=len(extraction.relationships),
            rows_in=rows_in,
            rows_dropped=rows_dropped,
        )

        if not extraction.entities:
            return IngestResult(
                entities_extracted=0, rows_in=rows_in, rows_dropped=rows_dropped,
            )

        # Step 3: Resolve types and attributes, validate, insert
        batch_id = str(uuid4())
        result = IngestResult(
            entities_extracted=len(extraction.entities),
            batch_id=batch_id,
            rows_in=rows_in,
            rows_dropped=rows_dropped,
        )
        entity_uri_map: dict[str, str] = {}  # entity id → URI
        entity_type_map: dict[str, str] = {}  # entity id → resolved type name

        try:
            final = await self._resolve_and_insert(
                extraction, graph_uri, existing_types, existing_attrs,
                source, result, entity_uri_map, entity_type_map, batch_id,
            )
            # Never present a run as complete while a whole chunk was lost to
            # truncation (FIX 1): a non-zero drop count after recovery is an
            # ERROR-level signal carried back on the result for the caller.
            if final.rows_dropped:
                logger.error(
                    "ingest_rows_dropped",
                    batch_id=batch_id,
                    rows_in=final.rows_in,
                    rows_dropped=final.rows_dropped,
                )
            return final
        except Exception:
            logger.error(
                "ingest_failed_rolling_back",
                batch_id=batch_id,
                entities_so_far=result.entities_resolved,
                exc_info=True,
            )
            instance_graph = getattr(self, "_instance_graph", graph_uri)
            try:
                sparql = delete_batch_query(instance_graph, batch_id)
                await self._neptune.update(sparql)
                logger.info("batch_rollback_complete", batch_id=batch_id)
            except Exception:
                logger.error("batch_rollback_failed", batch_id=batch_id, exc_info=True)
            raise

    async def _resolve_and_insert(
        self,
        extraction: ExtractionResult,
        graph_uri: str,
        existing_types: dict[str, str],
        existing_attrs: dict[str, dict[str, AttributeSchema]],
        source: str,
        result: IngestResult,
        entity_uri_map: dict[str, str],
        entity_type_map: dict[str, str],
        batch_id: str,
    ) -> IngestResult:
        """Inner pipeline: resolve entities, insert triples. Separated for rollback.

        Two-pass architecture for I/O efficiency:
          Pass 1: Resolve types for all entities, compute URIs
          Batch check: Which URIs already exist in Neptune (one query per 500)
          Pass 2: Resolve attributes, validate, insert triples
        """
        instance_graph = getattr(self, "_instance_graph", graph_uri)

        # Pass 1: Resolve types and compute entity URIs
        resolved_types: dict[str, str] = {}  # entity.id → resolved_type
        pending_uris: list[str] = []
        # ER index triples (block keys + denormalized signals) for newly minted
        # entities. Empty for merged/dedup'd entities.
        er_index_triples: list[tuple[str, str, str]] = []
        # Genuine independent co-classifications per entity id (ADR rule 1).
        # Empty for the common single-type case.
        entity_also_types: dict[str, list[str]] = {}
        # Track which entity IDs were merged into existing URIs (for telemetry)
        er_merged_count = 0
        for i, entity in enumerate(extraction.entities):
            if i > 0 and i % self.ONTOLOGY_REFRESH_INTERVAL == 0:
                await self._refresh_ontology(graph_uri, existing_types, existing_attrs)

            resolved_type = await self._resolve_type(
                entity, graph_uri, existing_types, existing_attrs, result,
            )
            if resolved_type:
                resolved_types[entity.id] = resolved_type
                # Resolve genuine co-types so they exist in the ontology; record
                # them for the multi-type write in pass 2. The declared primary
                # type (resolved_type) still owns URI minting + ER.
                also = await self._resolve_also_types(
                    entity, resolved_type, graph_uri, existing_types, existing_attrs, result,
                )
                if also:
                    entity_also_types[entity.id] = also
                entity_uri = f"https://cograph.tech/entities/{resolved_type}/{_safe_id(entity.id)}"

                # Cross-file ER: see if this entity matches an existing one.
                # Failures here MUST never block ingest — log and fall through.
                if self._er_enabled:
                    try:
                        from cograph_client.resolver.er import MergeAction, config_for_with_hierarchy
                        # Climb the subclass chain so a granular leaf (HotelGuest)
                        # inherits a configured ancestor's (Guest) ER config and
                        # ER fires on the subtype.
                        er_config = config_for_with_hierarchy(resolved_type, self._parent_of)
                        er_applies = er_config is not None
                        type_uri = f"https://cograph.tech/types/{resolved_type}"
                        decision = await self._er.find_match(
                            entity, resolved_type, type_uri, instance_graph,
                            config=er_config, parent_of=self._parent_of,
                        )
                        if decision.action == MergeAction.AUTO_MERGE and decision.canonical_uri:
                            entity_uri = decision.canonical_uri
                            er_merged_count += 1
                            # Merge expansion: write the incoming entity's
                            # ER signals onto the CANONICAL URI so future
                            # ingests can find this same person via the new
                            # signals (e.g. a CRM merge adds the secondary
                            # email as an alias of the canonical Guest,
                            # letting a Loyalty ingest match later via that
                            # email). Triples are idempotent on Neptune.
                            normalized, keys = self._er.signals_and_keys(entity)
                            if normalized and keys:
                                er_index_triples.extend(
                                    self._er._blocker.index_triples(entity_uri, normalized, keys)
                                )
                        else:
                            # No match — mint a new URI. For ER-enabled types
                            # we add a short signal-hash suffix so two unrelated
                            # humans sharing a name (e.g. two distinct John
                            # Smiths) get distinct URIs and don't quietly
                            # contaminate each other's signal store.
                            if er_applies:
                                import hashlib
                                normalized, keys = self._er.signals_and_keys(entity)
                                if normalized is not None:
                                    fingerprint_parts = [
                                        normalized.email or "",
                                        normalized.phone_e164 or "",
                                        normalized.dob_iso or "",
                                        "|".join(normalized.email_aliases),
                                    ]
                                    fp = hashlib.sha1("|".join(fingerprint_parts).encode("utf-8")).hexdigest()[:8]
                                    entity_uri = f"{entity_uri}-{fp}"
                                if normalized and keys:
                                    er_index_triples.extend(
                                        self._er._blocker.index_triples(entity_uri, normalized, keys)
                                    )
                            else:
                                normalized, keys = self._er.signals_and_keys(entity)
                                if normalized and keys:
                                    er_index_triples.extend(
                                        self._er._blocker.index_triples(entity_uri, normalized, keys)
                                    )
                    except Exception as e:
                        logger.warning("er_pipeline_failed", error=str(e), entity_id=entity.id)

                entity_uri_map[entity.id] = entity_uri
                entity_type_map[entity.id] = resolved_type
                pending_uris.append(entity_uri)
        if er_merged_count:
            logger.info("er_merged_entities", count=er_merged_count, total=len(extraction.entities))

        # Batch existence check: one SPARQL query per 500 URIs instead of N individual ASKs
        existing_uris: set[str] = set()
        BATCH_CHECK_SIZE = 500
        for i in range(0, len(pending_uris), BATCH_CHECK_SIZE):
            batch = pending_uris[i : i + BATCH_CHECK_SIZE]
            sparql = batch_entity_exists_query(instance_graph, batch)
            found = await self._neptune.batch_exists(sparql)
            existing_uris.update(found)
        if existing_uris:
            logger.info("batch_dedup_found", existing=len(existing_uris), total=len(pending_uris))

        # Pass 2: Resolve attributes, validate, collect triples
        # All entity triples are collected into one list, then batch-inserted
        # in a single call. This is ~10-50x faster than per-entity INSERT.
        all_entity_triples: list[tuple[str, str, str]] = []
        # Provenance collector (COG-46): statement-metadata triples for the
        # COMPANION provenance graph accumulate here during entity processing
        # and flush in one batched INSERT below, instead of one awaited
        # Neptune update per entity. Stays empty unless the flag is on.
        all_provenance_triples: list[tuple[str, str, str]] = []
        for entity in extraction.entities:
            if entity.id not in resolved_types:
                continue
            resolved_type = resolved_types[entity.id]
            entity_uri = entity_uri_map[entity.id]
            is_duplicate = entity_uri in existing_uris

            if is_duplicate:
                result.entities_deduplicated += 1

            await self._resolve_and_insert_entity(
                entity, resolved_type, entity_uri, is_duplicate,
                graph_uri, existing_types, existing_attrs, source, result, batch_id,
                _collect_triples=all_entity_triples,
                _collect_provenance=all_provenance_triples,
                also_types=entity_also_types.get(entity.id),
            )

        # Append ER index triples (block keys + denormalized signals) to the
        # same batch so future ingests can find these entities in O(1).
        if er_index_triples:
            all_entity_triples.extend(er_index_triples)

        # Single shared write path (graph/kg_writer.py) — the SAME function the
        # enrichment writer uses: batched instance-triple insert + the companion
        # provenance graph, in one place, so ingestion and enrichment can never
        # drift on HOW facts are written. (Per-fact provenance is flushed in one
        # batched INSERT per ingest, COG-46 — the exact triples a per-entity
        # write would produce; only the write pattern is batched.)
        if all_entity_triples or all_provenance_triples:
            instance_graph = getattr(self, "_instance_graph", graph_uri)
            await insert_facts(
                self._neptune,
                instance_graph,
                all_entity_triples,
                provenance_triples=all_provenance_triples or None,
            )

        # Incrementally embed newly created types for future embedding pre-filter matches
        if result.types_created and self._embedding_service is not None:
            try:
                await self._embedding_service.embed_types(
                    graph_uri, result.types_created, self._neptune,
                )
                logger.info("embedded_new_types", count=len(result.types_created))
            except Exception:
                logger.warning("embed_new_types_failed", exc_info=True)

        # Step 4: Insert relationships (instance triples to instance graph, ontology to base graph)
        instance_graph = getattr(self, "_instance_graph", graph_uri)
        rel_triples: list[tuple[str, str, str]] = []
        for rel in extraction.relationships:
            source_uri = entity_uri_map.get(rel.source_id)
            target_uri = entity_uri_map.get(rel.target_id)
            if source_uri and target_uri:
                # Normalize predicate against existing predicates on this type
                source_type = entity_type_map.get(rel.source_id)
                existing_preds = set()
                if source_type:
                    for attr_name, schema in existing_attrs.get(source_type, {}).items():
                        if schema.datatype not in PRIMITIVE_TYPES:
                            existing_preds.add(attr_name)
                canonical_pred = normalize_predicate(rel.predicate, existing_preds)

                predicate = f"https://cograph.tech/onto/{canonical_pred}"
                rel_triples.append((source_uri, predicate, target_uri))

                # Register relationship as object property in ontology
                target_type = entity_type_map.get(rel.target_id)
                if source_type and target_type:
                    type_attrs = existing_attrs.get(source_type, {})
                    existing = type_attrs.get(canonical_pred)
                    if existing is None:
                        sparql = insert_attribute(
                            graph_uri, source_type, canonical_pred, "", target_type,
                        )
                        await self._neptune.update(sparql)
                        result.attributes_added.append(f"{source_type}.{canonical_pred}")
                        existing_attrs.setdefault(source_type, {})[canonical_pred] = AttributeSchema(
                            name=canonical_pred, datatype=target_type,
                        )
                    elif existing.datatype in PRIMITIVE_TYPES:
                        # First seen as a primitive attribute, now carrying an
                        # entity object: upgrade its ontology range to the target
                        # type so the schema-only Explorer overview draws the edge
                        # (the detail view already shows it from instance data).
                        await self._neptune.update(
                            set_object_property_range(
                                graph_uri, source_type, canonical_pred, target_type,
                            )
                        )
                        existing_attrs[source_type][canonical_pred] = AttributeSchema(
                            name=canonical_pred, datatype=target_type,
                        )

        # Batch insert relationship triples
        if rel_triples:
            for sparql in batched_insert_triples(instance_graph, rel_triples):
                await self._neptune.update(sparql)
            result.triples_inserted += len(rel_triples)

        result.entities_resolved = len(entity_uri_map)
        logger.info(
            "ingest_complete",
            entities_resolved=result.entities_resolved,
            triples_inserted=result.triples_inserted,
            types_created=result.types_created,
            rejections=len(result.rejections),
        )
        return result

    async def _ingest_csv(
        self,
        content: str,
        graph_uri: str,
        existing_types: dict[str, str],
        existing_attrs: dict[str, dict[str, AttributeSchema]],
        source: str,
    ) -> IngestResult:
        """CSV ingestion: 1 LLM call for schema inference, deterministic mapping for all rows."""
        import csv
        import io
        from cograph_client.resolver.csv_resolver import CSVResolver

        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        if not rows:
            return IngestResult(entities_extracted=0)

        headers = list(rows[0].keys())
        logger.info("csv_ingest_start", rows=len(rows), columns=len(headers))

        # Step 1: Infer schema from sample (1 LLM call)
        csv_resolver = CSVResolver(self._anthropic, self._openrouter_key)
        mapping = await csv_resolver.infer_schema(headers, rows[:10], existing_types, total_rows=len(rows))

        # Step 2+: apply the mapping and run the shared resolve→dedup→insert
        # tail (also reused by web-discovery ingest via ingest_mapped_records).
        return await self._ingest_mapped(
            mapping, rows, graph_uri, existing_types, existing_attrs, source,
        )

    async def ingest_mapped_records(
        self,
        rows: list[dict[str, str]],
        mapping: CSVSchemaMapping,
        tenant_id: str,
        source: str = "",
        instance_graph: str | None = None,
    ) -> IngestResult:
        """Ingest pre-mapped records (no schema inference) — the fixed-mapping seam.

        A caller infers a :class:`CSVSchemaMapping` once (e.g. from a sample at
        plan time) and applies that SAME mapping to the full record set here. The
        mapping is applied DETERMINISTICALLY (no LLM, no re-inference), so the
        schema previewed to the user is exactly the schema committed
        (preview == commit). This is the CSV path's guarantee; the web-DISCOVERY
        path instead routes through :meth:`ingest` (the non-deterministic
        ``_extract``), where the previewed shape is only a sample-based estimate,
        not an exact match. Records flow through the identical type-resolution,
        batch existence-dedup, ER and batch-insert path CSV ingest uses.

        Mirrors :meth:`ingest`'s per-call setup (instance graph, type-matcher
        graph URI, ontology + parent-map fetch) so it can be called standalone,
        not only inside the CSV pipeline.
        """
        graph_uri = tenant_graph_uri(tenant_id)
        # Ontology always goes to the base tenant graph; instance data goes to
        # instance_graph when a specific KG is targeted, else the base graph.
        self._instance_graph = instance_graph or graph_uri
        self._type_matcher._graph_uri = graph_uri
        existing_types, existing_attrs = await self._fetch_ontology(graph_uri)
        self._parent_of = await self._fetch_parent_map(graph_uri)
        return await self._ingest_mapped(
            mapping, rows, graph_uri, existing_types, existing_attrs, source,
        )

    async def _ingest_mapped(
        self,
        mapping: CSVSchemaMapping,
        rows: list[dict[str, str]],
        graph_uri: str,
        existing_types: dict[str, str],
        existing_attrs: dict[str, dict[str, AttributeSchema]],
        source: str,
    ) -> IngestResult:
        """Apply a pre-inferred mapping to rows and run the resolve→insert tail.

        Extracted verbatim from the former ``_ingest_csv`` body (Step 2 onward)
        so CSV ingest and web-discovery ingest commit through one code path.
        """
        from cograph_client.resolver.csv_resolver import CSVResolver

        # Step 2: Apply mapping deterministically to ALL rows (no LLM)
        applied = CSVResolver.apply_mapping(mapping, rows)
        entities, relationships = applied.entities, applied.relationships

        # Step 3: Resolve entities + insert in batches
        batch_id = str(uuid4())
        result = IngestResult(
            entities_extracted=len(entities),
            chunks_processed=1,
            batch_id=batch_id,
            # Row-conservation accounting (ADR 0003 §2).
            rows_in=applied.rows_in,
            rows_dropped=applied.rows_dropped,
            drops_by_entity=applied.drops_by_entity,
        )
        entity_uri_map: dict[str, str] = {}
        entity_type_map: dict[str, str] = {}

        try:
            # Pass 1: Resolve types and compute URIs
            pending_uris: list[str] = []
            resolved_types: dict[str, str] = {}
            for i, entity in enumerate(entities):
                if i > 0 and i % self.ONTOLOGY_REFRESH_INTERVAL == 0:
                    await self._refresh_ontology(graph_uri, existing_types, existing_attrs)

                resolved_type = await self._resolve_type(
                    entity, graph_uri, existing_types, existing_attrs, result,
                )
                if resolved_type:
                    resolved_types[entity.id] = resolved_type
                    entity_uri = f"https://cograph.tech/entities/{resolved_type}/{_safe_id(entity.id)}"
                    entity_uri_map[entity.id] = entity_uri
                    entity_type_map[entity.id] = resolved_type
                    pending_uris.append(entity_uri)

            # Batch existence check
            instance_graph = getattr(self, "_instance_graph", graph_uri)
            existing_uris: set[str] = set()
            BATCH_CHECK_SIZE = 500
            for i in range(0, len(pending_uris), BATCH_CHECK_SIZE):
                batch = pending_uris[i : i + BATCH_CHECK_SIZE]
                sparql = batch_entity_exists_query(instance_graph, batch)
                found = await self._neptune.batch_exists(sparql)
                existing_uris.update(found)
            if existing_uris:
                logger.info("csv_batch_dedup_found", existing=len(existing_uris), total=len(pending_uris))

            # Pass 2: Resolve attributes and insert
            for entity in entities:
                if entity.id not in resolved_types:
                    continue
                resolved_type = resolved_types[entity.id]
                entity_uri = entity_uri_map[entity.id]
                is_duplicate = entity_uri in existing_uris
                if is_duplicate:
                    result.entities_deduplicated += 1
                await self._resolve_and_insert_entity(
                    entity, resolved_type, entity_uri, is_duplicate,
                    graph_uri, existing_types, existing_attrs, source, result, batch_id,
                )

            # Step 4: Batch-insert relationships
            rel_triples: list[tuple[str, str, str]] = []
            for rel in relationships:
                source_uri = entity_uri_map.get(rel.source_id)
                target_uri = entity_uri_map.get(rel.target_id)
                if source_uri and target_uri:
                    # Normalize predicate against existing predicates on this type
                    source_type = entity_type_map.get(rel.source_id)
                    existing_preds = set()
                    if source_type:
                        for attr_name, schema in existing_attrs.get(source_type, {}).items():
                            if schema.datatype not in PRIMITIVE_TYPES:
                                existing_preds.add(attr_name)
                    canonical_pred = normalize_predicate(rel.predicate, existing_preds)

                    predicate = f"https://cograph.tech/onto/{canonical_pred}"
                    rel_triples.append((source_uri, predicate, target_uri))

                    # Register relationship as object property in ontology
                    target_type = entity_type_map.get(rel.target_id)
                    if source_type and target_type:
                        type_attrs = existing_attrs.get(source_type, {})
                        existing = type_attrs.get(canonical_pred)
                        if existing is None:
                            sparql = insert_attribute(graph_uri, source_type, canonical_pred, "", target_type)
                            await self._neptune.update(sparql)
                            result.attributes_added.append(f"{source_type}.{canonical_pred}")
                            existing_attrs.setdefault(source_type, {})[canonical_pred] = AttributeSchema(
                                name=canonical_pred, datatype=target_type,
                            )
                        elif existing.datatype in PRIMITIVE_TYPES:
                            # Upgrade a primitive attribute to a relationship range
                            # so the Explorer overview draws the edge (see entity
                            # ingest path above for the full rationale).
                            await self._neptune.update(
                                set_object_property_range(
                                    graph_uri, source_type, canonical_pred, target_type,
                                )
                            )
                            existing_attrs[source_type][canonical_pred] = AttributeSchema(
                                name=canonical_pred, datatype=target_type,
                            )

            for sparql in batched_insert_triples(graph_uri, rel_triples):
                await self._neptune.update(sparql)
            result.triples_inserted += len(rel_triples)

            result.entities_resolved = len(entity_uri_map)
            logger.info(
                "csv_ingest_complete",
                rows=len(rows),
                entities=result.entities_resolved,
                triples=result.triples_inserted,
                types=result.types_created,
            )
            return result

        except Exception:
            logger.error(
                "csv_ingest_failed_rolling_back",
                batch_id=batch_id,
                entities_so_far=result.entities_resolved,
                exc_info=True,
            )
            instance_graph = getattr(self, "_instance_graph", graph_uri)
            try:
                sparql = delete_batch_query(instance_graph, batch_id)
                await self._neptune.update(sparql)
                logger.info("csv_batch_rollback_complete", batch_id=batch_id)
            except Exception:
                logger.error("csv_batch_rollback_failed", batch_id=batch_id, exc_info=True)
            raise

    async def _extract(
        self, content: str, content_type: str, existing_types: dict[str, str] | None = None,
    ) -> ExtractionResult:
        """Extract entities and relationships from raw content."""
        if existing_types:
            types_str = "\n".join(f"- {name}" for name in existing_types)
        else:
            types_str = "(none — this is a fresh ontology)"

        user_content = EXTRACTION_USER_TEMPLATE.format(
            content=content,
            existing_types=types_str,
        )

        truncated = False
        if self.EXTRACT_PROVIDER == "openrouter" and self._openrouter_key:
            text = await self._extract_via_openrouter(user_content)
        else:
            msg = await self._anthropic.messages.create(
                model=self.INFER_MODEL,
                max_tokens=self.EXTRACT_MAX_TOKENS,
                system=EXTRACTION_SYSTEM,
                messages=[{"role": "user", "content": user_content}],
            )
            text = msg.content[0].text
            # Explicit truncation signal from the Anthropic SDK: the model hit
            # the token ceiling mid-output, so the JSON is almost certainly
            # incomplete. Surface it so a JSON chunk can be split + retried
            # instead of silently dropping the whole batch. (openrouter_chat
            # doesn't expose finish_reason; there the json-parse-failure path
            # below is the truncation signal.)
            if getattr(msg, "stop_reason", None) == "max_tokens":
                truncated = True

        try:
            # Strip code fences if present
            stripped = text.strip()
            if stripped.startswith("```"):
                lines = [l for l in stripped.split("\n") if not l.strip().startswith("```")]
                stripped = "\n".join(lines)
            data = json.loads(stripped)
            entities = [ExtractedEntity(**e) for e in data.get("entities", [])]
            relationships = [ExtractedRelationship(**r) for r in data.get("relationships", [])]
            return ExtractionResult(
                entities=entities,
                relationships=relationships,
                source_text=content,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # A parse failure on a TRUNCATED response is the expected symptom of
            # the output exceeding max_tokens (the recovery loop will split +
            # retry); log it distinctly so it isn't mistaken for a malformed
            # model reply.
            logger.warning(
                "extraction_parse_error",
                error=str(e),
                truncated=truncated,
                raw=text[:500],
            )
            return ExtractionResult(source_text=content)

    async def _extract_via_openrouter(self, user_content: str) -> str:
        """Extract entities via OpenRouter, with primary→fallback routing."""
        return await openrouter_chat(
            self._openrouter_key,
            EXTRACTION_SYSTEM,
            user_content,
            model=self.EXTRACT_MODEL,
            temperature=0,
            max_tokens=self.EXTRACT_MAX_TOKENS,
            timeout=60,
        )

    # Floor below which a JSON chunk is no longer worth splitting: a handful of
    # records can't overflow max_tokens, so a still-empty extraction is a genuine
    # extraction failure to account for, not a truncation to recover.
    _RECOVERY_MIN_RECORDS = 3

    async def _extract_json_chunk_with_recovery(
        self, chunk: str, existing_types: dict[str, str],
    ) -> tuple[ExtractionResult, int]:
        """Extract one JSON-array chunk, RECOVERING from a silent batch loss.

        The reification/lift prompt makes each record emit many entities +
        relationships, so a dense chunk's JSON output can exceed the model's
        ``max_tokens``, get truncated, fail to parse, and return an EMPTY
        :class:`ExtractionResult` — silently dropping every record in the chunk.

        When that happens (zero entities extracted from a chunk that actually
        held records) we SPLIT the chunk's JSON array in half and retry each
        half, recursing down to :attr:`_RECOVERY_MIN_RECORDS`. Smaller chunks
        produce smaller outputs that fit under the cap. If a minimal chunk still
        yields nothing it is a real extraction failure: we log at ERROR and
        return its record count as ``dropped`` so the caller can surface it in
        row-conservation accounting instead of presenting the run as complete.

        Returns ``(merged_extraction, dropped_record_count)``.
        """
        from cograph_client.resolver.chunker import split_json_array_chunk, json_array_len

        extraction = await self._extract(chunk, "json", existing_types)
        n_records = json_array_len(chunk)
        # Success, or a genuinely empty chunk (no records to lose) → nothing to recover.
        if extraction.entities or n_records == 0:
            return extraction, 0

        # Too small to split further: a few records can't overflow the token
        # cap, so this is a real extraction failure — account for the loss.
        if n_records <= self._RECOVERY_MIN_RECORDS:
            logger.error(
                "extraction_chunk_dropped",
                records=n_records,
                reason="empty_extraction_at_min_chunk",
            )
            return extraction, n_records

        halves = split_json_array_chunk(chunk)
        if not halves:
            # Couldn't split (not a parseable array) — count the loss.
            logger.error("extraction_chunk_dropped", records=n_records, reason="unsplittable")
            return extraction, n_records

        logger.warning(
            "extraction_chunk_split_retry", records=n_records, halves=len(halves),
        )
        merged_entities: list[ExtractedEntity] = []
        merged_relationships: list[ExtractedRelationship] = []
        seen_ids: set[str] = set()
        total_dropped = 0
        for half in halves:
            sub_extraction, sub_dropped = await self._extract_json_chunk_with_recovery(
                half, existing_types,
            )
            total_dropped += sub_dropped
            for e in sub_extraction.entities:
                if e.id not in seen_ids:
                    merged_entities.append(e)
                    seen_ids.add(e.id)
            merged_relationships.extend(sub_extraction.relationships)
        return (
            ExtractionResult(
                entities=merged_entities,
                relationships=merged_relationships,
                source_text=chunk[:500],
            ),
            total_dropped,
        )

    async def _fetch_ontology(
        self, graph_uri: str
    ) -> tuple[dict[str, str], dict[str, dict[str, AttributeSchema]]]:
        """Fetch existing types and attributes from Neptune.

        Returns:
            (types: {name: description}, attrs: {type_name: {attr_name: schema}})
        """
        try:
            raw = await self._neptune.query(get_full_ontology_query(graph_uri))
            _, bindings = parse_sparql_results(raw)
        except Exception:
            logger.warning("ontology_fetch_failed", exc_info=True)
            return {}, {}

        types: dict[str, str] = {}
        attrs: dict[str, dict[str, AttributeSchema]] = {}

        for row in bindings:
            type_label = row.get("typeLabel", "")
            if not type_label:
                continue
            if type_label not in types:
                types[type_label] = ""
                attrs[type_label] = {}
            if row.get("attrLabel"):
                range_str = row.get("range", "")
                type_uri_prefix = "https://cograph.tech/types/"
                if range_str.startswith(type_uri_prefix):
                    # Range is a reference to another ontology type
                    datatype = range_str[len(type_uri_prefix):]
                elif "#" in range_str:
                    fragment = range_str.split("#")[-1]
                    # Map XSD names to our datatype names
                    dt_map = {
                        "string": "string", "integer": "integer", "float": "float",
                        "boolean": "boolean", "dateTime": "datetime", "Resource": "uri",
                    }
                    datatype = dt_map.get(fragment, "string")
                else:
                    datatype = "string"
                attrs[type_label][row["attrLabel"]] = AttributeSchema(
                    name=row["attrLabel"], datatype=datatype,
                )

        return types, attrs

    async def _fetch_parent_map(
        self, graph_uri: str, layer_stack: LayerStack | None = None
    ) -> dict[str, str]:
        """Fetch the child->parent subclass map (keyed by type *name*).

        Reads every rdfs:subClassOf edge via parent_map_query and reduces each
        URI to its type name so it can feed the pure hierarchy helpers
        (ancestor_chain / config_for_with_hierarchy). Returns {} on any error —
        callers degrade to flat (zero-hierarchy) behavior.

        Layer-aware variant (ADR 0002 §1, COG-37): pass a LayerStack and the
        edges are read from the UNION of the tenant's visible layer graphs in
        one query — subClassOf edges may span layers (a tenant leaf under a
        Public parent). Duplicate child names are resolved by shadowing: edges
        from higher-precedence layers (Tenant > Enhanced > Public) win. With
        no layer_stack the single-graph behavior is exactly as before.
        """
        if layer_stack is None:
            try:
                raw = await self._neptune.query(parent_map_query(graph_uri))
                _, bindings = parse_sparql_results(raw)
            except Exception:
                logger.warning("parent_map_fetch_failed", exc_info=True)
                return {}
            return self._parent_map_from_bindings(bindings)

        try:
            raw = await self._neptune.query(
                parent_map_query(layer_stack.visible_graph_uris())
            )
            _, bindings = parse_sparql_results(raw)
        except Exception:
            logger.warning("parent_map_fetch_failed", exc_info=True)
            return {}

        rows_by_graph: dict[str, list[dict]] = {}
        for row in bindings:
            rows_by_graph.setdefault(row.get("graph", ""), []).append(row)
        # Merge lowest-precedence layer first so higher layers overwrite
        # duplicate child keys — Tenant > Enhanced > Public shadowing.
        parent_of: dict[str, str] = {}
        for g in reversed(layer_stack.visible_graph_uris()):
            parent_of.update(self._parent_map_from_bindings(rows_by_graph.get(g, [])))
        return parent_of

    @staticmethod
    def _parent_map_from_bindings(bindings: list[dict]) -> dict[str, str]:
        """Reduce ?child/?parent URI bindings to a {child_name: parent_name} map.

        Names are extracted via type_name_from_uri, which understands every
        layer namespace — so a tenant-graph edge whose PARENT is a Public-layer
        URI (`types/public/Person`) keys correctly instead of being dropped.
        Edges with either end outside all layer namespaces are skipped, as are
        self-edges.
        """
        parent_of: dict[str, str] = {}
        for row in bindings:
            child_name = type_name_from_uri(row.get("child", ""))
            parent_name = type_name_from_uri(row.get("parent", ""))
            if child_name and parent_name and child_name != parent_name:
                parent_of[child_name] = parent_name
        return parent_of

    async def _synthesize_ancestors(
        self,
        child_type: str,
        parent_type: str | None,
        graph_uri: str,
        existing_types: dict[str, str],
        existing_attrs: dict[str, dict[str, AttributeSchema]],
        result: IngestResult,
        parent_chain: list[str] | None = None,
        emit_child_edge: bool = False,
    ) -> None:
        """Close the rdfs:subClassOf lineage from `child_type` up to the nearest
        existing root (ADR 0001 rule 3).

        `parent_type` is the immediate parent (may be None when only an extractor
        chain is available). `parent_chain` is the extractor's full ancestor list
        for `child_type`, most-specific first — seeding it lets a brand-new
        MULTI-LEVEL lineage (e.g. Condo < Property < Asset, all new) close in a
        single pass. `emit_child_edge=True` makes this method emit the
        child->immediate-parent subClassOf edge itself; callers that already
        emitted it (the SUBTYPE branches) pass False to avoid a redundant write.

        For each ancestor NOT yet in existing_types, emits insert_type +
        insert_subtype and registers it in existing_types / existing_attrs /
        result.types_created. Idempotent: ancestors already present are skipped.
        """
        from cograph_client.resolver.er import ancestor_chain

        parent_chain = parent_chain or []
        # Immediate parent: explicit hint wins; otherwise top of the extractor chain.
        if not parent_type:
            parent_type = parent_chain[0] if parent_chain else None
        if not parent_type:
            return

        # Record the child->parent edge so later entities in this batch can climb it.
        if child_type and child_type != parent_type:
            self._parent_of[child_type] = parent_type
        # Seed the deeper extractor lineage (ancestors of child, most-specific
        # first) without clobbering edges already recorded (setdefault).
        prev = child_type
        for anc in parent_chain:
            if prev and anc and prev != anc:
                self._parent_of.setdefault(prev, anc)
            prev = anc

        # Brand-new lineage: the caller couldn't link child->parent because the
        # parent didn't exist yet. Emit that edge here.
        if emit_child_edge and child_type and child_type != parent_type:
            await self._neptune.update(insert_subtype(graph_uri, parent_type, child_type))

        # Walk root-ward from the immediate parent. ancestor_chain is cycle-guarded.
        chain = ancestor_chain(parent_type, self._parent_of)
        for i, ancestor in enumerate(chain):
            grandparent = chain[i + 1] if i + 1 < len(chain) else None
            if ancestor not in existing_types:
                await self._neptune.update(insert_type(graph_uri, ancestor, ""))
                if grandparent:
                    await self._neptune.update(insert_subtype(graph_uri, grandparent, ancestor))
                    self._parent_of[ancestor] = grandparent
                result.types_created.append(ancestor)
                existing_types[ancestor] = ""
                existing_attrs[ancestor] = {}

    async def _link_parent(
        self,
        entity: ExtractedEntity,
        graph_uri: str,
        existing_types: dict[str, str],
        existing_attrs: dict[str, dict[str, AttributeSchema]],
        result: IngestResult,
    ) -> None:
        """Attach a freshly-created type to its parent lineage.

        Two cases:
        - immediate parent already exists → link directly, then synthesize any
          deeper ancestors the extractor named (parent_chain);
        - brand-new lineage (parent not in the ontology, or only a parent_chain) →
          let _synthesize_ancestors create every missing ancestor AND the
          child->parent edge (emit_child_edge=True). This closes a fully-new
          multi-level chain like Condo < Property < Asset in one row (ADR rule 3).
        """
        pt = entity.parent_type
        linked_as_subtype = False
        if pt and pt in existing_types:
            # Immediate parent exists — link directly, then synthesize any deeper
            # ancestors the extractor named.
            await self._neptune.update(insert_subtype(graph_uri, pt, entity.type_name))
            await self._synthesize_ancestors(
                entity.type_name, pt, graph_uri, existing_types, existing_attrs, result,
                parent_chain=entity.parent_chain,
            )
            logger.info("type_new_with_parent", child=entity.type_name, parent=pt)
            linked_as_subtype = True
        elif entity.parent_chain:
            # Brand-new lineage. We DON'T trust a parent_type that names a
            # non-existing type (preserves the "parent_type must be existing"
            # contract); the full chain comes from parent_chain instead.
            await self._synthesize_ancestors(
                entity.type_name, None, graph_uri, existing_types, existing_attrs, result,
                parent_chain=entity.parent_chain, emit_child_edge=True,
            )
            logger.info(
                "type_new_lineage", child=entity.type_name, parent=entity.parent_chain[0],
            )
            linked_as_subtype = True

        # The caller's top-level mint wrote NO comment (FIX 3): subtype_description
        # may only describe a real subtype. Now that a parent linkage has made
        # this type a genuine subtype, write the description here. Use the
        # COMMENT-ONLY upsert: the subClassOf edge was just created above (by
        # insert_subtype / _synthesize_ancestors), and plain upsert_type would
        # DELETE it (it clears subClassOf when no parent_type is passed) — the
        # new-parent-edge bug. upsert_type_comment touches only rdfs:comment, so
        # the edge survives while the description stays idempotent on re-ingest.
        if linked_as_subtype and entity.subtype_description:
            await self._neptune.update(
                upsert_type_comment(graph_uri, entity.type_name, entity.subtype_description)
            )

    async def _refresh_ontology(
        self,
        graph_uri: str,
        existing_types: dict[str, str],
        existing_attrs: dict[str, dict[str, AttributeSchema]],
    ) -> None:
        """Re-fetch ontology from Neptune and merge into in-memory state.

        Additive merge only: new types/attrs from concurrent ingestions are added,
        but nothing is removed (this ingestion may have added types not yet visible).
        """
        fresh_types, fresh_attrs = await self._fetch_ontology(graph_uri)
        added = 0
        for t, desc in fresh_types.items():
            if t not in existing_types:
                existing_types[t] = desc
                added += 1
        for t, attrs in fresh_attrs.items():
            if t not in existing_attrs:
                existing_attrs[t] = attrs
            else:
                for a, schema in attrs.items():
                    if a not in existing_attrs[t]:
                        existing_attrs[t][a] = schema
        if added:
            logger.info("ontology_refreshed", new_types=added)

    async def _resolve_also_types(
        self,
        entity: ExtractedEntity,
        primary_resolved: str,
        graph_uri: str,
        existing_types: dict[str, str],
        existing_attrs: dict[str, dict[str, AttributeSchema]],
        result: IngestResult,
    ) -> list[str]:
        """Resolve genuine co-classifications (entity.also_types) so each exists
        in the ontology (ADR rule 1). Returns the resolved co-type names, deduped.

        Skips any co-type that is actually in the primary's subClassOf lineage
        (an ancestor or descendant) — those are recovered by query-time closure,
        not asserted. Only genuinely INDEPENDENT types are returned.
        """
        if not entity.also_types:
            return []
        from cograph_client.resolver.er import ancestor_chain

        resolved: list[str] = []
        seen = {primary_resolved}
        for co in entity.also_types:
            if not co:
                continue
            proxy = ExtractedEntity(type_name=co, id=entity.id)
            rt = await self._resolve_type(
                proxy, graph_uri, existing_types, existing_attrs, result,
            )
            if not rt or rt in seen:
                continue
            # Same-lineage guard: skip if one is an ancestor of the other.
            if rt in ancestor_chain(primary_resolved, self._parent_of) or \
               primary_resolved in ancestor_chain(rt, self._parent_of):
                logger.info("also_type_in_lineage_skipped", primary=primary_resolved, co_type=rt)
                continue
            resolved.append(rt)
            seen.add(rt)
        return resolved

    async def _mint_subtype(
        self, graph_uri: str, type_name: str, subtype_description: str | None,
    ) -> None:
        """Create a NEW subtype's type declaration, carrying its description
        idempotently (FIX 3 + FIX 4).

        When a ``subtype_description`` is present it is written via
        :func:`upsert_type_comment`, which REPLACES the single-valued
        ``rdfs:comment`` instead of appending — so re-minting the same subtype
        across ingests can't accumulate duplicate comments — while leaving
        ``rdfs:subClassOf`` untouched (plain :func:`upsert_type` would CLEAR the
        edge a caller's ``insert_subtype`` creates). With no description we emit a
        plain ``insert_type`` (no comment), keeping the common no-description write
        byte-identical to before.
        """
        if subtype_description:
            await self._neptune.update(upsert_type_comment(graph_uri, type_name, subtype_description))
        else:
            await self._neptune.update(insert_type(graph_uri, type_name, ""))

    async def _resolve_type(
        self,
        entity: ExtractedEntity,
        graph_uri: str,
        existing_types: dict[str, str],
        existing_attrs: dict[str, dict[str, AttributeSchema]],
        result: IngestResult,
    ) -> str | None:
        """Pass 1: Resolve the type for an entity. Returns resolved type name or None."""
        if entity.type_name in existing_types:
            return entity.type_name
        elif entity.same_as and entity.same_as in existing_types:
            match = await self._type_matcher.match(entity.type_name, "", existing_types)
            if match.verdict == MatchVerdict.SAME:
                logger.info("type_same_as_verified", proposed=entity.type_name, resolved=match.resolved)
                return match.resolved
            elif match.verdict == MatchVerdict.SUBTYPE:
                # SUBTYPE branch — subtype_description legitimately describes this
                # NEW subtype (FIX 3). Written idempotently (FIX 4): upsert
                # REPLACES the single-valued rdfs:comment so re-minting the same
                # type across ingests can't accumulate duplicate comments.
                await self._mint_subtype(graph_uri, entity.type_name, entity.subtype_description)
                sparql = insert_subtype(graph_uri, match.parent_type, entity.type_name)
                await self._neptune.update(sparql)
                logger.info("type_same_as_was_subtype", child=entity.type_name, parent=match.parent_type)
                result.types_created.append(entity.type_name)
                existing_types[entity.type_name] = ""
                existing_attrs[entity.type_name] = {}
                await self._synthesize_ancestors(
                    entity.type_name, match.parent_type, graph_uri,
                    existing_types, existing_attrs, result,
                    parent_chain=entity.parent_chain,
                )
                return entity.type_name
            elif match.inconclusive:
                # Verifier couldn't reach a real decision (e.g. LLM unavailable).
                # Trust the extractor's explicit same_as rather than fabricating a
                # duplicate type — creating "Home" alongside "Property" is exactly
                # the ontology pollution this verification step exists to prevent.
                logger.info("type_same_as_trusted", proposed=entity.type_name, resolved=entity.same_as)
                return entity.same_as
            else:
                # same_as REJECTED → this is a genuine TOP-LEVEL type, not a
                # subtype. subtype_description must NOT be written here (FIX 3):
                # the field's contract is "describes a NEW SUBTYPE" only.
                sparql = insert_type(graph_uri, entity.type_name, "")
                await self._neptune.update(sparql)
                logger.info("type_same_as_rejected", proposed=entity.type_name, claimed=entity.same_as)
                result.types_created.append(entity.type_name)
                existing_types[entity.type_name] = ""
                existing_attrs[entity.type_name] = {}
                return entity.type_name
        else:
            match = await self._type_matcher.match(entity.type_name, "", existing_types)
            if match.verdict == MatchVerdict.SAME:
                logger.info("type_matched_existing", proposed=entity.type_name, resolved=match.resolved)
                return match.resolved
            elif match.verdict == MatchVerdict.SUBTYPE:
                # SUBTYPE branch — subtype_description describes this NEW subtype
                # (FIX 3), written idempotently via upsert (FIX 4).
                await self._mint_subtype(graph_uri, entity.type_name, entity.subtype_description)
                sparql = insert_subtype(graph_uri, match.parent_type, entity.type_name)
                await self._neptune.update(sparql)
                logger.info("type_subtype", child=entity.type_name, parent=match.parent_type)
                result.types_created.append(entity.type_name)
                existing_types[entity.type_name] = ""
                existing_attrs[entity.type_name] = {}
                await self._synthesize_ancestors(
                    entity.type_name, match.parent_type, graph_uri,
                    existing_types, existing_attrs, result,
                    parent_chain=entity.parent_chain,
                )
                return entity.type_name
            elif match.verdict == MatchVerdict.FLAGGED:
                # Top-level mint: do NOT write subtype_description here (FIX 3).
                # If _link_parent then establishes a parent (the entity carried a
                # parent_type/parent_chain), it upserts the description there —
                # the only place the type is actually a subtype.
                sparql = insert_type(graph_uri, entity.type_name, "")
                await self._neptune.update(sparql)
                result.types_created.append(entity.type_name)
                existing_types[entity.type_name] = ""
                existing_attrs[entity.type_name] = {}
                await self._link_parent(entity, graph_uri, existing_types, existing_attrs, result)
                logger.warning("type_flagged_for_review", proposed=entity.type_name)
                result.flagged_types.append(entity.type_name)
                return entity.type_name
            else:
                # Top-level mint: no subtype_description here (FIX 3). _link_parent
                # upserts it iff this turns out to be a subtype (parent_chain).
                sparql = insert_type(graph_uri, entity.type_name, "")
                await self._neptune.update(sparql)
                result.types_created.append(entity.type_name)
                existing_types[entity.type_name] = ""
                existing_attrs[entity.type_name] = {}
                await self._link_parent(entity, graph_uri, existing_types, existing_attrs, result)
                # Governance seam: the genuinely-new type MAY also be proposed
                # for the Global-Public layer. No-op unless the flag is on.
                await self._maybe_govern_new_type(entity, graph_uri)
                return entity.type_name

    async def _maybe_govern_new_type(self, entity: ExtractedEntity, graph_uri: str) -> None:
        """Governance seam (ADR 0002 §2, COG-43): propose a brand-new type for
        the shared Global-Public layer and, on majority judge approval, write
        a governed copy there with provenance + changelog.

        The tenant-layer write has ALREADY happened (today's behavior — the
        tenant uses the type immediately whatever the verdict); approval only
        ADDS a Public-layer copy.

        Scheduling (COG-46): the judge panel + Public-layer write run as a
        BACKGROUND task — ingest never waits on LLM judges. Semantics are
        eventually consistent: an approved type appears in the Public layer
        shortly AFTER ingest returns. Task references are retained on
        ``self._governance_tasks``; await :meth:`drain_governance` to
        deterministically wait for all scheduled outcomes. Best-effort: any
        failure (scheduling or in-task) is logged and never blocks or crashes
        ingest. No-op when COGRAPH_GOVERNANCE_ENABLED is off (default).
        """
        if not self._governance_enabled:
            return
        from cograph_client.resolver.governance import TypeProposal
        try:
            graphs_prefix = "https://cograph.tech/graphs/"
            tenant_id = (
                graph_uri[len(graphs_prefix):] if graph_uri.startswith(graphs_prefix) else graph_uri
            )
            proposal = TypeProposal(
                type_name=entity.type_name,
                parent_chain=list(entity.parent_chain),
                tenant_id=tenant_id,
                reasoning=(
                    f"Extractor proposed brand-new type '{entity.type_name}' "
                    f"matching no existing ontology type"
                ),
                proposer_model=self.EXTRACT_MODEL,
            )
            # Drop references to finished tasks so the list stays bounded on
            # long-lived resolvers, then schedule the panel off the ingest path.
            self._governance_tasks = [t for t in self._governance_tasks if not t.done()]
            self._governance_tasks.append(
                asyncio.create_task(self._govern_in_background(proposal))
            )
        except Exception:
            logger.warning("governance_failed", type_name=entity.type_name, exc_info=True)

    async def _govern_in_background(self, proposal) -> None:
        """Run propose-and-judge + the Public-layer write off the ingest path
        (COG-46). Exceptions are logged and swallowed here, inside the task —
        a governance failure never crashes ingest and never surfaces as an
        unretrieved task exception.
        """
        try:
            decision = await self._governance.propose_and_judge(proposal, self._judge_panel)
            if decision.approved:
                await self._governance.write_governed_type(proposal, decision)
            else:
                logger.info("governance_type_tenant_only", type_name=proposal.type_name)
        except Exception:
            logger.warning("governance_failed", type_name=proposal.type_name, exc_info=True)

    async def drain_governance(self) -> None:
        """Await all pending background governance tasks (COG-46).

        Governance is eventually consistent: :meth:`_maybe_govern_new_type`
        schedules the judge panel + Public-layer write as background tasks,
        so an approved type appears in the Public layer shortly after ingest
        returns. Call this to deterministically wait for every scheduled
        outcome — tests, and callers that need the Public layer settled
        before reading it. Safe to call any time (no-op with nothing
        pending). Task failures were already logged inside the tasks and are
        never re-raised here.
        """
        tasks, self._governance_tasks = self._governance_tasks, []
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _resolve_and_insert_entity(
        self,
        entity: ExtractedEntity,
        resolved_type: str,
        entity_uri: str,
        is_duplicate: bool,
        graph_uri: str,
        existing_types: dict[str, str],
        existing_attrs: dict[str, dict[str, AttributeSchema]],
        source: str,
        result: IngestResult,
        batch_id: str = "",
        _collect_triples: list[tuple[str, str, str]] | None = None,
        _collect_provenance: list[tuple[str, str, str]] | None = None,
        also_types: list[str] | None = None,
    ) -> None:
        """Pass 2: Resolve attributes, validate, and collect triples for one entity.

        If _collect_triples is provided, triples are appended to that list instead of
        being inserted immediately. The caller is responsible for batch-inserting them.
        This is ~10-50x faster because it avoids per-entity Neptune INSERT calls.

        If _collect_provenance is provided (COG-46), per-fact provenance triples
        (when COGRAPH_PROVENANCE_ENABLED is on) are likewise appended for the
        caller to flush in one batched INSERT into the companion provenance
        graph, instead of being inserted here per entity.

        `also_types` are genuine independent co-classifications (ADR rule 1): each
        gets its own asserted rdf:type triple alongside the primary resolved_type.
        """
        type_attrs = existing_attrs.get(resolved_type, {})

        # Option D promotions
        promotions = check_promotion(entity, type_attrs)
        promoted_type_names: set[str] = set()
        for promo in promotions:
            if promo.promoted_type and promo.promoted_type not in promoted_type_names:
                promoted_type_names.add(promo.promoted_type)

        for ptype in promoted_type_names:
            if ptype not in existing_types:
                sparql = insert_type(graph_uri, ptype, f"Promoted from {resolved_type} attributes")
                await self._neptune.update(sparql)
                result.types_created.append(ptype)
                existing_types[ptype] = ""
                existing_attrs[ptype] = {}

        rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        rdfs_label = "http://www.w3.org/2000/01/rdf-schema#label"

        # Duplicate entities skip rdf:type triple but still merge attributes
        if is_duplicate:
            triples_to_insert: list[tuple[str, str, str]] = []
        else:
            triples_to_insert: list[tuple[str, str, str]] = [
                (entity_uri, rdf_type, type_uri(resolved_type)),
                (entity_uri, rdfs_label, entity.id),
            ]
            # Multi-typing: emit an additional asserted rdf:type per genuine
            # co-classification (ADR rule 1). Ancestors are NOT asserted here —
            # they are recovered via query-time subclass closure.
            for co_type in (also_types or ()):
                if co_type and co_type != resolved_type:
                    triples_to_insert.append((entity_uri, rdf_type, type_uri(co_type)))

        promoted_entities: dict[str, str] = {}
        # Attribute assertions made for this entity — mirrors the attribute
        # appends to triples_to_insert so per-fact provenance (ADR 0002 §4)
        # can be emitted for them when enabled.
        attr_facts: list[tuple[str, str, str]] = []

        for attr in entity.attributes:
            promo_match = next(
                (p for p in promotions if p.name == attr.name.lower().replace(" ", "_").split("_", 1)[-1]
                 and p.promoted_type is not None),
                None,
            )
            if promo_match and promo_match.promoted_type:
                ptype = promo_match.promoted_type
                if ptype not in promoted_entities:
                    p_uri = f"https://cograph.tech/entities/{ptype}/{_safe_id(entity.id)}-{ptype.lower()}"
                    promoted_entities[ptype] = p_uri
                    triples_to_insert.append((p_uri, rdf_type, type_uri(ptype)))
                    rel_pred = f"https://cograph.tech/onto/has_{ptype.lower()}"
                    triples_to_insert.append((entity_uri, rel_pred, p_uri))

                p_uri = promoted_entities[ptype]
                attr_name = promo_match.name
                p_attrs = existing_attrs.get(ptype, {})
                if attr_name not in p_attrs:
                    sparql = insert_attribute(graph_uri, ptype, attr_name, "", attr.datatype)
                    await self._neptune.update(sparql)
                    result.attributes_added.append(f"{ptype}.{attr_name}")
                    existing_attrs.setdefault(ptype, {})[attr_name] = AttributeSchema(
                        name=attr_name, datatype=attr.datatype,
                    )

                pred_uri = attr_uri(ptype, attr_name)
                validated = validate_triple(
                    p_uri, pred_uri, attr.value, attr.datatype,
                    entity_id=entity.id, attribute_name=attr_name,
                )
                if isinstance(validated, ValidatedTriple):
                    triples_to_insert.append((validated.subject, validated.predicate, validated.object))
                    attr_facts.append((validated.subject, validated.predicate, validated.object))
                    result.triples_inserted += 1
                else:
                    result.rejections.append(validated)

                resolved = resolve_attribute(attr, type_attrs)
                if resolved.action == AttrAction.EXTEND:
                    sparql = insert_attribute(graph_uri, resolved_type, resolved.name, "", resolved.datatype)
                    await self._neptune.update(sparql)
                    result.attributes_added.append(f"{resolved_type}.{resolved.name}")
                    type_attrs[resolved.name] = AttributeSchema(name=resolved.name, datatype=resolved.datatype)

                pred_uri = attr_uri(resolved_type, resolved.name)
                validated = validate_triple(
                    entity_uri, pred_uri, resolved.value, resolved.datatype,
                    entity_id=entity.id, attribute_name=resolved.name,
                )
                if isinstance(validated, ValidatedTriple):
                    triples_to_insert.append((validated.subject, validated.predicate, validated.object))
                    attr_facts.append((validated.subject, validated.predicate, validated.object))
                    result.triples_inserted += 1
                else:
                    result.rejections.append(validated)
                continue

            resolved = resolve_attribute(attr, type_attrs)

            if resolved.action == AttrAction.EXTEND:
                sparql = insert_attribute(graph_uri, resolved_type, resolved.name, "", resolved.datatype)
                await self._neptune.update(sparql)
                result.attributes_added.append(f"{resolved_type}.{resolved.name}")
                type_attrs[resolved.name] = AttributeSchema(name=resolved.name, datatype=resolved.datatype)

            if resolved.datatype not in PRIMITIVE_TYPES and resolved.datatype in existing_types:
                target_uri = f"https://cograph.tech/entities/{resolved.datatype}/{_safe_id(resolved.value)}"
                pred_uri = attr_uri(resolved_type, resolved.name)
                triples_to_insert.append((entity_uri, pred_uri, target_uri))
                attr_facts.append((entity_uri, pred_uri, target_uri))
                result.triples_inserted += 1
            else:
                pred_uri = attr_uri(resolved_type, resolved.name)
                validated = validate_triple(
                    entity_uri, pred_uri, resolved.value, resolved.datatype,
                    entity_id=entity.id, attribute_name=resolved.name,
                )
                if isinstance(validated, ValidatedTriple):
                    triples_to_insert.append((validated.subject, validated.predicate, validated.object))
                    attr_facts.append((validated.subject, validated.predicate, validated.object))
                    result.triples_inserted += 1
                else:
                    result.rejections.append(validated)

        # Per-fact provenance (ADR 0002 §4), gated by COGRAPH_PROVENANCE_ENABLED
        # (default off). Statement-metadata triples target the COMPANION
        # provenance graph — a different graph than the instance-triple
        # collector. With a _collect_provenance collector (the batched fast
        # path, COG-46) they accumulate for ONE batched INSERT by the caller;
        # without one they are inserted here per entity (legacy path).
        # Confidence is 1.0 for directly-ingested facts.
        if self._provenance_enabled and attr_facts:
            instance_graph = getattr(self, "_instance_graph", graph_uri)
            prov_ts = datetime.now(timezone.utc)
            prov_triples: list[tuple[str, str, str]] = []
            for s, p, o in attr_facts:
                prov_triples.extend(build_provenance_triples(
                    s, p, o, source=source, confidence=1.0,
                    timestamp=prov_ts, graph_uri=instance_graph,
                ))
            if _collect_provenance is not None:
                _collect_provenance.extend(prov_triples)
            else:
                for sparql in batched_insert_triples(provenance_graph_uri(instance_graph), prov_triples):
                    await self._neptune.update(sparql)

        # Provenance triples
        now = datetime.now(timezone.utc).isoformat()
        triples_to_insert.append((entity_uri, "https://cograph.tech/onto/ingested_at", now))
        if source:
            triples_to_insert.append((entity_uri, "https://cograph.tech/onto/source", source))
        if batch_id:
            triples_to_insert.append((entity_uri, BATCH_PREDICATE, batch_id))

        # Collect triples for batch insert (or insert immediately if no collector)
        if triples_to_insert:
            if _collect_triples is not None:
                _collect_triples.extend(triples_to_insert)
                result.triples_inserted += len(triples_to_insert)
            else:
                # Legacy path: insert per-entity (used when called without collector)
                instance_graph = getattr(self, "_instance_graph", graph_uri)
                for sparql in batched_insert_triples(instance_graph, triples_to_insert):
                    await self._neptune.update(sparql)
                result.triples_inserted += len(triples_to_insert)


def _safe_id(raw_id: str) -> str:
    """Sanitize an entity ID for use in a URI."""
    import re
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_id.strip())
    return safe[:200] if safe else "unknown"
