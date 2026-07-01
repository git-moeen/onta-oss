"""Web-discovery capability — find a NEW set of records on the web and ingest them.

This is the discovery counterpart to enrichment. Enrichment fills a missing
``(entity, attribute)`` cell on entities that ALREADY exist; discovery CREATES a
whole set of new entities from a natural-language query ("a list of models
offered by OpenRouter"). So it reuses the **ingest** engine, not the enrichment
engine.

The flow deliberately confirms the SHAPE before fetching, so the ontology expands
accurately and the user doesn't have to run a separate enrichment afterward:

1. ``plan`` resolves the target ENTITY type and the ATTRIBUTES to collect. If the
   user only named the entity ("a list of models"), it proposes a sensible
   attribute set and returns a CLARIFY turn ("I'll collect Model with name —
   also want provider, open_source, context_length, pricing?"). The user's reply
   (a clicked option carrying the list, or free text) enters the accumulated
   instruction so the next turn converges.
2. Once attributes are confirmed, ``plan`` fetches a cheap SAMPLE constrained to
   those attributes and runs the SAME multi-type + relationship extractor the
   commit uses against it — so the plan card shows an ESTIMATE of the ontology
   shape the ingest will mint (the distinct entity types, their attributes, and
   the edges between them), not a flat pre-named type. The estimate comes from an
   8-row sample run through a non-deterministic extractor, so the full commit
   (over many more records) may surface additional types/relationships or differ
   in detail. What IS stable across preview and commit is the FETCH hint
   (``hint_columns``) — the column projection sent to the provider. If the
   extractor can't run, the preview degrades to a flat single-type card (the turn
   never 500s).
3. ``execute`` fetches the FULL set (targeting the same attributes) and ingests
   it through :meth:`SchemaResolver.ingest` (``content_type="json"``) — the
   identical extract→resolve→insert path document ingest commits through, which
   infers MULTIPLE types and registers relationships as object-properties — as a
   background job. Returns an ack.

OSS ships with NO web-source provider registered, so the capability degrades
gracefully: ``plan`` returns a plain "not enabled" answer until a downstream
deployment registers a provider (the dev stub, or a paid Exa/Perplexity fan-out).
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import structlog

from cograph_client.agent.registry import AgentContext, PlanStep
from cograph_client.enrichment.models import (
    ConflictPolicy,
    EnrichJob,
    EnrichmentTier,
    JobCategory,
    JobErrorItem,
    JobStatus,
    ProviderLog,
)
from cograph_client.graph.kg_writer import refresh_after_write
from cograph_client.graph.queries import kg_graph_uri, tenant_graph_uri
from cograph_client.resolver.llm_router import PRIMARY_MODEL, openrouter_chat
from cograph_client.web_sources.base import (
    WebSourceProvider,
    get_web_source,
    provider_cost,
)
from cograph_client.web_sources.url_extract import extract_urls

logger = structlog.stdlib.get_logger("cograph.agent.web_ingest")

_bg_tasks: set[asyncio.Task] = set()

# Rows requested for the cheap plan-time sample (preview + datatype inference).
_SAMPLE_ROWS = 8
_PREVIEW_SAMPLE = 5
_PREVIEW_SOURCES = 5
# Conservative default cap so a first (paid) discovery is BOUNDED and cheap to
# inspect. Mirrors the enrich plan's _DEFAULT_PLAN_LIMIT. User-overridable.
_DEFAULT_PLAN_CAP = 200


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


class WebIngestCapability:
    name = "web_ingest"

    def describe(self) -> str:
        return (
            "Discover a NEW set of records from the web and ingest them as a new "
            "dataset/type. Use for 'find a list of X from the web', 'pull all Y', "
            "'add data about Z from the web', 'get me <records> and add them'. Use "
            "when the user wants to CREATE entities that don't exist in the graph "
            "yet — NOT to fill attributes on existing entities (that is enrich)."
        )

    async def plan(
        self,
        ctx: AgentContext,
        instruction: str,
        parsed: dict | None = None,
    ) -> list[PlanStep]:
        # Explicit URLs the user handed us — from structured request context
        # (ctx.urls, read defensively so this works before that field lands) or
        # parsed out of the message. When present we run URL-TARGETED extraction:
        # pull records FROM those pages instead of web-searching for a query.
        urls = (getattr(ctx, "urls", None) or []) or extract_urls(instruction)

        # URL mode needs a URL-capable provider; query mode the default one. When
        # the required provider isn't registered, degrade gracefully the same way.
        provider = get_web_source(for_urls=bool(urls))
        if provider is None:
            if urls:
                return [
                    _answer_step(
                        "I can see the link(s) you shared, but URL extraction isn't "
                        "enabled in this deployment. An admin can configure a "
                        "URL-capable web-source provider to parse pages like these "
                        "into ingested data."
                    )
                ]
            return [
                _answer_step(
                    "Web discovery isn't enabled in this deployment. An admin can "
                    "configure a web-source provider (e.g. Exa or Perplexity) to "
                    "turn a request like this into ingested data."
                )
            ]

        # 1. Resolve the entity type, the attributes to collect, and a CLEAN search
        #    subject — so we search for "OpenRouter TTS models", NOT the user's raw
        #    conversational sentence ("can we ingest open-router's TTS models that
        #    it currently offers"). If the user only named the entity, propose a set
        #    and confirm before spending anything.
        spec = parsed or await _resolve_spec(ctx, instruction)
        type_name = spec.get("entity_type") or "WebRecord"
        query = (spec.get("query") or "").strip() or _clean_query(instruction)
        if not query:
            return []
        key_attr = spec.get("key_attribute") or "name"
        confirmed = _dedupe([key_attr, *spec.get("confirmed_attributes", [])])
        suggested = _dedupe([key_attr, *spec.get("suggested_attributes", [])])

        already_asked = int(ctx.extras.get("prior_clarify_count", 0)) >= 1
        if len(confirmed) <= 1 and not already_asked:
            # Only the key is "confirmed" (i.e. the user just named the entity).
            # Ask which attributes to collect — clickable options carry the list
            # so the next turn converges without new UI.
            return [_clarify_step(type_name, key_attr, suggested)]

        # Commit: use the confirmed set, or fall back to the suggested set if we
        # already asked once (don't loop). These drive entity naming + the
        # preview card — NOT the fetch breadth.
        attributes = confirmed if len(confirmed) > 1 else suggested

        # Decouple the PROVIDER FETCH from the user's minimal named attributes
        # (Cause 1): every provider PROJECTS rows to hint_columns, so passing the
        # confirmed minimal set (e.g. [name, score]) drops the rest of the table
        # (provider, rating, latency, price, votes) before extraction can model
        # the domain. Build a COMPREHENSIVE hint = key ∪ confirmed ∪ suggested
        # (the suggested set is the LLM's richer guess at web-discoverable
        # columns), so the provider returns a rich table the extractor can
        # normalize into Model/Organization/Score/etc. The confirmed set still
        # drives naming + preview above.
        hint_columns = _dedupe([key_attr, *confirmed, *suggested])

        # 2. Cheap sample fetched with the COMPREHENSIVE hint so the preview sees
        #    the same rich table the commit will. In URL mode the provider
        #    extracts the sample FROM the supplied pages.
        try:
            sample = await provider.discover(
                query,
                sample=True,
                max_rows=_SAMPLE_ROWS,
                hint_columns=hint_columns,
                context=_provider_context(ctx),
                urls=urls or None,
            )
        except Exception:  # noqa: BLE001 — a sample failure must never 500 the turn
            logger.warning("web_ingest_sample_failed", exc_info=True)
            return [
                _answer_step(
                    "I couldn't reach the web source to preview that just now. "
                    "Try again in a moment or rephrase the request."
                )
            ]
        if not sample.rows:
            return [_answer_step(_empty_sample_message(query, urls, sample))]

        # Thread the per-record source URL onto the sampled rows so the PREVIEW
        # matches the COMMIT (the same invariant the URL persistence keeps): the
        # discovered-types card + sample rows show the `source_url` citation column
        # the ingest will mint. No-op when the provider supplied no provenance.
        _attach_source_urls(sample.rows, getattr(sample, "provenance", None) or {})

        # 3. Estimate the DISCOVERED ontology shape from the sample — run the same
        #    multi-type + relationship extractor the commit will, so the plan card
        #    shows the LIKELY types/edges the ingest will mint (not a flat
        #    mapping). It's an estimate from the small sample, not a guarantee:
        #    the full commit may surface more types/edges or differ in detail.
        est_total = sample.estimated_total or len(sample.rows)
        cap = _DEFAULT_PLAN_CAP
        cost = _estimate_cost(provider, est_total, cap)
        resolver = _build_resolver(ctx)
        try:
            existing_types, _existing_attrs = await resolver._fetch_ontology(
                tenant_graph_uri(ctx.tenant_id)
            )
            shape = await _preview_shape(
                resolver, sample.rows, set(existing_types.keys())
            )
        except Exception:  # noqa: BLE001 — preview must NEVER 500 the turn
            logger.warning("web_ingest_preview_failed", exc_info=True)
            shape = _flat_shape(type_name, attributes, set())
        discovered_types = shape["discovered_types"]
        relationships = shape["relationships"]

        step = PlanStep(
            capability=self.name,
            action="discover_ingest",
            params={
                "query": query,
                "proposed_type": type_name,
                "attributes": attributes,
                # The COMPREHENSIVE fetch hint (key ∪ confirmed ∪ suggested) —
                # persisted so the full fetch in execute() uses the SAME rich
                # projection the sample did. The FETCH is the part that's stable
                # preview→commit; the discovered TYPES/edges are only an estimate
                # from the sample.
                "hint_columns": hint_columns,
                "max_rows": cap,
                "kg_name": ctx.kg_name,
                "provider": provider.name,
                # Persist the explicit URLs so execute() re-passes them (the same
                # pages are fetched at commit). Empty in plain query-discovery mode.
                "urls": urls,
            },
            rationale=(
                f"Find {query} on the web and add them to this graph as "
                f"{type_name} records."
            ),
            confidence=0.7,
            preview={
                "summary": (
                    f"Estimated ~{len(discovered_types)} type(s) and "
                    f"{len(relationships)} relationship(s) from a sample (the "
                    f"full pull may differ); capped at {cap}, staged for review."
                ),
                "discovered_types": discovered_types,
                "relationships": relationships,
                "sample_rows": sample.rows[:_PREVIEW_SAMPLE],
                "sources": sample.sources[:_PREVIEW_SOURCES],
                "estimated_total": est_total,
                "cost_estimate": cost.get("note", ""),
            },
            cost=cost,
        )
        return [step]

    async def execute(self, ctx: AgentContext, step: PlanStep) -> dict:
        p = step.params
        # URLs persisted at plan time (empty for plain query discovery). Provider
        # selection mirrors plan(): by the persisted name, falling back to the
        # mode-appropriate default (for_urls=bool(urls)) so the same provider runs.
        urls = list(p.get("urls") or [])
        provider = get_web_source(p.get("provider")) or get_web_source(
            for_urls=bool(urls)
        )
        if provider is None:
            raise RuntimeError("web-source provider not available at execute time")

        query = p["query"]
        attributes = p.get("attributes") or []
        # COMPREHENSIVE fetch hint persisted at plan time so the full pull uses the
        # SAME rich projection the sample did — the column projection is the stable
        # part of the preview (the discovered shape was only an estimate). Older
        # persisted steps predate this key — fall back to the named attributes so
        # they still run (graceful degradation).
        hint_columns = p.get("hint_columns") or attributes
        proposed_type = p.get("proposed_type") or "WebRecord"
        cap = int(p.get("max_rows") or _DEFAULT_PLAN_CAP)
        kg_name = p.get("kg_name") or ctx.kg_name
        instance_graph = kg_graph_uri(ctx.tenant_id, kg_name) if kg_name else None
        resolver = _build_resolver(ctx)
        source = f"web:{provider.name}:{query}"
        pctx = _provider_context(ctx)

        # Track the discovery as a real job so the client polls a LIVE status
        # (queued → running → applied/failed) with a result count, the platforms
        # consulted, and the run cost — instead of a synchronous "done" the
        # instant the background task is spawned. The job store is the same
        # unified store enrichment/dedupe use (injected on ctx.extras by the
        # agent route); when it's absent (a bare/test context) we degrade to the
        # previous fire-and-forget behavior so nothing breaks.
        job_store = ctx.extras.get("enrichment_job_store")
        cost_usd, cost_note = _step_cost(step)
        job: Optional[EnrichJob] = None
        if job_store is not None:
            job = EnrichJob(
                id=str(uuid.uuid4()),
                tenant_id=ctx.tenant_id,
                kg_name=kg_name or "",
                type_name=proposed_type,
                attributes=attributes,
                tier=EnrichmentTier.lite,
                status=JobStatus.queued,
                created_at=datetime.now(timezone.utc),
                conflict_policy=ConflictPolicy.stage,
                category=JobCategory.discovery,
                cost=cost_usd,
                cost_note=cost_note,
                # Chat provenance: link the job to the conversation that spawned it.
                thread_id=getattr(ctx, "session_id", None),
            )
            await job_store.create(job)

        # Thread the tracked job id into the provider context so a URL-targeted
        # provider that resumes asynchronously (e.g. a webhook-driven adapter) can
        # correlate its callback back to THIS job. Generic + optional: providers
        # that don't need it ignore the key, and it's absent when discovery runs
        # without a job store (bare/test context), so nothing depends on it.
        if job is not None:
            pctx = {**pctx, "job_id": job.id}

        async def _run() -> None:
            if job is not None and job_store is not None:
                job.status = JobStatus.running
                job.started_at = datetime.now(timezone.utc)
                await job_store.update(job)
            # Per-provider activity log for "which provider we used" + its outcome,
            # surfaced in the run-detail view alongside the platforms list. A
            # single provider drives one discovery run, so this is one entry.
            plog = ProviderLog(provider=provider.name)
            discover_ok = False
            try:
                full = await provider.discover(
                    query,
                    sample=False,
                    max_rows=cap,
                    hint_columns=hint_columns,
                    context=pctx,
                    urls=urls or None,
                )
                discover_ok = True
                rows = full.rows[:cap]
                # Per-record source-URL provenance (ONTA-151): stamp each row with
                # the page it was drawn from (DiscoverResult.provenance) BEFORE
                # serialization, so it rides through the SAME extract → ingest →
                # insert_facts path as the rest of the row's data (no bespoke write
                # path) and lands as a `source_url` citation on the entity. See
                # SOURCE_URL_ATTR on the best-effort (LLM-carried) reliability.
                source_urls = _attach_source_urls(
                    rows, getattr(full, "provenance", None) or {}
                )
                platforms = _platforms(getattr(full, "sources", None), provider)
                # The provider ran: record one attempt with the discovered-record
                # count as its "matches", or a no_match when it came back empty.
                plog.attempts = 1
                plog.matches = len(rows)
                if rows:
                    plog.status = "ok"
                else:
                    plog.no_match = 1
                    plog.status = "no_match"
                # Surface the row count + platforms + provider log as soon as
                # discovery returns, before the (slower) ingest — so a poll
                # mid-run already shows progress and which provider was consulted.
                if job is not None and job_store is not None:
                    job.progress.total = len(rows)
                    job.platforms = platforms
                    job.provider_logs = [plog]
                    await job_store.update(job)
                if not rows:
                    logger.info("web_ingest_no_rows", query=query)
                    await _finish_job(
                        job, job_store, processed=0, entities=0,
                        platforms=platforms,
                    )
                    return
                content = json.dumps(rows, default=str, ensure_ascii=False)
                result = await resolver.ingest(
                    content,
                    ctx.tenant_id,
                    content_type="json",
                    source=source,
                    instance_graph=instance_graph,
                )
                entities = int(getattr(result, "entities_resolved", 0) or 0)
                logger.info(
                    "web_ingest_complete",
                    query=query,
                    rows=len(rows),
                    entities=entities,
                    types=getattr(result, "types_created", None),
                    source_urls=source_urls,
                )
                # Single shared post-write housekeeping path (graph/kg_writer.py) —
                # the SAME refresh ingestion + enrichment run: invalidate the
                # NL-planning ontology cache, re-embed affected types (new types +
                # types that gained an attribute), and recompute Explorer type-stats.
                # Without this the web-discovery ontology expansion stays invisible to
                # query planning + Explorer. Best-effort: a refresh hiccup must NOT
                # present as a failed ingest — the data + ontology already landed.
                affected_types = set(result.types_created)
                for attr_added in result.attributes_added:
                    affected_types.add(attr_added.split(".")[0])
                try:
                    await refresh_after_write(
                        ctx.neptune,
                        tenant_id=ctx.tenant_id,
                        kg_name=kg_name,
                        affected_types=affected_types,
                    )
                except Exception:  # noqa: BLE001 — refresh failure must not fail a landed ingest
                    logger.warning("web_ingest_refresh_failed", exc_info=True)
                await _finish_job(
                    job, job_store, processed=len(rows), entities=entities,
                    platforms=platforms,
                )
            except Exception as exc:  # noqa: BLE001 — background job self-contains errors
                logger.error("web_ingest_failed", query=query, exc_info=True)
                msg = str(exc)
                # Attribute the failure: a crash BEFORE discovery returned is the
                # provider's; a later crash (ingest/refresh) is a job-level error
                # while the provider log stays whatever it recorded as "ok".
                if not discover_ok:
                    plog.attempts = 1
                    plog.errors = 1
                    plog.status = "error"
                    plog.last_error = msg[:300]
                if job is not None:
                    job.provider_logs = [plog]
                    job.error_summary = [
                        JobErrorItem(
                            # A crash before discovery returned is the provider's
                            # ("error"); a later crash (ingest/refresh) is a
                            # job-level failure ("job"), matching the enrichment
                            # executor's fatal-path classification.
                            provider=provider.name if not discover_ok else None,
                            kind="error" if not discover_ok else "job",
                            message=msg[:300],
                        )
                    ]
                await _fail_job(job, job_store, msg)

        _spawn(_run())
        ack = {
            "kind": "ack",
            "capability": self.name,
            "action": step.action,
            "message": (
                f"Searching the web for “{query}” and ingesting the results "
                f"as {proposed_type} ({', '.join(attributes)}) in the background."
            ),
        }
        if job is not None:
            # Hand the job id + initial status back so the client can poll the
            # live status (GET /enrich/jobs/{id} or the unified /jobs feed).
            ack["job_id"] = job.id
            ack["job_status"] = job.status.value
        return ack


# --- entity + attribute resolution ------------------------------------------- #

_SPEC_SYSTEM = """\
You plan a web-discovery ingest: the user wants to pull a NEW set of records from \
the web and add them to a knowledge graph. From the WHOLE conversation, output \
STRICT JSON only (no markdown):
{
  "entity_type": "<PascalCase singular type for the records, e.g. Model, Company, Drug>",
  "key_attribute": "<the natural identifier, usually 'name', snake_case>",
  "query": "<a clean, concise SEARCH SUBJECT — the thing to find on the web, with all conversational framing removed>",
  "confirmed_attributes": ["<attributes the user EXPLICITLY named; [] if they only named the entity>"],
  "suggested_attributes": ["<a COMPREHENSIVE set (6-12) of web-discoverable columns for this entity, snake_case, excluding the key>"]
}
RULES:
- query: the SUBJECT to search for, NOT the user's literal sentence. Strip \
questions, meta-framing and filler. "can we ingest open-router's TTS models that \
it currently offers" -> "OpenRouter text-to-speech (TTS) models". "I'm looking \
for a list of models offered by OpenRouter" -> "models offered by OpenRouter". \
Keep it short and specific; do NOT include words like "ingest", "add", "list of", \
"can we", "I'm looking for".
- entity_type: specific but clean — "a list of models offered by OpenRouter" -> \
"Model" (prefer the domain term the user used; singular).
- key_attribute: the human-readable identifier (name/title), snake_case.
- confirmed_attributes: ONLY what the user actually asked for. "models with their \
names and pricing" -> ["name","pricing"]; "a list of models" -> []. When the user \
replies with a list (e.g. "Use these: name, provider, pricing" or "just the name") \
treat THOSE as confirmed. snake_case; exclude nothing they named.
- suggested_attributes: a COMPREHENSIVE set (aim for 6-12) of the columns this \
entity is typically described by ON THE WEB — every web-discoverable property a \
rich source table (leaderboard, catalog, listing) would carry, snake_case, \
EXCLUDING the key. This is the FETCH hint: the provider projects rows to it, so a \
thin list silently drops the rest of the table before extraction. Be generous and \
include any recurring provider/vendor/organization column and any score/rating/ \
price/ranking column (those become reified entities downstream). For Model: \
["provider","organization","open_source","context_length","input_price",\
"output_price","modality","latency","rating","score","votes","release_date"]."""


async def _resolve_spec(ctx: AgentContext, instruction: str) -> dict:
    """LLM-resolve {entity_type, key_attribute, confirmed/suggested attributes}.

    Degrades to a minimal deterministic spec when there is no key or the LLM
    errors, so the turn never 500s — that minimal spec triggers the clarify path.
    """
    if ctx.openrouter_key:
        try:
            text = await openrouter_chat(
                ctx.openrouter_key,
                _SPEC_SYSTEM,
                instruction,
                model=PRIMARY_MODEL,
                temperature=0,
                max_tokens=400,
                timeout=30,
            )
            parsed = _parse_json_object(text)
            if parsed:
                return _normalize_spec(parsed)
        except Exception:  # noqa: BLE001
            logger.warning("web_ingest_spec_failed", exc_info=True)
    # No-LLM fallback: name the records generically and ask.
    return {
        "entity_type": "WebRecord",
        "key_attribute": "name",
        "confirmed_attributes": [],
        "suggested_attributes": ["name", "description", "url"],
    }


def _normalize_spec(parsed: dict) -> dict:
    et = str(parsed.get("entity_type") or "WebRecord").strip() or "WebRecord"
    key = _slug(parsed.get("key_attribute") or "name") or "name"
    confirmed = [_slug(a) for a in _as_list(parsed.get("confirmed_attributes"))]
    suggested = [_slug(a) for a in _as_list(parsed.get("suggested_attributes"))]
    return {
        "entity_type": _pascal(et),
        "key_attribute": key,
        # Free-text search subject (NOT slugged — it's prose for the provider/card).
        "query": str(parsed.get("query") or "").strip(),
        "confirmed_attributes": [a for a in confirmed if a],
        "suggested_attributes": [a for a in suggested if a],
    }


def _and_join(items: list[str], limit: int = 6) -> str:
    """Human-readable list: 'a, b and c'; '+N more' beyond ``limit``."""
    items = [i for i in items if i]
    if not items:
        return "their details"
    extra = len(items) - limit
    shown = items[:limit]
    head = ", ".join(shown[:-1])
    tail = shown[-1]
    joined = f"{head} and {tail}" if head else tail
    return f"{joined} (+{extra} more)" if extra > 0 else joined


def _clarify_step(type_name: str, key_attr: str, suggested: list[str]) -> PlanStep:
    """Ask which attributes to collect. Both clickable options carry the concrete
    attribute list, so whichever the user clicks lands in the accumulated
    instruction and the next turn converges. The user can also type their own."""
    full = _dedupe([key_attr, *suggested])
    extras = [a for a in full if a != key_attr]
    question = (
        f"I'll collect **{type_name}** records and always include **{key_attr}**. "
        + (
            f"Want these attributes too: {', '.join(extras)}? "
            if extras
            else ""
        )
        + "Pick a set below, or type the attributes you want."
    )
    options = [f"Use these: {', '.join(full)}", f"Just the {key_attr}"]
    return PlanStep(
        capability=WebIngestCapability.name,
        action="clarify",
        params={"question": question, "options": options},
        rationale="Confirm the entity and attributes before fetching from the web.",
        confidence=1.0,
    )


async def _preview_shape(
    resolver, sample_rows: list[dict], existing_types: set[str]
) -> dict:
    """Run the SAME multi-type extractor the commit uses against the sample so the
    plan card ESTIMATES the ontology shape the ingest will mint: the distinct
    entity types (with their attributes + parent chain + is_new flag) and the
    relationships between them, mapped from entity ids to their types.

    This is an estimate from the small sample, not a guarantee — the extractor is
    non-deterministic and the full commit runs over many more records, so it may
    surface additional types/relationships or differ in detail. Mirrors the engine
    that document ingest routes through — instead of forcing one flat pre-named
    type. Caller wraps this in try/except so any extractor failure degrades to a
    flat single-type preview (the turn never 500s)."""
    extraction = await resolver._extract(
        json.dumps(sample_rows, default=str, ensure_ascii=False),
        "json",
        existing_types,
    )
    id_to_type: dict[str, str] = {e.id: e.type_name for e in extraction.entities}

    discovered: list[dict] = []
    seen_types: set[str] = set()
    for e in extraction.entities:
        if e.type_name in seen_types:
            continue
        seen_types.add(e.type_name)
        discovered.append(
            {
                "name": e.type_name,
                "attributes": [a.name for a in e.attributes],
                "parent_chain": list(e.parent_chain),
                "is_new": e.type_name not in existing_types,
            }
        )

    relationships: list[dict] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for r in extraction.relationships:
        src = id_to_type.get(r.source_id)
        tgt = id_to_type.get(r.target_id)
        if not src or not tgt:
            continue
        edge = (src, r.predicate, tgt)
        if edge in seen_edges:
            continue
        seen_edges.add(edge)
        relationships.append({"source": src, "predicate": r.predicate, "target": tgt})

    return {"discovered_types": discovered, "relationships": relationships}


def _flat_shape(
    type_name: str, attributes: list[str], existing_types: set[str]
) -> dict:
    """Degraded preview when the multi-type extractor can't run: a single
    discovered type carrying the confirmed/suggested attributes, no relationships.
    Keeps the plan card confirmable so the turn never 500s."""
    return {
        "discovered_types": [
            {
                "name": type_name,
                "attributes": list(attributes),
                "parent_chain": [],
                "is_new": type_name not in existing_types,
            }
        ],
        "relationships": [],
    }


# --- helpers ----------------------------------------------------------------- #


def _provider_context(ctx: AgentContext) -> dict:
    return {
        "tenant_id": ctx.tenant_id,
        "kg_name": ctx.kg_name,
        "type_name": ctx.type_name,
    }


def _build_resolver(ctx: AgentContext):
    """Build a SchemaResolver from the agent context (same wiring the ingest
    route uses). Constructed per call — cheap, and keeps no cross-request state."""
    import tempfile
    from pathlib import Path

    from cograph_client.resolver.schema_resolver import SchemaResolver
    from cograph_client.resolver.verdict_cache import JsonVerdictCache

    cache = JsonVerdictCache(Path(tempfile.gettempdir()) / "omnix-verdict-cache.json")
    return SchemaResolver(
        neptune=ctx.neptune,
        anthropic_key=ctx.anthropic_key,
        verdict_cache=cache,
    )


# Leading filler we can safely drop so the provider sees a cleaner query. We also
# strip a leading "Use these:" / "just the …" confirmation prefix so the cleaned
# query is the discovery subject, not the attribute reply.
_LEAD_FILLER = re.compile(
    r"^\s*(?:i['’]?m\s+looking\s+for|i\s+want|i\s+need|please\s+|can\s+you\s+|"
    r"could\s+you\s+|find\s+me|find|get\s+me|get|pull|fetch|add|search\s+for)\s+"
    r"(?:a\s+|an\s+|the\s+|me\s+)?",
    re.IGNORECASE,
)


def _empty_sample_message(query: str, urls: list[str], sample) -> str:
    """The user-facing message when a discovery SAMPLE came back with no rows.

    URL mode and query mode fail for DIFFERENT reasons and warrant DIFFERENT
    advice, so we never tell a user who pasted a specific page to "rephrase their
    search" (the old bug — a search-flavoured dead-end shown after a URL scrape):

    * URL mode + provider ERROR (``DiscoverResult.error`` set) → we couldn't READ
      the page(s): surface the reason and suggest retry, not rephrasing.
    * URL mode + no error → we read the page(s) but found no extractable records:
      the page may render its data in a way we can't parse, or hold no list.
    * query mode → an open-web search genuinely found nothing: rephrase/narrow.
    """
    err = getattr(sample, "error", None)
    if urls:
        target = urls[0] if len(urls) == 1 else f"the {len(urls)} pages you shared"
        if err:
            return (
                f"I couldn't read {target}: {err}. The page may be blocking "
                "automated reading or be temporarily unavailable — try again in a "
                "moment, or share a different link."
            )
        return (
            f"I reached {target} but couldn't find a list or table of records to "
            "pull from it. The data may be rendered in a way I can't parse, or the "
            "page may not hold a structured list — try a page whose main content "
            "is the records you want."
        )
    return (
        f"I couldn't find anything on the web for “{query}”. "
        "Try rephrasing or narrowing it."
    )


def _clean_query(instruction: str) -> str:
    """Best-effort tidy of the instruction into a discovery query. Uses the FIRST
    line (the original ask), dropping later attribute-confirmation replies, then
    strips one leading filler phrase."""
    if not instruction:
        return ""
    first = next(
        (ln.strip() for ln in instruction.splitlines() if ln.strip()),
        instruction.strip(),
    )
    q = _LEAD_FILLER.sub("", first, count=1).strip()
    return q or first


def _estimate_cost(
    provider: WebSourceProvider, estimated_total: int, cap: int
) -> dict:
    """Plan-time cost estimate, using the SAME contract keys the plan card reads
    (``estimated_usd`` / ``paid_calls`` / ``note``)."""
    is_paid, cost_per_call = provider_cost(provider)
    rows = min(estimated_total or 0, cap) if cap else (estimated_total or 0)
    if not is_paid:
        return {
            "paid_calls": 0,
            "estimated_usd": 0.0,
            "note": "No paid calls (the configured web source is free).",
        }
    estimated_usd = round(cost_per_call, 4)
    return {
        "paid_calls": 1,
        "paid_calls_estimated": True,
        "estimated_usd": estimated_usd,
        "per_call_cost_usd": round(cost_per_call, 4),
        "note": (
            f"Paid web discovery via '{provider.name}': ≈ ${estimated_usd:.2f} "
            f"to fetch up to {rows} record(s) (estimate; provider may fan out "
            f"across sub-queries)."
        ),
    }


# --- per-record source-URL provenance (ONTA-151) ----------------------------- #

# Attribute minted on each discovered entity citing the exact page it was drawn
# from — the discovery counterpart to enrichment's `<attr>_source_url` citations
# and the user-facing source the Explorer renders (any URL-valued attribute is a
# clickable link in the records table). The run-level provenance the resolver
# already writes (`onto/source` = web:<provider>:<query>, `onto/ingested_at`, the
# batch id) is unchanged; this adds the missing PER-RECORD citation so "this exact
# data point came from this exact page" is answerable, not just "this came from a
# discovery for query X".
#
# Threaded as an ordinary row field so it flows through the SAME ingest →
# insert_facts write path as every other attribute (write-path convergence) — no
# bespoke writer, no separate provenance graph. NOTE on the reliability contract:
# unlike enrichment, which writes `<attr>_source_url` DETERMINISTICALLY onto the
# entity URI (no LLM), discovery carries `source_url` as a row field THROUGH the
# multi-type LLM extractor. So it is best-effort: exactly as reliable as the row's
# OTHER discovered attributes (name, pricing, …) — the same extractor decides them
# all — but not a hard guarantee, and on a multi-type row the extractor chooses
# which entity it lands on. `uri` is a declared attribute datatype, so a field
# named `source_url` is overwhelmingly kept as a literal at temperature 0. If
# GUARANTEED per-record citations are ever required, stamp this deterministically
# post-extraction keyed by entity id (a follow-up; would touch the shared resolver).
SOURCE_URL_ATTR = "source_url"


def _row_source_url(
    row: dict, index: int, provenance: dict[str, str]
) -> Optional[str]:
    """Resolve the source URL a discovered ``row`` was drawn from, using the
    provider's per-row ``provenance`` map (:attr:`DiscoverResult.provenance`).

    Providers key the map by the row's natural name, falling back to the row's
    positional index as a string — the convention every bundled adapter and the
    stub use (``{r.get("name", str(i)): url}``). Mirror that exact key here (name
    when the row carries one, else the index), then fall back to the positional
    index so an index-keyed provider also resolves. Returns ``None`` when no URL
    is known for the row (e.g. a free/stub provider that supplied no provenance)."""
    if not provenance or not isinstance(row, dict):
        return None
    key = row.get("name", str(index))
    url = provenance.get(str(key))
    if url:
        return url
    return provenance.get(str(index))


def _attach_source_urls(rows: list[dict], provenance: dict[str, str]) -> int:
    """Stamp each discovered row (in place) with its per-record ``source_url`` so
    the entity it mints carries a traceable citation to its origin page. Returns
    the number of rows stamped.

    A no-op when the provider supplied no provenance (free/stub providers may omit
    it). Never clobbers a ``source_url`` the provider already set on the row, and
    leaves a row with no resolvable URL untouched rather than stamping a blank — so
    the column appears only where there is a real citation to show."""
    if not provenance:
        return 0
    stamped = 0
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or row.get(SOURCE_URL_ATTR):
            continue
        url = _row_source_url(row, i, provenance)
        if url:
            row[SOURCE_URL_ATTR] = url
            stamped += 1
    return stamped


# --- job tracking ------------------------------------------------------------ #


def _step_cost(step: PlanStep) -> tuple[Optional[float], Optional[str]]:
    """Pull the plan card's cost estimate (estimated_usd + note) off the step so
    it can be stamped on the job — that's the "how much did it cost" detail the
    job-status view shows. Returns (usd, note); either may be None."""
    cost = step.cost or {}
    usd = cost.get("estimated_usd")
    note = cost.get("note")
    usd_f = (
        float(usd)
        if isinstance(usd, (int, float)) and not isinstance(usd, bool)
        else None
    )
    return usd_f, (str(note) if note else None)


def _host(url: str) -> str:
    """Hostname of a URL with a leading ``www.`` dropped; a bare token (already a
    host/provider name) is returned trimmed/lower-cased. '' if unparseable."""
    try:
        netloc = urlparse(url).netloc
    except Exception:  # noqa: BLE001 — never let URL parsing break a run
        netloc = ""
    host = (netloc or url or "").strip().lower()
    return host[4:] if host.startswith("www.") else host


def _platforms(sources, provider) -> list[str]:
    """Distinct platforms consulted during a discovery run — the host of each
    source URL (de-duplicated, order-preserved, capped), falling back to the
    provider name when no URLs were returned. Surfaced in the job-details view
    as "what platforms were used"."""
    out: list[str] = []
    seen: set[str] = set()
    for s in sources or []:
        host = _host(str(s))
        if host and host not in seen:
            seen.add(host)
            out.append(host)
        if len(out) >= 8:
            break
    if not out:
        name = (getattr(provider, "name", "") or "").strip()
        if name:
            out.append(name)
    return out


async def _finish_job(
    job: Optional[EnrichJob],
    job_store,
    *,
    processed: int,
    entities: int,
    platforms: list[str],
) -> None:
    """Mark a discovery job applied with its result count + final progress."""
    if job is None or job_store is None:
        return
    now = datetime.now(timezone.utc)
    job.progress.processed = processed
    job.progress.filled = entities
    job.result_count = entities
    if platforms:
        job.platforms = platforms
    job.status = JobStatus.applied
    job.completed_at = now
    job.last_run = now
    await job_store.update(job)


async def _fail_job(job: Optional[EnrichJob], job_store, error: str) -> None:
    """Mark a discovery job failed, carrying a (truncated) error for the UI."""
    if job is None or job_store is None:
        return
    now = datetime.now(timezone.utc)
    job.status = JobStatus.failed
    job.error = (error or "discovery failed")[:500]
    job.completed_at = now
    job.last_run = now
    await job_store.update(job)


def _answer_step(text: str) -> PlanStep:
    """A single no-write 'answer' step (planner short-circuits it to kind:answer)."""
    return PlanStep(
        capability=WebIngestCapability.name,
        action="answer",
        params={"answer_payload": {"answer": text, "narrative": text}},
        rationale=text,
        confidence=1.0,
    )


def _parse_json_object(text: str) -> dict | None:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = "\n".join(
            l for l in stripped.split("\n") if not l.strip().startswith("```")
        )
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        stripped = stripped[start : end + 1]
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _as_list(v) -> list[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(x) for x in v]
    return []


def _slug(v) -> str:
    """snake_case a single attribute name; drop surrounding junk."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(v or "").strip().lower()).strip("_")
    return s


def _pascal(v: str) -> str:
    parts = re.split(r"[^0-9a-zA-Z]+", str(v or "").strip())
    return "".join(p[:1].upper() + p[1:] for p in parts if p) or "WebRecord"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        s = (x or "").strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out
