"""HTTP routes for the auto-enrichment feature."""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from cograph_client.api.deps import (
    get_enrichment_job_store,
    get_executor,
    get_neptune_client,
)
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.enrichment.executor import EnrichmentExecutor
from cograph_client.enrichment.job_store import InMemoryJobStore
from cograph_client.enrichment.models import (
    ConflictReview,
    EnrichJob,
    EnrichRequest,
    EnrichmentTier,
    JobStatus,
    JobSummary,
)
from cograph_client.enrichment.strategy import resolve_type_name, unknown_type_message
from cograph_client.graph.client import NeptuneClient
from cograph_client.enrichment.tier_router import (
    DEFAULT_CONFIDENCE_MIN,
    WEB_CONFIDENCE_MIN,
    chain_has_paid,
    resolve_auto_tier,
)

router = APIRouter(prefix="/graphs/{tenant}/enrich")


def _openrouter_key() -> str:
    """The OpenRouter key the auto-tier resolver needs for its classify call.

    Same source the rest of OSS uses (``normalization.inference._openrouter_key``):
    the app ``settings`` (env ``OMNIX_OPENROUTER_API_KEY``) with a plain
    ``OPENROUTER_API_KEY`` env fallback. When empty, ``resolve_auto_tier`` falls
    back to its deterministic heuristic — it never requires a key.
    """
    from cograph_client.config import settings

    return settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")


def _effective_confidence_min(tier: EnrichmentTier, requested: float) -> float:
    """Apply the web confidence floor (COG-121) for a PAID/web tier.

    When the EFFECTIVE tier's chain has a paid adapter AND the caller left
    ``confidence_min`` at the default sentinel (0.85 — i.e. did not ask for a
    specific value), lower it to ``WEB_CONFIDENCE_MIN`` (0.4) so the low-prior web
    verdicts are written instead of all being filtered out → 0 fills. A
    user-supplied non-default confidence is respected unchanged. Free tiers are
    untouched. Generic: "paid" comes from adapter-declared metadata, never an
    adapter name (COG-123).
    """
    user_set = abs(requested - DEFAULT_CONFIDENCE_MIN) > 1e-9
    if not user_set and chain_has_paid(tier):
        return WEB_CONFIDENCE_MIN
    return requested


class CreateJobResponse(BaseModel):
    """Uniform create-job response across all branches (COG-124).

    Every branch — explicit tier, auto→lite, auto→core, and needs-clarification —
    returns the SAME keys so clients have one stable contract. ``job_id`` /
    ``resolved_tier`` are ``None`` only on the needs-clarification branch (no job
    is created); ``candidates`` is populated only there.
    """

    job_id: Optional[str] = None
    status: str
    matched_entities: Optional[int] = None
    resolved_tier: Optional[str] = None
    routing_note: Optional[str] = None
    needs_clarification: bool = False
    candidates: Optional[list[str]] = None


CONFLICT_RESULT_TRUNCATE = 100


# Background fire-and-forget tasks: CPython only holds a *weak* reference to a
# bare ``asyncio.create_task(...)`` result, so it can be garbage-collected at the
# first ``await`` after the request returns — silently stranding the enrichment
# job right after it selects entities (COG-112). Keep a strong reference in a
# module-level set and drop it on completion (mirrors actions.py / explore.py's
# schedule_recompute).
_bg_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    """Schedule a background coroutine, keeping a strong ref until it finishes."""
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


class ApplyRequest(BaseModel):
    decisions: list[ConflictReview]


@router.post("/jobs", status_code=202, response_model=CreateJobResponse)
async def create_job(
    body: EnrichRequest,
    tenant: TenantContext = Depends(get_tenant),
    executor: EnrichmentExecutor = Depends(get_executor),
    job_store: InMemoryJobStore = Depends(get_enrichment_job_store),
    neptune: NeptuneClient = Depends(get_neptune_client),
) -> CreateJobResponse:
    routing_note: Optional[str] = None

    # COG-124: resolve the smart ``auto`` tier BEFORE creating a job. The shared
    # router (re)uses the agent's web-fact judgment to decide free Wikidata
    # (``lite``) vs paid web search (``core``), leaning paid when Wikidata is
    # weak; it only asks for clarification when genuinely ambiguous.
    if body.tier == EnrichmentTier.auto:
        decision = await resolve_auto_tier(
            body.attributes, body.type_name, _openrouter_key()
        )
        if decision.needs_clarification:
            # Genuinely ambiguous — do NOT create a job. The client picks a tier
            # and re-submits with an explicit ``lite``/``core``. Still 202: the
            # request was understood, it just needs a follow-up choice.
            return CreateJobResponse(
                job_id=None,
                status="needs_clarification",
                matched_entities=None,
                resolved_tier=None,
                routing_note=decision.routing_note,
                needs_clarification=True,
                candidates=decision.candidates,
            )
        effective_tier = EnrichmentTier(decision.resolved_tier)
        routing_note = decision.routing_note
    else:
        effective_tier = body.tier

    # WEB CONFIDENCE FLOOR (COG-121): for a paid/web EFFECTIVE tier with an UNSET
    # confidence, lower confidence_min to the web floor so the low-prior web
    # verdicts actually land instead of all being filtered → 0 fills. Applies to
    # auto-resolved ``core`` AND an explicit ``core``/``pro`` (the direct API path
    # previously skipped this, so a direct core enrich wrote nothing). A
    # user-supplied confidence is respected.
    effective_confidence = _effective_confidence_min(
        effective_tier, body.confidence_min
    )

    # Resolve the target type to the tenant's canonical declared name up front:
    # a miscased type (e.g. `organization` vs the declared `Organization`)
    # auto-corrects, and a type that genuinely doesn't exist is rejected with an
    # actionable 422 — instead of creating a job that finishes "Completed" having
    # silently enriched nothing (the entity SELECT keys on <types/Name>). One
    # bounded ontology read, never an instance scan (COG-112 safe); fail-open when
    # the read fails or no types are declared (known == []). The executor applies
    # the SAME guard as a safety net for jobs created outside this route
    # (schedules, actions).
    resolved_type, known_types = await resolve_type_name(
        neptune, tenant.tenant_id, body.type_name
    )
    if known_types and resolved_type is None:
        raise HTTPException(
            status_code=422,
            detail=unknown_type_message(body.type_name, known_types),
        )
    type_name = resolved_type or body.type_name
    if resolved_type and resolved_type != body.type_name:
        correction = f"Interpreted type '{body.type_name}' as '{resolved_type}'."
        routing_note = f"{routing_note} {correction}".strip() if routing_note else correction

    # NON-BLOCKING create (COG-112): we deliberately do NOT count_entities() in
    # the request path. A scoped COUNT over a large type (e.g. ~13.5k Mentors)
    # can be slow and was timing out the create (~55s → 504). The executor's
    # background SELECT already resolves the scope, selects only the matched
    # subset, and sets `progress.total` to that count — so the matched count
    # surfaces via the job's progress, not at create time. This decouples create
    # latency from type size so it can NEVER time out again.
    job = EnrichJob(
        id=str(uuid.uuid4()),
        tenant_id=tenant.tenant_id,
        kg_name=body.kg_name,
        type_name=type_name,
        attributes=body.attributes,
        tier=effective_tier,
        status=JobStatus.queued,
        created_at=datetime.now(timezone.utc),
        conflict_policy=body.conflict_policy,
        confidence_min=effective_confidence,
        limit=body.limit,
        scope=body.scope,
        entity_uris=body.entity_uris,
        instructions=body.instructions,
        sources=body.sources,
        # Explicit pages to extract attribute values FROM (the URL-targeted
        # enrichment rail). Defaults to [] so a normal enrich is unchanged; a
        # URL-aware adapter reads them via the executor's lookup context.
        source_urls=body.target_urls or [],
        # Chat provenance: link the job to the conversation it was kicked off
        # from (None for direct-API / CLI callers).
        thread_id=body.thread_id,
    )
    await job_store.create(job)

    _spawn(executor.run(job, tenant.tenant_id))

    return CreateJobResponse(
        job_id=job.id,
        status=job.status.value,
        # Matched count is resolved by the background executor (progress.total),
        # not at create time — so create never blocks on a scoped COUNT. The web
        # dialog reads job_id + status; matched_entities is optional in the UI.
        matched_entities=None,
        resolved_tier=effective_tier.value,
        routing_note=routing_note,
        needs_clarification=False,
        candidates=None,
    )


@router.get("/jobs", response_model=list[JobSummary])
async def list_jobs(
    tenant: TenantContext = Depends(get_tenant),
    job_store: InMemoryJobStore = Depends(get_enrichment_job_store),
):
    return await job_store.list_for_tenant(tenant.tenant_id)


@router.get("/jobs/{job_id}", response_model=EnrichJob)
async def get_job(
    job_id: str,
    tenant: TenantContext = Depends(get_tenant),
    job_store: InMemoryJobStore = Depends(get_enrichment_job_store),
):
    job = await job_store.get(job_id)
    if not job or job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="job not found")
    if job.results and len(job.results) > CONFLICT_RESULT_TRUNCATE:
        job.results = job.results[:CONFLICT_RESULT_TRUNCATE]
    return job


@router.get("/jobs/{job_id}/conflicts", response_model=list[ConflictReview])
async def list_conflicts(
    job_id: str,
    tenant: TenantContext = Depends(get_tenant),
    job_store: InMemoryJobStore = Depends(get_enrichment_job_store),
):
    job = await job_store.get(job_id)
    if not job or job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="job not found")
    out: list[ConflictReview] = []
    for r in job.results:
        if r.action != "conflict" or r.verdict is None:
            continue
        out.append(
            ConflictReview(
                entity_uri=r.entity_uri,
                attribute=r.attribute,
                existing_value=r.existing_value or "",
                proposed=r.verdict,
            )
        )
    return out


@router.post("/jobs/{job_id}/apply")
async def apply_job(
    job_id: str,
    body: ApplyRequest,
    tenant: TenantContext = Depends(get_tenant),
    executor: EnrichmentExecutor = Depends(get_executor),
    job_store: InMemoryJobStore = Depends(get_enrichment_job_store),
):
    job = await job_store.get(job_id)
    if not job or job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="job not found")
    applied = await executor.apply_decisions(job_id, body.decisions)
    return {"applied": applied}


@router.delete("/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    tenant: TenantContext = Depends(get_tenant),
    job_store: InMemoryJobStore = Depends(get_enrichment_job_store),
):
    job = await job_store.get(job_id)
    if not job or job.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="job not found")
    job.status = JobStatus.cancelled
    await job_store.update(job)
    return {"status": "cancelled", "job_id": job_id}
