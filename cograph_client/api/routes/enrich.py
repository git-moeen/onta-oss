"""HTTP routes for the auto-enrichment feature."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from cograph_client.api.deps import get_enrichment_job_store, get_executor
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.enrichment.executor import EnrichmentExecutor
from cograph_client.enrichment.job_store import InMemoryJobStore
from cograph_client.enrichment.models import (
    ConflictReview,
    EnrichJob,
    EnrichRequest,
    JobStatus,
    JobSummary,
)

router = APIRouter(prefix="/graphs/{tenant}/enrich")


CONFLICT_RESULT_TRUNCATE = 100


class ApplyRequest(BaseModel):
    decisions: list[ConflictReview]


@router.post("/jobs", status_code=202)
async def create_job(
    body: EnrichRequest,
    tenant: TenantContext = Depends(get_tenant),
    executor: EnrichmentExecutor = Depends(get_executor),
    job_store: InMemoryJobStore = Depends(get_enrichment_job_store),
):
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
        type_name=body.type_name,
        attributes=body.attributes,
        tier=body.tier,
        status=JobStatus.queued,
        created_at=datetime.now(timezone.utc),
        conflict_policy=body.conflict_policy,
        confidence_min=body.confidence_min,
        limit=body.limit,
        scope=body.scope,
        entity_uris=body.entity_uris,
    )
    await job_store.create(job)

    asyncio.create_task(executor.run(job, tenant.tenant_id))

    return {
        "job_id": job.id,
        "status": job.status.value,
        # Matched count is resolved by the background executor (progress.total),
        # not at create time — so create never blocks on a scoped COUNT. The web
        # dialog reads job_id + status; matched_entities is optional in the UI.
        "matched_entities": None,
    }


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
