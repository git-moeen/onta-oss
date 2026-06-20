"""Unified jobs list (COG-101).

A single endpoint that lists ALL tracked jobs for a tenant — dedupe,
enrichment, and reconciliation — from the configured job store (in-memory or
Postgres). This complements, and does not replace, the enrichment-specific
``/graphs/{tenant}/enrich/jobs`` routes: those remain the place to create,
inspect conflicts for, apply, and cancel enrichment jobs.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from cograph_client.api.deps import get_enrichment_job_store
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.enrichment.models import JobCategory, JobSummary

router = APIRouter(prefix="/graphs/{tenant}/jobs")


@router.get("", response_model=list[JobSummary])
async def list_jobs(
    category: Optional[JobCategory] = Query(
        None, description="Filter to a single job category."
    ),
    tenant: TenantContext = Depends(get_tenant),
    job_store=Depends(get_enrichment_job_store),
):
    """List a tenant's jobs across all categories, newest first.

    Pass ``?category=dedupe|enrichment|reconciliation`` to filter. Each item is
    a ``JobSummary`` carrying the unified fields the Jobs page renders:
    category, trigger, last_run, next_run, cost (+ note), status, and the
    derived ``progress_pct``.
    """
    summaries = await job_store.list_for_tenant(tenant.tenant_id)
    if category is not None:
        summaries = [s for s in summaries if s.category == category]
    return summaries
