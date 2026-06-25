"""Schedule CRUD endpoints (COG-135).

Recurring-action schedules for a tenant: create / list / get / update / delete.
A schedule describes a recurring action (find-merge-duplicates, enrich,
suggest-relationships) over a KG and recurs on a cron expression OR a fixed
interval. This is the DATA SEAM only — these routes persist schedules and
compute their initial/updated ``next_run``; the firing loop that turns due
schedules into jobs is a separate task and lives elsewhere.

Boundary note: this is OSS storage + CRUD + SDK only. No cloud identifiers,
ARNs, hostnames, or secrets. The same ``get_tenant`` auth dependency the other
tenant-scoped routes use authorizes the tenant in the path.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ValidationError

from cograph_client.api.deps import get_schedule_store
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.enrichment.models import JobCategory
from cograph_client.scheduling.models import Schedule, ScheduleAction
from cograph_client.scheduling.next_run import compute_next_run

router = APIRouter(prefix="/graphs/{tenant}/schedules")


# --- Request bodies -----------------------------------------------------------


class ScheduleCreateRequest(BaseModel):
    kg_name: str
    category: JobCategory
    action: ScheduleAction
    params: dict = {}
    cron: Optional[str] = None
    interval_seconds: Optional[int] = None
    enabled: bool = True


class ScheduleUpdateRequest(BaseModel):
    """All fields optional — a PATCH updates only what is provided.

    Common cases: ``{"enabled": false}`` to pause, ``{"enabled": true}`` to
    resume, or change the recurrence / params. Changing the recurrence
    recomputes ``next_run``.
    """

    kg_name: Optional[str] = None
    category: Optional[JobCategory] = None
    action: Optional[ScheduleAction] = None
    params: Optional[dict] = None
    cron: Optional[str] = None
    interval_seconds: Optional[int] = None
    enabled: Optional[bool] = None


# --- Helpers ------------------------------------------------------------------


def _build_schedule(**kwargs) -> Schedule:
    """Construct a Schedule, mapping its validation error to a 422.

    The exactly-one-of cron/interval_seconds rule lives in the model validator,
    so a bad combination raised here is a client error, not a 500.
    """
    try:
        return Schedule(**kwargs)
    except ValidationError as exc:
        # Surface only the JSON-safe message/location for each error. The raw
        # errors() payload can embed the offending INPUT value (e.g. a datetime),
        # which Starlette's JSONResponse can't serialize — so we project to
        # primitives here.
        detail = [
            {"loc": list(e.get("loc", ())), "msg": e.get("msg", "")}
            for e in exc.errors()
        ]
        raise HTTPException(status_code=422, detail=detail)


# --- Routes -------------------------------------------------------------------


@router.post("", response_model=Schedule, status_code=201)
async def create_schedule(
    body: ScheduleCreateRequest,
    tenant: TenantContext = Depends(get_tenant),
    store=Depends(get_schedule_store),
):
    """Create a recurring schedule and compute its initial ``next_run``.

    Exactly one of ``cron`` / ``interval_seconds`` must be set (422 otherwise).
    ``next_run`` is seeded from ``created_at`` so the firing loop can pick it up
    on the next sweep.
    """
    now = datetime.now(timezone.utc)
    schedule = _build_schedule(
        id=str(uuid.uuid4()),
        tenant_id=tenant.tenant_id,
        kg_name=body.kg_name,
        category=body.category,
        action=body.action,
        params=body.params,
        cron=body.cron,
        interval_seconds=body.interval_seconds,
        enabled=body.enabled,
        created_at=now,
    )
    # Seed the initial next_run from creation time. compute_next_run raises a
    # clear NotImplementedError for cron without croniter → surface as 501.
    try:
        schedule.next_run = compute_next_run(schedule, now)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    await store.create(schedule)
    return schedule


@router.get("", response_model=list[Schedule])
async def list_schedules(
    tenant: TenantContext = Depends(get_tenant),
    store=Depends(get_schedule_store),
):
    """List a tenant's schedules, oldest first (creation order)."""
    return await store.list_for_tenant(tenant.tenant_id)


@router.get("/{schedule_id}", response_model=Schedule)
async def get_schedule(
    schedule_id: str,
    tenant: TenantContext = Depends(get_tenant),
    store=Depends(get_schedule_store),
):
    """Fetch a single schedule by id (scoped to the authorized tenant)."""
    schedule = await store.get(schedule_id)
    if schedule is None or schedule.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="schedule not found")
    return schedule


@router.patch("/{schedule_id}", response_model=Schedule)
async def update_schedule(
    schedule_id: str,
    body: ScheduleUpdateRequest,
    tenant: TenantContext = Depends(get_tenant),
    store=Depends(get_schedule_store),
):
    """Enable/disable or update a schedule.

    Only provided fields change. If the recurrence (``cron``/
    ``interval_seconds``) changes, ``next_run`` is recomputed from now.
    """
    existing = await store.get(schedule_id)
    if existing is None or existing.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="schedule not found")

    updates = body.model_dump(exclude_unset=True)
    recurrence_changed = "cron" in updates or "interval_seconds" in updates
    merged = existing.model_dump()
    merged.update(updates)
    # Changing one recurrence field clears the other so the exactly-one-of
    # invariant holds (e.g. switching an interval schedule to cron).
    if recurrence_changed:
        if "cron" in updates and updates.get("cron"):
            merged["interval_seconds"] = None
        elif "interval_seconds" in updates and updates.get("interval_seconds") is not None:
            merged["cron"] = None
    updated = _build_schedule(**merged)

    if recurrence_changed:
        try:
            updated.next_run = compute_next_run(
                updated, datetime.now(timezone.utc)
            )
        except NotImplementedError as exc:
            raise HTTPException(status_code=501, detail=str(exc))
    await store.update(updated)
    return updated


@router.delete("/{schedule_id}", status_code=204)
async def delete_schedule(
    schedule_id: str,
    tenant: TenantContext = Depends(get_tenant),
    store=Depends(get_schedule_store),
):
    """Delete a schedule (scoped to the authorized tenant)."""
    existing = await store.get(schedule_id)
    if existing is None or existing.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="schedule not found")
    await store.delete(schedule_id)
    return None
