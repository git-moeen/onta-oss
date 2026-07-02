"""Schedule CRUD endpoints (COG-135).

Recurring-action schedules for a tenant: create / list / get / update / delete.
A schedule describes a recurring action (find-merge-duplicates, enrich,
suggest-relationships) over a KG and recurs on a cron expression OR a fixed
interval. Create/update only accept the USER_SCHEDULABLE_ACTIONS subset of the
schedule vocabulary — the semantic maintenance actions are system-managed rows
the reconciler creates internally (see scheduling/models.py for the WHY). This is the DATA SEAM only — these routes persist schedules and
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
from pydantic import BaseModel, ValidationError, field_validator

from cograph_client.api.deps import get_schedule_store
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.enrichment.models import JobCategory
from cograph_client.scheduling.models import (
    USER_SCHEDULABLE_ACTIONS,
    Schedule,
    ScheduleAction,
)
from cograph_client.scheduling.next_run import compute_next_run

router = APIRouter(prefix="/graphs/{tenant}/schedules")


# --- Request bodies -----------------------------------------------------------


def _require_user_schedulable(action: Optional[str]) -> Optional[str]:
    """Reject system-managed actions on the tenant-facing request models.

    ``ScheduleAction`` is the FULL vocabulary the store/runner understand, but
    the semantic maintenance actions are system-managed rows created internally
    by ``semantic/reconciler.py`` — see ``USER_SCHEDULABLE_ACTIONS`` in
    ``scheduling/models.py`` for why tenants must not mint them (global
    cross-tenant sweep cadence; uuid rows that KG-delete cleanup never removes).
    Raising ``ValueError`` here surfaces as a standard 422 on the request body.
    """
    if action is not None and action not in USER_SCHEDULABLE_ACTIONS:
        raise ValueError(
            f"action '{action}' is system-managed (its schedule rows are "
            "created and tuned by the platform, not via this API); "
            f"user-schedulable actions: {sorted(USER_SCHEDULABLE_ACTIONS)}"
        )
    return action


class ScheduleCreateRequest(BaseModel):
    kg_name: str
    category: JobCategory
    action: ScheduleAction
    params: dict = {}
    cron: Optional[str] = None
    interval_seconds: Optional[int] = None
    enabled: bool = True

    @field_validator("action")
    @classmethod
    def _action_user_schedulable(cls, v: str) -> str:
        return _require_user_schedulable(v)  # type: ignore[return-value]


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

    @field_validator("action")
    @classmethod
    def _action_user_schedulable(cls, v: Optional[str]) -> Optional[str]:
        return _require_user_schedulable(v)


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

    System-managed rows (action outside ``USER_SCHEDULABLE_ACTIONS`` — the
    per-KG ``semantic-reconcile:{tenant}:{kg}`` rows the reconciler auto-creates
    carry the caller's tenant_id, so they ARE reachable here) are tenant-visible
    but not tenant-tunable: every PATCH to such a row is rejected with 403.
    Rejecting wholesale (rather than field-by-field) keeps the policy simple and
    airtight — cadence/action/params are platform-owned (the reconciler re-tunes
    them from env knobs), and even an ``enabled`` flip would silently switch off
    index maintenance while the row still looks healthy. The 403 (vs 404) is
    deliberate: the row is visible via GET/list, so pretending it doesn't exist
    would be misleading.
    """
    existing = await store.get(schedule_id)
    if existing is None or existing.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="schedule not found")
    if existing.action not in USER_SCHEDULABLE_ACTIONS:
        raise HTTPException(
            status_code=403,
            detail=(
                f"schedule '{schedule_id}' is system-managed (action "
                f"'{existing.action}') and cannot be modified via this API"
            ),
        )

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
    """Delete a schedule (scoped to the authorized tenant).

    Deleting a system-managed semantic row (action outside
    ``USER_SCHEDULABLE_ACTIONS``) is allowed but is NOT a durable opt-out: the
    reconciler's ensure-* hooks recreate the row on the next KG write or
    reindex request. Disabling semantic maintenance is done via its feature
    flag, not by deleting rows.
    """
    existing = await store.get(schedule_id)
    if existing is None or existing.tenant_id != tenant.tenant_id:
        raise HTTPException(status_code=404, detail="schedule not found")
    await store.delete(schedule_id)
    return None
