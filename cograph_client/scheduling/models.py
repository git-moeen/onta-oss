"""Pydantic models for recurring schedules (COG-135).

A ``Schedule`` is the *data* description of a recurring action — "re-run the
enrich/dedupe/suggest action for this KG every N seconds (or on this cron)".
This module (and the store + routes alongside it) is the SCHEDULING DATA SEAM
only: storage + CRUD + the next-run computation. It does NOT contain the
firing/scheduler loop — a separate task owns waking up and creating jobs from
due schedules.

``category`` reuses :class:`~cograph_client.enrichment.models.JobCategory` so a
schedule maps 1:1 onto the kind of job it will later fire, and the unified Jobs
feed stays consistent.

Vendor-neutral by construction: no cloud-provider identifiers, ARNs, hostnames,
or secrets live here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

from cograph_client.enrichment.models import JobCategory

# The action a schedule fires — mirrors the Ask-AI action endpoints (COG-99):
# find-merge-duplicates (dedupe), enrich (enrichment), suggest-relationships
# (reconciliation). A schedule's ``category`` should agree with its action.
ScheduleAction = Literal["find-merge-duplicates", "enrich", "suggest-relationships"]


class Schedule(BaseModel):
    """A recurring action definition for a tenant's KG.

    Exactly one of ``cron`` / ``interval_seconds`` must be set — a schedule
    recurs either on a cron expression or a fixed interval, never both and
    never neither (validated below). ``params`` carries the action-specific
    payload (e.g. ``type_name``/``attributes``/``tier``/``conflict_policy`` for
    an enrich schedule), passed through to the job the scheduler later creates.

    ``next_run`` is the timestamp the firing loop (separate task) will compare
    against ``now`` to decide whether this schedule is due; ``last_run`` records
    the most recent firing. Both are managed by the scheduler, not the data
    seam — they default to ``None`` on creation (the route computes the initial
    ``next_run``).
    """

    id: str
    tenant_id: str
    kg_name: str
    category: JobCategory
    action: ScheduleAction
    # Action-specific payload threaded through to the fired job (type_name,
    # attributes, tier, conflict_policy, scope, ...). Kept generic so adding a
    # field to an action body never requires a schema change here.
    params: dict = Field(default_factory=dict)
    # Recurrence: EXACTLY ONE of these is set (see _check_recurrence).
    cron: Optional[str] = None
    interval_seconds: Optional[int] = None
    enabled: bool = True
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    created_at: datetime

    @model_validator(mode="after")
    def _check_recurrence(self) -> "Schedule":
        has_cron = self.cron is not None and self.cron.strip() != ""
        has_interval = self.interval_seconds is not None
        if has_cron and has_interval:
            raise ValueError(
                "set exactly one of cron / interval_seconds, not both"
            )
        if not has_cron and not has_interval:
            raise ValueError(
                "set exactly one of cron / interval_seconds (one is required)"
            )
        if has_interval and self.interval_seconds is not None and self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be a positive integer")
        return self
