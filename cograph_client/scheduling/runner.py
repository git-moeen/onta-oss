"""In-process schedule firing loop (COG-136).

This is the FIRING/SCHEDULER side of the scheduling feature — the counterpart to
the data seam (COG-135: models + store + CRUD). A :class:`ScheduleRunner` wakes
up periodically, finds due schedules, advances each one's ``next_run`` past any
missed ticks, and dispatches the corresponding action job (reusing the exact
``/actions/*`` workers via :func:`dispatch_scheduled_action`).

Two claim strategies, picked by the configured store:

- **Postgres** (durable, possibly multi-replica): each tick claims a batch of
  due rows in ONE transaction with ``SELECT ... FOR UPDATE SKIP LOCKED``, so two
  replicas never fire the same schedule. Each claimed row's ``next_run`` /
  ``last_run`` are advanced and the row UPDATEd inside that transaction; the
  action is dispatched only AFTER the commit (so a slow/failing action can't
  hold the row lock or roll back the advance).
- **In-memory** (single process, zero-config default): no inter-process race, so
  it uses the store's ``due_before(now)`` then advances + updates each schedule
  before dispatching.

Vendor-neutral by construction: no cloud-provider identifiers, ARNs, hostnames,
or secrets live here — only a generic poll interval and the generic DSN already
carried by the store.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

from cograph_client.scheduling.models import Schedule
from cograph_client.scheduling.next_run import compute_next_run
from cograph_client.scheduling.store import (
    PostgresScheduleStore,
    ScheduleStore,
    make_schedule_store,
)

logger = structlog.stdlib.get_logger("cograph.scheduler")

# How often the loop wakes to look for due schedules. A schedule fires at most
# one poll-interval late, which is fine for hour/day-scale recurrences.
_DEFAULT_POLL_SECONDS = 30.0
# Max due rows claimed per tick (bounds work per wake + the FOR UPDATE batch).
_DEFAULT_BATCH = 50


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _poll_seconds() -> float:
    raw = os.environ.get("COGRAPH_SCHEDULER_POLL_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_POLL_SECONDS
    try:
        val = float(raw)
    except ValueError:
        return _DEFAULT_POLL_SECONDS
    # A non-positive poll would spin; clamp to the default.
    return val if val > 0 else _DEFAULT_POLL_SECONDS


def _advance(schedule: Schedule, now: datetime) -> Schedule:
    """Return a copy of ``schedule`` with ``next_run`` advanced past ``now``.

    ``compute_next_run`` advances exactly one step; if many ticks were missed
    (process was down) we step forward until ``next_run`` is in the future so the
    schedule fires ONCE now and resumes on cadence — it does not fire once per
    missed tick. ``last_run`` is set to ``now``.
    """
    advanced = schedule.model_copy(deep=True)
    advanced.last_run = now
    nxt = compute_next_run(advanced, now)
    # compute_next_run is strictly-after ``now`` for intervals/cron, so a single
    # call already clears ``now``; loop defensively (bounded) in case a custom
    # recurrence returns <= now, and to skip any backlog deterministically.
    guard = 0
    while nxt <= now and guard < 10000:
        nxt = compute_next_run(advanced, nxt)
        guard += 1
    advanced.next_run = nxt
    return advanced


class ScheduleRunner:
    """Owns a single asyncio task that polls for and fires due schedules.

    Construct with the per-process deps (schedule store + the action-dispatch
    deps: neptune client, job store, enrichment executor). Premium suggestion is
    handled inside :func:`dispatch_scheduled_action` (degrades when unwired), so
    no recommender is threaded here.

    ``start()`` is idempotent; ``stop()`` cancels and awaits the loop.
    """

    def __init__(
        self,
        *,
        store: ScheduleStore,
        neptune_client: Any,
        job_store: Any,
        executor: Any,
        poll_seconds: Optional[float] = None,
        batch_size: int = _DEFAULT_BATCH,
    ) -> None:
        self._store = store
        self._neptune = neptune_client
        self._jobs = job_store
        self._executor = executor
        self._poll = poll_seconds if poll_seconds is not None else _poll_seconds()
        self._batch = batch_size
        self._task: Optional[asyncio.Task] = None
        self._running = False

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("scheduler_started", poll_seconds=self._poll)

    async def stop(self) -> None:
        self._running = False
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        logger.info("scheduler_stopped")

    async def _loop(self) -> None:
        while self._running:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad tick must not kill the loop
                logger.warning("scheduler_tick_failed", error=str(exc))
            try:
                await asyncio.sleep(self._poll)
            except asyncio.CancelledError:
                raise

    # -- one sweep -------------------------------------------------------------

    async def tick(self) -> int:
        """Fire all currently-due schedules once. Returns the number dispatched."""
        now = _now()
        if isinstance(self._store, PostgresScheduleStore):
            claimed = await self._claim_due_postgres(now)
        else:
            claimed = await self._claim_due_inmemory(now)
        for schedule in claimed:
            await self._dispatch(schedule)
        if claimed:
            logger.info("scheduler_fired", count=len(claimed))
        return len(claimed)

    async def _dispatch(self, schedule: Schedule) -> None:
        # Imported here (not at module top) to avoid a route<->scheduler import
        # cycle and to keep the dispatch wiring in one place (actions.py).
        from cograph_client.api.routes.actions import dispatch_scheduled_action

        try:
            await dispatch_scheduled_action(
                schedule,
                client=self._neptune,
                job_store=self._jobs,
                executor=self._executor,
            )
        except Exception as exc:  # noqa: BLE001 - a failed action must not stop the sweep
            logger.warning(
                "scheduler_dispatch_failed",
                schedule_id=schedule.id,
                tenant=schedule.tenant_id,
                action=schedule.action,
                error=str(exc),
            )

    # -- in-memory claim -------------------------------------------------------

    async def _claim_due_inmemory(self, now: datetime) -> list[Schedule]:
        """Single-process path: read due, advance + persist, return the advanced
        copies. No inter-process claim is needed (one process owns the store)."""
        due = await self._store.due_before(now)
        claimed: list[Schedule] = []
        for schedule in due[: self._batch]:
            advanced = _advance(schedule, now)
            await self._store.update(advanced)
            claimed.append(advanced)
        return claimed

    # -- postgres claim (FOR UPDATE SKIP LOCKED) -------------------------------

    # One transaction selects+locks the due batch, advances each row, and UPDATEs
    # it; SKIP LOCKED means a second replica's tick claims a disjoint batch. The
    # action is dispatched only after this transaction COMMITs (see tick()).
    _SELECT_DUE_FOR_UPDATE = (
        "SELECT id, payload FROM {table} "
        "WHERE enabled = true AND next_run IS NOT NULL AND next_run <= $1 "
        "ORDER BY next_run "
        "FOR UPDATE SKIP LOCKED "
        "LIMIT $2"
    )
    _UPDATE_AFTER_FIRE = (
        "UPDATE {table} SET next_run = $2, updated_at = $3, payload = $4::jsonb "
        "WHERE id = $1"
    )

    def _select_sql(self) -> str:
        return self._SELECT_DUE_FOR_UPDATE.format(
            table=PostgresScheduleStore._TABLE
        )

    def _update_sql(self) -> str:
        return self._UPDATE_AFTER_FIRE.format(table=PostgresScheduleStore._TABLE)

    async def _claim_due_postgres(self, now: datetime) -> list[Schedule]:
        store: PostgresScheduleStore = self._store  # type: ignore[assignment]
        pool = await store._ensure_pool()
        select_sql = self._select_sql()
        update_sql = self._update_sql()
        claimed: list[Schedule] = []
        async with pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(select_sql, now, self._batch)
                for row in rows:
                    schedule = store._from_payload(row["payload"])
                    advanced = _advance(schedule, now)
                    await conn.execute(
                        update_sql,
                        advanced.id,
                        advanced.next_run,
                        _now(),
                        advanced.model_dump_json(),
                    )
                    claimed.append(advanced)
        # COMMIT has happened (transaction context exited); now it's safe to
        # dispatch the actions outside the lock.
        return claimed


def make_schedule_runner(app_state: Any) -> Optional[ScheduleRunner]:
    """Build a :class:`ScheduleRunner` from FastAPI ``app.state`` deps.

    Mirrors ``make_schedule_store()``: gating is configuration-driven.

    Enablement: the runner is enabled when a database_url is configured (the
    durable, shared backend) OR ``COGRAPH_SCHEDULER_ENABLED`` is explicitly
    truthy. It is disabled when ``COGRAPH_SCHEDULER_ENABLED`` is explicitly
    falsy. Returns ``None`` when disabled, so startup constructs no loop.

    Pulls the schedule store / neptune client / job store / executor off
    ``app_state`` if already attached, falling back to the same factories the
    deps use so this is safe to call before the first request populates them.
    """
    enabled_raw = os.environ.get("COGRAPH_SCHEDULER_ENABLED", "").strip().lower()
    from cograph_client.config import settings

    if enabled_raw in ("0", "false", "no", "off"):
        return None
    if enabled_raw in ("1", "true", "yes", "on"):
        enabled = True
    else:
        # Default: enabled when a durable store is configured.
        enabled = bool(settings.database_url)
    if not enabled:
        return None

    store = getattr(app_state, "schedule_store", None)
    if store is None:
        store = make_schedule_store()
        app_state.schedule_store = store

    # The job store + executor are lazily built by api.deps._ensure_enrichment_state
    # on first request; build them now if absent so the runner can fire before
    # any HTTP traffic arrives.
    from cograph_client.api.deps import _ensure_enrichment_state

    _ensure_enrichment_state(app_state)

    return ScheduleRunner(
        store=store,
        neptune_client=getattr(app_state, "neptune_client", None),
        job_store=app_state.enrichment_job_store,
        executor=app_state.enrichment_executor,
    )
