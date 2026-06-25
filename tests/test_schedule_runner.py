"""Tests for COG-136 (schedule firing loop: ScheduleRunner + dispatch).

Covers the FIRING side that COG-135's tests deliberately skipped: tick()
dispatches due schedules and advances next_run, the start/stop lifecycle, the
Postgres ``FOR UPDATE SKIP LOCKED`` claim SQL shape (asserted as text, no live
DB — same approach as the store SQL tests), the scheduled-dispatch reuse of the
route workers, and the disabled-when-no-DSN gating.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from cograph_client.enrichment.job_store import InMemoryJobStore
from cograph_client.enrichment.models import JobCategory, JobStatus, JobTrigger
from cograph_client.scheduling.models import Schedule
from cograph_client.scheduling.runner import (
    ScheduleRunner,
    _advance,
    make_schedule_runner,
)
from cograph_client.scheduling.store import (
    InMemoryScheduleStore,
    PostgresScheduleStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_schedule(
    *,
    schedule_id: str = "sched-1",
    tenant_id: str = "test-tenant",
    category: JobCategory = JobCategory.enrichment,
    action: str = "enrich",
    params: dict | None = None,
    cron: str | None = None,
    interval_seconds: int | None = 3600,
    enabled: bool = True,
    next_run: datetime | None = None,
    created_at: datetime | None = None,
) -> Schedule:
    return Schedule(
        id=schedule_id,
        tenant_id=tenant_id,
        kg_name="kg",
        category=category,
        action=action,
        params=params or {"type_name": "Product", "attributes": ["manufacturer"]},
        cron=cron,
        interval_seconds=interval_seconds,
        enabled=enabled,
        next_run=next_run,
        created_at=created_at or datetime.now(timezone.utc),
    )


class _CapturingExecutor:
    """Stands in for EnrichmentExecutor — records run() calls, no real work."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def run(self, job, tenant_id) -> None:  # noqa: ANN001
        self.calls.append((job.id, tenant_id))
        job.status = JobStatus.applied


def _runner(store, *, executor=None, job_store=None) -> ScheduleRunner:
    return ScheduleRunner(
        store=store,
        neptune_client=object(),
        job_store=job_store or InMemoryJobStore(),
        executor=executor or _CapturingExecutor(),
        poll_seconds=0.01,
        batch_size=50,
    )


# ---------------------------------------------------------------------------
# _advance — next_run moves strictly past now; backlog is skipped, not replayed
# ---------------------------------------------------------------------------


def test_advance_steps_past_now_for_interval():
    now = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
    # Due 1h ago, 15-min interval → next_run must land in the FUTURE (one fire
    # now, then resume on cadence), not 15 min after the stale next_run.
    s = _make_schedule(
        interval_seconds=900, next_run=now - timedelta(hours=1)
    )
    advanced = _advance(s, now)
    assert advanced.last_run == now
    assert advanced.next_run > now
    # And it is on the interval grid relative to now (within one step).
    assert advanced.next_run <= now + timedelta(seconds=900)


# ---------------------------------------------------------------------------
# tick() — in-memory path: dispatches due, advances next_run, tags scheduled
# ---------------------------------------------------------------------------


def test_tick_dispatches_due_and_advances(monkeypatch):
    captured: list[Schedule] = []

    async def fake_dispatch(schedule, *, client, job_store, executor):  # noqa: ANN001
        captured.append(schedule)

    import cograph_client.api.routes.actions as actions_mod

    monkeypatch.setattr(actions_mod, "dispatch_scheduled_action", fake_dispatch)

    store = InMemoryScheduleStore()
    now = datetime.now(timezone.utc)

    async def run():
        due = _make_schedule(
            schedule_id="due", next_run=now - timedelta(minutes=5)
        )
        future = _make_schedule(
            schedule_id="future", next_run=now + timedelta(minutes=30)
        )
        await store.create(due)
        await store.create(future)

        runner = _runner(store)
        fired = await runner.tick()
        assert fired == 1
        # Only the due schedule was dispatched, tagged via trigger=scheduled by
        # the dispatch (here captured pre-dispatch).
        assert [s.id for s in captured] == ["due"]
        # Its next_run advanced into the future and last_run was set.
        refreshed = await store.get("due")
        assert refreshed.next_run > now
        assert refreshed.last_run is not None
        # The not-yet-due one is untouched.
        assert (await store.get("future")).last_run is None

    asyncio.run(run())


def test_tick_no_due_dispatches_nothing(monkeypatch):
    calls: list = []

    async def fake_dispatch(schedule, **kw):  # noqa: ANN001
        calls.append(schedule)

    import cograph_client.api.routes.actions as actions_mod

    monkeypatch.setattr(actions_mod, "dispatch_scheduled_action", fake_dispatch)

    store = InMemoryScheduleStore()

    async def run():
        runner = _runner(store)
        assert await runner.tick() == 0

    asyncio.run(run())
    assert calls == []


# ---------------------------------------------------------------------------
# dispatch_scheduled_action — reuses the route workers, tags trigger=scheduled
# ---------------------------------------------------------------------------


def test_dispatch_enrich_runs_executor_with_scheduled_trigger():
    from cograph_client.api.routes.actions import dispatch_scheduled_action

    store = InMemoryJobStore()
    executor = _CapturingExecutor()
    sched = _make_schedule(
        action="enrich",
        category=JobCategory.enrichment,
        params={"type_name": "Product", "attributes": ["manufacturer"]},
    )

    async def run():
        job = await dispatch_scheduled_action(
            sched, client=object(), job_store=store, executor=executor
        )
        assert job.trigger == JobTrigger.scheduled
        assert job.category == JobCategory.enrichment
        assert job.type_name == "Product"
        # The same executor.run the /actions/enrich route calls was invoked.
        assert executor.calls == [(job.id, sched.tenant_id)]
        # The job is persisted in the store.
        assert (await store.get(job.id)) is not None

    asyncio.run(run())


def test_dispatch_dedupe_runs_dedupe_worker(monkeypatch):
    import cograph_client.api.routes.actions as actions_mod
    import cograph_client.api.routes.explore as explore_mod
    import cograph_client.resolver.er.rebuild as rebuild_mod
    from cograph_client.api.routes.actions import dispatch_scheduled_action

    async def fake_rebuild(client_, instance_graph):  # noqa: ANN001
        return {"types": [{"type": "Product"}], "fragments_absorbed_total": 2}

    monkeypatch.setattr(rebuild_mod, "rebuild_kg", fake_rebuild)
    monkeypatch.setattr(
        explore_mod, "schedule_recompute", lambda *a, **k: None
    )

    store = InMemoryJobStore()
    sched = _make_schedule(
        action="find-merge-duplicates",
        category=JobCategory.dedupe,
        params={},
    )

    async def run():
        from unittest.mock import AsyncMock

        job = await dispatch_scheduled_action(
            sched, client=AsyncMock(), job_store=store, executor=_CapturingExecutor()
        )
        assert job.trigger == JobTrigger.scheduled
        assert job.category == JobCategory.dedupe
        final = await store.get(job.id)
        assert final.status == JobStatus.applied

    asyncio.run(run())


def test_dispatch_suggest_degrades_without_recommender():
    from cograph_client.api.routes import actions
    from cograph_client.api.routes.actions import dispatch_scheduled_action

    actions.register_relationship_recommender(None)
    store = InMemoryJobStore()
    sched = _make_schedule(
        action="suggest-relationships",
        category=JobCategory.reconciliation,
        params={},
    )

    async def run():
        job = await dispatch_scheduled_action(
            sched, client=object(), job_store=store, executor=_CapturingExecutor()
        )
        assert job.trigger == JobTrigger.scheduled
        assert job.category == JobCategory.reconciliation
        # Mirrors the route's graceful degrade: terminal failed job, clear note.
        assert job.status == JobStatus.failed
        assert "premium" in (job.cost_note or "").lower()
        assert (await store.get(job.id)).status == JobStatus.failed

    asyncio.run(run())


# ---------------------------------------------------------------------------
# start() / stop() lifecycle
# ---------------------------------------------------------------------------


def test_start_stop_lifecycle(monkeypatch):
    ticks = {"n": 0}

    store = InMemoryScheduleStore()
    runner = _runner(store)

    async def counting_tick():
        ticks["n"] += 1

    monkeypatch.setattr(runner, "tick", counting_tick)

    async def run():
        runner.start()
        # start() is idempotent — a second call doesn't spawn a second task.
        first_task = runner._task
        runner.start()
        assert runner._task is first_task
        # Let the loop run a few polls (poll_seconds=0.01).
        await asyncio.sleep(0.05)
        await runner.stop()
        assert ticks["n"] >= 1
        # After stop the task is cleared and not running.
        assert runner._task is None
        assert runner._running is False

    asyncio.run(run())


def test_loop_survives_a_failing_tick(monkeypatch):
    calls = {"n": 0}
    store = InMemoryScheduleStore()
    runner = _runner(store)

    async def flaky_tick():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr(runner, "tick", flaky_tick)

    async def run():
        runner.start()
        await asyncio.sleep(0.05)
        await runner.stop()
        # The first tick raised but the loop kept going and ticked again.
        assert calls["n"] >= 2

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Postgres claim path — FOR UPDATE SKIP LOCKED SQL shape (no live DB)
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self, recorder: list[tuple], rows: list[dict]):
        self._rec = recorder
        self._rows = rows

    async def execute(self, sql, *params):
        self._rec.append(("execute", sql, params))
        return "OK"

    async def fetch(self, sql, *params):
        self._rec.append(("fetch", sql, params))
        return self._rows


class _Tx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _AcquireCtx:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return _AcquireCtx(self._conn)


def _conn_with_transaction(rec, rows):
    conn = _FakeConn(rec, rows)
    conn.transaction = lambda: _Tx(conn)  # type: ignore[attr-defined]
    return conn


def test_postgres_claim_uses_for_update_skip_locked(monkeypatch):
    """The Postgres tick must claim with SELECT ... FOR UPDATE SKIP LOCKED in a
    transaction, then UPDATE the advanced next_run. Asserted as SQL text shape,
    mirroring the store's DDL/SQL tests (no live DB)."""
    rec: list[tuple] = []
    now = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)
    due = _make_schedule(
        schedule_id="due", interval_seconds=900, next_run=now - timedelta(hours=1)
    )
    conn = _conn_with_transaction(rec, [{"id": "due", "payload": due.model_dump_json()}])

    store = PostgresScheduleStore(dsn="postgresql://fake/db")
    # Bypass the lazy pool creation with our fake pool.
    store._pool = _FakePool(conn)

    captured: list[Schedule] = []

    async def fake_dispatch(schedule, **kw):  # noqa: ANN001
        captured.append(schedule)

    import cograph_client.api.routes.actions as actions_mod

    monkeypatch.setattr(actions_mod, "dispatch_scheduled_action", fake_dispatch)

    async def run():
        runner = _runner(store)
        # Drive tick at a fixed now via the internal claim.
        fired = await runner.tick()
        assert fired == 1

    asyncio.run(run())

    select = next(r for r in rec if r[0] == "fetch")
    sql = select[1]
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "cograph_schedules" in sql
    assert "enabled = true" in sql
    assert "next_run <= $1" in sql
    assert "ORDER BY next_run" in sql
    assert "LIMIT $2" in sql
    # An UPDATE advancing next_run ran inside the transaction.
    update = next(r for r in rec if r[0] == "execute" and "UPDATE cograph_schedules" in r[1])
    assert "SET next_run = $2" in update[1]
    # And the due schedule was dispatched AFTER claim (one fire).
    assert [s.id for s in captured] == ["due"]
    # Its advanced next_run (param $2 of the update) is in the future.
    assert update[2][1] > now - timedelta(hours=1)


# ---------------------------------------------------------------------------
# make_schedule_runner — gating
# ---------------------------------------------------------------------------


class _State:
    """Minimal app.state stand-in."""


def test_make_runner_disabled_when_no_dsn(monkeypatch):
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "database_url", "")
    monkeypatch.delenv("COGRAPH_SCHEDULER_ENABLED", raising=False)
    assert make_schedule_runner(_State()) is None


def test_make_runner_enabled_when_dsn_set(monkeypatch):
    from cograph_client.config import settings
    from unittest.mock import AsyncMock

    monkeypatch.setattr(settings, "database_url", "postgresql://x/y")
    monkeypatch.delenv("COGRAPH_SCHEDULER_ENABLED", raising=False)

    state = _State()
    state.neptune_client = AsyncMock()
    runner = make_schedule_runner(state)
    assert isinstance(runner, ScheduleRunner)
    # The schedule store it picked is the durable Postgres one (DSN set).
    assert isinstance(runner._store, PostgresScheduleStore)


def test_make_runner_explicit_disable_overrides_dsn(monkeypatch):
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "database_url", "postgresql://x/y")
    monkeypatch.setenv("COGRAPH_SCHEDULER_ENABLED", "false")
    assert make_schedule_runner(_State()) is None


def test_make_runner_explicit_enable_without_dsn(monkeypatch):
    from cograph_client.config import settings
    from unittest.mock import AsyncMock

    monkeypatch.setattr(settings, "database_url", "")
    monkeypatch.setenv("COGRAPH_SCHEDULER_ENABLED", "true")

    state = _State()
    state.neptune_client = AsyncMock()
    runner = make_schedule_runner(state)
    assert isinstance(runner, ScheduleRunner)
    # No DSN → the in-memory store.
    assert isinstance(runner._store, InMemoryScheduleStore)
