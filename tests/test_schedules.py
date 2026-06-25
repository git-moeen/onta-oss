"""Tests for COG-135 (scheduling data seam: models + store + CRUD routes + SDK).

This covers the DATA SEAM only — storage, CRUD, and next-run computation. The
firing/scheduler loop is a separate task and is not exercised here.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from cograph_client.enrichment.models import JobCategory
from cograph_client.scheduling.models import Schedule
from cograph_client.scheduling.next_run import compute_next_run
from cograph_client.scheduling.store import (
    InMemoryScheduleStore,
    PostgresScheduleStore,
    make_schedule_store,
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


# ---------------------------------------------------------------------------
# Model validation (exactly one of cron / interval_seconds)
# ---------------------------------------------------------------------------


def test_schedule_requires_exactly_one_recurrence():
    # Both set → error.
    with pytest.raises(ValueError):
        _make_schedule(cron="0 * * * *", interval_seconds=3600)
    # Neither set → error.
    with pytest.raises(ValueError):
        _make_schedule(cron=None, interval_seconds=None)
    # Non-positive interval → error.
    with pytest.raises(ValueError):
        _make_schedule(interval_seconds=0)
    # Exactly one → ok.
    assert _make_schedule(interval_seconds=60).interval_seconds == 60
    assert _make_schedule(cron="0 * * * *", interval_seconds=None).cron == "0 * * * *"


def test_schedule_defaults():
    s = _make_schedule()
    assert s.enabled is True
    assert s.next_run is None
    assert s.last_run is None
    assert s.category == JobCategory.enrichment


# ---------------------------------------------------------------------------
# compute_next_run
# ---------------------------------------------------------------------------


def test_compute_next_run_interval():
    after = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    s = _make_schedule(interval_seconds=900)
    assert compute_next_run(s, after) == after + timedelta(seconds=900)


def test_compute_next_run_cron_best_effort():
    """Cron is best-effort: with croniter it computes; without it raises a clear
    NotImplementedError. Either outcome is acceptable (croniter is optional)."""
    after = datetime(2026, 6, 1, 12, 30, 0, tzinfo=timezone.utc)
    s = _make_schedule(cron="0 * * * *", interval_seconds=None)
    try:
        import croniter  # noqa: F401
    except ImportError:
        with pytest.raises(NotImplementedError):
            compute_next_run(s, after)
    else:
        nxt = compute_next_run(s, after)
        # Next top-of-hour after 12:30 is 13:00.
        assert nxt == datetime(2026, 6, 1, 13, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Store factory (in-memory vs postgres)
# ---------------------------------------------------------------------------


def test_make_schedule_store_inmemory_when_no_db(monkeypatch):
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "database_url", "")
    store = make_schedule_store()
    assert isinstance(store, InMemoryScheduleStore)


def test_make_schedule_store_postgres_when_db_set(monkeypatch):
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "database_url", "postgresql://x/y")
    store = make_schedule_store()
    assert isinstance(store, PostgresScheduleStore)


# ---------------------------------------------------------------------------
# InMemoryScheduleStore — create / list / due_before
# ---------------------------------------------------------------------------


def test_inmemory_create_get_list_delete():
    store = InMemoryScheduleStore()

    async def run():
        a = _make_schedule(schedule_id="a", created_at=datetime(2026, 6, 1, tzinfo=timezone.utc))
        b = _make_schedule(schedule_id="b", created_at=datetime(2026, 6, 2, tzinfo=timezone.utc))
        # Insert out of order; list_for_tenant must return oldest-first (created order).
        await store.create(b)
        await store.create(a)
        got = await store.get("a")
        assert got is not None and got.id == "a"
        listed = await store.list_for_tenant("test-tenant")
        assert [s.id for s in listed] == ["a", "b"]
        # A different tenant sees nothing.
        assert await store.list_for_tenant("other") == []
        await store.delete("a")
        assert await store.get("a") is None

    asyncio.run(run())


def test_inmemory_due_before_filters_enabled_and_time():
    store = InMemoryScheduleStore()
    now = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)

    async def run():
        # Due (enabled, next_run in the past).
        due = _make_schedule(
            schedule_id="due", next_run=now - timedelta(minutes=5)
        )
        # Not yet due (next_run in the future).
        future = _make_schedule(
            schedule_id="future", next_run=now + timedelta(minutes=5)
        )
        # Disabled (even though next_run is past) → excluded.
        disabled = _make_schedule(
            schedule_id="disabled", enabled=False, next_run=now - timedelta(minutes=5)
        )
        # next_run unset → never due.
        unset = _make_schedule(schedule_id="unset", next_run=None)
        for s in (due, future, disabled, unset):
            await store.create(s)
        result = await store.due_before(now)
        assert [s.id for s in result] == ["due"]

    asyncio.run(run())


# ---------------------------------------------------------------------------
# PostgresScheduleStore — no real DB, asyncpg.create_pool monkeypatched
# ---------------------------------------------------------------------------


class _FakeConn:
    def __init__(self, recorder: list[tuple]):
        self._rec = recorder
        self.rows: list[dict] = []
        self.row: dict | None = None

    async def execute(self, sql, *params):
        self._rec.append(("execute", sql, params))
        return "OK"

    async def fetchrow(self, sql, *params):
        self._rec.append(("fetchrow", sql, params))
        return self.row

    async def fetch(self, sql, *params):
        self._rec.append(("fetch", sql, params))
        return self.rows


class _AcquireCtx:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn

    def acquire(self):
        return _AcquireCtx(self._conn)


def _patch_asyncpg(monkeypatch, conn: _FakeConn):
    import asyncpg

    async def fake_create_pool(*args, **kwargs):
        return _FakePool(conn)

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)


def test_postgres_store_create_runs_ddl_and_insert(monkeypatch):
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    _patch_asyncpg(monkeypatch, conn)

    async def run():
        store = PostgresScheduleStore(dsn="postgresql://fake/db")
        await store.create(_make_schedule(next_run=datetime(2026, 6, 1, tzinfo=timezone.utc)))

    asyncio.run(run())

    sqls = [r[1] for r in rec]
    assert any("CREATE TABLE IF NOT EXISTS cograph_schedules" in s for s in sqls)
    assert any("CREATE INDEX IF NOT EXISTS" in s and "tenant_id" in s for s in sqls)
    insert = next(
        r for r in rec if r[0] == "execute" and "INSERT INTO cograph_schedules" in r[1]
    )
    assert "ON CONFLICT (id) DO UPDATE" in insert[1]
    params = insert[2]
    assert params[0] == "sched-1"      # id
    assert params[1] == "test-tenant"  # tenant_id
    assert params[2] is True           # enabled
    # payload is the JSON-serialized schedule (param 7).
    assert '"id":"sched-1"' in params[6].replace(" ", "")


def test_postgres_store_get_roundtrip(monkeypatch):
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    sched = _make_schedule()
    conn.row = {"payload": sched.model_dump_json()}
    _patch_asyncpg(monkeypatch, conn)

    async def run():
        store = PostgresScheduleStore(dsn="postgresql://fake/db")
        got = await store.get("sched-1")
        assert got is not None and got.id == "sched-1"
        assert got.category == JobCategory.enrichment
        conn.row = None
        assert await store.get("nope") is None

    asyncio.run(run())
    assert any(r[0] == "fetchrow" and "WHERE id = $1" in r[1] for r in rec)


def test_postgres_store_list_orders_by_created_at(monkeypatch):
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    s1 = _make_schedule(schedule_id="a")
    conn.rows = [{"payload": s1.model_dump_json()}]
    _patch_asyncpg(monkeypatch, conn)

    async def run():
        store = PostgresScheduleStore(dsn="postgresql://fake/db")
        listed = await store.list_for_tenant("test-tenant")
        assert [s.id for s in listed] == ["a"]

    asyncio.run(run())
    fetch = next(r for r in rec if r[0] == "fetch")
    assert "WHERE tenant_id = $1" in fetch[1]
    assert "ORDER BY created_at" in fetch[1]
    assert fetch[2] == ("test-tenant",)


def test_postgres_store_due_before_query(monkeypatch):
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    _patch_asyncpg(monkeypatch, conn)
    now = datetime(2026, 6, 10, tzinfo=timezone.utc)

    async def run():
        store = PostgresScheduleStore(dsn="postgresql://fake/db")
        await store.due_before(now)

    asyncio.run(run())
    fetch = next(r for r in rec if r[0] == "fetch")
    assert "enabled = true" in fetch[1]
    assert "next_run <= $1" in fetch[1]
    assert fetch[2] == (now,)


def test_postgres_store_update_and_delete(monkeypatch):
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    _patch_asyncpg(monkeypatch, conn)

    async def run():
        store = PostgresScheduleStore(dsn="postgresql://fake/db")
        await store.update(_make_schedule())   # update == upsert
        await store.delete("sched-1")

    asyncio.run(run())
    assert any(
        r[0] == "execute" and "INSERT INTO cograph_schedules" in r[1] for r in rec
    )
    delete = next(
        r for r in rec if r[0] == "execute" and "DELETE FROM cograph_schedules" in r[1]
    )
    assert delete[2] == ("sched-1",)


# ---------------------------------------------------------------------------
# CRUD routes (COG-135)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_schedule_singleton():
    from cograph_client.scheduling.store import reset_schedule_store

    reset_schedule_store()
    yield
    reset_schedule_store()


def test_route_create_and_list_happy_path(client, auth_headers):
    # Force the in-memory store regardless of any ambient OMNIX_DATABASE_URL.
    from cograph_client.config import settings

    settings.database_url = ""

    r = client.post(
        "/graphs/test-tenant/schedules",
        headers=auth_headers,
        json={
            "kg_name": "kg",
            "category": "enrichment",
            "action": "enrich",
            "params": {"type_name": "Product", "attributes": ["manufacturer"]},
            "interval_seconds": 3600,
        },
    )
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["id"]
    assert created["enabled"] is True
    # Initial next_run is computed (interval path) → created_at + interval.
    assert created["next_run"] is not None
    assert created["tenant_id"] == "test-tenant"

    listing = client.get("/graphs/test-tenant/schedules", headers=auth_headers).json()
    assert any(s["id"] == created["id"] for s in listing)


def test_route_create_rejects_both_recurrences(client, auth_headers):
    from cograph_client.config import settings

    settings.database_url = ""
    r = client.post(
        "/graphs/test-tenant/schedules",
        headers=auth_headers,
        json={
            "kg_name": "kg",
            "category": "dedupe",
            "action": "find-merge-duplicates",
            "cron": "0 * * * *",
            "interval_seconds": 3600,
        },
    )
    assert r.status_code == 422


def test_route_get_patch_delete(client, auth_headers):
    from cograph_client.config import settings

    settings.database_url = ""
    created = client.post(
        "/graphs/test-tenant/schedules",
        headers=auth_headers,
        json={
            "kg_name": "kg",
            "category": "enrichment",
            "action": "enrich",
            "interval_seconds": 3600,
        },
    ).json()
    sid = created["id"]

    # GET one
    got = client.get(f"/graphs/test-tenant/schedules/{sid}", headers=auth_headers)
    assert got.status_code == 200
    assert got.json()["id"] == sid

    # PATCH disable
    patched = client.patch(
        f"/graphs/test-tenant/schedules/{sid}",
        headers=auth_headers,
        json={"enabled": False},
    )
    assert patched.status_code == 200
    assert patched.json()["enabled"] is False

    # DELETE
    deleted = client.delete(
        f"/graphs/test-tenant/schedules/{sid}", headers=auth_headers
    )
    assert deleted.status_code == 204
    missing = client.get(f"/graphs/test-tenant/schedules/{sid}", headers=auth_headers)
    assert missing.status_code == 404
