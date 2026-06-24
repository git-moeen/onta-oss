"""Tests for COG-101 (unified Jobs API + Postgres JobStore) and COG-99
(Ask-AI action endpoints)."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import pytest

from cograph_client.enrichment.job_store import (
    InMemoryJobStore,
    PostgresJobStore,
    make_job_store,
)
from cograph_client.enrichment.models import (
    ConflictPolicy,
    EnrichJob,
    EnrichmentTier,
    JobCategory,
    JobProgress,
    JobStatus,
    JobTrigger,
    job_to_summary,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(
    *,
    job_id: str = "job-1",
    tenant_id: str = "test-tenant",
    category: JobCategory = JobCategory.enrichment,
    trigger: JobTrigger = JobTrigger.manual,
    status: JobStatus = JobStatus.queued,
    progress: JobProgress | None = None,
    created_at: datetime | None = None,
) -> EnrichJob:
    return EnrichJob(
        id=job_id,
        tenant_id=tenant_id,
        kg_name="kg",
        type_name="Product",
        attributes=["manufacturer"],
        tier=EnrichmentTier.lite,
        status=status,
        progress=progress or JobProgress(),
        created_at=created_at or datetime.now(timezone.utc),
        conflict_policy=ConflictPolicy.stage,
        category=category,
        trigger=trigger,
    )


# ---------------------------------------------------------------------------
# Model fields + job_to_summary (COG-101)
# ---------------------------------------------------------------------------


def test_new_model_fields_default():
    job = _make_job()
    # New optional fields default to backward-compatible values.
    assert job.category == JobCategory.enrichment
    assert job.trigger == JobTrigger.manual
    assert job.last_run is None
    assert job.next_run is None
    assert job.cost is None
    assert job.cost_note is None


def test_job_to_summary_populates_new_fields():
    job = _make_job(
        category=JobCategory.dedupe,
        trigger=JobTrigger.scheduled,
        progress=JobProgress(total=4, processed=1),
    )
    job.cost = 0.5
    job.cost_note = "estimate"
    job.last_run = datetime(2026, 6, 1, tzinfo=timezone.utc)
    summary = job_to_summary(job)
    assert summary.category == JobCategory.dedupe
    assert summary.trigger == JobTrigger.scheduled
    assert summary.cost == 0.5
    assert summary.cost_note == "estimate"
    assert summary.last_run == job.last_run
    # progress_pct = round(1/4 * 100) = 25
    assert summary.progress_pct == 25


def test_progress_pct_edge_cases():
    # total == 0 → 0 (no division by zero)
    assert job_to_summary(_make_job(progress=JobProgress())).progress_pct == 0
    # over-count clamps to 100
    full = job_to_summary(
        _make_job(progress=JobProgress(total=2, processed=5))
    )
    assert full.progress_pct == 100


# ---------------------------------------------------------------------------
# Store factory (COG-101)
# ---------------------------------------------------------------------------


def test_make_job_store_inmemory_when_no_db(monkeypatch):
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "database_url", "")
    store = make_job_store()
    assert isinstance(store, InMemoryJobStore)


def test_make_job_store_postgres_when_db_set(monkeypatch):
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "database_url", "postgresql://x/y")
    store = make_job_store()
    assert isinstance(store, PostgresJobStore)


def test_inmemory_list_for_tenant_newest_first():
    """InMemoryJobStore must match the unified /jobs "newest first" contract
    (PostgresJobStore sorts ORDER BY created_at DESC) even when jobs are created
    out of chronological order — not dict-insertion order."""
    store = InMemoryJobStore()
    # Created OUT of chronological order: insert middle, then oldest, then newest.
    mid = _make_job(job_id="mid", created_at=datetime(2026, 6, 2, tzinfo=timezone.utc))
    old = _make_job(job_id="old", created_at=datetime(2026, 6, 1, tzinfo=timezone.utc))
    new = _make_job(job_id="new", created_at=datetime(2026, 6, 3, tzinfo=timezone.utc))

    async def run():
        await store.create(mid)
        await store.create(old)
        await store.create(new)
        return await store.list_for_tenant("test-tenant")

    summaries = asyncio.run(run())
    assert [s.id for s in summaries] == ["new", "mid", "old"]


# ---------------------------------------------------------------------------
# PostgresJobStore — no real DB, asyncpg.create_pool monkeypatched
# ---------------------------------------------------------------------------


class _FakeConn:
    """Records SQL + params; returns canned rows for fetch/fetchrow."""

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
    def __init__(self, conn: _FakeConn, pool: "_FakePool", timeout):
        self._conn = conn
        self._pool = pool
        self._timeout = timeout

    async def __aenter__(self):
        # Mimic asyncpg: a saturated pool waits up to `timeout`, then raises
        # asyncio.TimeoutError. The bounded acquire (COG-112) is what converts
        # that wait from "forever" into a surfaced failure.
        if self._pool.block_forever:
            await asyncio.wait_for(asyncio.Event().wait(), timeout=self._timeout)
        return self._conn

    async def __aexit__(self, *exc):
        return False


class _FakePool:
    def __init__(self, conn: _FakeConn):
        self._conn = conn
        # When True, acquire() never hands out a connection (saturated pool) and
        # relies on the caller's `timeout` to break the wait.
        self.block_forever = False
        # Records the `timeout` value passed to each acquire() call so tests can
        # assert the store bounds its acquires (COG-112).
        self.acquire_timeouts: list = []

    def acquire(self, timeout=None):
        self.acquire_timeouts.append(timeout)
        return _AcquireCtx(self._conn, self, timeout)


def _patch_asyncpg(monkeypatch, conn: _FakeConn) -> "_FakePool":
    import asyncpg

    pool = _FakePool(conn)

    async def fake_create_pool(*args, **kwargs):
        return pool

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    return pool


def test_postgres_store_create_runs_ddl_and_insert(monkeypatch):
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    _patch_asyncpg(monkeypatch, conn)

    async def run():
        store = PostgresJobStore(dsn="postgresql://fake/db")
        await store.create(_make_job())

    asyncio.run(run())

    sqls = [r[1] for r in rec]
    # DDL (table + index) ran on first use.
    assert any("CREATE TABLE IF NOT EXISTS cograph_jobs" in s for s in sqls)
    assert any("CREATE INDEX IF NOT EXISTS" in s and "tenant_id" in s for s in sqls)
    # INSERT (upsert) with the queryable columns mirrored.
    insert = next(r for r in rec if r[0] == "execute" and "INSERT INTO cograph_jobs" in r[1])
    assert "ON CONFLICT (id) DO UPDATE" in insert[1]
    params = insert[2]
    assert params[0] == "job-1"           # id
    assert params[1] == "test-tenant"     # tenant_id
    assert params[2] == "enrichment"      # category
    assert params[3] == "manual"          # trigger
    assert params[4] == "queued"          # status
    # payload is the JSON-serialized job (param 11).
    assert '"id":"job-1"' in params[10].replace(" ", "")


def test_postgres_store_get_roundtrip(monkeypatch):
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    job = _make_job()
    conn.row = {"payload": job.model_dump_json()}
    _patch_asyncpg(monkeypatch, conn)

    async def run():
        store = PostgresJobStore(dsn="postgresql://fake/db")
        got = await store.get("job-1")
        assert got is not None
        assert got.id == "job-1"
        assert got.category == JobCategory.enrichment
        # Missing row → None
        conn.row = None
        assert await store.get("nope") is None

    asyncio.run(run())

    assert any(r[0] == "fetchrow" and "WHERE id = $1" in r[1] for r in rec)


def test_postgres_store_list_returns_summaries(monkeypatch):
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    j1 = _make_job(job_id="a", category=JobCategory.dedupe,
                   progress=JobProgress(total=2, processed=1))
    j2 = _make_job(job_id="b", category=JobCategory.enrichment)
    conn.rows = [
        {"payload": j1.model_dump_json()},
        {"payload": j2.model_dump_json()},
    ]
    _patch_asyncpg(monkeypatch, conn)

    async def run():
        store = PostgresJobStore(dsn="postgresql://fake/db")
        summaries = await store.list_for_tenant("test-tenant")
        assert {s.id for s in summaries} == {"a", "b"}
        a = next(s for s in summaries if s.id == "a")
        assert a.category == JobCategory.dedupe
        assert a.progress_pct == 50

    asyncio.run(run())

    fetch = next(r for r in rec if r[0] == "fetch")
    assert "WHERE tenant_id = $1" in fetch[1]
    assert fetch[2] == ("test-tenant",)


def test_postgres_store_update_and_delete(monkeypatch):
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    _patch_asyncpg(monkeypatch, conn)

    async def run():
        store = PostgresJobStore(dsn="postgresql://fake/db")
        job = _make_job()
        await store.update(job)   # update == upsert
        await store.delete("job-1")

    asyncio.run(run())

    assert any(r[0] == "execute" and "INSERT INTO cograph_jobs" in r[1] for r in rec)
    delete = next(r for r in rec if r[0] == "execute" and "DELETE FROM cograph_jobs" in r[1])
    assert delete[2] == ("job-1",)


def test_postgres_store_bounds_acquire_timeout(monkeypatch):
    """COG-112: the store must pass a finite ``timeout`` to ``pool.acquire`` so a
    saturated/cold pool fails fast instead of hanging forever. asyncpg's
    ``acquire(timeout=None)`` waits indefinitely — the executor's first
    post-select ``jobs.update`` stalled there, with no exception and no
    completion (the production hang)."""
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    pool = _patch_asyncpg(monkeypatch, conn)

    async def run():
        store = PostgresJobStore(dsn="postgresql://fake/db", acquire_timeout=7.5)
        await store.create(_make_job())
        await store.get("job-1")

    asyncio.run(run())

    # Every acquire (DDL bootstrap + create + get) carried the finite timeout,
    # never None (the unbounded asyncpg default).
    assert pool.acquire_timeouts, "expected at least one acquire"
    assert all(t == 7.5 for t in pool.acquire_timeouts), pool.acquire_timeouts
    assert None not in pool.acquire_timeouts


def test_postgres_store_saturated_pool_raises_not_hangs(monkeypatch):
    """COG-112 root-cause guard: a saturated pool (acquire can never be granted)
    must raise ``asyncio.TimeoutError`` within the bound, NOT hang forever. We
    wrap in ``wait_for(..., 3s)`` so a regression to the unbounded acquire fails
    the test (timeout) instead of hanging CI."""
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    pool = _patch_asyncpg(monkeypatch, conn)
    pool.block_forever = True  # connection is never handed out

    async def run():
        store = PostgresJobStore(dsn="postgresql://fake/db", acquire_timeout=0.2)
        with pytest.raises(asyncio.TimeoutError):
            await store.create(_make_job())

    asyncio.run(asyncio.wait_for(run(), timeout=3))


@pytest.mark.integration
def test_postgres_store_real_db():
    """Optional real-DB smoke test; skipped unless OMNIX_DATABASE_URL is set."""
    dsn = os.environ.get("OMNIX_DATABASE_URL")
    if not dsn:
        pytest.skip("OMNIX_DATABASE_URL not set")

    async def run():
        store = PostgresJobStore(dsn=dsn)
        job = _make_job(job_id=f"it-{datetime.now().timestamp()}")
        await store.create(job)
        got = await store.get(job.id)
        assert got is not None and got.id == job.id
        summaries = await store.list_for_tenant("test-tenant")
        assert any(s.id == job.id for s in summaries)
        await store.delete(job.id)
        assert await store.get(job.id) is None

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Unified GET /graphs/{tenant}/jobs (COG-101) + actions (COG-99)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    from cograph_client.enrichment.cache import reset_enrichment_cache
    from cograph_client.enrichment.job_store import reset_job_store

    reset_job_store()
    reset_enrichment_cache()
    yield
    reset_job_store()
    reset_enrichment_cache()


def _seed(job: EnrichJob):
    from cograph_client.enrichment.job_store import get_job_store

    async def go():
        await get_job_store().create(job)

    asyncio.run(go())


def test_unified_jobs_list_and_category_filter(client, auth_headers):
    _seed(_make_job(job_id="d1", category=JobCategory.dedupe))
    _seed(_make_job(job_id="e1", category=JobCategory.enrichment))

    r = client.get("/graphs/test-tenant/jobs", headers=auth_headers)
    assert r.status_code == 200
    ids = {j["id"] for j in r.json()}
    assert ids == {"d1", "e1"}
    # Summaries carry the unified fields.
    sample = r.json()[0]
    assert "category" in sample and "trigger" in sample and "progress_pct" in sample

    r2 = client.get(
        "/graphs/test-tenant/jobs?category=dedupe", headers=auth_headers
    )
    assert r2.status_code == 200
    ids2 = {j["id"] for j in r2.json()}
    assert ids2 == {"d1"}


def test_unified_jobs_list_newest_first(client, auth_headers):
    # Seed out of chronological order; the endpoint must return newest first.
    _seed(_make_job(job_id="mid", created_at=datetime(2026, 6, 2, tzinfo=timezone.utc)))
    _seed(_make_job(job_id="old", created_at=datetime(2026, 6, 1, tzinfo=timezone.utc)))
    _seed(_make_job(job_id="new", created_at=datetime(2026, 6, 3, tzinfo=timezone.utc)))

    r = client.get("/graphs/test-tenant/jobs", headers=auth_headers)
    assert r.status_code == 200
    assert [j["id"] for j in r.json()] == ["new", "mid", "old"]


def test_action_find_merge_duplicates_creates_dedupe_job(
    client, auth_headers, monkeypatch
):
    # Patch rebuild_kg so no real ER runs.
    async def fake_rebuild(client_, instance_graph):
        return {"types": [{"type": "Product"}], "fragments_absorbed_total": 3}

    import cograph_client.resolver.er.rebuild as rebuild_mod

    monkeypatch.setattr(rebuild_mod, "rebuild_kg", fake_rebuild)

    r = client.post(
        "/graphs/test-tenant/actions/find-merge-duplicates",
        headers=auth_headers,
        json={"kg_name": "kg"},
    )
    assert r.status_code == 202
    data = r.json()
    assert "job_id" in data
    assert data["poll_url"].endswith(data["job_id"])

    # The job exists in the store and is a dedupe job. The background task may
    # or may not have completed; either way category is dedupe.
    listing = client.get(
        "/graphs/test-tenant/jobs?category=dedupe", headers=auth_headers
    ).json()
    assert any(j["id"] == data["job_id"] for j in listing)


def test_run_dedupe_schedules_stats_recompute(monkeypatch):
    """A successful dedupe must refresh the Explorer stats (recompute) for the
    job's (tenant, kg) — the dedupe worker collapses fragments and changes
    per-type counts, mirroring the er-rebuild route."""
    from unittest.mock import AsyncMock

    import cograph_client.api.routes.actions as actions_mod
    import cograph_client.api.routes.explore as explore_mod
    import cograph_client.resolver.er.rebuild as rebuild_mod

    async def fake_rebuild(client_, instance_graph):
        return {"types": [{"type": "Product"}], "fragments_absorbed_total": 3}

    monkeypatch.setattr(rebuild_mod, "rebuild_kg", fake_rebuild)

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        explore_mod,
        "schedule_recompute",
        lambda client, tenant_id, kg_name: calls.append((tenant_id, kg_name)),
    )

    async def run():
        store = InMemoryJobStore()
        job = _make_job(job_id="dd-1", category=JobCategory.dedupe)
        await store.create(job)
        await actions_mod._run_dedupe(
            AsyncMock(), store, job.id, "test-tenant", "kg"
        )
        final = await store.get(job.id)
        assert final.status == JobStatus.applied

    asyncio.run(run())
    assert calls == [("test-tenant", "kg")]


def test_run_dedupe_failure_does_not_recompute(monkeypatch):
    """A failed rebuild writes nothing → no recompute scheduled."""
    from unittest.mock import AsyncMock

    import cograph_client.api.routes.actions as actions_mod
    import cograph_client.api.routes.explore as explore_mod
    import cograph_client.resolver.er.rebuild as rebuild_mod

    async def boom(client_, instance_graph):
        raise RuntimeError("rebuild blew up")

    monkeypatch.setattr(rebuild_mod, "rebuild_kg", boom)

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        explore_mod,
        "schedule_recompute",
        lambda client, tenant_id, kg_name: calls.append((tenant_id, kg_name)),
    )

    async def run():
        store = InMemoryJobStore()
        job = _make_job(job_id="dd-2", category=JobCategory.dedupe)
        await store.create(job)
        await actions_mod._run_dedupe(
            AsyncMock(), store, job.id, "test-tenant", "kg"
        )
        final = await store.get(job.id)
        assert final.status == JobStatus.failed

    asyncio.run(run())
    assert calls == []


def test_action_suggest_relationships_degrades(client, auth_headers):
    from cograph_client.api.routes import actions

    # No recommender wired (default) → terminal failed job, but still a job_id.
    actions.register_relationship_recommender(None)
    r = client.post(
        "/graphs/test-tenant/actions/suggest-relationships",
        headers=auth_headers,
        json={"kg_name": "kg"},
    )
    assert r.status_code == 202
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "failed"

    listing = client.get(
        "/graphs/test-tenant/jobs?category=reconciliation", headers=auth_headers
    ).json()
    job = next(j for j in listing if j["id"] == data["job_id"])
    assert job["category"] == "reconciliation"
    assert "premium" in (job["cost_note"] or "").lower()


def test_action_enrich_creates_enrichment_job(client, auth_headers, mock_neptune):
    # count_entities → 0 so the executor run loop is a no-op.
    mock_neptune.query.return_value = {
        "head": {"vars": ["n"]},
        "results": {"bindings": [{"n": {"type": "literal", "value": "0"}}]},
    }
    r = client.post(
        "/graphs/test-tenant/actions/enrich",
        headers=auth_headers,
        json={
            "type_name": "Product",
            "attributes": ["manufacturer"],
            "kg_name": "kg",
            "tier": "lite",
        },
    )
    assert r.status_code == 202
    data = r.json()
    assert "job_id" in data

    listing = client.get(
        "/graphs/test-tenant/jobs?category=enrichment", headers=auth_headers
    ).json()
    assert any(j["id"] == data["job_id"] for j in listing)


def test_action_enrich_threads_scope(client, auth_headers, mock_neptune):
    """The /actions/enrich body accepts scope (COG-112) and persists it on the job."""
    mock_neptune.query.return_value = {
        "head": {"vars": ["n"]},
        "results": {"bindings": [{"n": {"type": "literal", "value": "0"}}]},
    }
    r = client.post(
        "/graphs/test-tenant/actions/enrich",
        headers=auth_headers,
        json={
            "type_name": "Mentor",
            "attributes": ["bio"],
            "kg_name": "kg",
            "scope": {"predicate": "haslevel", "value": "Manager"},
        },
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    job = client.get(
        f"/graphs/test-tenant/enrich/jobs/{job_id}", headers=auth_headers
    ).json()
    assert job["scope"] == {"predicate": "haslevel", "value": "Manager"}


# ---------------------------------------------------------------------------
# COG-112 review: injection rejected at the API boundary (422)
# ---------------------------------------------------------------------------


def test_create_job_rejects_injecting_scope_predicate(
    client, auth_headers, mock_neptune
):
    """An injecting scope.predicate is rejected with 422 and never reaches the
    enrich pipeline (count_entities must not be called)."""
    r = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Mentor",
            "attributes": ["bio"],
            "kg_name": "kg",
            # `>` would break out of the predicate IRI in a naive builder.
            "scope": {"predicate": "haslevel> } UNION", "value": "Manager"},
        },
    )
    assert r.status_code == 422
    mock_neptune.query.assert_not_called()


def test_create_job_rejects_injecting_entity_uri(client, auth_headers, mock_neptune):
    """An injecting entity_uris entry is rejected with 422; never reaches SPARQL."""
    r = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Mentor",
            "attributes": ["bio"],
            "kg_name": "kg",
            "entity_uris": ["https://evil> } INSERT { ?s ?p ?o }"],
        },
    )
    assert r.status_code == 422
    mock_neptune.query.assert_not_called()


def test_action_enrich_rejects_injecting_entity_uri(
    client, auth_headers, mock_neptune
):
    """The /actions/enrich body applies the same entity_uris validation (422)."""
    r = client.post(
        "/graphs/test-tenant/actions/enrich",
        headers=auth_headers,
        json={
            "type_name": "Mentor",
            "attributes": ["bio"],
            "kg_name": "kg",
            "entity_uris": ["not-a-url"],
        },
    )
    assert r.status_code == 422
