"""Tests for the durable plan store (COG-124).

Mirrors ``tests/test_jobs_actions.py``'s store tests: an in-memory round-trip,
the ``make_plan_store`` selection logic, the ``reset_plan_store`` helper, and the
``PostgresPlanStore`` lazy-pool/DDL/upsert path with ``asyncpg.create_pool``
monkeypatched (no live DB). A final end-to-end test confirms a confirm→execute
resolves the persisted plan by id through the store.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from cograph_client.agent.plan_store import (
    InMemoryPlanStore,
    PostgresPlanStore,
    StoredPlan,
    get_plan_store,
    make_plan_store,
    reset_plan_store,
)
from cograph_client.agent.registry import PlanStep


def _make_plan(
    plan_id: str = "plan-1",
    tenant_id: str = "test-tenant",
    session_id: str | None = "sess-1",
    created_at: datetime | None = None,
) -> StoredPlan:
    return StoredPlan(
        plan_id=plan_id,
        tenant_id=tenant_id,
        kg_name="kg1",
        type_name="Mentor",
        message="enrich company for mentors",
        steps=[
            PlanStep(
                capability="enrich",
                action="run_enrichment",
                params={"attributes": ["company"]},
                rationale="fill missing company",
                confidence=0.9,
            )
        ],
        session_id=session_id,
        created_at=created_at or datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# InMemoryPlanStore — round-trip + scoping
# ---------------------------------------------------------------------------


def test_inmemory_save_get_delete_roundtrip():
    store = InMemoryPlanStore()

    async def run():
        plan = _make_plan()
        await store.save(plan)
        got = await store.get("plan-1", "test-tenant")
        assert got is not None
        assert got.plan_id == "plan-1"
        assert got.tenant_id == "test-tenant"
        assert got.session_id == "sess-1"
        assert got.steps[0].capability == "enrich"
        assert got.steps[0].params == {"attributes": ["company"]}
        # Wrong tenant → not visible (tenant scoping).
        assert await store.get("plan-1", "other-tenant") is None
        # Delete removes it.
        await store.delete("plan-1", "test-tenant")
        assert await store.get("plan-1", "test-tenant") is None

    asyncio.run(run())


def test_inmemory_get_returns_copy_not_reference():
    """A mutation of a returned plan must not leak back into the store."""
    store = InMemoryPlanStore()

    async def run():
        await store.save(_make_plan())
        got = await store.get("plan-1", "test-tenant")
        got.status = "MUTATED"
        got.steps[0].action = "MUTATED"
        again = await store.get("plan-1", "test-tenant")
        assert again.status == "proposed"
        assert again.steps[0].action == "run_enrichment"

    asyncio.run(run())


def test_inmemory_delete_scoped_by_tenant():
    store = InMemoryPlanStore()

    async def run():
        await store.save(_make_plan())
        # A delete from the wrong tenant is a no-op.
        await store.delete("plan-1", "other-tenant")
        assert await store.get("plan-1", "test-tenant") is not None

    asyncio.run(run())


def test_inmemory_list_for_tenant_newest_first():
    """Mirrors the job-store contract: list returns newest first regardless of
    insertion order, matching PostgresPlanStore's ORDER BY created_at DESC."""
    store = InMemoryPlanStore()
    mid = _make_plan("mid", created_at=datetime(2026, 6, 2, tzinfo=timezone.utc))
    old = _make_plan("old", created_at=datetime(2026, 6, 1, tzinfo=timezone.utc))
    new = _make_plan("new", created_at=datetime(2026, 6, 3, tzinfo=timezone.utc))

    async def run():
        await store.save(mid)
        await store.save(old)
        await store.save(new)
        # A plan for a different tenant must not appear.
        await store.save(_make_plan("other", tenant_id="t2"))
        return await store.list_for_tenant("test-tenant")

    plans = asyncio.run(run())
    assert [p.plan_id for p in plans] == ["new", "mid", "old"]


def test_inmemory_list_for_session():
    store = InMemoryPlanStore()

    async def run():
        await store.save(_make_plan("a", session_id="s1"))
        await store.save(_make_plan("b", session_id="s2"))
        await store.save(_make_plan("c", session_id="s1"))
        return await store.list_for_session("s1")

    plans = asyncio.run(run())
    assert {p.plan_id for p in plans} == {"a", "c"}


# ---------------------------------------------------------------------------
# StoredPlan serialization round-trip (payload jsonb)
# ---------------------------------------------------------------------------


def test_storedplan_json_roundtrip_preserves_steps_and_meta():
    plan = _make_plan()
    rebuilt = StoredPlan.from_payload(plan.to_json())
    assert rebuilt.plan_id == plan.plan_id
    assert rebuilt.tenant_id == plan.tenant_id
    assert rebuilt.session_id == plan.session_id
    assert rebuilt.kg_name == plan.kg_name
    assert rebuilt.type_name == plan.type_name
    assert rebuilt.status == plan.status
    assert rebuilt.created_at == plan.created_at
    assert len(rebuilt.steps) == 1
    assert rebuilt.steps[0].capability == "enrich"
    assert rebuilt.steps[0].action == "run_enrichment"
    assert rebuilt.steps[0].params == {"attributes": ["company"]}
    # A pre-decoded dict payload (asyncpg with a codec) also works.
    import json as _json

    rebuilt2 = StoredPlan.from_payload(_json.loads(plan.to_json()))
    assert rebuilt2.plan_id == plan.plan_id


# ---------------------------------------------------------------------------
# Store factory + reset helper
# ---------------------------------------------------------------------------


def test_make_plan_store_inmemory_when_no_db(monkeypatch):
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "database_url", "")
    reset_plan_store()
    store = make_plan_store()
    assert isinstance(store, InMemoryPlanStore)
    # The zero-config selection returns the shared singleton.
    assert store is get_plan_store()


def test_make_plan_store_postgres_when_db_set(monkeypatch):
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "database_url", "postgresql://x/y")
    store = make_plan_store()
    assert isinstance(store, PostgresPlanStore)


def test_reset_plan_store_clears_singleton(monkeypatch):
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "database_url", "")

    async def run():
        reset_plan_store()
        s1 = make_plan_store()
        await s1.save(_make_plan())
        # Same process: the singleton still has it.
        assert await make_plan_store().get("plan-1", "test-tenant") is not None
        # Reset → a fresh, empty store.
        reset_plan_store()
        assert await make_plan_store().get("plan-1", "test-tenant") is None

    asyncio.run(run())


# ---------------------------------------------------------------------------
# PostgresPlanStore — no real DB, asyncpg.create_pool monkeypatched
# (mirrors tests/test_jobs_actions.py)
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


def test_postgres_store_save_runs_ddl_and_upsert(monkeypatch):
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    _patch_asyncpg(monkeypatch, conn)

    async def run():
        store = PostgresPlanStore(dsn="postgresql://fake/db")
        await store.save(_make_plan())

    asyncio.run(run())

    sqls = [r[1] for r in rec]
    # DDL (table + indexes) ran lazily on first use.
    assert any("CREATE TABLE IF NOT EXISTS cograph_plans" in s for s in sqls)
    assert any("CREATE INDEX IF NOT EXISTS" in s and "tenant_id" in s for s in sqls)
    assert any("CREATE INDEX IF NOT EXISTS" in s and "session_id" in s for s in sqls)
    # INSERT upsert with the mirrored queryable columns.
    insert = next(
        r for r in rec if r[0] == "execute" and "INSERT INTO cograph_plans" in r[1]
    )
    assert "ON CONFLICT (plan_id) DO UPDATE" in insert[1]
    params = insert[2]
    assert params[0] == "plan-1"        # plan_id
    assert params[1] == "test-tenant"   # tenant_id
    assert params[2] == "sess-1"        # session_id
    assert params[3] == "proposed"      # status
    # payload (param 7) is the JSON-serialized plan.
    assert '"plan_id":"plan-1"' in params[6].replace(" ", "")


def test_postgres_store_get_roundtrip(monkeypatch):
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    plan = _make_plan()
    conn.row = {"payload": plan.to_json()}
    _patch_asyncpg(monkeypatch, conn)

    async def run():
        store = PostgresPlanStore(dsn="postgresql://fake/db")
        got = await store.get("plan-1", "test-tenant")
        assert got is not None
        assert got.plan_id == "plan-1"
        assert got.steps[0].capability == "enrich"
        # Missing row → None.
        conn.row = None
        assert await store.get("nope", "test-tenant") is None

    asyncio.run(run())

    fetch = next(r for r in rec if r[0] == "fetchrow")
    assert "WHERE plan_id = $1 AND tenant_id = $2" in fetch[1]
    assert fetch[2] == ("plan-1", "test-tenant")


def test_postgres_store_list_for_tenant(monkeypatch):
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    p1 = _make_plan("a")
    p2 = _make_plan("b")
    conn.rows = [{"payload": p1.to_json()}, {"payload": p2.to_json()}]
    _patch_asyncpg(monkeypatch, conn)

    async def run():
        store = PostgresPlanStore(dsn="postgresql://fake/db")
        plans = await store.list_for_tenant("test-tenant")
        assert {p.plan_id for p in plans} == {"a", "b"}

    asyncio.run(run())

    fetch = next(r for r in rec if r[0] == "fetch")
    assert "WHERE tenant_id = $1" in fetch[1]
    assert "ORDER BY created_at DESC" in fetch[1]
    assert fetch[2] == ("test-tenant",)


def test_postgres_store_delete(monkeypatch):
    rec: list[tuple] = []
    conn = _FakeConn(rec)
    _patch_asyncpg(monkeypatch, conn)

    async def run():
        store = PostgresPlanStore(dsn="postgresql://fake/db")
        await store.delete("plan-1", "test-tenant")

    asyncio.run(run())

    delete = next(
        r for r in rec if r[0] == "execute" and "DELETE FROM cograph_plans" in r[1]
    )
    assert "WHERE plan_id = $1 AND tenant_id = $2" in delete[1]
    assert delete[2] == ("plan-1", "test-tenant")


@pytest.mark.integration
def test_postgres_store_real_db():
    """Optional real-DB smoke test; skipped unless OMNIX_DATABASE_URL is set."""
    import os

    dsn = os.environ.get("OMNIX_DATABASE_URL")
    if not dsn:
        pytest.skip("OMNIX_DATABASE_URL not set")

    async def run():
        store = PostgresPlanStore(dsn=dsn)
        plan = _make_plan(plan_id=f"it-{datetime.now().timestamp()}")
        await store.save(plan)
        got = await store.get(plan.plan_id, plan.tenant_id)
        assert got is not None and got.plan_id == plan.plan_id
        await store.delete(plan.plan_id, plan.tenant_id)
        assert await store.get(plan.plan_id, plan.tenant_id) is None

    asyncio.run(run())


# ---------------------------------------------------------------------------
# confirm→execute resolves a plan by id through the store
# ---------------------------------------------------------------------------


def test_confirm_execute_resolves_plan_via_store(monkeypatch):
    """The durable store backs the confirm→execute handoff: a plan saved by one
    call to ``make_plan_store()`` is resolved by ``execute_plan`` through another
    call — exactly the cross-call handoff the durable store exists to survive."""
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "database_url", "")
    reset_plan_store()

    from cograph_client.agent import planner as planner_mod
    from cograph_client.agent.registry import (
        AgentContext,
        register_capability,
        reset_capabilities,
    )

    reset_capabilities()

    executed: list[str] = []

    class _StubCap:
        name = "enrich"

        def describe(self):
            return "enrich attributes"

        async def plan(self, ctx, instruction):
            return [PlanStep(capability="enrich", action="run_enrichment")]

        async def execute(self, ctx, step):
            executed.append(step.id)
            return {"kind": "ack"}

    register_capability(_StubCap())

    async def fake_classify(ctx, message):
        return {"intent": "enrich", "clarify": ""}

    monkeypatch.setattr(planner_mod, "_classify", fake_classify)

    class _FakeNeptune:
        async def query(self, q):
            return {"head": {"vars": []}, "results": {"bindings": []}}

        async def update(self, q):
            return None

    ctx = AgentContext(tenant_id="t1", kg_name="kg1", neptune=_FakeNeptune())

    async def run():
        # Plan is persisted via the store, carrying the supplied session_id.
        plan_out = await planner_mod.handle(
            ctx, "enrich company", session={"id": "sess-42"}
        )
        assert plan_out["kind"] == "plan"
        plan_id = plan_out["plan_id"]
        # It's actually in the store, with the session_id threaded through.
        stored = await make_plan_store().get(plan_id, "t1")
        assert stored is not None
        assert stored.session_id == "sess-42"
        # confirm→execute resolves the SAME plan by id through the store.
        result = await planner_mod.execute_plan(ctx, plan_id)
        assert result["kind"] == "result"
        assert all(s["status"] == "ok" for s in result["steps"])
        # The plan's terminal status was persisted back.
        done = await make_plan_store().get(plan_id, "t1")
        assert done.status == "done"
        # A wrong-tenant confirm cannot resolve another tenant's plan.
        other = AgentContext(
            tenant_id="t2", kg_name="kg1", neptune=_FakeNeptune()
        )
        miss = await planner_mod.execute_plan(other, plan_id)
        assert miss["kind"] == "error"

    asyncio.run(run())
    reset_capabilities()
    reset_plan_store()
    assert len(executed) == 1
