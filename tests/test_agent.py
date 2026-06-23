"""Tests for the unified Ask-AI agent: registry, planner, clean-before-enrich.

Everything is stubbed so the suite is deterministic and fast (no network, no
real Neptune): the classifier LLM, the inference LLM, predicate sampling, and the
underlying enrichment executor / normalization apply. Each async test is wrapped
in ``asyncio.wait_for`` to fail loudly rather than hang.
"""

from __future__ import annotations

import asyncio

import pytest

from cograph_client.agent import planner as planner_mod
from cograph_client.agent.capabilities.enrich_cap import EnrichCapability
from cograph_client.agent.capabilities.normalize_cap import NormalizeCapability
from cograph_client.agent.capabilities.query import QueryCapability
from cograph_client.agent.planner import (
    execute_plan,
    handle,
    register_default_capabilities,
    reset_plan_store,
)
from cograph_client.agent.registry import (
    AgentContext,
    PlanStep,
    get_capabilities,
    get_capability,
    order_steps,
    register_capability,
    reset_capabilities,
)
from cograph_client.normalization.rules import NormalizationRule

TIMEOUT = 5.0


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeNeptune:
    """Returns no rows by default; tests that need sampled values patch the
    inference sampling helper directly instead of teaching this a SPARQL dialect."""

    def __init__(self):
        self.updates: list[str] = []

    async def query(self, q):
        return {"head": {"vars": []}, "results": {"bindings": []}}

    async def update(self, q):
        self.updates.append(q)
        return None


class FakeJobStore:
    def __init__(self):
        self.created = []

    async def create(self, job):
        self.created.append(job)

    async def get(self, job_id):
        for j in self.created:
            if j.id == job_id:
                return j
        return None


class FakeExecutor:
    def __init__(self):
        self.ran = []

    async def run(self, job, tenant_id):
        self.ran.append((job, tenant_id))


def _ctx(neptune=None, **extras_kw):
    return AgentContext(
        tenant_id="t1",
        kg_name="kg1",
        neptune=neptune or FakeNeptune(),
        type_name="Mentor",
        openrouter_key="fake-key",
        anthropic_key="fake-anthropic",
        extras={
            "enrichment_executor": extras_kw.get("executor", FakeExecutor()),
            "enrichment_job_store": extras_kw.get("job_store", FakeJobStore()),
        },
    )


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Each test starts from the default OSS capability set + an empty plan store."""
    reset_capabilities()
    reset_plan_store()
    register_default_capabilities()
    yield
    reset_capabilities()
    reset_plan_store()


@pytest.fixture(autouse=True)
def _track_bg_tasks(monkeypatch):
    """Schedule capability-spawned background coroutines as TRACKED tasks.

    Capabilities use the strong-ref ``_spawn`` (create_task) for long work
    (normalize apply / enrichment run). In tests we still want that work to run
    (so we can assert the executor/store was actually driven), just tracked so
    nothing leaks. We replace ``_spawn`` with one that creates a real task and
    keeps a strong ref — the underlying apply/run is itself stubbed per-test.
    """
    import cograph_client.agent.capabilities.enrich_cap as enrich_cap
    import cograph_client.agent.capabilities.normalize_cap as norm_cap

    spawned: list = []

    def tracking_spawn(coro):
        task = asyncio.ensure_future(coro)
        spawned.append(task)
        task.add_done_callback(lambda t: None)

    monkeypatch.setattr(norm_cap, "_spawn", tracking_spawn)
    monkeypatch.setattr(enrich_cap, "_spawn", tracking_spawn)
    return spawned


def _stub_classifier(monkeypatch, intent: str, clarify: str = ""):
    async def fake_chat(*args, **kwargs):
        import json

        return json.dumps({"intent": intent, "clarify": clarify})

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_chat)


# --------------------------------------------------------------------------- #
# 1. Registry roundtrip — adding a capability needs no route change
# --------------------------------------------------------------------------- #
def test_register_and_get_capability_roundtrip():
    names_before = {c.name for c in get_capabilities()}
    assert {"query", "normalize", "enrich"} <= names_before

    class DedupCapability:
        name = "dedup"

        def describe(self):
            return "merge duplicate entities"

        async def plan(self, ctx, instruction):
            return []

        async def execute(self, ctx, step):
            return {"kind": "ack"}

    register_capability(DedupCapability())
    assert get_capability("dedup") is not None
    assert "dedup" in {c.name for c in get_capabilities()}
    # No route/endpoint was added — the single endpoint dispatches by name.


def test_order_steps_respects_depends_on():
    a = PlanStep(capability="normalize", action="x")
    b = PlanStep(capability="enrich", action="y", depends_on=[a.id])
    ordered = order_steps([b, a])  # deliberately reversed input
    assert [s.id for s in ordered] == [a.id, b.id]


# --------------------------------------------------------------------------- #
# 2. Classifier routing
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_question_routes_to_answer(monkeypatch):
    _stub_classifier(monkeypatch, "question")

    async def fake_answer(self, ctx, q):
        return {"answer": "42", "sparql": "SELECT ...", "rows": [], "narrative": ""}

    monkeypatch.setattr(QueryCapability, "answer", fake_answer)

    out = await asyncio.wait_for(
        handle(_ctx(), "how many mentors are there?"), TIMEOUT
    )
    assert out["kind"] == "answer"
    assert out["answer"] == "42"
    assert out["sparql"].startswith("SELECT")


@pytest.mark.asyncio
async def test_enrich_routes_to_plan(monkeypatch):
    _stub_classifier(monkeypatch, "enrich")
    out = await asyncio.wait_for(
        handle(_ctx(), "enrich company for managers"), TIMEOUT
    )
    assert out["kind"] == "plan"
    assert out["plan_id"]
    steps = out["steps"]
    assert len(steps) == 1
    assert steps[0]["capability"] == "enrich"
    assert steps[0]["action"] == "run_enrichment"
    assert steps[0]["params"]["attributes"] == ["company"]


@pytest.mark.asyncio
async def test_ambiguous_routes_to_clarify(monkeypatch):
    _stub_classifier(monkeypatch, "ambiguous", clarify="Which field?")
    out = await asyncio.wait_for(handle(_ctx(), "do the thing"), TIMEOUT)
    assert out["kind"] == "clarify"
    assert out["question"] == "Which field?"


@pytest.mark.asyncio
async def test_unregistered_intent_clarifies(monkeypatch):
    # dedup is recognized by the classifier but no capability is registered in A1.
    _stub_classifier(monkeypatch, "dedup")
    out = await asyncio.wait_for(handle(_ctx(), "merge the duplicates"), TIMEOUT)
    assert out["kind"] == "clarify"


# --------------------------------------------------------------------------- #
# 3. clean-before-enrich composition
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_clean_before_enrich_composes_depends_on(monkeypatch):
    """'enrich company for mentors who speak Persian' where the speaks target
    sample is composite ('English__Persian') → the plan contains a normalize
    step that the enrich step depends_on, ordered normalize-first."""
    _stub_classifier(monkeypatch, "enrich")

    # The scope predicate 'speaks' has a composite target sample.
    async def fake_sample(neptune, tenant_id, kg, type_name, pred_leaf):
        assert pred_leaf == "speaks"
        return (["English__Persian", "English"], "relationship")

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.sample_predicate_values",
        fake_sample,
    )

    # The normalize capability's plan() infers a list_explode rule for 'speaks'.
    async def fake_suggest(neptune, tenant_id, kg, type_name, leaves):
        assert leaves == ["speaks"]
        return [
            NormalizationRule(
                id="kg1__Mentor__speaks",
                kg_name="kg1",
                type_name="Mentor",
                predicate="speaks",
                target_kind="relationship",
                rule_type="list_explode",
                params={"delimiters": ["__"], "target": "entity"},
                confidence=0.95,
                rationale="composite language values",
                sample_values=["English__Persian"],
            )
        ]

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.normalize_cap.suggest_rules_for_predicates",
        fake_suggest,
    )

    out = await asyncio.wait_for(
        handle(_ctx(), "enrich company for mentors who speak Persian"), TIMEOUT
    )
    assert out["kind"] == "plan"
    steps = out["steps"]
    assert len(steps) == 2
    normalize_step, enrich_step = steps[0], steps[1]
    assert normalize_step["capability"] == "normalize"
    assert enrich_step["capability"] == "enrich"
    # The enrich step depends on the normalize step → ordered normalize-first.
    assert enrich_step["depends_on"] == [normalize_step["id"]]
    # Scope was parsed from the NL.
    assert enrich_step["params"]["scope"] == {
        "predicate": "speaks",
        "value": "Persian",
    }
    # Dry-run preview shows the split.
    assert normalize_step["preview"]["rule_type"] == "list_explode"


@pytest.mark.asyncio
async def test_no_prereq_when_scope_target_atomic(monkeypatch):
    """If the scope target sample is already atomic, no normalize prerequisite."""
    _stub_classifier(monkeypatch, "enrich")

    async def fake_sample(neptune, tenant_id, kg, type_name, pred_leaf):
        return (["English", "Persian"], "relationship")  # atomic

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.sample_predicate_values",
        fake_sample,
    )
    out = await asyncio.wait_for(
        handle(_ctx(), "enrich company for mentors who speak Persian"), TIMEOUT
    )
    assert out["kind"] == "plan"
    assert len(out["steps"]) == 1
    assert out["steps"][0]["capability"] == "enrich"


# --------------------------------------------------------------------------- #
# 4. confirm/execute runs steps in dependency order
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_execute_plan_runs_in_dependency_order(monkeypatch):
    """A persisted [normalize→enrich] plan executes normalize before enrich."""
    _stub_classifier(monkeypatch, "enrich")

    async def fake_sample(neptune, tenant_id, kg, type_name, pred_leaf):
        return (["English__Persian"], "relationship")

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.sample_predicate_values",
        fake_sample,
    )

    async def fake_suggest(neptune, tenant_id, kg, type_name, leaves):
        return [
            NormalizationRule(
                id="kg1__Mentor__speaks",
                kg_name="kg1",
                type_name="Mentor",
                predicate="speaks",
                target_kind="relationship",
                rule_type="list_explode",
                params={"delimiters": ["__"], "target": "entity"},
                confidence=0.9,
                sample_values=["English__Persian"],
            )
        ]

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.normalize_cap.suggest_rules_for_predicates",
        fake_suggest,
    )

    # Record execution order across both capabilities.
    order: list[str] = []
    orig_norm_execute = NormalizeCapability.execute
    orig_enrich_execute = EnrichCapability.execute

    async def norm_execute(self, ctx, step):
        order.append("normalize")
        return await orig_norm_execute(self, ctx, step)

    async def enrich_execute(self, ctx, step):
        order.append("enrich")
        return await orig_enrich_execute(self, ctx, step)

    monkeypatch.setattr(NormalizeCapability, "execute", norm_execute)
    monkeypatch.setattr(EnrichCapability, "execute", enrich_execute)

    # Stub the normalize store save so execute() doesn't touch a real store.
    async def fake_save(self, tenant_id, rule):
        return None

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.normalize_cap.NormalizationRuleStore.save",
        fake_save,
    )

    job_store = FakeJobStore()
    executor = FakeExecutor()
    ctx = _ctx(executor=executor, job_store=job_store)

    plan_out = await asyncio.wait_for(
        handle(ctx, "enrich company for mentors who speak Persian"), TIMEOUT
    )
    assert plan_out["kind"] == "plan"
    plan_id = plan_out["plan_id"]

    result = await asyncio.wait_for(execute_plan(ctx, plan_id), TIMEOUT)
    assert result["kind"] == "result"
    assert order == ["normalize", "enrich"]  # dependency order honored
    # Let the spawned background tasks (executor.run, normalize apply) run.
    await asyncio.sleep(0)
    # The enrich step actually created + ran a job through the real executor path.
    assert len(job_store.created) == 1
    assert len(executor.ran) == 1
    statuses = [s["status"] for s in result["steps"]]
    assert statuses == ["ok", "ok"]


@pytest.mark.asyncio
async def test_execute_plan_unknown_id_errors():
    out = await asyncio.wait_for(execute_plan(_ctx(), "nope"), TIMEOUT)
    assert out["kind"] == "error"


# --------------------------------------------------------------------------- #
# 5. Route-level: confirm:{plan_id} runs the persisted plan
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_route_confirm_executes_plan(monkeypatch):
    """End-to-end through the HTTP route: handle → plan, then confirm → result."""
    import os

    os.environ.setdefault("OMNIX_API_KEYS", '{"test-key": "test-tenant"}')
    os.environ.setdefault("OMNIX_NEPTUNE_ENDPOINT", "http://fake:8182")
    # monkeypatch (NOT os.environ[...]=) so the key is reverted after the test
    # and never leaks into other tests' NLQueryPipeline construction.
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")

    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    from cograph_client.api.app import create_app
    from cograph_client.graph.client import NeptuneClient

    _stub_classifier(monkeypatch, "enrich")

    async def fake_sample(neptune, tenant_id, kg, type_name, pred_leaf):
        return (["English"], "relationship")  # atomic → single enrich step

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.sample_predicate_values",
        fake_sample,
    )

    app = create_app()
    n = AsyncMock(spec=NeptuneClient)
    n.query.return_value = {"head": {"vars": []}, "results": {"bindings": []}}
    n.update.return_value = None
    app.state.neptune_client = n
    client = TestClient(app)
    headers = {"X-API-Key": "test-key"}

    r1 = client.post(
        "/graphs/test-tenant/agent",
        json={
            "message": "enrich company for mentors who speak Persian",
            "context": {"kg_name": "kg1", "type_name": "Mentor"},
        },
        headers=headers,
    )
    assert r1.status_code == 200, r1.text
    body = r1.json()
    assert body["kind"] == "plan"
    plan_id = body["plan_id"]

    r2 = client.post(
        "/graphs/test-tenant/agent",
        json={
            "message": "",
            "context": {"kg_name": "kg1", "type_name": "Mentor"},
            "confirm": {"plan_id": plan_id},
        },
        headers=headers,
    )
    assert r2.status_code == 200, r2.text
    result = r2.json()
    assert result["kind"] == "result"
    assert all(s["status"] == "ok" for s in result["steps"])
