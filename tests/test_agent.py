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
from cograph_client.agent.conversation_store import reset_conversation_store
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
        self.updated = []

    async def create(self, job):
        self.created.append(job)

    async def get(self, job_id):
        for j in self.created:
            if j.id == job_id:
                return j
        return None

    async def update(self, job):
        self.updated.append(job)


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
    reset_conversation_store()
    register_default_capabilities()
    yield
    reset_capabilities()
    reset_plan_store()
    reset_conversation_store()


@pytest.fixture(autouse=True)
def _track_bg_tasks(monkeypatch):
    """Schedule capability-spawned background coroutines as TRACKED tasks.

    Capabilities use the strong-ref ``_spawn`` (create_task) for long work
    (normalize apply / enrichment run). In tests we still want that work to run
    (so we can assert the executor/store was actually driven), just tracked so
    nothing leaks. We replace ``_spawn`` with one that creates a real task and
    keeps a strong ref — the underlying apply/run is itself stubbed per-test.
    """
    import cograph_client.agent.capabilities.dedup_cap as dedup_cap
    import cograph_client.agent.capabilities.enrich_cap as enrich_cap
    import cograph_client.agent.capabilities.normalize_cap as norm_cap

    spawned: list = []

    def tracking_spawn(coro):
        task = asyncio.ensure_future(coro)
        spawned.append(task)
        task.add_done_callback(lambda t: None)

    monkeypatch.setattr(norm_cap, "_spawn", tracking_spawn)
    monkeypatch.setattr(enrich_cap, "_spawn", tracking_spawn)
    monkeypatch.setattr(dedup_cap, "_spawn", tracking_spawn)
    return spawned


def _stub_classifier(monkeypatch, intent: str, clarify: str = ""):
    async def fake_chat(*args, **kwargs):
        import json

        return json.dumps({"intent": intent, "clarify": clarify})

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_chat)


# The Mentor type's schema the agent grounds extraction in: ``company`` is
# ABSENT (so it must be proposed as a new attribute), ``title`` + ``skills`` are
# present attributes, and ``speaks`` is a relationship to a Language type.
_MENTOR_SCHEMA = {
    "attributes": ["title", "skills"],
    "relationships": [{"name": "speaks", "target_type": "Language"}],
}


def _stub_schema(monkeypatch, schema: dict | None = None):
    """Stub ``list_type_schema`` in BOTH capabilities to the Mentor schema."""
    schema = schema if schema is not None else _MENTOR_SCHEMA

    async def fake_schema(neptune, tenant_id, type_name):
        return schema

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.list_type_schema", fake_schema
    )
    monkeypatch.setattr(
        "cograph_client.agent.capabilities.normalize_cap.list_type_schema", fake_schema
    )


def _stub_enrich_extract(monkeypatch, payload: dict):
    """Stub the enrich capability's extraction LLM to return ``payload`` JSON."""
    import json

    async def fake_chat(*args, **kwargs):
        return json.dumps(payload)

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.openrouter_chat", fake_chat
    )


def _stub_normalize_extract(monkeypatch, payload: dict):
    """Stub the normalize capability's directive LLM to return ``payload`` JSON."""
    import json

    async def fake_chat(*args, **kwargs):
        return json.dumps(payload)

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.normalize_cap.openrouter_chat", fake_chat
    )


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
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch,
        {"attributes": ["company"], "scope": None, "tier": "core"},
    )
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


def _stub_kg_types(monkeypatch, names: list[str]):
    """Stub the enrich capability's KG type listing to ``names``."""

    async def fake_list_types(ctx):
        return list(names)

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap._list_types", fake_list_types
    )


@pytest.mark.asyncio
async def test_enrich_infers_type_from_message_over_selection(monkeypatch):
    """The type NAMED in the message wins over the UI selection: "enrich brokers
    with their websites" targets Broker even though ctx.type_name is Mentor (the
    selected-but-wrong type). Regression for the enrich-uses-selection bug."""
    _stub_classifier(monkeypatch, "enrich")
    _stub_kg_types(monkeypatch, ["Broker", "PropertyListing", "Mentor"])

    captured: dict = {}

    async def fake_schema(neptune, tenant_id, type_name):
        captured["schema_type"] = type_name
        return {"attributes": ["website"], "relationships": []}

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.list_type_schema", fake_schema
    )
    _stub_enrich_extract(
        monkeypatch, {"attributes": ["website"], "scope": None, "tier": "core"}
    )

    out = await asyncio.wait_for(
        handle(_ctx(), "enrich brokers with their websites"), TIMEOUT
    )
    assert out["kind"] == "plan"
    step = out["steps"][0]
    assert step["params"]["type_name"] == "Broker"  # message won, not Mentor
    assert captured["schema_type"] == "Broker"  # grounded in Broker's schema
    assert step["params"]["attributes"] == ["website"]


@pytest.mark.asyncio
async def test_enrich_infers_type_with_no_selection(monkeypatch):
    """With NO type selected (ctx.type_name is None), a message that names the
    type still plans (resolves Broker) instead of bailing to clarify."""
    _stub_classifier(monkeypatch, "enrich")
    _stub_kg_types(monkeypatch, ["Broker", "PropertyListing"])

    async def fake_schema(neptune, tenant_id, type_name):
        return {"attributes": ["website"], "relationships": []}

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.list_type_schema", fake_schema
    )
    _stub_enrich_extract(
        monkeypatch, {"attributes": ["website"], "scope": None, "tier": "core"}
    )

    ctx = AgentContext(
        tenant_id="t1",
        kg_name="kg1",
        neptune=FakeNeptune(),
        type_name=None,  # nothing selected in the UI
        openrouter_key="fake-key",
        extras={
            "enrichment_executor": FakeExecutor(),
            "enrichment_job_store": FakeJobStore(),
        },
    )
    out = await asyncio.wait_for(
        handle(ctx, "look up the websites for the top 5 brokers"), TIMEOUT
    )
    assert out["kind"] == "plan"
    assert out["steps"][0]["params"]["type_name"] == "Broker"


@pytest.mark.asyncio
async def test_ambiguous_routes_to_clarify(monkeypatch):
    _stub_classifier(monkeypatch, "ambiguous", clarify="Which field?")
    out = await asyncio.wait_for(handle(_ctx(), "do the thing"), TIMEOUT)
    assert out["kind"] == "clarify"
    assert out["question"] == "Which field?"


@pytest.mark.asyncio
async def test_unregistered_intent_clarifies(monkeypatch):
    # 'ontology' is recognized by the classifier but no capability is registered
    # yet (A2) → clarify. (dedup IS registered now; see the dedup tests below.)
    _stub_classifier(monkeypatch, "ontology")
    out = await asyncio.wait_for(handle(_ctx(), "rename the type"), TIMEOUT)
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
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch,
        {
            "attributes": ["company"],
            "scope": {"predicate": "speaks", "value": "Persian"},
            "tier": "core",
        },
    )

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
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch,
        {
            "attributes": ["company"],
            "scope": {"predicate": "speaks", "value": "Persian"},
            "tier": "core",
        },
    )

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
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch,
        {
            "attributes": ["company"],
            "scope": {"predicate": "speaks", "value": "Persian"},
            "tier": "core",
        },
    )

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
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch,
        {
            "attributes": ["company"],
            "scope": {"predicate": "speaks", "value": "Persian"},
            "tier": "core",
        },
    )

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


# --------------------------------------------------------------------------- #
# 6. Schema-grounded plan() extraction (COG-119)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_enrich_extracts_real_attr_and_web_tier(monkeypatch):
    """'enrich the current company for mentors who speak Persian':
    - attribute is the real noun 'company' (NOT the modifier 'current'),
    - tier is 'core' (company is an open-web fact Wikidata lacks),
    - scope is the validated 'speaks' relationship = Persian.
    company is ABSENT from the schema, so it is proposed as a new attribute."""
    _stub_classifier(monkeypatch, "enrich")
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch,
        {
            "attributes": ["company"],
            "scope": {"predicate": "speaks", "value": "Persian"},
            "tier": "core",
            "confidence_min": 0.85,
        },
    )

    # speaks samples are atomic → no clean-before-enrich prerequisite.
    async def fake_sample(neptune, tenant_id, kg, type_name, pred_leaf):
        return (["English", "Persian"], "relationship")

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.sample_predicate_values",
        fake_sample,
    )

    out = await asyncio.wait_for(
        handle(_ctx(), "enrich the current company for mentors who speak Persian"),
        TIMEOUT,
    )
    assert out["kind"] == "plan"
    steps = out["steps"]
    assert len(steps) == 1
    enrich_step = steps[0]
    assert enrich_step["capability"] == "enrich"
    assert enrich_step["params"]["attributes"] == ["company"]
    assert enrich_step["params"]["tier"] == "core"
    assert enrich_step["params"]["scope"] == {
        "predicate": "speaks",
        "value": "Persian",
    }


@pytest.mark.asyncio
async def test_enrich_drops_stray_modifier_word_on_fallback(monkeypatch):
    """When the LLM extraction is unavailable, the deterministic fallback still
    yields a real attribute ('company') from 'the current company' — never the
    stray modifier 'current' — and defaults the tier to the paid web 'core'."""
    _stub_classifier(monkeypatch, "enrich")
    _stub_schema(monkeypatch)

    async def boom(*args, **kwargs):
        raise RuntimeError("no llm")

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.openrouter_chat", boom
    )

    out = await asyncio.wait_for(
        handle(_ctx(), "enrich the current company"), TIMEOUT
    )
    assert out["kind"] == "plan"
    step = out["steps"][0]
    assert step["params"]["attributes"] == ["company"]
    assert step["params"]["tier"] == "core"  # web-fact backstop


@pytest.mark.asyncio
async def test_normalize_strip_emoji_on_title_is_a_plan(monkeypatch):
    """'remove emojis from the title field' → a strip_emoji PLAN on 'title',
    NOT a clarify (the old behavior, because the live sample had no emoji)."""
    _stub_classifier(monkeypatch, "clean")
    _stub_schema(monkeypatch)
    _stub_normalize_extract(
        monkeypatch,
        {
            "rule_type": "strip_emoji",
            "predicate": "title",
            "params": {},
            "confidence": 0.9,
            "rationale": "user asked to remove emoji from title",
        },
    )

    # sample_predicate_values powers the dry-run preview for the built rule.
    async def fake_sample(neptune, tenant_id, kg, type_name, pred_leaf):
        assert pred_leaf == "title"
        return (["🚀 Founder", "CTO"], "attribute")

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.normalize_cap.sample_predicate_values",
        fake_sample,
    )

    out = await asyncio.wait_for(
        handle(_ctx(), "remove emojis from the title field"), TIMEOUT
    )
    assert out["kind"] == "plan", out
    step = out["steps"][0]
    assert step["capability"] == "normalize"
    rule = step["params"]["rule"]
    assert rule["rule_type"] == "strip_emoji"
    assert rule["predicate"] == "title"
    assert step["preview"]["rule_type"] == "strip_emoji"


@pytest.mark.asyncio
async def test_normalize_list_explode_maps_languages_to_speaks(monkeypatch):
    """'split the languages into separate ones' → list_explode on the 'speaks'
    relationship (the NL phrase 'languages' maps onto the real predicate)."""
    _stub_classifier(monkeypatch, "clean")
    _stub_schema(monkeypatch)
    _stub_normalize_extract(
        monkeypatch,
        {
            "rule_type": "list_explode",
            "predicate": "speaks",
            "params": {},
            "confidence": 0.92,
            "rationale": "languages packed together",
        },
    )

    async def fake_sample(neptune, tenant_id, kg, type_name, pred_leaf):
        assert pred_leaf == "speaks"
        return (["English__Persian", "French"], "relationship")

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.normalize_cap.sample_predicate_values",
        fake_sample,
    )

    out = await asyncio.wait_for(
        handle(_ctx(), "split the languages into separate ones"), TIMEOUT
    )
    assert out["kind"] == "plan", out
    rule = out["steps"][0]["params"]["rule"]
    assert rule["rule_type"] == "list_explode"
    assert rule["predicate"] == "speaks"
    assert rule["target_kind"] == "relationship"
    assert rule["params"]["target"] == "entity"


@pytest.mark.asyncio
async def test_normalize_vague_message_still_clarifies(monkeypatch):
    """A genuinely vague instruction ('fix this data') → clarify: no field can
    be identified, the LLM returns no rule_type, and no predicate is spotted."""
    _stub_classifier(monkeypatch, "clean")
    _stub_schema(monkeypatch)
    _stub_normalize_extract(
        monkeypatch,
        {"rule_type": None, "predicate": None, "params": {}, "confidence": 0.0},
    )

    out = await asyncio.wait_for(handle(_ctx(), "fix this data"), TIMEOUT)
    assert out["kind"] == "clarify"


# --------------------------------------------------------------------------- #
# 7. COG-123 cost estimate + COG-121 confidence — the agent's plan is honest
# --------------------------------------------------------------------------- #
from cograph_client.enrichment.models import EnrichmentTier  # noqa: E402
from cograph_client.enrichment.sources.base import (  # noqa: E402
    _adapters,
    register_adapter,
)
from cograph_client.enrichment.tiers import register_tier, reset_tiers  # noqa: E402


class _CountingExecutor:
    """A FakeExecutor that also answers count_entities (the matched-count path
    the plan reuses for its cost estimate, COG-123)."""

    def __init__(self, count: int = 0, raises: bool = False):
        self.ran = []
        self._count = count
        self._raises = raises
        self.count_calls = []

    async def run(self, job, tenant_id):
        self.ran.append((job, tenant_id))

    async def count_entities(self, tenant_id, kg_name, type_name, scope=None,
                             entity_uris=None):
        self.count_calls.append((type_name, scope))
        if self._raises:
            raise RuntimeError("neptune down")
        return self._count


class _MockPaidAdapter:
    """A generic PAID adapter declaring cost via the protocol's metadata — stands
    in for a proprietary web adapter (Exa/Parallel) WITHOUT importing one. The
    cost model must derive everything from these declared attributes, never the
    name (COG-123 boundary)."""

    name = "mock_paid_web"
    is_paid = True
    cost_per_call = 0.01

    async def lookup(self, entity_label, attribute, context):
        return []


class _MockFreeAdapter:
    name = "mock_free"
    is_paid = False
    cost_per_call = 0.0

    async def lookup(self, entity_label, attribute, context):
        return []


@pytest.fixture
def _adapters_and_tiers():
    """Register mock adapters + wire tiers, restoring global state afterwards so
    the registry/tier mutations never leak into other tests."""
    saved = dict(_adapters)
    register_adapter(_MockPaidAdapter())
    register_adapter(_MockFreeAdapter())
    # core => a paid/web chain; lite => an all-free chain.
    register_tier(EnrichmentTier.core, ["cache", "mock_paid_web"])
    register_tier(EnrichmentTier.lite, ["cache", "mock_free"])
    yield
    _adapters.clear()
    _adapters.update(saved)
    reset_tiers()


@pytest.mark.asyncio
async def test_paid_tier_cost_scales_with_matched_count(monkeypatch, _adapters_and_tiers):
    """COG-123: a paid chain yields a non-zero cost that scales with the matched
    count, the plan proposes a bounded limit, and the preview no longer says
    'no paid calls'."""
    _stub_classifier(monkeypatch, "enrich")
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch,
        {"attributes": ["company"], "scope": None, "tier": "core"},
    )
    executor = _CountingExecutor(count=50)
    ctx = _ctx(executor=executor)

    out = await asyncio.wait_for(handle(ctx, "enrich company"), TIMEOUT)
    assert out["kind"] == "plan"
    step = out["steps"][0]
    cost = step["cost"]
    # 50 matched × $0.01/entity = $0.50, paid_calls = 50 (under the 200 cap).
    assert cost["paid_calls"] == 50
    assert cost["estimated_usd"] == 0.5
    assert cost["per_entity_cost_usd"] == 0.01
    assert cost["paid_calls_estimated"] is False  # exact count, not an upper bound
    assert "no paid calls" not in cost["note"].lower()
    # A bounded limit was proposed + surfaced.
    assert step["params"]["limit"] == 200
    assert step["preview"]["limit"] == 200
    # The matched-count COUNT was reused (not a new query engine).
    assert executor.count_calls == [("Mentor", None)]


@pytest.mark.asyncio
async def test_paid_cost_capped_by_limit(monkeypatch, _adapters_and_tiers):
    """COG-123: cost is bounded by the proposed limit — matched 5000 but the
    plan caps paid calls at the default limit (200)."""
    _stub_classifier(monkeypatch, "enrich")
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch,
        {"attributes": ["company"], "scope": None, "tier": "core"},
    )
    ctx = _ctx(executor=_CountingExecutor(count=5000))
    out = await asyncio.wait_for(handle(ctx, "enrich company"), TIMEOUT)
    cost = out["steps"][0]["cost"]
    assert cost["paid_calls"] == 200  # min(5000, limit)
    assert cost["estimated_usd"] == 2.0  # 200 × $0.01


@pytest.mark.asyncio
async def test_free_tier_cost_is_zero(monkeypatch, _adapters_and_tiers):
    """COG-123: an all-free chain (Wikidata-style) costs nothing, even with a
    large matched count."""
    _stub_classifier(monkeypatch, "enrich")
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch,
        # 'lite' resolves to the all-free chain; confidence stays at the strict
        # default for a free/structured source (not a web override).
        {"attributes": ["country"], "scope": None, "tier": "lite"},
    )
    ctx = _ctx(executor=_CountingExecutor(count=1000))
    out = await asyncio.wait_for(handle(ctx, "look up the country code"), TIMEOUT)
    cost = out["steps"][0]["cost"]
    assert cost["paid_calls"] == 0
    assert cost["estimated_usd"] == 0.0
    # A free tier keeps the strict default confidence (no web override).
    assert out["steps"][0]["params"]["confidence_min"] == 0.85


@pytest.mark.asyncio
async def test_cost_falls_back_to_limit_when_count_unavailable(
    monkeypatch, _adapters_and_tiers
):
    """COG-123: when the matched COUNT can't be computed (executor raises), the
    paid cost is reported as a clearly-labeled UPPER BOUND (the cap), never a
    silent 0 for a paid tier."""
    _stub_classifier(monkeypatch, "enrich")
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch,
        {"attributes": ["company"], "scope": None, "tier": "core"},
    )
    ctx = _ctx(executor=_CountingExecutor(raises=True))
    out = await asyncio.wait_for(handle(ctx, "enrich company"), TIMEOUT)
    cost = out["steps"][0]["cost"]
    assert cost["paid_calls"] == 200          # bounded by the proposed cap
    assert cost["paid_calls_estimated"] is True  # flagged as upper-bound
    assert "up to" in cost["note"].lower()
    assert cost["estimated_usd"] == 2.0


@pytest.mark.asyncio
async def test_paid_cost_scales_with_attribute_count(monkeypatch, _adapters_and_tiers):
    """COG-123 (review fix): the executor calls the adapter chain once per
    (entity, attribute) pair, so a multi-attribute enrich must scale paid_calls
    AND the dollar estimate by len(attributes). Quoting only by entities
    under-counts by n_attributes×. This test FAILS against the entity-only
    estimate."""
    _stub_classifier(monkeypatch, "enrich")
    _stub_schema(monkeypatch)
    # TWO paid attributes on a paid (core) chain.
    _stub_enrich_extract(
        monkeypatch,
        {"attributes": ["company", "website"], "scope": None, "tier": "core"},
    )
    executor = _CountingExecutor(count=50)
    ctx = _ctx(executor=executor)

    out = await asyncio.wait_for(handle(ctx, "enrich company and website"), TIMEOUT)
    assert out["kind"] == "plan"
    cost = out["steps"][0]["cost"]
    # 50 entities × 2 attributes = 100 paid calls; 100 × $0.01 = $1.00.
    assert cost["paid_calls"] == 100
    assert cost["estimated_usd"] == 1.0
    assert cost["attributes"] == 2
    # The note states the entities × attributes = calls basis.
    note = cost["note"].lower()
    assert "2 attributes" in note
    assert "100 paid lookups" in note


def test_estimate_cost_keys_match_web_contract():
    """MAJOR review fix: the emitted cost dict MUST use the EXACT keys the web
    plan-step cost contract reads — ``estimated_usd`` and ``paid_calls`` (see
    web/app/components/explore/useAgentChat.ts ``AgentStepCost`` and
    AgentChat.tsx ``PlanStepRow``). Asserting on the literal keys here pins the
    contract so a future rename can't silently blank the cost badge again. Covers
    both the free and paid branches of _estimate_cost."""
    from cograph_client.agent.capabilities.enrich_cap import _estimate_cost

    # Free branch (no paid adapter).
    free = _estimate_cost(
        tier=EnrichmentTier.lite,
        per_entity_cost=0.0,
        paid_adapters=0,
        has_paid=False,
        matched=10,
        matched_exact=True,
        limit=200,
        n_attributes=1,
    )
    assert "estimated_usd" in free
    assert "paid_calls" in free
    assert "estimated_cost_usd" not in free  # the old (wrong) key is gone

    # Paid branch.
    paid = _estimate_cost(
        tier=EnrichmentTier.core,
        per_entity_cost=0.01,
        paid_adapters=1,
        has_paid=True,
        matched=10,
        matched_exact=True,
        limit=200,
        n_attributes=2,
    )
    assert "estimated_usd" in paid
    assert "paid_calls" in paid
    assert "estimated_cost_usd" not in paid
    # The web UI only reads these two; confirm both are populated/typed.
    assert isinstance(paid["paid_calls"], int)
    assert isinstance(paid["estimated_usd"], float)


@pytest.mark.asyncio
async def test_web_tier_lowers_confidence_min(monkeypatch, _adapters_and_tiers):
    """COG-121: a web-sourced (paid-chain) enrich lowers confidence_min from the
    strict 0.85 default to a functional floor, surfaced in the preview, so web
    verdicts aren't all silently filtered → 0 writes."""
    _stub_classifier(monkeypatch, "enrich")
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch,
        # No explicit confidence → the default; the plan should override it.
        {"attributes": ["company"], "scope": None, "tier": "core"},
    )
    ctx = _ctx(executor=_CountingExecutor(count=10))
    out = await asyncio.wait_for(handle(ctx, "enrich company"), TIMEOUT)
    step = out["steps"][0]
    assert step["params"]["confidence_min"] == 0.4  # functional web floor
    assert step["preview"]["confidence_min"] == 0.4
    assert "confidence_min lowered" in step["preview"]["confidence_note"]


@pytest.mark.asyncio
async def test_user_confidence_not_overridden_on_web_tier(
    monkeypatch, _adapters_and_tiers
):
    """COG-121: an EXPLICIT user confidence is respected even on a web tier — we
    only override the unset (default 0.85) value."""
    _stub_classifier(monkeypatch, "enrich")
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch,
        {
            "attributes": ["company"],
            "scope": None,
            "tier": "core",
            "confidence_min": 0.9,  # user asked for stricter
        },
    )
    ctx = _ctx(executor=_CountingExecutor(count=10))
    out = await asyncio.wait_for(handle(ctx, "enrich company strictly"), TIMEOUT)
    step = out["steps"][0]
    assert step["params"]["confidence_min"] == 0.9  # respected, not lowered


@pytest.mark.asyncio
async def test_plan_limit_carried_into_enrich_job(monkeypatch, _adapters_and_tiers):
    """COG-123: the proposed limit is honored at execute time — the EnrichJob the
    capability builds carries the cap."""
    _stub_classifier(monkeypatch, "enrich")
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch,
        {"attributes": ["company"], "scope": None, "tier": "core"},
    )
    job_store = FakeJobStore()
    executor = _CountingExecutor(count=10)
    ctx = _ctx(executor=executor, job_store=job_store)

    plan_out = await asyncio.wait_for(handle(ctx, "enrich company"), TIMEOUT)
    result = await asyncio.wait_for(
        execute_plan(ctx, plan_out["plan_id"]), TIMEOUT
    )
    assert result["kind"] == "result"
    await asyncio.sleep(0)
    assert len(job_store.created) == 1
    job = job_store.created[0]
    assert job.limit == 200
    assert abs(job.confidence_min - 0.4) < 1e-9  # web floor carried through


# --------------------------------------------------------------------------- #
# 8. Dedup capability (COG-122) — registered → plans + drives the ER engine
# --------------------------------------------------------------------------- #
from cograph_client.agent.capabilities.dedup_cap import DedupCapability  # noqa: E402
from cograph_client.enrichment.models import JobCategory, JobStatus  # noqa: E402


def test_dedup_capability_is_registered_by_default():
    """register_default_capabilities() appends DedupCapability → the single
    endpoint can dispatch 'dedup' with no route change."""
    names = {c.name for c in get_capabilities()}
    assert "dedup" in names
    cap = get_capability("dedup")
    assert isinstance(cap, DedupCapability)


class _TypedNeptune(FakeNeptune):
    """Neptune fake whose query() returns the given rdf:type URIs so the dedup
    plan can enumerate ER-enabled types for its preview."""

    def __init__(self, type_uris: list[str]):
        super().__init__()
        self._type_uris = type_uris

    async def query(self, q):
        return {
            "head": {"vars": ["t"]},
            "results": {"bindings": [{"t": {"value": u}} for u in self._type_uris]},
        }


@pytest.mark.asyncio
async def test_dedup_routes_to_plan_with_er_types(monkeypatch):
    """'merge the duplicates' → a dedup PLAN (not clarify), grounded in the KG's
    real ER-enabled types. 'Person' resolves to an ERConfig (kept); 'Skill' does
    not (filtered out)."""
    _stub_classifier(monkeypatch, "dedup")
    prefix = "https://cograph.tech/types/"
    neptune = _TypedNeptune([f"{prefix}Person", f"{prefix}Skill"])

    out = await asyncio.wait_for(
        handle(_ctx(neptune=neptune), "merge the duplicate people"), TIMEOUT
    )
    assert out["kind"] == "plan"
    assert len(out["steps"]) == 1
    step = out["steps"][0]
    assert step["capability"] == "dedup"
    assert step["action"] == "run_dedup"
    assert step["params"]["kg_name"] == "kg1"
    # Only ER-enabled types are previewed; 'Skill' has no ERConfig → dropped.
    assert step["params"]["er_types"] == ["Person"]
    # Dedup is compute, not paid web calls.
    assert step["cost"]["paid_calls"] == 0
    assert step["cost"]["estimated_usd"] == 0.0


@pytest.mark.asyncio
async def test_dedup_plan_degrades_when_type_enum_fails(monkeypatch):
    """If type enumeration raises (Neptune down), the plan still proposes a dedup
    step with an empty er_types list rather than failing."""
    _stub_classifier(monkeypatch, "dedup")

    class _BoomNeptune(FakeNeptune):
        async def query(self, q):
            raise RuntimeError("neptune down")

    out = await asyncio.wait_for(
        handle(_ctx(neptune=_BoomNeptune()), "find and merge duplicates"), TIMEOUT
    )
    assert out["kind"] == "plan"
    step = out["steps"][0]
    assert step["capability"] == "dedup"
    assert step["params"]["er_types"] == []
    assert "all ER-enabled types" in step["preview"]["summary"]


@pytest.mark.asyncio
async def test_dedup_execute_drives_rebuild_engine(monkeypatch):
    """execute() creates a dedupe-category job and drives the EXISTING ER engine
    (rebuild_kg) as a tracked background worker, then records the merge volume."""
    captured: dict = {}

    async def fake_rebuild_kg(client, instance_graph):
        captured["instance_graph"] = instance_graph
        return {
            "types": [{"type": "Person", "fragments_absorbed": 7}],
            "fragments_absorbed_total": 7,
        }

    # Patch the engine entry point + the recompute hook the worker imports
    # lazily from the route module.
    monkeypatch.setattr(
        "cograph_client.resolver.er.rebuild.rebuild_kg", fake_rebuild_kg
    )

    recompute_calls: list = []
    monkeypatch.setattr(
        "cograph_client.api.routes.explore.schedule_recompute",
        lambda client, tenant_id, kg_name: recompute_calls.append((tenant_id, kg_name)),
    )

    job_store = FakeJobStore()
    ctx = _ctx(job_store=job_store)
    cap = get_capability("dedup")

    plan = await asyncio.wait_for(cap.plan(ctx, "merge duplicates"), TIMEOUT)
    ack = await asyncio.wait_for(cap.execute(ctx, plan[0]), TIMEOUT)

    assert ack["kind"] == "ack"
    assert ack["capability"] == "dedup"
    assert ack["job_id"]
    # A dedupe-category job was created in the queued state.
    assert len(job_store.created) == 1
    job = job_store.created[0]
    assert job.category == JobCategory.dedupe
    assert job.type_name == ""  # KG-wide, not type-scoped

    # Let the spawned background rebuild worker run.
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    # The engine ran against the KG's instance graph (the same primitive the
    # er-rebuild route uses), the job landed in 'applied' with the merge volume
    # recorded, and a type-stats recompute was scheduled.
    assert captured["instance_graph"] == "https://cograph.tech/graphs/t1/kg/kg1"
    assert job.status == JobStatus.applied
    assert job.progress.processed == 7
    assert "merged 7 duplicate fragment" in (job.error or "")
    assert recompute_calls == [("t1", "kg1")]


@pytest.mark.asyncio
async def test_dedup_execute_records_failure(monkeypatch):
    """If the rebuild engine raises, the worker records a failed job (detached —
    the error is captured on the job, never propagated)."""

    async def boom_rebuild(client, instance_graph):
        raise RuntimeError("merge blew up")

    monkeypatch.setattr(
        "cograph_client.resolver.er.rebuild.rebuild_kg", boom_rebuild
    )
    monkeypatch.setattr(
        "cograph_client.api.routes.explore.schedule_recompute",
        lambda *a, **k: None,
    )

    job_store = FakeJobStore()
    ctx = _ctx(job_store=job_store)
    cap = get_capability("dedup")
    plan = await asyncio.wait_for(cap.plan(ctx, "dedupe"), TIMEOUT)
    await asyncio.wait_for(cap.execute(ctx, plan[0]), TIMEOUT)

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    job = job_store.created[0]
    assert job.status == JobStatus.failed
    assert "dedup failed" in (job.error or "")


@pytest.mark.asyncio
async def test_dedup_execute_requires_job_store():
    """execute() raises a clear error when the job store isn't in the context."""
    ctx = AgentContext(
        tenant_id="t1", kg_name="kg1", neptune=FakeNeptune(), type_name="Person",
        extras={},
    )
    cap = get_capability("dedup")
    step = PlanStep(capability="dedup", action="run_dedup", params={"kg_name": "kg1"})
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(cap.execute(ctx, step), TIMEOUT)


@pytest.mark.asyncio
async def test_dedup_execute_via_plan_store(monkeypatch):
    """End-to-end through the planner: handle → dedup plan, confirm → result with
    an 'ok' step that drove the engine (the only mutating path)."""
    _stub_classifier(monkeypatch, "dedup")

    async def fake_rebuild_kg(client, instance_graph):
        return {"types": [], "fragments_absorbed_total": 0}

    monkeypatch.setattr(
        "cograph_client.resolver.er.rebuild.rebuild_kg", fake_rebuild_kg
    )
    monkeypatch.setattr(
        "cograph_client.api.routes.explore.schedule_recompute",
        lambda *a, **k: None,
    )

    job_store = FakeJobStore()
    ctx = _ctx(job_store=job_store)
    plan_out = await asyncio.wait_for(handle(ctx, "merge duplicates"), TIMEOUT)
    assert plan_out["kind"] == "plan"
    result = await asyncio.wait_for(execute_plan(ctx, plan_out["plan_id"]), TIMEOUT)
    assert result["kind"] == "result"
    assert [s["status"] for s in result["steps"]] == ["ok"]
    assert result["steps"][0]["capability"] == "dedup"
    await asyncio.sleep(0)
    assert len(job_store.created) == 1
    assert job_store.created[0].category == JobCategory.dedupe
