"""Tests for the ontology capability behind the unified Ask-AI agent (COG-126).

Everything is stubbed so the suite is deterministic and fast (no network, no
real Neptune): the classifier LLM, the ontology directive-extraction LLM, the
type-schema read, and the underlying Neptune update. Each async test is wrapped
in ``asyncio.wait_for`` to fail loudly rather than hang.

Covers the two operations the capability exposes:
  * inspect  → an ANSWER (the planner surfaces it directly, no confirm plan),
  * declare  → a well-formed PlanStep, whose ``execute`` drives the EXISTING
               ontology engine (the atomic upsert builders) via Neptune.
"""

from __future__ import annotations

import asyncio

import pytest

from cograph_client.agent import planner as planner_mod
from cograph_client.agent.capabilities import ontology_cap as ontology_mod
from cograph_client.agent.capabilities.ontology_cap import OntologyCapability
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
    reset_capabilities,
)

TIMEOUT = 5.0


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeNeptune:
    """Records updates; answers a types query for the no-type-in-scope path."""

    def __init__(self, type_rows: list[dict] | None = None):
        self.updates: list[str] = []
        self._type_rows = type_rows or []

    async def query(self, q):
        # Render the rows in SPARQL-results JSON shape so parse_sparql_results
        # (used by _list_types) returns them.
        return {
            "head": {"vars": ["type", "label", "comment", "parent"]},
            "results": {
                "bindings": [
                    {k: {"value": v} for k, v in row.items()}
                    for row in self._type_rows
                ]
            },
        }

    async def update(self, q):
        self.updates.append(q)
        return None


_MENTOR_SCHEMA = {
    "attributes": ["title", "skills"],
    "relationships": [{"name": "speaks", "target_type": "Language"}],
}


def _ctx(neptune=None, type_name="Mentor"):
    return AgentContext(
        tenant_id="t1",
        kg_name="kg1",
        neptune=neptune or FakeNeptune(),
        type_name=type_name,
        openrouter_key="fake-key",
        anthropic_key="fake-anthropic",
        extras={},
    )


@pytest.fixture(autouse=True)
def _fresh_registry():
    reset_capabilities()
    reset_plan_store()
    register_default_capabilities()
    yield
    reset_capabilities()
    reset_plan_store()


def _stub_classifier(monkeypatch, intent: str, clarify: str = ""):
    async def fake_chat(*args, **kwargs):
        import json

        return json.dumps({"intent": intent, "clarify": clarify})

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_chat)


def _stub_directive(monkeypatch, payload: dict):
    """Stub the ontology capability's extraction LLM to return ``payload``."""
    import json

    async def fake_chat(*args, **kwargs):
        return json.dumps(payload)

    monkeypatch.setattr(ontology_mod, "openrouter_chat", fake_chat)


def _stub_schema(monkeypatch, schema: dict | None = None):
    schema = schema if schema is not None else _MENTOR_SCHEMA

    async def fake_schema(neptune, tenant_id, type_name):
        return schema

    monkeypatch.setattr(ontology_mod, "list_type_schema", fake_schema)


# --------------------------------------------------------------------------- #
# 1. Registration — ontology is now a first-class capability
# --------------------------------------------------------------------------- #
def test_ontology_capability_registered():
    names = {c.name for c in get_capabilities()}
    assert "ontology" in names
    cap = get_capability("ontology")
    assert cap is not None
    assert "ontology" in cap.describe().lower() or "schema" in cap.describe().lower()


# --------------------------------------------------------------------------- #
# 2. Inspect → answer (read-only, no confirm plan)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_inspect_returns_answer_with_type_schema(monkeypatch):
    """'what attributes does this type have' → {kind:"answer"} carrying the
    type's declared attributes + relationships, NOT a plan."""
    _stub_classifier(monkeypatch, "ontology")
    _stub_schema(monkeypatch)
    _stub_directive(monkeypatch, {"op": "inspect", "type_name": "Mentor"})

    out = await asyncio.wait_for(
        handle(_ctx(), "what attributes does this type have?"), TIMEOUT
    )
    assert out["kind"] == "answer"
    onto = out["ontology"]
    assert onto["type"] == "Mentor"
    assert onto["attributes"] == ["title", "skills"]
    assert onto["relationships"] == [{"name": "speaks", "target_type": "Language"}]
    # The human-facing answer text names the attributes.
    assert "title" in out["answer"] and "speaks" in out["answer"]


@pytest.mark.asyncio
async def test_inspect_without_type_lists_tenant_types(monkeypatch):
    """With no type in scope, inspect lists the tenant's declared types."""
    _stub_classifier(monkeypatch, "ontology")
    _stub_schema(monkeypatch)  # unused on this path but harmless
    _stub_directive(monkeypatch, {"op": "inspect", "type_name": None})

    neptune = FakeNeptune(
        type_rows=[
            {"label": "Mentor", "comment": "a mentor"},
            {"label": "Company", "comment": ""},
        ]
    )
    out = await asyncio.wait_for(
        handle(_ctx(neptune=neptune, type_name=None), "list the types"), TIMEOUT
    )
    assert out["kind"] == "answer"
    names = [t["name"] for t in out["ontology"]["types"]]
    assert names == ["Mentor", "Company"]
    assert "Mentor" in out["answer"] and "Company" in out["answer"]


# --------------------------------------------------------------------------- #
# 3. Declare attribute → a well-formed PlanStep (mutation, awaits confirm)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_declare_attribute_produces_plan_step(monkeypatch):
    """'add a website attribute' → a declare_attribute PlanStep with free cost,
    grounded params, and a preview — NOT an answer."""
    _stub_classifier(monkeypatch, "ontology")
    _stub_schema(monkeypatch)
    _stub_directive(
        monkeypatch,
        {
            "op": "declare_attribute",
            "type_name": None,  # falls back to the active type
            "attribute": "website",
            "datatype": "string",
            "description": "the company website",
            "confidence": 0.9,
        },
    )

    out = await asyncio.wait_for(
        handle(_ctx(), "add a website attribute to mentors"), TIMEOUT
    )
    assert out["kind"] == "plan"
    assert out["plan_id"]
    steps = out["steps"]
    assert len(steps) == 1
    step = steps[0]
    assert step["capability"] == "ontology"
    assert step["action"] == "declare_attribute"
    assert step["params"]["type_name"] == "Mentor"  # active type
    assert step["params"]["attribute"] == "website"
    assert step["params"]["datatype"] == "string"
    # Ontology edits are free.
    assert step["cost"] == {"paid_calls": 0, "estimated_usd": 0.0}
    assert "website" in step["preview"]["summary"]


@pytest.mark.asyncio
async def test_declare_type_produces_plan_step(monkeypatch):
    """'create a Product type under Thing' → a declare_type PlanStep."""
    _stub_classifier(monkeypatch, "ontology")
    _stub_schema(monkeypatch)
    _stub_directive(
        monkeypatch,
        {
            "op": "declare_type",
            "type_name": "Product",
            "parent_type": "Thing",
            "description": "a sellable product",
            "confidence": 0.85,
        },
    )

    out = await asyncio.wait_for(
        handle(_ctx(), "create a Product type under Thing"), TIMEOUT
    )
    assert out["kind"] == "plan"
    step = out["steps"][0]
    assert step["action"] == "declare_type"
    assert step["params"]["type_name"] == "Product"
    assert step["params"]["parent_type"] == "Thing"
    assert step["cost"] == {"paid_calls": 0, "estimated_usd": 0.0}


@pytest.mark.asyncio
async def test_declare_attribute_underspecified_clarifies(monkeypatch):
    """An attribute declare with no resolvable attribute name → no step → the
    planner clarifies rather than declaring a junk predicate."""
    _stub_classifier(monkeypatch, "ontology")
    _stub_schema(monkeypatch)
    _stub_directive(
        monkeypatch,
        {"op": "declare_attribute", "type_name": "Mentor", "attribute": "the"},
    )
    out = await asyncio.wait_for(handle(_ctx(), "add a thing"), TIMEOUT)
    assert out["kind"] == "clarify"


# --------------------------------------------------------------------------- #
# 4. Execute → drives the EXISTING ontology engine (atomic upsert builders)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_execute_declare_attribute_calls_ontology_engine(monkeypatch):
    """Confirming a declare_attribute plan runs the upsert_attribute builder
    against the tenant graph (the SAME engine the /ontology endpoint uses)."""
    _stub_classifier(monkeypatch, "ontology")
    _stub_schema(monkeypatch)
    _stub_directive(
        monkeypatch,
        {
            "op": "declare_attribute",
            "type_name": "Mentor",
            "attribute": "website",
            "datatype": "string",
        },
    )
    neptune = FakeNeptune()
    ctx = _ctx(neptune=neptune)

    plan_out = await asyncio.wait_for(
        handle(ctx, "add a website attribute"), TIMEOUT
    )
    assert plan_out["kind"] == "plan"
    result = await asyncio.wait_for(
        execute_plan(ctx, plan_out["plan_id"]), TIMEOUT
    )
    assert result["kind"] == "result"
    assert all(s["status"] == "ok" for s in result["steps"])
    # The ontology engine actually ran one update, against the TENANT graph, for
    # the website attribute on Mentor.
    assert len(neptune.updates) == 1
    update = neptune.updates[0]
    assert "graphs/t1" in update  # tenant ontology graph
    assert "Mentor" in update
    assert "website" in update


@pytest.mark.asyncio
async def test_execute_declare_type_calls_ontology_engine(monkeypatch):
    """Confirming a declare_type plan runs the insert_type builder."""
    cap = OntologyCapability()
    neptune = FakeNeptune()
    ctx = _ctx(neptune=neptune)
    step = PlanStep(
        capability="ontology",
        action="declare_type",
        params={"type_name": "Product", "description": "", "parent_type": "Thing"},
    )
    ack = await asyncio.wait_for(cap.execute(ctx, step), TIMEOUT)
    assert ack["kind"] == "ack"
    assert ack["type_name"] == "Product"
    assert len(neptune.updates) == 1
    update = neptune.updates[0]
    assert "Product" in update
    assert "Thing" in update  # parent (subClassOf)


# --------------------------------------------------------------------------- #
# 5. Fallback heuristic (no LLM key) still inspects + declares
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_inspect_via_heuristic_without_llm_key(monkeypatch):
    """When the LLM is unavailable, an obvious inspect ('describe the schema')
    is still recognized deterministically → an answer."""
    _stub_classifier(monkeypatch, "ontology")
    _stub_schema(monkeypatch)

    async def boom(*args, **kwargs):
        raise RuntimeError("no llm")

    monkeypatch.setattr(ontology_mod, "openrouter_chat", boom)

    out = await asyncio.wait_for(
        handle(_ctx(), "describe the schema for this type"), TIMEOUT
    )
    assert out["kind"] == "answer"
    assert out["ontology"]["type"] == "Mentor"


def test_heuristic_directive_recognizes_ops():
    """Unit-level: the deterministic fallback classifies the common shapes."""
    assert ontology_mod._heuristic_directive("describe the schema")["op"] == "inspect"
    assert (
        ontology_mod._heuristic_directive("what attributes does Mentor have")["op"]
        == "inspect"
    )
    d = ontology_mod._heuristic_directive("add a website attribute")
    assert d["op"] == "declare_attribute"
    assert d["attribute"] == "website"
    # A genuinely vague instruction yields no op → planner clarifies.
    assert ontology_mod._heuristic_directive("do something")["op"] is None
