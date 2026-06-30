"""Unit tests for the web-discovery capability (web_ingest).

No network and no LLM: the web-source provider is a fake returning canned rows,
and the entity/attribute spec is injected via plan()'s ``parsed`` hook (so the
LLM resolver never runs). These exercise the full rail — graceful degradation,
the attribute-confirmation clarify, the confirmed-attributes plan (which now
previews the DISCOVERED multi-type ontology shape from the sample), and
execute → SchemaResolver.ingest (the same multi-type extract→resolve→insert path
document ingest commits through).
"""

from __future__ import annotations

import asyncio
import json

import pytest
from unittest.mock import MagicMock

from cograph_client.agent.capabilities import web_ingest_cap
from cograph_client.agent.capabilities.web_ingest_cap import WebIngestCapability
from cograph_client.agent.registry import AgentContext
from cograph_client.resolver.models import (
    ExtractedAttribute,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult,
    IngestResult,
)
from cograph_client.resolver.schema_resolver import SchemaResolver
from cograph_client.web_sources import (
    DiscoverResult,
    register_web_source,
    reset_web_sources,
)

FULL_ROWS = [
    {"name": "anthropic/claude-opus-4-8", "context_length": "200000"},
    {"name": "openai/gpt-5", "context_length": "400000"},
    {"name": "google/gemini-2.5-flash", "context_length": "1000000"},
    {"name": "meta/llama-4", "context_length": "128000"},
]

# Spec as the LLM resolver would return it (already normalized). ``query`` is the
# CLEAN search subject the resolver distills from the raw message.
CONFIRMED_SPEC = {
    "entity_type": "OpenRouterModel",
    "key_attribute": "name",
    "query": "OpenRouter models",
    "confirmed_attributes": ["context_length"],
    "suggested_attributes": ["provider", "context_length"],
}
ENTITY_ONLY_SPEC = {
    "entity_type": "OpenRouterModel",
    "key_attribute": "name",
    "query": "OpenRouter models",
    "confirmed_attributes": [],
    "suggested_attributes": ["provider", "context_length", "pricing"],
}


class FakeProvider:
    """Canned provider that honors hint_columns (projects rows to them)."""

    def __init__(self, *, is_paid: bool = False, cost_per_call: float = 0.0, rows=None) -> None:
        self.name = "fake"
        self.is_paid = is_paid
        self.cost_per_call = cost_per_call
        self._rows = FULL_ROWS if rows is None else rows
        self.calls: list[tuple] = []

    async def discover(self, query, *, sample, max_rows, hint_columns, context, urls=None):
        self.calls.append((query, sample, max_rows, tuple(hint_columns or ())))
        rows = self._rows[: (5 if sample else max_rows)]
        if hint_columns:
            rows = [{c: r.get(c, "unknown") for c in hint_columns} for r in rows]
        return DiscoverResult(
            rows=rows,
            sources=["https://openrouter.ai/models"],
            estimated_total=len(self._rows),
            is_partial=sample,
        )


def _ctx(prior_clarify: int = 0) -> AgentContext:
    return AgentContext(
        tenant_id="demo-tenant",
        kg_name="models",
        neptune=MagicMock(),
        anthropic_key="sk-ant-test",
        openrouter_key="",
        extras={"prior_clarify_count": prior_clarify},
    )


def _patch_preview(monkeypatch, *, entities, relationships=(), existing=None):
    """Make plan-time previewing deterministic: stub _fetch_ontology (existing
    types) and _extract (the multi-type extraction the preview reads)."""
    existing_types = {name: "" for name in (existing or [])}

    async def fake_fetch_ontology(self, graph_uri):
        return existing_types, {}

    async def fake_extract(self, content, content_type, existing=None):
        return ExtractionResult(
            entities=list(entities), relationships=list(relationships)
        )

    monkeypatch.setattr(SchemaResolver, "_fetch_ontology", fake_fetch_ontology)
    monkeypatch.setattr(SchemaResolver, "_extract", fake_extract)


# A simple single-type extraction the FakeProvider rows would yield.
def _single_type_entities():
    return [
        ExtractedEntity(
            type_name="OpenRouterModel",
            id=r["name"],
            attributes=[
                ExtractedAttribute(name="context_length", value=r["context_length"])
            ],
        )
        for r in FULL_ROWS[:5]
    ]


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_web_sources()
    yield
    reset_web_sources()


async def test_no_provider_degrades_to_not_enabled_answer():
    steps = await WebIngestCapability().plan(_ctx(), "find a list of OpenRouter models")
    assert len(steps) == 1 and steps[0].action == "answer"
    assert "isn't enabled" in steps[0].params["answer_payload"]["answer"]


async def test_entity_only_asks_to_confirm_attributes():
    register_web_source(FakeProvider())
    steps = await WebIngestCapability().plan(
        _ctx(), "a list of OpenRouter models", parsed=ENTITY_ONLY_SPEC
    )
    assert len(steps) == 1
    step = steps[0]
    assert step.action == "clarify"
    # Both clickable options carry the concrete attribute set so the next turn
    # converges without new UI.
    opts = step.params["options"]
    assert opts[0].startswith("Use these: name")
    assert "provider" in opts[0] and "context_length" in opts[0]
    assert opts[1] == "Just the name"


async def test_confirmed_attributes_builds_discovery_plan(monkeypatch):
    provider = FakeProvider()
    register_web_source(provider)
    _patch_preview(monkeypatch, entities=_single_type_entities())

    steps = await WebIngestCapability().plan(
        _ctx(), "can we ingest the models OpenRouter currently offers?",
        parsed=CONFIRMED_SPEC,
    )
    assert len(steps) == 1
    step = steps[0]
    assert step.action == "discover_ingest"

    # Sample fetched with the CLEAN search subject (from spec.query) + the
    # COMPREHENSIVE hint (key ∪ confirmed ∪ suggested) as hint_columns — NOT the
    # raw conversational sentence, and NOT the confirmed minimal list (Cause 1:
    # the provider projects to hint_columns, so a thin hint starves the fetch).
    q, sample, _max, cols = provider.calls[0]
    assert sample is True
    assert q == "OpenRouter models"
    # key=name, confirmed=[context_length], suggested=[provider, context_length].
    assert set(cols) == {"name", "context_length", "provider"}
    # The card text uses the clean subject, never echoes the raw question.
    assert "OpenRouter models" in step.rationale
    assert "can we ingest" not in step.rationale

    # Preview surfaces the DISCOVERED ontology shape (multi-type engine output).
    names = {t["name"] for t in step.preview["discovered_types"]}
    assert "OpenRouterModel" in names
    assert step.preview["relationships"] == []
    # No more flat mapping persisted; proposed_type stays as a useful label.
    assert "mapping" not in step.params
    assert step.params["proposed_type"] == "OpenRouterModel"
    # attributes = the confirmed naming set; hint_columns = the comprehensive
    # fetch union, persisted so execute() fetches the same rich projection.
    assert step.params["attributes"] == ["name", "context_length"]
    assert set(step.params["hint_columns"]) == {"name", "context_length", "provider"}


async def test_preview_surfaces_multiple_types_and_relationships(monkeypatch):
    """The plan card previews the multi-type ontology + the relationship the
    extractor inferred between two distinct entity types."""
    provider = FakeProvider()
    register_web_source(provider)
    entities = [
        ExtractedEntity(
            type_name="Model", id="claude-opus",
            attributes=[ExtractedAttribute(name="context_length", value="200000")],
        ),
        ExtractedEntity(
            type_name="Provider", id="anthropic",
            attributes=[ExtractedAttribute(name="homepage", value="anthropic.com")],
        ),
    ]
    rels = [ExtractedRelationship(
        source_id="claude-opus", predicate="provided_by", target_id="anthropic",
    )]
    _patch_preview(monkeypatch, entities=entities, relationships=rels, existing=["Provider"])

    steps = await WebIngestCapability().plan(
        _ctx(), "models OpenRouter offers", parsed=CONFIRMED_SPEC
    )
    step = steps[0]
    types = {t["name"]: t for t in step.preview["discovered_types"]}
    assert set(types) == {"Model", "Provider"}
    # is_new reflects the existing ontology (Provider exists, Model is new).
    assert types["Model"]["is_new"] is True
    assert types["Provider"]["is_new"] is False
    assert "context_length" in types["Model"]["attributes"]

    rels_out = step.preview["relationships"]
    assert len(rels_out) == 1
    assert rels_out[0] == {
        "source": "Model", "predicate": "provided_by", "target": "Provider",
    }


async def test_preview_summary_frames_shape_as_estimate(monkeypatch):
    """FIX 5: the discovered TYPES/relationships are an ESTIMATE from the small
    sample, not a guarantee — the user-facing summary must say so (only the
    column projection is stable preview→commit). Wording-only assertion."""
    provider = FakeProvider()
    register_web_source(provider)
    _patch_preview(monkeypatch, entities=_single_type_entities())

    steps = await WebIngestCapability().plan(
        _ctx(), "models OpenRouter offers", parsed=CONFIRMED_SPEC
    )
    summary = steps[0].preview["summary"].lower()
    # Must NOT over-claim certainty ("Discovered N types") and must signal the
    # commit may differ.
    assert "estimated" in summary
    assert "may differ" in summary
    assert "discovered " not in summary


async def test_preview_degrades_to_flat_when_extract_fails(monkeypatch):
    """If the plan-time extractor raises, plan() still returns a confirmable plan
    card (degraded flat single-type preview) — no exception propagates."""
    provider = FakeProvider()
    register_web_source(provider)

    async def fake_fetch_ontology(self, graph_uri):
        return {}, {}

    async def boom_extract(self, content, content_type, existing=None):
        raise RuntimeError("extractor unavailable")

    monkeypatch.setattr(SchemaResolver, "_fetch_ontology", fake_fetch_ontology)
    monkeypatch.setattr(SchemaResolver, "_extract", boom_extract)

    steps = await WebIngestCapability().plan(
        _ctx(), "models OpenRouter offers", parsed=CONFIRMED_SPEC
    )
    assert len(steps) == 1
    step = steps[0]
    assert step.action == "discover_ingest"
    # Degraded: one flat discovered type = the proposed type with the attributes.
    dts = step.preview["discovered_types"]
    assert len(dts) == 1
    assert dts[0]["name"] == "OpenRouterModel"
    assert dts[0]["attributes"] == ["name", "context_length"]
    assert step.preview["relationships"] == []


async def test_commit_to_suggested_after_prior_clarify(monkeypatch):
    register_web_source(FakeProvider())
    _patch_preview(monkeypatch, entities=_single_type_entities())
    # Entity-only spec, but we've already asked once → commit to suggested,
    # don't clarify again.
    steps = await WebIngestCapability().plan(
        _ctx(prior_clarify=1), "a list of OpenRouter models", parsed=ENTITY_ONLY_SPEC
    )
    assert len(steps) == 1
    assert steps[0].action == "discover_ingest"
    assert steps[0].params["attributes"] == ["name", "provider", "context_length", "pricing"]


async def test_paid_provider_quotes_cost(monkeypatch):
    register_web_source(FakeProvider(is_paid=True, cost_per_call=0.01))
    _patch_preview(monkeypatch, entities=_single_type_entities())
    steps = await WebIngestCapability().plan(
        _ctx(), "list of OpenRouter models", parsed=CONFIRMED_SPEC
    )
    cost = steps[0].cost
    assert cost["paid_calls"] == 1
    assert cost["estimated_usd"] == pytest.approx(0.01)
    assert "Paid web discovery" in cost["note"]


async def test_empty_sample_returns_message():
    register_web_source(FakeProvider(rows=[]))
    steps = await WebIngestCapability().plan(
        _ctx(), "find a list of nonsense xyzzy", parsed=CONFIRMED_SPEC
    )
    assert len(steps) == 1 and steps[0].action == "answer"
    assert "couldn't find anything" in steps[0].params["answer_payload"]["answer"]


async def test_execute_runs_full_discover_and_ingests(monkeypatch):
    provider = FakeProvider()
    register_web_source(provider)
    _patch_preview(monkeypatch, entities=_single_type_entities())

    captured: dict = {}

    async def fake_ingest(self, content, tenant_id, content_type="text", source="", instance_graph=None):
        captured.update(
            content=content, tenant_id=tenant_id,
            content_type=content_type, source=source,
        )
        rows = json.loads(content)
        return IngestResult(entities_extracted=len(rows), entities_resolved=len(rows))

    monkeypatch.setattr(SchemaResolver, "ingest", fake_ingest)

    spawned: dict = {}
    monkeypatch.setattr(
        web_ingest_cap, "_spawn",
        lambda coro: spawned.__setitem__("task", asyncio.ensure_future(coro)),
    )

    cap = WebIngestCapability()
    step = (await cap.plan(_ctx(), "find a list of OpenRouter models", parsed=CONFIRMED_SPEC))[0]
    ack = await cap.execute(_ctx(), step)
    assert ack["kind"] == "ack" and "background" in ack["message"]

    await spawned["task"]

    # Full pull (sample=False) with the COMPREHENSIVE hint (key ∪ confirmed ∪
    # suggested) — the SAME rich projection the sample used (the FETCH is the
    # stable part preview→commit; the discovered shape is only an estimate),
    # NOT the confirmed minimal list. Committed through the multi-type ingest
    # path (content_type="json").
    assert provider.calls[-1][1] is False
    assert set(provider.calls[-1][3]) == {"name", "context_length", "provider"}
    assert captured["content_type"] == "json"
    # The JSON round-trips back to the rows the provider returned (projected to
    # the comprehensive hint).
    rows_back = json.loads(captured["content"])
    assert len(rows_back) == len(FULL_ROWS)
    assert set(rows_back[0].keys()) == {"name", "context_length", "provider"}
    # The clean search subject (spec.query) is what the provider + source use.
    assert captured["source"] == "web:fake:OpenRouter models"


async def test_run_routes_multi_type_and_refreshes(monkeypatch):
    """The commit routes through ingest (content_type="json") and the post-write
    refresh is driven by the multi-type result: every type the ingest created or
    extended is in the affected_types passed to refresh_after_write."""
    provider = FakeProvider()
    register_web_source(provider)
    _patch_preview(monkeypatch, entities=_single_type_entities())

    ingest_calls: dict = {}

    async def spy_ingest(self, content, tenant_id, content_type="text", source="", instance_graph=None):
        ingest_calls.update(content=content, content_type=content_type)
        return IngestResult(
            types_created=["Model", "Provider"],
            attributes_added=["Model.provided_by"],
            entities_resolved=4,
        )

    monkeypatch.setattr(SchemaResolver, "ingest", spy_ingest)

    refreshed: dict = {}

    async def fake_refresh(neptune, *, tenant_id, kg_name, affected_types):
        refreshed.update(affected_types=affected_types, kg_name=kg_name)

    monkeypatch.setattr(web_ingest_cap, "refresh_after_write", fake_refresh)

    spawned: dict = {}
    monkeypatch.setattr(
        web_ingest_cap, "_spawn",
        lambda coro: spawned.__setitem__("task", asyncio.ensure_future(coro)),
    )

    cap = WebIngestCapability()
    step = (await cap.plan(_ctx(), "list of OpenRouter models", parsed=CONFIRMED_SPEC))[0]
    await cap.execute(_ctx(), step)
    await spawned["task"]

    assert ingest_calls["content_type"] == "json"
    # affected_types = types_created ∪ owning-types of attributes_added.
    assert refreshed["affected_types"] == {"Model", "Provider"}


def _ctx_with_store(store) -> AgentContext:
    """Agent context carrying a job store, as the agent route injects it."""
    return AgentContext(
        tenant_id="demo-tenant",
        kg_name="models",
        neptune=MagicMock(),
        anthropic_key="sk-ant-test",
        openrouter_key="",
        extras={"prior_clarify_count": 0, "enrichment_job_store": store},
    )


async def test_execute_tracks_job_with_results_and_platforms(monkeypatch):
    """With a job store present, execute creates a tracked discovery job, returns
    its id + initial status, and drives it to applied with a result count, the
    platforms consulted, and the run cost — so the client can poll a live status."""
    from cograph_client.enrichment.job_store import InMemoryJobStore
    from cograph_client.enrichment.models import JobCategory, JobStatus

    provider = FakeProvider(is_paid=True, cost_per_call=0.09)
    register_web_source(provider)
    _patch_preview(monkeypatch, entities=_single_type_entities())

    async def fake_ingest(self, content, tenant_id, content_type="text", source="", instance_graph=None):
        rows = json.loads(content)
        return IngestResult(entities_extracted=len(rows), entities_resolved=len(rows))

    monkeypatch.setattr(SchemaResolver, "ingest", fake_ingest)

    spawned: dict = {}
    monkeypatch.setattr(
        web_ingest_cap, "_spawn",
        lambda coro: spawned.__setitem__("task", asyncio.ensure_future(coro)),
    )

    store = InMemoryJobStore()
    cap = WebIngestCapability()
    step = (
        await cap.plan(_ctx_with_store(store), "list of OpenRouter models", parsed=CONFIRMED_SPEC)
    )[0]
    ack = await cap.execute(_ctx_with_store(store), step)

    # The ack hands back a job id + initial status to poll on.
    assert ack["kind"] == "ack"
    job_id = ack["job_id"]
    assert ack["job_status"] == "queued"

    # The job is in the store immediately (queued), then completes after the run.
    queued = await store.get(job_id)
    assert queued is not None
    assert queued.category == JobCategory.discovery
    assert queued.cost == pytest.approx(0.09)

    await spawned["task"]

    done = await store.get(job_id)
    assert done.status == JobStatus.applied
    assert done.result_count == len(FULL_ROWS)
    assert done.progress.total == len(FULL_ROWS)
    assert done.progress.processed == len(FULL_ROWS)
    assert "openrouter.ai" in (done.platforms or [])
    assert done.type_name == "OpenRouterModel"
    assert done.completed_at is not None


async def test_execute_marks_job_failed_on_error(monkeypatch):
    """A discovery that raises mid-ingest leaves the job failed with an error, not
    silently dropped — so the live status can show the failure."""
    from cograph_client.enrichment.job_store import InMemoryJobStore
    from cograph_client.enrichment.models import JobStatus

    register_web_source(FakeProvider())
    _patch_preview(monkeypatch, entities=_single_type_entities())

    async def boom(self, *a, **k):
        raise RuntimeError("ingest exploded")

    monkeypatch.setattr(SchemaResolver, "ingest", boom)
    spawned: dict = {}
    monkeypatch.setattr(
        web_ingest_cap, "_spawn",
        lambda coro: spawned.__setitem__("task", asyncio.ensure_future(coro)),
    )

    store = InMemoryJobStore()
    cap = WebIngestCapability()
    step = (
        await cap.plan(_ctx_with_store(store), "list of OpenRouter models", parsed=CONFIRMED_SPEC)
    )[0]
    ack = await cap.execute(_ctx_with_store(store), step)
    await spawned["task"]

    failed = await store.get(ack["job_id"])
    assert failed.status == JobStatus.failed
    assert "ingest exploded" in (failed.error or "")


def test_capability_registered_by_default():
    from cograph_client.agent.planner import register_default_capabilities
    from cograph_client.agent.registry import get_capability

    register_default_capabilities()
    assert get_capability("web_ingest") is not None


async def test_planner_short_circuits_capability_clarify(monkeypatch):
    """End-to-end: a discover turn whose capability needs attribute confirmation
    returns {kind:"clarify"} via the planner's clarify short-circuit."""
    import json as _json

    from cograph_client.agent import planner as planner_mod
    from cograph_client.agent.registry import register_capability, reset_capabilities

    async def fake_classify_chat(*_a, **_k):
        return _json.dumps({"intents": ["discover"]})

    async def fake_spec_chat(*_a, **_k):
        return _json.dumps(ENTITY_ONLY_SPEC)

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_classify_chat)
    monkeypatch.setattr(web_ingest_cap, "openrouter_chat", fake_spec_chat)

    reset_capabilities()
    register_capability(WebIngestCapability())
    register_web_source(FakeProvider())

    ctx = AgentContext(
        tenant_id="demo-tenant", kg_name="models", neptune=MagicMock(),
        anthropic_key="sk-ant-test", openrouter_key="k",
    )
    result = await planner_mod.handle(ctx, "a list of OpenRouter models")
    assert result["kind"] == "clarify"
    assert any(o.startswith("Use these:") for o in result["options"])
    reset_capabilities()
