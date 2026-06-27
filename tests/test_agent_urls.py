"""Tests for URL-targeted web extraction at the OSS interface layer (WP3).

A user can hand the agent explicit links — in the chat message OR as structured
request context (``context.urls``). This pins the seams that thread those URLs
from the request into the agent context and route a URL-bearing turn:

  * the agent route carries ``context.urls`` into :class:`AgentContext.urls`,
  * the planner deterministically routes a URL-bearing message — an enrich-type
    verb → ``enrich`` (fill attributes on existing entities), else → ``discover``
    (bring in a new set) — so it is never mis-filed as a plain question, and
  * the enrich route maps ``EnrichRequest.target_urls`` onto the created job's
    ``EnrichJob.source_urls`` (the URL-targeted enrichment rail).

Everything is stubbed (classifier LLM, enrich extraction, executor) so the suite
is deterministic and offline. Each async test is wrapped in ``asyncio.wait_for``
to fail loudly rather than hang.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

# Mirror the test env so settings/auth construct the same way as in CI.
os.environ.setdefault("OMNIX_API_KEYS", '{"test-key": "test-tenant"}')
os.environ.setdefault("OMNIX_NEPTUNE_ENDPOINT", "http://fake:8182")

from cograph_client.agent import planner as planner_mod  # noqa: E402
from cograph_client.agent.planner import (  # noqa: E402
    handle,
    register_default_capabilities,
    reset_plan_store,
)
from cograph_client.agent.registry import (  # noqa: E402
    AgentContext,
    get_capability,
    reset_capabilities,
)
from cograph_client.agent.conversation_store import (  # noqa: E402
    reset_conversation_store,
)

TIMEOUT = 5.0


# --------------------------------------------------------------------------- #
# Fakes (mirrors test_agent.py so the two suites stay consistent).
# --------------------------------------------------------------------------- #
class FakeNeptune:
    async def query(self, q):
        return {"head": {"vars": []}, "results": {"bindings": []}}

    async def update(self, q):
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

    async def update(self, job):
        return None


class FakeExecutor:
    def __init__(self):
        self.ran = []

    async def run(self, job, tenant_id):
        self.ran.append((job, tenant_id))


def _ctx(urls=None, **extras_kw):
    return AgentContext(
        tenant_id="t1",
        kg_name="kg1",
        neptune=FakeNeptune(),
        type_name="Company",
        urls=list(urls or []),
        openrouter_key="fake-key",
        anthropic_key="fake-anthropic",
        extras={
            "enrichment_executor": extras_kw.get("executor", FakeExecutor()),
            "enrichment_job_store": extras_kw.get("job_store", FakeJobStore()),
        },
    )


@pytest.fixture(autouse=True)
def _fresh_registry():
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
    """Run capability-spawned background work as TRACKED tasks (see test_agent)."""
    import cograph_client.agent.capabilities.dedup_cap as dedup_cap
    import cograph_client.agent.capabilities.enrich_cap as enrich_cap
    import cograph_client.agent.capabilities.normalize_cap as norm_cap

    def tracking_spawn(coro):
        asyncio.ensure_future(coro)

    monkeypatch.setattr(norm_cap, "_spawn", tracking_spawn)
    monkeypatch.setattr(enrich_cap, "_spawn", tracking_spawn)
    monkeypatch.setattr(dedup_cap, "_spawn", tracking_spawn)


def _stub_classifier(monkeypatch, intent: str, clarify: str = ""):
    async def fake_chat(*args, **kwargs):
        return json.dumps({"intent": intent, "clarify": clarify})

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_chat)


def _stub_enrich(monkeypatch, payload: dict):
    """Stub the enrich capability's schema + extraction so plan() can ground."""

    async def fake_schema(neptune, tenant_id, type_name):
        return {"attributes": ["website"], "relationships": []}

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.list_type_schema", fake_schema
    )

    async def fake_kg_types(ctx):
        return ["Company"]

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap._list_types", fake_kg_types
    )

    async def fake_chat(*args, **kwargs):
        return json.dumps(payload)

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.openrouter_chat", fake_chat
    )

    async def fake_sample(*a, **k):
        return ([], "attribute")

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.sample_predicate_values",
        fake_sample,
    )

    class _Exec:
        async def count_entities(self, *a, **k):
            return 3

        async def run(self, job, tenant_id):
            return None

    return _Exec()


# --------------------------------------------------------------------------- #
# 1. The request context carries urls into AgentContext.
# --------------------------------------------------------------------------- #
def test_request_context_carries_urls_into_agent_context(monkeypatch):
    """``AgentRequestContext.urls`` is threaded into ``AgentContext.urls`` by the
    route's ``_build_ctx`` — so capabilities see the user's attached links."""
    from cograph_client.api.routes.agent import (
        AgentRequest,
        AgentRequestContext,
        _build_ctx,
    )
    from cograph_client.auth.api_keys import TenantContext

    urls = ["https://example.com/a", "https://example.com/b"]
    body = AgentRequest(
        message="enrich from these",
        context=AgentRequestContext(kg_name="kg1", type_name="Company", urls=urls),
    )
    ctx = _build_ctx(
        TenantContext(tenant_id="t1", api_key="k"),
        body,
        FakeNeptune(),
        FakeExecutor(),
        FakeJobStore(),
    )
    assert ctx.urls == urls


def test_agent_request_context_urls_defaults_empty():
    """``urls`` is optional with a safe default → existing clients unaffected."""
    from cograph_client.api.routes.agent import AgentRequestContext

    assert AgentRequestContext(kg_name="kg1").urls == []


# --------------------------------------------------------------------------- #
# 2. Planner routes a URL-bearing ENRICH-verb message to enrich.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_url_with_enrich_verb_routes_to_enrich(monkeypatch):
    """'enrich Company website from https://x' carries a URL + an enrich verb →
    the guard routes to enrich even though the classifier mis-files it as a
    question (the message reads query-like)."""
    _stub_classifier(monkeypatch, "question")  # deliberately wrong
    executor = _stub_enrich(
        monkeypatch,
        {"attributes": ["website"], "scope": None, "tier": "core"},
    )

    out = await asyncio.wait_for(
        handle(
            _ctx(executor=executor),
            "enrich the Company website from https://acme.example.com/about",
        ),
        TIMEOUT,
    )
    assert out["kind"] == "plan"
    assert out["steps"][0]["capability"] == "enrich"


@pytest.mark.asyncio
async def test_ctx_urls_with_enrich_verb_routes_to_enrich(monkeypatch):
    """URLs supplied via structured request context (ctx.urls, no URL in the
    message text) also trigger the guard."""
    _stub_classifier(monkeypatch, "question")
    executor = _stub_enrich(
        monkeypatch,
        {"attributes": ["website"], "scope": None, "tier": "core"},
    )

    out = await asyncio.wait_for(
        handle(
            _ctx(urls=["https://acme.example.com/about"], executor=executor),
            "fill in the website for these companies",
        ),
        TIMEOUT,
    )
    assert out["kind"] == "plan"
    assert out["steps"][0]["capability"] == "enrich"


# --------------------------------------------------------------------------- #
# 3. Planner routes a URL-bearing NON-enrich message to discover.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_url_without_enrich_verb_routes_to_discover(monkeypatch):
    """'add companies from https://x' carries a URL + a NON-enrich verb → the
    guard routes to discovery (a NEW set of records, Rail A). With no URL-capable
    provider registered in OSS, discover degrades to a clear 'not enabled' answer
    — proof it handled the turn (and was not answered as a plain question)."""
    _stub_classifier(monkeypatch, "question")

    from cograph_client.agent.capabilities.query import QueryCapability

    async def fake_answer(self, ctx, q):
        return {"answer": "SHOULD_NOT_RUN", "sparql": "", "rows": [], "narrative": ""}

    monkeypatch.setattr(QueryCapability, "answer", fake_answer)

    out = await asyncio.wait_for(
        handle(_ctx(), "add the companies from https://list.example.com/companies"),
        TIMEOUT,
    )
    assert out.get("answer") != "SHOULD_NOT_RUN"  # not the read-only path
    body = f"{out.get('narrative', '')} {out.get('answer', '')}".lower()
    assert "enabled" in body  # routed to discovery (degrades to not-enabled in OSS)


@pytest.mark.asyncio
async def test_no_urls_leaves_question_untouched(monkeypatch):
    """A plain question with NO URLs is unaffected by the guard — it still
    answers (the guard must not over-trigger)."""
    _stub_classifier(monkeypatch, "question")

    from cograph_client.agent.capabilities.query import QueryCapability

    async def fake_answer(self, ctx, q):
        return {"answer": "42", "sparql": "SELECT", "rows": [], "narrative": ""}

    monkeypatch.setattr(QueryCapability, "answer", fake_answer)

    out = await asyncio.wait_for(
        handle(_ctx(), "how many companies are there?"), TIMEOUT
    )
    assert out["kind"] == "answer"
    assert out["answer"] == "42"


@pytest.mark.asyncio
async def test_url_in_question_form_defers_to_classifier(monkeypatch):
    """A genuine read-only QUESTION that merely contains a link in its text (no
    attached ctx.urls, no enrich verb) must be ANSWERED, not hijacked into an
    ingest/enrich plan. Mirrors the web-discovery interrogative guard."""
    _stub_classifier(monkeypatch, "question")

    from cograph_client.agent.capabilities.query import QueryCapability

    async def fake_answer(self, ctx, q):
        return {"answer": "ANSWERED", "sparql": "", "rows": [], "narrative": ""}

    monkeypatch.setattr(QueryCapability, "answer", fake_answer)

    out = await asyncio.wait_for(
        handle(
            _ctx(),
            "what does https://acme.example.com/about say about their pricing?",
        ),
        TIMEOUT,
    )
    assert out["kind"] == "answer"
    assert out["answer"] == "ANSWERED"  # deferred to the read-only path


@pytest.mark.asyncio
async def test_question_form_url_with_enrich_verb_still_routes(monkeypatch):
    """An explicit enrich verb is an unambiguous action — it routes even when the
    message is phrased as a question (the verb wins over the interrogative guard)."""
    _stub_classifier(monkeypatch, "question")
    executor = _stub_enrich(
        monkeypatch,
        {"attributes": ["website"], "scope": None, "tier": "core"},
    )

    out = await asyncio.wait_for(
        handle(
            _ctx(executor=executor),
            "can you enrich the Company website from https://acme.example.com/about?",
        ),
        TIMEOUT,
    )
    assert out["kind"] == "plan"
    assert out["steps"][0]["capability"] == "enrich"


@pytest.mark.asyncio
async def test_attached_ctx_urls_route_even_in_question_form(monkeypatch):
    """Links ATTACHED as structured context (ctx.urls) are a deliberate action
    signal — they route even when the message is a bare question with no verb."""
    _stub_classifier(monkeypatch, "question")

    from cograph_client.agent.capabilities.query import QueryCapability

    async def fake_answer(self, ctx, q):
        return {"answer": "SHOULD_NOT_RUN", "sparql": "", "rows": [], "narrative": ""}

    monkeypatch.setattr(QueryCapability, "answer", fake_answer)

    out = await asyncio.wait_for(
        handle(_ctx(urls=["https://list.example.com/companies"]), "what's here?"),
        TIMEOUT,
    )
    assert out.get("answer") != "SHOULD_NOT_RUN"  # not the read-only path
    body = f"{out.get('narrative', '')} {out.get('answer', '')}".lower()
    assert "enabled" in body  # routed to discovery (degrades to not-enabled in OSS)


# --------------------------------------------------------------------------- #
# 4. The _url_intent helper: enrich-type verb vs everything else.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "message,expected",
    [
        ("enrich the website from https://x", "enrich"),
        ("fill in the prices from https://x", "enrich"),
        ("update the addresses from https://x", "enrich"),
        ("complete the missing fields from https://x", "enrich"),
        ("populate the company data from https://x", "enrich"),
        ("add companies from https://x", "discover"),
        ("import the records at https://x", "discover"),
        ("pull the data from https://x", "discover"),
        ("scrape https://x", "discover"),
    ],
)
def test_url_intent_classifies_by_verb(message, expected):
    assert planner_mod._url_intent(message) == expected


def test_message_has_urls_detects_both_sources():
    assert planner_mod._message_has_urls("see https://x", None) is True
    assert planner_mod._message_has_urls("no links here", ["https://x"]) is True
    assert planner_mod._message_has_urls("no links here", None) is False
    assert planner_mod._message_has_urls("no links here", []) is False


# --------------------------------------------------------------------------- #
# 5. Enrich route maps target_urls onto the created job's source_urls (WP2 fields).
# --------------------------------------------------------------------------- #
def test_enrich_route_maps_target_urls_to_job_source_urls(monkeypatch):
    """POST /enrich/jobs with ``target_urls`` creates an EnrichJob carrying those
    as ``source_urls`` (the URL-targeted enrichment rail). Asserted via the job
    store the route writes to — the same primitive the existing route tests use."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    from cograph_client.api.app import create_app
    from cograph_client.graph.client import NeptuneClient

    app = create_app()
    n = AsyncMock(spec=NeptuneClient)
    n.query.return_value = {"head": {"vars": []}, "results": {"bindings": []}}
    n.update.return_value = None
    app.state.neptune_client = n

    # Capture the job the route hands the executor without doing real work.
    captured: dict = {}

    async def fake_run(self, job, tenant_id):
        captured["job"] = job

    monkeypatch.setattr(
        "cograph_client.enrichment.executor.EnrichmentExecutor.run", fake_run
    )

    client = TestClient(app)
    urls = ["https://acme.example.com/about", "https://acme.example.com/team"]
    r = client.post(
        "/graphs/test-tenant/enrich/jobs",
        json={
            "type_name": "Company",
            "attributes": ["website"],
            "kg_name": "kg1",
            "tier": "core",  # explicit tier → no auto-resolve LLM call needed
            "target_urls": urls,
        },
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 202, r.text
    # Let the spawned executor task run so it captures the job.
    job = captured.get("job")
    assert job is not None, "executor.run was not invoked with a job"
    assert job.source_urls == urls


def test_enrich_route_defaults_source_urls_empty(monkeypatch):
    """Omitting ``target_urls`` leaves the job's ``source_urls`` empty — a normal
    enrich is unchanged (backward compatible)."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    from cograph_client.api.app import create_app
    from cograph_client.graph.client import NeptuneClient

    app = create_app()
    n = AsyncMock(spec=NeptuneClient)
    n.query.return_value = {"head": {"vars": []}, "results": {"bindings": []}}
    n.update.return_value = None
    app.state.neptune_client = n

    captured: dict = {}

    async def fake_run(self, job, tenant_id):
        captured["job"] = job

    monkeypatch.setattr(
        "cograph_client.enrichment.executor.EnrichmentExecutor.run", fake_run
    )

    client = TestClient(app)
    r = client.post(
        "/graphs/test-tenant/enrich/jobs",
        json={
            "type_name": "Company",
            "attributes": ["website"],
            "kg_name": "kg1",
            "tier": "core",
        },
        headers={"X-API-Key": "test-key"},
    )
    assert r.status_code == 202, r.text
    job = captured.get("job")
    assert job is not None
    assert job.source_urls == []
