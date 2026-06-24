"""COG-130: the Ask-AI agent must converge to a plan instead of looping on
``clarify``. These tests pin the fix:

1. A session-keyed conversation store persists the rolling transcript.
2. ``handle`` threads that transcript into the classifier (so a terse answer to
   a prior clarify is classified in context, not in isolation) and into the
   capabilities (so the accumulated answers ground field extraction).
3. A multi-action ask ("clean the names and remove duplicates") composes ONE
   ordered plan (clean → dedup) instead of re-asking which one.

Everything is stubbed so the suite is deterministic and fast (no network, no
real Neptune). Mirrors the fakes/fixtures in ``test_agent.py``.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from cograph_client.agent import planner as planner_mod
from cograph_client.agent.conversation_store import (
    InMemoryConversationStore,
    Turn,
    make_conversation_store,
    reset_conversation_store,
)
from cograph_client.auth.api_keys import AuthVerdict, register_external_verifier
from cograph_client.agent.planner import (
    handle,
    register_default_capabilities,
    reset_plan_store,
)
from cograph_client.agent.registry import (
    AgentContext,
    reset_capabilities,
)

TIMEOUT = 5.0


# --------------------------------------------------------------------------- #
# Fakes (same shape as test_agent.py)
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
        return next((j for j in self.created if j.id == job_id), None)

    async def update(self, job):
        return None


class FakeExecutor:
    async def run(self, job, tenant_id):
        return None


def _ctx():
    return AgentContext(
        tenant_id="t1",
        kg_name="kg1",
        neptune=FakeNeptune(),
        type_name="Manufacturer",
        openrouter_key="fake-key",
        extras={
            "enrichment_executor": FakeExecutor(),
            "enrichment_job_store": FakeJobStore(),
        },
    )


# The Manufacturer type schema the normalize extraction is grounded in.
_SCHEMA = {"attributes": ["name", "country"], "relationships": []}


@pytest.fixture(autouse=True)
def _fresh():
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
    """Run capability-spawned background work as tracked tasks (no leaks)."""
    import cograph_client.agent.capabilities.dedup_cap as dedup_cap
    import cograph_client.agent.capabilities.enrich_cap as enrich_cap
    import cograph_client.agent.capabilities.normalize_cap as norm_cap

    def tracking_spawn(coro):
        asyncio.ensure_future(coro)

    monkeypatch.setattr(norm_cap, "_spawn", tracking_spawn)
    monkeypatch.setattr(enrich_cap, "_spawn", tracking_spawn)
    monkeypatch.setattr(dedup_cap, "_spawn", tracking_spawn)


def _stub_schema(monkeypatch):
    async def fake_schema(neptune, tenant_id, type_name):
        return _SCHEMA

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.normalize_cap.list_type_schema",
        fake_schema,
    )


def _stub_normalize_directive(monkeypatch, predicate="name"):
    """Normalize directive LLM → a strip_emoji rule on the named field."""

    async def fake_chat(*args, **kwargs):
        return json.dumps(
            {
                "rule_type": "strip_emoji",
                "predicate": predicate,
                "params": {},
                "confidence": 0.9,
                "rationale": "tidy up the manufacturer names",
            }
        )

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.normalize_cap.openrouter_chat", fake_chat
    )

    async def fake_sample(neptune, tenant_id, kg, type_name, pred_leaf):
        return (["Acme Corp 🚀", "Acme Corp"], "attribute")

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.normalize_cap.sample_predicate_values",
        fake_sample,
    )


def _stub_history_aware_classifier(monkeypatch):
    """A classifier that mimics the real failure-then-recovery behavior.

    Turn 1 (no transcript in the prompt) → ``ambiguous`` (a single message like
    "clean up the names and remove duplicates" reads as under-specified to the
    isolated classifier — this is exactly what produced the live loop). Once the
    transcript is threaded in (the prompt contains "Conversation so far"), it
    commits to BOTH clean and dedup. This keys off the prompt text, so it only
    recovers if ``handle`` actually fed the history through.
    """

    async def fake_chat(key, system, user, **kwargs):
        if "Conversation so far" not in user:
            return json.dumps(
                {
                    "intents": ["ambiguous"],
                    "clarify": "Do you want to clean the names, merge duplicates, "
                    "or both?",
                }
            )
        return json.dumps({"intents": ["clean", "dedup"], "clarify": ""})

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_chat)


# --------------------------------------------------------------------------- #
# 1. Conversation store
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_conversation_store_roundtrip_and_tenant_scope():
    store = InMemoryConversationStore()
    assert await store.load("s1", "t1") == []

    await store.append("s1", "t1", [Turn(role="user", text="hi")])
    await store.append(
        "s1", "t1", [Turn(role="assistant", text="hello", kind="clarify")]
    )
    turns = await store.load("s1", "t1")
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[1].kind == "clarify"

    # Tenant-scoped: a different tenant with the same session id is isolated.
    assert await store.load("s1", "other") == []
    # A returned copy can't mutate stored state by reference.
    turns[0].text = "MUTATED"
    assert (await store.load("s1", "t1"))[0].text == "hi"


@pytest.mark.asyncio
async def test_conversation_store_trims_to_max():
    from cograph_client.agent import conversation_store as cs

    store = InMemoryConversationStore()
    for i in range(cs._MAX_TURNS + 8):
        await store.append("s", "t", [Turn(role="user", text=str(i))])
    turns = await store.load("s", "t")
    assert len(turns) == cs._MAX_TURNS
    # The OLDEST were dropped; the most-recent are kept.
    assert turns[-1].text == str(cs._MAX_TURNS + 7)


@pytest.mark.asyncio
async def test_no_session_skips_history_and_does_not_persist(monkeypatch):
    """Without a session id, nothing is loaded or stored (single-turn behavior)."""
    _stub_schema(monkeypatch)
    _stub_normalize_directive(monkeypatch)

    async def fake_chat(key, system, user, **kwargs):
        # No history can ever be present without a session id.
        assert "Conversation so far" not in user
        return json.dumps({"intents": ["clean"], "clarify": ""})

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_chat)

    out = await asyncio.wait_for(handle(_ctx(), "clean the names"), TIMEOUT)
    assert out["kind"] == "plan"
    assert "session_id" not in out  # nothing echoed when none supplied


# --------------------------------------------------------------------------- #
# 2. The COG-130 repro: clarify once, then converge to a plan
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_clarify_then_converges_to_plan(monkeypatch):
    """The exact bug: turn 1 may clarify, but turn 2 (after the user answers)
    MUST produce a plan — never another clarify on the same dimension."""
    _stub_schema(monkeypatch)
    _stub_normalize_directive(monkeypatch)
    _stub_history_aware_classifier(monkeypatch)

    ctx = _ctx()
    session = {"id": "sess-1"}

    # Turn 1: under-specified in isolation → one clarify.
    t1 = await asyncio.wait_for(
        handle(ctx, "I want to clean up the names and remove duplicates.", session),
        TIMEOUT,
    )
    assert t1["kind"] == "clarify"
    assert t1["session_id"] == "sess-1"  # echoed back

    # Turn 2: the user answers "both". With the transcript threaded in, the
    # classifier commits and the agent returns a PLAN, not another clarify.
    t2 = await asyncio.wait_for(handle(ctx, "I wanna do both.", session), TIMEOUT)
    assert t2["kind"] == "plan", t2
    caps = [s["capability"] for s in t2["steps"]]
    # Composed clean-before-dedup: a normalize step and a dedup step.
    assert "normalize" in caps and "dedup" in caps
    norm = next(s for s in t2["steps"] if s["capability"] == "normalize")
    dedup = next(s for s in t2["steps"] if s["capability"] == "dedup")
    # dedup runs AFTER the clean step (depends_on wired across capabilities).
    assert norm["id"] in dedup["depends_on"]
    # normalize is ordered first.
    assert t2["steps"][0]["capability"] == "normalize"


@pytest.mark.asyncio
async def test_transcript_is_persisted_across_turns(monkeypatch):
    """Each turn appends user + assistant turns to the session transcript."""
    _stub_schema(monkeypatch)
    _stub_normalize_directive(monkeypatch)
    _stub_history_aware_classifier(monkeypatch)

    ctx = _ctx()
    session = {"id": "sess-2"}
    await asyncio.wait_for(handle(ctx, "clean names and remove dupes", session), TIMEOUT)

    turns = await make_conversation_store().load("sess-2", "t1")
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[1].role == "assistant"
    assert turns[1].kind == "clarify"  # turn 1 clarified

    # Second turn appends two more and the assistant turn records a plan.
    await asyncio.wait_for(handle(ctx, "both please", session), TIMEOUT)
    turns = await make_conversation_store().load("sess-2", "t1")
    assert len(turns) == 4
    assert turns[-1].kind == "plan"


@pytest.mark.asyncio
async def test_does_not_reask_after_max_clarify_rounds(monkeypatch):
    """Convergence guard: even if the model stays unsure, once the clarify cap is
    reached the classifier is told to commit — the prompt carries the guard and
    the accumulated answers let a capability ground a step, so we get a plan."""
    _stub_schema(monkeypatch)
    _stub_normalize_directive(monkeypatch)

    seen_guard = {"value": False}

    async def fake_chat(key, system, user, **kwargs):
        # Turn 1: ambiguous (no transcript yet).
        if "Conversation so far" not in user:
            return json.dumps(
                {"intents": ["ambiguous"], "clarify": "Which field?"}
            )
        # Later turns: the convergence guard must be present in the prompt.
        if "ALREADY asked" in user:
            seen_guard["value"] = True
        return json.dumps({"intents": ["clean"], "clarify": ""})

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_chat)

    ctx = _ctx()
    session = {"id": "sess-3"}
    t1 = await asyncio.wait_for(handle(ctx, "tidy the data", session), TIMEOUT)
    assert t1["kind"] == "clarify"
    t2 = await asyncio.wait_for(handle(ctx, "the name field", session), TIMEOUT)
    assert t2["kind"] == "plan"
    assert seen_guard["value"], "the convergence guard was not injected into the prompt"


# --------------------------------------------------------------------------- #
# 3. Multi-intent composition works without a session too (single turn)
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_multi_intent_single_turn_composes_plan(monkeypatch):
    """A single message asking for two actions composes one ordered plan."""
    _stub_schema(monkeypatch)
    _stub_normalize_directive(monkeypatch)

    async def fake_chat(key, system, user, **kwargs):
        return json.dumps({"intents": ["clean", "dedup"], "clarify": ""})

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_chat)

    out = await asyncio.wait_for(
        handle(_ctx(), "clean the names and merge duplicates"), TIMEOUT
    )
    assert out["kind"] == "plan"
    caps = {s["capability"] for s in out["steps"]}
    assert caps == {"normalize", "dedup"}


# --------------------------------------------------------------------------- #
# 4. Clickable clarify options
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_clarify_carries_classifier_options(monkeypatch):
    """An ambiguous clarify surfaces the classifier's own suggested options so
    the UI can render them as clickable choices."""

    async def fake_chat(key, system, user, **kwargs):
        return json.dumps(
            {
                "intents": ["ambiguous"],
                "clarify": "Clean the values, merge duplicates, or both?",
                "options": ["Clean up the values", "Merge duplicates", "Both"],
            }
        )

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_chat)
    out = await asyncio.wait_for(handle(_ctx(), "tidy these up"), TIMEOUT)
    assert out["kind"] == "clarify"
    assert out["options"] == ["Clean up the values", "Merge duplicates", "Both"]


@pytest.mark.asyncio
async def test_clarify_falls_back_to_default_options(monkeypatch):
    """When the classifier offers no options (or none parse), the clarify still
    carries the generic action menu — every clarify is clickable."""

    async def fake_chat(key, system, user, **kwargs):
        return json.dumps({"intents": ["ambiguous"], "clarify": "What next?"})

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_chat)
    out = await asyncio.wait_for(handle(_ctx(), "help"), TIMEOUT)
    assert out["kind"] == "clarify"
    assert out["options"], "a fallback clarify must still offer clickable options"
    assert "Merge duplicate records" in out["options"]


@pytest.mark.asyncio
async def test_options_are_capped_and_cleaned(monkeypatch):
    """Classifier options are de-duped and capped at 4 (defensive against a
    runaway list)."""

    async def fake_chat(key, system, user, **kwargs):
        return json.dumps(
            {
                "intents": ["ambiguous"],
                "clarify": "Which?",
                "options": ["A", "A", "B", "C", "D", "E", "  ", "F"],
            }
        )

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_chat)
    out = await asyncio.wait_for(handle(_ctx(), "x"), TIMEOUT)
    assert out["options"] == ["A", "B", "C", "D"]



# --------------------------------------------------------------------------- #
# 5. Thread history (COG-131): per-user listing + retrieval
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_list_for_owner_get_and_title():
    store = InMemoryConversationStore()
    await store.append(
        "s1", "t1",
        [Turn(role="user", text="Clean up the manufacturer names please"),
         Turn(role="assistant", text="ok", kind="plan")],
        owner="user_a",
    )
    await store.append("s2", "t1", [Turn(role="user", text="merge dupes")], owner="user_a")

    listed = await store.list_for_owner("t1", "user_a")
    assert {c.session_id for c in listed} == {"s1", "s2"}
    # Title derived from the first user message; newest-first ordering.
    s1 = next(c for c in listed if c.session_id == "s1")
    assert s1.title.startswith("Clean up the manufacturer names")
    assert s1.turn_count == 2

    full = await store.get("s1", "t1", owner="user_a")
    assert full is not None and [t.text for t in full.turns][0].startswith("Clean up")


@pytest.mark.asyncio
async def test_history_is_scoped_per_owner():
    store = InMemoryConversationStore()
    await store.append("s1", "t1", [Turn(role="user", text="mine")], owner="user_a")
    await store.append("s2", "t1", [Turn(role="user", text="theirs")], owner="user_b")

    assert [c.session_id for c in await store.list_for_owner("t1", "user_a")] == ["s1"]
    # user_b cannot fetch user_a's thread (owner mismatch -> None, surfaced as 404).
    assert await store.get("s1", "t1", owner="user_b") is None


@pytest.mark.asyncio
async def test_ownerless_session_never_listed():
    """A demo (no-owner) session is persisted for in-session grounding but never
    appears in any user's history."""
    store = InMemoryConversationStore()
    await store.append("demo-sess", "t1", [Turn(role="user", text="hi")])  # no owner
    assert await store.list_for_owner("t1", "user_a") == []
    assert await store.list_for_owner("t1", "") == []


@pytest.mark.asyncio
async def test_planner_records_owner(monkeypatch):
    """handle(..., session={'owner': ...}) tags the thread so it lists for that
    user."""

    async def fake_chat(key, system, user, **kwargs):
        return json.dumps({"intents": ["ambiguous"], "clarify": "Which?"})

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_chat)
    out = await asyncio.wait_for(
        handle(_ctx(), "do something", session={"id": "sx", "owner": "user_z"}),
        TIMEOUT,
    )
    assert out["kind"] == "clarify"
    listed = await make_conversation_store().list_for_owner("t1", "user_z")
    assert [c.session_id for c in listed] == ["sx"]


# --- Route-level: per-user history endpoints ------------------------------- #
@pytest.fixture
def _history_client(monkeypatch):
    """A TestClient whose auth verifier maps two keys to two subjects, with no
    OpenRouter key so the agent returns a clarify (records a turn) sans LLM."""
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    from cograph_client.api.app import create_app
    from cograph_client.graph.client import NeptuneClient

    monkeypatch.setattr("cograph_client.auth.api_keys.settings.api_keys", "{}")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "cograph_client.api.routes.agent.settings.openrouter_api_key", "", raising=False
    )

    subjects = {"key-a": "user_a", "key-b": "user_b"}

    def verifier(key):
        subj = subjects.get(key)
        return AuthVerdict(tenants=["test-tenant"], subject=subj) if subj else None

    register_external_verifier(verifier)

    app = create_app()
    n = AsyncMock(spec=NeptuneClient)
    n.query.return_value = {"head": {"vars": []}, "results": {"bindings": []}}
    n.update.return_value = None
    app.state.neptune_client = n
    client = TestClient(app)
    yield client
    register_external_verifier(None)


def _send(client, key, sid, msg):
    return client.post(
        "/graphs/test-tenant/agent",
        json={"message": msg, "context": {"kg_name": "kg1"}, "session_id": sid},
        headers={"X-API-Key": key},
    )


def test_route_history_lists_and_scopes_per_user(_history_client):
    client = _history_client
    # user_a starts two threads; user_b one.
    assert _send(client, "key-a", "a1", "clean the names").status_code == 200
    assert _send(client, "key-a", "a2", "merge dupes").status_code == 200
    assert _send(client, "key-b", "b1", "enrich stuff").status_code == 200

    ra = client.get("/graphs/test-tenant/conversations", headers={"X-API-Key": "key-a"})
    assert ra.status_code == 200
    a_sessions = {c["session_id"] for c in ra.json()["conversations"]}
    assert a_sessions == {"a1", "a2"}  # only user_a's threads

    rb = client.get("/graphs/test-tenant/conversations", headers={"X-API-Key": "key-b"})
    assert {c["session_id"] for c in rb.json()["conversations"]} == {"b1"}


def test_route_get_thread_is_owner_scoped(_history_client):
    client = _history_client
    _send(client, "key-a", "a1", "clean the names")

    # Owner can fetch the full transcript.
    ok = client.get(
        "/graphs/test-tenant/conversations/a1", headers={"X-API-Key": "key-a"}
    )
    assert ok.status_code == 200
    body = ok.json()
    assert body["session_id"] == "a1"
    assert body["turns"] and body["turns"][0]["role"] == "user"
    assert body["title"].startswith("clean the names")

    # A different user cannot read it (404, not enumerable).
    denied = client.get(
        "/graphs/test-tenant/conversations/a1", headers={"X-API-Key": "key-b"}
    )
    assert denied.status_code == 404
