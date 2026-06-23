"""Contract tests for the MCP ``agent`` tool's backend surface (COG-125).

The OSS MCP server (``packages/cograph-mcp``) is a thin TypeScript client over
the HTTP API: its ``agent`` tool calls ``POST /graphs/{tenant}/agent`` via the
``cograph`` SDK. These tests pin the HTTP contract the tool depends on, through
the FastAPI ``TestClient`` (the same path the SDK hits), with the planner stubbed
so the suite is deterministic and offline:

  * a message returns a PLAN (action intent),
  * a message returns an ANSWER (question intent),
  * a ``confirm_plan_id`` routes to execute → RESULT (the only mutating path),
  * the agent endpoint is mounted/reachable (the tool's target exists), and
  * paid execution RESPECTS ENTITLEMENT — the agent route authorizes the tenant
    with the SAME ``get_tenant`` dependency the direct paid routes use, so a
    confirm for an ungranted tenant 403s exactly like the direct ``/enrich``
    path. There is no OSS-side entitlement gate the agent could bypass (gating
    is proprietary, enforced behind the API); these tests prove the agent path
    is authorized identically to the direct path.

The MCP tool itself is exercised by TypeScript typecheck + build in CI's npm
job; the request/response *contract* it relies on is locked here.
"""

from __future__ import annotations

import os

import pytest
from fastapi import HTTPException

# Mirror the test.yml env so settings/auth construct the same way as in CI.
os.environ.setdefault("OMNIX_API_KEYS", '{"test-key": "test-tenant"}')
os.environ.setdefault("OMNIX_NEPTUNE_ENDPOINT", "http://fake:8182")

from unittest.mock import AsyncMock  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from cograph_client.agent import planner as planner_mod  # noqa: E402
from cograph_client.agent.planner import (  # noqa: E402
    register_default_capabilities,
    reset_plan_store,
)
from cograph_client.agent.registry import (  # noqa: E402
    reset_capabilities,
)
from cograph_client.api.app import create_app  # noqa: E402
from cograph_client.auth.api_keys import (  # noqa: E402
    TenantContext,
    get_tenant,
    register_external_verifier,
)
from cograph_client.graph.client import NeptuneClient  # noqa: E402

HEADERS = {"X-API-Key": "test-key"}
TENANT = "test-tenant"


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Each test starts from the default OSS capability set + empty plan store."""
    reset_capabilities()
    reset_plan_store()
    register_default_capabilities()
    yield
    reset_capabilities()
    reset_plan_store()
    register_external_verifier(None)


@pytest.fixture
def app_client(monkeypatch):
    """A TestClient with a mocked Neptune (no live graph) — the SDK/MCP target."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")
    app = create_app()
    n = AsyncMock(spec=NeptuneClient)
    n.query.return_value = {"head": {"vars": []}, "results": {"bindings": []}}
    n.update.return_value = None
    app.state.neptune_client = n
    return TestClient(app)


def _stub_classifier(monkeypatch, intent: str, clarify: str = ""):
    async def fake_chat(*args, **kwargs):
        import json

        return json.dumps({"intent": intent, "clarify": clarify})

    monkeypatch.setattr(planner_mod, "openrouter_chat", fake_chat)


def _stub_schema(monkeypatch):
    async def fake_schema(neptune, tenant_id, type_name):
        return {
            "attributes": ["title", "skills"],
            "relationships": [{"name": "speaks", "target_type": "Language"}],
        }

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.list_type_schema", fake_schema
    )


def _stub_enrich_extract(monkeypatch, payload: dict):
    import json

    async def fake_chat(*args, **kwargs):
        return json.dumps(payload)

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.openrouter_chat", fake_chat
    )


def _agent_body(message: str = "", *, confirm_plan_id: str | None = None) -> dict:
    """Build the body the MCP ``agent`` tool sends (message + context, or confirm).

    Mirrors ``packages/cograph/src/client.ts`` ``Client.agent`` exactly: a turn is
    either a new ``message`` with ``context`` or a ``confirm`` of a plan_id.
    """
    body: dict = {
        "message": message,
        "context": {"kg_name": "kg1", "type_name": "Mentor"},
    }
    if confirm_plan_id is not None:
        body["confirm"] = {"plan_id": confirm_plan_id}
    return body


# --------------------------------------------------------------------------- #
# 1. A message returns a PLAN (action intent).
# --------------------------------------------------------------------------- #
def test_agent_tool_message_returns_plan(app_client, monkeypatch):
    _stub_classifier(monkeypatch, "enrich")
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch, {"attributes": ["company"], "scope": None, "tier": "core"}
    )

    async def fake_sample(*a, **k):
        return (["English"], "relationship")

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.sample_predicate_values",
        fake_sample,
    )

    r = app_client.post(
        f"/graphs/{TENANT}/agent",
        json=_agent_body("enrich the company for managers"),
        headers=HEADERS,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "plan"
    assert body["plan_id"]
    assert body["steps"][0]["capability"] == "enrich"


# --------------------------------------------------------------------------- #
# 2. A message returns an ANSWER (question intent).
# --------------------------------------------------------------------------- #
def test_agent_tool_question_returns_answer(app_client, monkeypatch):
    _stub_classifier(monkeypatch, "question")

    from cograph_client.agent.capabilities.query import QueryCapability

    async def fake_answer(self, ctx, q):
        return {"answer": "42", "sparql": "SELECT ...", "rows": [], "narrative": ""}

    monkeypatch.setattr(QueryCapability, "answer", fake_answer)

    r = app_client.post(
        f"/graphs/{TENANT}/agent",
        json=_agent_body("how many mentors are there?"),
        headers=HEADERS,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "answer"
    assert body["answer"] == "42"


# --------------------------------------------------------------------------- #
# 3. A confirm routes to EXECUTE → RESULT (the only mutating path).
# --------------------------------------------------------------------------- #
def test_agent_tool_confirm_routes_to_execute(app_client, monkeypatch):
    _stub_classifier(monkeypatch, "enrich")
    _stub_schema(monkeypatch)
    _stub_enrich_extract(
        monkeypatch, {"attributes": ["company"], "scope": None, "tier": "core"}
    )

    async def fake_sample(*a, **k):
        return (["English"], "relationship")

    monkeypatch.setattr(
        "cograph_client.agent.capabilities.enrich_cap.sample_predicate_values",
        fake_sample,
    )

    r1 = app_client.post(
        f"/graphs/{TENANT}/agent",
        json=_agent_body("enrich the company for managers"),
        headers=HEADERS,
    )
    assert r1.status_code == 200, r1.text
    plan_id = r1.json()["plan_id"]

    # The confirm turn (MCP tool: confirm_plan_id set) → execute_plan → result.
    r2 = app_client.post(
        f"/graphs/{TENANT}/agent",
        json=_agent_body(confirm_plan_id=plan_id),
        headers=HEADERS,
    )
    assert r2.status_code == 200, r2.text
    result = r2.json()
    assert result["kind"] == "result"
    assert all(s["status"] == "ok" for s in result["steps"])


# --------------------------------------------------------------------------- #
# 4. The agent endpoint the tool targets is mounted/reachable ("advertised").
#    The MCP ``agent`` tool is advertised unconditionally; this proves its
#    backend target exists (a POST returns something other than 404 Not Found).
# --------------------------------------------------------------------------- #
def test_agent_endpoint_is_mounted(app_client, monkeypatch):
    _stub_classifier(monkeypatch, "ambiguous", clarify="What would you like to do?")
    r = app_client.post(
        f"/graphs/{TENANT}/agent",
        json=_agent_body("hello"),
        headers=HEADERS,
    )
    # Reachable + dispatched (not a 404/405 from a missing route).
    assert r.status_code == 200, r.text
    assert r.json()["kind"] == "clarify"


# --------------------------------------------------------------------------- #
# 5. Paid execution RESPECTS ENTITLEMENT: the agent route authorizes the tenant
#    with the SAME get_tenant dependency the direct paid routes use, so the agent
#    path can't bypass a gate the direct path enforces.
# --------------------------------------------------------------------------- #
def test_agent_confirm_for_ungranted_tenant_is_403(monkeypatch):
    """A multi-tenant key that does NOT own the requested tenant is rejected 403
    on the agent confirm — identical to the direct /enrich path, because both go
    through get_tenant. (Entitlement of paid steps lives behind the API; the OSS
    layer only proves the tenant is authorized, the same for both paths.)"""
    monkeypatch.setattr("cograph_client.auth.api_keys.settings.api_keys", "{}")
    register_external_verifier(lambda key: ["alpha", "beta"])

    app = create_app()
    client = TestClient(app, raise_server_exceptions=False)

    # Confirm a (would-be paid) plan against a tenant the key does NOT own.
    r = client.post(
        "/graphs/not-owned/agent",
        json={
            "message": "",
            "context": {"kg_name": "kg1", "type_name": "Mentor"},
            "confirm": {"plan_id": "any-plan"},
        },
        headers={"X-API-Key": "k"},
    )
    assert r.status_code == 403, r.text


def test_agent_route_uses_same_tenant_dep_as_direct_path():
    """Lock the safeguard structurally: the agent route depends on the very same
    ``get_tenant`` callable the direct enrich route depends on — so paid work
    routed through the agent is authorized identically and cannot bypass it."""
    from cograph_client.api.routes import agent as agent_route
    from cograph_client.api.routes import enrich as enrich_route

    def _tenant_deps(module):
        deps = set()
        for route in module.router.routes:
            for dep in getattr(getattr(route, "dependant", None), "dependencies", []):
                if dep.call is get_tenant:
                    deps.add(dep.call)
        return deps

    # Both routers wire get_tenant; same callable object → same authorization.
    assert get_tenant in _tenant_deps(agent_route)
    assert get_tenant in _tenant_deps(enrich_route)


def test_get_tenant_grants_owned_tenant(monkeypatch):
    """Sanity: the same dependency DOES grant an owned tenant (no false 403)."""
    monkeypatch.setattr("cograph_client.auth.api_keys.settings.api_keys", "{}")
    register_external_verifier(lambda key: ["alpha", "beta"])
    try:
        ctx = get_tenant(tenant="alpha", api_key="k")
        assert ctx == TenantContext(tenant_id="alpha", api_key="k")
        with pytest.raises(HTTPException) as exc:
            get_tenant(tenant="gamma", api_key="k")
        assert exc.value.status_code == 403
    finally:
        register_external_verifier(None)
