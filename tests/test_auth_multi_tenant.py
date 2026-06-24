"""User-scoped multi-tenant API keys.

A verifier may return a SEQUENCE of tenant ids (the user's owned tenants):
every key the user creates then works for all of them, authorized against
the tenant requested in the route path. Single-tenant verdicts (str) and
static keys keep the legacy behavior: they route to their own tenant
regardless of the path.
"""

import pytest
from fastapi import HTTPException

from cograph_client.auth.api_keys import (
    AuthVerdict,
    TenantContext,
    get_tenant,
    register_external_verifier,
)


@pytest.fixture(autouse=True)
def _clear_verifier():
    yield
    register_external_verifier(None)


@pytest.fixture
def open_access(monkeypatch):
    monkeypatch.setattr("cograph_client.auth.api_keys.settings.api_keys", "{}")


def test_multi_tenant_key_routes_to_requested_tenant(open_access):
    register_external_verifier(lambda key: ["alpha", "beta"])
    ctx = get_tenant(tenant="beta", api_key="k")
    assert ctx == TenantContext(tenant_id="beta", api_key="k")


def test_multi_tenant_key_defaults_to_first_without_path(open_access):
    register_external_verifier(lambda key: ["alpha", "beta"])
    ctx = get_tenant(tenant=None, api_key="k")
    assert ctx.tenant_id == "alpha"


def test_multi_tenant_key_rejects_unowned_tenant_with_403(open_access):
    register_external_verifier(lambda key: ["alpha", "beta"])
    with pytest.raises(HTTPException) as exc:
        get_tenant(tenant="other-tenant", api_key="k")
    assert exc.value.status_code == 403


def test_empty_allowed_list_is_invalid_key(open_access):
    register_external_verifier(lambda key: [])
    with pytest.raises(HTTPException) as exc:
        get_tenant(tenant="alpha", api_key="k")
    assert exc.value.status_code == 401


def test_legacy_single_tenant_str_ignores_path(open_access):
    """Back-compat: a str verdict routes to ITS tenant even if the path
    names another — exactly today's behavior for claims.tenant keys."""
    register_external_verifier(lambda key: "their-tenant")
    ctx = get_tenant(tenant="something-else", api_key="k")
    assert ctx.tenant_id == "their-tenant"


def test_unrecognized_key_still_401(open_access):
    register_external_verifier(lambda key: None)
    with pytest.raises(HTTPException) as exc:
        get_tenant(tenant="alpha", api_key="k")
    assert exc.value.status_code == 401


def test_open_access_honors_requested_tenant(open_access):
    ctx = get_tenant(tenant="my-local-tenant", api_key=None)
    assert ctx.tenant_id == "my-local-tenant"


def test_open_access_defaults_without_path(open_access):
    ctx = get_tenant(tenant=None, api_key=None)
    assert ctx.tenant_id == "default"


def test_auth_verdict_carries_subject_to_context(open_access):
    """A verifier may return an AuthVerdict (tenants + subject); the subject is
    threaded onto TenantContext so per-user resources can scope by it."""
    register_external_verifier(
        lambda key: AuthVerdict(tenants=["alpha", "beta"], subject="user_123")
    )
    ctx = get_tenant(tenant="beta", api_key="k")
    assert ctx.tenant_id == "beta"
    assert ctx.subject == "user_123"


def test_auth_verdict_authorizes_path_tenant(open_access):
    """AuthVerdict tenants are authorized against the path like a sequence."""
    register_external_verifier(
        lambda key: AuthVerdict(tenants=["alpha"], subject="u1")
    )
    with pytest.raises(HTTPException) as exc:
        get_tenant(tenant="not-owned", api_key="k")
    assert exc.value.status_code == 403


def test_auth_verdict_without_subject_is_none(open_access):
    register_external_verifier(lambda key: AuthVerdict(tenants=["alpha"]))
    ctx = get_tenant(tenant="alpha", api_key="k")
    assert ctx.subject is None


def test_sequence_verdict_has_no_subject(open_access):
    """A bare sequence verdict (legacy) yields subject=None."""
    register_external_verifier(lambda key: ["alpha"])
    ctx = get_tenant(tenant="alpha", api_key="k")
    assert ctx.subject is None


def test_static_key_keeps_legacy_routing(monkeypatch):
    monkeypatch.setattr(
        "cograph_client.auth.api_keys.settings.api_keys",
        '{"static-key": "static-tenant"}',
    )
    ctx = get_tenant(tenant="another", api_key="static-key")
    assert ctx.tenant_id == "static-tenant"
