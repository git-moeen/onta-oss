"""The /v1/me/tenants routes and the tenant-provider plugin protocol.

Exercises the OSS surface only: a fake in-memory provider stands in for the
premium Clerk integration, the same way the auth tests use a fake verifier.
"""

import pytest
from fastapi.testclient import TestClient

from cograph_client.api.app import create_app
from cograph_client.auth.tenant_directory import (
    Tenant,
    TenantProviderError,
    register_tenant_provider,
    validate_new_tenant,
)


class FakeProvider:
    """In-memory tenant directory keyed by api_key → list[Tenant]."""

    def __init__(self):
        self.store: dict[str, list[Tenant]] = {"good-key": []}

    def _user(self, api_key: str) -> list[Tenant]:
        if api_key not in self.store:
            raise TenantProviderError(401, "Invalid API key")
        return self.store[api_key]

    def list_tenants(self, api_key):
        return list(self._user(api_key))

    def add_tenant(self, api_key, tenant_id, label):
        owned = self._user(api_key)
        if any(t.id == tenant_id for t in owned):
            raise TenantProviderError(409, f'Tenant "{tenant_id}" already exists.')
        t = Tenant(id=tenant_id, label=label)
        owned.append(t)
        return t

    def remove_tenant(self, api_key, tenant_id):
        owned = self._user(api_key)
        if not any(t.id == tenant_id for t in owned):
            raise TenantProviderError(404, f'Tenant "{tenant_id}" not found.')
        self.store[api_key] = [t for t in owned if t.id != tenant_id]


@pytest.fixture
def app():
    # Open access (no static keys) so get_tenant isn't in play; these routes
    # authenticate via the provider, not the path-tenant dependency.
    import os

    os.environ["OMNIX_API_KEYS"] = "{}"
    return create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_provider():
    yield
    register_tenant_provider(None)


# --- routes with a provider registered ---------------------------------------


def test_add_list_remove_roundtrip(client):
    register_tenant_provider(FakeProvider())
    h = {"X-API-Key": "good-key"}

    assert client.get("/v1/me/tenants", headers=h).json() == []

    r = client.post(
        "/v1/me/tenants", headers=h, json={"id": "acme-co", "label": "Acme"}
    )
    assert r.status_code == 201
    assert r.json() == {"id": "acme-co", "label": "Acme"}

    assert client.get("/v1/me/tenants", headers=h).json() == [
        {"id": "acme-co", "label": "Acme"}
    ]

    r = client.delete("/v1/me/tenants/acme-co", headers=h)
    assert r.status_code == 200
    assert r.json() == {"removed": "acme-co"}
    assert client.get("/v1/me/tenants", headers=h).json() == []


def test_duplicate_add_is_409(client):
    register_tenant_provider(FakeProvider())
    h = {"X-API-Key": "good-key"}
    client.post("/v1/me/tenants", headers=h, json={"id": "acme-co", "label": "A"})
    r = client.post("/v1/me/tenants", headers=h, json={"id": "acme-co", "label": "A"})
    assert r.status_code == 409


def test_remove_unknown_is_404(client):
    register_tenant_provider(FakeProvider())
    r = client.delete("/v1/me/tenants/nope", headers={"X-API-Key": "good-key"})
    assert r.status_code == 404


def test_invalid_key_is_401(client):
    register_tenant_provider(FakeProvider())
    r = client.get("/v1/me/tenants", headers={"X-API-Key": "bogus"})
    assert r.status_code == 401


def test_missing_key_is_401(client):
    register_tenant_provider(FakeProvider())
    assert client.get("/v1/me/tenants").status_code == 401


@pytest.mark.parametrize(
    "tid,label",
    [
        ("UPPER", "x"),  # uppercase
        ("ab", "x"),  # too short
        ("a" * 41, "x"),  # too long
        ("demo-tenant", "x"),  # reserved
        ("acme-co", ""),  # empty label
        ("acme-co", "y" * 65),  # label too long
    ],
)
def test_invalid_input_is_400_before_provider(client, tid, label):
    register_tenant_provider(FakeProvider())
    r = client.post(
        "/v1/me/tenants", headers={"X-API-Key": "good-key"}, json={"id": tid, "label": label}
    )
    assert r.status_code == 400


# --- no provider registered (OSS-only deployment) ----------------------------


def test_no_provider_is_501(client):
    r = client.get("/v1/me/tenants", headers={"X-API-Key": "good-key"})
    assert r.status_code == 501


# --- shared validation helper ------------------------------------------------


def test_validate_new_tenant_trims_and_returns():
    assert validate_new_tenant("  acme-co ", "  Acme  ") == ("acme-co", "Acme")


def test_validate_new_tenant_rejects_reserved():
    with pytest.raises(TenantProviderError) as exc:
        validate_new_tenant("spider-bench", "x")
    assert exc.value.status_code == 400
