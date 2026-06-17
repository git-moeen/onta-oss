"""Tenant directory plugin protocol — user-owned tenant management.

Tenants belong to USERS (see auth/api_keys.py): a user owns N tenants and every
API key they create works for all of them. *Reading and mutating* that ownership
list (list/add/remove tenants) is identity-provider specific — it lives in the
user's Clerk/WorkOS/... profile — so cograph-oss does not implement it directly.
Instead a deployment registers a provider here, exactly as it registers an API
key verifier via ``register_external_verifier``. The premium Clerk integration
(``cograph.auth.clerk``) registers one; without a provider the ``/v1/me/tenants``
routes report 501.

The provider authenticates the caller from their own API key (the same key used
for ``X-API-Key`` auth) — no admin/identity-provider secret ever leaves the
backend. This is what lets the CLI and the Explorer manage tenants over one
shared backend route instead of each holding the Clerk secret.

Validation rules (slug shape, reserved ids, label length) are product rules, not
provider specifics, so they live here as the single source of truth shared by the
route and any caller. Keep ``TENANT_ID_RE``/``RESERVED_TENANT_IDS`` in sync with
the web Explorer's client-side preview (TenantSwitcher.tsx).
"""

import re
from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

# Slug rule: lowercase alphanumeric + interior dashes, 3–40 chars.
TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,38}[a-z0-9]$")

# Ids that must never be self-served: shared/env tenants, the backend fallback
# tenant, and the disposable benchmark tenant.
RESERVED_TENANT_IDS = frozenset(
    {"demo-tenant", "hotel-design-partner", "default", "spider-bench"}
)

MAX_LABEL_LEN = 64


@dataclass
class Tenant:
    id: str
    label: str


class TenantProviderError(Exception):
    """A client-facing failure from a provider, carrying an HTTP status.

    Providers raise this for auth/conflict/not-found conditions so the route can
    translate them into the right status without knowing provider internals.
    """

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


@runtime_checkable
class TenantProvider(Protocol):
    """Manages the caller's owned tenants, authenticated by their API key.

    Implementations should raise ``TenantProviderError`` for caller-facing
    failures (401 invalid key, 404 unknown tenant, 409 already exists) and fail
    closed on identity-provider outages.
    """

    def list_tenants(self, api_key: str) -> list[Tenant]: ...

    def add_tenant(self, api_key: str, tenant_id: str, label: str) -> Tenant: ...

    def remove_tenant(self, api_key: str, tenant_id: str) -> None: ...


_provider: Optional[TenantProvider] = None


def register_tenant_provider(provider: Optional[TenantProvider]) -> None:
    """Register (or clear) the tenant directory provider. Pass None to clear."""
    global _provider
    _provider = provider


def get_tenant_provider() -> Optional[TenantProvider]:
    return _provider


def validate_new_tenant(tenant_id: str, label: str) -> tuple[str, str]:
    """Validate + normalize a tenant id/label for creation.

    Returns the trimmed (id, label). Raises ``TenantProviderError(400, ...)`` on
    a bad slug, a reserved id, or a missing/over-long label — mirroring the web
    Explorer's createTenant checks so both surfaces enforce identical rules.
    """
    tid = tenant_id.strip()
    lbl = label.strip()
    if not TENANT_ID_RE.match(tid):
        raise TenantProviderError(
            400,
            "Tenant id must be 3–40 characters: lowercase letters, numbers, "
            "and interior dashes.",
        )
    if tid in RESERVED_TENANT_IDS:
        raise TenantProviderError(400, f'"{tid}" is reserved.')
    if not lbl:
        raise TenantProviderError(400, "Label is required.")
    if len(lbl) > MAX_LABEL_LEN:
        raise TenantProviderError(
            400, f"Label must be {MAX_LABEL_LEN} characters or fewer."
        )
    return tid, lbl
