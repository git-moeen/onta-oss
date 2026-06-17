"""User-owned tenant management — list / add / remove the caller's tenants.

These are the *single* backend routes both the CLI and the web Explorer use to
manage tenants, so the two surfaces can never drift. The work itself (reading and
writing the user's tenant list on their identity profile) is delegated to a
registered ``TenantProvider``; cograph-oss ships none, so an OSS-only deployment
returns 501 here. The premium Clerk integration registers a provider.

Auth: the caller proves identity with their own ``X-API-Key`` — the same key used
everywhere else. The provider resolves key → user and operates on that user's
tenants; no identity-provider admin secret is ever required client-side.
"""

from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel, Field

from cograph_client.auth.api_keys import api_key_header
from cograph_client.auth.tenant_directory import (
    Tenant,
    TenantProvider,
    TenantProviderError,
    get_tenant_provider,
    validate_new_tenant,
)

router = APIRouter(prefix="/v1/me/tenants")


class TenantOut(BaseModel):
    id: str
    label: str


class TenantCreate(BaseModel):
    id: str = Field(..., description="Tenant slug (lowercase, 3–40 chars).")
    label: str = Field(..., description="Human-readable label.")


def _require_provider() -> TenantProvider:
    provider = get_tenant_provider()
    if provider is None:
        raise HTTPException(
            status_code=501,
            detail="Tenant management is not configured for this deployment.",
        )
    return provider


def _require_key(api_key: str | None) -> str:
    if not api_key:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return api_key


def _out(t: Tenant) -> TenantOut:
    return TenantOut(id=t.id, label=t.label)


@router.get("", response_model=list[TenantOut])
def list_tenants(api_key: str | None = Security(api_key_header)):
    provider = _require_provider()
    key = _require_key(api_key)
    try:
        return [_out(t) for t in provider.list_tenants(key)]
    except TenantProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)


@router.post("", response_model=TenantOut, status_code=201)
def add_tenant(body: TenantCreate, api_key: str | None = Security(api_key_header)):
    provider = _require_provider()
    key = _require_key(api_key)
    try:
        # Validate before touching the provider so bad input is a clean 400 and
        # the rules stay identical to the Explorer's (validate_new_tenant is the
        # shared source of truth; it raises TenantProviderError(400)).
        tenant_id, label = validate_new_tenant(body.id, body.label)
        return _out(provider.add_tenant(key, tenant_id, label))
    except TenantProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)


@router.delete("/{tenant_id}")
def remove_tenant(tenant_id: str, api_key: str | None = Security(api_key_header)):
    provider = _require_provider()
    key = _require_key(api_key)
    try:
        provider.remove_tenant(key, tenant_id)
    except TenantProviderError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)
    return {"removed": tenant_id}
