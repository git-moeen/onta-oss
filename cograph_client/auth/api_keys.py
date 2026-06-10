import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from cograph_client.config import settings

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class TenantContext:
    tenant_id: str
    api_key: str


# A verifier takes a raw API key and returns either:
#   - a single tenant_id (legacy single-tenant keys), or
#   - a sequence of tenant_ids the key may access (user-scoped keys: a user
#     owns N tenants and every key they create works for all of them), or
#   - None if the key is not recognized.
# Implementations are expected to fail closed (return None) on network or
# timeout errors rather than raising — raising would turn an auth provider
# outage into a 500.
ExternalVerifier = Callable[[str], Optional[Union[str, Sequence[str]]]]

_external_verifier: Optional[ExternalVerifier] = None


def register_external_verifier(verifier: Optional[ExternalVerifier]) -> None:
    """Register (or clear) an external API key verifier.

    Downstream deployments can use this to plug in a third-party auth
    provider (Clerk, WorkOS, a custom keystore, etc.) without forking
    omnix-oss. Pass None to clear.
    """
    global _external_verifier
    _external_verifier = verifier


def _resolve_allowed(
    allowed: Sequence[str], requested: Optional[str], api_key: str
) -> TenantContext:
    """Pick the tenant for a key that may access several.

    The requested tenant comes from the route path (/graphs/{tenant}/...).
    No request → the key's first tenant; a request outside the allowed set
    is a 403 (the key is valid, the tenant grant is not).
    """
    allowed = [t for t in allowed if t]
    if not allowed:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if requested is None or requested == "":
        return TenantContext(tenant_id=allowed[0], api_key=api_key)
    if requested in allowed:
        return TenantContext(tenant_id=requested, api_key=api_key)
    raise HTTPException(
        status_code=403,
        detail=f"API key does not grant access to tenant '{requested}'",
    )


def get_tenant(
    tenant: Optional[str] = None,
    api_key: Optional[str] = Security(api_key_header),
) -> TenantContext:
    """Resolve the tenant for a request.

    `tenant` is injected from the route path (/graphs/{tenant}/...) when
    present. Single-tenant keys (static map, legacy claims.tenant) keep
    today's behavior: they route to THEIR tenant regardless of the path.
    Multi-tenant keys (verifier returned a sequence) are authorized against
    the requested path tenant.
    """
    keys_map = settings.get_api_keys_map()
    has_static_keys = bool(keys_map) and keys_map != {"": ""}
    has_external = _external_verifier is not None

    # No auth configured at all — open access; honor the requested tenant.
    if not has_static_keys and not has_external:
        return TenantContext(tenant_id=tenant or "default", api_key="")

    if not api_key:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Static keys take precedence: cheap dict lookup, no network round-trip.
    if has_static_keys:
        tenant_id = keys_map.get(api_key)
        if tenant_id is not None:
            return TenantContext(tenant_id=tenant_id, api_key=api_key)

    # Fall back to the external verifier, if one is registered.
    if has_external:
        try:
            verdict = _external_verifier(api_key)  # type: ignore[misc]
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("external verifier raised: %s", exc)
            verdict = None
        if isinstance(verdict, str):
            return TenantContext(tenant_id=verdict, api_key=api_key)
        if verdict is not None:
            return _resolve_allowed(verdict, tenant, api_key)

    raise HTTPException(status_code=401, detail="Invalid API key")
