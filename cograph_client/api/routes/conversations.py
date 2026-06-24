"""Ask-AI conversation history (COG-131).

Read-only endpoints that let a signed-in user list their past Ask-AI threads and
re-open one. Threads are persisted by the agent as it runs (see
:mod:`cograph_client.agent.conversation_store`); these routes only read them.

Scoping is per-user: a thread is owned by the auth *subject* (the user id behind
the API key, surfaced generically as :attr:`TenantContext.subject`). Listing is
therefore available only to a request that carries a subject — the public demo's
shared static key has none, so it gets an empty list and keeps its single-thread
behavior. A thread can only be fetched by its owner.

There is no write endpoint here: the agent endpoint
(``POST /graphs/{tenant}/agent``) remains the single conversational surface that
creates/updates a thread. This mirrors the interface-convergence rule — one
canonical route per operation (CLAUDE.md / COG-128).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from cograph_client.agent.conversation_store import make_conversation_store
from cograph_client.auth.api_keys import TenantContext, get_tenant

router = APIRouter(prefix="/graphs/{tenant}/conversations")


@router.get("")
async def list_conversations(tenant: TenantContext = Depends(get_tenant)) -> dict:
    """List the caller's threads for this tenant, newest-first.

    Returns ``{"conversations": []}`` for a request without a subject (the demo
    shared key) — history is an authenticated, per-user feature.
    """
    if not tenant.subject:
        return {"conversations": []}
    store = make_conversation_store()
    summaries = await store.list_for_owner(tenant.tenant_id, tenant.subject)
    return {"conversations": [s.to_dict() for s in summaries]}


@router.get("/{session_id}")
async def get_conversation(
    session_id: str, tenant: TenantContext = Depends(get_tenant)
) -> dict:
    """Return one thread's full transcript so the UI can re-render it.

    Scoped to the caller's subject: a user can only open their own thread (a
    mismatch is a 404, not a 403, so thread ids aren't enumerable).
    """
    store = make_conversation_store()
    convo = await store.get(session_id, tenant.tenant_id, owner=tenant.subject)
    if convo is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {
        "session_id": convo.session_id,
        "title": convo.title,
        "created_at": convo.created_at.isoformat() if convo.created_at else None,
        "updated_at": convo.updated_at.isoformat() if convo.updated_at else None,
        "turns": [t.to_dict() for t in convo.turns],
    }
