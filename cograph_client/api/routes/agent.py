"""THE single conversational surface for the unified Ask-AI agent (COG-118).

``POST /graphs/{tenant}/agent`` is the ONLY endpoint. Everything the agent can do
is a capability behind the registry — there is no per-task conversational
endpoint. The legacy ``/ask``, ``/enrich/*`` and ``/normalize/*`` routes stay for
back-compat (existing dialogs), but the agent does NOT call them: it drives the
underlying engines directly through the capability registry.

Request/response contract:

  POST body {message, context:{kg_name, type_name, selection?}, session_id?,
             confirm?:{plan_id}}
    - confirm.plan_id present → execute_plan → {kind:"result", steps:[...]}
      (execute is the only mutating path; long work runs as background jobs)
    - else → planner.handle → {kind: "answer"|"clarify"|"plan"}

Capabilities are registered at import time via
:func:`cograph_client.agent.planner.register_default_capabilities` (also invoked
from ``app.py`` at startup, import-safe + idempotent).
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from cograph_client.agent import planner
from cograph_client.agent.planner import register_default_capabilities
from cograph_client.agent.registry import AgentContext
from cograph_client.api.deps import (
    get_enrichment_job_store,
    get_executor,
    get_neptune_client,
)
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.config import settings
from cograph_client.enrichment.executor import EnrichmentExecutor
from cograph_client.graph.client import NeptuneClient

router = APIRouter(prefix="/graphs/{tenant}/agent")

# Register the default OSS capabilities on import so the single endpoint works
# even if app.py's explicit startup call is bypassed (e.g. a test mounting only
# this router). Idempotent — last-write-wins.
register_default_capabilities()


class AgentRequestContext(BaseModel):
    kg_name: str = ""
    type_name: str | None = None
    selection: dict | None = None


class Confirm(BaseModel):
    plan_id: str


class AgentRequest(BaseModel):
    message: str = Field("", description="The user's message to the agent")
    context: AgentRequestContext = Field(default_factory=AgentRequestContext)
    session_id: str | None = None
    confirm: Confirm | None = None


def _build_ctx(
    tenant: TenantContext,
    body: AgentRequest,
    client: NeptuneClient,
    executor: EnrichmentExecutor,
    job_store,
) -> AgentContext:
    return AgentContext(
        tenant_id=tenant.tenant_id,
        kg_name=body.context.kg_name,
        neptune=client,
        type_name=body.context.type_name,
        selection=body.context.selection,
        openrouter_key=settings.openrouter_api_key
        or os.environ.get("OPENROUTER_API_KEY", ""),
        anthropic_key=settings.anthropic_api_key,
        extras={
            "enrichment_executor": executor,
            "enrichment_job_store": job_store,
        },
    )


@router.post("")
async def agent_turn(
    body: AgentRequest,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
    executor: EnrichmentExecutor = Depends(get_executor),
    job_store=Depends(get_enrichment_job_store),
):
    """One agent turn: confirm→execute a plan, or classify+respond to a message."""
    ctx = _build_ctx(tenant, body, client, executor, job_store)
    if body.confirm is not None:
        return await planner.execute_plan(ctx, body.confirm.plan_id)
    return await planner.handle(ctx, body.message, session={"id": body.session_id})
