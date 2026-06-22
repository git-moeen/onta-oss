"""HTTP routes for the inferred data-normalization subsystem (OSS core).

Lifecycle: suggest → (human) confirm/reject → apply.

  POST /graphs/{tenant}/normalize/suggest?kg=&type=   run inference, persist
                                                       suggested rules, return ranked
  GET  /graphs/{tenant}/normalize/rules?kg=&status=    list stored rules
  POST /graphs/{tenant}/normalize/rules/{id}/confirm   status -> confirmed
  POST /graphs/{tenant}/normalize/rules/{id}/reject    status -> rejected
  POST /graphs/{tenant}/normalize/rules/{id}/apply     run apply_rule in the
                                                       background (confirmed only),
                                                       set status=applied on success

Apply uses the SAME strong-ref ``_spawn`` background-task pattern as enrich.py so
the task can't be garbage-collected after the request returns.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from cograph_client.api.deps import get_neptune_client
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.graph.client import NeptuneClient
from cograph_client.normalization.execute import apply_rule
from cograph_client.normalization.inference import suggest_rules
from cograph_client.normalization.rules import NormalizationRule, NormalizationRuleStore

logger = structlog.stdlib.get_logger("cograph.normalization.routes")

router = APIRouter(prefix="/graphs/{tenant}/normalize")


# Strong refs to background apply tasks. CPython only weakly references a bare
# create_task() result, so without this the apply could be GC'd at its first
# await right after the request returns (mirrors enrich.py / explore.py).
_bg_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def _store(client: NeptuneClient) -> NormalizationRuleStore:
    return NormalizationRuleStore(client)


@router.post("/suggest", response_model=list[NormalizationRule])
async def suggest(
    kg: str = Query(..., description="KG name to analyze"),
    type: str = Query(..., description="Type name to analyze"),
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Infer normalization rules for a type's predicates, persist them as
    ``suggested``, and return them ranked by confidence (desc)."""
    rules = await suggest_rules(client, tenant.tenant_id, kg, type)
    store = _store(client)
    for rule in rules:
        await store.save(tenant.tenant_id, rule)
    return rules


@router.get("/rules", response_model=list[NormalizationRule])
async def list_rules(
    kg: str | None = Query(None),
    status: str | None = Query(None),
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """List stored rules, optionally filtered by KG and/or status."""
    return await _store(client).list(tenant.tenant_id, kg=kg, status=status)


@router.post("/rules/{rule_id}/confirm", response_model=NormalizationRule)
async def confirm_rule(
    rule_id: str,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    rule = await _store(client).update_status(tenant.tenant_id, rule_id, "confirmed")
    if rule is None:
        raise HTTPException(status_code=404, detail="rule not found")
    return rule


@router.post("/rules/{rule_id}/reject", response_model=NormalizationRule)
async def reject_rule(
    rule_id: str,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    rule = await _store(client).update_status(tenant.tenant_id, rule_id, "rejected")
    if rule is None:
        raise HTTPException(status_code=404, detail="rule not found")
    return rule


@router.post("/rules/{rule_id}/apply", status_code=202)
async def apply_rule_route(
    rule_id: str,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Apply a confirmed rule in the background; ack immediately (202).

    Apply runs ONLY when the rule is ``confirmed`` (or already ``applied`` — a
    re-run is idempotent). ``suggested`` / ``rejected`` rules are refused.
    On success the rule's status flips to ``applied`` with ``applied_at`` set.
    """
    store = _store(client)
    rule = await store.get(tenant.tenant_id, rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="rule not found")
    if rule.status not in ("confirmed", "applied"):
        raise HTTPException(
            status_code=409,
            detail=f"rule must be confirmed before apply (status={rule.status})",
        )
    _spawn(_apply_and_mark(client, tenant.tenant_id, rule))
    return {"status": "accepted", "rule_id": rule_id}


async def _apply_and_mark(
    client: NeptuneClient, tenant_id: str, rule: NormalizationRule
) -> None:
    """Run apply_rule, then mark the rule applied. Errors are logged, not raised
    (this runs detached as a background task)."""
    try:
        summary = await apply_rule(client, tenant_id, rule)
        await NormalizationRuleStore(client).update_status(
            tenant_id,
            rule.id,
            "applied",
            applied_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("normalize_apply_done", rule_id=rule.id, **summary)
    except Exception:
        logger.error("normalize_apply_failed", rule_id=rule.id, exc_info=True)
