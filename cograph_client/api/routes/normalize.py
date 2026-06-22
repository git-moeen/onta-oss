"""HTTP routes for the inferred data-normalization subsystem (OSS core).

Lifecycle: suggest → (human) confirm/reject → apply.

  POST /graphs/{tenant}/normalize/suggest?kg=&type=   run inference, persist
                                                       suggested rules, return ranked
  POST /graphs/{tenant}/normalize/rules                create a USER-AUTHORED rule
                                                       directly (no inference); upsert
  GET  /graphs/{tenant}/normalize/rules?kg=&status=    list stored rules
  POST /graphs/{tenant}/normalize/rules/{id}/confirm   status -> confirmed
  POST /graphs/{tenant}/normalize/rules/{id}/reject    status -> rejected
  POST /graphs/{tenant}/normalize/rules/{id}/apply     run apply_rule in the
                                                       background (confirmed only),
                                                       set status=applied on success

The user-authored create path complements inference: for sparse issues (e.g. an
emoji in ~0.4% of values) random-sample inference is unreliable, and a user may
simply KNOW they want a rule. It shares ids with inferred rules of the same
(kg, type, predicate, rule_type) via :func:`make_rule_id`, so creating one that
already exists UPSERTs rather than duplicating.

Apply uses the SAME strong-ref ``_spawn`` background-task pattern as enrich.py so
the task can't be garbage-collected after the request returns.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from cograph_client.api.deps import get_neptune_client
from cograph_client.auth.api_keys import TenantContext, get_tenant
from cograph_client.graph.client import NeptuneClient
from cograph_client.normalization.execute import apply_rule
from cograph_client.normalization.inference import suggest_rules
from cograph_client.normalization.rules import (
    NormalizationRule,
    NormalizationRuleStore,
    make_rule_id,
)

# Rule types a user may author directly. Mirrors inference's
# ``_SUPPORTED_RULE_TYPES`` (kept local so the create route doesn't depend on an
# inference internal): the two normalizations the executor knows how to apply.
_SUPPORTED_RULE_TYPES = {"list_explode", "strip_emoji"}
_VALID_LIST_EXPLODE_TARGETS = {"entity", "literal"}

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


class CreateRuleRequest(BaseModel):
    """Body for the user-authored rule create endpoint.

    A user supplies the full rule shape directly (no inference). ``params`` is
    rule-type-specific (see :class:`NormalizationRule`); validation below enforces
    the per-type minimums so a malformed rule never reaches the store.
    ``status`` defaults to ``"suggested"`` (matching inferred rules) but may be
    ``"confirmed"`` so the rule can be applied without a separate confirm step.
    """

    kg_name: str
    type_name: str
    predicate: str
    target_kind: Literal["attribute", "relationship"]
    rule_type: str
    params: dict = Field(default_factory=dict)
    confidence: float = 1.0
    rationale: str = "user-authored"
    status: Literal["suggested", "confirmed"] = "suggested"


def _validate_create_request(req: CreateRuleRequest) -> None:
    """Reject malformed create requests with a 422. Mirrors the per-rule-type
    invariants inference applies, but enforced as hard validation since there is
    no LLM to fall back on defaults for a user-authored rule."""
    if not req.kg_name.strip():
        raise HTTPException(status_code=422, detail="kg_name must be non-empty")
    if not req.type_name.strip():
        raise HTTPException(status_code=422, detail="type_name must be non-empty")
    if not req.predicate.strip():
        raise HTTPException(status_code=422, detail="predicate must be non-empty")
    if req.rule_type not in _SUPPORTED_RULE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"unknown rule_type {req.rule_type!r}; "
                f"supported: {sorted(_SUPPORTED_RULE_TYPES)}"
            ),
        )
    if req.rule_type == "list_explode":
        delimiters = req.params.get("delimiters")
        if not isinstance(delimiters, list) or not delimiters:
            raise HTTPException(
                status_code=422,
                detail="list_explode requires params.delimiters (non-empty list)",
            )
        target = req.params.get("target")
        if target not in _VALID_LIST_EXPLODE_TARGETS:
            raise HTTPException(
                status_code=422,
                detail=(
                    "list_explode requires params.target in "
                    f"{sorted(_VALID_LIST_EXPLODE_TARGETS)}"
                ),
            )
    # strip_emoji: params may be empty — the executor defaults targets to
    # attribute literals, so no required keys to validate.


@router.post("/rules", response_model=NormalizationRule)
async def create_rule(
    req: CreateRuleRequest,
    tenant: TenantContext = Depends(get_tenant),
    client: NeptuneClient = Depends(get_neptune_client),
):
    """Create a USER-AUTHORED normalization rule directly (no inference).

    The id is derived from ``(kg, type, predicate, rule_type)`` via
    :func:`make_rule_id`, so it shares ids with inferred rules of the same shape:
    creating one whose id already exists UPSERTs (the store clears prior triples
    before re-writing), never duplicates. ``created_at`` is stamped by the
    :class:`NormalizationRule` model's default_factory. Persists with the
    requested ``status`` and returns the persisted rule.
    """
    _validate_create_request(req)
    rule = NormalizationRule(
        id=make_rule_id(req.kg_name, req.type_name, req.predicate, req.rule_type),
        kg_name=req.kg_name,
        type_name=req.type_name,
        predicate=req.predicate,
        target_kind=req.target_kind,
        rule_type=req.rule_type,
        params=req.params,
        confidence=req.confidence,
        rationale=req.rationale,
        status=req.status,
    )
    await _store(client).save(tenant.tenant_id, rule)
    return rule


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
