"""Cleanup capability — propose + apply normalization rules through the agent.

Reuses the existing normalization engine end-to-end (no reimplementation):

* ``plan`` → :func:`cograph_client.normalization.inference.suggest_rules_for_predicates`
  (the TARGETED variant: only the predicate(s) the instruction names, never a
  whole-type scan) and produces one :class:`PlanStep` per inferred rule with a
  **dry-run preview** computed in-memory from sampled values (no writes).
* ``execute`` → persists the (confirmed) rule via
  :class:`cograph_client.normalization.rules.NormalizationRuleStore` and applies
  it via :func:`cograph_client.normalization.execute.apply_rule`, as a background
  job using the same strong-ref ``_spawn`` pattern as ``enrich.py`` /
  ``normalize.py`` (so the task can't be GC'd after the request returns).

The agent never calls the ``/normalize/*`` HTTP routes — it drives the same
engine functions directly.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone

import structlog

from cograph_client.agent.registry import AgentContext, PlanStep
from cograph_client.normalization.execute import apply_rule
from cograph_client.normalization.inference import suggest_rules_for_predicates
from cograph_client.normalization.rules import NormalizationRule, NormalizationRuleStore

logger = structlog.stdlib.get_logger("cograph.agent.normalize")

# Strong refs to background apply tasks (mirrors enrich.py / normalize.py): a
# bare create_task() is only weakly held by CPython and can be GC'd at its first
# await once the request returns.
_bg_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


# Same fallback delimiters inference defaults to — used only to build an
# in-memory dry-run preview of what the rule WOULD produce.
_PREVIEW_DELIMITERS = [", ", "; ", " / ", " | ", " - ", "__"]
_EMOJI_PATTERN = re.compile(
    "["
    "\U0000200d\U0000fe0e\U0000fe0f"
    "\U0001f3fb-\U0001f3ff\U0001f1e6-\U0001f1ff"
    "\U00002190-\U000021ff\U00002300-\U000023ff"
    "\U00002600-\U000027bf\U00002b00-\U00002bff"
    "\U0001f000-\U0001faff"
    "]+"
)
_WS = re.compile(r"\s+")


class NormalizeCapability:
    name = "normalize"

    def describe(self) -> str:
        return (
            "Clean up messy values on a named attribute/relationship: split "
            "composite multi-value cells into atomic ones (list_explode) and "
            "strip emoji/junk from text (strip_emoji). Use for 'clean', "
            "'normalize', 'split', 'tidy', 'fix the values of <field>' requests."
        )

    async def plan(
        self,
        ctx: AgentContext,
        instruction: str,
        predicate_leaves: list[str] | None = None,
    ) -> list[PlanStep]:
        """Infer rules for the predicate(s) named in the instruction.

        The planner (or the enrich capability composing a prerequisite) may pass
        ``predicate_leaves`` explicitly; otherwise we extract candidate
        predicate names from the instruction. We deliberately target only those
        predicate(s) — never scan every predicate (COG-118 perf).
        """
        type_name = ctx.type_name or ""
        leaves = predicate_leaves or _extract_predicate_leaves(instruction)
        if not type_name or not leaves:
            return []
        rules = await suggest_rules_for_predicates(
            ctx.neptune, ctx.tenant_id, ctx.kg_name, type_name, leaves
        )
        steps: list[PlanStep] = []
        for rule in rules:
            steps.append(
                PlanStep(
                    capability=self.name,
                    action="apply_rule",
                    params={
                        "rule": rule.model_dump(),
                    },
                    rationale=rule.rationale
                    or f"Normalize '{rule.predicate}' on {type_name}.",
                    confidence=rule.confidence,
                    preview=_dry_run_preview(rule),
                    cost={},  # normalization is free (no paid calls)
                )
            )
        return steps

    async def execute(self, ctx: AgentContext, step: PlanStep) -> dict:
        """Persist the rule as confirmed + apply it in the background; ack now."""
        rule = NormalizationRule(**step.params["rule"])
        # The user confirmed the plan, so the rule is confirmed.
        rule.status = "confirmed"
        store = NormalizationRuleStore(ctx.neptune)
        await store.save(ctx.tenant_id, rule)
        _spawn(_apply_and_mark(ctx.neptune, ctx.tenant_id, rule))
        return {
            "kind": "ack",
            "capability": self.name,
            "action": step.action,
            "rule_id": rule.id,
            "predicate": rule.predicate,
            "rule_type": rule.rule_type,
            "rule_status": "accepted",
            "message": (
                f"Normalizing '{rule.predicate}' ({rule.rule_type}) on "
                f"{rule.type_name} in the background."
            ),
        }


async def _apply_and_mark(neptune, tenant_id: str, rule: NormalizationRule) -> None:
    """Run apply_rule, then mark applied. Detached — errors logged, not raised."""
    try:
        summary = await apply_rule(neptune, tenant_id, rule)
        await NormalizationRuleStore(neptune).update_status(
            tenant_id,
            rule.id,
            "applied",
            applied_at=datetime.now(timezone.utc).isoformat(),
        )
        logger.info("agent_normalize_done", rule_id=rule.id, **summary)
    except Exception:
        logger.error("agent_normalize_failed", rule_id=rule.id, exc_info=True)


def _dry_run_preview(rule: NormalizationRule) -> dict:
    """Build a before/after preview from the rule's sampled values — IN MEMORY.

    No writes, no Neptune round-trip: we apply the same split/strip logic the
    executor uses to a few sampled values so the user sees what WOULD change.
    """
    samples = rule.sample_values[:5]
    changes: list[dict] = []
    for v in samples:
        after = _apply_in_memory(rule, v)
        if after != [v]:
            changes.append({"before": v, "after": after})
    return {
        "rule_type": rule.rule_type,
        "predicate": rule.predicate,
        "samples": changes,
        "summary": _summary_line(rule, changes),
    }


def _apply_in_memory(rule: NormalizationRule, value: str) -> list[str]:
    """Apply the rule's transform to a single value, returning the result list."""
    if rule.rule_type == "strip_emoji":
        cleaned = _WS.sub(" ", _EMOJI_PATTERN.sub("", value)).strip()
        return [cleaned] if cleaned else []
    # list_explode
    delims = (rule.params or {}).get("delimiters") or _PREVIEW_DELIMITERS
    parts = [value]
    for d in delims:
        nxt: list[str] = []
        for p in parts:
            nxt.extend(p.split(d))
        parts = nxt
    out = [p.strip() for p in parts if p.strip()]
    return out or [value]


def _summary_line(rule: NormalizationRule, changes: list[dict]) -> str:
    if not changes:
        return f"No sampled '{rule.predicate}' values need {rule.rule_type}."
    ex = changes[0]
    if rule.rule_type == "list_explode":
        return (
            f"Split composite '{rule.predicate}' values, e.g. "
            f"{ex['before']!r} → {ex['after']}."
        )
    return f"Strip junk from '{rule.predicate}', e.g. {ex['before']!r} → {ex['after']}."


# Quoted-token | word-after-trigger extraction for predicate names in NL.
_QUOTED = re.compile(r"['\"`]([A-Za-z_][\w-]*)['\"`]")
_TRIGGER = re.compile(
    r"\b(?:field|attribute|predicate|column|property|values? of|the)\s+"
    r"([A-Za-z_][\w-]*)",
    re.IGNORECASE,
)


def _extract_predicate_leaves(instruction: str) -> list[str]:
    """Best-effort pull of predicate leaf names from an NL instruction.

    Prefers quoted tokens (``clean the 'speaks' field``); falls back to a word
    after a trigger phrase (``clean the speaks values``). Returns a de-duped
    list. The planner usually passes predicates explicitly; this is the fallback
    when a user types a free-form clean request straight at the normalize intent.
    """
    found: list[str] = []
    for m in _QUOTED.finditer(instruction):
        found.append(m.group(1))
    if not found:
        for m in _TRIGGER.finditer(instruction):
            tok = m.group(1)
            if tok.lower() not in {"the", "a", "an", "of", "values", "value"}:
                found.append(tok)
    # De-dupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for f in found:
        if f.lower() not in seen:
            seen.add(f.lower())
            out.append(f)
    return out
