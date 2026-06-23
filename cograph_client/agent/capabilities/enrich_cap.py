"""Enrichment capability — with clean-before-enrich composition.

Reuses the existing enrichment engine (no reimplementation):

* ``plan`` parses the NL instruction into the existing :class:`EnrichRequest`
  shape (attributes + optional scope ``predicate=value`` + tier + confidence).
  THEN it detects a prerequisite: if the **scope predicate's target values are
  composite** (un-normalized — a delimiter shows up in the sampled target
  labels), scoping by ``value`` would MISS the rows packed inside a composite
  cell (e.g. scope ``speaks=Persian`` misses an entity whose ``speaks`` points
  at ``English__Persian``). In that case it emits a NORMALIZE step FIRST (reusing
  :class:`NormalizeCapability.plan` so the cleanup logic isn't duplicated) and
  sets the enrich step's ``depends_on`` to it. Returns ``[normalize_step?,
  enrich_step]``. No writes.

* ``execute`` runs the enrichment as a background job, building the EXACT same
  :class:`EnrichJob` + ``EnrichmentExecutor.run`` the ``/enrich/jobs`` route
  builds (strong-ref ``_spawn`` so the task can't be GC'd). Returns an ack.

The agent never calls the ``/enrich`` HTTP route — it drives the executor + job
store directly via the same primitives.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone

import structlog

from cograph_client.agent.capabilities.normalize_cap import NormalizeCapability
from cograph_client.agent.registry import AgentContext, PlanStep
from cograph_client.enrichment.models import (
    EnrichJob,
    EnrichScope,
    EnrichmentTier,
    JobStatus,
)
from cograph_client.normalization.inference import sample_predicate_values

logger = structlog.stdlib.get_logger("cograph.agent.enrich")

_bg_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


# Delimiters that signal a composite (un-normalized) target value. "__" is the
# slugified list separator the ingest produces; the rest are raw list delimiters.
_COMPOSITE_DELIMS = ["__", ", ", "; ", " / ", " | "]


class EnrichCapability:
    name = "enrich"

    def __init__(self, normalize: NormalizeCapability | None = None) -> None:
        # Reuse the normalize capability to BUILD the prerequisite step so the
        # clean-before-enrich logic lives in exactly one place.
        self._normalize = normalize or NormalizeCapability()

    def describe(self) -> str:
        return (
            "Fill in or verify missing attributes on a type by looking them up "
            "from external sources (enrichment). Use for 'enrich', 'fill in', "
            "'look up', 'find the <attribute> for <type>' requests, optionally "
            "scoped (e.g. 'for managers', 'who speak Persian')."
        )

    async def plan(
        self,
        ctx: AgentContext,
        instruction: str,
        parsed: dict | None = None,
    ) -> list[PlanStep]:
        """Build [normalize_step?, enrich_step] from the instruction.

        ``parsed`` (optional) lets the planner pass an already-parsed request
        (attributes/scope/tier/confidence). When absent we parse it here.
        """
        type_name = ctx.type_name or ""
        if not type_name:
            return []
        req = parsed or _parse_enrich_instruction(instruction)
        attributes: list[str] = req.get("attributes") or []
        if not attributes:
            return []
        tier = _coerce_tier(req.get("tier"))
        confidence_min = float(req.get("confidence_min", 0.85) or 0.85)
        scope = req.get("scope")  # {"predicate":..., "value":...} | None

        steps: list[PlanStep] = []
        depends_on: list[str] = []

        # clean-before-enrich: if a scope predicate's target is composite,
        # normalize it FIRST so the scope actually matches the packed rows.
        if scope and scope.get("predicate"):
            samples, _kind = await sample_predicate_values(
                ctx.neptune,
                ctx.tenant_id,
                ctx.kg_name,
                type_name,
                scope["predicate"],
            )
            if _looks_composite(samples):
                norm_steps = await self._normalize.plan(
                    ctx, instruction, predicate_leaves=[scope["predicate"]]
                )
                if norm_steps:
                    norm = norm_steps[0]
                    norm.rationale = (
                        f"Clean '{scope['predicate']}' before enrichment: its "
                        f"values are composite, so scoping by "
                        f"{scope.get('value')!r} would miss packed rows."
                    )
                    steps.append(norm)
                    depends_on = [norm.id]

        cost = _estimate_cost(tier)
        enrich_step = PlanStep(
            capability=self.name,
            action="run_enrichment",
            params={
                "type_name": type_name,
                "attributes": attributes,
                "tier": tier.value,
                "confidence_min": confidence_min,
                "scope": scope,
            },
            rationale=(
                f"Enrich {', '.join(attributes)} on {type_name}"
                + (f" scoped to {scope['predicate']}={scope['value']}" if scope else "")
                + f" via the {tier.value} tier."
            ),
            confidence=0.8,
            preview={
                "summary": (
                    f"Look up {', '.join(attributes)} for matched {type_name} "
                    f"entities and stage the results for review."
                ),
                "scope": scope,
                "tier": tier.value,
            },
            cost=cost,
            depends_on=depends_on,
        )
        steps.append(enrich_step)
        return steps

    async def execute(self, ctx: AgentContext, step: PlanStep) -> dict:
        """Create + run an EnrichJob in the background (same as /enrich/jobs)."""
        p = step.params
        executor = ctx.extras.get("enrichment_executor")
        job_store = ctx.extras.get("enrichment_job_store")
        if executor is None or job_store is None:
            raise RuntimeError(
                "enrichment executor/job_store not available in agent context"
            )
        scope = None
        if p.get("scope") and p["scope"].get("predicate"):
            scope = EnrichScope(
                predicate=p["scope"]["predicate"], value=p["scope"]["value"]
            )
        job = EnrichJob(
            id=str(uuid.uuid4()),
            tenant_id=ctx.tenant_id,
            kg_name=ctx.kg_name,
            type_name=p["type_name"],
            attributes=p["attributes"],
            tier=_coerce_tier(p.get("tier")),
            status=JobStatus.queued,
            created_at=datetime.now(timezone.utc),
            conflict_policy=_default_conflict_policy(),
            confidence_min=float(p.get("confidence_min", 0.85) or 0.85),
            scope=scope,
        )
        await job_store.create(job)
        _spawn(executor.run(job, ctx.tenant_id))
        return {
            "kind": "ack",
            "capability": self.name,
            "action": step.action,
            "job_id": job.id,
            "job_status": job.status.value,
            "message": (
                f"Enriching {', '.join(job.attributes)} on {job.type_name} "
                "in the background; results will be staged for review."
            ),
        }


def _default_conflict_policy():
    from cograph_client.enrichment.models import ConflictPolicy

    return ConflictPolicy.stage


def _looks_composite(samples: list[str]) -> bool:
    """Cheap composite check: any sampled target value carries a list delimiter."""
    for v in samples:
        for d in _COMPOSITE_DELIMS:
            if d in v:
                return True
    return False


def _coerce_tier(tier) -> EnrichmentTier:
    if isinstance(tier, EnrichmentTier):
        return tier
    try:
        return EnrichmentTier(str(tier))
    except ValueError:
        return EnrichmentTier.lite


def _estimate_cost(tier: EnrichmentTier) -> dict:
    """Cost estimate. The matched count is resolved by the executor at run time
    (not in the request path — COG-112), so at plan time we report the tier and
    a note rather than a blocking COUNT. ``lite`` is free (Wikidata only)."""
    if tier == EnrichmentTier.lite:
        return {"paid_calls": 0, "note": "lite tier (Wikidata) — no paid calls"}
    return {
        "paid_calls": None,
        "note": (
            f"{tier.value} tier may use paid sources; matched-entity count is "
            "resolved when the job runs (no blocking COUNT at plan time)."
        ),
    }


# --- NL parsing -------------------------------------------------------------- #

_ATTR_TRIGGER = re.compile(
    r"\b(?:enrich|fill in|fill|look up|lookup|find|get|add)\s+(?:the\s+)?"
    r"([A-Za-z_][\w-]*(?:\s*,\s*[A-Za-z_][\w-]*)*)",
    re.IGNORECASE,
)
# "for managers", "for the mentors", "for X who speak Persian"
_SCOPE_FOR = re.compile(
    r"\bfor\s+(?:the\s+)?([A-Za-z_][\w-]*)", re.IGNORECASE
)
# Relationship scope: "<verb> <Value>" e.g. "speak Persian", "speaks French".
# group(1) = verb, group(2) = value. Verb is lemmatized to its predicate leaf.
_SCOPE_REL = re.compile(
    r"\b(speak|speaks|speaking|knows?|knowing|using|uses?)\s+"
    r"([A-Z][\w-]+)",
)


def _parse_enrich_instruction(instruction: str) -> dict:
    """Parse an NL enrich instruction into the EnrichRequest shape (best-effort).

    Extracts attributes (the noun(s) after the enrich verb) and an optional
    relationship scope. This is a lightweight deterministic parser used when the
    planner does not supply a structured ``parsed`` dict. The planner's LLM
    classification may pass a richer parse in future; the contract is the same.

    Examples:
      "enrich company for managers"
        → attributes=["company"], scope=None
      "enrich company for mentors who speak Persian"
        → attributes=["company"], scope={"predicate":"speaks","value":"Persian"}
    """
    attributes: list[str] = []
    m = _ATTR_TRIGGER.search(instruction)
    if m:
        attributes = [a.strip() for a in m.group(1).split(",") if a.strip()]

    scope = None
    rel = _SCOPE_REL.search(instruction)
    if rel:
        verb = rel.group(1).lower()
        pred = _SCOPE_VERB_LEMMA.get(verb, verb)
        scope = {"predicate": pred, "value": rel.group(2)}
    return {"attributes": attributes, "scope": scope, "tier": "lite"}


# Map inflected scope verbs to their predicate leaf (the ontology stores the
# bare relationship name, e.g. "speaks").
_SCOPE_VERB_LEMMA = {
    "speak": "speaks",
    "speaks": "speaks",
    "speaking": "speaks",
    "know": "knows",
    "knows": "knows",
    "knowing": "knows",
    "use": "uses",
    "uses": "uses",
    "using": "uses",
}
