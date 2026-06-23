"""Capability protocol + module-level registry for the unified Ask-AI agent.

The agent has exactly ONE conversational surface (``POST /graphs/{tenant}/agent``).
Everything a user can ask the agent to do is a *capability* registered here —
question answering, normalization, enrichment, and (later) dedup / ontology
edits. Adding a capability is a one-liner :func:`register_capability` call; no
route changes, no new endpoints. This is the whole point: there is no per-task
divergence (no separate ``/enrich``-flavoured conversational endpoint, no
``/normalize`` chat, etc.) — the agent classifies intent once and dispatches to
the matching capability through this registry.

The registry mirrors the existing plugin-registry style in this package
(:func:`cograph_client.enrichment.sources.base.register_adapter`,
:func:`cograph_client.enrichment.tiers.register_tier`): a module-level dict, an
idempotent register, and accessors. So a downstream (proprietary) deployment can
register paid capabilities (e.g. an embedding-matcher dedup capability) at app
boot exactly the way it registers paid enrichment adapters today — without
touching OSS routes.

Capabilities NEVER call the existing HTTP routes; they reuse the underlying
engines directly (the ask pipeline, ``EnrichmentExecutor``, the normalization
``suggest_rules`` / ``apply_rule``). The HTTP routes stay for back-compat.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from cograph_client.graph.client import NeptuneClient


@dataclass
class AgentContext:
    """Per-request context handed to every capability ``plan`` / ``execute``.

    Carries the tenant + KG scope the agent operates within plus the live
    Neptune client and the optional OpenRouter/Anthropic keys the engines need.
    A capability reads only what it needs; new fields are added with safe
    defaults so existing capabilities keep working.
    """

    tenant_id: str
    kg_name: str
    neptune: NeptuneClient
    type_name: Optional[str] = None
    selection: Optional[dict] = None
    openrouter_key: str = ""
    anthropic_key: str = ""
    # Free-form extras (e.g. the enrichment executor/job-store stashed on
    # app.state) so a capability can reuse app-scoped singletons without the
    # context model needing to know about every engine.
    extras: dict = field(default_factory=dict)


@dataclass
class PlanStep:
    """One proposed unit of work in a plan.

    A plan is an ORDERED-once-resolved list of these. ``depends_on`` names the
    ids of steps that must run first (e.g. an enrich step depends on a
    clean-before-enrich normalize step). Nothing in a PlanStep is executed at
    plan time — it is a *proposal* with a dry-run ``preview`` and a ``cost``
    estimate so the user can confirm before any write happens.
    """

    capability: str
    action: str
    params: dict = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 0.0
    # Human-/UI-facing dry-run preview: before/after samples or a short summary.
    preview: dict = field(default_factory=dict)
    # Cost estimate, e.g. {"paid_calls": 1200} or {} for free work.
    cost: dict = field(default_factory=dict)
    # Ids of prerequisite steps that must execute before this one.
    depends_on: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "capability": self.capability,
            "action": self.action,
            "params": self.params,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "preview": self.preview,
            "cost": self.cost,
            "depends_on": list(self.depends_on),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlanStep":
        return cls(
            id=d.get("id") or str(uuid.uuid4()),
            capability=d["capability"],
            action=d.get("action", ""),
            params=d.get("params", {}),
            rationale=d.get("rationale", ""),
            confidence=float(d.get("confidence", 0.0) or 0.0),
            preview=d.get("preview", {}),
            cost=d.get("cost", {}),
            depends_on=list(d.get("depends_on", [])),
        )


@runtime_checkable
class AgentCapability(Protocol):
    """A registered capability behind the single agent endpoint.

    * ``name`` — stable id, matched against the classifier's chosen intent.
    * ``describe()`` — one line telling the classifier when this applies.
    * ``plan(ctx, instruction)`` — propose 0+ :class:`PlanStep`\\ s (may include
      prerequisite steps wired via ``depends_on``). NO writes.
    * ``execute(ctx, step)`` — run one step; for long work, spawn a background
      job (strong-ref pattern, like ``enrich.py``) and return an ack summary.
    """

    name: str

    def describe(self) -> str: ...

    async def plan(self, ctx: AgentContext, instruction: str) -> list[PlanStep]: ...

    async def execute(self, ctx: AgentContext, step: PlanStep) -> dict: ...


# Module-level registry — same shape as register_adapter / register_tier.
_capabilities: dict[str, AgentCapability] = {}


def register_capability(cap: AgentCapability) -> None:
    """Register (or replace) a capability by name. Idempotent — last write wins,
    so re-running default registration at app boot is safe."""
    _capabilities[cap.name] = cap


def get_capability(name: str) -> Optional[AgentCapability]:
    return _capabilities.get(name)


def get_capabilities() -> list[AgentCapability]:
    return list(_capabilities.values())


def reset_capabilities() -> None:
    """Clear the registry. For tests."""
    _capabilities.clear()


def order_steps(steps: list[PlanStep]) -> list[PlanStep]:
    """Topologically order steps so every step's ``depends_on`` runs first.

    A small Kahn's-algorithm sort. Steps referencing an unknown dependency id
    are kept (the dep is simply treated as already-satisfied) so a malformed
    plan still runs in a sensible order rather than dropping steps. A dependency
    cycle falls back to input order for the remaining steps (never loops).
    """
    by_id = {s.id: s for s in steps}
    indeg = {s.id: 0 for s in steps}
    for s in steps:
        for dep in s.depends_on:
            if dep in by_id:
                indeg[s.id] += 1
    # Preserve input order among ready steps for deterministic output.
    ready = [s for s in steps if indeg[s.id] == 0]
    ordered: list[PlanStep] = []
    seen: set[str] = set()
    while ready:
        s = ready.pop(0)
        if s.id in seen:
            continue
        seen.add(s.id)
        ordered.append(s)
        for other in steps:
            if s.id in other.depends_on and other.id not in seen:
                indeg[other.id] -= 1
                if indeg[other.id] == 0:
                    ready.append(other)
    # Append any steps left out by a cycle, in input order, so none are dropped.
    for s in steps:
        if s.id not in seen:
            ordered.append(s)
    return ordered


__all__ = [
    "AgentCapability",
    "AgentContext",
    "PlanStep",
    "get_capabilities",
    "get_capability",
    "order_steps",
    "register_capability",
    "reset_capabilities",
]
