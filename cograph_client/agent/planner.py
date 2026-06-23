"""The planner — the brain behind the single agent endpoint.

One bounded LLM call classifies the user's intent (question | enrich | clean |
dedup | ontology | ambiguous); the planner then dispatches to the matching
registered capability. There is NO per-task fan-out and NO per-task endpoint —
the classifier picks ONE capability and we call its ``plan()`` (or, for a
question, answer directly). Writes happen ONLY on ``execute_plan`` (after the
user confirms a returned plan), never during ``handle``.

Contract (the single conversational surface):
  handle(message) →
    {kind:"answer",  answer, sparql, rows}        # questions, no confirm
    {kind:"clarify", question}                    # ambiguous
    {kind:"plan",    plan_id, steps:[...]}         # actions, awaiting confirm
  execute_plan(plan_id) →
    {kind:"result",  steps:[summaries]}           # the only mutating path

Plan persistence (A2, COG-124): a swappable, tenant-scoped store keyed by
plan_id, mirroring the dual-backend :class:`JobStore` pattern. ``make_plan_store``
returns a :class:`PostgresPlanStore` when ``settings.database_url`` is set
(durable, shared across ECS tasks — so a confirm→execute survives a process
restart or a different task than the one that planned), else an
:class:`InMemoryPlanStore` (zero-config default). See
:mod:`cograph_client.agent.plan_store`.
"""

from __future__ import annotations

import json

import structlog

from cograph_client.agent.capabilities.enrich_cap import EnrichCapability
from cograph_client.agent.capabilities.normalize_cap import NormalizeCapability
from cograph_client.agent.capabilities.query import QueryCapability
from cograph_client.agent.plan_store import (  # noqa: F401  (re-exported for back-compat)
    InMemoryPlanStore,
    PlanStore,
    PostgresPlanStore,
    StoredPlan,
    get_plan_store,
    make_plan_store,
    reset_plan_store,
)
from cograph_client.agent.registry import (
    AgentContext,
    get_capabilities,
    get_capability,
    order_steps,
)
from cograph_client.resolver.llm_router import PRIMARY_MODEL, openrouter_chat

logger = structlog.stdlib.get_logger("cograph.agent.planner")

# Intents the classifier may return. "question" → answer; "ambiguous" → clarify;
# the rest map to a capability name (clean→normalize).
_INTENT_TO_CAPABILITY = {
    "enrich": "enrich",
    "clean": "normalize",
    "dedup": "dedup",  # A2 — capability not registered yet → clarify
    "ontology": "ontology",  # A2 — not registered yet → clarify
}


_CLASSIFY_SYSTEM = """\
You are the intent router for a knowledge-graph data assistant. Classify the \
user's message into EXACTLY ONE intent:

- "question": a read-only question about the data (counts, lookups, "how many", \
"which", "list", "show me"). The assistant will answer with SPARQL.
- "enrich": fill in / look up / find missing ATTRIBUTE values for a type from \
external sources ("enrich", "fill in the X", "look up the Y for Z").
- "clean": normalize / clean / split / tidy messy VALUES of a field \
("clean the speaks field", "split the skills", "strip emoji from titles").
- "dedup": find and merge duplicate entities.
- "ontology": change the schema / types / attributes / relationships.
- "ambiguous": the message is unclear or could be several of the above and you \
need to ask the user a clarifying question.

You are also given the available capabilities (one line each). Pick the intent \
whose capability best matches. Respond with STRICT JSON only:
{"intent": "<one of the above>", "clarify": "<a clarifying question, only if \
intent is ambiguous>"}"""


async def _classify(ctx: AgentContext, message: str) -> dict:
    """One bounded LLM call → {"intent": ..., "clarify": ...}.

    On any error / missing key we degrade to "ambiguous" with a generic clarify
    so the agent never 500s on classification.
    """
    caps = "\n".join(f"- {c.name}: {c.describe()}" for c in get_capabilities())
    user = f"Available capabilities:\n{caps}\n\nUser message: {message}"
    if not ctx.openrouter_key:
        return {"intent": "ambiguous", "clarify": "What would you like me to do?"}
    try:
        text = await openrouter_chat(
            ctx.openrouter_key,
            _CLASSIFY_SYSTEM,
            user,
            model=PRIMARY_MODEL,
            temperature=0,
            max_tokens=200,
            timeout=30,
        )
    except Exception:
        logger.warning("agent_classify_failed", exc_info=True)
        return {"intent": "ambiguous", "clarify": "What would you like me to do?"}
    return _parse_classification(text)


def _parse_classification(text: str) -> dict:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = "\n".join(
            l for l in stripped.split("\n") if not l.strip().startswith("```")
        )
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        stripped = stripped[start : end + 1]
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return {"intent": "ambiguous", "clarify": "What would you like me to do?"}
    intent = data.get("intent", "ambiguous")
    return {"intent": intent, "clarify": data.get("clarify", "")}


async def handle(ctx: AgentContext, message: str, session: dict | None = None) -> dict:
    """Classify the message and respond — answer, clarify, or propose a plan.

    NO writes happen here. An action returns a persisted plan the caller confirms
    via :func:`execute_plan`.
    """
    classification = await _classify(ctx, message)
    intent = classification.get("intent", "ambiguous")

    if intent == "question":
        cap = get_capability("query") or QueryCapability()
        out = await cap.answer(ctx, message)  # type: ignore[attr-defined]
        return {"kind": "answer", **out}

    if intent == "ambiguous":
        return {
            "kind": "clarify",
            "question": classification.get("clarify")
            or "Could you clarify what you'd like me to do?",
        }

    cap_name = _INTENT_TO_CAPABILITY.get(intent)
    cap = get_capability(cap_name) if cap_name else None
    if cap is None:
        # Recognized intent but no registered capability (e.g. dedup/ontology in
        # A1) → ask for clarification rather than fail.
        return {
            "kind": "clarify",
            "question": (
                f"I can't yet handle '{intent}' requests. I can answer questions, "
                "enrich attributes, and clean up values — what would you like?"
            ),
        }

    steps = await cap.plan(ctx, message)
    if not steps:
        return {
            "kind": "clarify",
            "question": (
                "I understood you want to "
                f"{intent}, but I couldn't determine the specifics (which "
                "field/attribute and value). Could you be more specific?"
            ),
        }

    steps = order_steps(steps)
    plan_id = _new_plan_id()
    session_id = (session or {}).get("id")
    await make_plan_store().save(
        StoredPlan(
            plan_id=plan_id,
            tenant_id=ctx.tenant_id,
            kg_name=ctx.kg_name,
            type_name=ctx.type_name,
            message=message,
            steps=steps,
            session_id=session_id,
        )
    )
    return {
        "kind": "plan",
        "plan_id": plan_id,
        "steps": [s.to_dict() for s in steps],
    }


async def execute_plan(ctx: AgentContext, plan_id: str) -> dict:
    """Run a persisted plan's steps in dependency order. The ONLY mutating path.

    Each step runs via its capability's ``execute`` (long work is spawned as a
    background job inside the capability). Records per-step status; idempotent-ish
    (re-running a done plan re-issues the acks — the underlying applies are
    themselves idempotent / staged).
    """
    store = make_plan_store()
    plan = await store.get(plan_id, ctx.tenant_id)
    if plan is None:
        return {"kind": "error", "error": "plan not found", "plan_id": plan_id}

    plan.status = "executing"
    await store.save(plan)
    ordered = order_steps(plan.steps)
    summaries: list[dict] = []
    for step in ordered:
        cap = get_capability(step.capability)
        if cap is None:
            summaries.append(
                {
                    "step_id": step.id,
                    "capability": step.capability,
                    "status": "skipped",
                    "error": "capability not registered",
                }
            )
            continue
        try:
            result = await cap.execute(ctx, step)
            # Spread the capability ack first, then stamp the orchestration
            # status LAST so a capability's own "status" field (e.g. a job's
            # "queued") can't clobber the step-level success marker.
            summaries.append({"step_id": step.id, **result, "status": "ok"})
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "agent_step_failed", step_id=step.id, capability=step.capability,
                exc_info=True,
            )
            summaries.append(
                {
                    "step_id": step.id,
                    "capability": step.capability,
                    "status": "failed",
                    "error": str(exc),
                }
            )
    plan.status = "done"
    await store.save(plan)
    return {"kind": "result", "plan_id": plan_id, "steps": summaries}


def _new_plan_id() -> str:
    import uuid

    return str(uuid.uuid4())


def register_default_capabilities() -> None:
    """Register the OSS A1 capabilities. Import-safe + idempotent.

    Called from app startup (and tests). A downstream/proprietary deployment can
    register additional capabilities (dedup with embedding matchers, ontology
    edits) the same way — no route change. ``register_capability`` is last-write-
    wins, so calling this more than once is harmless.
    """
    from cograph_client.agent.registry import register_capability

    normalize = NormalizeCapability()
    register_capability(QueryCapability())
    register_capability(normalize)
    register_capability(EnrichCapability(normalize=normalize))
