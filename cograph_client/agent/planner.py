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

from cograph_client.agent.capabilities.dedup_cap import DedupCapability
from cograph_client.agent.capabilities.enrich_cap import EnrichCapability
from cograph_client.agent.capabilities.normalize_cap import NormalizeCapability
from cograph_client.agent.capabilities.ontology_cap import OntologyCapability
from cograph_client.agent.capabilities.query import QueryCapability
from cograph_client.agent.conversation_store import (  # noqa: F401  (re-exported)
    Turn,
    make_conversation_store,
    reset_conversation_store,
)
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
    "dedup": "dedup",  # registered (DedupCapability) → plans an ER rebuild
    "ontology": "ontology",  # registered (OntologyCapability) → inspect/declare
}

# When the user asks for SEVERAL actions in one breath ("clean the names and
# dedupe"), we plan each capability and compose them into one ordered plan. This
# is the order they run in: cleaning the VALUES first means the dedup/enrich pass
# operates on already-normalized data — the documented clean-before-dedup /
# clean-before-enrich pattern. Lower number = earlier.
_INTENT_PLAN_ORDER = {"clean": 0, "enrich": 1, "dedup": 2, "ontology": 3}

# Convergence guard (COG-130): once the agent has asked this many clarifying
# questions in a session, the classifier is told to STOP asking and commit. The
# real fix is feeding it the dialogue (below) so it rarely needs to; this caps
# the worst case so the panel can never loop forever on `clarify`.
_MAX_CLARIFY_ROUNDS = 1

# How many recent turns of a (possibly long, history-backed) transcript to feed
# the classifier prompt + accumulate for capability extraction. The store keeps
# a longer tail for the history UI; the prompt only needs the recent context.
_PROMPT_HISTORY_TURNS = 16


_CLASSIFY_SYSTEM = """\
You are the intent router for a knowledge-graph data assistant. Read the WHOLE \
conversation (not just the latest message) and classify what the user wants into \
one or MORE of these intents:

- "question": a read-only question about the data (counts, lookups, "how many", \
"which", "list", "show me"). The assistant will answer with SPARQL.
- "enrich": fill in / look up / find missing ATTRIBUTE values for a type from \
external sources ("enrich", "fill in the X", "look up the Y for Z").
- "clean": normalize / clean / split / tidy messy VALUES of a field \
("clean the speaks field", "split the skills", "strip emoji from titles", \
"clean up the names").
- "dedup": find and merge duplicate entities ("remove duplicates", "de-dupe", \
"merge duplicate records").
- "ontology": change the schema / types / attributes / relationships.
- "ambiguous": you genuinely cannot tell what is wanted and must ask ONE \
clarifying question.

CRITICAL rules:
- The user may ask for several things at once. "clean up the names and remove \
duplicates" is BOTH "clean" AND "dedup" — return both, do not ask which one.
- USE THE PRIOR TURNS. If you already asked a clarifying question and the user \
answered it (even tersely, e.g. "both", "yes", "just the names"), treat the \
question as ANSWERED and commit — never re-ask the same dimension.
- Only return "ambiguous" when the conversation as a whole still does not say \
what to do. If you can act, act.

You are also given the available capabilities (one line each). Respond with \
STRICT JSON only:
{"intents": ["<one or more of the above>"], "clarify": "<a clarifying question, \
ONLY when the single intent is ambiguous>", "options": ["<2-4 short clickable \
answer choices>"]}

When you ask a clarifying question, ALSO provide "options": a short list (2-4) of \
the distinct answers the user is choosing between, each a few words, phrased as \
the user would say them (e.g. for clean-vs-merge: ["Clean up the values", "Merge \
duplicates", "Both"]). The user can click one instead of typing. Omit "options" \
(or use []) only when the answer is genuinely free-form (a field name, a value) \
and no small set of choices fits."""

# Generic action options offered on a fall-back clarify (greeting, "I can't yet
# handle X", or when the classifier didn't suggest its own). Each maps cleanly to
# an intent when the user clicks it, so the next turn routes straight to a plan.
_DEFAULT_ACTION_OPTIONS = [
    "Ask a question about the data",
    "Clean up messy values",
    "Enrich missing attributes",
    "Merge duplicate records",
    "Change the schema",
]


def _format_history(history: list[Turn] | None) -> str:
    """Render the prior turns as a transcript block for the classifier prompt."""
    if not history:
        return ""
    lines = []
    for t in history:
        if t.role == "assistant":
            who = f"Assistant ({t.kind})" if t.kind else "Assistant"
        else:
            who = "User"
        text = (t.text or "").strip()
        if text:
            lines.append(f"{who}: {text}")
    if not lines:
        return ""
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


def _effective_instruction(history: list[Turn] | None, message: str) -> str:
    """Accumulate the user's answers so capability extraction sees the full ask.

    A capability's parameter extraction (which field, which attribute, which
    rule) runs on a single string. Feeding it only the latest reply ("I wanna do
    both") loses the field the user named two turns ago. Concatenating every
    user turn in the session — oldest first, current last — gives the extractor
    the whole dialogue, which is what lets a clarify→answer exchange converge to
    a concrete plan. With no prior turns this is just the message (unchanged
    single-turn behavior).
    """
    prior_user = [t.text for t in (history or []) if t.role == "user" and t.text]
    if not prior_user:
        return message
    return "\n".join([*prior_user, message])


async def _classify(
    ctx: AgentContext,
    message: str,
    history: list[Turn] | None = None,
    prior_clarify_count: int = 0,
) -> dict:
    """One bounded LLM call → {"intents": [...], "clarify": ...}.

    Sees the running transcript (``history``) so a terse answer to a prior
    clarify is classified in context instead of in isolation. On any error /
    missing key we degrade to "ambiguous" with a generic clarify so the agent
    never 500s on classification.
    """
    caps = "\n".join(f"- {c.name}: {c.describe()}" for c in get_capabilities())
    convo = _format_history(history)
    guard = ""
    if prior_clarify_count >= _MAX_CLARIFY_ROUNDS:
        guard = (
            f"You have ALREADY asked {prior_clarify_count} clarifying "
            "question(s) in this conversation and the user has responded. Do NOT "
            "ask again — use their answers above and commit to the intent(s).\n\n"
        )
    user = (
        f"Available capabilities:\n{caps}\n\n{convo}{guard}"
        f"Latest user message: {message}"
    )
    if not ctx.openrouter_key:
        return _ambiguous()
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
        return _ambiguous()
    return _parse_classification(text)


def _ambiguous(clarify: str = "What would you like me to do?") -> dict:
    return {
        "intents": ["ambiguous"],
        "clarify": clarify,
        "options": list(_DEFAULT_ACTION_OPTIONS),
    }


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
        return _ambiguous()
    return _normalize_classification(data)


def _normalize_classification(data: dict) -> dict:
    """Coerce a classifier reply to {"intents": [...], "clarify": str}.

    Accepts both the new ``intents`` array and the legacy single ``intent`` key
    (so older prompts/clients — and the existing test stubs — keep working).
    De-dupes preserving order and never returns an empty list.
    """
    raw = data.get("intents")
    if not isinstance(raw, list) or not raw:
        one = data.get("intent")
        raw = [one] if one else []
    intents: list[str] = []
    seen: set[str] = set()
    for i in raw:
        s = str(i).strip().lower()
        if s and s not in seen:
            seen.add(s)
            intents.append(s)
    if not intents:
        intents = ["ambiguous"]
    return {
        "intents": intents,
        "clarify": data.get("clarify", "") or "",
        "options": _clean_options(data.get("options")),
    }


def _clean_options(raw) -> list[str]:
    """Sanitize classifier-suggested clickable options: strings, capped at 4."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for o in raw:
        s = str(o).strip()
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
        if len(out) >= 4:
            break
    return out


async def handle(ctx: AgentContext, message: str, session: dict | None = None) -> dict:
    """Classify the message and respond — answer, clarify, or propose a plan.

    Multi-turn aware (COG-130): when ``session.id`` is supplied, the running
    transcript is loaded and threaded into both the classifier and the
    capabilities so a clarify→answer exchange converges instead of looping. Each
    turn (the user message + the assistant's reply) is appended to the session's
    transcript. NO data writes happen here — an action returns a persisted plan
    the caller confirms via :func:`execute_plan`.
    """
    session_id = (session or {}).get("id")
    owner = (session or {}).get("owner")
    history = await _load_history(ctx, session_id)
    prior_clarify_count = sum(
        1 for t in history if t.role == "assistant" and t.kind == "clarify"
    )

    result = await _respond(ctx, message, session_id, history, prior_clarify_count)

    await _record_turn(ctx, session_id, message, result, owner)
    if session_id:
        result.setdefault("session_id", session_id)
    return result


async def _respond(
    ctx: AgentContext,
    message: str,
    session_id: str | None,
    history: list[Turn],
    prior_clarify_count: int,
) -> dict:
    """The classify → dispatch core, factored out of transcript bookkeeping."""
    # Only the recent tail grounds the prompt — a long history-backed thread
    # shouldn't blow up the classifier context (COG-131).
    recent = (
        history[-_PROMPT_HISTORY_TURNS:]
        if len(history) > _PROMPT_HISTORY_TURNS
        else history
    )
    classification = await _classify(ctx, message, recent, prior_clarify_count)
    intents = classification.get("intents", ["ambiguous"])

    # A read-only question is terminal and does not compose with actions.
    if "question" in intents:
        cap = get_capability("query") or QueryCapability()
        out = await cap.answer(ctx, message)  # type: ignore[attr-defined]
        return {"kind": "answer", **out}

    actionable = [i for i in intents if i in _INTENT_TO_CAPABILITY]
    if not actionable:
        return {
            "kind": "clarify",
            "question": classification.get("clarify")
            or "Could you clarify what you'd like me to do?",
            # Offer the model's own choices when it gave them, else the generic
            # action menu — so the user can click instead of typing.
            "options": classification.get("options") or list(_DEFAULT_ACTION_OPTIONS),
        }

    # Resolve the registered capabilities. A recognized intent with no capability
    # registered in THIS deployment (a downstream may map an intent it hasn't
    # registered) is skipped; if none resolve, clarify rather than fail.
    available = [
        (i, get_capability(_INTENT_TO_CAPABILITY[i])) for i in actionable
    ]
    available = [(i, c) for i, c in available if c is not None]
    if not available:
        return {
            "kind": "clarify",
            "question": (
                f"I can't yet handle '{actionable[0]}' requests. I can answer "
                "questions, enrich attributes, clean up values, merge duplicates, "
                "and inspect or extend the ontology — what would you like?"
            ),
            "options": list(_DEFAULT_ACTION_OPTIONS),
        }

    # Accumulate the user's answers across the dialogue so each capability's
    # field/attribute extraction sees the full ask, not just the latest reply.
    instruction = _effective_instruction(recent, message)
    steps = await _plan_intents(ctx, available, instruction)
    if not steps:
        labels = " and ".join(i for i, _ in available)
        return {
            "kind": "clarify",
            "question": (
                f"I understood you want to {labels}, but I couldn't determine the "
                "specifics (which field/attribute and value). Could you be more "
                "specific?"
            ),
        }

    # Read-only answer step: a capability may answer a question-like request
    # directly (e.g. the ontology capability's INSPECT op renders the schema)
    # instead of proposing a mutation. Such a step carries action="answer" and an
    # ``answer_payload``; surface it as {kind:"answer"} (no confirm round-trip),
    # exactly like the question fast-path. Only a SINGLE no-write answer step
    # short-circuits — a mutation plan always goes through confirm.
    if len(steps) == 1 and steps[0].action == "answer":
        payload = steps[0].params.get("answer_payload")
        if payload is not None:
            return {"kind": "answer", **payload}

    steps = order_steps(steps)
    plan_id = _new_plan_id()
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


async def _plan_intents(
    ctx: AgentContext,
    available: list[tuple[str, object]],
    instruction: str,
) -> list:
    """Plan each requested capability and compose them into one ordered plan.

    Capabilities are planned clean-first (``_INTENT_PLAN_ORDER``) so a "clean and
    dedup"/"clean and enrich" ask wires the dedup/enrich step's ``depends_on`` to
    the clean (normalize) step(s) — the documented clean-before-* pattern — and
    :func:`order_steps` then runs normalize first. A capability that can't ground
    a concrete step (returns ``[]``) simply contributes nothing; as long as ANY
    requested capability produces a step the turn converges to a plan instead of
    re-asking. A single requested intent collapses to exactly the prior
    single-capability behavior (no cross-capability dependency is added).
    """
    available = sorted(
        available, key=lambda pair: _INTENT_PLAN_ORDER.get(pair[0], 9)
    )
    all_steps: list = []
    normalize_ids: list[str] = []
    for intent, cap in available:
        steps = await cap.plan(ctx, instruction)  # type: ignore[attr-defined]
        if not steps:
            continue
        if intent in ("dedup", "enrich") and normalize_ids:
            for s in steps:
                s.depends_on = list(dict.fromkeys([*s.depends_on, *normalize_ids]))
        if intent == "clean":
            normalize_ids.extend(
                s.id for s in steps if s.capability == "normalize"
            )
        all_steps.extend(steps)
    return all_steps


async def _load_history(ctx: AgentContext, session_id: str | None) -> list[Turn]:
    """Load the session transcript; never fail the turn on a store hiccup."""
    if not session_id:
        return []
    try:
        return await make_conversation_store().load(session_id, ctx.tenant_id)
    except Exception:  # noqa: BLE001 — a transcript read must never 500 the turn
        logger.warning("agent_history_load_failed", exc_info=True)
        return []


def _result_summary(result: dict) -> tuple[str, str | None]:
    """Derive (assistant_text, intent_label) to store for an agent response."""
    kind = result.get("kind")
    if kind == "clarify":
        return result.get("question", ""), None
    if kind == "answer":
        return result.get("answer") or result.get("narrative") or "", "question"
    if kind == "plan":
        caps = ", ".join(
            dict.fromkeys(s.get("capability", "") for s in result.get("steps", []))
        )
        return f"Proposed a plan ({caps}).", caps or None
    return "", None


async def _record_turn(
    ctx: AgentContext,
    session_id: str | None,
    message: str,
    result: dict,
    owner: str | None = None,
) -> None:
    """Append the user message + assistant reply to the session transcript.

    ``owner`` (the auth subject) tags the thread so a signed-in user can find it
    in their history; it's None for ownerless (demo) sessions.
    """
    if not session_id:
        return
    text, intent = _result_summary(result)
    turns = [
        Turn(role="user", text=message),
        Turn(role="assistant", text=text, kind=result.get("kind"), intent=intent),
    ]
    try:
        await make_conversation_store().append(
            session_id, ctx.tenant_id, turns, owner=owner
        )
    except Exception:  # noqa: BLE001 — persistence is best-effort, never 500
        logger.warning("agent_history_append_failed", exc_info=True)


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
    register_capability(DedupCapability())
    register_capability(OntologyCapability())
