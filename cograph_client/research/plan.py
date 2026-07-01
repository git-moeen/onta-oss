"""The planner — schema-first research planning (ADR 0006 §Plan).

The single highest-leverage stage of the harness is deciding, up front, the
SHAPE of the answer. Given a natural-language question, the planner derives a
:class:`~cograph_client.research.types.TargetSchema` (what each row IS + its
columns) and a :class:`~cograph_client.research.types.ResearchPlan` (does this
need the web at all, is it a trivial fast-path, what discovery queries + seed
URLs to start from). Fixing the schema first is what makes the downstream
extraction and cite-or-abstain verification tractable — the extractor pulls
into a fixed shape, the verifier gates a fixed shape.

The planning call uses the STRONG model (``PRIMARY_MODEL``): a good schema pays
for itself across every later stage, and this is one call per run. It mirrors
the OSS strict-JSON pattern already used by the enrichment extractor — one
:func:`~cograph_client.resolver.llm_router.openrouter_chat` call with a
``json_object`` response format at ``temperature=0``, parsed defensively — and,
like that extractor, it NEVER raises: any error / missing key / malformed JSON
collapses to a deterministic fallback plan so a planner hiccup can't sink a run.

Boundary: OSS file. Imports only stdlib / ``cograph_client.*`` / ``httpx``.
No ``from cograph.*`` and no proprietary identifiers.
"""

from __future__ import annotations

from typing import Optional

from cograph_client.enrichment.extraction import _try_parse_json
from cograph_client.research.types import (
    Budget,
    ResearchPlan,
    SchemaField,
    TargetSchema,
)
from cograph_client.resolver.llm_router import PRIMARY_MODEL, openrouter_chat

__all__ = ["plan_research"]

_PLAN_SYSTEM = (
    "You are a precise research planner. Given a question, you decide the SHAPE "
    "of the answer (a target schema: what each output row represents and its "
    "columns) and how to go find it on the open web. You return ONLY a single "
    "JSON object and nothing else. You never invent columns the question does "
    "not ask for, and you keep discovery queries concrete and searchable. You "
    "DEFAULT to answering with the most reasonable interpretation; you ask the "
    "user to clarify ONLY when the question is genuinely ambiguous — when two or "
    "more readings would lead to materially different answers."
)


def _build_prompt(
    question: str,
    hint_columns: Optional[list[str]],
    seed_urls: Optional[list[str]],
) -> str:
    hints = [c for c in (hint_columns or []) if str(c).strip()]
    seeds = [u for u in (seed_urls or []) if str(u).strip()]
    parts = [
        "Plan the research for the QUESTION below. Return ONLY a JSON object "
        "shaped EXACTLY like:\n"
        '{"entity": "<what each output row represents>", '
        '"fields": [{"name": "<column>", "description": "<what it holds>", '
        '"type": "string|number|boolean|date|url", "required": <bool>}], '
        '"needs_web": <bool>, "fast_path": <bool>, '
        '"queries": ["<concrete discovery query>", ...], '
        '"needs_clarification": <bool>, '
        '"clarifying_questions": [{"question": "<question to ask the user>", '
        '"options": ["<suggested answer>", ...]}, ...], '
        '"rationale": "<one line>"}\n\n'
        "Rules:\n"
        "- `entity` names one row (e.g. \"TTS model\", \"company\").\n"
        "- `fields` are ONLY the columns the question asks for; mark the "
        "identifying column(s) `required: true`.\n"
        "- `needs_web` is false ONLY when the question can be answered with no "
        "web lookup at all.\n"
        "- `fast_path` is true for a trivial single-fact question one cited "
        "page covers.\n"
        "- `queries` are concrete search strings that would surface the "
        "authoritative sources.\n"
        "- `needs_clarification` is true ONLY when the question is genuinely "
        "ambiguous — two or more readings give materially different answers "
        "(e.g. \"list the best models\" — best by what metric? which modality? "
        "over what time window?). DEFAULT to false and just pick the most "
        "reasonable interpretation; do NOT ask about minor details you can "
        "sensibly assume.\n"
        "- `clarifying_questions` holds 1–3 short, specific questions ONLY when "
        "`needs_clarification` is true; otherwise an empty list. When you ask, "
        "you may still fill `entity`/`fields`/`queries` with your best guess.\n"
        "- Each clarifying question SHOULD carry 2–4 short `options` when the "
        "plausible answers are enumerable (e.g. modality: LLMs / image / speech; "
        "metric: benchmark score / price / popularity) — they render as one-tap "
        "choices. Use an empty `options` list when a free-form answer is better "
        "(e.g. \"which company?\"). Options are suggestions, never a closed set.\n"
        "- `rationale` is one short line.\n",
    ]
    if hints:
        parts.append(
            "\nThe caller REQUIRES these columns — include every one of them as "
            "a field (add others only if the question clearly needs them): "
            + ", ".join(hints)
        )
    if seeds:
        parts.append(
            "\nThe caller already knows these authoritative URLs (do not repeat "
            "them as queries): " + ", ".join(seeds)
        )
    parts.append(f"\n\nQUESTION:\n{question}")
    return "".join(parts)


def _ensure_hint_fields(
    schema: TargetSchema, hint_columns: Optional[list[str]]
) -> TargetSchema:
    """Ensure every caller-supplied hint column appears as a schema field.

    The caller's hints are authoritative: if the planner omitted one (or there
    was no planner at all), append it as a plain string field. Case-insensitive
    de-dupe against the fields already present, order-preserving.
    """
    hints = [str(c).strip() for c in (hint_columns or []) if str(c).strip()]
    if not hints:
        return schema
    present = {f.name.lower() for f in schema.fields if f.name}
    for col in hints:
        if col.lower() not in present:
            schema.fields.append(SchemaField(name=col))
            present.add(col.lower())
    return schema


def _fallback_plan(
    question: str,
    hint_columns: Optional[list[str]],
    seed_urls: Optional[list[str]],
) -> ResearchPlan:
    """Deterministic plan used whenever the planner LLM is unavailable/failed.

    Schema comes from the caller's hint columns when given (else a single
    ``answer`` field); we assume the web is needed and start discovery from the
    raw question. Never raises.
    """
    hints = [str(c).strip() for c in (hint_columns or []) if str(c).strip()]
    fields = (
        [SchemaField(name=c) for c in hints]
        if hints
        else [SchemaField(name="answer")]
    )
    seeds = [str(u).strip() for u in (seed_urls or []) if str(u).strip()]
    return ResearchPlan(
        question=question,
        schema=TargetSchema(entity="item", fields=fields),
        needs_web=True,
        fast_path=False,
        queries=[question] if str(question).strip() else [],
        seed_urls=seeds,
        rationale="fallback: no planner LLM",
    )


async def plan_research(
    question: str,
    *,
    hint_columns: Optional[list[str]] = None,
    seed_urls: Optional[list[str]] = None,
    openrouter_key: str = "",
    model: Optional[str] = None,
    budget: "Optional[Budget]" = None,
) -> ResearchPlan:
    """Turn a natural-language question into a schema-first research plan.

    One :func:`~cograph_client.resolver.llm_router.openrouter_chat` call (strict
    ``json_object``, ``temperature=0``, the strong ``PRIMARY_MODEL`` by default —
    a good schema pays off across every later stage) parses the question into a
    :class:`~cograph_client.research.types.TargetSchema` + a
    :class:`~cograph_client.research.types.ResearchPlan`.

    Caller inputs are merged authoritatively: ``question`` and ``seed_urls`` are
    threaded onto the returned plan, and every ``hint_columns`` entry is
    guaranteed to appear as a schema field.

    NEVER raises. With no ``openrouter_key``, an exhausted ``budget``, or any
    error / malformed JSON it returns a deterministic fallback plan
    (``needs_web=True``, ``fast_path=False``, ``queries=[question]``).

    Args:
        question: The natural-language research question.
        hint_columns: Optional caller-required output columns; each is ensured to
            appear as a schema field (and seeds the fallback schema).
        seed_urls: Optional authoritative URLs to start fetching from; passed
            through onto the plan verbatim.
        openrouter_key: OpenRouter API key. Empty → deterministic fallback.
        model: Optional model override; defaults to ``PRIMARY_MODEL``.
        budget: Optional per-run :class:`~cograph_client.research.types.Budget`;
            when it can no longer afford an LLM call the planner falls back
            without calling out. The one planning call is metered via
            ``note_llm(1)``.

    Returns:
        A :class:`~cograph_client.research.types.ResearchPlan`.
    """
    # No key → deterministic fallback (mirrors the enrichment extractor).
    if not openrouter_key:
        return _fallback_plan(question, hint_columns, seed_urls)

    # Respect the budget: if we can't afford the planning call, fall back
    # cleanly rather than exceeding the cap.
    if budget is not None and not budget.can_call_llm():
        return _fallback_plan(question, hint_columns, seed_urls)

    if budget is not None:
        budget.note_llm(1)

    try:
        content = await openrouter_chat(
            openrouter_key,
            _PLAN_SYSTEM,
            _build_prompt(question, hint_columns, seed_urls),
            model=model or PRIMARY_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception:
        # Network error, HTTP error after fallback chain, timeout, malformed
        # response shape — all collapse to the deterministic fallback plan.
        return _fallback_plan(question, hint_columns, seed_urls)

    # OpenRouter can return ``content: null`` (empty / refused completion),
    # which surfaces as Python ``None`` — guard before parsing.
    if not content:
        return _fallback_plan(question, hint_columns, seed_urls)

    obj = _try_parse_json(content)
    if obj is None:
        return _fallback_plan(question, hint_columns, seed_urls)

    # The LLM contract is FLAT (`entity` + `fields` at the top level), but
    # ``ResearchPlan.from_dict`` reads the schema from a nested ``schema`` key —
    # lift the flat pair into it (preferring an already-nested ``schema`` if the
    # model happened to nest it). ``from_dict`` then normalizes the schema,
    # queries, and needs_web/fast_path. We thread question + seed_urls (union of
    # planner-proposed and caller-supplied) onto the payload before parsing.
    plan_payload = dict(obj)
    if not isinstance(plan_payload.get("schema"), dict):
        plan_payload["schema"] = {
            "entity": plan_payload.get("entity", "item"),
            "fields": plan_payload.get("fields") or [],
        }
    plan_payload["question"] = question

    seeds_out: list[str] = []
    seen_seed: set[str] = set()
    for u in list(plan_payload.get("seed_urls") or []) + list(seed_urls or []):
        s = str(u).strip()
        if s and s not in seen_seed:
            seen_seed.add(s)
            seeds_out.append(s)
    plan_payload["seed_urls"] = seeds_out

    try:
        plan = ResearchPlan.from_dict(plan_payload)
    except Exception:
        return _fallback_plan(question, hint_columns, seed_urls)

    # Clarification only counts when the planner actually supplied questions; cap
    # to a few crisp ones so we ask, not interrogate. A `needs_clarification: true`
    # with no questions is a model slip — treat it as "proceed".
    plan.clarifying_questions = plan.clarifying_questions[:3]
    plan.needs_clarification = plan.needs_clarification and bool(
        plan.clarifying_questions
    )

    # Caller's hint columns are authoritative — guarantee they all appear.
    plan.schema = _ensure_hint_fields(plan.schema, hint_columns)

    # A schema with no fields at all is useless downstream; backstop it the same
    # way the fallback does so extraction always has a shape to target.
    if plan.schema.is_empty():
        plan.schema = _fallback_plan(question, hint_columns, seed_urls).schema

    # If the planner proposed no discovery queries and named no seeds, seed
    # discovery from the raw question so the run has somewhere to start.
    if not plan.queries and not plan.seed_urls and str(question).strip():
        plan.queries = [question]

    return plan
