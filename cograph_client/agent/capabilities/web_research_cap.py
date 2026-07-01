"""Web-research capability — answer a question FROM THE WEB, cite it, don't store it.

This is the READ-ONLY counterpart to :class:`WebIngestCapability`. Discovery
INGEST creates new graph entities from a query; research ANSWERS a question by
reading the web and returns a cited answer plus a downloadable table
(CSV/JSON) — it writes NOTHING to the knowledge graph. Representative ask:
"list every TTS model on the VAPI Humanness Index with its score → CSV"
(ADR 0006).

It runs the staged harness (:class:`cograph_client.research.harness.WebResearchHarness`):
plan (schema-first) → discover (reuses the registered ``WebSourceProvider``) →
fetch ladder (OSS static → premium JS-render) → extract → verify (cite-or-abstain)
→ synthesize → reflect, all bounded by a per-request :class:`Budget`.

Like enrich / web-ingest, a run that spends on paid web tools goes through the
plan → confirm → execute gate (confirm-before-spend). ``plan`` does only the
cheap schema-first step (no web spend) and stashes the schema so ``execute``
commits the SAME shape it previewed. When neither a discovery provider nor an
explicit URL is available, the capability degrades gracefully to a plain "not
enabled" answer — the same no-op pattern web-ingest uses.

Boundary: OSS. Imports only stdlib / ``cograph_client.*``.
"""

from __future__ import annotations

import structlog

from cograph_client.agent.registry import AgentContext, PlanStep
from cograph_client.research.fetch import fetcher_cost, get_page_fetchers
from cograph_client.research.harness import WebResearchHarness
from cograph_client.research.types import (
    Budget,
    TargetSchema,
    normalize_clarifying_questions,
)
from cograph_client.web_sources.base import get_web_source, provider_cost
from cograph_client.web_sources.url_extract import extract_urls

logger = structlog.stdlib.get_logger("cograph.agent.web_research")

# Default per-request caps for a research run — kept modest so an interactive
# turn stays bounded and cheap; user/deployment can widen later.
_DEFAULT_BUDGET = {
    "max_iterations": 2,
    "max_fetches": 6,
    "max_llm_calls": 12,
    "max_wall_clock_s": 90.0,
}
_DEFAULT_MAX_ROWS = 200


def _answer_step(text: str) -> PlanStep:
    """A single no-write 'answer' step (planner short-circuits it to kind:answer)."""
    return PlanStep(
        capability=WebResearchCapability.name,
        action="answer",
        params={"answer_payload": {"answer": text, "narrative": text}},
        rationale=text,
        confidence=1.0,
    )


def _clarify_step(questions: list) -> PlanStep:
    """A no-spend 'answer' step that asks the user to disambiguate a genuinely
    ambiguous question BEFORE running (and paying for) the research loop. The
    questions ride the payload STRUCTURED (``{"question", "options"}``) so a
    client can render each one's suggested answers as one-tap reply chips;
    options are also inlined in the text for plain-text clients."""
    qs = normalize_clarifying_questions(questions)
    lines = [
        f"- {q.question}" + (f" ({' / '.join(q.options)})" if q.options else "")
        for q in qs
    ]
    text = (
        "This question has more than one reasonable reading — a quick "
        "clarification will get you a sharper answer:\n" + "\n".join(lines)
    )
    return PlanStep(
        capability=WebResearchCapability.name,
        action="answer",
        params={
            "answer_payload": {
                "answer": text,
                "narrative": text,
                "clarifying_questions": [q.to_dict() for q in qs],
            }
        },
        rationale="Question is ambiguous — asking before spending on web tools.",
        confidence=1.0,
    )


def _estimate_cost(max_fetches: int) -> dict:
    """A conservative plan-card cost estimate, read generically from the registered
    provider + fetcher cost signals (never by vendor name)."""
    est = 0.0
    paid_calls = 0
    provider = get_web_source()
    if provider is not None:
        p_paid, p_cost = provider_cost(provider)
        if p_paid:
            paid_calls += 2  # up to two discovery queries per run
            est += 2 * p_cost
    # The priciest paid rung we might escalate to, charged across the fetch cap.
    paid_fetch_costs = [
        c for f in get_page_fetchers() for p, c in [fetcher_cost(f)] if p
    ]
    if paid_fetch_costs:
        est += max_fetches * max(paid_fetch_costs)
        paid_calls += max_fetches
    return {"estimated_usd": round(est, 4), "paid_calls": paid_calls}


class WebResearchCapability:
    name = "web_research"

    def describe(self) -> str:
        return (
            "Answer a question using the WEB and return a cited answer + a "
            "downloadable table (CSV/JSON), WITHOUT adding anything to the graph. "
            "Use for 'what/which/list <things> on <site/topic>', 'find the "
            "<values> for <set> from the web', 'research X and give me a CSV / "
            "report'. Distinct from 'discover' (which INGESTS web records as NEW "
            "graph entities) and from 'question' (read-only about data ALREADY in "
            "the graph)."
        )

    async def plan(self, ctx: AgentContext, instruction: str) -> list[PlanStep]:
        urls = (getattr(ctx, "urls", None) or []) or extract_urls(instruction)
        provider = get_web_source()

        # Availability: research needs SOMETHING to read — a discovery provider for
        # open-web search, or explicit URLs for the static fetcher. Absent both,
        # degrade to a plain answer (the same no-op pattern web-ingest uses).
        if provider is None and not urls:
            return [
                _answer_step(
                    "Web research isn't fully enabled in this deployment. Share one "
                    "or more page URLs and I'll read and structure them, or an admin "
                    "can configure a web-discovery provider (e.g. Exa or Perplexity) "
                    "for open-web search."
                )
            ]

        # Schema-first planning — the only plan-time work, and it does NOT touch the
        # web (no spend). Stash the schema so execute commits the same shape.
        schema = TargetSchema()
        rationale = "Research the web and return a cited, structured answer."
        try:
            from cograph_client.research.plan import plan_research

            plan = await plan_research(
                instruction,
                seed_urls=urls or None,
                openrouter_key=ctx.openrouter_key,
            )
            # Genuinely ambiguous → ask before spending. This replaces the research
            # step entirely with a no-write answer step carrying the questions, so
            # the confirm-before-spend gate is never even reached.
            if plan.needs_clarification and plan.clarifying_questions:
                return [_clarify_step(plan.clarifying_questions)]
            schema = plan.schema
            if plan.rationale:
                rationale = plan.rationale
        except Exception:  # noqa: BLE001 — planning must never 500 the turn
            logger.warning("web_research_plan_failed", exc_info=True)

        cols = schema.field_names()
        preview = {
            "target_entity": schema.entity,
            "columns": cols or ["answer"],
            "sources": urls[:5],
            "approach": (
                "read the URL(s) you supplied"
                if urls
                else "search the web for authoritative sources"
            ),
            "writes_to_graph": False,
        }
        step = PlanStep(
            capability=self.name,
            action="research",
            params={
                "question": instruction,
                "schema": schema.to_dict(),
                "urls": urls,
                "max_rows": _DEFAULT_MAX_ROWS,
                "budget": dict(_DEFAULT_BUDGET),
            },
            rationale=rationale,
            confidence=0.8,
            preview=preview,
            cost=_estimate_cost(_DEFAULT_BUDGET["max_fetches"]),
        )
        return [step]

    async def execute(self, ctx: AgentContext, step: PlanStep) -> dict:
        params = step.params or {}
        question = params.get("question", "")
        schema = TargetSchema.from_dict(params.get("schema") or {})
        urls = list(params.get("urls") or [])
        max_rows = int(params.get("max_rows", _DEFAULT_MAX_ROWS))
        # Filter to Budget's known fields so an unexpected/typo'd key in a stored
        # plan can't raise TypeError and lose the turn.
        merged_budget = {**_DEFAULT_BUDGET, **(params.get("budget") or {})}
        budget = Budget(
            **{k: v for k, v in merged_budget.items() if k in _DEFAULT_BUDGET}
        )

        harness = WebResearchHarness(openrouter_key=ctx.openrouter_key)
        result = await harness.run(
            question,
            schema=schema if not schema.is_empty() else None,
            urls=urls,
            max_rows=max_rows,
            budget=budget,
            context={"tenant_id": ctx.tenant_id, "kg_name": ctx.kg_name},
        )

        return {
            "capability": self.name,
            "action": "research",
            "kind": "research_result",
            "answer": result.answer,
            "narrative": result.answer,
            "rows": [r.to_dict() for r in result.rows],
            "citations": [c.to_dict() for c in result.citations],
            "sources": [c.url for c in result.citations],
            "schema": result.schema.to_dict(),
            "artifact_csv": result.to_csv(),
            "confidence": result.confidence,
            "abstained": result.abstained,
            "is_complete": result.is_complete,
            "iterations": result.iterations,
            "budget": budget.to_dict(),
        }


__all__ = ["WebResearchCapability"]
