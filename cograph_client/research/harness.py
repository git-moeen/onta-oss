"""The web-research harness — the staged agent loop (ADR 0006 §Decision).

``WebResearchHarness.run`` answers a natural-language question from the web and
returns a cited :class:`ResearchResult` (answer + rows + CSV/JSON artifact). It
is READ-ONLY — it never writes to a knowledge graph (that is the discovery-ingest
capability's job).

The loop:

1. **Plan** — schema-first: derive a :class:`TargetSchema` + discovery queries.
2. **Discover** — reuse the registered :class:`WebSourceProvider` (premium) to
   get candidate rows + source URLs. Absent a provider, work from the user's URLs.
3. **Fetch (laddered)** — walk the :func:`default_ladder` cheapest-first; escalate
   to a pricier rung only when a page looks incomplete (empty / JS-gated).
4. **Extract** — schema-valid rows via constrained generation.
5. **Verify** — cite-or-abstain (drop unsourced rows; abstain if none survive).
6. **Synthesize** — compose the answer + artifact + citations.
7. **Reflect** — if thin and budget remains, broaden queries and loop.

Every metered action is charged against a per-request :class:`Budget`; the loop
stops cleanly the moment a cap is hit, returning what it has. Each stage is
INJECTABLE (constructor overrides) so unit tests run with fakes and zero network,
and so a premium deployment can swap any stage.

Boundary: OSS. Imports only stdlib / ``cograph_client.*``.
"""

from __future__ import annotations

import time
from typing import Awaitable, Callable, Optional

import structlog

from cograph_client.research.fetch import (
    FetchedPage,
    PageFetcher,
    default_ladder,
    fetcher_cost,
    is_fetchable_url,
)
from cograph_client.research.synthesize import (
    clarification_result,
    synthesize_result,
)
from cograph_client.research.types import (
    Budget,
    ResearchPlan,
    ResearchResult,
    ResearchRow,
    ResearchTrace,
    SchemaField,
    TargetSchema,
)
from cograph_client.research.verify import (
    ResearchVerifier,
    VerifyOutcome,
    get_research_verifier,
)
from cograph_client.web_sources.base import (
    DiscoverResult,
    get_web_source,
    provider_cost,
)

logger = structlog.stdlib.get_logger("cograph.research.harness")

# A fetched page is treated as "incomplete" — a signal to escalate to the next,
# pricier ladder rung — when it is empty, truly tiny, JS-gated, or a nav-only
# shell (see _looks_incomplete).
_INCOMPLETE_CHARS = 200
_JS_GATE_MARKERS = ("enable javascript", "requires javascript", "please enable js")
# A line at least this long reads as prose/data (a sentence, a table row); shorter
# lines are nav labels ("Docs", "Pricing"). A client-side-rendered page whose JS
# never ran returns only its shell — a stack of short nav labels — so it scores
# ~0 on the substantive-char measure below even when it clears _INCOMPLETE_CHARS.
_PROSE_LINE_CHARS = 40
# Minimum characters living in prose/data-length lines for a page over
# _INCOMPLETE_CHARS to count as real content. Calibrated to sit between an
# unrendered SPA shell (~0) and genuine content (models.dev static ≈730, census
# ≈4900), and at the tiny-page bar so a single long content line still counts.
_MIN_SUBSTANTIVE_CHARS = 200

# Confidence at/above which the reflect loop considers the answer done.
_DONE_CONFIDENCE = 0.6

DiscoverFn = Callable[..., Awaitable["DiscoverResult"]]
PlannerFn = Callable[..., Awaitable[ResearchPlan]]
ExtractorFn = Callable[..., Awaitable[list[ResearchRow]]]


def _substantive_chars(text: str) -> int:
    """Characters that live in prose/data-length lines — the signal that a page
    carries real content and not just navigation chrome. A client-side-rendered
    page that hasn't executed its JS returns only its shell (a list of short nav
    labels), which scores ~0 here even when it clears :data:`_INCOMPLETE_CHARS`."""
    return sum(
        len(stripped)
        for line in text.splitlines()
        if len(stripped := line.strip()) >= _PROSE_LINE_CHARS
    )


def _looks_incomplete(page: FetchedPage) -> bool:
    if not page.has_content():
        return True
    text = page.text
    if len(text) < _INCOMPLETE_CHARS:
        return True
    if any(marker in text.lower()[:4000] for marker in _JS_GATE_MARKERS):
        return True
    # Over the size bar but almost all short nav labels → an unrendered SPA shell
    # (e.g. a 407-char menu with zero model data). Escalate to a JS-rendering rung,
    # which the OSS static tier is not. This is what distinguishes "407 chars of
    # navigation" from "407 chars of answer".
    return _substantive_chars(text) < _MIN_SUBSTANTIVE_CHARS


def _row_key(values: dict, cols: list[str]) -> str:
    keys = cols or list(values.keys())
    return "|".join(str(values.get(c, "")).strip().lower() for c in keys)


def _note_stage(
    trace: ResearchTrace,
    stage: str,
    *,
    detail: str = "",
    elapsed_ms: float = 0.0,
    cost_usd: float = 0.0,
    ok: bool = True,
    **meta: object,
) -> None:
    """Record one metered action on the run's trace AND emit it as a structured
    log line, so per-stage cost/latency is observable both in the response
    (``ResearchResult.trace``) and in ops logs — tagged with the requesting
    ``medium`` (cli / explorer / mcp / …) in both places."""
    trace.add(
        stage, detail=detail, elapsed_ms=elapsed_ms, cost_usd=cost_usd, ok=ok, **meta
    )
    logger.info(
        "research_stage",
        stage=stage,
        medium=trace.medium,
        detail=detail,
        elapsed_ms=round(elapsed_ms, 1),
        cost_usd=round(cost_usd, 6),
        ok=ok,
        **meta,
    )


class WebResearchHarness:
    def __init__(
        self,
        *,
        openrouter_key: str = "",
        plan_model: Optional[str] = None,
        extract_model: Optional[str] = None,
        fetchers: Optional[list[PageFetcher]] = None,
        verifier: Optional[ResearchVerifier] = None,
        discover: Optional[DiscoverFn] = None,
        planner: Optional[PlannerFn] = None,
        extractor: Optional[ExtractorFn] = None,
    ) -> None:
        self.key = openrouter_key
        self.plan_model = plan_model
        self.extract_model = extract_model
        self._fetchers = fetchers if fetchers is not None else default_ladder()
        self._verifier = verifier if verifier is not None else get_research_verifier()
        self._discover_fn = discover if discover is not None else _default_discover()
        self._planner = planner
        self._extractor = extractor
        # Marginal cost of ONE discovery call, read from the registered provider's
        # declared pricing — known only for the default binding (an injected
        # ``discover`` carries no cost signal, so it meters as free).
        self._discover_cost = 0.0
        if discover is None:
            provider = get_web_source()
            if provider is not None:
                _, self._discover_cost = provider_cost(provider)

    # -- stage callables (imported lazily so a bare import of this module doesn't
    #    require the LLM-stage siblings; also lets tests inject fakes) ---------- #
    async def _plan(self, question, hint_columns, seed_urls, budget) -> ResearchPlan:
        planner = self._planner
        if planner is None:
            from cograph_client.research.plan import plan_research

            planner = plan_research
        return await planner(
            question,
            hint_columns=hint_columns,
            seed_urls=seed_urls,
            openrouter_key=self.key,
            model=self.plan_model,
            budget=budget,
        )

    async def _extract(self, pages, schema, question, max_rows, budget):
        extractor = self._extractor
        if extractor is None:
            from cograph_client.research.extract import extract_rows

            extractor = extract_rows
        return await extractor(
            pages,
            schema,
            question=question,
            openrouter_key=self.key,
            model=self.extract_model,
            max_rows=max_rows,
            budget=budget,
        )

    async def run(
        self,
        question: str,
        *,
        schema: Optional[TargetSchema] = None,
        hint_columns: Optional[list[str]] = None,
        seed_urls: Optional[list[str]] = None,
        urls: Optional[list[str]] = None,
        max_rows: int = 200,
        budget: Optional[Budget] = None,
        context: Optional[dict] = None,
    ) -> ResearchResult:
        budget = (budget or Budget()).start()
        context = context or {}
        seed_urls = list(seed_urls or [])
        user_urls = list(urls or [])
        # Per-run cost/latency trace, tagged with the requesting interface
        # (cli / explorer / mcp / …) as threaded through the canonical /agent
        # request context.
        trace = ResearchTrace(medium=str(context.get("medium", "") or ""))
        # A caller-pinned schema IS the disambiguation — never ask in that case.
        pinned_schema = schema is not None

        # 1. Plan (schema-first) unless the caller pinned a schema. The planner
        # sees EVERY caller-supplied URL (seed_urls + urls) — a planner blind to
        # the user's URLs would ask "which page?" about a page it was handed.
        if schema is None:
            plan_urls = self._dedupe_urls(seed_urls + user_urls)
            llm0 = budget.llm_calls_used
            t0 = time.monotonic()
            plan_ok = True
            try:
                plan = await self._plan(question, hint_columns, plan_urls, budget)
            except Exception as exc:  # planner must never sink the run
                logger.warning("research_plan_failed", error=str(exc))
                plan = ResearchPlan(question=question, queries=[question])
                plan_ok = False
            _note_stage(
                trace,
                "plan",
                detail=self.plan_model or "primary",
                elapsed_ms=(time.monotonic() - t0) * 1000,
                ok=plan_ok,
                llm_calls=budget.llm_calls_used - llm0,
                needs_clarification=plan.needs_clarification,
            )
            schema = plan.schema
        else:
            plan = ResearchPlan(
                question=question, schema=schema, queries=[question], seed_urls=seed_urls
            )

        # A genuinely ambiguous question: ask rather than guess. Return the
        # planner's questions immediately — no discovery, fetch, or LLM spend
        # beyond the one planning call already made (ADR 0006 §Plan).
        if (
            not pinned_schema
            and plan.needs_clarification
            and plan.clarifying_questions
        ):
            result = clarification_result(question, plan.clarifying_questions)
            result.trace = trace
            return result

        if schema.is_empty():
            schema = TargetSchema(
                entity=schema.entity or "item", fields=[SchemaField(name="answer")]
            )
            plan.schema = schema

        cols = schema.field_names()
        all_rows: list[ResearchRow] = []
        pages: list[FetchedPage] = []
        extracted_from: set[str] = set()
        seen_urls: set[str] = set()
        sources_consulted: list[str] = []

        candidate_urls = self._dedupe_urls(
            user_urls + seed_urls + list(plan.seed_urls)
        )
        queries = plan.queries or [question]

        outcome = VerifyOutcome(abstained=True)
        iterations = 0
        max_iters = max(1, budget.max_iterations)
        while iterations < max_iters:
            iterations += 1

            # 2. Discover (premium provider) → candidate rows + source URLs.
            #    Discovery is a paid/metered source consultation, so it is gated
            #    and charged against the fetch budget just like a page fetch —
            #    and it spends ONLY when no already-known candidate URL is still
            #    waiting to be read. The pages the user (or planner) handed us
            #    come first; a small fetch budget must never be starved by
            #    discovery before the explicit URLs are fetched. A reflect
            #    iteration re-enters here after those pages proved thin.
            pending_urls = any(u not in seen_urls for u in candidate_urls)
            if plan.needs_web and queries and not pending_urls and budget.can_fetch():
                disc_rows, disc_urls = await self._discover(
                    queries, cols, max_rows, context, budget, trace
                )
                all_rows.extend(disc_rows)
                for u in disc_urls:
                    if u not in seen_urls:
                        candidate_urls.append(u)
                candidate_urls = self._dedupe_urls(candidate_urls)

            # 3. Fetch ladder over any not-yet-seen candidate URLs.
            for url in candidate_urls:
                if url in seen_urls:
                    continue
                if not budget.can_fetch():
                    break
                seen_urls.add(url)
                sources_consulted.append(url)
                page = await self._fetch_laddered(url, question, budget, trace)
                if page is not None:
                    pages.append(page)

            # 4. Extract from freshly fetched pages.
            new_pages = [
                p for p in pages if p.has_content() and p.url not in extracted_from
            ]
            if new_pages and budget.can_call_llm():
                llm0 = budget.llm_calls_used
                t0 = time.monotonic()
                ext_ok = True
                n_rows = 0
                try:
                    ext = await self._extract(
                        new_pages, schema, question, max_rows, budget
                    )
                    all_rows.extend(ext)
                    n_rows = len(ext)
                except Exception as exc:
                    logger.warning("research_extract_failed", error=str(exc))
                    ext_ok = False
                extracted_from.update(p.url for p in new_pages)
                _note_stage(
                    trace,
                    "extract",
                    detail=self.extract_model or "extract-default",
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    ok=ext_ok,
                    llm_calls=budget.llm_calls_used - llm0,
                    pages=len(new_pages),
                    rows=n_rows,
                )

            merged = self._merge_rows(all_rows, cols, max_rows)

            # 5. Verify (cite-or-abstain).
            t0 = time.monotonic()
            outcome = await self._safe_verify(question, merged, pages, schema)
            _note_stage(
                trace,
                "verify",
                detail=getattr(self._verifier, "name", ""),
                elapsed_ms=(time.monotonic() - t0) * 1000,
                rows_in=len(merged),
                kept=len(outcome.rows),
                dropped=outcome.dropped,
                abstained=outcome.abstained,
                confidence=outcome.confidence,
            )

            # 6. Reflect: done, or out of budget → stop; else broaden + loop.
            done = (
                not outcome.abstained
                and bool(outcome.rows)
                and outcome.confidence >= _DONE_CONFIDENCE
            )
            out_of_budget = budget.timed_out() or (
                not budget.can_fetch() and not budget.can_call_llm()
            )
            if done or out_of_budget or iterations >= max_iters:
                break
            queries = self._followup_queries(question, cols)

        complete = (
            not outcome.abstained
            and bool(outcome.rows)
            and outcome.confidence >= _DONE_CONFIDENCE
        )
        result = synthesize_result(
            question,
            schema,
            outcome,
            pages,
            iterations=iterations,
            sources_consulted=sources_consulted,
            complete=complete,
        )
        result.trace = trace
        totals = trace.totals()
        logger.info(
            "research_run",
            medium=trace.medium,
            iterations=iterations,
            abstained=result.abstained,
            confidence=result.confidence,
            rows=len(result.rows),
            cost_usd=totals["cost_usd"],
            elapsed_ms=totals["elapsed_ms"],
            fetches_used=budget.fetches_used,
            llm_calls_used=budget.llm_calls_used,
        )
        return result

    # -- helpers --------------------------------------------------------------- #
    @staticmethod
    def _dedupe_urls(urls: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for u in urls:
            u = (u or "").strip()
            if u and u not in seen and is_fetchable_url(u):
                seen.add(u)
                out.append(u)
        return out

    async def _discover(
        self,
        queries: list[str],
        cols: list[str],
        max_rows: int,
        context: dict,
        budget: Budget,
        trace: ResearchTrace,
    ) -> tuple[list[ResearchRow], list[str]]:
        if self._discover_fn is None:
            return [], []
        rows_out: list[ResearchRow] = []
        urls_out: list[str] = []
        for q in queries[:2]:
            # Meter each provider call: discovery is a paid source consultation,
            # so stop once the budget can no longer afford one.
            if not budget.can_fetch():
                break
            budget.note_fetch(1)
            t0 = time.monotonic()
            try:
                result: DiscoverResult = await self._discover_fn(
                    q,
                    sample=False,
                    max_rows=max_rows,
                    hint_columns=cols or None,
                    context=context,
                    urls=None,
                )
            except Exception as exc:
                logger.warning("research_discover_failed", query=q, error=str(exc))
                _note_stage(
                    trace,
                    "discover",
                    detail=q,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    cost_usd=self._discover_cost,
                    ok=False,
                    error=str(exc)[:160],
                )
                continue
            _note_stage(
                trace,
                "discover",
                detail=q,
                elapsed_ms=(time.monotonic() - t0) * 1000,
                cost_usd=self._discover_cost,
                rows=len(result.rows or []),
                sources=len(result.sources or []),
            )
            for i, row in enumerate(result.rows or []):
                key = str(row.get("name", i))
                url = (result.provenance or {}).get(key, "")
                cites = [url] if url else list((result.sources or [])[:1])
                rows_out.append(
                    ResearchRow(
                        values={k: str(v) for k, v in row.items()},
                        citations=[c for c in cites if c],
                        confidence=0.5,
                    )
                )
            urls_out.extend(result.sources or [])
        return rows_out, urls_out

    async def _fetch_laddered(
        self, url: str, want: str, budget: Budget, trace: ResearchTrace
    ) -> Optional[FetchedPage]:
        """Walk the ladder cheapest-first; escalate only when a rung's page looks
        incomplete and budget remains. Returns the best page seen."""
        best: Optional[FetchedPage] = None
        for fetcher in self._fetchers:
            if not budget.can_fetch():
                break
            budget.note_fetch(1)
            _, rung_cost = fetcher_cost(fetcher)
            t0 = time.monotonic()
            try:
                page = await fetcher.fetch(url, want=want)
            except Exception as exc:  # fetchers shouldn't raise, but never trust
                logger.warning(
                    "research_fetch_raised", url=url, fetcher=fetcher.name, error=str(exc)
                )
                _note_stage(
                    trace,
                    "fetch",
                    detail=url,
                    elapsed_ms=(time.monotonic() - t0) * 1000,
                    cost_usd=rung_cost,
                    ok=False,
                    fetcher=fetcher.name,
                    tier=int(getattr(fetcher, "tier", 0)),
                    error=str(exc)[:160],
                )
                continue
            _note_stage(
                trace,
                "fetch",
                detail=url,
                elapsed_ms=(time.monotonic() - t0) * 1000,
                cost_usd=rung_cost,
                ok=page.ok if page is not None else False,
                fetcher=fetcher.name,
                tier=int(getattr(fetcher, "tier", 0)),
                chars=len(page.text) if page is not None else 0,
            )
            if page is None:
                continue
            if page.has_content():
                # Keep the page with the MOST usable text — a later rung that
                # returns a sliver must not clobber an earlier, richer read.
                if (
                    best is None
                    or not best.has_content()
                    or len(page.text) > len(best.text)
                ):
                    best = page
                if not _looks_incomplete(page):
                    return page  # good enough; don't pay for a higher rung
            elif best is None:
                best = page  # keep an error page only if nothing better seen
        return best

    @staticmethod
    def _merge_rows(
        rows: list[ResearchRow], cols: list[str], max_rows: int
    ) -> list[ResearchRow]:
        out: list[ResearchRow] = []
        by_key: dict[str, ResearchRow] = {}
        for r in rows:
            if not any(str(v).strip() for v in r.values.values()):
                continue
            # Clamp confidence to [0,1] — a discovery provider (or bug) can hand
            # us an out-of-range score that would otherwise ride onto the artifact.
            r.confidence = min(1.0, max(0.0, r.confidence))
            key = _row_key(r.values, cols)
            if key in by_key:
                existing = by_key[key]
                for c in r.citations:
                    if c and c not in existing.citations:
                        existing.citations.append(c)
                existing.confidence = max(existing.confidence, r.confidence)
                continue
            by_key[key] = r
            out.append(r)
            if len(out) >= max_rows:
                break
        return out

    async def _safe_verify(
        self, question, rows, pages, schema
    ) -> VerifyOutcome:
        try:
            return await self._verifier.verify(question, rows, pages, schema=schema)
        except Exception as exc:  # a verifier fault must not sink the whole run
            logger.warning("research_verify_failed", error=str(exc))
            return VerifyOutcome(
                rows=rows, confidence=0.3, abstained=not rows, notes="verifier error"
            )

    @staticmethod
    def _followup_queries(question: str, cols: list[str]) -> list[str]:
        extra = " ".join(cols[:3])
        out = [f"{question} full list"]
        if extra:
            out.append(f"{question} {extra}")
        return out


def _default_discover() -> Optional[DiscoverFn]:
    """Bind the registered query-discovery provider's ``discover`` as the default,
    or ``None`` when OSS ships without one (graceful degradation to URL-only)."""
    provider = get_web_source()
    if provider is None:
        return None

    async def _discover(query, **kwargs):
        return await provider.discover(query, **kwargs)

    return _discover


__all__ = ["WebResearchHarness"]
