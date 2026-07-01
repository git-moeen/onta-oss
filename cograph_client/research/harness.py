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

from typing import Awaitable, Callable, Optional

import structlog

from cograph_client.research.fetch import (
    FetchedPage,
    PageFetcher,
    default_ladder,
    is_fetchable_url,
)
from cograph_client.research.synthesize import synthesize_result
from cograph_client.research.types import (
    Budget,
    ResearchPlan,
    ResearchResult,
    ResearchRow,
    SchemaField,
    TargetSchema,
)
from cograph_client.research.verify import (
    ResearchVerifier,
    VerifyOutcome,
    get_research_verifier,
)
from cograph_client.web_sources.base import DiscoverResult, get_web_source

logger = structlog.stdlib.get_logger("cograph.research.harness")

# A fetched page shorter than this (or flagged as JS-gated) is treated as
# "incomplete" and the harness escalates to the next, pricier ladder rung.
_INCOMPLETE_CHARS = 200
_JS_GATE_MARKERS = ("enable javascript", "requires javascript", "please enable js")

# Confidence at/above which the reflect loop considers the answer done.
_DONE_CONFIDENCE = 0.6

DiscoverFn = Callable[..., Awaitable["DiscoverResult"]]
PlannerFn = Callable[..., Awaitable[ResearchPlan]]
ExtractorFn = Callable[..., Awaitable[list[ResearchRow]]]


def _looks_incomplete(page: FetchedPage) -> bool:
    if not page.has_content():
        return True
    text = page.text.lower()
    if len(page.text) < _INCOMPLETE_CHARS:
        return True
    head = text[:4000]
    return any(marker in head for marker in _JS_GATE_MARKERS)


def _row_key(values: dict, cols: list[str]) -> str:
    keys = cols or list(values.keys())
    return "|".join(str(values.get(c, "")).strip().lower() for c in keys)


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

        # 1. Plan (schema-first) unless the caller pinned a schema.
        if schema is None:
            try:
                plan = await self._plan(question, hint_columns, seed_urls, budget)
            except Exception as exc:  # planner must never sink the run
                logger.warning("research_plan_failed", error=str(exc))
                plan = ResearchPlan(question=question, queries=[question])
            schema = plan.schema
        else:
            plan = ResearchPlan(
                question=question, schema=schema, queries=[question], seed_urls=seed_urls
            )
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
            #    and charged against the fetch budget just like a page fetch.
            if plan.needs_web and queries and budget.can_fetch():
                disc_rows, disc_urls = await self._discover(
                    queries, cols, max_rows, context, budget
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
                page = await self._fetch_laddered(url, question, budget)
                if page is not None:
                    pages.append(page)

            # 4. Extract from freshly fetched pages.
            new_pages = [
                p for p in pages if p.has_content() and p.url not in extracted_from
            ]
            if new_pages and budget.can_call_llm():
                try:
                    ext = await self._extract(
                        new_pages, schema, question, max_rows, budget
                    )
                    all_rows.extend(ext)
                except Exception as exc:
                    logger.warning("research_extract_failed", error=str(exc))
                extracted_from.update(p.url for p in new_pages)

            merged = self._merge_rows(all_rows, cols, max_rows)

            # 5. Verify (cite-or-abstain).
            outcome = await self._safe_verify(question, merged, pages, schema)

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
        return synthesize_result(
            question,
            schema,
            outcome,
            pages,
            iterations=iterations,
            sources_consulted=sources_consulted,
            complete=complete,
        )

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
                continue
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
        self, url: str, want: str, budget: Budget
    ) -> Optional[FetchedPage]:
        """Walk the ladder cheapest-first; escalate only when a rung's page looks
        incomplete and budget remains. Returns the best page seen."""
        best: Optional[FetchedPage] = None
        for fetcher in self._fetchers:
            if not budget.can_fetch():
                break
            budget.note_fetch(1)
            try:
                page = await fetcher.fetch(url, want=want)
            except Exception as exc:  # fetchers shouldn't raise, but never trust
                logger.warning(
                    "research_fetch_raised", url=url, fetcher=fetcher.name, error=str(exc)
                )
                continue
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
