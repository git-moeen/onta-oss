"""Hardening tests for the web-research harness — driven by the independent
code review of ONTA-166. Cover the SSRF fixes, budget metering/stops, the
reflect loop's 2nd iteration, extract dedupe/cap, plan fallback, ladder
best-page selection, and CSV formula-injection escaping. All offline/deterministic
(fakes + httpx MockTransport + `max_wall_clock_s=0.0`, never a real sleep/network).
"""

from __future__ import annotations

import dataclasses

import httpx
import pytest

from cograph_client.research import extract as extract_mod
from cograph_client.research import fetch as fetch_mod
from cograph_client.research.fetch import (
    StaticHttpFetcher,
    is_fetchable_url,
    reset_page_fetchers,
)
from cograph_client.research.harness import WebResearchHarness, _looks_incomplete
from cograph_client.research.plan import plan_research
from cograph_client.research.types import (
    Budget,
    FetchedPage,
    ResearchPlan,
    ResearchResult,
    ResearchRow,
    SchemaField,
    TargetSchema,
)
from cograph_client.research.verify import reset_research_verifier
from cograph_client.web_sources.base import DiscoverResult, reset_web_sources


@pytest.fixture(autouse=True)
def _clean():
    reset_page_fetchers()
    reset_research_verifier()
    reset_web_sources()
    yield
    reset_page_fetchers()
    reset_research_verifier()
    reset_web_sources()


def _schema() -> TargetSchema:
    return TargetSchema(
        entity="model", fields=[SchemaField(name="name"), SchemaField(name="score")]
    )


class FakeFetcher:
    def __init__(self, default=None, *, name="static", tier=0, is_paid=False, cost_per_call=0.0):
        self.default = default
        self.name = name
        self.tier = tier
        self.is_paid = is_paid
        self.cost_per_call = cost_per_call
        self.calls = 0
        self.seen: list[str] = []

    async def fetch(self, url, *, want=""):
        self.calls += 1
        self.seen.append(url)
        if self.default is not None:
            return dataclasses.replace(self.default, url=url)
        return FetchedPage(url=url, ok=False, error="none")


# --- BLOCKER 1: SSRF IP-encoding bypasses now blocked --------------------------- #
@pytest.mark.parametrize(
    "url",
    [
        "http://2130706433/",       # decimal 127.0.0.1
        "http://0x7f000001/",       # hex
        "http://0177.0.0.1/",       # octal
        "http://127.1/",            # short form
        "http://[::1]/",            # ipv6 loopback
        "http://0.0.0.0/",          # unspecified
        "http://169.254.169.254/",  # cloud metadata
        "http://127.0.0.1./",       # trailing dot
    ],
)
def test_ssrf_blocks_ip_encodings(url):
    assert not is_fetchable_url(url)


@pytest.mark.parametrize("url", ["https://example.com/x", "http://public.example.org/a"])
def test_ssrf_allows_public_hosts_offline(url):
    # Deterministic without DNS — a real hostname is not resolved by the guard.
    assert is_fetchable_url(url)


# --- BLOCKER 2: redirects are re-validated per hop ------------------------------ #
def _mock_client_factory(handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    return factory


async def test_static_fetch_refuses_redirect_to_internal_host(monkeypatch):
    def handler(request):
        if request.url.host == "public.example.org":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/secret"})
        return httpx.Response(200, text="INTERNAL SECRET")

    monkeypatch.setattr(fetch_mod.httpx, "AsyncClient", _mock_client_factory(handler))
    page = await StaticHttpFetcher().fetch("http://public.example.org/x")
    assert not page.ok
    assert "redirect" in (page.error or "")
    assert "SECRET" not in page.text


async def test_static_fetch_follows_public_redirect(monkeypatch):
    def handler(request):
        if request.url.host == "a.example.org":
            return httpx.Response(302, headers={"location": "http://b.example.org/final"})
        return httpx.Response(
            200, text="FINAL CONTENT " * 20, headers={"content-type": "text/plain"}
        )

    monkeypatch.setattr(fetch_mod.httpx, "AsyncClient", _mock_client_factory(handler))
    page = await StaticHttpFetcher().fetch("http://a.example.org/x")
    assert page.ok
    assert "FINAL CONTENT" in page.text


async def test_static_fetch_bounds_redirect_loops(monkeypatch):
    def handler(request):
        return httpx.Response(302, headers={"location": "http://public.example.org/loop"})

    monkeypatch.setattr(fetch_mod.httpx, "AsyncClient", _mock_client_factory(handler))
    page = await StaticHttpFetcher().fetch("http://public.example.org/loop")
    assert not page.ok
    assert "too many redirects" in (page.error or "")


# --- static fetcher blocked-URL path (review F6) ------------------------------- #
async def test_static_fetch_blocked_url_is_honest():
    page = await StaticHttpFetcher().fetch("http://localhost/x")
    assert not page.ok
    assert page.error
    assert not page.has_content()
    assert page.tier == "static"


# --- MAJOR 3: discovery is metered + gated by budget --------------------------- #
async def test_discovery_is_metered_and_bounded_by_budget():
    calls = {"n": 0}

    async def _discover(query, **kw):
        calls["n"] += 1
        return DiscoverResult(rows=[], sources=[])

    async def _empty(pages, schema, **kw):
        return []

    fetcher = FakeFetcher(default=FetchedPage(url="x", ok=False))
    harness = WebResearchHarness(discover=_discover, fetchers=[fetcher], extractor=_empty)
    # Only 1 fetch of budget → at most one discovery call, then the loop stops.
    res = await harness.run(
        "q", schema=_schema(), budget=Budget(max_iterations=5, max_fetches=1)
    )
    assert calls["n"] == 1
    assert res.abstained


# --- review F3: the reflect loop actually runs a 2nd iteration ----------------- #
async def test_reflect_loop_second_iteration_recovers():
    seen_queries: list[str] = []
    state = {"n": 0}

    async def _discover(query, **kw):
        seen_queries.append(query)
        state["n"] += 1
        if state["n"] == 1:
            return DiscoverResult(rows=[], sources=[])
        return DiscoverResult(
            rows=[{"name": "A", "score": "1"}],
            provenance={"A": "https://src.example/a"},
            sources=["https://src.example/a"],
        )

    async def _empty(pages, schema, **kw):
        return []

    fetcher = FakeFetcher(default=FetchedPage(url="x", ok=False))
    harness = WebResearchHarness(discover=_discover, fetchers=[fetcher], extractor=_empty)
    res = await harness.run(
        "q", schema=_schema(), budget=Budget(max_iterations=2, max_fetches=20)
    )
    assert res.iterations == 2
    assert not res.abstained
    assert any(r.values.get("name") == "A" for r in res.rows)
    assert any("full list" in q for q in seen_queries)  # followup query fired


# --- review F4: budget wall-clock + max_llm_calls stops ------------------------ #
async def test_wall_clock_zero_stops_before_any_fetch():
    fetcher = FakeFetcher(default=FetchedPage(url="x", text="rich " * 100, ok=True))

    async def _rows(pages, schema, **kw):
        return [ResearchRow(values={"name": "A"}, citations=[pages[0].url], confidence=0.9)]

    harness = WebResearchHarness(fetchers=[fetcher], extractor=_rows)
    res = await harness.run(
        "q", schema=_schema(), urls=["https://ex.example/a"], budget=Budget(max_wall_clock_s=0.0)
    )
    assert fetcher.calls == 0
    assert res.abstained


async def test_max_llm_calls_zero_fetches_but_skips_extraction():
    fetcher = FakeFetcher(default=FetchedPage(url="x", text="rich " * 100, ok=True))

    async def _rows(pages, schema, **kw):  # would return rows if ever called
        return [ResearchRow(values={"name": "A"}, citations=[pages[0].url], confidence=0.9)]

    harness = WebResearchHarness(fetchers=[fetcher], extractor=_rows)
    res = await harness.run(
        "q",
        schema=_schema(),
        urls=["https://ex.example/a"],
        budget=Budget(max_llm_calls=0, max_fetches=5, max_iterations=1),
    )
    assert fetcher.calls == 1  # fetched
    assert res.abstained  # but extraction was budget-gated off


# --- review F5: a raising discovery provider is swallowed ---------------------- #
async def test_discovery_exception_is_swallowed():
    async def _discover(query, **kw):
        raise RuntimeError("boom")

    async def _empty(pages, schema, **kw):
        return []

    fetcher = FakeFetcher(default=FetchedPage(url="x", ok=False))
    harness = WebResearchHarness(discover=_discover, fetchers=[fetcher], extractor=_empty)
    res = await harness.run("q", schema=_schema())  # must not raise
    assert res.abstained


# --- review F7: _looks_incomplete boundary ------------------------------------- #
def test_looks_incomplete_boundary():
    assert _looks_incomplete(FetchedPage(url="u", text="x" * 199, ok=True))
    assert not _looks_incomplete(FetchedPage(url="u", text="x" * 200, ok=True))
    assert _looks_incomplete(
        FetchedPage(url="u", text="Please enable JavaScript to continue " + "y" * 300, ok=True)
    )
    assert _looks_incomplete(FetchedPage(url="u", ok=False, error="boom"))


# --- ONTA-166 local test: nav-only SPA shell must escalate --------------------- #
# A client-side-rendered page whose JS never ran (openrouter.ai/models static ≈407
# chars of pure nav) used to PASS the completeness gate — it clears
# _INCOMPLETE_CHARS but carries zero prose/data — so the ladder never escalated to
# the render rung and URL-only research abstained. Guard the substantive-char fix.
_NAV_SHELL = "\n".join(
    [
        "Chat", "Rankings", "Apps", "Models", "Providers", "Pricing",
        "Enterprise", "Docs", "API Reference", "SDK", "Status", "Discord",
        "GitHub", "Careers", "Privacy", "Terms of Service", "Support",
    ]
    * 4
)


def test_looks_incomplete_flags_nav_only_shell():
    assert len(_NAV_SHELL) > 200  # clears the size bar…
    assert _looks_incomplete(FetchedPage(url="u", text=_NAV_SHELL, ok=True))
    # …but a real paragraph of the same/greater size is complete (prose-length line).
    prose = (
        "Net international migration drove the highest US population growth in "
        "years. California remained the most populous state, followed by Texas "
        "and Florida, according to the Census Bureau's vintage 2024 estimates."
    )
    assert len(prose) > 200
    assert not _looks_incomplete(FetchedPage(url="u", text=prose, ok=True))


# --- ONTA-166 local: clarify-on-true-ambiguity -------------------------------- #
async def test_plan_emits_clarifying_questions_with_options(monkeypatch):
    async def _chat(key, system, user, **kw):
        return (
            '{"entity":"model","fields":[{"name":"name","required":true}],'
            '"needs_web":true,"fast_path":false,"queries":["best models"],'
            '"needs_clarification":true,'
            '"clarifying_questions":['
            '{"question":"Best by what metric?",'
            '"options":["benchmark score","price","popularity"]},'
            '{"question":"Which modality?","options":["LLMs","image","speech"]}],'
            '"rationale":"ambiguous"}'
        )

    monkeypatch.setattr("cograph_client.research.plan.openrouter_chat", _chat)
    plan = await plan_research("list the best models", openrouter_key="k")
    assert plan.needs_clarification is True
    assert [q.question for q in plan.clarifying_questions] == [
        "Best by what metric?",
        "Which modality?",
    ]
    assert plan.clarifying_questions[0].options == [
        "benchmark score",
        "price",
        "popularity",
    ]


async def test_plan_accepts_bare_string_clarifying_questions(monkeypatch):
    # Defensive parsing: a model that returns plain strings (the pre-options
    # shape) still works — options just come back empty (free-form).
    async def _chat(key, system, user, **kw):
        return (
            '{"entity":"item","fields":[{"name":"answer"}],"needs_web":true,'
            '"needs_clarification":true,'
            '"clarifying_questions":["Which region?", {"question":"What year?"}]}'
        )

    monkeypatch.setattr("cograph_client.research.plan.openrouter_chat", _chat)
    plan = await plan_research("q", openrouter_key="k")
    assert plan.needs_clarification is True
    assert [(q.question, q.options) for q in plan.clarifying_questions] == [
        ("Which region?", []),
        ("What year?", []),
    ]


def test_normalize_clarifying_questions_caps_and_dedupes():
    from cograph_client.research.types import normalize_clarifying_questions

    qs = normalize_clarifying_questions(
        [
            {"question": "Q1", "options": ["a", "A", " b ", "", "c", "d", "e", "f"]},
            "Q2",
            {"question": "  "},  # blank → dropped
            42,  # junk → dropped
            "Q3",
            "Q4",  # over the 3-question cap → dropped
        ]
    )
    assert [q.question for q in qs] == ["Q1", "Q2", "Q3"]
    assert qs[0].options == ["a", "b", "c", "d", "e"]  # deduped (case), capped at 5


def test_clarification_result_renders_options_inline():
    from cograph_client.research.synthesize import clarification_result

    res = clarification_result(
        "best models?",
        [{"question": "Which modality?", "options": ["LLMs", "image", "speech"]}],
    )
    assert res.needs_clarification is True
    assert "Which modality? (LLMs / image / speech)" in res.answer
    d = res.to_dict()
    assert d["clarifying_questions"] == [
        {"question": "Which modality?", "options": ["LLMs", "image", "speech"]}
    ]


async def test_plan_clarification_ignored_when_no_questions(monkeypatch):
    async def _chat(key, system, user, **kw):
        return (
            '{"entity":"item","fields":[{"name":"answer"}],"needs_web":true,'
            '"needs_clarification":true,"clarifying_questions":[]}'
        )

    monkeypatch.setattr("cograph_client.research.plan.openrouter_chat", _chat)
    plan = await plan_research("q", openrouter_key="k")
    assert plan.needs_clarification is False  # a flag with no questions → proceed


async def test_plan_fallback_never_asks_for_clarification():
    plan = await plan_research("anything", openrouter_key="")  # keyless → fallback
    assert plan.needs_clarification is False
    assert plan.clarifying_questions == []


async def test_harness_asks_for_clarification_without_web_spend():
    async def _ambiguous_planner(question, **kw):
        return ResearchPlan(
            question=question,
            needs_clarification=True,
            clarifying_questions=["By what metric — speed, cost, or quality?"],
            schema=TargetSchema(entity="item", fields=[SchemaField(name="answer")]),
        )

    fetcher = FakeFetcher(default=FetchedPage(url="u", text="x" * 500, ok=True))
    extracted: list[int] = []

    async def _extract(pages, schema, **kw):
        extracted.append(1)
        return []

    harness = WebResearchHarness(
        openrouter_key="k",
        fetchers=[fetcher],
        planner=_ambiguous_planner,
        extractor=_extract,
    )
    res = await harness.run("list the best models", urls=["https://ex.example/x"])
    assert res.needs_clarification is True
    assert [q.question for q in res.clarifying_questions] == [
        "By what metric — speed, cost, or quality?"
    ]
    assert res.abstained is False  # asking is not abstaining
    assert res.rows == []
    assert fetcher.calls == 0  # short-circuited BEFORE any fetch → no web spend
    assert extracted == []  # …and before any extraction


async def test_harness_pinned_schema_skips_clarification():
    called: list[int] = []

    async def _ambiguous_planner(question, **kw):  # pragma: no cover - must NOT run
        called.append(1)
        return ResearchPlan(
            question=question, needs_clarification=True, clarifying_questions=["?"]
        )

    fetcher = FakeFetcher(default=FetchedPage(url="u", text="rich " * 100, ok=True))

    async def _extract(pages, schema, **kw):
        return [
            ResearchRow(
                values={"name": "X", "score": "1"},
                citations=[pages[0].url],
                confidence=0.9,
            )
        ]

    harness = WebResearchHarness(
        openrouter_key="k",
        fetchers=[fetcher],
        planner=_ambiguous_planner,
        extractor=_extract,
    )
    res = await harness.run(
        "q",
        schema=_schema(),  # a pinned schema IS the disambiguation
        urls=["https://ex.example/x"],
        budget=Budget(max_iterations=1, max_fetches=2, max_llm_calls=2),
    )
    assert called == []  # planner never consulted when schema is pinned
    assert res.needs_clarification is False
    assert not res.abstained
    assert len(res.rows) == 1


async def test_ladder_escalates_past_nav_only_shell():
    url = "https://spa.example/models"
    cheap = FakeFetcher(
        default=FetchedPage(url=url, text=_NAV_SHELL, tier="static", ok=True),
        name="static",
        tier=0,
    )
    pricey = FakeFetcher(
        default=FetchedPage(
            url=url,
            text="GPT-5 offers a 400000 token context window on OpenRouter. " * 20,
            tier="render",
            ok=True,
        ),
        name="render",
        tier=2,
        is_paid=True,
        cost_per_call=0.02,
    )
    seen_tiers: list[str] = []

    async def _record(pages, schema, **kw):
        seen_tiers.extend(p.tier for p in pages)
        return [ResearchRow(values={"name": "GPT-5"}, citations=[url], confidence=0.9)]

    harness = WebResearchHarness(fetchers=[cheap, pricey], extractor=_record)
    res = await harness.run(
        "list models", schema=_schema(), urls=[url], budget=Budget(max_fetches=4)
    )
    assert pricey.calls == 1  # escalated past the >200-char nav-only shell
    assert "render" in seen_tiers  # extraction ran on the rendered page
    assert not res.abstained


# --- review F8: plan_research keyless + budget-exhausted fallback --------------- #
async def test_plan_research_keyless_fallback():
    plan = await plan_research("find models", hint_columns=["name", "score"], openrouter_key="")
    assert plan.rationale.startswith("fallback")
    assert set(["name", "score"]).issubset(set(plan.schema.field_names()))
    assert plan.queries == ["find models"]


async def test_plan_research_budget_exhausted_makes_no_call():
    b = Budget(max_llm_calls=0)
    plan = await plan_research("q", openrouter_key="would-be-used", budget=b)
    assert b.llm_calls_used == 0  # gated before the call
    assert plan.rationale.startswith("fallback")


# --- MAJOR 4 + review F9: extract dedupe (all fields) + cap -------------------- #
async def test_extract_keeps_records_sharing_a_required_field(monkeypatch):
    async def _chat(key, system, user, **kw):
        return (
            '{"rows":[{"values":{"name":"GPT","score":"90"},"confidence":0.9},'
            '{"values":{"name":"GPT","score":"85"},"confidence":0.8}]}'
        )

    monkeypatch.setattr(extract_mod, "openrouter_chat", _chat)
    schema = TargetSchema(
        entity="m",
        fields=[SchemaField(name="name", required=True), SchemaField(name="score")],
    )
    pages = [FetchedPage(url="u1", text="content " * 50, ok=True)]
    rows = await extract_mod.extract_rows(pages, schema, openrouter_key="k")
    # Two distinct records sharing the required `name` must BOTH survive (M4 fix).
    assert len(rows) == 2
    assert {r.values["score"] for r in rows} == {"90", "85"}


async def test_extract_dedupes_identical_rows_and_merges_citations(monkeypatch):
    async def _chat(key, system, user, **kw):
        return '{"rows":[{"values":{"name":"X","score":"1"},"confidence":0.9}]}'

    monkeypatch.setattr(extract_mod, "openrouter_chat", _chat)
    schema = _schema()
    pages = [
        FetchedPage(url="u1", text="c " * 50, ok=True),
        FetchedPage(url="u2", text="c " * 50, ok=True),
    ]
    rows = await extract_mod.extract_rows(pages, schema, openrouter_key="k")
    assert len(rows) == 1
    assert rows[0].citations == ["u1", "u2"]


async def test_extract_caps_rows(monkeypatch):
    async def _chat(key, system, user, **kw):
        return (
            '{"rows":[{"values":{"name":"A"}},{"values":{"name":"B"}},'
            '{"values":{"name":"C"}}]}'
        )

    monkeypatch.setattr(extract_mod, "openrouter_chat", _chat)
    pages = [FetchedPage(url="u1", text="c " * 50, ok=True)]
    rows = await extract_mod.extract_rows(pages, _schema(), openrouter_key="k", max_rows=2)
    assert len(rows) == 2


# --- MINOR 5: ladder returns the richest page, not the last -------------------- #
async def test_ladder_returns_richest_page_when_all_incomplete():
    url = "https://ex.example/js"
    big = FakeFetcher(
        default=FetchedPage(url=url, text="Please enable JavaScript " + "z" * 400, ok=True),
        name="static",
        tier=0,
    )
    tiny = FakeFetcher(
        default=FetchedPage(url=url, text="tiny", ok=True),
        name="render",
        tier=2,
        is_paid=True,
        cost_per_call=0.02,
    )
    seen_lens: list[int] = []

    async def _record(pages, schema, **kw):
        seen_lens.append(len(pages[0].text))
        return [ResearchRow(values={"name": "R"}, citations=[pages[0].url], confidence=0.8)]

    harness = WebResearchHarness(fetchers=[big, tiny], extractor=_record)
    await harness.run("q", schema=_schema(), urls=[url], budget=Budget(max_fetches=4))
    # Both rungs looked incomplete, so the richer (bigger) page must be the one
    # handed to extraction — not the 4-char one.
    assert seen_lens and max(seen_lens) > 400 - 1


# --- review F10: provenance keyed by index ------------------------------------- #
async def test_discovery_provenance_by_index():
    async def _discover(query, **kw):
        return DiscoverResult(
            rows=[{"score": "1"}],  # no "name" key → provenance keyed by index "0"
            provenance={"0": "https://src.example/a"},
            sources=["https://src.example/a"],
        )

    async def _empty(pages, schema, **kw):
        return []

    fetcher = FakeFetcher(default=FetchedPage(url="x", ok=False))
    harness = WebResearchHarness(discover=_discover, fetchers=[fetcher], extractor=_empty)
    res = await harness.run("q", schema=_schema())
    assert not res.abstained
    assert res.rows[0].citations == ["https://src.example/a"]


# --- review F11: non-http URLs are filtered from harness input ----------------- #
async def test_harness_filters_non_fetchable_input_urls():
    fetcher = FakeFetcher(default=FetchedPage(url="x", text="rich " * 100, ok=True))

    async def _rows(pages, schema, **kw):
        return [ResearchRow(values={"name": "A"}, citations=[pages[0].url], confidence=0.9)]

    harness = WebResearchHarness(fetchers=[fetcher], extractor=_rows)
    await harness.run(
        "q",
        schema=_schema(),
        urls=["ftp://x.example/a", "http://localhost/x", "https://ok.example/a"],
        budget=Budget(max_fetches=5),
    )
    assert fetcher.seen == ["https://ok.example/a"]


# --- MINOR 6: CSV formula injection is neutralized ----------------------------- #
def test_to_csv_escapes_formula_injection():
    res = ResearchResult(
        question="q",
        schema=_schema(),
        rows=[ResearchRow(values={"name": "=1+2", "score": "@SUM(A1)"}, citations=["http://u"])],
    )
    csv = res.to_csv()
    assert "'=1+2" in csv
    assert "'@SUM(A1)" in csv
    # a benign value is untouched
    res2 = ResearchResult(
        question="q",
        schema=_schema(),
        rows=[ResearchRow(values={"name": "Alpha", "score": "94"}, citations=["u"])],
    )
    assert "Alpha,94,u" in res2.to_csv()
