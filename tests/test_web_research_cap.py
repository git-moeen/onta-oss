"""Tests for the web-research agent capability (ONTA-166).

Covers the plan/execute contract offline: graceful degradation with nothing to
read, a research plan when the user supplies URLs, and an end-to-end execute
against a fake fetcher + fake extractor (no network, no LLM).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from cograph_client.agent.capabilities.web_research_cap import WebResearchCapability
from cograph_client.agent.registry import AgentContext, PlanStep
from cograph_client.research import extract as research_extract
from cograph_client.research.fetch import (
    FetchedPage,
    register_page_fetcher,
    reset_page_fetchers,
)
from cograph_client.research.types import ResearchRow
from cograph_client.research.verify import reset_research_verifier
from cograph_client.web_sources.base import register_web_source, reset_web_sources


@pytest.fixture(autouse=True)
def _clean_registries():
    reset_page_fetchers()
    reset_research_verifier()
    reset_web_sources()
    yield
    reset_page_fetchers()
    reset_research_verifier()
    reset_web_sources()


def _ctx(urls=None) -> AgentContext:
    return AgentContext(
        tenant_id="t1",
        kg_name="kg",
        neptune=MagicMock(),
        openrouter_key="",  # keyless → planner uses its deterministic fallback
        urls=list(urls or []),
    )


class _FakeFetcher:
    name = "static"
    tier = 0
    is_paid = False
    cost_per_call = 0.0

    async def fetch(self, url: str, *, want: str = "") -> FetchedPage:
        return FetchedPage(url=url, text="Alpha score 94. " * 30, ok=True)


# --- plan -------------------------------------------------------------------- #
async def test_plan_degrades_when_nothing_to_read():
    cap = WebResearchCapability()
    steps = await cap.plan(_ctx(), "what are the top TTS models by score")
    assert len(steps) == 1
    assert steps[0].action == "answer"
    payload = steps[0].params["answer_payload"]
    assert "isn't fully enabled" in payload["answer"]


async def test_plan_with_urls_returns_research_step():
    cap = WebResearchCapability()
    steps = await cap.plan(
        _ctx(urls=["https://example.com/board"]),
        "pull the scores from this page",
    )
    assert len(steps) == 1
    step = steps[0]
    assert step.action == "research"
    assert step.params["urls"] == ["https://example.com/board"]
    assert step.preview["writes_to_graph"] is False
    assert "estimated_usd" in step.cost


async def test_plan_available_via_registered_provider_without_urls():
    # A registered discovery provider makes open-web research available even with
    # no URLs supplied.
    class _Provider:
        name = "fake"

        async def discover(self, query, **kw):  # pragma: no cover - not called in plan
            from cograph_client.web_sources.base import DiscoverResult

            return DiscoverResult()

    register_web_source(_Provider())
    cap = WebResearchCapability()
    steps = await cap.plan(_ctx(), "research the S&P 500 and give me a CSV")
    assert steps[0].action == "research"


async def test_plan_asks_for_clarification_when_ambiguous(monkeypatch):
    # A genuinely ambiguous question surfaces a no-write 'answer' step carrying the
    # clarifying questions — the confirm-before-spend research step is never made.
    # Questions ride the payload STRUCTURED (question + options) for reply chips,
    # with the options also inlined in the plain-text answer.
    from cograph_client.research.types import (
        ClarifyingQuestion,
        ResearchPlan,
        SchemaField,
        TargetSchema,
    )

    async def _ambiguous_plan(instruction, **kw):
        return ResearchPlan(
            question=instruction,
            needs_clarification=True,
            clarifying_questions=[
                ClarifyingQuestion(
                    question="Best by what metric?",
                    options=["benchmark score", "price"],
                ),
                ClarifyingQuestion(question="Which modality?"),
            ],
            schema=TargetSchema(entity="model", fields=[SchemaField(name="name")]),
        )

    monkeypatch.setattr(
        "cograph_client.research.plan.plan_research", _ambiguous_plan
    )
    cap = WebResearchCapability()
    steps = await cap.plan(
        _ctx(urls=["https://example.com/board"]), "list the best models"
    )
    assert len(steps) == 1
    assert steps[0].action == "answer"  # not "research" — nothing will be spent
    payload = steps[0].params["answer_payload"]
    assert payload["clarifying_questions"] == [
        {"question": "Best by what metric?", "options": ["benchmark score", "price"]},
        {"question": "Which modality?", "options": []},
    ]
    assert "clarification" in payload["answer"].lower()
    assert "(benchmark score / price)" in payload["answer"]


async def test_paid_fetcher_shows_up_in_cost_estimate():
    class _PaidFetcher(_FakeFetcher):
        name = "render"
        tier = 2
        is_paid = True
        cost_per_call = 0.02

    register_page_fetcher(_PaidFetcher())
    cap = WebResearchCapability()
    steps = await cap.plan(_ctx(urls=["https://example.com/x"]), "read this")
    assert steps[0].cost["estimated_usd"] > 0
    assert steps[0].cost["paid_calls"] > 0


# --- execute ----------------------------------------------------------------- #
async def test_execute_runs_harness_and_returns_cited_artifact(monkeypatch):
    register_page_fetcher(_FakeFetcher())

    async def _fake_extract(pages, schema, **kw):
        return [
            ResearchRow(
                values={"name": "Alpha", "score": "94"},
                citations=[pages[0].url],
                confidence=0.9,
            )
        ]

    # The harness imports extract_rows lazily from this module — patch it there.
    monkeypatch.setattr(research_extract, "extract_rows", _fake_extract)

    cap = WebResearchCapability()
    step = PlanStep(
        capability="web_research",
        action="research",
        params={
            "question": "list the models with scores",
            "schema": {
                "entity": "model",
                "fields": [{"name": "name"}, {"name": "score"}],
            },
            "urls": ["https://example.com/board"],
            "max_rows": 50,
            "budget": {"max_iterations": 1, "max_fetches": 2, "max_llm_calls": 4},
        },
    )
    out = await cap.execute(_ctx(urls=["https://example.com/board"]), step)

    assert out["kind"] == "research_result"
    assert out["abstained"] is False
    assert len(out["rows"]) == 1
    assert out["rows"][0]["values"]["name"] == "Alpha"
    assert out["sources"] == ["https://example.com/board"]
    assert "Alpha,94," in out["artifact_csv"]
    assert out["confidence"] > 0.5


async def test_execute_abstains_without_readable_sources(monkeypatch):
    async def _empty(pages, schema, **kw):
        return []

    monkeypatch.setattr(research_extract, "extract_rows", _empty)
    register_page_fetcher(_FakeFetcher())

    cap = WebResearchCapability()
    step = PlanStep(
        capability="web_research",
        action="research",
        params={
            "question": "q",
            "schema": {"entity": "item", "fields": [{"name": "answer"}]},
            "urls": [],
            "budget": {"max_iterations": 1, "max_fetches": 2},
        },
    )
    out = await cap.execute(_ctx(), step)
    assert out["kind"] == "research_result"
    assert out["abstained"] is True
    assert out["rows"] == []
