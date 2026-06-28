"""Unit tests for the URL-targeted mode of the web-discovery capability.

These cover WP1: when the user hands the agent explicit URLs (in the message or
via structured ctx), the web_ingest capability routes through a URL-capable
provider (``get_web_source(for_urls=True)``), passes the URLs into
``discover(..., urls=...)``, persists them in the PlanStep params so preview ==
commit, and degrades gracefully when no URL-capable provider is registered. The
plain query-discovery path stays unchanged.

No network and no LLM: providers are fakes returning canned rows, and the
entity/attribute spec is injected via plan()'s ``parsed`` hook.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from unittest.mock import MagicMock

from cograph_client.agent.capabilities import web_ingest_cap
from cograph_client.agent.capabilities.web_ingest_cap import WebIngestCapability
from cograph_client.agent.registry import AgentContext
from cograph_client.resolver.models import (
    ExtractedAttribute,
    ExtractedEntity,
    ExtractionResult,
    IngestResult,
)
from cograph_client.resolver.schema_resolver import SchemaResolver
from cograph_client.web_sources import (
    DiscoverResult,
    register_web_source,
    reset_web_sources,
)

URL_A = "https://example.com/catalog"
URL_B = "https://docs.example.org/models"


@pytest.fixture(autouse=True)
def _patch_preview(monkeypatch):
    """Make the plan-time multi-type preview deterministic so URL-routing tests
    don't depend on a real LLM extractor: a fresh ontology + a single Model type."""

    async def fake_fetch_ontology(self, graph_uri):
        return {}, {}

    async def fake_extract(self, content, content_type, existing=None):
        return ExtractionResult(
            entities=[
                ExtractedEntity(
                    type_name="Model", id="row-0",
                    attributes=[ExtractedAttribute(name="description", value="x")],
                )
            ],
            relationships=[],
        )

    monkeypatch.setattr(SchemaResolver, "_fetch_ontology", fake_fetch_ontology)
    monkeypatch.setattr(SchemaResolver, "_extract", fake_extract)

# Spec as the LLM resolver would return it (already normalized). In URL mode the
# query is still "what to pull from these pages".
CONFIRMED_SPEC = {
    "entity_type": "Model",
    "key_attribute": "name",
    "query": "models on these pages",
    "confirmed_attributes": ["description"],
    "suggested_attributes": ["description", "url"],
}


class UrlProvider:
    """URL-capable provider: when ``urls`` are passed it extracts one canned row
    per URL with provenance mapping back to the URL; otherwise it serves a plain
    query result. Records every discover call for assertions."""

    def __init__(self, *, url_only: bool = False, is_paid: bool = False, cost_per_call: float = 0.0) -> None:
        self.name = "urlfake"
        self.is_paid = is_paid
        self.cost_per_call = cost_per_call
        self.supports_urls = True
        self.url_only = url_only
        self.calls: list[dict] = []

    async def discover(self, query, *, sample, max_rows, hint_columns, context, urls=None):
        self.calls.append(
            {"query": query, "sample": sample, "max_rows": max_rows,
             "hint_columns": tuple(hint_columns or ()), "urls": list(urls or [])}
        )
        if urls:
            rows = [
                {"name": f"row-{i}", "description": f"from {u}", "url": u}
                for i, u in enumerate(urls)
            ]
            if hint_columns:
                rows = [{c: r.get(c, "unknown") for c in hint_columns} for r in rows]
            return DiscoverResult(
                rows=rows,
                provenance={r.get("name", str(i)): u for i, (r, u) in enumerate(zip(rows, urls))},
                sources=list(urls),
                estimated_total=len(urls),
                is_partial=sample,
            )
        # Query mode fallback (should not be hit in URL tests).
        rows = [{"name": "q-row", "description": "query result", "url": "https://q"}]
        if hint_columns:
            rows = [{c: r.get(c, "unknown") for c in hint_columns} for r in rows]
        return DiscoverResult(rows=rows, sources=["https://q"], estimated_total=1)


class QueryOnlyProvider:
    """A plain query provider that does NOT support URLs (no supports_urls)."""

    def __init__(self) -> None:
        self.name = "queryonly"
        self.is_paid = False
        self.cost_per_call = 0.0
        self.calls: list[dict] = []

    async def discover(self, query, *, sample, max_rows, hint_columns, context, urls=None):
        self.calls.append({"query": query, "urls": list(urls or [])})
        rows = [{"name": "q-row", "description": "query result", "url": "https://q"}]
        if hint_columns:
            rows = [{c: r.get(c, "unknown") for c in hint_columns} for r in rows]
        return DiscoverResult(rows=rows, sources=["https://q"], estimated_total=1)


def _ctx(prior_clarify: int = 0, urls: list[str] | None = None) -> AgentContext:
    ctx = AgentContext(
        tenant_id="demo-tenant",
        kg_name="models",
        neptune=MagicMock(),
        anthropic_key="sk-ant-test",
        openrouter_key="",
        extras={"prior_clarify_count": prior_clarify},
    )
    if urls is not None:
        # ctx.urls is added by a sibling WP; the capability reads it defensively.
        ctx.urls = urls
    return ctx


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_web_sources()
    yield
    reset_web_sources()


async def test_urls_in_message_route_through_for_urls_provider():
    provider = UrlProvider(url_only=True)
    register_web_source(provider)

    steps = await WebIngestCapability().plan(
        _ctx(),
        f"add the models listed on {URL_A} and {URL_B}",
        parsed=CONFIRMED_SPEC,
    )
    assert len(steps) == 1
    step = steps[0]
    assert step.action == "discover_ingest"

    # The provider was called in URL mode with the extracted URLs.
    assert provider.calls[0]["sample"] is True
    assert provider.calls[0]["urls"] == [URL_A, URL_B]

    # URLs persisted in params for execute() to re-pass (preview == commit).
    assert step.params["urls"] == [URL_A, URL_B]
    assert step.params["provider"] == "urlfake"

    # Preview surfaces the URLs as the sources consulted.
    assert URL_A in step.preview["sources"]


async def test_urls_from_ctx_route_through_for_urls_provider():
    provider = UrlProvider()
    register_web_source(provider)

    # No URLs in the message; they arrive via structured ctx.urls.
    steps = await WebIngestCapability().plan(
        _ctx(urls=[URL_A]),
        "import these into the graph",
        parsed=CONFIRMED_SPEC,
    )
    step = steps[0]
    assert step.action == "discover_ingest"
    assert provider.calls[0]["urls"] == [URL_A]
    assert step.params["urls"] == [URL_A]


async def test_no_url_capable_provider_degrades_gracefully():
    # Only a query-only provider is registered; a URL request can't be served.
    register_web_source(QueryOnlyProvider())
    steps = await WebIngestCapability().plan(
        _ctx(),
        f"parse the data on {URL_A}",
        parsed=CONFIRMED_SPEC,
    )
    assert len(steps) == 1 and steps[0].action == "answer"
    ans = steps[0].params["answer_payload"]["answer"]
    assert "URL extraction isn't enabled" in ans


async def test_url_only_provider_ignored_for_plain_query_when_query_provider_present():
    # When BOTH a query provider and a url_only extractor are registered, a plain
    # (no-URL) query must NOT pick the url_only extractor — the query provider is
    # selected. (With ONLY the url_only provider registered, the single-provider
    # backward-compat fallback still returns it; see test_both_providers below for
    # the multi-provider routing the skip protects.)
    q = QueryOnlyProvider()
    register_web_source(q)
    register_web_source(UrlProvider(url_only=True))
    steps = await WebIngestCapability().plan(
        _ctx(),
        "find a list of OpenRouter models",
        parsed=CONFIRMED_SPEC,
    )
    assert len(steps) == 1 and steps[0].action == "discover_ingest"
    assert steps[0].params["provider"] == "queryonly"


async def test_both_providers_registered_route_by_mode():
    # Query provider + url_only extractor coexist: query mode → query provider,
    # URL mode → the extractor.
    q = QueryOnlyProvider()
    u = UrlProvider(url_only=True)
    register_web_source(q)
    register_web_source(u)

    # Plain query → query provider selected.
    steps_q = await WebIngestCapability().plan(
        _ctx(), "find a list of OpenRouter models", parsed=CONFIRMED_SPEC
    )
    assert steps_q[0].params["provider"] == "queryonly"
    assert steps_q[0].params["urls"] == []

    # URL request → url extractor selected.
    steps_u = await WebIngestCapability().plan(
        _ctx(), f"add the models on {URL_A}", parsed=CONFIRMED_SPEC
    )
    assert steps_u[0].params["provider"] == "urlfake"
    assert steps_u[0].params["urls"] == [URL_A]


async def test_execute_repasses_urls_to_provider(monkeypatch):
    """preview == commit: execute() re-passes the persisted URLs to the full
    discover call and ingests the extracted rows."""
    provider = UrlProvider(url_only=True)
    register_web_source(provider)

    captured: dict = {}

    async def fake_ingest(self, content, tenant_id, content_type="text", source="", instance_graph=None):
        captured.update(content=content, content_type=content_type, source=source)
        rows = json.loads(content)
        return IngestResult(entities_extracted=len(rows), entities_resolved=len(rows))

    monkeypatch.setattr(SchemaResolver, "ingest", fake_ingest)

    spawned: dict = {}
    monkeypatch.setattr(
        web_ingest_cap, "_spawn",
        lambda coro: spawned.__setitem__("task", asyncio.ensure_future(coro)),
    )

    cap = WebIngestCapability()
    step = (
        await cap.plan(_ctx(), f"add the models on {URL_A} and {URL_B}", parsed=CONFIRMED_SPEC)
    )[0]
    ack = await cap.execute(_ctx(), step)
    assert ack["kind"] == "ack"

    await spawned["task"]

    # Full pull (sample=False) re-passed the SAME URLs persisted at plan time.
    full_call = provider.calls[-1]
    assert full_call["sample"] is False
    assert full_call["urls"] == [URL_A, URL_B]
    # The extracted rows were committed through the multi-type ingest path; the
    # JSON content round-trips back to the extracted rows.
    assert captured["content_type"] == "json"
    assert len(json.loads(captured["content"])) == 2


async def test_query_path_unchanged_when_no_urls():
    """No URLs anywhere → plain query discovery, urls param empty, provider called
    in query mode (urls=[])."""
    provider = UrlProvider()  # supports urls, but none are supplied
    register_web_source(provider)

    steps = await WebIngestCapability().plan(
        _ctx(), "find a list of OpenRouter models", parsed=CONFIRMED_SPEC
    )
    step = steps[0]
    assert step.action == "discover_ingest"
    assert step.params["urls"] == []
    # Provider invoked in query mode (no urls forwarded).
    assert provider.calls[0]["urls"] == []


# --- get_web_source selection + stub URL-mode (base/stub coverage) ----------- #


def test_get_web_source_for_urls_picks_supporting_provider():
    from cograph_client.web_sources.base import get_web_source

    register_web_source(QueryOnlyProvider())  # no supports_urls
    register_web_source(UrlProvider(url_only=True))  # supports_urls

    assert get_web_source(for_urls=True).name == "urlfake"
    # Query mode skips the url_only provider, returning the query provider.
    assert get_web_source().name == "queryonly"


def test_get_web_source_for_urls_none_when_unsupported():
    from cograph_client.web_sources.base import get_web_source

    register_web_source(QueryOnlyProvider())
    assert get_web_source(for_urls=True) is None


def test_get_web_source_backward_compatible_single_provider():
    from cograph_client.web_sources.base import get_web_source

    # A lone query provider is still returned by the no-arg convenience.
    register_web_source(QueryOnlyProvider())
    assert get_web_source() is not None


async def test_stub_url_mode_maps_provenance_to_urls():
    from cograph_client.web_sources.stub import StubWebSource

    p = StubWebSource()
    assert getattr(p, "supports_urls", False) is True
    res = await p.discover(
        "models on these pages",
        sample=False, max_rows=100, hint_columns=None, context={},
        urls=[URL_A, URL_B],
    )
    assert res.sources == [URL_A, URL_B]
    assert len(res.rows) == 2
    # provenance maps each row's natural key to the URL it came from.
    assert set(res.provenance.values()) == {URL_A, URL_B}
    # Query behavior still works (no urls → catalogue/synthesized rows).
    q = await p.discover(
        "openrouter models", sample=False, max_rows=100, hint_columns=None, context={},
    )
    assert q.sources == ["https://openrouter.ai/models"]
