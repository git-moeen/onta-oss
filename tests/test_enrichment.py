"""Tests for the auto-enrichment feature (lite tier)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from cograph_client.enrichment.cache import EnrichmentCache
from cograph_client.enrichment.executor import (
    EnrichmentExecutor,
    _build_select_query,
    _parse_vals,
    _values_match,
)
from cograph_client.enrichment.job_store import InMemoryJobStore
from cograph_client.enrichment.models import (
    ConflictPolicy,
    ConflictReview,
    EnrichJob,
    EnrichmentTier,
    JobStatus,
    Verdict,
)
from cograph_client.enrichment.sources.wikidata import (
    WikidataAdapter,
    _clean_label_candidates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(
    *,
    type_name: str = "Product",
    attributes: list[str] | None = None,
    policy: ConflictPolicy = ConflictPolicy.stage,
    confidence_min: float = 0.85,
) -> EnrichJob:
    return EnrichJob(
        id="job-1",
        tenant_id="test-tenant",
        kg_name="kg",
        type_name=type_name,
        attributes=attributes or ["manufacturer"],
        tier=EnrichmentTier.lite,
        status=JobStatus.queued,
        created_at=datetime.now(timezone.utc),
        conflict_policy=policy,
        confidence_min=confidence_min,
    )


def _entities_query_response(rows: list[dict]) -> dict:
    bindings = []
    for r in rows:
        b: dict = {"e": {"type": "uri", "value": r["uri"]}}
        if r.get("label") is not None:
            b["label"] = {"type": "literal", "value": r["label"]}
        if r.get("vals") is not None:
            b["vals"] = {"type": "literal", "value": r["vals"]}
        bindings.append(b)
    return {"head": {"vars": ["e", "label", "nameAttr", "vals"]}, "results": {"bindings": bindings}}


def _count_response(n: int) -> dict:
    return {
        "head": {"vars": ["n"]},
        "results": {"bindings": [{"n": {"type": "literal", "value": str(n)}}]},
    }


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------


def test_job_store_crud():
    async def run():
        store = InMemoryJobStore()
        job = _make_job()
        await store.create(job)

        got = await store.get("job-1")
        assert got is not None
        assert got.id == "job-1"

        # Update
        got.status = JobStatus.running
        await store.update(got)
        again = await store.get("job-1")
        assert again.status == JobStatus.running

        summaries = await store.list_for_tenant("test-tenant")
        assert len(summaries) == 1
        assert summaries[0].id == "job-1"

        await store.delete("job-1")
        assert await store.get("job-1") is None

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_get_put():
    async def run():
        cache = EnrichmentCache()
        # Miss
        assert await cache.get("Bosch", "manufacturer", "wikidata") is None

        v = Verdict(value="Bosch GmbH", confidence=0.95, source="wikidata")
        await cache.put("Bosch", "manufacturer", "wikidata", [v])

        # Case-insensitive on entity_label
        hit = await cache.get("bosch", "manufacturer", "wikidata")
        assert hit is not None and len(hit) == 1
        assert hit[0].value == "Bosch GmbH"

        # Different attribute → still miss
        assert await cache.get("Bosch", "country", "wikidata") is None

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Wikidata adapter
# ---------------------------------------------------------------------------


def _mk_response(payload: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    return resp


def test_wikidata_adapter_unknown_attribute_returns_empty():
    async def run():
        adapter = WikidataAdapter()
        result = await adapter.lookup("Bosch", "not_a_known_attr", {})
        assert result == []

    asyncio.run(run())


def test_wikidata_adapter_resolves_entity_id_claim():
    async def run():
        adapter = WikidataAdapter()
        # Inject a fake httpx client.
        client = AsyncMock()
        # Sequence: search → entities (claims) → entities (label for target)
        client.get.side_effect = [
            _mk_response({"search": [{"id": "Q176"}]}),
            _mk_response(
                {
                    "entities": {
                        "Q176": {
                            "claims": {
                                "P17": [
                                    {
                                        "mainsnak": {
                                            "datavalue": {
                                                "type": "wikibase-entityid",
                                                "value": {"id": "Q183"},
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            ),
            _mk_response(
                {
                    "entities": {
                        "Q183": {"labels": {"en": {"value": "Germany"}}}
                    }
                }
            ),
        ]
        adapter._client = client
        verdicts = await adapter.lookup("Bosch", "country", {})
        assert len(verdicts) == 1
        assert verdicts[0].value == "Germany"
        assert verdicts[0].source == "wikidata"
        assert verdicts[0].source_url == "https://www.wikidata.org/wiki/Q176"
        assert verdicts[0].confidence == 0.95

    asyncio.run(run())


def test_wikidata_adapter_handles_429_gracefully():
    async def run():
        adapter = WikidataAdapter()
        client = AsyncMock()
        client.get.side_effect = [_mk_response({}, status=429)]
        adapter._client = client
        verdicts = await adapter.lookup("Bosch", "country", {})
        assert verdicts == []

    asyncio.run(run())


def test_wikidata_adapter_no_search_results():
    async def run():
        adapter = WikidataAdapter()
        client = AsyncMock()
        # All 4 fallback candidates return no hits — capped at 4 search calls.
        client.get.side_effect = [_mk_response({"search": []})] * 4
        adapter._client = client
        verdicts = await adapter.lookup("ZZZNOPE", "country", {})
        assert verdicts == []

    asyncio.run(run())


def test_wikidata_label_strips_trailing_sku():
    """First search (full label) misses; SKU-stripped candidate hits.

    Confidence is reduced by 0.05 because we used the first fallback step.
    """
    async def run():
        adapter = WikidataAdapter()
        client = AsyncMock()
        # 1) original "Apple MacBook Pro M3" → empty
        # 2) "Apple MacBook Pro" → hit Q312 (Apple Inc.)
        # 3) entity claims for manufacturer (P176) → string value
        client.get.side_effect = [
            _mk_response({"search": []}),
            _mk_response({"search": [{"id": "Q312"}]}),
            _mk_response(
                {
                    "entities": {
                        "Q312": {
                            "claims": {
                                "P176": [
                                    {
                                        "mainsnak": {
                                            "datavalue": {
                                                "type": "string",
                                                "value": "Apple Inc.",
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            ),
        ]
        adapter._client = client
        verdicts = await adapter.lookup(
            "Apple MacBook Pro M3", "manufacturer", {}
        )
        assert len(verdicts) == 1
        assert verdicts[0].value == "Apple Inc."
        # Direct hit would be 0.95; one fallback step → 0.90.
        assert verdicts[0].confidence == pytest.approx(0.90)

    asyncio.run(run())


def test_wikidata_label_falls_back_to_first_two_tokens():
    """Original + SKU-strip both miss; first-2-tokens candidate hits.

    Confidence reduced by 0.10 (two fallback steps).
    """
    async def run():
        adapter = WikidataAdapter()
        client = AsyncMock()
        # Candidates for "Bosch fuel injector 0261545109":
        #   ["...", "Bosch fuel injector", "Bosch fuel", "Bosch"]
        # 1) original → empty
        # 2) "Bosch fuel injector" → empty
        # 3) "Bosch fuel" → hit Q234021
        # 4) entity claims for country (P17) → entity-id
        # 5) label for Q183 → "Germany"
        client.get.side_effect = [
            _mk_response({"search": []}),
            _mk_response({"search": []}),
            _mk_response({"search": [{"id": "Q234021"}]}),
            _mk_response(
                {
                    "entities": {
                        "Q234021": {
                            "claims": {
                                "P17": [
                                    {
                                        "mainsnak": {
                                            "datavalue": {
                                                "type": "wikibase-entityid",
                                                "value": {"id": "Q183"},
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            ),
            _mk_response(
                {
                    "entities": {
                        "Q183": {"labels": {"en": {"value": "Germany"}}}
                    }
                }
            ),
        ]
        adapter._client = client
        verdicts = await adapter.lookup(
            "Bosch fuel injector 0261545109", "country", {}
        )
        assert len(verdicts) == 1
        assert verdicts[0].value == "Germany"
        # Two fallback steps → 0.95 - 0.10 = 0.85.
        assert verdicts[0].confidence == pytest.approx(0.85)

    asyncio.run(run())


def test_wikidata_label_cleaning_unit():
    """Pure tokenizer/cleaner behavior."""
    assert _clean_label_candidates("Apple MacBook Pro M3") == [
        "Apple MacBook Pro M3",
        "Apple MacBook Pro",
        "Apple MacBook",
        "Apple",
    ]
    assert _clean_label_candidates("Bosch fuel injector 0261545109") == [
        "Bosch fuel injector 0261545109",
        "Bosch fuel injector",
        "Bosch fuel",
        "Bosch",
    ]
    # Sony case: trailing-only stripping leaves "headphones" in place;
    # SKU "WH-1000XM5" sits in the middle and is not stripped. Length is 3
    # so Candidate B (first 2 tokens) fires from the original list.
    assert _clean_label_candidates("Sony WH-1000XM5 headphones") == [
        "Sony WH-1000XM5 headphones",
        "Sony WH-1000XM5",
        "Sony",
    ]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_vals():
    assert _parse_vals("") == {}
    out = _parse_vals("p1::v1||p2::v2||p1::dup")
    assert out == {"p1": "v1", "p2": "v2"}


def test_values_match():
    assert _values_match("Bosch", "Bosch GmbH")
    assert _values_match("Germany", "germany")
    assert not _values_match("Bosch", "Siemens")
    assert not _values_match("", "Bosch")


def test_build_select_query_includes_limit_and_attrs():
    q = _build_select_query("https://g/x", "Product", ["manufacturer", "country"], 50)
    assert "<https://cograph.tech/types/Product>" in q
    assert "<https://cograph.tech/types/Product/attrs/manufacturer>" in q
    assert "<https://cograph.tech/types/Product/attrs/country>" in q
    assert "LIMIT 50" in q


# ---------------------------------------------------------------------------
# Executor end-to-end
# ---------------------------------------------------------------------------


class FakeWikidata:
    name = "wikidata"

    def __init__(self, mapping: dict[tuple[str, str], list[Verdict]]):
        self._mapping = mapping
        self.calls: list[tuple[str, str]] = []

    async def lookup(self, entity_label, attribute, context):
        self.calls.append((entity_label, attribute))
        return list(self._mapping.get((entity_label, attribute), []))


def test_executor_end_to_end_filled_verified_conflict():
    async def run():
        # Three entities: one missing manufacturer (filled), one with matching
        # value (verified), one with different value (conflict).
        mfr_pred = "https://cograph.tech/types/Product/attrs/manufacturer"
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
            {
                "uri": "https://cograph.tech/entities/Product/p2",
                "label": "Drill 18V",
                "vals": f"{mfr_pred}::Bosch",
            },
            {
                "uri": "https://cograph.tech/entities/Product/p3",
                "label": "Saw",
                "vals": f"{mfr_pred}::Acme Tools",
            },
        ]

        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")
                ],
                ("Drill 18V", "manufacturer"): [
                    Verdict(value="Bosch", confidence=0.95, source="wikidata")
                ],
                ("Saw", "manufacturer"): [
                    Verdict(value="Bosch", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["manufacturer"], policy=ConflictPolicy.stage)
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        assert final is not None
        assert final.status == JobStatus.review
        assert final.progress.total == 3
        assert final.progress.processed == 3
        assert final.progress.filled == 1
        assert final.progress.verified == 1
        assert final.progress.conflicts == 1
        # Only conflicts retained in results.
        assert len(final.results) == 1
        assert final.results[0].action == "conflict"
        assert final.results[0].existing_value == "Acme Tools"
        # No SPARQL update should happen for stage policy.
        neptune.update.assert_not_called()

    asyncio.run(run())


def test_executor_overwrite_writes_triples():
    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        assert final.status == JobStatus.applied
        # Triple insert called.
        assert neptune.update.await_count >= 1

    asyncio.run(run())


def test_executor_cache_hit_increment():
    async def run():
        mfr_pred = "https://cograph.tech/types/Product/attrs/manufacturer"
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
            {"uri": "https://cograph.tech/entities/Product/p2", "label": "Bosch", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(policy=ConflictPolicy.stage)
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        # Second entity (same label) should hit cache.
        assert final.progress.cache_hits >= 1

    asyncio.run(run())


def test_executor_no_match_when_no_verdict():
    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Unknown", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata({})
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)
        job = _make_job()
        await store.create(job)
        await executor.run(job, "test-tenant")
        final = await store.get(job.id)
        assert final.progress.filled == 0
        assert final.progress.conflicts == 0
        assert final.progress.processed == 1

    asyncio.run(run())


def test_apply_decisions_writes_accepted_only():
    async def run():
        neptune = AsyncMock()
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata({})
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)
        job = _make_job(policy=ConflictPolicy.stage)
        await store.create(job)

        decisions = [
            ConflictReview(
                entity_uri="https://cograph.tech/entities/Product/p1",
                attribute="manufacturer",
                existing_value="Acme",
                proposed=Verdict(value="Bosch", confidence=0.95, source="wikidata"),
                decision="accept",
            ),
            ConflictReview(
                entity_uri="https://cograph.tech/entities/Product/p2",
                attribute="manufacturer",
                existing_value="X",
                proposed=Verdict(value="Y", confidence=0.95, source="wikidata"),
                decision="reject",
            ),
        ]

        applied = await executor.apply_decisions(job.id, decisions)
        assert applied == 1
        neptune.update.assert_awaited()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    from cograph_client.enrichment.cache import reset_enrichment_cache
    from cograph_client.enrichment.job_store import reset_job_store

    reset_job_store()
    reset_enrichment_cache()
    yield
    reset_job_store()
    reset_enrichment_cache()


def test_post_jobs_returns_job_id(client, auth_headers, mock_neptune):
    # count_entities query returns 0 entities, plus the executor's run loop
    # query once started. We don't care about the run loop's outcome here.
    mock_neptune.query.return_value = _count_response(0)

    response = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Product",
            "attributes": ["manufacturer"],
            "kg_name": "kg",
            "tier": "lite",
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["total_entities"] == 0
    assert data["estimated_cost_usd"] == 0


def test_get_jobs_lists_jobs(client, auth_headers, mock_neptune):
    mock_neptune.query.return_value = _count_response(0)
    r = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Product",
            "attributes": ["manufacturer"],
            "kg_name": "kg",
        },
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    listing = client.get(
        "/graphs/test-tenant/enrich/jobs", headers=auth_headers
    )
    assert listing.status_code == 200
    rows = listing.json()
    ids = [j["id"] for j in rows]
    assert job_id in ids


def test_get_job_404(client, auth_headers, mock_neptune):
    response = client.get(
        "/graphs/test-tenant/enrich/jobs/does-not-exist", headers=auth_headers
    )
    assert response.status_code == 404


def test_conflicts_and_apply_flow(client, auth_headers, mock_neptune):
    """Seed a job directly, set a conflict result, then call /conflicts and /apply."""
    from cograph_client.enrichment.job_store import get_job_store
    from cograph_client.enrichment.models import RowResult

    job = _make_job(policy=ConflictPolicy.stage)
    job.tenant_id = "test-tenant"
    job.status = JobStatus.review
    verdict = Verdict(value="Bosch", confidence=0.95, source="wikidata")
    job.results = [
        RowResult(
            entity_uri="https://cograph.tech/entities/Product/p1",
            attribute="manufacturer",
            existing_value="Acme",
            verdict=verdict,
            action="conflict",
        )
    ]

    async def _seed():
        store = get_job_store()
        await store.create(job)

    asyncio.run(_seed())

    r = client.get(
        f"/graphs/test-tenant/enrich/jobs/{job.id}/conflicts", headers=auth_headers
    )
    assert r.status_code == 200
    conflicts = r.json()
    assert len(conflicts) == 1
    assert conflicts[0]["entity_uri"].endswith("/p1")

    apply_resp = client.post(
        f"/graphs/test-tenant/enrich/jobs/{job.id}/apply",
        headers=auth_headers,
        json={
            "decisions": [
                {
                    "entity_uri": "https://cograph.tech/entities/Product/p1",
                    "attribute": "manufacturer",
                    "existing_value": "Acme",
                    "proposed": verdict.model_dump(),
                    "decision": "accept",
                }
            ]
        },
    )
    assert apply_resp.status_code == 200
    assert apply_resp.json()["applied"] == 1
    assert mock_neptune.update.await_count >= 1


def test_cancel_job(client, auth_headers, mock_neptune):
    mock_neptune.query.return_value = _count_response(0)
    r = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Product",
            "attributes": ["manufacturer"],
            "kg_name": "kg",
        },
    )
    job_id = r.json()["job_id"]
    cancel = client.delete(
        f"/graphs/test-tenant/enrich/jobs/{job_id}", headers=auth_headers
    )
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Tier registry
# ---------------------------------------------------------------------------


def test_register_tier_and_get_chain():
    from cograph_client.enrichment.tiers import (
        get_chain,
        register_tier,
        reset_tiers,
    )

    reset_tiers()
    try:
        assert get_chain(EnrichmentTier.lite) == ["wikidata"]
        register_tier(EnrichmentTier.base, ["wikidata", "web"])
        assert get_chain(EnrichmentTier.base) == ["wikidata", "web"]
        # Idempotent: last write wins.
        register_tier(EnrichmentTier.base, ["wikidata"])
        assert get_chain(EnrichmentTier.base) == ["wikidata"]
        # Returned list is a copy: mutating it does not affect the registry.
        chain = get_chain(EnrichmentTier.lite)
        chain.append("mutated")
        assert get_chain(EnrichmentTier.lite) == ["wikidata"]
    finally:
        reset_tiers()


def test_executor_skips_unregistered_adapter(caplog):
    """Chain with a missing adapter name should log a warning and not fail."""
    import logging

    from cograph_client.enrichment.tiers import (
        get_chain,
        register_tier,
        reset_tiers,
    )

    async def run():
        rows = [
            {
                "uri": "https://cograph.tech/entities/Product/p1",
                "label": "Bosch",
                "vals": "",
            },
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(
                        value="Robert Bosch GmbH",
                        confidence=0.95,
                        source="wikidata",
                    )
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        register_tier(EnrichmentTier.lite, ["wikidata", "nonexistent"])
        assert get_chain(EnrichmentTier.lite) == ["wikidata", "nonexistent"]

        job = _make_job(policy=ConflictPolicy.stage)
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        # Job did not fail.
        assert final is not None
        assert final.status != JobStatus.failed
        # Wikidata produced a verdict, so the job filled the empty slot.
        assert final.progress.filled == 1

    reset_tiers()
    caplog.set_level(logging.WARNING)
    try:
        asyncio.run(run())
    finally:
        reset_tiers()


# ---------------------------------------------------------------------------
# Strategy loader
# ---------------------------------------------------------------------------


def _strategy_query_response(rows: list[dict]) -> dict:
    """Build a SPARQL response for the strategy SELECT.

    rows: list of {"subj": uri, "p": uri, "o": value}
    """
    bindings = []
    for r in rows:
        b = {
            "subj": {"type": "uri", "value": r["subj"]},
            "p": {"type": "uri", "value": r["p"]},
            "o": {"type": "literal", "value": r["o"]},
        }
        bindings.append(b)
    return {
        "head": {"vars": ["subj", "p", "o"]},
        "results": {"bindings": bindings},
    }


def test_load_strategy_returns_empty_when_no_triples():
    from cograph_client.enrichment.strategy import load_strategy

    async def run():
        neptune = AsyncMock()
        neptune.query.return_value = _strategy_query_response([])
        s = await load_strategy(neptune, "test-tenant", "LineItem")
        assert s.type_name == "LineItem"
        assert s.match_key is None
        assert s.lookup_priority is None
        assert s.attributes == {}

    asyncio.run(run())


def test_load_strategy_parses_attribute_triples():
    from cograph_client.enrichment.strategy import load_strategy

    type_uri = "https://cograph.tech/types/LineItem"
    mpn_uri = "https://cograph.tech/types/LineItem/attrs/mpn"
    brand_uri = "https://cograph.tech/types/LineItem/attrs/brand"
    onto = "https://cograph.tech/onto"

    async def run():
        neptune = AsyncMock()
        neptune.query.return_value = _strategy_query_response(
            [
                {"subj": type_uri, "p": f"{onto}/matchKey", "o": "description"},
                {"subj": type_uri, "p": f"{onto}/lookupPriority", "o": "1"},
                {"subj": mpn_uri, "p": f"{onto}/enrichmentSource", "o": "wikidata"},
                {"subj": mpn_uri, "p": f"{onto}/enrichmentSource", "o": "web"},
                {"subj": mpn_uri, "p": f"{onto}/confidenceMin", "o": "0.9"},
                {"subj": mpn_uri, "p": f"{onto}/idPattern", "o": "^[A-Z0-9-]{6,20}$"},
                {"subj": mpn_uri, "p": f"{onto}/conflictPolicy", "o": "stage"},
                {"subj": brand_uri, "p": f"{onto}/canonicalizer", "o": "title-case"},
                {"subj": brand_uri, "p": f"{onto}/alias", "o": "KN→K&N"},
                {"subj": brand_uri, "p": f"{onto}/alias", "o": "Mfg→Manufacturing"},
                # Malformed alias should be silently skipped.
                {"subj": brand_uri, "p": f"{onto}/alias", "o": "bogus-no-arrow"},
            ]
        )
        s = await load_strategy(neptune, "test-tenant", "LineItem")
        assert s.match_key == "description"
        assert s.lookup_priority == 1
        assert "mpn" in s.attributes
        mpn = s.attributes["mpn"]
        assert mpn.sources == ["wikidata", "web"]
        assert mpn.confidence_min == 0.9
        assert mpn.id_pattern == "^[A-Z0-9-]{6,20}$"
        assert mpn.conflict_policy == "stage"
        brand = s.attributes["brand"]
        assert brand.canonicalizer == "title-case"
        assert brand.aliases == {"KN": "K&N", "Mfg": "Manufacturing"}

    asyncio.run(run())


def test_aliases_resolve_conflicts_to_verified():
    """Existing brand=KN, alias KN->K&N, verdict K&N -> verified, not conflict."""
    from cograph_client.enrichment.tiers import reset_tiers

    type_uri = "https://cograph.tech/types/Product"
    brand_uri = "https://cograph.tech/types/Product/attrs/brand"
    onto = "https://cograph.tech/onto"
    brand_pred = brand_uri  # the predicate stored on the entity row

    async def run():
        rows = [
            {
                "uri": "https://cograph.tech/entities/Product/p1",
                "label": "Filter",
                "vals": f"{brand_pred}::KN",
            },
        ]
        neptune = AsyncMock()

        async def query(sparql, *args, **kwargs):
            # First call inside run() is the strategy load (tenant graph URI),
            # subsequent calls are the entity SELECT.
            if "FROM <https://cograph.tech/graphs/test-tenant>" in sparql and "alias" in sparql:
                return _strategy_query_response(
                    [
                        {"subj": brand_uri, "p": f"{onto}/alias", "o": "KN→K&N"},
                    ]
                )
            return _entities_query_response(rows)

        neptune.query.side_effect = query
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Filter", "brand"): [
                    Verdict(value="K&N", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        reset_tiers()
        job = _make_job(
            type_name="Product",
            attributes=["brand"],
            policy=ConflictPolicy.stage,
        )
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        assert final is not None
        assert final.status == JobStatus.review
        assert final.progress.verified == 1, (
            f"expected verified, got progress={final.progress}"
        )
        assert final.progress.conflicts == 0

    reset_tiers()
    try:
        asyncio.run(run())
    finally:
        reset_tiers()


def test_canonicalize_title_case_handles_ampersand():
    from cograph_client.enrichment.canonicalize import apply_canonicalizer

    assert apply_canonicalizer("title-case", "k&n filters") == "K&N Filters"
    assert apply_canonicalizer("title-case", "AT&T") == "AT&T"
    assert apply_canonicalizer("title-case", "  bosch  gmbh  ").strip() == "Bosch Gmbh"
    # Unknown canonicalizer returns value unchanged.
    assert apply_canonicalizer("nope", "anything") == "anything"
    assert apply_canonicalizer(None, "x") == "x"
    assert apply_canonicalizer("trim", "  hi  ") == "hi"
