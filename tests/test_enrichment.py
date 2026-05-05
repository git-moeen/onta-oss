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
from cograph_client.enrichment.sources.wikidata import WikidataAdapter


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
        client.get.side_effect = [_mk_response({"search": []})]
        adapter._client = client
        verdicts = await adapter.lookup("ZZZNOPE", "country", {})
        assert verdicts == []

    asyncio.run(run())


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
