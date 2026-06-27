"""Per-KG dashboard-summary stats: durable store, write-path population,
lazy backfill from the existing stats graph, and live enriching status.

The dashboard reads entity/edge counts from a relational store rather than
scanning Neptune; these tests pin the contract: the shared recompute populates
the store, delete drops the row, a KG that predates the store is backfilled from
its precomputed stats graph, and status is derived from in-flight jobs.
"""

import pytest

from cograph_client.api.routes.explore import (
    ENTITY_URI_PREFIX,
    RDF_TYPE,
    TYPE_URI_PREFIX,
    backfill_kg_summary,
    drop_kg_stats,
    recompute_kg_stats,
)
from cograph_client.api.routes.knowledge_graphs import _enriching_kgs
from cograph_client.enrichment.models import JobCategory, JobStatus
from cograph_client.graph.kg_stats_store import (
    InMemoryKgStatsStore,
    KgStats,
    get_kg_stats_store,
    reset_kg_stats_store,
)

TENANT = "test-tenant"
KG = "demo-live"


@pytest.fixture(autouse=True)
def _fresh_store():
    """Reset the process-wide store singleton around each test."""
    reset_kg_stats_store()
    yield
    reset_kg_stats_store()


async def test_inmemory_store_crud():
    store = InMemoryKgStatsStore()
    await store.upsert(
        KgStats(tenant_id=TENANT, kg_name=KG, entity_count=5, edge_count=2,
                type_breakdown={"Person": 5})
    )
    got = await store.get(TENANT, KG)
    assert got is not None
    assert (got.entity_count, got.edge_count) == (5, 2)
    assert got.type_breakdown == {"Person": 5}
    assert [r.kg_name for r in await store.list_for_tenant(TENANT)] == [KG]
    assert await store.list_for_tenant("other") == []

    # upsert overwrites
    await store.upsert(KgStats(tenant_id=TENANT, kg_name=KG, entity_count=9, edge_count=4))
    assert (await store.get(TENANT, KG)).entity_count == 9

    await store.delete(TENANT, KG)
    assert await store.get(TENANT, KG) is None


def _scan_rows(person_count: int, knows_edges: int):
    """One whole-KG scan result: Person rdf:type row + a relationship predicate."""
    return {
        "head": {"vars": ["type", "p", "cnt", "sample", "rel"]},
        "results": {"bindings": [
            {"type": {"value": TYPE_URI_PREFIX + "Person"},
             "p": {"value": RDF_TYPE},
             "cnt": {"value": str(person_count)}, "rel": {"value": "0"}},
            {"type": {"value": TYPE_URI_PREFIX + "Person"},
             "p": {"value": "https://cograph.tech/onto/knows"},
             "cnt": {"value": str(knows_edges)},
             "sample": {"value": ENTITY_URI_PREFIX + "Person/x"},
             "rel": {"value": str(knows_edges)}},
        ]},
    }


async def test_recompute_populates_store(mock_neptune):
    def route(sparql, *a, **k):
        if "GROUP BY ?type ?p" in sparql:
            return _scan_rows(person_count=10, knows_edges=4)
        return {"head": {"vars": []}, "results": {"bindings": []}}

    mock_neptune.query.side_effect = route
    await recompute_kg_stats(mock_neptune, TENANT, KG)

    row = await get_kg_stats_store().get(TENANT, KG)
    assert row is not None
    assert row.entity_count == 10          # sum of per-type entityCount
    assert row.edge_count == 4             # sum of entity-valued-object totals
    assert row.type_breakdown == {"Person": 10}


async def test_drop_kg_stats_removes_store_row(mock_neptune):
    await get_kg_stats_store().upsert(
        KgStats(tenant_id=TENANT, kg_name=KG, entity_count=10, edge_count=4)
    )
    await drop_kg_stats(mock_neptune, TENANT, KG)
    assert await get_kg_stats_store().get(TENANT, KG) is None


async def test_backfill_from_existing_stats_graph(mock_neptune):
    """A KG that predates the store is seeded from its precomputed stats graph."""
    def route(sparql, *a, **k):
        if "SELECT ?t ?ec" in sparql and "entityCount" in sparql:
            return {"head": {"vars": ["t", "ec"]}, "results": {"bindings": [
                {"t": {"value": TYPE_URI_PREFIX + "Person"}, "ec": {"value": "7"}},
                {"t": {"value": TYPE_URI_PREFIX + "Company"}, "ec": {"value": "3"}},
            ]}}
        if "SUM(?rel)" in sparql:
            return {"head": {"vars": ["total"]},
                    "results": {"bindings": [{"total": {"value": "12"}}]}}
        return {"head": {"vars": []}, "results": {"bindings": []}}

    mock_neptune.query.side_effect = route
    row = await backfill_kg_summary(mock_neptune, TENANT, KG)
    assert row is not None
    assert row.entity_count == 10              # 7 + 3
    assert row.edge_count == 12
    assert row.type_breakdown == {"Person": 7, "Company": 3}
    # persisted for next time
    assert (await get_kg_stats_store().get(TENANT, KG)).entity_count == 10


async def test_backfill_returns_none_when_no_stats_yet(mock_neptune):
    """No materialized entity counts → None (caller schedules a recompute)."""
    mock_neptune.query.return_value = {"head": {"vars": []}, "results": {"bindings": []}}
    assert await backfill_kg_summary(mock_neptune, TENANT, KG) is None
    assert await get_kg_stats_store().get(TENANT, KG) is None


class _FakeJobStore:
    def __init__(self, jobs):
        self._jobs = jobs

    async def list_for_tenant(self, tenant_id):
        return self._jobs


class _Job:
    def __init__(self, kg_name, status, category):
        self.kg_name = kg_name
        self.status = status
        self.category = category


async def test_enriching_status_from_inflight_jobs():
    jobs = [
        _Job(KG, JobStatus.running, JobCategory.enrichment),       # in-flight → enriching
        _Job("other", JobStatus.applied, JobCategory.enrichment),  # done → not
        _Job("dedup-kg", JobStatus.running, JobCategory.dedupe),   # wrong category → not
        _Job("disco", JobStatus.queued, JobCategory.discovery),    # queued discovery → enriching
    ]
    enriching = await _enriching_kgs(_FakeJobStore(jobs), TENANT)
    assert enriching == {KG, "disco"}


async def test_list_kgs_endpoint_serves_store_stats(client, mock_neptune, auth_headers):
    """GET /kgs surfaces entity/edge counts from the store with no Neptune scan."""
    await get_kg_stats_store().upsert(
        KgStats(tenant_id=TENANT, kg_name=KG, entity_count=482000, edge_count=1300000)
    )

    def route(sparql, *a, **k):
        if "kg_name" in sparql:  # the metadata listing query
            return {"head": {"vars": ["name", "desc", "count"]},
                    "results": {"bindings": [
                        {"name": {"value": KG}, "count": {"value": "999"}},
                    ]}}
        return {"head": {"vars": []}, "results": {"bindings": []}}

    mock_neptune.query.side_effect = route
    resp = client.get(f"/graphs/{TENANT}/kgs", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    kg = body[0]
    assert kg["name"] == KG
    assert kg["entity_count"] == 482000
    assert kg["edge_count"] == 1300000
    assert kg["status"] == "active"          # no in-flight job in the (empty) store
    assert kg["triple_count"] == 999
