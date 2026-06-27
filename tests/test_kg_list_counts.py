"""`list_kgs` must serve triple counts from stored metadata, not a live scan.

Counting every triple in a KG graph is a full scan (seconds for a large KG).
The Explorer page calls `/graphs/{tenant}/kgs` on every load, so the count is
stored alongside the KG registration and read in the same metadata query. KGs
with no stored count yet fall back to a live COUNT(*) — which is then persisted
so the next read is again a single tiny lookup.
"""

import pytest

from cograph_client.api.routes.knowledge_graphs import KG_TRIPLE_COUNT
from cograph_client.graph.kg_stats_store import reset_kg_stats_store

TENANT = "test-tenant"


@pytest.fixture(autouse=True)
def _fresh_kg_stats_store():
    """Isolate the process-wide dashboard-summary store across these tests."""
    reset_kg_stats_store()
    yield
    reset_kg_stats_store()


def _binding(**vals):
    return {k: {"value": v} for k, v in vals.items()}


def _route(*, stored_count: str | None, live_count: str = "999"):
    """Steer the two query shapes list_kgs issues: the metadata list and the
    fallback COUNT(*). `stored_count=None` omits ?count so the fallback fires."""

    def route(sparql, *args, **kwargs):
        if "COUNT(*)" in sparql:
            return {
                "head": {"vars": ["c"]},
                "results": {"bindings": [_binding(c=live_count)]},
            }
        # Dashboard-summary backfill reads (per-KG stats graph): no stats
        # materialized in this test → empty, so the store row stays unset.
        if "entityCount" in sparql or "SUM(?rel)" in sparql or "forType" in sparql:
            return {"head": {"vars": []}, "results": {"bindings": []}}
        # The metadata list query.
        row = {"name": "kg-a", "desc": "A"}
        if stored_count is not None:
            row["count"] = stored_count
        return {
            "head": {"vars": ["name", "desc", "count"]},
            "results": {"bindings": [_binding(**row)]},
        }

    return route


def test_stored_count_served_without_live_scan(client, mock_neptune, auth_headers):
    mock_neptune.query.side_effect = _route(stored_count="218261")

    resp = client.get(f"/graphs/{TENANT}/kgs", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == [
        {
            "name": "kg-a",
            "description": "A",
            "triple_count": 218261,
            "entity_count": 0,
            "edge_count": 0,
            "status": "active",
            "stats_updated_at": None,
        }
    ]

    # The hot path must NOT issue a full-graph COUNT(*) when the count is stored.
    queries = [c.args[0] for c in mock_neptune.query.call_args_list if c.args]
    assert not any("COUNT(*)" in q for q in queries), (
        f"stored count should avoid a live scan; queries={queries}"
    )


def test_missing_count_falls_back_to_live_scan_and_persists(
    client, mock_neptune, auth_headers
):
    mock_neptune.query.side_effect = _route(stored_count=None, live_count="42")

    resp = client.get(f"/graphs/{TENANT}/kgs", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()[0]["triple_count"] == 42

    # Fallback path: a live COUNT(*) was issued...
    queries = [c.args[0] for c in mock_neptune.query.call_args_list if c.args]
    assert any("COUNT(*)" in q for q in queries)

    # ...and the freshly computed count was written back for next time.
    updates = [c.args[0] for c in mock_neptune.update.call_args_list if c.args]
    assert any(
        KG_TRIPLE_COUNT in u and "42" in u for u in updates
    ), f"computed count should be persisted; updates={updates}"
