"""Deleting a KG must invalidate its precomputed type-stats.

The stats graph URI and the in-memory summary cache are both keyed by KG name,
so a KG recreated under the same name would serve the deleted graph's stale
counts unless delete busts them. Regression test for that bug (seen while
recording the live ER demo: a re-ingested `demo-live` showed the prior run's
post-resolution count instead of the fresh fragmented count).
"""
from cograph_client.api.routes.explore import _stats_graph_uri, _summary_cache

TENANT = "test-tenant"
KG = "demo-live"


def test_delete_kg_drops_stats_graph_and_busts_cache(client, mock_neptune, auth_headers):
    stats_uri = _stats_graph_uri(TENANT, KG)

    # Seed an in-memory summary as if a prior read had warmed the cache.
    cache_key = (TENANT, KG, "Person")
    _summary_cache[cache_key] = (0.0, {"entity_count": 43})
    assert cache_key in _summary_cache

    resp = client.delete(f"/graphs/{TENANT}/kgs/{KG}", headers=auth_headers)
    assert resp.status_code == 200

    # The stats graph must have been dropped (not just the data graph).
    updates = [c.args[0] for c in mock_neptune.update.call_args_list if c.args]
    assert any(
        "DROP SILENT GRAPH" in u and stats_uri in u for u in updates
    ), f"stats graph {stats_uri} was never dropped; updates={updates}"

    # And the hot cache entry must be gone.
    assert cache_key not in _summary_cache
