"""Tests for the shared KG write path (graph/kg_writer.py).

This module is the single insertion + post-write housekeeping path that BOTH
ingestion and enrichment must use, so these tests pin the behaviors that keep the
two from drifting: always-batched inserts, provenance routed to the companion
graph, and the three post-write refreshes (cache-invalidate, re-embed, recompute
stats) running best-effort.
"""

import asyncio
from unittest.mock import AsyncMock

import cograph_client.api.routes.explore as explore_mod
import cograph_client.nlp.pipeline as pipeline_mod
from cograph_client.graph.kg_writer import insert_facts, refresh_after_write
from cograph_client.graph.provenance import provenance_graph_uri


def test_insert_facts_batches_and_routes_provenance():
    """Instance triples are written in batches of 500; provenance triples go to
    the companion provenance graph (never the data graph)."""

    async def run():
        neptune = AsyncMock()
        instance_graph = "https://cograph.tech/graphs/t/kg/k"
        instance_triples = [
            (f"https://cograph.tech/entities/E/{i}", "https://cograph.tech/onto/p", f"v{i}")
            for i in range(1200)
        ]
        prov_triples = [("https://cograph.tech/prov/stmt/abc", "https://cograph.tech/prov/source", "csv")]

        await insert_facts(
            neptune, instance_graph, instance_triples, provenance_triples=prov_triples,
        )

        # 1200 / 500 = 3 instance batches + 1 provenance batch.
        assert neptune.update.await_count == 4
        statements = [c.args[0] for c in neptune.update.await_args_list]
        # Provenance landed in the companion graph; instance data did not.
        prov_graph = provenance_graph_uri(instance_graph)
        assert any(prov_graph in s for s in statements)
        instance_only = [s for s in statements if prov_graph not in s]
        assert len(instance_only) == 3
        assert all(instance_graph in s for s in instance_only)

    asyncio.run(run())


def test_insert_facts_noop_on_empty():
    async def run():
        neptune = AsyncMock()
        await insert_facts(neptune, "https://g", [], provenance_triples=None)
        neptune.update.assert_not_awaited()

    asyncio.run(run())


def test_refresh_after_write_runs_all_three(monkeypatch):
    """Cache-invalidate, re-embed affected types, and recompute stats all fire
    with the right args — the housekeeping enrichment used to skip entirely."""

    async def run():
        calls = {"invalidate": [], "embed": [], "recompute": []}

        monkeypatch.setattr(
            pipeline_mod.NLQueryPipeline,
            "invalidate_cache",
            lambda graph: calls["invalidate"].append(graph),
        )

        class FakeSvc:
            async def embed_types(self, graph, types, neptune):
                calls["embed"].append((graph, list(types)))

        monkeypatch.setattr(pipeline_mod, "get_embedding_service", lambda: FakeSvc())
        monkeypatch.setattr(
            explore_mod,
            "schedule_recompute",
            lambda neptune, tenant_id, kg_name: calls["recompute"].append((tenant_id, kg_name)),
        )

        neptune = AsyncMock()
        await refresh_after_write(
            neptune, tenant_id="t", kg_name="k", affected_types={"Company"},
        )

        onto = "https://cograph.tech/graphs/t"
        assert calls["invalidate"] == [onto]
        assert calls["embed"] == [(onto, ["Company"])]
        assert calls["recompute"] == [("t", "k")]

    asyncio.run(run())


def test_refresh_after_write_skips_embed_without_types(monkeypatch):
    """No affected types → no embed call (but cache-invalidate + recompute still run)."""

    async def run():
        embedded = []
        recomputed = []
        monkeypatch.setattr(pipeline_mod.NLQueryPipeline, "invalidate_cache", lambda graph: None)

        class FakeSvc:
            async def embed_types(self, graph, types, neptune):
                embedded.append(types)

        monkeypatch.setattr(pipeline_mod, "get_embedding_service", lambda: FakeSvc())
        monkeypatch.setattr(
            explore_mod, "schedule_recompute",
            lambda neptune, tenant_id, kg_name: recomputed.append(kg_name),
        )

        await refresh_after_write(AsyncMock(), tenant_id="t", kg_name="k", affected_types=set())
        assert embedded == []
        assert recomputed == ["k"]

    asyncio.run(run())


def test_refresh_after_write_is_best_effort(monkeypatch):
    """An embedding failure must NOT propagate, and must not block the stats
    recompute that follows it."""

    async def run():
        recomputed = []
        monkeypatch.setattr(pipeline_mod.NLQueryPipeline, "invalidate_cache", lambda graph: None)

        class BadSvc:
            async def embed_types(self, graph, types, neptune):
                raise RuntimeError("embedding backend down")

        monkeypatch.setattr(pipeline_mod, "get_embedding_service", lambda: BadSvc())
        monkeypatch.setattr(
            explore_mod, "schedule_recompute",
            lambda neptune, tenant_id, kg_name: recomputed.append(kg_name),
        )

        # Should not raise.
        await refresh_after_write(AsyncMock(), tenant_id="t", kg_name="k", affected_types={"X"})
        assert recomputed == ["k"]

    asyncio.run(run())


def test_refresh_after_write_skips_recompute_without_kg(monkeypatch):
    """No kg_name (tenant-graph-only write) → no stats recompute."""

    async def run():
        recomputed = []
        monkeypatch.setattr(pipeline_mod.NLQueryPipeline, "invalidate_cache", lambda graph: None)
        monkeypatch.setattr(pipeline_mod, "get_embedding_service", lambda: None)
        monkeypatch.setattr(
            explore_mod, "schedule_recompute",
            lambda neptune, tenant_id, kg_name: recomputed.append(kg_name),
        )
        await refresh_after_write(AsyncMock(), tenant_id="t", kg_name=None, affected_types={"X"})
        assert recomputed == []

    asyncio.run(run())
