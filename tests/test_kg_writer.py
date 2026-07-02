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
from cograph_client.graph.kg_writer import (
    delete_facts,
    insert_facts,
    refresh_after_write,
    rewrite_subject,
)
from cograph_client.graph.provenance import provenance_graph_uri
from cograph_client.spatiotemporal.memory import InMemorySpatioTemporalIndex
from cograph_client.spatiotemporal.protocol import SpatioTemporalFact
from cograph_client.spatiotemporal.registry import (
    register_spatiotemporal_index,
    reset_spatiotemporal_index,
)


def _count_response(n: int) -> dict:
    """A SPARQL SELECT (COUNT(*) AS ?n) response for the delete_facts count query."""
    return {"head": {"vars": ["n"]}, "results": {"bindings": [{"n": {"value": str(n)}}]}}


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


# --- delete_facts: batching, counting, provenance tombstone (ADR 0007) ---------


def test_delete_facts_batches_concrete_triples_and_counts_exactly():
    """Concrete-triple deletes go out as batched DELETE DATA (500/batch), the
    count is exact (len), and NO count query is needed."""

    async def run():
        neptune = AsyncMock()
        g = "https://cograph.tech/graphs/t/kg/k"
        triples = [
            (f"https://cograph.tech/entities/E/{i}", "https://cograph.tech/onto/p", f"v{i}")
            for i in range(1200)
        ]
        removed = await delete_facts(neptune, g, triples=triples)
        assert removed == 1200
        assert neptune.update.await_count == 3  # 1200 / 500 = 3 DELETE DATA batches
        assert neptune.query.await_count == 0  # exact count, no COUNT query
        stmts = [c.args[0] for c in neptune.update.await_args_list]
        assert all("DELETE DATA" in s and g in s for s in stmts)

    asyncio.run(run())


def test_delete_facts_predicate_scoped_uses_wildcard_and_counts():
    """An object=None triple is a predicate-scoped delete (VALUES (?s ?p)), whose
    removed count comes from a COUNT query."""

    async def run():
        neptune = AsyncMock()
        neptune.query.return_value = _count_response(2)
        g = "https://cograph.tech/graphs/t/kg/k"
        removed = await delete_facts(
            neptune, g, triples=[("e1", "p1", None), ("e1", "p2", None)]
        )
        assert removed == 2
        assert neptune.query.await_count == 1
        assert neptune.update.await_count == 1
        stmt = neptune.update.await_args_list[0].args[0]
        assert "VALUES (?s ?p)" in stmt and "DELETE { GRAPH" in stmt

    asyncio.run(run())


def test_delete_facts_whole_subject_uses_values_and_counts():
    """subjects= deletes every triple of each URI (VALUES ?s), counted via COUNT."""

    async def run():
        neptune = AsyncMock()
        neptune.query.return_value = _count_response(5)
        g = "https://cograph.tech/graphs/t/kg/k"
        removed = await delete_facts(neptune, g, subjects=["e1", "e2"])
        assert removed == 5
        assert neptune.update.await_count == 1
        stmt = neptune.update.await_args_list[0].args[0]
        assert "VALUES ?s" in stmt and "DELETE { GRAPH" in stmt

    asyncio.run(run())


def test_delete_facts_noop_on_empty():
    async def run():
        neptune = AsyncMock()
        removed = await delete_facts(neptune, "https://g")
        assert removed == 0
        neptune.update.assert_not_awaited()

    asyncio.run(run())


def test_delete_facts_writes_tombstone_when_provenance_enabled(monkeypatch):
    """With COGRAPH_PROVENANCE_ENABLED=1 a tombstone event lands in the companion
    provenance graph (never the data graph)."""

    async def run():
        monkeypatch.setenv("COGRAPH_PROVENANCE_ENABLED", "1")
        neptune = AsyncMock()
        neptune.query.return_value = _count_response(1)
        g = "https://cograph.tech/graphs/t/kg/k"
        subj = "https://cograph.tech/entities/E/1"
        await delete_facts(neptune, g, subjects=[subj], reason="unit-delete")
        stmts = [c.args[0] for c in neptune.update.await_args_list]
        prov_graph = provenance_graph_uri(g)
        prov_stmts = [s for s in stmts if prov_graph in s]
        assert prov_stmts, "a tombstone must be written to the provenance graph"
        assert any("tombstone" in s for s in prov_stmts)
        assert any(subj in s for s in prov_stmts)

    asyncio.run(run())


def test_delete_facts_no_tombstone_when_provenance_disabled(monkeypatch):
    async def run():
        monkeypatch.delenv("COGRAPH_PROVENANCE_ENABLED", raising=False)
        neptune = AsyncMock()
        neptune.query.return_value = _count_response(1)
        g = "https://cograph.tech/graphs/t/kg/k"
        await delete_facts(neptune, g, subjects=["e1"], reason="x")
        prov_graph = provenance_graph_uri(g)
        assert not any(prov_graph in c.args[0] for c in neptune.update.await_args_list)

    asyncio.run(run())


# --- rewrite_subject: two-direction move + rewrite provenance -------------------


def test_rewrite_subject_moves_both_directions_and_records_event(monkeypatch):
    async def run():
        monkeypatch.setenv("COGRAPH_PROVENANCE_ENABLED", "1")
        neptune = AsyncMock()
        g = "https://cograph.tech/graphs/t/kg/k"
        await rewrite_subject(neptune, g, "urn:old", "urn:new", reason="er-merge")
        stmts = [c.args[0] for c in neptune.update.await_args_list]
        # The merge SPARQL moves outgoing + incoming references.
        assert any(
            "DELETE { <urn:old> ?p ?o }" in s and "INSERT { <urn:new> ?p ?o }" in s
            for s in stmts
        )
        # The rewrite event lands in the provenance graph, old -> new.
        prov_graph = provenance_graph_uri(g)
        prov = [s for s in stmts if prov_graph in s]
        assert prov and any("rewrite" in s and "urn:new" in s for s in prov)

    asyncio.run(run())


def test_rewrite_subject_noop_on_same_uri():
    async def run():
        neptune = AsyncMock()
        await rewrite_subject(neptune, "g", "same", "same")
        neptune.update.assert_not_awaited()

    asyncio.run(run())


# --- refresh_after_write: derived-index eviction / re-key (ADR 0007) ------------


def _quiet_housekeeping(monkeypatch):
    """Silence the ontology-cache / embed / stats steps so a refresh test isolates
    the derived-index maintenance."""
    monkeypatch.setattr(pipeline_mod.NLQueryPipeline, "invalidate_cache", lambda g: None)
    monkeypatch.setattr(pipeline_mod, "get_embedding_service", lambda: None)
    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)


def test_refresh_after_write_evicts_deleted_subjects(monkeypatch):
    async def run():
        _quiet_housekeeping(monkeypatch)
        index = InMemorySpatioTemporalIndex()
        register_spatiotemporal_index(index)
        try:
            await index.upsert(
                SpatioTemporalFact(entity_uri="E1", tenant_id="t", kg_name="k", lon=1.0, lat=2.0)
            )
            await index.upsert(
                SpatioTemporalFact(entity_uri="E2", tenant_id="t", kg_name="k", lon=3.0, lat=4.0)
            )
            await refresh_after_write(
                AsyncMock(), tenant_id="t", kg_name="k", deleted_subjects=["E1"]
            )
            hits = await index.query_bbox("t", -180, -90, 180, 90, kg_name="k")
            assert {h.entity_uri for h in hits} == {"E2"}
        finally:
            reset_spatiotemporal_index()

    asyncio.run(run())


def test_refresh_after_write_rekeys_rewritten_subjects(monkeypatch):
    async def run():
        _quiet_housekeeping(monkeypatch)
        index = InMemorySpatioTemporalIndex()
        register_spatiotemporal_index(index)
        try:
            await index.upsert(
                SpatioTemporalFact(entity_uri="loser", tenant_id="t", kg_name="k", lon=1.0, lat=2.0)
            )
            await refresh_after_write(
                AsyncMock(), tenant_id="t", kg_name="k", rewritten_subjects={"loser": "canon"},
            )
            hits = await index.query_bbox("t", -180, -90, 180, 90, kg_name="k")
            assert {h.entity_uri for h in hits} == {"canon"}
        finally:
            reset_spatiotemporal_index()

    asyncio.run(run())


def test_refresh_after_write_evicts_deleted_subjects_from_semantic_index(monkeypatch):
    """The ONTA-173 half of the _deindex_secondary seam: deletes and rewrites
    evict semantic docs exactly like spatiotemporal rows (rewrites evict the
    stale key; re-indexing the new key is the hook/reconciler's job). Gated:
    with the env gate off the semantic backend must not even be touched."""
    from cograph_client.semantic.memory import InMemorySemanticIndex
    from cograph_client.semantic.protocol import SemanticChunk
    from cograph_client.semantic.registry import (
        register_semantic_index,
        reset_semantic_index,
    )

    async def run():
        _quiet_housekeeping(monkeypatch)
        monkeypatch.setenv("COGRAPH_SEMANTIC_INDEX_ENABLED", "true")
        sem = InMemorySemanticIndex()
        register_semantic_index(sem)
        try:
            for uri in ("E1", "E2", "loser"):
                await sem.upsert_chunks(
                    [
                        SemanticChunk(
                            tenant_id="t",
                            kg_name="k",
                            entity_uri=uri,
                            attr="desc",
                            chunk_ix=0,
                            chunk_text=f"text of {uri}",
                            content_hash=f"h-{uri}",
                        )
                    ]
                )
            await refresh_after_write(
                AsyncMock(),
                tenant_id="t",
                kg_name="k",
                deleted_subjects=["E1"],
                rewritten_subjects={"loser": "canon"},
            )
            remaining = {e for e, _a, _h, _at in await sem.list_docs("t", kg_name="k")}
            assert remaining == {"E2"}

            # Gate off: the backend must not be touched at all.
            monkeypatch.delenv("COGRAPH_SEMANTIC_INDEX_ENABLED", raising=False)

            class Exploding:
                def __getattr__(self, name):
                    raise AssertionError("semantic backend touched with gate off")

            register_semantic_index(Exploding())
            await refresh_after_write(
                AsyncMock(), tenant_id="t", kg_name="k", deleted_subjects=["E2"]
            )
        finally:
            reset_semantic_index()

    asyncio.run(run())


def test_refresh_after_write_deindex_is_noop_without_removals(monkeypatch):
    """No deleted/rewritten subjects → the derived-index step touches nothing."""

    async def run():
        _quiet_housekeeping(monkeypatch)

        class BoomIndex(InMemorySpatioTemporalIndex):
            async def delete(self, *a, **k):  # pragma: no cover - must not be called
                raise AssertionError("delete must not run without deleted_subjects")

            async def rekey(self, *a, **k):  # pragma: no cover
                raise AssertionError("rekey must not run without rewritten_subjects")

        register_spatiotemporal_index(BoomIndex())
        try:
            # Should not raise — the deindex step early-returns.
            await refresh_after_write(AsyncMock(), tenant_id="t", kg_name="k")
        finally:
            reset_spatiotemporal_index()

    asyncio.run(run())
