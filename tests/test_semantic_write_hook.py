"""Tests for the semantic-index write hook in graph/kg_writer.py (ONTA-181).

The hook is the FRESHNESS half of the ONTA-173 consistency model and rides
inside :func:`insert_facts` — so the regression class that matters most here is
"the KG write must NEVER fail on an index hiccup": timeout, backend error, and
disabled-by-default gating all leave the Neptune write untouched. Also covered:
the marker-driven extraction path, the empty-doc delete contract, the non-KG
graph scope guard, and the reconcile-schedule ensure.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import cograph_client.graph.text_markers as tm
import cograph_client.semantic.reconciler as rec
from cograph_client.graph.kg_writer import insert_facts
from cograph_client.graph.queries import kg_graph_uri, tenant_graph_uri
from cograph_client.scheduling.store import get_schedule_store, reset_schedule_store
from cograph_client.semantic.memory import InMemorySemanticIndex
from cograph_client.semantic.protocol import SemanticChunk
from cograph_client.semantic.registry import (
    register_semantic_index,
    reset_semantic_index,
)

TENANT = "t1"
KG = "kg1"
GRAPH = kg_graph_uri(TENANT, KG)
DESC_PRED = "https://cograph.tech/types/Doc/attrs/description"
ENTITY = "https://cograph.tech/entities/Doc/e1"
PROSE = (
    "The committee heard extensive testimony about the proposed changes to the "
    "watershed management plan and debated the funding formula for well over "
    "two hours before adjourning without a final vote on the matter."
)


def _marker_json(markers: dict[str, str]) -> dict:
    """SPARQL-results JSON for the textKind map query."""
    return {
        "head": {"vars": ["attr", "kind"]},
        "results": {
            "bindings": [
                {
                    "attr": {"type": "uri", "value": attr},
                    "kind": {"type": "literal", "value": kind},
                }
                for attr, kind in markers.items()
            ]
        },
    }


def _neptune(markers: dict[str, str] | None = None) -> AsyncMock:
    client = AsyncMock()
    client.query.return_value = _marker_json(markers or {})
    client.update.return_value = None
    return client


@pytest.fixture(autouse=True)
def _clean_state():
    reset_semantic_index()
    tm.reset_for_tests()
    rec.reset_for_tests()
    reset_schedule_store()
    yield
    reset_semantic_index()
    tm.reset_for_tests()
    rec.reset_for_tests()
    reset_schedule_store()


def _enable(monkeypatch) -> None:
    monkeypatch.setenv("COGRAPH_SEMANTIC_INDEX_ENABLED", "true")


async def _all_rows(index: InMemorySemanticIndex) -> list[SemanticChunk]:
    return await index.fetch_pending(limit=10_000)


# --- gating -------------------------------------------------------------------


def test_hook_disabled_by_default(monkeypatch):
    """No env knob → the hook is OFF: no marker fetch, no index writes — but the
    Neptune write itself proceeds normally (cost/rollout control, ONTA-181)."""
    monkeypatch.delenv("COGRAPH_SEMANTIC_INDEX_ENABLED", raising=False)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _neptune({DESC_PRED: "free_text"})

    async def run():
        await insert_facts(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        assert await _all_rows(index) == []

    asyncio.run(run())
    # The KG write happened; the marker map was never consulted (query() is
    # only used by the semantic hook in this call path).
    assert neptune.update.await_count >= 1
    assert neptune.query.await_count == 0


def test_hook_enabled_writes_pending_chunks(monkeypatch):
    """Enabled via env → chunks of marked attrs land with embedding=None (the
    durable queue) and are lexically searchable instantly."""
    _enable(monkeypatch)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _neptune({DESC_PRED: "free_text"})

    async def run():
        await insert_facts(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        rows = await _all_rows(index)
        assert len(rows) == 1
        row = rows[0]
        assert (row.tenant_id, row.kg_name) == (TENANT, KG)
        assert (row.entity_uri, row.attr) == (ENTITY, "description")
        assert row.embedding is None  # queued for the embed-fill sweep
        # Lexical search works before any embedding exists.
        hits = await index.search(TENANT, "watershed management", kg_name=KG)
        assert [h.entity_uri for h in hits.hits] == [ENTITY]

    asyncio.run(run())


def test_hook_ignores_unmarked_predicates(monkeypatch):
    """Marker-driven: an unmarked literal (a SKU is TEXT-datatyped too) is never
    indexed — marking is the gate, not the datatype."""
    _enable(monkeypatch)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _neptune({})  # nothing marked

    async def run():
        await insert_facts(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        assert await _all_rows(index) == []

    asyncio.run(run())


def test_hook_skips_non_kg_graphs(monkeypatch):
    """A write to a non-KG graph (tenant ontology graph) has no (tenant, kg)
    scope to index under — the scope guard skips it entirely."""
    _enable(monkeypatch)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _neptune({DESC_PRED: "free_text"})

    async def run():
        await insert_facts(
            neptune, tenant_graph_uri(TENANT), [(ENTITY, DESC_PRED, PROSE)]
        )
        assert await _all_rows(index) == []

    asyncio.run(run())
    assert neptune.query.await_count == 0


# --- the KG write must NEVER fail on an index hiccup ---------------------------


class _HangingIndex(InMemorySemanticIndex):
    """upsert_chunks hangs — simulates a partitioned/hung index backend."""

    async def upsert_chunks(self, chunks):  # noqa: ANN001
        await asyncio.sleep(30)


class _ExplodingIndex(InMemorySemanticIndex):
    async def upsert_chunks(self, chunks):  # noqa: ANN001
        raise RuntimeError("index backend down")


def test_hook_timeout_never_fails_the_write(monkeypatch):
    """A hung index backend is converted to a caught TimeoutError by the hook's
    own env knob (COGRAPH_SEMANTIC_UPSERT_TIMEOUT_S) — the KG write succeeds
    (same regression class as the spatio-temporal timeout guard)."""
    _enable(monkeypatch)
    monkeypatch.setenv("COGRAPH_SEMANTIC_UPSERT_TIMEOUT_S", "0.05")
    register_semantic_index(_HangingIndex())
    neptune = _neptune({DESC_PRED: "free_text"})

    async def run():
        await asyncio.wait_for(
            insert_facts(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)]),
            # Generous outer bound: proves insert_facts returned long before
            # the 30s hang, i.e. the hook's own timeout fired and was caught.
            timeout=5,
        )

    asyncio.run(run())
    assert neptune.update.await_count >= 1  # the Neptune write happened


def test_hook_backend_error_never_fails_the_write(monkeypatch):
    _enable(monkeypatch)
    register_semantic_index(_ExplodingIndex())
    neptune = _neptune({DESC_PRED: "free_text"})

    async def run():
        await insert_facts(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])

    asyncio.run(run())  # no exception = pass
    assert neptune.update.await_count >= 1


def test_hook_marker_fetch_error_never_fails_the_write(monkeypatch):
    """get_free_text_map is best-effort ({} on failure) — the write proceeds
    with nothing indexed rather than failing."""
    _enable(monkeypatch)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = AsyncMock()
    neptune.query.side_effect = RuntimeError("neptune down")
    neptune.update.return_value = None

    async def run():
        await insert_facts(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        assert await _all_rows(index) == []

    asyncio.run(run())
    assert neptune.update.await_count >= 1


# --- empty-doc delete contract --------------------------------------------------


def test_hook_empty_doc_deletes_that_attrs_rows(monkeypatch):
    """A marked attr present in the write whose canonicalized doc is EMPTY
    (whitespace values) → delete(entity, tenant, kg_name=…, attr=…) per the
    ONTA-175 contract: an empty doc has no chunk rows to carry its key through
    upsert, so the hook must issue the delete explicitly."""
    _enable(monkeypatch)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _neptune({DESC_PRED: "free_text"})

    async def run():
        # Seed the doc via a first write, then empty it via a second.
        await insert_facts(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        assert len(await _all_rows(index)) == 1
        await insert_facts(neptune, GRAPH, [(ENTITY, DESC_PRED, "   ")])
        assert await _all_rows(index) == []

    asyncio.run(run())


def test_hook_empty_doc_delete_is_attr_scoped(monkeypatch):
    """Emptying one marked attr must not touch the entity's OTHER marked docs."""
    _enable(monkeypatch)
    notes_pred = "https://cograph.tech/types/Doc/attrs/notes"
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _neptune({DESC_PRED: "free_text", notes_pred: "free_text"})

    async def run():
        await insert_facts(
            neptune,
            GRAPH,
            [(ENTITY, DESC_PRED, PROSE), (ENTITY, notes_pred, "Some standing notes.")],
        )
        assert {r.attr for r in await _all_rows(index)} == {"description", "notes"}
        await insert_facts(neptune, GRAPH, [(ENTITY, DESC_PRED, " ")])
        assert {r.attr for r in await _all_rows(index)} == {"notes"}

    asyncio.run(run())


# --- reconcile-schedule ensure ---------------------------------------------------


def test_hook_ensures_per_kg_reconcile_schedule(monkeypatch):
    """The hook memoizes an ensure of the KG's recurring reconcile row — how a
    write-active KG gets periodic ghost repair without operator action."""
    _enable(monkeypatch)
    register_semantic_index(InMemorySemanticIndex())
    neptune = _neptune({DESC_PRED: "free_text"})

    async def run():
        await insert_facts(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        store = get_schedule_store()  # the in-memory singleton (no DSN in tests)
        schedule = await store.get(rec.reconcile_schedule_id(TENANT, KG))
        assert schedule is not None
        assert schedule.action == "semantic-reconcile"
        assert schedule.kg_name == KG
        assert schedule.next_run is not None  # first run = the backfill

    asyncio.run(run())
