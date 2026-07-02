"""Tests for the semantic-index write hook in graph/kg_writer.py (ONTA-181).

The hook is the FRESHNESS half of the ONTA-173 consistency model and rides
inside :func:`insert_facts` — so the regression class that matters most here is
"the KG write must NEVER fail on an index hiccup": timeout, backend error, and
disabled-by-default gating all leave the Neptune write untouched. Also covered:
the marker-driven extraction path, the COMPLETENESS re-read (docs are rebuilt
from the touched entities' full Neptune state, never from the write's own
triples — the ONTA-173 partial-doc-wipe fix), the empty-doc delete contract,
the entity cap, the non-KG graph scope guard, and the reconcile-schedule
ensure.

The fake Neptune here is STATEFUL (``_FakeNeptune.kg`` holds the graph's
current triples) because the hook's correctness now depends on what the
re-read returns, not on the write's payload: tests put the intended post-write
KG state into ``kg`` (``_write`` appends the write's triples first, exactly
like the real insert commits before the hook runs) and the hook must index
THAT state.
"""

from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock

import pytest
import structlog

import cograph_client.graph.text_markers as tm
import cograph_client.semantic.reconciler as rec
from cograph_client.graph.kg_writer import insert_facts
from cograph_client.graph.queries import kg_graph_uri, tenant_graph_uri
from cograph_client.scheduling.store import get_schedule_store, reset_schedule_store
from cograph_client.semantic.extract import (
    canonicalize_values,
    extract_semantic_chunks,
)
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
SUMMARY_PRED = "https://cograph.tech/types/Doc/attrs/summary"
ENTITY = "https://cograph.tech/entities/Doc/e1"
PROSE = (
    "The committee heard extensive testimony about the proposed changes to the "
    "watershed management plan and debated the funding formula for well over "
    "two hours before adjourning without a final vote on the matter."
)
PROSE_TAIL = (
    "A follow-up session was scheduled for the next quarter, where the revised "
    "funding formula and the amended watershed boundaries will be put to a "
    "binding vote of the full committee."
)


class _FakeNeptune:
    """Stateful Neptune stand-in for the write hook.

    Serves exactly the two reads the hook issues — the tenant textKind marker
    map, and the VALUES-scoped touched-entity re-read — from mutable state
    (``markers`` / ``kg``), so tests control the FULL KG state the hook
    rebuilds docs from (the ONTA-173 completeness contract). ``update`` only
    counts calls (the hook never parses its own INSERTs back).
    """

    def __init__(self, markers=None, kg=None):
        self.markers: dict[str, str] = dict(markers or {})
        self.kg: list[tuple[str, str, str]] = list(kg or [])
        self.queries: list[str] = []
        self.update_count = 0
        self.fail_fetch = False

    async def update(self, sparql: str) -> None:
        self.update_count += 1

    async def query(self, sparql: str) -> dict:
        self.queries.append(sparql)
        if "textKind" in sparql:
            return {
                "head": {"vars": ["attr", "kind"]},
                "results": {
                    "bindings": [
                        {
                            "attr": {"type": "uri", "value": attr},
                            "kind": {"type": "literal", "value": kind},
                        }
                        for attr, kind in self.markers.items()
                    ]
                },
            }
        # The touched-entity completeness re-read (VALUES-scoped ?e ?p ?o).
        if self.fail_fetch:
            raise RuntimeError("neptune fetch down")
        m = re.search(r"VALUES \?e \{([^}]*)\}", sparql)
        uris = set(re.findall(r"<([^>]+)>", m.group(1))) if m else set()
        rows = sorted(t for t in self.kg if t[0] in uris)  # ORDER BY ?e ?p ?o
        return {
            "head": {"vars": ["e", "p", "o"]},
            "results": {
                "bindings": [
                    {
                        "e": {"type": "uri", "value": s},
                        "p": {"type": "uri", "value": p},
                        "o": {"type": "literal", "value": o},
                    }
                    for s, p, o in rows
                ]
            },
        }

    def fetches(self) -> list[str]:
        """The entity re-read queries issued (excludes the marker-map read)."""
        return [q for q in self.queries if "VALUES ?e" in q]


async def _write(neptune: _FakeNeptune, graph: str, triples) -> None:
    """Commit-then-hook, like production: the write's triples land in the fake
    KG state FIRST (Neptune is written before the hook runs inside
    insert_facts), then the shared write path runs."""
    neptune.kg.extend(triples)
    await insert_facts(neptune, graph, list(triples))


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
    neptune = _FakeNeptune({DESC_PRED: "free_text"})

    async def run():
        await _write(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        assert await _all_rows(index) == []

    asyncio.run(run())
    # The KG write happened; the marker map was never consulted (query() is
    # only used by the semantic hook in this call path).
    assert neptune.update_count >= 1
    assert neptune.queries == []


def test_hook_enabled_writes_pending_chunks(monkeypatch):
    """Enabled via env → chunks of marked attrs land with embedding=None (the
    durable queue) and are lexically searchable instantly."""
    _enable(monkeypatch)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _FakeNeptune({DESC_PRED: "free_text"})

    async def run():
        await _write(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
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
    neptune = _FakeNeptune({})  # nothing marked

    async def run():
        await _write(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        assert await _all_rows(index) == []

    asyncio.run(run())
    assert neptune.fetches() == []  # and no entity re-read was paid for it


def test_hook_unmarked_write_does_zero_entity_fetches(monkeypatch):
    """A write that touches NO marked attrs must not pay the completeness
    re-read — the Neptune fetch only happens for writes that touch marked
    (entity, attr) docs (ONTA-173 bounded-cost constraint)."""
    _enable(monkeypatch)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    sku_pred = "https://cograph.tech/types/Doc/attrs/sku"
    neptune = _FakeNeptune({DESC_PRED: "free_text"})  # description IS marked

    async def run():
        # …but this write only touches the unmarked sku attribute.
        await _write(neptune, GRAPH, [(ENTITY, sku_pred, "SKU-001")])
        assert await _all_rows(index) == []

    asyncio.run(run())
    assert len(neptune.queries) == 1  # the marker-map read only
    assert neptune.fetches() == []


def test_hook_not_text_marker_is_a_decided_no(monkeypatch):
    """ONTA-173 decided-no overrule guard: an attr carrying the durable
    ``not_text`` marker is False-and-PRESENT in the marker map — the hook must
    NOT index it even when its values are ≥120-char prose the name-blind auto
    tier would classify FREE_TEXT. The persisted NO wins."""
    _enable(monkeypatch)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _FakeNeptune({DESC_PRED: "not_text"})
    long_prose = PROSE + " " + PROSE_TAIL  # comfortably over the auto threshold

    async def run():
        await _write(neptune, GRAPH, [(ENTITY, DESC_PRED, long_prose)])
        assert await _all_rows(index) == []

    asyncio.run(run())
    assert neptune.fetches() == []  # not marked free_text → not even fetched


def test_hook_skips_non_kg_graphs(monkeypatch):
    """A write to a non-KG graph (tenant ontology graph) has no (tenant, kg)
    scope to index under — the scope guard skips it entirely."""
    _enable(monkeypatch)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _FakeNeptune({DESC_PRED: "free_text"})

    async def run():
        await _write(neptune, tenant_graph_uri(TENANT), [(ENTITY, DESC_PRED, PROSE)])
        assert await _all_rows(index) == []

    asyncio.run(run())
    assert neptune.queries == []


# --- completeness: docs come from the touched-entity re-read (ONTA-173) --------


def test_hook_append_merges_with_existing_values_no_tail_wipe(monkeypatch):
    """THE partial-doc-wipe regression: appending a value to an existing
    multi-valued marked attr (schema_resolver's duplicate-merge path writes
    only the NEW value's triples) must yield the MERGED doc — upsert is
    replace-per-doc, so building the doc from the write's triples alone would
    wipe the previously indexed text."""
    _enable(monkeypatch)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _FakeNeptune({DESC_PRED: "free_text"})

    async def run():
        await _write(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        # Second write carries ONLY the appended value; Neptune (the fake kg)
        # now holds both values — and so must the rebuilt doc.
        await _write(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE_TAIL)])
        rows = await _all_rows(index)
        assert len(rows) == 1
        merged = canonicalize_values([PROSE, PROSE_TAIL])
        assert rows[0].chunk_text == merged
        assert PROSE in rows[0].chunk_text and PROSE_TAIL in rows[0].chunk_text

    asyncio.run(run())


def test_hook_mirrored_attr_dedups_identically_to_full_scan(monkeypatch):
    """Cross-attr dedup symmetry: a mirrored attr written in a SECOND write
    must dedup exactly like a reconciler full-scan build — same winner
    (sorted-triple order), no hook-indexes/reconcile-deletes flip-flop."""
    _enable(monkeypatch)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    markers = {DESC_PRED: "free_text", SUMMARY_PRED: "free_text"}
    neptune = _FakeNeptune(markers)

    async def run():
        # Write 1: only `summary` exists → it is the doc.
        await _write(neptune, GRAPH, [(ENTITY, SUMMARY_PRED, PROSE)])
        assert {(r.entity_uri, r.attr) for r in await _all_rows(index)} == {
            (ENTITY, "summary")
        }
        # Write 2: the mirroring `description` lands. The re-read sees BOTH
        # attrs; dedup keeps the winner a full scan would keep and deletes the
        # mirror's doc.
        await _write(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        hook_docs = {(r.entity_uri, r.attr) for r in await _all_rows(index)}

        # Reference: the reconciler's build — extract over the FULL sorted
        # triples (its scan is ORDER BY ?e ?p ?o).
        full_scan_chunks = extract_semantic_chunks(
            sorted(neptune.kg),
            tenant_id=TENANT,
            kg_name=KG,
            marked_predicates=set(markers),
        )
        scan_docs = {(c.entity_uri, c.attr) for c in full_scan_chunks}

        assert hook_docs == scan_docs == {(ENTITY, "description")}

        # And a THIRD write touching the entity again must not flip the winner
        # back (the old bug: hook indexes the mirror, reconcile ghost-deletes
        # it, repeat).
        await _write(neptune, GRAPH, [(ENTITY, SUMMARY_PRED, PROSE)])
        assert {(r.entity_uri, r.attr) for r in await _all_rows(index)} == scan_docs

    asyncio.run(run())


def test_hook_fetch_failure_skips_index_never_fails_write(monkeypatch):
    """A completeness re-read failure must SKIP the semantic index for this
    write (warning logged) — NEVER degrade to write-local partial docs (that
    is the bug the re-read fixes) and NEVER fail the KG write. Existing index
    docs stay untouched; the reconciler repairs on its cadence."""
    _enable(monkeypatch)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _FakeNeptune({DESC_PRED: "free_text"})

    async def run():
        await _write(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        before = [(r.entity_uri, r.attr, r.chunk_text) for r in await _all_rows(index)]
        assert before  # seeded

        neptune.fail_fetch = True
        updates_before = neptune.update_count
        with structlog.testing.capture_logs() as logs:
            await _write(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE_TAIL)])

        # KG write happened; index untouched (no partial-doc upsert, no delete).
        assert neptune.update_count > updates_before
        after = [(r.entity_uri, r.attr, r.chunk_text) for r in await _all_rows(index)]
        assert after == before
        events = [l for l in logs if l["event"] == "semantic_index_hook_fetch_failed"]
        assert events and events[0]["log_level"] == "warning"

    asyncio.run(run())


def test_hook_entity_cap_bounds_the_fetch_and_logs(monkeypatch):
    """The re-read is bounded: past COGRAPH_SEMANTIC_HOOK_MAX_ENTITIES touched
    entities the hook indexes the first (sorted) N, logs the cap loudly, and
    leaves the rest to the reconciler."""
    _enable(monkeypatch)
    monkeypatch.setenv("COGRAPH_SEMANTIC_HOOK_MAX_ENTITIES", "1")
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _FakeNeptune({DESC_PRED: "free_text"})
    e_a = "https://cograph.tech/entities/Doc/a"
    e_b = "https://cograph.tech/entities/Doc/b"

    async def run():
        with structlog.testing.capture_logs() as logs:
            await _write(
                neptune,
                GRAPH,
                [(e_a, DESC_PRED, PROSE), (e_b, DESC_PRED, PROSE_TAIL)],
            )
        rows = await _all_rows(index)
        # Deterministic under the cap: the sorted-first entity was indexed.
        assert {r.entity_uri for r in rows} == {e_a}
        cap_events = [
            l for l in logs if l["event"] == "semantic_index_hook_entity_cap"
        ]
        assert cap_events and cap_events[0]["log_level"] == "warning"
        assert cap_events[0]["touched_entities"] == 2
        assert cap_events[0]["cap"] == 1
        # Only the capped entity set was fetched.
        assert len(neptune.fetches()) == 1
        assert e_a in neptune.fetches()[0] and e_b not in neptune.fetches()[0]

    asyncio.run(run())


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
    neptune = _FakeNeptune({DESC_PRED: "free_text"})

    async def run():
        await asyncio.wait_for(
            _write(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)]),
            # Generous outer bound: proves insert_facts returned long before
            # the 30s hang, i.e. the hook's own timeout fired and was caught.
            timeout=5,
        )

    asyncio.run(run())
    assert neptune.update_count >= 1  # the Neptune write happened


def test_hook_backend_error_never_fails_the_write(monkeypatch):
    _enable(monkeypatch)
    register_semantic_index(_ExplodingIndex())
    neptune = _FakeNeptune({DESC_PRED: "free_text"})

    async def run():
        await _write(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])

    asyncio.run(run())  # no exception = pass
    assert neptune.update_count >= 1


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
    """A marked attr on a touched entity whose RE-READ canonical doc is EMPTY
    (all values whitespace in Neptune — e.g. a normalization replaced them) →
    delete(entity, tenant, kg_name=…, attr=…) per the ONTA-175 contract: an
    empty doc has no chunk rows to carry its key through upsert, so the hook
    must issue the delete explicitly. Note the values come from the FETCH, not
    the write: if the old prose still existed in Neptune the doc would (
    correctly) survive."""
    _enable(monkeypatch)
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _FakeNeptune({DESC_PRED: "free_text"})

    async def run():
        # Seed the doc via a first write.
        await _write(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        assert len(await _all_rows(index)) == 1
        # Simulate a value REPLACEMENT (normalization delete+insert): the KG
        # state now holds only a whitespace value for the attr.
        neptune.kg = [(ENTITY, DESC_PRED, "   ")]
        await insert_facts(neptune, GRAPH, [(ENTITY, DESC_PRED, "   ")])
        assert await _all_rows(index) == []

    asyncio.run(run())


def test_hook_empty_doc_delete_is_attr_scoped(monkeypatch):
    """Emptying one marked attr must not touch the entity's OTHER marked docs."""
    _enable(monkeypatch)
    notes_pred = "https://cograph.tech/types/Doc/attrs/notes"
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    neptune = _FakeNeptune({DESC_PRED: "free_text", notes_pred: "free_text"})

    async def run():
        await _write(
            neptune,
            GRAPH,
            [(ENTITY, DESC_PRED, PROSE), (ENTITY, notes_pred, "Some standing notes.")],
        )
        assert {r.attr for r in await _all_rows(index)} == {"description", "notes"}
        # Replace description's value with whitespace in the KG state; notes
        # stays intact in Neptune and must stay intact in the index.
        neptune.kg = [
            (ENTITY, DESC_PRED, " "),
            (ENTITY, notes_pred, "Some standing notes."),
        ]
        await insert_facts(neptune, GRAPH, [(ENTITY, DESC_PRED, " ")])
        assert {r.attr for r in await _all_rows(index)} == {"notes"}

    asyncio.run(run())


# --- reconcile-schedule ensure ---------------------------------------------------


def test_hook_ensures_per_kg_reconcile_schedule(monkeypatch):
    """The hook memoizes an ensure of the KG's recurring reconcile row — how a
    write-active KG gets periodic ghost repair without operator action."""
    _enable(monkeypatch)
    register_semantic_index(InMemorySemanticIndex())
    neptune = _FakeNeptune({DESC_PRED: "free_text"})

    async def run():
        await _write(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        store = get_schedule_store()  # the in-memory singleton (no DSN in tests)
        schedule = await store.get(rec.reconcile_schedule_id(TENANT, KG))
        assert schedule is not None
        assert schedule.action == "semantic-reconcile"
        assert schedule.kg_name == KG
        assert schedule.next_run is not None  # first run = the backfill

    asyncio.run(run())


def test_hook_fetch_failure_still_ensures_reconcile_schedule(monkeypatch):
    """Even when the re-read fails (index skipped for this write), the hook
    still ensures the KG's reconcile row — the reconciler IS the repair path
    for exactly this skip."""
    _enable(monkeypatch)
    register_semantic_index(InMemorySemanticIndex())
    neptune = _FakeNeptune({DESC_PRED: "free_text"})
    neptune.fail_fetch = True

    async def run():
        with structlog.testing.capture_logs():
            await _write(neptune, GRAPH, [(ENTITY, DESC_PRED, PROSE)])
        store = get_schedule_store()
        assert await store.get(rec.reconcile_schedule_id(TENANT, KG)) is not None

    asyncio.run(run())
