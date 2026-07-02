"""Tests for the claim-based semantic reconciler (ONTA-181).

Covers both duties over the InMemory backend (no DSN required):

* **Embed-fill sweep**: drain, deploy-kill-mid-fill resume (the NULL-embedding
  column IS the durable queue), poison-row dead-letter (attempt cutoff via
  ``fetch_pending(max_attempts=…)`` — the sweep is never wedged), no-key
  degrade.
* **Neptune-scan reconcile**: first-run backfill (the parliamentary-speeches
  scenario), unchanged-hash skip preserving filled embeddings, ghost deletion
  (ER merges / normalization deletes bypass the write hook), candidacy flips
  (decided-no removes rows), the reconciler-side default candidacy heuristic
  (ONTA-177 hand-off), and partial-run resume (interrupt → rerun converges).
* **Claim exclusivity**: semantic schedule rows ride the existing
  ``ScheduleRunner`` claim (advance-before-dispatch in memory; the SAME
  ``FOR UPDATE SKIP LOCKED`` SQL on Postgres), so two runner instances fire a
  due row exactly once.

Ghost enumeration uses the ``SemanticIndex.list_docs`` Protocol method, which
the real InMemory backend implements — the reconcile tests below run against
plain :class:`InMemorySemanticIndex` (ghost repair works on the OSS default,
no helper subclass). :class:`LegacyThirdPartyIndex` simulates a third-party
backend compiled against the pre-``list_docs`` Protocol to pin the
graceful-degrade branch (ghost repair skipped loudly, everything else
converges).
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest
import structlog

import cograph_client.graph.text_markers as tm
import cograph_client.nlp.embed_client as embed_client_mod
import cograph_client.semantic.reconciler as rec
from cograph_client.graph.ontology_queries import attr_uri
from cograph_client.scheduling.models import Schedule
from cograph_client.scheduling.store import InMemoryScheduleStore, reset_schedule_store
from cograph_client.semantic.memory import InMemorySemanticIndex
from cograph_client.semantic.protocol import SemanticChunk
from cograph_client.semantic.registry import reset_semantic_index

TENANT = "t1"
KG = "kg1"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
DOC_TYPE = "https://cograph.tech/types/Doc"
DESC_PRED = attr_uri("Doc", "description")
SUMMARY_PRED = attr_uri("Doc", "summary")
SKU_PRED = attr_uri("Doc", "sku")

PROSE = (
    "The committee heard extensive testimony about the proposed changes to the "
    "watershed management plan and debated the funding formula for well over "
    "two hours before adjourning without a final vote on the matter."
)


def _entity(n: int) -> str:
    return f"https://cograph.tech/entities/Doc/e{n}"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeNeptune:
    """Pattern-matching SPARQL stub for the reconciler's query shapes.

    ``entities`` is ``{entity_uri: {predicate_uri: [values]}}``; ``markers`` is
    ``{(type_name, attr_name): kind}`` and is MUTATED by textKind updates, so a
    heuristic verdict written mid-reconcile is visible to the refetch — exactly
    like the real ontology graph."""

    def __init__(self, entities: dict, markers: dict | None = None) -> None:
        self.entities = entities
        self.markers: dict[tuple[str, str], str] = dict(markers or {})
        self.updates: list[str] = []
        self.queries: list[str] = []

    @staticmethod
    def _rows(var_rows: list[dict[str, str]], variables: list[str]) -> dict:
        return {
            "head": {"vars": variables},
            "results": {
                "bindings": [
                    {k: {"type": "literal", "value": v} for k, v in row.items()}
                    for row in var_rows
                ]
            },
        }

    async def query(self, sparql: str) -> dict:
        self.queries.append(sparql)
        if "textKind" in sparql and "SELECT ?attr ?kind" in sparql:
            return self._rows(
                [
                    {"attr": attr_uri(t, a), "kind": kind}
                    for (t, a), kind in sorted(self.markers.items())
                ],
                ["attr", "kind"],
            )
        if "SELECT DISTINCT ?p" in sparql:
            preds = sorted(
                {
                    p
                    for pv in self.entities.values()
                    for p, vals in pv.items()
                    if p != RDF_TYPE and vals
                }
            )
            return self._rows([{"p": p} for p in preds], ["p"])
        if sparql.startswith("SELECT ?o FROM"):
            pred = re.search(r"\?e <([^>]+)> \?o", sparql).group(1)
            limit = int(re.search(r"LIMIT (\d+)", sparql).group(1))
            values = [
                v for pv in self.entities.values() for v in pv.get(pred, [])
            ][:limit]
            return self._rows([{"o": v} for v in values], ["o"])
        if "SELECT ?e ?p ?o" in sparql and "VALUES ?p" in sparql:
            # Keyset pagination (ONTA-173 review fix): the scan must never use
            # LIMIT/OFFSET (O(pages²) on the store) — pin that here.
            assert "OFFSET" not in sparql, "scan regressed to OFFSET paging"
            block = re.search(r"VALUES \?p \{ ([^}]*)\}", sparql).group(1)
            preds = set(re.findall(r"<([^>]+)>", block))
            limit = int(re.search(r"LIMIT (\d+)", sparql).group(1))
            m = re.search(r'FILTER\(STR\(\?e\) > "((?:[^"\\]|\\.)*)"\)', sparql)
            after = m.group(1) if m else ""
            triples = sorted(
                (e, p, v)
                for e, pv in self.entities.items()
                if e > after
                for p, vals in pv.items()
                if p in preds
                for v in vals
            )
            page = triples[:limit]
            return self._rows(
                [{"e": e, "p": p, "o": o} for e, p, o in page], ["e", "p", "o"]
            )
        raise AssertionError(f"unexpected query shape:\n{sparql}")

    async def update(self, sparql: str) -> None:
        self.updates.append(sparql)
        m = re.search(
            r"<https://cograph\.tech/types/([^/]+)/attrs/([^>]+)> "
            r"<https://cograph\.tech/onto/textKind> \"([^\"]+)\"",
            sparql,
        )
        if m:
            self.markers[(m.group(1), m.group(2))] = m.group(3)


class LegacyThirdPartyIndex:
    """A backend compiled against the pre-``list_docs`` Protocol (the frozen
    window while ONTA-181 was built): delegates the whole contract to a real
    InMemory index but genuinely LACKS ``list_docs`` — ``getattr`` on it must
    miss, exercising the reconciler's graceful-degrade branch."""

    def __init__(self) -> None:
        self._inner = InMemorySemanticIndex()

    def __getattr__(self, name: str):
        if name == "list_docs":
            raise AttributeError(name)  # the whole point of this class
        return getattr(self._inner, name)


class CrashingIndex(InMemorySemanticIndex):
    """Raises on the Nth upsert call — simulates a deploy kill mid-reconcile."""

    def __init__(self, fail_after: int) -> None:
        super().__init__()
        self.fail_after: Optional[int] = fail_after
        self.upsert_calls = 0

    async def upsert_chunks(self, chunks):  # noqa: ANN001
        self.upsert_calls += 1
        if self.fail_after is not None and self.upsert_calls > self.fail_after:
            raise RuntimeError("deploy kill")
        await super().upsert_chunks(chunks)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    reset_semantic_index()
    tm.reset_for_tests()
    rec.reset_for_tests()
    reset_schedule_store()
    monkeypatch.setenv("COGRAPH_SEMANTIC_INDEX_ENABLED", "true")
    yield
    reset_semantic_index()
    tm.reset_for_tests()
    rec.reset_for_tests()
    reset_schedule_store()


def _kg(entities: dict, markers: dict | None = None) -> FakeNeptune:
    return FakeNeptune(entities, markers)


def _doc_entities(n: int = 2) -> dict:
    return {
        _entity(i): {
            RDF_TYPE: [DOC_TYPE],
            DESC_PRED: [f"{PROSE} Session {i}."],
        }
        for i in range(1, n + 1)
    }


def _chunk(entity_n: int, text: str, *, attr: str = "description") -> SemanticChunk:
    from cograph_client.semantic.extract import content_hash

    return SemanticChunk(
        tenant_id=TENANT,
        kg_name=KG,
        entity_uri=_entity(entity_n),
        attr=attr,
        chunk_ix=0,
        chunk_text=text,
        content_hash=content_hash(text),
    )


# ---------------------------------------------------------------------------
# Reconcile: backfill, unchanged-hash skip, ghosts, candidacy
# ---------------------------------------------------------------------------


def test_first_reconcile_is_the_backfill():
    """An already-ingested KG (marked attrs, empty index) gets fully indexed by
    the FIRST reconcile run — no re-ingest (the parliamentary-speeches case)."""
    neptune = _kg(_doc_entities(3), {("Doc", "description"): "free_text"})
    index = InMemorySemanticIndex()

    async def run():
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["chunks_written"] == 3
        assert counters["ghosts_deleted"] == 0
        hits = await index.search(TENANT, "watershed testimony", kg_name=KG)
        assert len(hits.hits) == 3
        # Display attrs came through the scan (type via rdf:type).
        assert hits.hits[0].attrs.get("type") == "Doc"

    asyncio.run(run())


def test_reconcile_skips_unchanged_docs_and_preserves_embeddings():
    """Rerunning reconcile over unchanged data writes nothing and NEVER
    re-queues an already-filled embedding (content_hash is the currency)."""
    neptune = _kg(_doc_entities(2), {("Doc", "description"): "free_text"})
    index = InMemorySemanticIndex()

    async def run():
        await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        pending = await index.fetch_pending(limit=100)
        assert await index.fill_embeddings(
            pending, [[0.5, 0.5]] * len(pending), embed_model="m1"
        ) == len(pending)

        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["chunks_written"] == 0
        assert counters["skipped_unchanged_hash"] == 2
        assert counters["ghosts_deleted"] == 0
        assert await index.fetch_pending(limit=100) == []  # embeddings survived

    asyncio.run(run())


def test_reconcile_deletes_ghosts_of_merged_entities():
    """ER merges / normalization deletes bypass the write hook: an entity gone
    from Neptune must lose its index rows on reconcile, siblings untouched."""
    entities = _doc_entities(2)
    neptune = _kg(entities, {("Doc", "description"): "free_text"})
    index = InMemorySemanticIndex()

    async def run():
        await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        # e1 is merged away (its triples vanish from the instance graph).
        del neptune.entities[_entity(1)]
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["ghosts_deleted"] == 1
        rows = await index.fetch_pending(limit=100)
        assert {r.entity_uri for r in rows} == {_entity(2)}

    asyncio.run(run())


def test_reconcile_ghost_deletion_is_batched_via_delete_docs():
    """Ghost deletion goes through the protocol's batched delete_docs — ONE
    call for the whole ghost set, not one per-doc delete round trip."""
    entities = _doc_entities(3)
    neptune = _kg(entities, {("Doc", "description"): "free_text"})

    class RecordingIndex(InMemorySemanticIndex):
        def __init__(self) -> None:
            super().__init__()
            self.delete_docs_calls: list[list[tuple[str, str]]] = []
            self.delete_calls = 0

        async def delete_docs(self, pairs, tenant_id, *, kg_name):  # noqa: ANN001
            self.delete_docs_calls.append(list(pairs))
            await super().delete_docs(pairs, tenant_id, kg_name=kg_name)

        async def delete(self, *a, **kw):  # noqa: ANN002, ANN003
            self.delete_calls += 1
            await super().delete(*a, **kw)

    index = RecordingIndex()

    async def run():
        await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        del neptune.entities[_entity(1)]
        del neptune.entities[_entity(2)]
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["ghosts_deleted"] == 2
        assert index.delete_docs_calls == [
            [(_entity(1), "description"), (_entity(2), "description")]
        ]
        assert index.delete_calls == 0  # no per-doc fallback on a capable backend

    asyncio.run(run())


def test_reconcile_decided_no_flip_removes_rows_without_reclassifying():
    """Candidacy flip: a marker rewritten to a decided-no kind removes that
    attr's rows via the ghost diff — and the default heuristic must NOT fight
    the decision (the attr is in the map, so it is decided, not undecided)."""
    neptune = _kg(_doc_entities(2), {("Doc", "description"): "free_text"})
    index = InMemorySemanticIndex()

    async def run():
        await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert len(await index.fetch_pending(limit=100)) == 2

        neptune.markers[("Doc", "description")] = "not_text"  # decided-no
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["ghosts_deleted"] == 2
        assert counters["attrs_marked_free_text"] == 0  # heuristic stayed out
        assert await index.fetch_pending(limit=100) == []
        assert not any("textKind" in u for u in neptune.updates)

    asyncio.run(run())


def test_reconcile_marker_added_indexes_existing_data():
    """Candidacy flip the other way: marking an attr AFTER its data landed gets
    it indexed on the next reconcile (no re-ingest)."""
    neptune = _kg(_doc_entities(2))  # data present, nothing marked
    index = InMemorySemanticIndex()

    async def run():
        # The heuristic will mark long prose itself; pin the flip explicitly by
        # pre-writing a decided-no, reconciling, then flipping to free_text.
        neptune.markers[("Doc", "description")] = "not_text"
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["chunks_written"] == 0

        neptune.markers[("Doc", "description")] = "free_text"
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["chunks_written"] == 2
        assert {r.attr for r in await index.fetch_pending(limit=100)} == {
            "description"
        }

    asyncio.run(run())


def test_reconcile_default_candidacy_heuristic():
    """ONTA-177 hand-off: attributes with NO verdict get the name-blind
    heuristic on sampled values — long prose → durable free_text (then indexed
    this same run); codes → durable decided-no; both written via
    upsert_attribute_text_kind so every consumer sees them."""
    entities = {
        _entity(i): {
            RDF_TYPE: [DOC_TYPE],
            SUMMARY_PRED: [f"{PROSE} Extended remarks for session {i}."],
            SKU_PRED: [f"SKU-{i:04d}"],
        }
        for i in range(1, 5)
    }
    neptune = _kg(entities)  # NO markers at all
    index = InMemorySemanticIndex()

    async def run():
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["attrs_marked_free_text"] == 1
        assert counters["attrs_marked_not_text"] == 1
        # Verdicts are durable in the ontology graph...
        assert neptune.markers[("Doc", "summary")] == "free_text"
        assert neptune.markers[("Doc", "sku")] == rec.TEXT_KIND_NOT_TEXT
        # ...and the freshly-marked attr was indexed in the SAME run.
        assert counters["chunks_written"] == 4
        rows = await index.fetch_pending(limit=100)
        assert {r.attr for r in rows} == {"summary"}

    asyncio.run(run())


def test_candidacy_cap_randomizes_selection_and_logs_the_backlog(monkeypatch):
    """Fairness past the per-run cap: with a deterministic prefix, attrs
    sorted after the cap would NEVER be sampled while ≥cap stay perpetually
    ambiguous. The selection must shuffle before truncating (so every
    undecided attr eventually gets its turn) and log the total backlog."""
    monkeypatch.setattr(rec, "_MAX_CANDIDACY_ATTRS_PER_RUN", 1)
    # A deterministic "shuffle" that reverses: the pick must become the attr a
    # stable prefix would have starved (proves shuffle happens BEFORE [:cap]).
    monkeypatch.setattr(rec.random, "shuffle", lambda seq: seq.reverse())

    aaa_pred = attr_uri("Doc", "aaa_first")
    zzz_pred = attr_uri("Doc", "zzz_last")
    entities = {
        _entity(i): {
            RDF_TYPE: [DOC_TYPE],
            aaa_pred: [f"{PROSE} First remarks {i}."],
            zzz_pred: [f"{PROSE} Last remarks {i}."],
        }
        for i in range(1, 4)
    }
    neptune = _kg(entities)  # both attrs undecided
    index = InMemorySemanticIndex()

    async def run():
        with structlog.testing.capture_logs() as logs:
            counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        # Only the capped, shuffled pick was classified this run…
        assert counters["attrs_marked_free_text"] == 1
        assert ("Doc", "zzz_last") in neptune.markers  # the reversed pick
        assert ("Doc", "aaa_first") not in neptune.markers  # waits for a later run
        # …and the truncation reported the full backlog, never silently.
        [capped] = [e for e in logs if e["event"] == "semantic_candidacy_capped"]
        assert capped["undecided"] == 2
        assert capped["cap"] == 1

    asyncio.run(run())


def test_reconcile_heuristic_verdicts_stick_across_runs():
    """A decided verdict (either way) is NOT re-sampled on later runs — absence
    means undecided, presence means decided."""
    entities = {
        _entity(1): {RDF_TYPE: [DOC_TYPE], SKU_PRED: ["SKU-1", "SKU-2", "SKU-3"]}
    }
    neptune = _kg(entities)
    index = InMemorySemanticIndex()

    async def run():
        await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        first_updates = len(neptune.updates)
        assert first_updates == 1  # the sku decided-no verdict
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert len(neptune.updates) == first_updates  # nothing rewritten
        assert counters["attrs_marked_not_text"] == 0

    asyncio.run(run())


def test_reconcile_aborts_on_marker_fetch_failure_without_ghost_deleting():
    """A Neptune hiccup during the marker fetch must ABORT the run (the runner
    retries next cadence) — acting on an empty map would ghost-delete the whole
    KG's index. This is why the reconciler does NOT use the best-effort
    get_free_text_map."""
    neptune = _kg(_doc_entities(1), {("Doc", "description"): "free_text"})
    index = InMemorySemanticIndex()

    async def run():
        await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert len(await index.fetch_pending(limit=100)) == 1

        async def boom(_sparql):
            raise RuntimeError("neptune down")

        neptune.query = boom  # type: ignore[assignment]
        with pytest.raises(RuntimeError):
            await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert len(await index.fetch_pending(limit=100)) == 1  # rows intact

    asyncio.run(run())


def test_reconcile_without_list_docs_still_upserts_and_skips_ghosts():
    """A third-party backend that predates the Protocol's list_docs still
    converges on upserts (hash-idempotent); ghost deletion is skipped LOUDLY
    (``doc_listing_supported=False``), not wrongly."""
    neptune = _kg(_doc_entities(2), {("Doc", "description"): "free_text"})
    index = LegacyThirdPartyIndex()  # genuinely lacks list_docs

    async def run():
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["chunks_written"] == 2
        del neptune.entities[_entity(1)]
        with structlog.testing.capture_logs() as logs:
            counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        # Ghost repair needs the method — nothing was (wrongly) deleted, and
        # the degrade is loud, never silent.
        assert counters["ghosts_deleted"] == 0
        assert len(await index.fetch_pending(limit=100)) == 2
        [summary] = [e for e in logs if e["event"] == "semantic_reconcile"]
        assert summary["doc_listing_supported"] is False
        assert any(
            e["event"] == "semantic_reconcile_ghost_scan_skipped" for e in logs
        )

    asyncio.run(run())


def test_reconcile_reports_doc_listing_supported_on_plain_inmemory():
    """The OSS-default InMemory backend implements the Protocol's list_docs,
    so the default path must report ``doc_listing_supported=True`` — the
    graceful-degrade branch is reserved for third-party backends that lag the
    Protocol, never the first-party defaults."""
    neptune = _kg(_doc_entities(1), {("Doc", "description"): "free_text"})
    index = InMemorySemanticIndex()

    async def run():
        with structlog.testing.capture_logs() as logs:
            await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        [summary] = [e for e in logs if e["event"] == "semantic_reconcile"]
        assert summary["doc_listing_supported"] is True
        assert not any(
            e["event"] == "semantic_reconcile_ghost_scan_skipped" for e in logs
        )

    asyncio.run(run())


def test_reconcile_disabled_is_a_noop(monkeypatch):
    monkeypatch.delenv("COGRAPH_SEMANTIC_INDEX_ENABLED", raising=False)
    neptune = _kg(_doc_entities(1), {("Doc", "description"): "free_text"})
    index = InMemorySemanticIndex()

    async def run():
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["chunks_written"] == 0
        assert neptune.queries == []  # not even a marker fetch

    asyncio.run(run())


def test_partial_reconcile_resumes_and_converges(monkeypatch):
    """Interrupt a reconcile mid-upsert (deploy kill) → the rerun skips what
    landed (unchanged hashes) and finishes the rest, including ghosts."""
    # One doc per upsert batch so the crash leaves genuinely partial state.
    monkeypatch.setattr(rec, "_UPSERT_BATCH_CHUNKS", 1)
    neptune = _kg(_doc_entities(3), {("Doc", "description"): "free_text"})
    index = CrashingIndex(fail_after=1)

    async def run():
        # Seed a ghost that the interrupted run never gets to delete.
        await index.upsert_chunks([_chunk(9, "stale merged-away doc")])

        with pytest.raises(RuntimeError):
            await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        partial = {
            r.entity_uri for r in await index.fetch_pending(limit=100)
        }
        assert _entity(9) in partial  # ghost still there mid-crash
        assert len(partial) < 4  # genuinely partial

        index.fail_after = None  # "next deploy": healthy again
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["chunks_written"] + counters["skipped_unchanged_hash"] == 3
        assert counters["ghosts_deleted"] == 1
        rows = await index.fetch_pending(limit=100)
        assert {r.entity_uri for r in rows} == {_entity(1), _entity(2), _entity(3)}

    asyncio.run(run())


def test_reconcile_scan_pages_through_large_kgs(monkeypatch):
    """The scan pages with keyset pagination (ORDER BY ?e ?p ?o + a
    strictly-after-entity FILTER, never OFFSET — the FakeNeptune handler
    asserts that) — a KG larger than one page is still fully indexed."""
    monkeypatch.setenv("COGRAPH_SEMANTIC_SCAN_PAGE_SIZE", "3")
    neptune = _kg(_doc_entities(4), {("Doc", "description"): "free_text"})
    index = InMemorySemanticIndex()

    async def run():
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["chunks_written"] == 4

    asyncio.run(run())


def test_scan_keyset_keeps_straddling_entity_group_whole(monkeypatch):
    """An entity whose triples straddle a page boundary must come out of the
    scan WHOLE (the trailing partial group is held back and re-fetched from
    the last complete entity) — extract's intra-entity dedup and its
    multi-value canonicalization assume whole-entity groups."""
    from cograph_client.semantic.extract import canonicalize_values, content_hash

    monkeypatch.setenv("COGRAPH_SEMANTIC_SCAN_PAGE_SIZE", "4")
    v_a = f"{PROSE} Straddle part alpha."
    v_b = f"{PROSE} Straddle part beta."
    entities = {
        # e1: 2 scan rows (rdf:type + description).
        _entity(1): {RDF_TYPE: [DOC_TYPE], DESC_PRED: [f"{PROSE} One."]},
        # e2: 3 scan rows — page 1 (size 4) cuts its group after 2 of them.
        _entity(2): {RDF_TYPE: [DOC_TYPE], DESC_PRED: [v_a, v_b]},
    }
    neptune = _kg(entities, {("Doc", "description"): "free_text"})
    index = InMemorySemanticIndex()

    async def run():
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["chunks_written"] == 2
        # e2's doc holds BOTH values: the straddled group was re-fetched whole
        # (a torn group would have hashed/canonicalized only the first value).
        docs = {(e, a): h for e, a, h, _attrs in await index.list_docs(TENANT, kg_name=KG)}
        assert docs[(_entity(2), "description")] == content_hash(
            canonicalize_values([v_a, v_b])
        )
        # And the scan really paged with keyset: page 2 filters past e1.
        scans = [q for q in neptune.queries if "VALUES ?p" in q]
        assert len(scans) == 2
        assert "FILTER" not in scans[0]
        assert f'FILTER(STR(?e) > "{_entity(1)}")' in scans[1]

    asyncio.run(run())


def test_reconcile_mid_scan_indexed_doc_is_not_ghost_deleted():
    """The reconcile-vs-ingest race: a doc the write hook indexes AFTER the
    reconciler's list_docs snapshot but DURING the Neptune scan lands in the
    index without being in this run's scan output. It must NOT be
    ghost-deleted — the snapshot is taken before the scan starts, so a
    mid-scan arrival is simply not a ghost candidate this run."""
    neptune = _kg(_doc_entities(2), {("Doc", "description"): "free_text"})
    index = InMemorySemanticIndex()

    fired = {"done": False}
    orig_query = neptune.query

    async def query_with_mid_scan_hook_write(sparql):
        if "VALUES ?p" in sparql and not fired["done"]:
            fired["done"] = True
            # Simulate kg_writer._index_semantic firing concurrently: the doc
            # lands in the index while the scan is mid-flight.
            await index.upsert_chunks([_chunk(7, "landed mid-scan via the hook")])
        return await orig_query(sparql)

    neptune.query = query_with_mid_scan_hook_write  # type: ignore[assignment]

    async def run():
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert fired["done"]  # the race actually happened
        assert counters["ghosts_deleted"] == 0
        docs = await index.list_docs(TENANT, kg_name=KG)
        assert _entity(7) in {e for e, *_rest in docs}  # survived the run

    asyncio.run(run())


def test_reconcile_truncated_scan_skips_ghost_deletion(monkeypatch):
    """A scan cut off at the page cap yields a PARTIAL expected set — driving
    ghost deletion with it would mass-delete every healthy doc past the
    cutoff. Truncation must skip ghost deletion entirely (0 deletes, loud
    warning), while upserts of what WAS scanned still land."""
    monkeypatch.setenv("COGRAPH_SEMANTIC_SCAN_PAGE_SIZE", "1")
    monkeypatch.setattr(rec, "_MAX_SCAN_PAGES", 1)
    neptune = _kg(_doc_entities(3), {("Doc", "description"): "free_text"})
    index = InMemorySemanticIndex()

    async def run():
        # A doc that IS a genuine ghost (entity absent from Neptune) — even it
        # must survive a truncated run: partial knowledge, no deletions.
        await index.upsert_chunks([_chunk(9, "stale merged-away doc")])
        with structlog.testing.capture_logs() as logs:
            counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["ghosts_deleted"] == 0
        docs = await index.list_docs(TENANT, kg_name=KG)
        assert _entity(9) in {e for e, *_rest in docs}
        assert any(e["event"] == "semantic_scan_truncated" for e in logs)
        assert any(
            e["event"] == "semantic_reconcile_ghosts_skipped_scan_truncated"
            for e in logs
        )

    asyncio.run(run())


def test_reconcile_repairs_attrs_when_text_unchanged():
    """The enrichment-shaped attrs drift: a chunk born with attrs={} (a hook
    write whose triple batch carried no rdf:type/label rows) must get its
    denormalized attrs repaired by reconcile even though the TEXT is unchanged
    — and the already-filled embedding must survive the repair."""
    neptune = _kg(_doc_entities(1), {("Doc", "description"): "free_text"})
    index = InMemorySemanticIndex()
    text = f"{PROSE} Session 1."  # exactly what the scan re-extracts

    async def run():
        await index.upsert_chunks([_chunk(1, text)])  # attrs={} (model default)
        [pending] = await index.fetch_pending(limit=10)
        assert await index.fill_embeddings([pending], [[0.1, 0.2]], embed_model="m1") == 1

        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["attrs_repaired"] == 1
        assert counters["chunks_written"] == 1  # the repair upsert
        assert counters["skipped_unchanged_hash"] == 0
        assert counters["ghosts_deleted"] == 0
        # Repaired attrs are live (type came from the scan's rdf:type rows)...
        [(_, _, _, attrs)] = await index.list_docs(TENANT, kg_name=KG)
        assert attrs.get("type") == "Doc"
        # ...and the filled embedding was preserved, not re-queued.
        assert await index.fetch_pending(limit=10) == []

        # A second run has nothing left to repair: pure unchanged-skip.
        counters = await rec.reconcile_kg(neptune, TENANT, KG, index=index)
        assert counters["attrs_repaired"] == 0
        assert counters["chunks_written"] == 0
        assert counters["skipped_unchanged_hash"] == 1

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Embed-fill sweep
# ---------------------------------------------------------------------------


def _seed_pending(index, n: int, *, poison_first: bool = False) -> list[str]:
    texts = []
    for i in range(1, n + 1):
        text = f"POISON {i}" if (poison_first and i == 1) else f"pending text {i}"
        texts.append(text)
    return texts


def test_embed_fill_drains_pending(monkeypatch):
    index = InMemorySemanticIndex()

    async def fake_embed(texts, *, api_key, timeout=30):  # noqa: ANN001
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(embed_client_mod, "embed_texts", fake_embed)

    async def run():
        await index.upsert_chunks([_chunk(i, f"pending {i}") for i in range(1, 4)])
        counters = await rec.run_embed_fill_sweep(index=index, api_key="k")
        assert counters == {
            "embeds_pending": 3,
            "embeds_filled": 3,
            "embed_failures": 0,
        }
        assert await index.fetch_pending(limit=100) == []
        # embed_model was stamped from the shared client's constant.
        hits = await index.search(
            TENANT, "pending", query_embedding=[0.1, 0.2], kg_name=KG
        )
        assert hits.degraded is False and hits.hits

    asyncio.run(run())


def test_embed_fill_deploy_kill_mid_fill_resumes(monkeypatch):
    """Crash after a partial fill → nothing is lost: filled rows stay filled,
    the rest are still NULL and drain on the NEXT sweep (the NULL-embedding
    column is the durable queue — no outbox to replay)."""
    index = InMemorySemanticIndex()
    calls = {"n": 0}

    async def flaky_embed(texts, *, api_key, timeout=30):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("killed mid-deploy")
        return [[0.3, 0.4] for _ in texts]

    monkeypatch.setattr(embed_client_mod, "embed_texts", flaky_embed)

    async def run():
        await index.upsert_chunks([_chunk(i, f"pending {i}") for i in range(1, 5)])
        first = await rec.run_embed_fill_sweep(index=index, api_key="k", limit=2)
        assert first["embeds_filled"] == 2
        assert first["embed_failures"] == 2
        assert len(await index.fetch_pending(limit=100)) == 2

        async def healthy_embed(texts, *, api_key, timeout=30):  # noqa: ANN001
            return [[0.3, 0.4] for _ in texts]

        monkeypatch.setattr(embed_client_mod, "embed_texts", healthy_embed)
        second = await rec.run_embed_fill_sweep(index=index, api_key="k", limit=2)
        assert second["embeds_filled"] == 2
        assert await index.fetch_pending(limit=100) == []

    asyncio.run(run())


def test_embed_fill_poison_row_dead_letters(monkeypatch):
    """A poison chunk fails its batch without wedging the sweep: healthy rows
    behind it still fill in the SAME sweep (in-sweep seen-set), its
    attempt_count climbs, and past the cutoff fetch_pending(max_attempts=…)
    dead-letters it — visible with a higher cutoff, never silently gone."""
    index = InMemorySemanticIndex()

    async def poison_embed(texts, *, api_key, timeout=30):  # noqa: ANN001
        if any(t.startswith("POISON") for t in texts):
            raise RuntimeError("model refuses this input")
        return [[0.5, 0.6] for _ in texts]

    monkeypatch.setattr(embed_client_mod, "embed_texts", poison_embed)

    async def run():
        # PK order puts a1 (poison) first — the wedge-prone position.
        await index.upsert_chunks(
            [
                _chunk(1, "POISON payload", attr="a1"),
                _chunk(1, "healthy text one", attr="b1"),
                _chunk(1, "healthy text two", attr="b2"),
            ]
        )
        first = await rec.run_embed_fill_sweep(
            index=index, api_key="k", limit=1, max_attempts=2
        )
        assert first["embeds_filled"] == 2  # sweep continued past the poison
        assert first["embed_failures"] == 1

        second = await rec.run_embed_fill_sweep(
            index=index, api_key="k", limit=1, max_attempts=2
        )
        assert second["embed_failures"] == 1  # attempt 2 (the cutoff)

        third = await rec.run_embed_fill_sweep(
            index=index, api_key="k", limit=1, max_attempts=2
        )
        assert third == {
            "embeds_pending": 0,
            "embeds_filled": 0,
            "embed_failures": 0,
        }  # dead-lettered: not retried, sweep unwedged

        # Never silently vanished: still inspectable past the cutoff.
        leftovers = await index.fetch_pending(limit=100, max_attempts=None)
        assert len(leftovers) == 1
        assert leftovers[0].attempt_count == 2
        assert "refuses" in (leftovers[0].last_error or "")

    asyncio.run(run())


def test_embed_fill_all_success_sweep_keeps_fetch_window_constant(monkeypatch):
    """A healthy sweep must NOT widen the fetch window as it progresses:
    successful fills DRAIN from the queue, so only FAILED rows (which stay
    pending) need sliding past. Growing the window with every processed row
    made a large healthy sweep quadratic in fetch work."""
    fetch_limits: list[int] = []

    class RecordingIndex(InMemorySemanticIndex):
        async def fetch_pending(self, *, limit=100, **kw):  # noqa: ANN001, ANN003
            fetch_limits.append(limit)
            return await super().fetch_pending(limit=limit, **kw)

    index = RecordingIndex()

    async def fake_embed(texts, *, api_key, timeout=30):  # noqa: ANN001
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(embed_client_mod, "embed_texts", fake_embed)

    async def run():
        await index.upsert_chunks([_chunk(i, f"pending {i}") for i in range(1, 6)])
        counters = await rec.run_embed_fill_sweep(index=index, api_key="k", limit=2)
        assert counters["embeds_filled"] == 5
        assert counters["embed_failures"] == 0
        # Every fetch used the caller's limit — the window never grew.
        assert fetch_limits and all(limit == 2 for limit in fetch_limits)

    asyncio.run(run())


def test_embed_fill_without_api_key_leaves_queue_intact(monkeypatch):
    index = InMemorySemanticIndex()

    async def run():
        await index.upsert_chunks([_chunk(1, "pending")])
        counters = await rec.run_embed_fill_sweep(index=index, api_key="")
        assert counters["embeds_filled"] == 0
        assert counters["embeds_pending"] == 1
        assert len(await index.fetch_pending(limit=100)) == 1

    asyncio.run(run())


def test_embed_fill_disabled_is_a_noop(monkeypatch):
    monkeypatch.delenv("COGRAPH_SEMANTIC_INDEX_ENABLED", raising=False)
    index = InMemorySemanticIndex()

    async def run():
        await index.upsert_chunks([_chunk(1, "pending")])
        counters = await rec.run_embed_fill_sweep(index=index, api_key="k")
        assert counters["embeds_pending"] == 0
        assert len(await index.fetch_pending(limit=100)) == 1

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Schedule rows + claim exclusivity through the existing runner
# ---------------------------------------------------------------------------


def test_ensure_embed_fill_schedule_is_idempotent():
    store = InMemoryScheduleStore()

    async def run():
        first = await rec.ensure_embed_fill_schedule(store)
        assert first.action == "semantic-embed-fill"
        assert first.interval_seconds == rec.embed_fill_interval_s()
        # A restart must not reset a live row's next_run.
        advanced = first.model_copy(deep=True)
        advanced.next_run = datetime.now(timezone.utc) + timedelta(minutes=4)
        await store.update(advanced)
        again = await rec.ensure_embed_fill_schedule(store)
        assert again.next_run == advanced.next_run

    asyncio.run(run())


def test_ensure_reconcile_schedule_due_now_pulls_forward():
    store = InMemoryScheduleStore()

    async def run():
        first = await rec.ensure_reconcile_schedule(store, TENANT, KG)
        assert first.action == "semantic-reconcile"
        # Push next_run an hour out, then an on-demand reindex pulls it to now.
        later = first.model_copy(deep=True)
        later.next_run = datetime.now(timezone.utc) + timedelta(hours=1)
        await store.update(later)
        pulled = await rec.ensure_reconcile_schedule(
            store, TENANT, KG, due_now=True
        )
        assert pulled.next_run <= datetime.now(timezone.utc)
        # Removal drops the row (the KG-delete path).
        await rec.remove_reconcile_schedule(store, TENANT, KG)
        assert await store.get(rec.reconcile_schedule_id(TENANT, KG)) is None

    asyncio.run(run())


def test_hook_ensure_memo_expires_so_crud_deletes_cannot_poison_it(monkeypatch):
    """The write hook's ensure memo is TTL-based, not process-lifetime: a
    schedules-CRUD DELETE of the auto-created reconcile row (which never
    touches this module) is re-ensured within one TTL window by the next hook
    write — deleting the row is NOT a durable opt-out (the env gate is)."""
    from cograph_client.scheduling.store import make_schedule_store

    async def run():
        store = make_schedule_store()  # the same in-memory singleton the hook uses
        sid = rec.reconcile_schedule_id(TENANT, KG)

        await rec.ensure_reconcile_schedule_from_hook(TENANT, KG)
        assert await store.get(sid) is not None

        # A schedules-CRUD delete bypasses remove_reconcile_schedule entirely.
        await store.delete(sid)

        # Within the TTL the memo (correctly) suppresses the round trip…
        await rec.ensure_reconcile_schedule_from_hook(TENANT, KG)
        assert await store.get(sid) is None

        # …but once the TTL passes, the next hook write resurrects the row.
        expired = rec._now_monotonic() + rec._ensure_memo_ttl_s() + 1
        monkeypatch.setattr(rec, "_now_monotonic", lambda: expired)
        await rec.ensure_reconcile_schedule_from_hook(TENANT, KG)
        assert await store.get(sid) is not None

    asyncio.run(run())


def _semantic_schedule(action: str, *, next_run=None) -> Schedule:
    now = datetime.now(timezone.utc)
    return Schedule(
        id=f"{action}:{TENANT}:{KG}",
        tenant_id=TENANT,
        kg_name=KG,
        category="reconciliation",
        action=action,
        interval_seconds=3600,
        enabled=True,
        next_run=next_run or (now - timedelta(minutes=1)),
        created_at=now,
    )


def test_two_runner_instances_fire_a_semantic_row_exactly_once(monkeypatch):
    """Claim exclusivity: semantic rows ride the SAME claim-then-advance path
    as every other schedule, so two runner instances over one store (the
    rolling-deploy overlap) dispatch a due row exactly once."""
    from cograph_client.enrichment.job_store import InMemoryJobStore
    from cograph_client.scheduling.runner import ScheduleRunner

    fired: list[tuple[str, str]] = []

    async def fake_reconcile(client, tenant_id, kg_name, **kw):  # noqa: ANN001
        fired.append((tenant_id, kg_name))
        return {}

    monkeypatch.setattr(rec, "reconcile_kg", fake_reconcile)

    store = InMemoryScheduleStore()

    def _runner():
        return ScheduleRunner(
            store=store,
            neptune_client=object(),
            job_store=InMemoryJobStore(),
            executor=object(),
            poll_seconds=0.01,
        )

    async def run():
        await store.create(_semantic_schedule("semantic-reconcile"))
        assert await _runner().tick() == 1  # instance A claims + fires
        assert await _runner().tick() == 0  # instance B finds nothing due

    asyncio.run(run())
    assert fired == [(TENANT, KG)]


def test_semantic_row_flows_through_for_update_skip_locked(monkeypatch):
    """On Postgres the semantic row is claimed by the runner's existing
    FOR UPDATE SKIP LOCKED transaction — asserted as SQL text shape against a
    fake pool, mirroring test_schedule_runner's approach (no live DB)."""
    from cograph_client.enrichment.job_store import InMemoryJobStore
    from cograph_client.scheduling.runner import ScheduleRunner
    from cograph_client.scheduling.store import PostgresScheduleStore

    fired: list[str] = []

    async def fake_sweep(**kw):  # noqa: ANN001
        fired.append("sweep")
        return {}

    monkeypatch.setattr(rec, "run_embed_fill_sweep", fake_sweep)

    rec_calls: list[tuple] = []

    class _Tx:
        def __init__(self, conn):
            self._conn = conn

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Conn:
        def __init__(self, rows):
            self._rows = rows

        def transaction(self):
            return _Tx(self)

        async def fetch(self, sql, *params):
            rec_calls.append(("fetch", sql))
            return self._rows

        async def execute(self, sql, *params):
            rec_calls.append(("execute", sql))
            return "OK"

    class _AcquireCtx:
        def __init__(self, conn):
            self._conn = conn

        async def __aenter__(self):
            return self._conn

        async def __aexit__(self, *exc):
            return False

    class _Pool:
        def __init__(self, conn):
            self._conn = conn

        def acquire(self):
            return _AcquireCtx(self._conn)

    due = _semantic_schedule("semantic-embed-fill")
    store = PostgresScheduleStore(dsn="postgresql://fake/db")
    store._pool = _Pool(_Conn([{"id": due.id, "payload": due.model_dump_json()}]))

    runner = ScheduleRunner(
        store=store,
        neptune_client=object(),
        job_store=InMemoryJobStore(),
        executor=object(),
        poll_seconds=0.01,
    )

    async def run():
        assert await runner.tick() == 1

    asyncio.run(run())
    assert fired == ["sweep"]
    select_sql = next(sql for kind, sql in rec_calls if kind == "fetch")
    assert "FOR UPDATE SKIP LOCKED" in select_sql


def test_dispatch_skips_semantic_rows_when_disabled(monkeypatch):
    """Stale schedule rows left behind after a disable are cheap no-ops."""
    monkeypatch.delenv("COGRAPH_SEMANTIC_INDEX_ENABLED", raising=False)
    fired: list[str] = []

    async def fake_sweep(**kw):  # noqa: ANN001
        fired.append("sweep")
        return {}

    monkeypatch.setattr(rec, "run_embed_fill_sweep", fake_sweep)

    async def run():
        await rec.dispatch_semantic_schedule(
            _semantic_schedule("semantic-embed-fill"), client=object()
        )

    asyncio.run(run())
    assert fired == []
