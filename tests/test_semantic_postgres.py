"""pgvector semantic-index backend (ONTA-176).

Two layers, mirroring ``test_spatiotemporal.py``:

* **Fake-pool unit tests** (no real Postgres): emitted DDL/SQL shape — the
  5-col PK, GENERATED tsvector, GIN + HNSW indexes, the replace-per-doc upsert
  (``ON CONFLICT ... WHERE content_hash IS DISTINCT FROM``), the hybrid RRF
  statement's leg pre-filters, the ANN mode selection (exact / hnsw_default /
  hnsw_iterative) incl. the pgvector-0.8 capability probe and its
  placeholder-GUC false-positive guard, the degraded lexical-only path, the
  queue SQL, and the registry seam.
* **Live integration tests** gated on ``OMNIX_DATABASE_URL`` (skipped without
  it): the full protocol contract against a real Postgres + pgvector. These
  run in CI against the ``pgvector/pgvector:pg16`` service container, and
  locally against a scratch cluster. They are written to pass on BOTH pgvector
  0.6 (no iterative_scan → the probe must fail cleanly and hnsw_default /
  exact fallbacks run) and 0.8+ (iterative_scan detected).
"""

from __future__ import annotations

import os
import uuid

import pytest

from cograph_client.semantic import (
    InMemorySemanticIndex,
    SemanticChunk,
    SemanticIndex,
    content_hash,
    make_semantic_index,
    reset_semantic_index,
)
from cograph_client.semantic.postgres import (
    HNSW_EF_SEARCH,
    PostgresSemanticIndex,
    _vector_text,
    _version_at_least,
)

DSN = os.environ.get("OMNIX_DATABASE_URL", "")

TENANT = "demo-tenant"
KG = "kg1"
OTHER_KG = "kg2"
FAKE_MODEL = "fake-embed-model"


def _chunk(
    uri: str,
    text: str,
    *,
    ix: int = 0,
    attr: str = "description",
    tenant: str = TENANT,
    kg: str = KG,
    doc_text: str | None = None,
    embedding: list[float] | None = None,
    attrs: dict | None = None,
) -> SemanticChunk:
    return SemanticChunk(
        tenant_id=tenant,
        kg_name=kg,
        entity_uri=uri,
        attr=attr,
        chunk_ix=ix,
        chunk_text=text,
        content_hash=content_hash(doc_text if doc_text is not None else text),
        embedding=embedding,
        embed_model=FAKE_MODEL if embedding is not None else None,
        attrs=attrs if attrs is not None else {"label": uri},
    )


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_semantic_index()
    yield
    reset_semantic_index()


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def test_version_at_least():
    assert _version_at_least("0.8.0", (0, 8))
    assert _version_at_least("0.8.1", (0, 8))
    assert _version_at_least("1.0.0", (0, 8))
    assert not _version_at_least("0.6.0", (0, 8))
    assert not _version_at_least("0.7.4", (0, 8))
    # Unparseable / missing → False (fall back to the SET probe alone).
    assert not _version_at_least(None, (0, 8))
    assert not _version_at_least("", (0, 8))
    assert not _version_at_least("weird", (0, 8))
    # Debian-style suffixes still parse their numeric prefix.
    assert _version_at_least("0.8.0-1.pgdg", (0, 8))


def test_vector_text():
    assert _vector_text(None) is None
    assert _vector_text([1, 2.5, -3]) == "[1.0,2.5,-3.0]"


def test_invalid_ts_config_rejected():
    with pytest.raises(ValueError):
        PostgresSemanticIndex(dsn="postgres://x", ts_config="bad config; DROP")


def test_invalid_embed_dim_rejected():
    with pytest.raises(ValueError):
        PostgresSemanticIndex(dsn="postgres://x", embed_dim=0)
    with pytest.raises(ValueError):
        PostgresSemanticIndex(dsn="postgres://x", embed_dim=99_999)


def test_env_knobs_are_read(monkeypatch):
    monkeypatch.setenv("OMNIX_SEMANTIC_TS_CONFIG", "english")
    monkeypatch.setenv("OMNIX_SEMANTIC_EMBED_DIM", "8")
    monkeypatch.setenv("OMNIX_SEMANTIC_EXACT_SCAN_THRESHOLD", "123")
    idx = PostgresSemanticIndex(dsn="postgres://x")
    assert idx._ts_config == "english"
    assert idx._embed_dim == 8
    assert idx._exact_scan_threshold == 123


# ---------------------------------------------------------------------------
# Factory seam (registry)
# ---------------------------------------------------------------------------


def test_factory_returns_postgres_when_dsn_set(monkeypatch):
    from cograph_client import config

    monkeypatch.setattr(config.settings, "database_url", "postgres://u@h/db", raising=False)
    idx = make_semantic_index()
    assert isinstance(idx, PostgresSemanticIndex)
    assert isinstance(idx, SemanticIndex)


def test_factory_returns_inmemory_without_dsn(monkeypatch):
    from cograph_client import config

    monkeypatch.setattr(config.settings, "database_url", "", raising=False)
    assert isinstance(make_semantic_index(), InMemorySemanticIndex)


def test_lazy_package_export():
    from cograph_client import semantic

    assert semantic.PostgresSemanticIndex is PostgresSemanticIndex
    with pytest.raises(AttributeError):
        semantic.NoSuchThing  # noqa: B018


# ---------------------------------------------------------------------------
# Fake asyncpg pool (no real DB) — emitted SQL shape
# ---------------------------------------------------------------------------


class FakeConn:
    """Records every call; canned results for fetch/fetchval; simulates the
    pgvector version via ``extversion`` + ``iterative_scan_ok``."""

    def __init__(self, recorder):
        self._rec = recorder
        self.rows: list[dict] = []
        self.extversion = "0.6.0"
        self.iterative_scan_ok = False  # SET LOCAL hnsw.iterative_scan raises
        self.gate_count = 0
        self.update_statuses: list[str] = []

    async def execute(self, sql, *args):
        self._rec.append(("execute", sql, args))
        if "hnsw.iterative_scan" in sql and not self.iterative_scan_ok:
            raise RuntimeError(
                'unrecognized configuration parameter "hnsw.iterative_scan"'
            )
        if sql.strip().upper().startswith("UPDATE"):
            return self.update_statuses.pop(0) if self.update_statuses else "UPDATE 1"
        return "OK"

    async def executemany(self, sql, rows):
        self._rec.append(("executemany", sql, [tuple(r) for r in rows]))

    async def fetch(self, sql, *args):
        self._rec.append(("fetch", sql, args))
        return self.rows

    async def fetchval(self, sql, *args):
        self._rec.append(("fetchval", sql, args))
        if "extversion" in sql:
            return self.extversion
        if "count(*)" in sql:
            return self.gate_count
        return None

    def transaction(self):
        conn = self

        class _Tx:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *exc):
                return False

        return _Tx()


class FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn
        self.expired = 0

    def acquire(self):
        return FakeAcquire(self._conn)

    def expire_connections(self):
        self.expired += 1


@pytest.fixture
def pg(monkeypatch):
    """A PostgresSemanticIndex wired to a fake asyncpg pool.

    Yields (store, recorder, conn, pool). Uses the shared-pool reset pattern
    from ONTA-174 so each test gets THIS test's fake.
    """
    recorder: list = []
    conn = FakeConn(recorder)
    pool = FakePool(conn)

    async def fake_create_pool(*a, **k):
        recorder.append(("create_pool", a, k))
        return pool

    import asyncpg

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    from cograph_client.db.pool import reset_pg_pools

    reset_pg_pools()
    store = PostgresSemanticIndex(
        dsn="postgres://user@host/db",
        embed_model=FAKE_MODEL,
        embed_dim=4,
        ts_config="simple",
        exact_scan_threshold=100,
    )
    yield store, recorder, conn, pool
    reset_pg_pools()


def _sqls(recorder, op=None) -> list[str]:
    return [entry[1] for entry in recorder if entry[0] != "create_pool" and (op is None or entry[0] == op)]


async def test_ddl_shape(pg):
    store, recorder, _conn, pool = pg
    await store._ensure_pool()
    ddl = " ".join(_sqls(recorder, "execute"))
    assert "CREATE EXTENSION IF NOT EXISTS vector" in ddl
    assert "CREATE TABLE IF NOT EXISTS entity_semantic_chunk" in ddl
    # Column shape (SemanticChunk 1:1 + GENERATED tsv).
    assert "embedding     vector(4)" in ddl
    assert "tsvector GENERATED ALWAYS AS" in ddl
    assert "to_tsvector('simple'::regconfig, chunk_text)) STORED" in ddl
    assert "attempt_count integer NOT NULL DEFAULT 0" in ddl
    assert "attrs         jsonb NOT NULL DEFAULT '{}'::jsonb" in ddl
    assert "PRIMARY KEY (tenant_id, kg_name, entity_uri, attr, chunk_ix)" in ddl
    # Both leg indexes.
    assert "USING GIN (tsv)" in ddl
    assert "USING hnsw (embedding vector_cosine_ops)" in ddl
    # First-time DDL recycles connections so the codec hook re-runs with the
    # extension present.
    assert pool.expired >= 1


async def test_extension_failure_tolerated(monkeypatch):
    """A permission error on CREATE EXTENSION must not abort setup (the infra
    layer bootstraps extensions in managed environments)."""
    recorder: list = []

    class PickyConn(FakeConn):
        async def execute(self, sql, *args):
            if "CREATE EXTENSION" in sql:
                recorder.append(("execute", sql, args))
                raise RuntimeError("permission denied to create extension")
            return await super().execute(sql, *args)

    conn = PickyConn(recorder)
    pool = FakePool(conn)

    async def fake_create_pool(*a, **k):
        return pool

    import asyncpg

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    from cograph_client.db.pool import reset_pg_pools

    reset_pg_pools()
    store = PostgresSemanticIndex(dsn="postgres://u@h/db", embed_dim=4)
    await store._ensure_pool()  # must NOT raise
    ddl = " ".join(s for (_o, s, *_r) in recorder)
    assert "CREATE EXTENSION" in ddl
    reset_pg_pools()


async def test_codec_hook_registered_and_tolerant(pg, monkeypatch):
    store, recorder, _conn, _pool = pg
    await store._ensure_pool()
    from cograph_client.db import pool as pool_mod
    from cograph_client.semantic.postgres import _register_vector_codec

    assert _register_vector_codec in pool_mod._init_hooks

    # The hook itself must tolerate a missing vector type (fresh DB before
    # DDL) — register_vector raising is a warning, not a failure.
    import pgvector.asyncpg as pga

    async def boom(conn):
        raise RuntimeError("type vector does not exist")

    monkeypatch.setattr(pga, "register_vector", boom)
    await _register_vector_codec(object())  # must NOT raise


# -- capability probe ---------------------------------------------------------


async def test_probe_pgvector_06_falls_back(pg):
    """pgvector 0.6: the SET raises (reserved GUC prefix) → capability False.
    This is the deployed-Aurora / local-test situation."""
    store, recorder, conn, _pool = pg
    conn.extversion = "0.6.0"
    conn.iterative_scan_ok = False
    await store._ensure_pool()
    assert store._iterative_scan is False
    assert any("SET LOCAL hnsw.iterative_scan" in s for s in _sqls(recorder, "execute"))


async def test_probe_pgvector_08_detected(pg):
    store, recorder, conn, _pool = pg
    conn.extversion = "0.8.0"
    conn.iterative_scan_ok = True
    await store._ensure_pool()
    assert store._iterative_scan is True


async def test_probe_placeholder_guc_false_positive_guard(pg):
    """SET 'succeeding' on a pre-0.8 pgvector (unreserved prefix → placeholder
    GUC) must NOT count as support — the extversion cross-check kills it."""
    store, recorder, conn, _pool = pg
    conn.extversion = "0.6.0"
    conn.iterative_scan_ok = True  # SET silently creates a placeholder
    await store._ensure_pool()
    assert store._iterative_scan is False


# -- upsert (replace-per-doc) -------------------------------------------------


async def test_upsert_sql_shape(pg):
    store, recorder, _conn, _pool = pg
    await store.upsert_chunks(
        [
            _chunk("e:1", "part one", ix=0, doc_text="d1", embedding=[1, 0, 0, 0]),
            _chunk("e:1", "part two", ix=1, doc_text="d1"),
        ]
    )
    batches = [e for e in recorder if e[0] == "executemany"]
    assert len(batches) == 2
    insert_sql, insert_rows = batches[0][1], batches[0][2]
    assert "INSERT INTO entity_semantic_chunk" in insert_sql
    assert (
        "ON CONFLICT (tenant_id, kg_name, entity_uri, attr, chunk_ix) DO UPDATE"
        in insert_sql
    )
    # The unchanged-hash guard: keep the row (and its filled embedding) as-is.
    assert (
        "WHERE entity_semantic_chunk.content_hash IS DISTINCT FROM EXCLUDED.content_hash"
        in insert_sql
    )
    # Vector bound as text + cast (works with or without the asyncpg codec).
    assert "$8::text::vector" in insert_sql
    assert insert_rows[0][7] == "[1.0,0.0,0.0,0.0]"
    assert insert_rows[1][7] is None

    tail_sql, tail_rows = batches[1][1], batches[1][2]
    assert "chunk_ix >= $5" in tail_sql
    # doc_len = max chunk_ix + 1 = 2 for the (e:1, description) doc.
    assert tail_rows == [(TENANT, KG, "e:1", "description", 2)]


async def test_upsert_tail_delete_per_doc(pg):
    store, recorder, _conn, _pool = pg
    await store.upsert_chunks(
        [
            _chunk("e:1", "a", ix=0),
            _chunk("e:1", "n", ix=0, attr="notes"),
            _chunk("e:2", "b", ix=0),
        ]
    )
    tail_rows = [e for e in recorder if e[0] == "executemany"][1][2]
    assert set(tail_rows) == {
        (TENANT, KG, "e:1", "description", 1),
        (TENANT, KG, "e:1", "notes", 1),
        (TENANT, KG, "e:2", "description", 1),
    }


async def test_upsert_empty_is_noop(pg):
    store, recorder, _conn, _pool = pg
    await store.upsert_chunks([])
    assert recorder == []  # not even pool/DDL setup


# -- search: mode selection + SQL shape ----------------------------------------


def _fetch_sql(recorder) -> str:
    return next(s for (op, s, *_a) in recorder if op == "fetch")


def _fetch_args(recorder) -> tuple:
    return next(a for (op, _s, a) in recorder if op == "fetch")


async def test_search_degraded_lexical_only(pg):
    store, recorder, conn, _pool = pg
    conn.rows = [
        {
            "entity_uri": "e:1",
            "attr": "description",
            "chunk_text": "solar text",
            "attrs": '{"label": "x"}',
            "score": 0.016,
        }
    ]
    res = await store.search(TENANT, "solar panels")
    assert res.degraded is True
    assert [h.entity_uri for h in res.hits] == ["e:1"]
    assert res.hits[0].attrs == {"label": "x"}
    assert store._last_search_mode == "lexical_only"
    sql = _fetch_sql(recorder)
    # FTS leg only — no ANN CTE, no gate count query.
    assert "websearch_to_tsquery" in sql
    assert "<=>" not in sql
    assert not any("count(*)" in s for (op, s, *_r) in recorder if op == "fetchval" and "extversion" not in s)
    # Leg pre-filters: tenant mandatory, kg/type optional, inside the leg.
    assert "tenant_id = $1" in sql
    assert "($2::text IS NULL OR kg_name = $2::text)" in sql
    assert "($3::text IS NULL OR attrs->>'type' = $3::text)" in sql
    # RRF + grouping shape.
    assert "1.0 / (60 + rank)" in sql
    assert "LIMIT 50" in sql
    assert "DISTINCT ON (entity_uri)" in sql
    args = _fetch_args(recorder)
    assert args == (TENANT, None, None, "simple", "solar panels", 10)


async def test_search_small_set_uses_exact_mode(pg):
    store, recorder, conn, _pool = pg
    conn.gate_count = 5  # <= threshold (100) → exact
    res = await store.search(TENANT, "solar", query_embedding=[1, 0, 0, 0])
    assert res.degraded is False
    assert store._last_search_mode == "ann_exact"
    sql = _fetch_sql(recorder)
    # Exact by construction: the filtered pool is a MATERIALIZED fence.
    assert "ann_pool AS MATERIALIZED" in sql
    assert "embedding <=> $7::text::vector" in sql
    assert "embedding IS NOT NULL" in sql
    assert "embed_model = $6" in sql
    assert "UNION ALL" in sql
    # No planner SETs needed in exact mode.
    assert not any("SET LOCAL hnsw.ef_search" in s for s in _sqls(recorder, "execute"))
    args = _fetch_args(recorder)
    assert args == (
        TENANT,
        None,
        None,
        "simple",
        "solar",
        FAKE_MODEL,
        "[1.0,0.0,0.0,0.0]",
        10,
    )


async def test_search_large_set_hnsw_default_below_08(pg):
    store, recorder, conn, _pool = pg
    conn.extversion = "0.6.0"
    conn.gate_count = 101  # > threshold → index mode
    await store.search(TENANT, "solar", query_embedding=[1, 0, 0, 0])
    assert store._last_search_mode == "hnsw_default"
    sql = _fetch_sql(recorder)
    assert "MATERIALIZED" not in sql
    assert "embedding <=> $7::text::vector" in sql
    sets = [s for s in _sqls(recorder, "execute") if "SET LOCAL" in s]
    # ef_search raised above the 50-row leg budget; NO iterative_scan SET in
    # the query path on a 0.6 pool (only the one-time probe attempts it).
    assert any(f"hnsw.ef_search = {HNSW_EF_SEARCH}" in s for s in sets)
    assert sum("hnsw.iterative_scan" in s for s in sets) == 1  # the probe only


async def test_search_large_set_iterative_on_08(pg):
    store, recorder, conn, _pool = pg
    conn.extversion = "0.8.0"
    conn.iterative_scan_ok = True
    conn.gate_count = 101
    await store.search(TENANT, "solar", query_embedding=[1, 0, 0, 0])
    assert store._last_search_mode == "hnsw_iterative"
    sets = [s for s in _sqls(recorder, "execute") if "SET LOCAL" in s]
    assert any("hnsw.ef_search" in s for s in sets)
    # probe + query path.
    assert sum("hnsw.iterative_scan = 'relaxed_order'" in s for s in sets) == 2


async def test_search_gate_is_bounded_and_prefiltered(pg):
    store, recorder, conn, _pool = pg
    conn.gate_count = 0
    await store.search(
        TENANT, "solar", query_embedding=[1, 0, 0, 0], kg_name=KG, type_filter="Event"
    )
    gate = next(
        (s, a)
        for (op, s, a) in recorder
        if op == "fetchval" and "count(*)" in s
    )
    sql, args = gate
    assert "LIMIT $5" in sql
    assert "embedding IS NOT NULL" in sql
    assert "embed_model = $4" in sql
    # threshold 100 → probe bounded at 101 rows.
    assert args == (TENANT, KG, "Event", FAKE_MODEL, 101)


async def test_search_dim_mismatch_degrades_lexical(pg):
    store, recorder, conn, _pool = pg
    res = await store.search(TENANT, "solar", query_embedding=[1.0, 0.0])  # dim 2 != 4
    assert res.degraded is True
    assert store._last_search_mode == "lexical_only"
    assert "<=>" not in _fetch_sql(recorder)


async def test_search_top_k_zero_short_circuits(pg):
    store, recorder, conn, _pool = pg
    res = await store.search(TENANT, "solar", top_k=0)
    assert res.hits == [] and res.degraded is True
    assert not any(op == "fetch" for (op, *_r) in recorder)


# -- deletes / clear ------------------------------------------------------------


async def test_delete_and_clear_sql(pg):
    store, recorder, _conn, _pool = pg
    await store.delete("e:1", TENANT)
    await store.delete("e:1", TENANT, kg_name=KG, attr="notes")
    await store.clear(TENANT)
    await store.clear(TENANT, kg_name=KG)
    deletes = [s for s in _sqls(recorder, "execute") if s.startswith("DELETE")]
    assert deletes[0] == (
        "DELETE FROM entity_semantic_chunk WHERE tenant_id = $1 AND entity_uri = $2"
    )
    assert deletes[1].endswith("AND kg_name = $3 AND attr = $4")
    assert deletes[2] == "DELETE FROM entity_semantic_chunk WHERE tenant_id = $1"
    assert deletes[3].endswith("AND kg_name = $2")


# -- embed queue ---------------------------------------------------------------


async def test_fetch_pending_sql_and_hydration(pg):
    store, recorder, conn, _pool = pg
    conn.rows = [
        {
            "tenant_id": TENANT,
            "kg_name": KG,
            "entity_uri": "e:1",
            "attr": "description",
            "chunk_ix": 0,
            "chunk_text": "text",
            "content_hash": "h",
            "embed_model": None,
            "attempt_count": 2,
            "last_error": "429",
            "attrs": b'{"label": "L"}',
        }
    ]
    rows = await store.fetch_pending(limit=7, max_attempts=3, tenant_id=TENANT, kg_name=KG)
    sql = _fetch_sql(recorder)
    assert "WHERE embedding IS NULL" in sql
    assert "attempt_count < $1::int" in sql
    assert "ORDER BY tenant_id, kg_name, entity_uri, attr, chunk_ix" in sql
    assert _fetch_args(recorder) == (3, TENANT, KG, 7)
    [c] = rows
    assert c.embedding is None
    assert c.attempt_count == 2 and c.last_error == "429"
    assert c.attrs == {"label": "L"}


async def test_fill_embeddings_counts_and_guards(pg):
    store, recorder, conn, _pool = pg
    conn.update_statuses = ["UPDATE 1", "UPDATE 0"]
    chunks = [_chunk("e:1", "a"), _chunk("e:2", "b")]
    n = await store.fill_embeddings(
        chunks, [[1, 0, 0, 0], [0, 1, 0, 0]], embed_model="m1"
    )
    assert n == 1  # only rows actually updated count
    updates = [s for s in _sqls(recorder, "execute") if s.strip().startswith("UPDATE")]
    assert len(updates) == 2
    # Optimistic-concurrency guard: same hash AND still unembedded.
    assert "content_hash = $8 AND embedding IS NULL" in updates[0]
    assert "embed_model = $7, last_error = NULL" in updates[0]


async def test_fill_embeddings_length_mismatch_raises(pg):
    store, _recorder, _conn, _pool = pg
    with pytest.raises(ValueError):
        await store.fill_embeddings([_chunk("e:1", "a")], [], embed_model="m")


async def test_mark_embed_failed_sql(pg):
    store, recorder, conn, _pool = pg
    conn.update_statuses = ["UPDATE 1"]
    n = await store.mark_embed_failed([_chunk("e:1", "a")], error="boom")
    assert n == 1
    [sql] = [s for s in _sqls(recorder, "execute") if s.strip().startswith("UPDATE")]
    assert "attempt_count = attempt_count + 1" in sql
    assert "content_hash = $7 AND embedding IS NULL" in sql


# ---------------------------------------------------------------------------
# Live DB integration tests (skip without OMNIX_DATABASE_URL)
# ---------------------------------------------------------------------------

needs_pg = pytest.mark.skipif(
    not DSN, reason="OMNIX_DATABASE_URL not set; needs live Postgres with pgvector"
)

#: All live tests use dim-8 vectors — the table is created once per database
#: with this dimension (the knob is DDL-time-only; see postgres.py docstring).
DIM = 8

V1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def _t() -> str:
    """A unique tenant per test so live runs never collide (CI reruns, the
    shared scratch cluster, the parity suite)."""
    return f"t-{uuid.uuid4().hex[:10]}"


@pytest.fixture
async def live():
    """A live PostgresSemanticIndex on a per-test shared pool.

    ONTA-174 pool-cache reset pattern: pools are cached per DSN but bound to
    the creating test's event loop, so each test resets the cache up front and
    CLOSES its pools afterwards (reset alone would leak real connections).
    """
    from cograph_client.db.pool import close_pg_pools, reset_pg_pools

    reset_pg_pools()
    idx = PostgresSemanticIndex(dsn=DSN, embed_model=FAKE_MODEL, embed_dim=DIM)
    yield idx
    await close_pg_pools()
    reset_pg_pools()


@needs_pg
@pytest.mark.integration
async def test_live_lexical_roundtrip_degraded(live):
    tenant = _t()
    await live.upsert_chunks(
        [
            _chunk(
                "e:solar",
                "Rooftop solar panel installation for residential homes.",
                tenant=tenant,
                attrs={"label": "Solar", "type": "Report"},
            )
        ]
    )
    # The GENERATED tsvector makes a just-written, unembedded chunk lexically
    # findable immediately (the freshness property in the design record).
    res = await live.search(tenant, "solar panel installation")
    assert res.degraded is True
    assert live._last_search_mode == "lexical_only"
    [hit] = res.hits
    assert hit.entity_uri == "e:solar"
    assert hit.attr == "description"
    assert hit.attrs == {"label": "Solar", "type": "Report"}
    assert hit.snippet.startswith("Rooftop solar")
    assert hit.score > 0


@needs_pg
@pytest.mark.integration
async def test_live_hybrid_vector_recall_exact_mode(live):
    tenant = _t()
    await live.upsert_chunks(
        [
            _chunk("e:vec", "cardiac arrhythmia treatment", tenant=tenant),
            _chunk("e:lex", "heart rhythm disorder care", tenant=tenant),
        ]
    )
    [pending_vec] = [
        c for c in await live.fetch_pending(tenant_id=tenant) if c.entity_uri == "e:vec"
    ]
    assert await live.fill_embeddings([pending_vec], [V1], embed_model=FAKE_MODEL) == 1

    # Zero token overlap with e:vec — reachable only through the ANN leg.
    res = await live.search(
        tenant, "heart rhythm disorder", query_embedding=[0.99, 0.1, 0, 0, 0, 0, 0, 0]
    )
    assert res.degraded is False
    assert live._last_search_mode == "ann_exact"  # small tenant → exact scan
    assert {h.entity_uri for h in res.hits} == {"e:vec", "e:lex"}


@needs_pg
@pytest.mark.integration
async def test_live_null_embedding_rows_fts_leg_only(live):
    tenant = _t()
    await live.upsert_chunks(
        [
            _chunk("e:queued", "supply chain resilience strategies", tenant=tenant),
            _chunk("e:filled", "warehouse robotics automation", tenant=tenant),
        ]
    )
    [f] = [
        c
        for c in await live.fetch_pending(tenant_id=tenant)
        if c.entity_uri == "e:filled"
    ]
    await live.fill_embeddings([f], [V2], embed_model=FAKE_MODEL)
    # No lexical overlap at all + a vector: only the embedded row is reachable.
    res = await live.search(tenant, "zzz qqq nothing", query_embedding=V2)
    uris = {h.entity_uri for h in res.hits}
    assert "e:filled" in uris
    assert "e:queued" not in uris  # NULL embedding never enters the ANN leg


@needs_pg
@pytest.mark.integration
async def test_live_replay_preserves_embedding_and_change_requeues(live):
    tenant = _t()
    await live.upsert_chunks([_chunk("e:1", "some text", tenant=tenant)])
    [pending] = await live.fetch_pending(tenant_id=tenant)
    assert await live.fill_embeddings([pending], [V1], embed_model=FAKE_MODEL) == 1
    # Replay the identical doc: unchanged hash → row kept, embedding survives.
    await live.upsert_chunks([_chunk("e:1", "some text", tenant=tenant)])
    assert await live.fetch_pending(tenant_id=tenant) == []
    # Changed doc → replaced, embedding reset to NULL (re-queued).
    await live.upsert_chunks([_chunk("e:1", "new text", tenant=tenant)])
    requeued = await live.fetch_pending(tenant_id=tenant)
    assert [c.chunk_text for c in requeued] == ["new text"]


@needs_pg
@pytest.mark.integration
async def test_live_shrunken_doc_deletes_stale_tail(live):
    tenant = _t()
    await live.upsert_chunks(
        [
            _chunk("e:1", "part one about quasars", ix=0, doc_text="v1", tenant=tenant),
            _chunk("e:1", "part two about pulsars", ix=1, doc_text="v1", tenant=tenant),
        ]
    )
    await live.upsert_chunks(
        [_chunk("e:1", "just quasars now", ix=0, doc_text="v2", tenant=tenant)]
    )
    assert (await live.search(tenant, "pulsars")).hits == []
    assert {h.entity_uri for h in (await live.search(tenant, "quasars")).hits} == {"e:1"}


@needs_pg
@pytest.mark.integration
async def test_live_tenant_kg_type_isolation(live):
    tenant_a, tenant_b = _t(), _t()
    await live.upsert_chunks(
        [
            _chunk("e:a", "identical secret text", tenant=tenant_a),
            _chunk("e:b", "identical secret text", tenant=tenant_b),
            _chunk("e:kg2", "identical secret text", tenant=tenant_a, kg=OTHER_KG),
            _chunk(
                "e:ev",
                "annual gathering downtown",
                tenant=tenant_a,
                attrs={"type": "Event"},
            ),
            _chunk(
                "e:org",
                "annual gathering downtown",
                tenant=tenant_a,
                attrs={"type": "Organization"},
            ),
        ]
    )
    # Security-grade: tenant A must never see B, even with identical text.
    assert {h.entity_uri for h in (await live.search(tenant_a, "identical secret")).hits} == {
        "e:a",
        "e:kg2",
    }
    assert {h.entity_uri for h in (await live.search(tenant_b, "identical secret")).hits} == {
        "e:b"
    }
    # KG narrowing.
    assert {
        h.entity_uri
        for h in (await live.search(tenant_a, "identical secret", kg_name=KG)).hits
    } == {"e:a"}
    # Type filter over the denormalized attrs->>'type'.
    assert {
        h.entity_uri
        for h in (
            await live.search(tenant_a, "annual gathering", type_filter="Event")
        ).hits
    } == {"e:ev"}


@needs_pg
@pytest.mark.integration
async def test_live_delete_and_clear_scoping(live):
    tenant = _t()
    await live.upsert_chunks(
        [
            _chunk("e:1", "delete probe description", tenant=tenant),
            _chunk("e:1", "delete probe notes", attr="notes", tenant=tenant),
            _chunk("e:1", "delete probe sibling", kg=OTHER_KG, tenant=tenant),
            _chunk("e:2", "delete probe other", tenant=tenant),
        ]
    )
    await live.delete("e:1", tenant, kg_name=KG, attr="notes")
    uris = {h.entity_uri for h in (await live.search(tenant, "delete probe")).hits}
    assert uris == {"e:1", "e:2"}
    await live.delete("e:1", tenant, kg_name=KG)
    assert {
        h.entity_uri
        for h in (await live.search(tenant, "delete probe", kg_name=KG)).hits
    } == {"e:2"}
    # The sibling KG's rows for e:1 survived the kg-scoped delete.
    assert {
        h.entity_uri
        for h in (await live.search(tenant, "delete probe", kg_name=OTHER_KG)).hits
    } == {"e:1"}
    await live.clear(tenant, kg_name=OTHER_KG)
    assert (await live.search(tenant, "delete probe", kg_name=OTHER_KG)).hits == []
    await live.clear(tenant)
    assert (await live.search(tenant, "delete probe")).hits == []


@needs_pg
@pytest.mark.integration
async def test_live_queue_order_limits_and_dead_letter(live):
    tenant = _t()
    await live.upsert_chunks(
        [
            _chunk("e:2", "beta text", tenant=tenant),
            _chunk("e:1", "alpha text", tenant=tenant),
        ]
    )
    pending = await live.fetch_pending(tenant_id=tenant)
    assert [c.entity_uri for c in pending] == ["e:1", "e:2"]  # PK order
    assert len(await live.fetch_pending(tenant_id=tenant, limit=1)) == 1

    assert await live.mark_embed_failed(pending[:1], error="429 rate limited") == 1
    [row] = [
        c for c in await live.fetch_pending(tenant_id=tenant) if c.entity_uri == "e:1"
    ]
    assert row.attempt_count == 1 and row.last_error == "429 rate limited"
    # Dead-letter cutoff skips (never deletes) the row.
    assert [
        c.entity_uri for c in await live.fetch_pending(tenant_id=tenant, max_attempts=1)
    ] == ["e:2"]
    # A successful fill clears the error and drains the queue entry.
    assert await live.fill_embeddings([row], [V1], embed_model=FAKE_MODEL) == 1
    assert [
        c.entity_uri for c in await live.fetch_pending(tenant_id=tenant)
    ] == ["e:2"]


@needs_pg
@pytest.mark.integration
async def test_live_stale_hash_guards(live):
    tenant = _t()
    await live.upsert_chunks([_chunk("e:1", "version one", tenant=tenant)])
    [stale] = await live.fetch_pending(tenant_id=tenant)
    await live.upsert_chunks([_chunk("e:1", "version two", tenant=tenant)])
    # The doc changed between fetch and fill: neither the stale vector nor the
    # stale failure may land on the new row.
    assert await live.fill_embeddings([stale], [V1], embed_model=FAKE_MODEL) == 0
    assert await live.mark_embed_failed([stale], error="boom") == 0
    [fresh] = await live.fetch_pending(tenant_id=tenant)
    assert fresh.chunk_text == "version two"
    assert fresh.attempt_count == 0 and fresh.last_error is None


@needs_pg
@pytest.mark.integration
async def test_live_capability_probe_matches_extversion(live):
    """The probe's verdict must be consistent with the installed pgvector —
    passes on both 0.6 (False) and 0.8+ (True)."""
    pool = await live._ensure_pool()
    async with pool.acquire() as conn:
        extversion = await conn.fetchval(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        )
    assert live._iterative_scan == _version_at_least(extversion, (0, 8))


@needs_pg
@pytest.mark.integration
async def test_live_hnsw_index_mode_fallback_never_crashes():
    """threshold=0 forces the index-scan ANN path even on a tiny table —
    exercising hnsw_default on pgvector 0.6 (the deployed-Aurora situation)
    and hnsw_iterative on 0.8+, and proving neither crashes nor silently
    changes the result contract."""
    from cograph_client.db.pool import close_pg_pools, reset_pg_pools

    reset_pg_pools()
    idx = PostgresSemanticIndex(
        dsn=DSN, embed_model=FAKE_MODEL, embed_dim=DIM, exact_scan_threshold=0
    )
    try:
        tenant = _t()
        await idx.upsert_chunks(
            [
                _chunk("e:vec", "cardiac arrhythmia treatment", tenant=tenant),
                _chunk("e:lex", "heart rhythm disorder care", tenant=tenant),
            ]
        )
        [pending_vec] = [
            c
            for c in await idx.fetch_pending(tenant_id=tenant)
            if c.entity_uri == "e:vec"
        ]
        await idx.fill_embeddings([pending_vec], [V1], embed_model=FAKE_MODEL)
        res = await idx.search(
            tenant,
            "heart rhythm disorder",
            query_embedding=[0.99, 0.1, 0, 0, 0, 0, 0, 0],
        )
        expected = "hnsw_iterative" if idx._iterative_scan else "hnsw_default"
        assert idx._last_search_mode == expected
        assert {h.entity_uri for h in res.hits} == {"e:vec", "e:lex"}
    finally:
        await close_pg_pools()
        reset_pg_pools()


@needs_pg
@pytest.mark.integration
async def test_live_dim_mismatch_degrades_not_crashes(live):
    tenant = _t()
    await live.upsert_chunks([_chunk("e:1", "resilient text", tenant=tenant)])
    res = await live.search(tenant, "resilient text", query_embedding=[1.0, 0.0])
    assert res.degraded is True
    assert live._last_search_mode == "lexical_only"
    assert {h.entity_uri for h in res.hits} == {"e:1"}


@needs_pg
@pytest.mark.integration
async def test_live_empty_query_returns_empty(live):
    tenant = _t()
    await live.upsert_chunks([_chunk("e:1", "anything", tenant=tenant)])
    # Punctuation-only text parses to an empty tsquery → empty FTS leg, no error.
    res = await live.search(tenant, "!!! ???")
    assert res.hits == []
