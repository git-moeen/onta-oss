"""pgvector-backed :class:`SemanticIndex` over a generic Postgres DSN (ONTA-176).

Durable, shared-across-tasks adapter using ``asyncpg`` on the process-wide
shared pool (``cograph_client/db/pool.py``, ONTA-174). Vendor-neutral by
construction — the *only* configuration is a plain DSN (``settings.database_url``
or an explicit ``dsn=`` arg) plus a few env knobs documented below. No
cloud-provider ARNs, account IDs, or hostnames live here (same contract as
``spatiotemporal/postgis.py``, which this module deliberately mirrors:
lazy idempotent DDL behind an ``asyncio.Lock``, tolerated extension bootstrap,
structlog, and — like postgis — no bespoke per-call timeouts: statement time
budgets belong to the DSN/server (``statement_timeout``), not to each adapter).

Schema (mirrors :class:`~cograph_client.semantic.protocol.SemanticChunk` 1:1,
plus the store-side GENERATED ``tsv``)::

    CREATE TABLE entity_semantic_chunk (
        tenant_id     text NOT NULL,
        kg_name       text NOT NULL,
        entity_uri    text NOT NULL,
        attr          text NOT NULL,
        chunk_ix      integer NOT NULL,
        chunk_text    text NOT NULL,
        content_hash  text NOT NULL,
        embed_model   text,
        attempt_count integer NOT NULL DEFAULT 0,
        last_error    text,
        embedding     vector(1536),               -- NULL == the durable embed queue
        tsv           tsvector GENERATED ALWAYS AS
                          (to_tsvector('simple'::regconfig, chunk_text)) STORED,
        attrs         jsonb NOT NULL DEFAULT '{}'::jsonb,
        PRIMARY KEY (tenant_id, kg_name, entity_uri, attr, chunk_ix)
    );
    CREATE INDEX entity_semantic_chunk_tsv  ON ... USING GIN (tsv);
    CREATE INDEX entity_semantic_chunk_hnsw ON ... USING hnsw (embedding vector_cosine_ops);

Why ``'simple'`` as the default text-search config: it is **deterministic and
language-agnostic** (no stemming dictionaries whose behavior differs across
locales/installations), which keeps ``tsv`` reproducible for the same
``chunk_text`` everywhere, and it plays predictably with
``websearch_to_tsquery`` quoting (a quoted phrase matches the literal tokens —
no stemmer surprises). Multi-tenant KGs hold mixed-language free text, so a
per-language stemmer would be wrong as often as right. The config is an env
knob (``OMNIX_SEMANTIC_TS_CONFIG``) — **applied at table-creation time only**
(it is baked into the GENERATED column): changing it after the table exists
does NOT re-parse existing rows; you must drop/recreate the table (the
reconciler backfill, ONTA-181, repopulates it). The same knob value is used at
query time for ``websearch_to_tsquery`` so parse rules always match the column.

Hybrid query (ONE SQL round-trip)
---------------------------------

``search`` fuses two CTE legs in a single statement — no Python-side fusion:

* **FTS leg**: top-50 by ``ts_rank_cd`` over ``tsv @@ websearch_to_tsquery`` —
  NULL-embedding rows participate here (a just-written chunk is lexically
  findable before the embed sweep fills it);
* **ANN leg**: top-50 by cosine distance ``embedding <=> $query_vec``, only
  over rows with a filled embedding **whose ``embed_model`` equals the current
  model** (vectors from an older model are not comparable to the query vector);
* both legs are pre-filtered by ``tenant_id`` (mandatory), ``kg_name`` and the
  optional ``type_filter`` (``attrs->>'type'``) **inside the leg**, before the
  LIMIT — filtering after the LIMIT would silently shrink recall;
* fusion is RRF with ``k=60``: chunks are grouped by PK summing
  ``1/(60+rank)`` across legs, then grouped into entities (an entity scores as
  its best fused chunk — ``DISTINCT ON (entity_uri)``), top_k entities return.

The constants (k=60, 50 per leg, snippet shape) are imported from
``memory.py`` so the two backends cannot drift apart — the smoke-parity suite
asserts loose top-k overlap between them.

**Type-filter staleness caveat:** ``attrs`` (including ``attrs->>'type'``) is
denormalized at write time. If an entity's type/label changes later without its
marked text changing, the filter matches the *stale* display value until the
ONTA-181 reconciler re-upserts the doc. This is the accepted cost of zero-join
hits; the reconciler is the corrective mechanism.

Filtered-HNSW recall trap + ANN mode selection
----------------------------------------------

An HNSW index scan returns ~``hnsw.ef_search`` candidates and only THEN applies
the WHERE filter — a highly selective tenant/kg/type filter can leave far fewer
than 50 rows (or zero) even though matches exist. pgvector **0.8.0** fixes this
with ``SET LOCAL hnsw.iterative_scan = 'relaxed_order'`` (keep scanning until
enough filtered rows are found), but the deployed Aurora (PostgreSQL 16.x) and
the test containers ship **pgvector 0.6.0**, which lacks the GUC. So the ANN
leg picks one of three modes per query — logged every time, **never silent**:

* ``ann_exact`` — when the (tenant, kg, type)-scoped embedded-row count is at
  or below ``OMNIX_SEMANTIC_EXACT_SCAN_THRESHOLD`` (default 10 000; counted
  with a ``LIMIT threshold+1`` bounded probe, so the gate itself stays cheap).
  The leg selects the filtered rows into a ``MATERIALIZED`` CTE and orders that
  by distance: a materialized CTE is an optimization fence, so the sort can
  never use the HNSW index — **exact by construction** on every pgvector
  version, and a top-50 heapsort over ≤10k rows is a few milliseconds. This is
  the portable exact-scan technique: forcing planner GUCs (``enable_indexscan``
  etc.) is global to the statement and version-fragile, whereas the fence is
  guaranteed semantics.
* ``hnsw_iterative`` — large filtered set AND the pool's pgvector supports
  iterative scans: ``SET LOCAL hnsw.iterative_scan = 'relaxed_order'`` (plus a
  raised ``hnsw.ef_search`` — the 40-row default would cap the 50-row leg).
* ``hnsw_default`` — large filtered set on pgvector < 0.8: plain HNSW scan with
  raised ``ef_search``. Recall on heavily-filtered queries may be reduced; a
  once-per-instance ``warning`` log says so explicitly.

Capability detection runs once per pool at DDL time: attempt the ``SET LOCAL``
in a throwaway transaction and catch the invalid/undefined-GUC error, then
cross-check ``pg_extension.extversion >= 0.8``. BOTH layers are load-bearing —
verified empirically on pgvector 0.6.0: whether the SET raises depends on
whether pgvector's shared library is already loaded in the session (its
``_PG_init`` reserves the ``hnsw.`` prefix → ``invalid configuration parameter
name``); in a session that has not yet touched a vector operator, the same SET
**silently succeeds** by minting a placeholder GUC. Trusting the SET alone
could therefore make us believe we are iterating when we are not — a silent
recall degrade, the one thing this module must never do. The extversion
cross-check kills that false positive deterministically.

Degraded mode: ``query_embedding=None`` (embed service down/unconfigured) runs
the FTS leg only and returns ``degraded=True`` — the protocol's explicit
reduced-recall signal. A query embedding whose dimension does not match the
table's is treated the same way (lexical-only + degraded + warning log) rather
than crashing the query.

Vector binding: vectors are bound as *text* (``'[1.0,2.0,...]'``) and cast
``$n::text::vector`` in SQL. This works with or without the pgvector asyncpg
codec, which matters because the codec's per-connection registration
(:func:`register_pool_init`) can only succeed once the ``vector`` extension
exists — on a fresh database the very first connections predate our DDL. After
the DDL runs we expire the pool's connections so every later connection gets
the codec; the text-cast binding keeps the store correct either way.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from typing import Any, Optional, Sequence

import structlog

from cograph_client.config import settings
from cograph_client.nlp.embed_client import EMBEDDING_DIM, EMBEDDING_MODEL
from cograph_client.semantic.memory import _CANDIDATES_PER_LEG, _RRF_K, _snippet
from cograph_client.semantic.protocol import (
    SemanticChunk,
    SemanticHit,
    SemanticSearchResult,
)

logger = structlog.stdlib.get_logger("cograph.semantic.postgres")

#: Env knobs (read at instance construction; constructor args override).
#: They deliberately live here rather than in ``config.Settings`` — they are
#: adapter-internal tuning, not deployment wiring, and the settings object is
#: shared surface owned by other subsystems.
TS_CONFIG_ENV = "OMNIX_SEMANTIC_TS_CONFIG"
EMBED_DIM_ENV = "OMNIX_SEMANTIC_EMBED_DIM"
EXACT_SCAN_THRESHOLD_ENV = "OMNIX_SEMANTIC_EXACT_SCAN_THRESHOLD"

DEFAULT_TS_CONFIG = "simple"
DEFAULT_EXACT_SCAN_THRESHOLD = 10_000

#: ``hnsw.ef_search`` used in the index-scan modes. Must exceed the per-leg
#: LIMIT (50): the default of 40 would cap the ANN leg below its candidate
#: budget even before any filtering.
HNSW_EF_SEARCH = 80

#: regconfig / identifier whitelist — the ts config is baked into DDL via
#: f-string, so it must never be attacker-controllable free text.
_IDENT_RE = re.compile(r"^[a-z_][a-z0-9_]*$")

_PK_COLS = "tenant_id, kg_name, entity_uri, attr, chunk_ix"
_CHUNK_COLS = f"{_PK_COLS}, chunk_text, attrs"


def _version_at_least(extversion: Optional[str], minimum: tuple[int, int]) -> bool:
    """Parse a pg_extension.extversion string ("0.8.0", "0.6.0", …) and compare.

    Unparseable / missing versions compare False — we then rely on the SET
    probe alone (documented in the module docstring's capability-detection
    section).
    """
    if not extversion:
        return False
    parts: list[int] = []
    for piece in str(extversion).split(".")[:2]:
        m = re.match(r"\d+", piece)
        if m is None:
            return False
        parts.append(int(m.group()))
    if len(parts) < 2:
        return False
    return (parts[0], parts[1]) >= minimum


def _vector_text(vec: Optional[Sequence[float]]) -> Optional[str]:
    """Serialize an embedding to pgvector's text literal (``[1.0,2.0]``).

    ``None`` stays ``None`` (binds as a typed NULL through ``::text::vector``).
    Bound as text + cast in SQL so it works with or without the asyncpg codec
    (see the module docstring's vector-binding note).
    """
    if vec is None:
        return None
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def _attrs_json(attrs: dict[str, Any]) -> str:
    """Serialize attrs to a JSON string for the ``$::jsonb`` bind."""
    return json.dumps(attrs or {})


def _parse_attrs(value: Any) -> dict[str, Any]:
    """asyncpg may hand jsonb back as str/bytes (no codec) or a decoded dict."""
    if value is None:
        return {}
    if isinstance(value, (bytes, bytearray)):
        value = value.decode()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {}
    if isinstance(value, dict):
        return value
    return {}


def _rowcount(status: str) -> int:
    """Rows affected from an asyncpg command status tag (``"UPDATE 3"``)."""
    try:
        return int(str(status).rsplit(" ", 1)[-1])
    except (ValueError, IndexError):
        return 0


async def _register_vector_codec(conn: Any) -> None:
    """Per-connection pool init hook: install pgvector's asyncpg codec.

    Tolerates failure (warn, continue): on a fresh database the ``vector``
    type does not exist until our lazy DDL runs ``CREATE EXTENSION`` — the
    store stays correct without the codec because every vector bind goes
    through ``$n::text::vector``. After DDL we expire the pool's connections so
    subsequent ones register cleanly.
    """
    try:
        from pgvector.asyncpg import register_vector

        await register_vector(conn)
    except Exception as exc:  # noqa: BLE001 — codec is an optimization, not a dependency
        logger.warning("semantic_pgvector_codec_skipped", error=str(exc))


class PostgresSemanticIndex:
    """Durable :class:`SemanticIndex` backed by Postgres + pgvector via asyncpg."""

    _TABLE = "entity_semantic_chunk"
    _GIN_INDEX = "entity_semantic_chunk_tsv"
    _HNSW_INDEX = "entity_semantic_chunk_hnsw"

    def __init__(
        self,
        dsn: Optional[str] = None,
        *,
        embed_model: Optional[str] = None,
        embed_dim: Optional[int] = None,
        ts_config: Optional[str] = None,
        exact_scan_threshold: Optional[int] = None,
    ) -> None:
        self._dsn = dsn if dsn is not None else settings.database_url
        # The "current" embed model: the ANN leg only trusts vectors produced
        # by it (a query vector is not comparable across models). Defaults to
        # the shared embed client's model so the embed-fill sweep (ONTA-181)
        # and the query side agree by construction.
        self._embed_model = embed_model or EMBEDDING_MODEL

        dim = embed_dim if embed_dim is not None else int(
            os.environ.get(EMBED_DIM_ENV, EMBEDDING_DIM)
        )
        if not (0 < dim <= 16_000):  # pgvector's hard column-dimension cap
            raise ValueError(f"invalid semantic embedding dimension: {dim}")
        self._embed_dim = dim

        cfg = ts_config or os.environ.get(TS_CONFIG_ENV, DEFAULT_TS_CONFIG)
        if not _IDENT_RE.match(cfg):
            # Baked into DDL via f-string — refuse anything but a plain
            # regconfig identifier (never SQL-injectable).
            raise ValueError(f"invalid text-search config name: {cfg!r}")
        self._ts_config = cfg

        threshold = (
            exact_scan_threshold
            if exact_scan_threshold is not None
            else int(
                os.environ.get(
                    EXACT_SCAN_THRESHOLD_ENV, DEFAULT_EXACT_SCAN_THRESHOLD
                )
            )
        )
        self._exact_scan_threshold = max(int(threshold), 0)

        self._pool: Any = None
        # DDL + capability probe run lazily once; guard concurrent first
        # callers. Pool creation is delegated to the process-wide shared pool
        # (cograph_client.db.pool, ONTA-174) — one pool per DSN across stores.
        self._lock = asyncio.Lock()
        # Per-pool capability: pgvector >= 0.8 iterative scans (see module
        # docstring). False until the probe says otherwise.
        self._iterative_scan = False
        self._warned_hnsw_default = False
        # Ops/test introspection: which ANN mode the last search ran
        # (lexical_only / ann_exact / hnsw_iterative / hnsw_default). The same
        # value is logged on every search — never a silent degrade.
        self._last_search_mode: Optional[str] = None

    # ------------------------------------------------------------------ setup
    async def _ensure_pool(self) -> Any:
        """Acquire the shared pool + apply DDL + probe capabilities, once."""
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is not None:
                return self._pool
            from cograph_client.db.pool import get_pg_pool, register_pool_init

            # Install the codec hook BEFORE creating/joining the pool so its
            # very first connections run it (register_pool_init also expires
            # any pre-existing pool's connections, so joining late is safe).
            register_pool_init(_register_vector_codec)

            pool = await get_pg_pool(self._dsn)
            async with pool.acquire() as conn:
                # Extension may need superuser; infra (COG-102 pattern)
                # bootstraps it in managed environments. Tolerate a permission
                # error — the table DDL below surfaces a clear "type vector
                # does not exist" if the extension is genuinely absent.
                try:
                    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                except Exception as exc:  # noqa: BLE001 - best-effort bootstrap
                    logger.warning(
                        "semantic_extension_bootstrap_skipped",
                        extension="vector",
                        error=str(exc),
                    )
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._TABLE} (
                        tenant_id     text NOT NULL,
                        kg_name       text NOT NULL,
                        entity_uri    text NOT NULL,
                        attr          text NOT NULL,
                        chunk_ix      integer NOT NULL,
                        chunk_text    text NOT NULL,
                        content_hash  text NOT NULL,
                        embed_model   text,
                        attempt_count integer NOT NULL DEFAULT 0,
                        last_error    text,
                        embedding     vector({self._embed_dim}),
                        tsv           tsvector GENERATED ALWAYS AS
                            (to_tsvector('{self._ts_config}'::regconfig, chunk_text)) STORED,
                        attrs         jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                        PRIMARY KEY (tenant_id, kg_name, entity_uri, attr, chunk_ix)
                    )
                    """
                )
                # Lexical leg: GIN over the generated tsvector.
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._GIN_INDEX} "
                    f"ON {self._TABLE} USING GIN (tsv)"
                )
                # ANN leg: HNSW, cosine (matches the memory backend + <=>).
                # NULL embeddings are not indexed — the queue costs nothing here.
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._HNSW_INDEX} "
                    f"ON {self._TABLE} USING hnsw (embedding vector_cosine_ops)"
                )
                await self._probe_capabilities(conn)
                # The codec hook may have failed on connections created before
                # CREATE EXTENSION (fresh DB). Recycle so every connection
                # handed out from now on registers the codec successfully.
                # NOTE: asyncpg's Pool.expire_connections is a coroutine
                # (test fakes may keep it sync) — await when awaitable.
                try:
                    result = pool.expire_connections()
                    if inspect.isawaitable(result):
                        await result
                except Exception:  # noqa: BLE001 — text-cast binds keep us correct anyway
                    logger.warning("semantic_pool_expire_failed", exc_info=True)
            self._pool = pool
            return self._pool

    async def _probe_capabilities(self, conn: Any) -> None:
        """Feature-detect pgvector iterative scans, once per pool (module
        docstring: SET probe + extversion cross-check)."""
        extversion: Optional[str] = None
        try:
            extversion = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
        except Exception as exc:  # noqa: BLE001 — probe must never break setup
            logger.warning("semantic_extversion_probe_failed", error=str(exc))

        supports = False
        try:
            async with conn.transaction():
                # Throwaway transaction: SET LOCAL evaporates at commit. On
                # pgvector builds that reserve the "hnsw." prefix this raises
                # (unrecognized/invalid configuration parameter) when the GUC
                # is absent — one of the two fallback triggers.
                await conn.execute(
                    "SET LOCAL hnsw.iterative_scan = 'relaxed_order'"
                )
            supports = True
        except Exception as exc:  # noqa: BLE001 — undefined GUC == unsupported
            logger.info(
                "semantic_hnsw_iterative_scan_unsupported", error=str(exc)
            )
        if supports and not _version_at_least(extversion, (0, 8)):
            # SET "succeeded" on a pre-0.8 pgvector: Postgres minted a
            # placeholder GUC (happens on 0.6.0 whenever the pgvector library
            # is not yet loaded in the probing session — VERIFIED locally).
            # Trusting it would silently degrade recall, exactly what we must
            # not do.
            logger.warning(
                "semantic_iterative_scan_placeholder_guc",
                extversion=extversion,
            )
            supports = False
        self._iterative_scan = supports
        logger.info(
            "semantic_pgvector_capabilities",
            extversion=extversion,
            iterative_scan=supports,
            exact_scan_threshold=self._exact_scan_threshold,
        )

    # ----------------------------------------------------------------- writes
    _UPSERT_SQL_TEMPLATE = """
        INSERT INTO {table}
            (tenant_id, kg_name, entity_uri, attr, chunk_ix, chunk_text,
             content_hash, embedding, embed_model, attempt_count, last_error, attrs)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8::text::vector, $9, $10, $11, $12::jsonb)
        ON CONFLICT (tenant_id, kg_name, entity_uri, attr, chunk_ix) DO UPDATE SET
            chunk_text    = EXCLUDED.chunk_text,
            content_hash  = EXCLUDED.content_hash,
            embedding     = EXCLUDED.embedding,
            embed_model   = EXCLUDED.embed_model,
            attempt_count = EXCLUDED.attempt_count,
            last_error    = EXCLUDED.last_error,
            attrs         = EXCLUDED.attrs
        WHERE {table}.content_hash IS DISTINCT FROM EXCLUDED.content_hash
    """

    async def upsert_chunks(self, chunks: Sequence[SemanticChunk]) -> None:
        """Replace-per-doc upsert (the Protocol's complete-document contract).

        One transaction, two batched statements:

        * an ``INSERT ... ON CONFLICT (pk) DO UPDATE ... WHERE content_hash IS
          DISTINCT FROM EXCLUDED.content_hash`` — an unchanged-hash row is left
          untouched (its filled embedding survives replay, never re-queued); a
          changed-hash row is replaced wholesale (embedding reset per the
          incoming chunk, normally NULL → queued);
        * a tail ``DELETE ... chunk_ix >= doc_len`` per (entity, attr) doc —
          the stale-tail case when a doc re-chunked shorter.
        """
        if not chunks:
            return
        pool = await self._ensure_pool()

        rows: list[tuple] = []
        doc_lens: dict[tuple[str, str, str, str], int] = {}
        for c in chunks:
            rows.append(
                (
                    c.tenant_id,
                    c.kg_name,
                    c.entity_uri,
                    c.attr,
                    c.chunk_ix,
                    c.chunk_text,
                    c.content_hash,
                    _vector_text(c.embedding),
                    c.embed_model,
                    c.attempt_count,
                    c.last_error,
                    _attrs_json(c.attrs),
                )
            )
            dk = c.doc_key()
            doc_lens[dk] = max(doc_lens.get(dk, 0), c.chunk_ix + 1)

        upsert_sql = self._UPSERT_SQL_TEMPLATE.format(table=self._TABLE)
        tail_sql = (
            f"DELETE FROM {self._TABLE} "
            "WHERE tenant_id = $1 AND kg_name = $2 AND entity_uri = $3 "
            "AND attr = $4 AND chunk_ix >= $5"
        )
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(upsert_sql, rows)
                await conn.executemany(
                    tail_sql,
                    [(t, k, u, a, n) for (t, k, u, a), n in doc_lens.items()],
                )

    async def delete(
        self,
        entity_uri: str,
        tenant_id: str,
        *,
        kg_name: Optional[str] = None,
        attr: Optional[str] = None,
    ) -> None:
        pool = await self._ensure_pool()
        sql = f"DELETE FROM {self._TABLE} WHERE tenant_id = $1 AND entity_uri = $2"
        params: list[Any] = [tenant_id, entity_uri]
        if kg_name is not None:
            params.append(kg_name)
            sql += f" AND kg_name = ${len(params)}"
        if attr is not None:
            params.append(attr)
            sql += f" AND attr = ${len(params)}"
        async with pool.acquire() as conn:
            await conn.execute(sql, *params)

    async def clear(self, tenant_id: str, *, kg_name: Optional[str] = None) -> None:
        pool = await self._ensure_pool()
        sql = f"DELETE FROM {self._TABLE} WHERE tenant_id = $1"
        params: list[Any] = [tenant_id]
        if kg_name is not None:
            params.append(kg_name)
            sql += f" AND kg_name = ${len(params)}"
        async with pool.acquire() as conn:
            await conn.execute(sql, *params)

    # ----------------------------------------------------------------- search
    @staticmethod
    def _leg_filter_sql() -> str:
        """The shared pre-filter both legs apply INSIDE the leg (before its
        LIMIT): tenant (mandatory), optional kg, optional denormalized type."""
        return (
            "tenant_id = $1\n"
            "          AND ($2::text IS NULL OR kg_name = $2::text)\n"
            "          AND ($3::text IS NULL OR attrs->>'type' = $3::text)"
        )

    def _fts_cte(self) -> str:
        # websearch_to_tsquery: never raises on arbitrary user text and maps
        # double quotes to phrase queries (the 'simple' config keeps phrase
        # tokens literal). Rank ties break on the PK for determinism.
        return f"""
        fts AS (
            SELECT {_CHUNK_COLS},
                   row_number() OVER (
                       ORDER BY ts_rank_cd(tsv, websearch_to_tsquery($4::text::regconfig, $5)) DESC,
                                {_PK_COLS}
                   ) AS rank
            FROM {self._TABLE}
            WHERE {self._leg_filter_sql()}
              AND tsv @@ websearch_to_tsquery($4::text::regconfig, $5)
            ORDER BY rank
            LIMIT {_CANDIDATES_PER_LEG}
        )"""

    def _ann_leg_source(self, exact: bool) -> str:
        """FROM-clause source for the ANN leg: the raw table (index-scannable)
        or the MATERIALIZED filtered pool (optimization fence → exact)."""
        if exact:
            return "ann_pool"
        return self._TABLE

    def _ann_cte(self, *, exact: bool) -> str:
        pool_cte = ""
        filters = (
            f"WHERE {self._leg_filter_sql()}\n"
            "              AND embedding IS NOT NULL\n"
            "              AND embed_model = $6"
        )
        if exact:
            # MATERIALIZED = optimization fence: the distance sort below can
            # never reach back to the HNSW index — exact by construction on
            # every pgvector version (the 0.6-portable exact-scan technique).
            pool_cte = f"""
        ann_pool AS MATERIALIZED (
            SELECT {_CHUNK_COLS}, embedding
            FROM {self._TABLE}
            {filters}
        ),"""
            inner_filters = ""
        else:
            inner_filters = "            " + filters + "\n"
        return f"""{pool_cte}
        ann AS (
            SELECT {_CHUNK_COLS},
                   row_number() OVER (ORDER BY dist, {_PK_COLS}) AS rank
            FROM (
                SELECT {_CHUNK_COLS},
                       embedding <=> $7::text::vector AS dist
                FROM {self._ann_leg_source(exact)}
    {inner_filters}            ORDER BY embedding <=> $7::text::vector
                LIMIT {_CANDIDATES_PER_LEG}
            ) ann_leg
        )"""

    def _fusion_tail(self, *, legs: int, top_k_param: int) -> str:
        """RRF fusion + entity grouping, identical shape to memory.py:
        chunk score = Σ 1/(60+rank) across legs; entity = its best chunk."""
        if legs == 2:
            unioned = f"""
        unioned AS (
            SELECT {_CHUNK_COLS}, rank FROM fts
            UNION ALL
            SELECT {_CHUNK_COLS}, rank FROM ann
        ),"""
            source = "unioned"
        else:
            unioned = ""
            source = "fts"
        return f"""{unioned}
        chunk_scores AS (
            SELECT entity_uri, attr, chunk_ix, chunk_text, attrs,
                   sum(1.0 / ({_RRF_K} + rank))::float8 AS score
            FROM {source}
            GROUP BY {_CHUNK_COLS}
        ),
        best AS (
            SELECT DISTINCT ON (entity_uri)
                   entity_uri, attr, chunk_text, attrs, score
            FROM chunk_scores
            ORDER BY entity_uri, score DESC, attr, chunk_ix
        )
        SELECT entity_uri, attr, chunk_text, attrs, score
        FROM best
        ORDER BY score DESC, entity_uri
        LIMIT ${top_k_param}"""

    def _hybrid_sql(self, *, exact: bool) -> str:
        """The single-round-trip hybrid statement: FTS CTE + ANN CTE + fusion."""
        return (
            "WITH"
            + self._fts_cte()
            + ","
            + self._ann_cte(exact=exact)
            + ","
            + self._fusion_tail(legs=2, top_k_param=8)
        )

    def _lexical_sql(self) -> str:
        """Degraded mode: FTS leg only (still grouped + RRF-scored so scores
        stay comparable in shape)."""
        return "WITH" + self._fts_cte() + "," + self._fusion_tail(
            legs=1, top_k_param=6
        )

    _GATE_SQL_TEMPLATE = """
        SELECT count(*) FROM (
            SELECT 1 FROM {table}
            WHERE tenant_id = $1
              AND ($2::text IS NULL OR kg_name = $2::text)
              AND ($3::text IS NULL OR attrs->>'type' = $3::text)
              AND embedding IS NOT NULL
              AND embed_model = $4
            LIMIT $5
        ) gate
    """

    async def search(
        self,
        tenant_id: str,
        query_text: str,
        *,
        query_embedding: Optional[Sequence[float]] = None,
        kg_name: Optional[str] = None,
        type_filter: Optional[str] = None,
        top_k: int = 10,
    ) -> SemanticSearchResult:
        pool = await self._ensure_pool()
        top_k = max(int(top_k), 0)

        degraded = query_embedding is None
        qvec: Optional[str] = None
        if query_embedding is not None:
            if len(query_embedding) != self._embed_dim:
                # A mis-dimensioned vector cannot run against the column.
                # Degrade to lexical-only EXPLICITLY (flag + log) rather than
                # crash the query or silently return a broken ANN leg.
                logger.warning(
                    "semantic_query_embedding_dim_mismatch",
                    got=len(query_embedding),
                    expected=self._embed_dim,
                )
                degraded = True
            else:
                qvec = _vector_text(query_embedding)

        if top_k == 0:
            return SemanticSearchResult(hits=[], degraded=degraded)

        async with pool.acquire() as conn:
            if degraded:
                mode = "lexical_only"
                rows = await conn.fetch(
                    self._lexical_sql(),
                    tenant_id,
                    kg_name,
                    type_filter,
                    self._ts_config,
                    query_text,
                    top_k,
                )
            else:
                # Bounded gate: how many embedded rows survive the leg filter?
                # LIMIT threshold+1 keeps the probe O(threshold) even on huge
                # tenants; the hybrid ranking itself stays ONE statement.
                gate = await conn.fetchval(
                    self._GATE_SQL_TEMPLATE.format(table=self._TABLE),
                    tenant_id,
                    kg_name,
                    type_filter,
                    self._embed_model,
                    self._exact_scan_threshold + 1,
                )
                params = [
                    tenant_id,
                    kg_name,
                    type_filter,
                    self._ts_config,
                    query_text,
                    self._embed_model,
                    qvec,
                    top_k,
                ]
                if int(gate or 0) <= self._exact_scan_threshold:
                    mode = "ann_exact"
                    rows = await conn.fetch(self._hybrid_sql(exact=True), *params)
                else:
                    mode = (
                        "hnsw_iterative" if self._iterative_scan else "hnsw_default"
                    )
                    if mode == "hnsw_default" and not self._warned_hnsw_default:
                        self._warned_hnsw_default = True
                        logger.warning(
                            "semantic_hnsw_default_mode",
                            detail=(
                                "pgvector < 0.8: filtered HNSW scans may return "
                                "fewer than the requested candidates on highly "
                                "selective filters (no iterative_scan); "
                                "recall may be reduced for such queries"
                            ),
                        )
                    async with conn.transaction():
                        # SET LOCAL scopes to this transaction only. ef_search
                        # must exceed the 50-row leg budget (default is 40).
                        await conn.execute(
                            f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}"
                        )
                        if self._iterative_scan:
                            await conn.execute(
                                "SET LOCAL hnsw.iterative_scan = 'relaxed_order'"
                            )
                        rows = await conn.fetch(
                            self._hybrid_sql(exact=False), *params
                        )

        self._last_search_mode = mode
        logger.debug(
            "semantic_search_mode",
            mode=mode,
            tenant_id=tenant_id,
            kg_name=kg_name,
            type_filter=type_filter,
            degraded=degraded,
            hits=len(rows),
        )
        hits = [
            SemanticHit(
                entity_uri=r["entity_uri"],
                attrs=_parse_attrs(r["attrs"]),
                snippet=_snippet(r["chunk_text"]),
                attr=r["attr"],
                score=float(r["score"]),
            )
            for r in rows
        ]
        return SemanticSearchResult(hits=hits, degraded=degraded)

    # -- embed-fill sweep seam (ONTA-181's reconciler drives these) ----------
    async def fetch_pending(
        self,
        *,
        limit: int = 100,
        max_attempts: Optional[int] = None,
        tenant_id: Optional[str] = None,
        kg_name: Optional[str] = None,
    ) -> list[SemanticChunk]:
        pool = await self._ensure_pool()
        sql = f"""
            SELECT tenant_id, kg_name, entity_uri, attr, chunk_ix, chunk_text,
                   content_hash, embed_model, attempt_count, last_error, attrs
            FROM {self._TABLE}
            WHERE embedding IS NULL
              AND ($1::int IS NULL OR attempt_count < $1::int)
              AND ($2::text IS NULL OR tenant_id = $2::text)
              AND ($3::text IS NULL OR kg_name = $3::text)
            ORDER BY {_PK_COLS}
            LIMIT $4
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                sql, max_attempts, tenant_id, kg_name, max(int(limit), 0)
            )
        return [
            SemanticChunk(
                tenant_id=r["tenant_id"],
                kg_name=r["kg_name"],
                entity_uri=r["entity_uri"],
                attr=r["attr"],
                chunk_ix=r["chunk_ix"],
                chunk_text=r["chunk_text"],
                content_hash=r["content_hash"],
                embedding=None,  # by selection: these are the pending rows
                embed_model=r["embed_model"],
                attempt_count=r["attempt_count"],
                last_error=r["last_error"],
                attrs=_parse_attrs(r["attrs"]),
            )
            for r in rows
        ]

    async def fill_embeddings(
        self,
        chunks: Sequence[SemanticChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        embed_model: str,
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must be parallel sequences"
            )
        if not chunks:
            return 0
        pool = await self._ensure_pool()
        # content_hash is the optimistic-concurrency token: a fill only lands
        # if the row still holds the SAME doc version and is still unembedded
        # (protocol contract — a stale vector must never land on new text).
        sql = f"""
            UPDATE {self._TABLE}
            SET embedding = $6::text::vector, embed_model = $7, last_error = NULL
            WHERE tenant_id = $1 AND kg_name = $2 AND entity_uri = $3
              AND attr = $4 AND chunk_ix = $5
              AND content_hash = $8 AND embedding IS NULL
        """
        filled = 0
        async with pool.acquire() as conn:
            async with conn.transaction():
                for c, vec in zip(chunks, embeddings):
                    status = await conn.execute(
                        sql,
                        c.tenant_id,
                        c.kg_name,
                        c.entity_uri,
                        c.attr,
                        c.chunk_ix,
                        _vector_text([float(x) for x in vec]),
                        embed_model,
                        c.content_hash,
                    )
                    filled += _rowcount(status)
        return filled

    async def mark_embed_failed(
        self, chunks: Sequence[SemanticChunk], *, error: str
    ) -> int:
        if not chunks:
            return 0
        pool = await self._ensure_pool()
        # Same content_hash + still-NULL guard as fill_embeddings; the row
        # stays queued (embedding NULL) until max_attempts dead-letters it.
        sql = f"""
            UPDATE {self._TABLE}
            SET attempt_count = attempt_count + 1, last_error = $6
            WHERE tenant_id = $1 AND kg_name = $2 AND entity_uri = $3
              AND attr = $4 AND chunk_ix = $5
              AND content_hash = $7 AND embedding IS NULL
        """
        marked = 0
        async with pool.acquire() as conn:
            async with conn.transaction():
                for c in chunks:
                    status = await conn.execute(
                        sql,
                        c.tenant_id,
                        c.kg_name,
                        c.entity_uri,
                        c.attr,
                        c.chunk_ix,
                        error,
                        c.content_hash,
                    )
                    marked += _rowcount(status)
        return marked
