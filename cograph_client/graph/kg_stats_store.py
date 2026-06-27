"""Durable per-KG dashboard-summary stats store.

Holds one materialized summary row per ``(tenant_id, kg_name)`` — total entity
count, total edge count, and the per-type entity breakdown — so the dashboard
(and any "list my graphs with their size" view) reads a tiny relational lookup
instead of querying Neptune on the hot path.

The numbers are computed exactly once, by the shared post-write housekeeping
path: :func:`cograph_client.api.routes.explore.recompute_kg_stats` already scans
the KG and writes the precomputed stats graph after every ingest/enrichment
write; it now also UPSERTs the aggregate into this store. Rows for KGs that
predate this store are backfilled lazily from their existing stats graph the
first time they're listed (mirrors how ``list_kgs`` lazily materializes a KG's
triple count). So this store is a read cache that is *kept fresh by the one
write path*, never a second source of truth.

Backends mirror the ``JobStore`` pattern so the deployment is swappable:

- :class:`InMemoryKgStatsStore` — the zero-config default; non-durable,
  per-process.
- :class:`PostgresKgStatsStore` — durable, shared across tasks, over a generic
  Postgres DSN (``settings.database_url`` / ``OMNIX_DATABASE_URL``). Vendor
  neutral: a plain DSN, no cloud-provider identifiers, works against any
  Postgres (local, Aurora, Neon, Supabase, ...).

This module lives in ``graph/`` and stays importable without pulling in the API
routes or ``nlp`` — it is pure storage.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field

from cograph_client.config import settings


class KgStats(BaseModel):
    """A materialized per-KG summary row."""

    tenant_id: str
    kg_name: str
    entity_count: int = 0
    edge_count: int = 0
    # Per-type entity counts, keyed by type leaf name (e.g. {"Person": 5000}).
    type_breakdown: dict[str, int] = Field(default_factory=dict)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KgStatsStore(Protocol):
    async def upsert(self, stats: KgStats) -> None: ...
    async def get(self, tenant_id: str, kg_name: str) -> Optional[KgStats]: ...
    async def list_for_tenant(self, tenant_id: str) -> list[KgStats]: ...
    async def delete(self, tenant_id: str, kg_name: str) -> None: ...


class InMemoryKgStatsStore:
    """Zero-config default; non-durable, per-process."""

    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], KgStats] = {}
        self._lock = asyncio.Lock()

    async def upsert(self, stats: KgStats) -> None:
        async with self._lock:
            self._rows[(stats.tenant_id, stats.kg_name)] = stats.model_copy(deep=True)

    async def get(self, tenant_id: str, kg_name: str) -> Optional[KgStats]:
        async with self._lock:
            row = self._rows.get((tenant_id, kg_name))
            return row.model_copy(deep=True) if row else None

    async def list_for_tenant(self, tenant_id: str) -> list[KgStats]:
        async with self._lock:
            return [
                r.model_copy(deep=True)
                for (t, _), r in self._rows.items()
                if t == tenant_id
            ]

    async def delete(self, tenant_id: str, kg_name: str) -> None:
        async with self._lock:
            self._rows.pop((tenant_id, kg_name), None)


class PostgresKgStatsStore:
    """Durable ``KgStatsStore`` over a generic Postgres DSN via asyncpg.

    The full :class:`KgStats` is serialized to a ``payload`` jsonb column; the
    columns the dashboard reads/aggregates on (``entity_count``, ``edge_count``,
    ``updated_at``) are mirrored alongside it so the common "list a tenant's KG
    sizes" query never has to parse jsonb.

    The pool and table are created lazily on first use so importing this module
    (and constructing the store) never touches the network; the DDL is
    idempotent (``CREATE TABLE IF NOT EXISTS``). Vendor-neutral by construction:
    the only configuration is a plain DSN.
    """

    _TABLE = "cograph_kg_stats"

    def __init__(self, dsn: Optional[str] = None) -> None:
        self._dsn = dsn if dsn is not None else settings.database_url
        self._pool: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is not None:
                return self._pool
            import asyncpg  # imported lazily so the dependency is optional

            pool = await asyncpg.create_pool(dsn=self._dsn)
            async with pool.acquire() as conn:
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._TABLE} (
                        tenant_id text NOT NULL,
                        kg_name text NOT NULL,
                        entity_count bigint NOT NULL DEFAULT 0,
                        edge_count bigint NOT NULL DEFAULT 0,
                        updated_at timestamptz,
                        payload jsonb NOT NULL,
                        PRIMARY KEY (tenant_id, kg_name)
                    )
                    """
                )
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._TABLE}_tenant_idx "
                    f"ON {self._TABLE} (tenant_id)"
                )
            self._pool = pool
            return self._pool

    @staticmethod
    def _row_to_stats(row: Any) -> KgStats:
        payload = row["payload"]
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode()
        if isinstance(payload, str):
            return KgStats.model_validate_json(payload)
        return KgStats.model_validate(payload)

    async def upsert(self, stats: KgStats) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._TABLE}
                    (tenant_id, kg_name, entity_count, edge_count, updated_at, payload)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (tenant_id, kg_name) DO UPDATE SET
                    entity_count = EXCLUDED.entity_count,
                    edge_count = EXCLUDED.edge_count,
                    updated_at = EXCLUDED.updated_at,
                    payload = EXCLUDED.payload
                """,
                stats.tenant_id,
                stats.kg_name,
                stats.entity_count,
                stats.edge_count,
                stats.updated_at,
                stats.model_dump_json(),
            )

    async def get(self, tenant_id: str, kg_name: str) -> Optional[KgStats]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT payload FROM {self._TABLE} "
                f"WHERE tenant_id = $1 AND kg_name = $2",
                tenant_id,
                kg_name,
            )
        return self._row_to_stats(row) if row is not None else None

    async def list_for_tenant(self, tenant_id: str) -> list[KgStats]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT payload FROM {self._TABLE} WHERE tenant_id = $1",
                tenant_id,
            )
        return [self._row_to_stats(r) for r in rows]

    async def delete(self, tenant_id: str, kg_name: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._TABLE} WHERE tenant_id = $1 AND kg_name = $2",
                tenant_id,
                kg_name,
            )


# A single per-process instance shared by BOTH the read path (`list_kgs`) and
# the background writer (`recompute_kg_stats`, which runs outside any request so
# it can't reach `app.state`). For the in-memory backend this sharing is a
# correctness requirement — writer and reader must see the same dict — so unlike
# the request-scoped job store this store is a module singleton.
_store: Optional[KgStatsStore] = None


def get_kg_stats_store() -> KgStatsStore:
    """Return the process-wide KG-stats store.

    :class:`PostgresKgStatsStore` when ``settings.database_url`` is set (durable,
    shared across ECS tasks), else :class:`InMemoryKgStatsStore` (zero-config
    default). The Postgres store creates its pool/table lazily, so calling this
    never touches the network.
    """
    global _store
    if _store is None:
        _store = (
            PostgresKgStatsStore() if settings.database_url else InMemoryKgStatsStore()
        )
    return _store


def reset_kg_stats_store() -> None:
    """Test helper — clear the singleton."""
    global _store
    _store = None
