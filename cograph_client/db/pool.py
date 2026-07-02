"""Process-wide shared asyncpg pool — ONE pool per DSN, not one per store.

Six durable stores each grew an identical private ``asyncpg.create_pool``
(scheduling/store, graph/kg_stats_store, agent/conversation_store,
agent/plan_store, enrichment/job_store, spatiotemporal/postgis) — at asyncpg's
default sizing that is up to ~60 connections per ECS task against the same
Postgres for no benefit. This module is the single place a pool is created;
stores keep their own DDL/queries and just acquire connections from here.

ONTA-174 migrates ``spatiotemporal/postgis.py`` (and the upcoming semantic
index backend builds on it directly); the other five stores can migrate
opportunistically — their private pools keep working meanwhile.

Connection-init hooks (:func:`register_pool_init`) let a consumer install
per-connection setup — e.g. the semantic index registers pgvector's
``register_vector`` codec (ONTA-176) — without this module importing optional
dependencies. Hooks run on every NEW connection; registering a hook after a
pool already exists expires that pool's connections so the hook applies to
every connection handed out from then on.

Lazy by construction: importing this module never touches the network, and
``asyncpg`` is imported only on first use so OSS installs without a DSN never
need it (same contract as the stores' original private pools).
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import structlog

logger = structlog.stdlib.get_logger("cograph.db.pool")

#: Per-connection init hooks, applied (in registration order) to every new
#: connection of every pool this module creates.
_init_hooks: list[Callable[[Any], Awaitable[None]]] = []

_pools: dict[str, Any] = {}
_lock: asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    # Created lazily so importing this module never binds an event loop.
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


async def _run_init_hooks(conn: Any) -> None:
    for hook in _init_hooks:
        await hook(conn)


def register_pool_init(hook: Callable[[Any], Awaitable[None]]) -> None:
    """Register a per-connection init hook (e.g. pgvector codec registration).

    Applies to every NEW connection of every pool. If pools already exist,
    their current connections are expired so subsequent checkouts are fresh
    connections that run the hook — a late-registered hook can therefore never
    leave a mixed pool (some connections with the codec, some without).
    """
    _init_hooks.append(hook)
    for pool in _pools.values():
        try:
            pool.expire_connections()
        except Exception:  # noqa: BLE001 — best-effort; new conns still get the hook
            logger.warning("pg_pool_expire_failed", exc_info=True)


async def get_pg_pool(dsn: str) -> Any:
    """The shared pool for ``dsn`` — created lazily once, then reused.

    Concurrent first callers are serialized by a lock so exactly one pool is
    built per DSN (the same guard each store's private ``_ensure_pool`` had).
    """
    pool = _pools.get(dsn)
    if pool is not None:
        return pool
    async with _get_lock():
        pool = _pools.get(dsn)
        if pool is not None:
            return pool
        import asyncpg  # imported lazily so the dependency stays optional

        pool = await asyncpg.create_pool(dsn=dsn, init=_run_init_hooks)
        _pools[dsn] = pool
        logger.info("pg_pool_created", pools=len(_pools))
        return pool


async def close_pg_pools() -> None:
    """Close every shared pool (app shutdown)."""
    global _pools
    pools, _pools = _pools, {}
    for pool in pools.values():
        try:
            await pool.close()
        except Exception:  # noqa: BLE001 — shutdown is best-effort
            logger.warning("pg_pool_close_failed", exc_info=True)


def reset_pg_pools() -> None:
    """Test helper — forget pools and hooks WITHOUT closing (tests own fakes)."""
    global _lock
    _pools.clear()
    _init_hooks.clear()
    _lock = None
