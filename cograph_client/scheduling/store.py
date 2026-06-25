"""Schedule stores for recurring action schedules (COG-135).

Defines an async ``ScheduleStore`` Protocol so the backend is swappable, exactly
mirroring the enrichment ``JobStore`` pattern:

- ``InMemoryScheduleStore`` — the zero-config default; non-durable, per-process.
- ``PostgresScheduleStore`` — a durable, shared-across-tasks backend over a
  generic Postgres DSN (``settings.database_url``). Vendor-neutral by
  construction: it reads a plain DSN, contains no cloud-provider identifiers,
  and works against any Postgres (local, Aurora, Neon, Supabase, ...).

This is the DATA SEAM only (storage + CRUD + due-query). The firing/scheduler
loop that turns due schedules into jobs is a separate task.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from cograph_client.config import settings
from cograph_client.scheduling.models import Schedule

_OLDEST = datetime.min.replace(tzinfo=timezone.utc)


class ScheduleStore(Protocol):
    async def create(self, schedule: Schedule) -> None: ...
    async def get(self, schedule_id: str) -> Optional[Schedule]: ...
    async def update(self, schedule: Schedule) -> None: ...
    async def delete(self, schedule_id: str) -> None: ...
    async def list_for_tenant(self, tenant_id: str) -> list[Schedule]: ...
    async def due_before(self, now: datetime) -> list[Schedule]: ...


class InMemoryScheduleStore:
    def __init__(self) -> None:
        self._schedules: dict[str, Schedule] = {}
        self._lock = asyncio.Lock()

    async def create(self, schedule: Schedule) -> None:
        async with self._lock:
            self._schedules[schedule.id] = schedule.model_copy(deep=True)

    async def get(self, schedule_id: str) -> Optional[Schedule]:
        async with self._lock:
            schedule = self._schedules.get(schedule_id)
            return schedule.model_copy(deep=True) if schedule else None

    async def update(self, schedule: Schedule) -> None:
        async with self._lock:
            self._schedules[schedule.id] = schedule.model_copy(deep=True)

    async def delete(self, schedule_id: str) -> None:
        async with self._lock:
            self._schedules.pop(schedule_id, None)

    async def list_for_tenant(self, tenant_id: str) -> list[Schedule]:
        async with self._lock:
            schedules = [
                s for s in self._schedules.values() if s.tenant_id == tenant_id
            ]
        # Oldest first (created_at ASC) to match PostgresScheduleStore's
        # ORDER BY created_at. created_at is required; guard a None defensively
        # (treat as oldest so it sorts first) rather than raising.
        schedules.sort(key=lambda s: s.created_at or _OLDEST)
        return [s.model_copy(deep=True) for s in schedules]

    async def due_before(self, now: datetime) -> list[Schedule]:
        async with self._lock:
            due = [
                s
                for s in self._schedules.values()
                if s.enabled and s.next_run is not None and s.next_run <= now
            ]
        due.sort(key=lambda s: s.next_run or _OLDEST)
        return [s.model_copy(deep=True) for s in due]


class PostgresScheduleStore:
    """Durable ``ScheduleStore`` backed by a generic Postgres DSN via asyncpg.

    The full ``Schedule`` is serialized to a ``payload`` jsonb column; the
    columns the firing loop / listing filter on (tenant, enabled, next_run) are
    mirrored alongside it so the due-query doesn't have to parse jsonb.

    The connection pool and table are created lazily on first use so importing
    this module (and constructing the store) never touches the network — the
    table DDL is idempotent (``CREATE TABLE IF NOT EXISTS``).

    Vendor-neutral by construction: the only configuration is a plain DSN. No
    cloud-provider ARNs, account IDs, or hostnames live here.
    """

    _TABLE = "cograph_schedules"

    def __init__(self, dsn: Optional[str] = None) -> None:
        self._dsn = dsn if dsn is not None else settings.database_url
        self._pool: Any = None
        self._lock = asyncio.Lock()

    async def _ensure_pool(self) -> Any:
        """Lazily create the asyncpg pool + table on first use."""
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
                        id text PRIMARY KEY,
                        tenant_id text NOT NULL,
                        enabled boolean,
                        next_run timestamptz,
                        created_at timestamptz,
                        updated_at timestamptz,
                        payload jsonb NOT NULL
                    )
                    """
                )
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._TABLE}_tenant_idx "
                    f"ON {self._TABLE} (tenant_id)"
                )
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._TABLE}_due_idx "
                    f"ON {self._TABLE} (enabled, next_run)"
                )
            self._pool = pool
            return self._pool

    @staticmethod
    def _columns(schedule: Schedule) -> tuple:
        """Mirror queryable columns from a schedule (payload stored separately)."""
        now = datetime.now(timezone.utc)
        return (
            schedule.id,
            schedule.tenant_id,
            schedule.enabled,
            schedule.next_run,
            schedule.created_at,
            now,
            schedule.model_dump_json(),
        )

    @staticmethod
    def _from_payload(payload: Any) -> Schedule:
        # asyncpg returns jsonb as a str unless a codec is registered; accept
        # str, bytes, and pre-decoded dict for robustness.
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode()
        if isinstance(payload, str):
            return Schedule.model_validate_json(payload)
        return Schedule.model_validate(payload)

    async def create(self, schedule: Schedule) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._TABLE}
                    (id, tenant_id, enabled, next_run, created_at, updated_at,
                     payload)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    tenant_id = EXCLUDED.tenant_id,
                    enabled = EXCLUDED.enabled,
                    next_run = EXCLUDED.next_run,
                    updated_at = EXCLUDED.updated_at,
                    payload = EXCLUDED.payload
                """,
                *self._columns(schedule),
            )

    async def get(self, schedule_id: str) -> Optional[Schedule]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT payload FROM {self._TABLE} WHERE id = $1", schedule_id
            )
        if row is None:
            return None
        return self._from_payload(row["payload"])

    async def update(self, schedule: Schedule) -> None:
        # create() is an UPSERT, so update is the same write path.
        await self.create(schedule)

    async def delete(self, schedule_id: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._TABLE} WHERE id = $1", schedule_id
            )

    async def list_for_tenant(self, tenant_id: str) -> list[Schedule]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT payload FROM {self._TABLE} WHERE tenant_id = $1 "
                f"ORDER BY created_at",
                tenant_id,
            )
        return [self._from_payload(row["payload"]) for row in rows]

    async def due_before(self, now: datetime) -> list[Schedule]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT payload FROM {self._TABLE} "
                f"WHERE enabled = true AND next_run IS NOT NULL "
                f"AND next_run <= $1 ORDER BY next_run",
                now,
            )
        return [self._from_payload(row["payload"]) for row in rows]


_store: Optional[InMemoryScheduleStore] = None


def get_schedule_store() -> InMemoryScheduleStore:
    global _store
    if _store is None:
        _store = InMemoryScheduleStore()
    return _store


def make_schedule_store() -> ScheduleStore:
    """Select the schedule-store backend from configuration.

    Returns a :class:`PostgresScheduleStore` when ``settings.database_url`` is
    set (durable, shared across tasks), else an :class:`InMemoryScheduleStore`
    (zero-config default). The Postgres store creates its pool/table lazily, so
    calling this never touches the network.
    """
    if settings.database_url:
        return PostgresScheduleStore()
    return get_schedule_store()


def reset_schedule_store() -> None:
    """Test helper — clear the singleton."""
    global _store
    _store = None
