"""Job stores for tracked jobs (enrichment / dedupe / reconciliation).

Defines an async ``JobStore`` Protocol so the backend is swappable:

- ``InMemoryJobStore`` — the zero-config default; non-durable, per-process.
- ``PostgresJobStore`` — a durable, shared-across-tasks backend over a generic
  Postgres DSN (``settings.database_url``). It is deliberately vendor-neutral:
  it reads a plain DSN, contains no cloud-provider identifiers, and works
  against any Postgres (local, Aurora, Neon, Supabase, ...).
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional, Protocol

from cograph_client.config import settings
from cograph_client.enrichment.models import EnrichJob, JobSummary, job_to_summary


class JobStore(Protocol):
    async def create(self, job: EnrichJob) -> None: ...
    async def get(self, job_id: str) -> Optional[EnrichJob]: ...
    async def update(self, job: EnrichJob) -> None: ...
    async def list_for_tenant(self, tenant_id: str) -> list[JobSummary]: ...
    async def delete(self, job_id: str) -> None: ...


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, EnrichJob] = {}
        self._lock = asyncio.Lock()

    async def create(self, job: EnrichJob) -> None:
        async with self._lock:
            self._jobs[job.id] = job

    async def get(self, job_id: str) -> Optional[EnrichJob]:
        async with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy(deep=True) if job else None

    async def update(self, job: EnrichJob) -> None:
        async with self._lock:
            self._jobs[job.id] = job.model_copy(deep=True)

    async def list_for_tenant(self, tenant_id: str) -> list[JobSummary]:
        async with self._lock:
            return [
                job_to_summary(j)
                for j in self._jobs.values()
                if j.tenant_id == tenant_id
            ]

    async def delete(self, job_id: str) -> None:
        async with self._lock:
            self._jobs.pop(job_id, None)


class PostgresJobStore:
    """Durable ``JobStore`` backed by a generic Postgres DSN via asyncpg.

    The full ``EnrichJob`` is serialized to a ``payload`` jsonb column; the
    columns that the unified Jobs list filters/sorts on (tenant, category,
    trigger, status, cost, run timestamps) are mirrored alongside it so common
    queries don't have to parse jsonb.

    The connection pool and table are created lazily on first use so importing
    this module (and constructing the store) never touches the network — the
    table DDL is idempotent (``CREATE TABLE IF NOT EXISTS``).

    Vendor-neutral by construction: the only configuration is a plain DSN. No
    cloud-provider ARNs, account IDs, or hostnames live here.
    """

    _TABLE = "cograph_jobs"

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
                        category text,
                        trigger text,
                        status text,
                        cost double precision,
                        last_run timestamptz,
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
            self._pool = pool
            return self._pool

    @staticmethod
    def _columns(job: EnrichJob) -> tuple:
        """Mirror queryable columns from a job (payload stored separately)."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        return (
            job.id,
            job.tenant_id,
            job.category.value,
            job.trigger.value,
            job.status.value,
            job.cost,
            job.last_run,
            job.next_run,
            job.created_at,
            now,
            job.model_dump_json(),
        )

    async def create(self, job: EnrichJob) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._TABLE}
                    (id, tenant_id, category, trigger, status, cost,
                     last_run, next_run, created_at, updated_at, payload)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb)
                ON CONFLICT (id) DO UPDATE SET
                    tenant_id = EXCLUDED.tenant_id,
                    category = EXCLUDED.category,
                    trigger = EXCLUDED.trigger,
                    status = EXCLUDED.status,
                    cost = EXCLUDED.cost,
                    last_run = EXCLUDED.last_run,
                    next_run = EXCLUDED.next_run,
                    updated_at = EXCLUDED.updated_at,
                    payload = EXCLUDED.payload
                """,
                *self._columns(job),
            )

    async def get(self, job_id: str) -> Optional[EnrichJob]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT payload FROM {self._TABLE} WHERE id = $1", job_id
            )
        if row is None:
            return None
        payload = row["payload"]
        # asyncpg returns jsonb as a str unless a codec is registered; accept
        # both str and pre-decoded dict for robustness.
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode()
        if isinstance(payload, str):
            return EnrichJob.model_validate_json(payload)
        return EnrichJob.model_validate(payload)

    async def update(self, job: EnrichJob) -> None:
        # create() is an UPSERT, so update is the same write path.
        await self.create(job)

    async def list_for_tenant(self, tenant_id: str) -> list[JobSummary]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT payload FROM {self._TABLE} WHERE tenant_id = $1 "
                f"ORDER BY created_at DESC",
                tenant_id,
            )
        out: list[JobSummary] = []
        for row in rows:
            payload = row["payload"]
            if isinstance(payload, (bytes, bytearray)):
                payload = payload.decode()
            job = (
                EnrichJob.model_validate_json(payload)
                if isinstance(payload, str)
                else EnrichJob.model_validate(payload)
            )
            out.append(job_to_summary(job))
        return out

    async def delete(self, job_id: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._TABLE} WHERE id = $1", job_id
            )


_store: Optional[InMemoryJobStore] = None


def get_job_store() -> InMemoryJobStore:
    global _store
    if _store is None:
        _store = InMemoryJobStore()
    return _store


def make_job_store() -> JobStore:
    """Select the job-store backend from configuration.

    Returns a :class:`PostgresJobStore` when ``settings.database_url`` is set
    (durable, shared across ECS tasks), else an :class:`InMemoryJobStore`
    (zero-config default). The Postgres store creates its pool/table lazily, so
    calling this never touches the network.
    """
    if settings.database_url:
        return PostgresJobStore()
    return get_job_store()


def reset_job_store() -> None:
    """Test helper — clear the singleton."""
    global _store
    _store = None
