"""Plan stores for the unified Ask-AI agent (COG-124).

A user asks the agent to do something → the planner proposes a plan (a list of
:class:`~cograph_client.agent.registry.PlanStep`) and persists it keyed by
``plan_id``; the user then confirms and the planner executes it. For that
confirm→execute to survive a process restart — or, on multi-task deployments
(e.g. ECS Fargate), to hit a different task than the one that planned — the
proposed plan must live in a durable, shared store, not in process memory.

This mirrors :mod:`cograph_client.enrichment.job_store` exactly:

- ``PlanStore`` — an async Protocol so the backend is swappable.
- ``InMemoryPlanStore`` — the zero-config default; non-durable, per-process.
- ``PostgresPlanStore`` — a durable, shared-across-tasks backend over a generic
  Postgres DSN (``settings.database_url``). Deliberately vendor-neutral: it reads
  a plain DSN, contains no cloud-provider identifiers, and works against any
  Postgres (local, Aurora, Neon, Supabase, ...).
- ``make_plan_store()`` — selects Postgres when ``settings.database_url`` is set,
  else in-memory.

The full plan is serialized to a ``payload`` jsonb column; the columns the agent
scopes/expires on (tenant, session, created_at) are mirrored alongside it.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol

from cograph_client.agent.registry import PlanStep
from cograph_client.config import settings


@dataclass
class StoredPlan:
    """A persisted, tenant-scoped plan awaiting confirmation/execution.

    Carries the tenant + KG scope it was proposed in, the originating
    ``session_id`` (when the caller supplied one), and a ``created_at`` so a
    durable store can scope listing and expire stale plans.
    """

    plan_id: str
    tenant_id: str
    kg_name: str
    type_name: str | None
    message: str
    steps: list[PlanStep]
    status: str = "proposed"  # proposed | executing | done | failed
    session_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_json(self) -> str:
        """Serialize to a JSON string for the jsonb payload column.

        ``PlanStep`` is a dataclass (not pydantic), so we round-trip it through
        its own ``to_dict``/``from_dict`` rather than ``model_dump_json``.
        """
        return json.dumps(
            {
                "plan_id": self.plan_id,
                "tenant_id": self.tenant_id,
                "kg_name": self.kg_name,
                "type_name": self.type_name,
                "message": self.message,
                "steps": [s.to_dict() for s in self.steps],
                "status": self.status,
                "session_id": self.session_id,
                "created_at": self.created_at.isoformat() if self.created_at else None,
            }
        )

    @classmethod
    def from_payload(cls, payload: Any) -> "StoredPlan":
        """Rebuild a :class:`StoredPlan` from a jsonb payload (str or dict)."""
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode()
        if isinstance(payload, str):
            data = json.loads(payload)
        else:
            data = payload
        created_raw = data.get("created_at")
        created = (
            datetime.fromisoformat(created_raw)
            if created_raw
            else datetime.now(timezone.utc)
        )
        return cls(
            plan_id=data["plan_id"],
            tenant_id=data["tenant_id"],
            kg_name=data.get("kg_name", ""),
            type_name=data.get("type_name"),
            message=data.get("message", ""),
            steps=[PlanStep.from_dict(s) for s in data.get("steps", [])],
            status=data.get("status", "proposed"),
            session_id=data.get("session_id"),
            created_at=created,
        )


class PlanStore(Protocol):
    async def save(self, plan: StoredPlan) -> None: ...
    async def get(self, plan_id: str, tenant_id: str) -> Optional[StoredPlan]: ...
    async def delete(self, plan_id: str, tenant_id: str) -> None: ...
    async def list_for_tenant(self, tenant_id: str) -> list[StoredPlan]: ...
    async def list_for_session(self, session_id: str) -> list[StoredPlan]: ...


class InMemoryPlanStore:
    """Tenant-scoped in-memory plan store — the zero-config default.

    Mirrors :class:`~cograph_client.enrichment.job_store.InMemoryJobStore`:
    an ``asyncio.Lock`` guards the dict and reads/writes deep-copy so a caller
    can't mutate stored state by reference. Plans do not survive a process
    restart; use :class:`PostgresPlanStore` for durability.
    """

    def __init__(self) -> None:
        self._plans: dict[str, StoredPlan] = {}
        self._lock = asyncio.Lock()

    async def save(self, plan: StoredPlan) -> None:
        async with self._lock:
            self._plans[plan.plan_id] = _copy_plan(plan)

    async def get(self, plan_id: str, tenant_id: str) -> Optional[StoredPlan]:
        async with self._lock:
            p = self._plans.get(plan_id)
            if p is None or p.tenant_id != tenant_id:
                return None
            return _copy_plan(p)

    async def delete(self, plan_id: str, tenant_id: str) -> None:
        async with self._lock:
            p = self._plans.get(plan_id)
            if p is not None and p.tenant_id == tenant_id:
                self._plans.pop(plan_id, None)

    async def list_for_tenant(self, tenant_id: str) -> list[StoredPlan]:
        async with self._lock:
            plans = [p for p in self._plans.values() if p.tenant_id == tenant_id]
        return _sorted_newest_first([_copy_plan(p) for p in plans])

    async def list_for_session(self, session_id: str) -> list[StoredPlan]:
        async with self._lock:
            plans = [p for p in self._plans.values() if p.session_id == session_id]
        return _sorted_newest_first([_copy_plan(p) for p in plans])


class PostgresPlanStore:
    """Durable ``PlanStore`` backed by a generic Postgres DSN via asyncpg.

    The full :class:`StoredPlan` is serialized to a ``payload`` jsonb column; the
    columns the agent scopes/expires on (tenant, session, status, created_at) are
    mirrored alongside it so common queries don't have to parse jsonb.

    The connection pool and table are created lazily on first use so importing
    this module (and constructing the store) never touches the network — the
    table DDL is idempotent (``CREATE TABLE IF NOT EXISTS``).

    Vendor-neutral by construction: the only configuration is a plain DSN. No
    cloud-provider ARNs, account IDs, or hostnames live here.
    """

    _TABLE = "cograph_plans"

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
                        plan_id text PRIMARY KEY,
                        tenant_id text NOT NULL,
                        session_id text,
                        status text,
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
                    f"CREATE INDEX IF NOT EXISTS {self._TABLE}_session_idx "
                    f"ON {self._TABLE} (session_id)"
                )
            self._pool = pool
            return self._pool

    @staticmethod
    def _columns(plan: StoredPlan) -> tuple:
        """Mirror queryable columns from a plan (payload stored separately)."""
        now = datetime.now(timezone.utc)
        return (
            plan.plan_id,
            plan.tenant_id,
            plan.session_id,
            plan.status,
            plan.created_at,
            now,
            plan.to_json(),
        )

    async def save(self, plan: StoredPlan) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                INSERT INTO {self._TABLE}
                    (plan_id, tenant_id, session_id, status,
                     created_at, updated_at, payload)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                ON CONFLICT (plan_id) DO UPDATE SET
                    tenant_id = EXCLUDED.tenant_id,
                    session_id = EXCLUDED.session_id,
                    status = EXCLUDED.status,
                    updated_at = EXCLUDED.updated_at,
                    payload = EXCLUDED.payload
                """,
                *self._columns(plan),
            )

    async def get(self, plan_id: str, tenant_id: str) -> Optional[StoredPlan]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"SELECT payload FROM {self._TABLE} "
                f"WHERE plan_id = $1 AND tenant_id = $2",
                plan_id,
                tenant_id,
            )
        if row is None:
            return None
        return StoredPlan.from_payload(row["payload"])

    async def delete(self, plan_id: str, tenant_id: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._TABLE} WHERE plan_id = $1 AND tenant_id = $2",
                plan_id,
                tenant_id,
            )

    async def list_for_tenant(self, tenant_id: str) -> list[StoredPlan]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT payload FROM {self._TABLE} WHERE tenant_id = $1 "
                f"ORDER BY created_at DESC",
                tenant_id,
            )
        return [StoredPlan.from_payload(r["payload"]) for r in rows]

    async def list_for_session(self, session_id: str) -> list[StoredPlan]:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT payload FROM {self._TABLE} WHERE session_id = $1 "
                f"ORDER BY created_at DESC",
                session_id,
            )
        return [StoredPlan.from_payload(r["payload"]) for r in rows]


def _copy_plan(plan: StoredPlan) -> StoredPlan:
    """Deep-ish copy so in-memory callers can't mutate stored state by ref."""
    return StoredPlan.from_payload(plan.to_json())


def _sorted_newest_first(plans: list[StoredPlan]) -> list[StoredPlan]:
    _OLDEST = datetime.min.replace(tzinfo=timezone.utc)
    plans.sort(key=lambda p: p.created_at or _OLDEST, reverse=True)
    return plans


_store: Optional[InMemoryPlanStore] = None


def get_plan_store() -> InMemoryPlanStore:
    global _store
    if _store is None:
        _store = InMemoryPlanStore()
    return _store


def make_plan_store() -> PlanStore:
    """Select the plan-store backend from configuration.

    Returns a :class:`PostgresPlanStore` when ``settings.database_url`` is set
    (durable, shared across ECS tasks), else an :class:`InMemoryPlanStore`
    (zero-config default). The Postgres store creates its pool/table lazily, so
    calling this never touches the network.
    """
    if settings.database_url:
        return PostgresPlanStore()
    return get_plan_store()


def reset_plan_store() -> None:
    """Test helper — clear the singleton."""
    global _store
    _store = None


__all__ = [
    "InMemoryPlanStore",
    "PlanStore",
    "PostgresPlanStore",
    "StoredPlan",
    "get_plan_store",
    "make_plan_store",
    "reset_plan_store",
]
