"""In-memory job store for enrichment jobs.

Defines an async Protocol so we can swap in a persistent backend later.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Protocol

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


_store: Optional[InMemoryJobStore] = None


def get_job_store() -> InMemoryJobStore:
    global _store
    if _store is None:
        _store = InMemoryJobStore()
    return _store


def reset_job_store() -> None:
    """Test helper — clear the singleton."""
    global _store
    _store = None
