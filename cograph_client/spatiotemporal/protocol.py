"""Spatio-temporal secondary index — models + swappable backend Protocol (COG-103).

Neptune (our RDF source of truth) has **no spatial or temporal index**: answering
"which entities are within 5 km of this point, valid during last March?" against
Neptune means scanning and post-filtering, which does not scale. This module adds a
**secondary index keyed by entity URI** that answers geo + time queries directly and
returns entity URIs (+ denormalized display attrs) so the hot read path is a single
hop with **no Neptune round-trip**.

Design (mirrors the existing swappable-backend pattern — ``JobStore`` /
``JobBackend`` / ``VerdictStore`` + ``register_*`` / default):

* :class:`SpatioTemporalIndex` is a ``Protocol`` so the backend is pluggable.
* :class:`InMemorySpatioTemporalIndex` (``memory.py``) is the zero-config OSS
  default — pure Python, fully functional for tests and Postgres-less OSS use.
* :class:`PostGISSpatioTemporalIndex` (``postgis.py``) is the durable adapter over a
  **generic** Postgres DSN (``settings.database_url``). It is vendor-neutral by
  construction: a plain DSN, no cloud-provider ARNs / account IDs / hostnames. The
  Aurora/Neon connection + infra that *provides* that DSN stay proprietary.

Consistency model: Neptune stays the source of truth; this index is **eventually
consistent**. Writes are idempotent upserts keyed on
``(tenant_id, kg_name, entity_uri, valid_time)`` so replaying the same fact is a
no-op. The full ingest/enrichment outbox reconciler
that keeps the index in lock-step with Neptune is **out of scope** here — see the
TODO seam at the bottom of this file.

**MobilityDB caveat (important):** this PostGIS schema models **discrete
"located-during-range" facts** — an entity is at a *fixed* point/geometry for a
``valid_time`` range. It is NOT suitable for *continuously moving objects /
trajectories* (a vehicle whose position is a function of time). For genuine moving
objects use `MobilityDB <https://mobilitydb.com/>`_ (temporal geometry / ``tgeompoint``)
instead; this index would otherwise explode into one row per sampled position.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, Field

#: A half-open time window ``[start, end)`` as a (start, end) tuple. Either bound may
#: be ``None`` for an unbounded side (``(None, end)`` = "up to end",
#: ``(start, None)`` = "from start on"). Maps to a PostGIS ``tstzrange`` overlap.
TimeWindow = tuple[Optional[datetime], Optional[datetime]]


class SpatioTemporalFact(BaseModel):
    """A single "entity located at a point during a time range" fact.

    The geometry is a WGS84 point (``lon``/``lat``, SRID 4326). ``valid_from`` /
    ``valid_to`` bound the half-open validity range ``[valid_from, valid_to)``;
    either may be ``None`` for an open-ended side (e.g. "valid from 2020 onward").

    ``kg_name`` scopes the fact to ONE knowledge graph within the tenant. Instance
    data lives in per-KG named graphs (``…/kg/<kg_name>``), so the index carries
    the same dimension: dropping a single KG must clear only its facts, never a
    sibling KG's, and a query may be narrowed to one KG. ``(tenant_id, kg_name)``
    is the isolation boundary; ``tenant_id`` alone is never enough.

    ``attrs`` holds **denormalized display fields** (label, type, thumbnail, …) so
    the hot read path returns everything the UI needs without a Neptune round-trip.
    Keep it small — it is stored verbatim as ``jsonb`` and shipped on every hit.
    """

    entity_uri: str
    tenant_id: str
    kg_name: str
    lon: float = Field(..., ge=-180.0, le=180.0)
    lat: float = Field(..., ge=-90.0, le=90.0)
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    attrs: dict[str, Any] = Field(default_factory=dict)


class STQueryResult(BaseModel):
    """A single spatio-temporal query hit: the entity URI + denormalized attrs.

    Deliberately does NOT carry geometry/time back — the index is a *finder*; batch
    hydration of full entities from Neptune (if needed) is the caller's concern.
    """

    entity_uri: str
    attrs: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class SpatioTemporalIndex(Protocol):
    """Swappable spatio-temporal secondary index over entity URIs.

    All methods are ``async``. Queries are always scoped to a single ``tenant_id``
    (tenant isolation is mandatory — a query MUST NEVER cross tenants) and may be
    further narrowed to one ``kg_name`` (``None`` = every KG in the tenant). Spatial
    predicates select candidate geometries; the optional temporal predicate further
    restricts by validity range:

    * ``time_window=(start, end)`` keeps facts whose ``valid_time`` **overlaps** the
      window (PostGIS ``valid_time && tstzrange(start, end)``).
    * ``as_of=<ts>`` keeps facts whose ``valid_time`` **contains** that instant
      (PostGIS ``as_of <@ valid_time``).
    * **Precedence:** if both are passed, ``as_of`` wins and ``time_window`` is
      ignored (a single instant is the more specific predicate). Implementations
      must honor this so backends are interchangeable.
    * Neither → no temporal filtering (purely spatial).

    Premium/alternate backends implement these and register via
    :func:`cograph_client.spatiotemporal.registry.register_spatiotemporal_index`.
    """

    async def upsert(self, fact: SpatioTemporalFact) -> None:
        """Idempotently insert/replace one fact, keyed on ``(entity_uri, valid_time)``."""
        ...

    async def upsert_many(self, facts: Sequence[SpatioTemporalFact]) -> None:
        """Idempotently upsert many facts (batched where the backend supports it)."""
        ...

    async def query_radius(
        self,
        tenant_id: str,
        lon: float,
        lat: float,
        radius_m: float,
        *,
        kg_name: Optional[str] = None,
        time_window: Optional[TimeWindow] = None,
        as_of: Optional[datetime] = None,
    ) -> list[STQueryResult]:
        """Entities within ``radius_m`` **metres** of ``(lon, lat)`` (geodesic).

        ``kg_name`` narrows to one KG; ``None`` searches every KG in the tenant."""
        ...

    async def query_bbox(
        self,
        tenant_id: str,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        *,
        kg_name: Optional[str] = None,
        time_window: Optional[TimeWindow] = None,
        as_of: Optional[datetime] = None,
    ) -> list[STQueryResult]:
        """Entities inside the axis-aligned lon/lat bounding box."""
        ...

    async def query_polygon(
        self,
        tenant_id: str,
        wkt_polygon: str,
        *,
        kg_name: Optional[str] = None,
        time_window: Optional[TimeWindow] = None,
        as_of: Optional[datetime] = None,
    ) -> list[STQueryResult]:
        """Entities inside the WGS84 polygon given as WKT (``POLYGON((...))``)."""
        ...

    async def delete(
        self, entity_uri: str, tenant_id: str, *, kg_name: Optional[str] = None
    ) -> None:
        """Remove all facts for one entity. ``kg_name`` narrows to a single KG;
        ``None`` removes the entity's facts across every KG in the tenant."""
        ...

    async def clear(self, tenant_id: str, *, kg_name: Optional[str] = None) -> None:
        """Remove facts for a tenant. ``kg_name`` clears just that KG (the KG-delete
        path); ``None`` clears the whole tenant (e.g. tenant teardown)."""
        ...


# ---------------------------------------------------------------------------
# TODO (out of scope for COG-103) — documented seams for follow-up work:
#
# * Outbox reconciler: a worker that tails the ingest/enrichment outbox and keeps
#   this index eventually consistent with Neptune (the source of truth). Until that
#   lands, callers must invoke upsert/upsert_many/delete at write time.
# * H3 column: add an ``h3`` text/bigint column + index to the PostGIS schema for
#   cheap cell-based aggregation / coarse pre-filtering at very large scale.
# * pg_partman time-partitioning of entity_spatiotemporal by valid_time for
#   retention + query pruning once row counts make a single table impractical.
# * MobilityDB backend: a separate adapter (tgeompoint) for continuously moving
#   objects / trajectories — see the module docstring caveat.
# ---------------------------------------------------------------------------
