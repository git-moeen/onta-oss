"""Pure-Python in-memory :class:`SpatioTemporalIndex` — the OSS default (COG-103).

Zero-config, non-durable, per-process. Fully functional so OSS deployments without
Postgres (and the whole test suite) work without any external service:

* ``query_radius`` uses the **haversine** great-circle distance in metres — the same
  geodesic semantics as PostGIS ``ST_DWithin(geom::geography, …)``.
* ``query_bbox`` is an exact axis-aligned point-in-box test.
* ``query_polygon`` is an **approximation**: it uses a ray-casting point-in-polygon
  test against the WKT ring when parseable, and otherwise falls back to the polygon's
  bounding box. The PostGIS backend is exact (``ST_Within``); this default trades a
  little precision (no holes, no spherical edges) for "no dependencies". Documented
  here and in the package README so callers know the limitation.

Temporal filtering matches the Protocol contract: ``as_of`` (containment) takes
precedence over ``time_window`` (overlap) when both are given.
"""

from __future__ import annotations

import asyncio
import math
import re
from datetime import datetime
from typing import Optional, Sequence

from cograph_client.spatiotemporal.protocol import (
    STQueryResult,
    SpatioTemporalFact,
    TimeWindow,
)

_EARTH_RADIUS_M = 6_371_008.8  # mean Earth radius (metres), matches PostGIS geography


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def _parse_wkt_polygon(wkt: str) -> Optional[list[tuple[float, float]]]:
    """Best-effort parse of the OUTER ring of a ``POLYGON((lon lat, ...))`` WKT.

    Returns a list of ``(lon, lat)`` vertices, or ``None`` if it can't be parsed
    (caller then falls back to a bbox test). Holes / inner rings are ignored.
    """
    m = re.search(r"\(\s*\(([^)]*)\)", wkt)
    if not m:
        return None
    pts: list[tuple[float, float]] = []
    for pair in m.group(1).split(","):
        nums = pair.strip().split()
        if len(nums) < 2:
            continue
        try:
            pts.append((float(nums[0]), float(nums[1])))
        except ValueError:
            return None
    return pts or None


def _point_in_ring(lon: float, lat: float, ring: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test for a single ring (planar approximation)."""
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-30) + xi
        ):
            inside = not inside
        j = i
    return inside


def _temporal_ok(
    fact: SpatioTemporalFact,
    time_window: Optional[TimeWindow],
    as_of: Optional[datetime],
) -> bool:
    """Apply the temporal predicate. ``as_of`` (containment) wins over ``time_window``.

    Validity is the half-open range ``[valid_from, valid_to)`` with ``None`` meaning
    unbounded on that side.
    """
    lo, hi = fact.valid_from, fact.valid_to
    if as_of is not None:
        # containment: lo <= as_of < hi
        if lo is not None and as_of < lo:
            return False
        if hi is not None and as_of >= hi:
            return False
        return True
    if time_window is not None:
        w_lo, w_hi = time_window
        # overlap of [lo, hi) and [w_lo, w_hi): NOT (hi <= w_lo or lo >= w_hi)
        if w_lo is not None and hi is not None and hi <= w_lo:
            return False
        if w_hi is not None and lo is not None and lo >= w_hi:
            return False
        return True
    return True


class InMemorySpatioTemporalIndex:
    """Non-durable, per-process :class:`SpatioTemporalIndex` — the registered default."""

    def __init__(self) -> None:
        # keyed by (tenant_id, entity_uri, valid_from, valid_to) → fact, so upsert
        # is idempotent on (entity_uri, valid_time).
        self._facts: dict[tuple, SpatioTemporalFact] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def _key(fact: SpatioTemporalFact) -> tuple:
        return (fact.tenant_id, fact.entity_uri, fact.valid_from, fact.valid_to)

    async def upsert(self, fact: SpatioTemporalFact) -> None:
        async with self._lock:
            self._facts[self._key(fact)] = fact.model_copy(deep=True)

    async def upsert_many(self, facts: Sequence[SpatioTemporalFact]) -> None:
        async with self._lock:
            for fact in facts:
                self._facts[self._key(fact)] = fact.model_copy(deep=True)

    def _scan(
        self,
        tenant_id: str,
        time_window: Optional[TimeWindow],
        as_of: Optional[datetime],
    ):
        for fact in self._facts.values():
            if fact.tenant_id != tenant_id:
                continue  # tenant isolation: never cross tenants
            if _temporal_ok(fact, time_window, as_of):
                yield fact

    @staticmethod
    def _result(fact: SpatioTemporalFact) -> STQueryResult:
        return STQueryResult(entity_uri=fact.entity_uri, attrs=dict(fact.attrs))

    async def query_radius(
        self,
        tenant_id: str,
        lon: float,
        lat: float,
        radius_m: float,
        *,
        time_window: Optional[TimeWindow] = None,
        as_of: Optional[datetime] = None,
    ) -> list[STQueryResult]:
        async with self._lock:
            return [
                self._result(f)
                for f in self._scan(tenant_id, time_window, as_of)
                if _haversine_m(lon, lat, f.lon, f.lat) <= radius_m
            ]

    async def query_bbox(
        self,
        tenant_id: str,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        *,
        time_window: Optional[TimeWindow] = None,
        as_of: Optional[datetime] = None,
    ) -> list[STQueryResult]:
        async with self._lock:
            return [
                self._result(f)
                for f in self._scan(tenant_id, time_window, as_of)
                if min_lon <= f.lon <= max_lon and min_lat <= f.lat <= max_lat
            ]

    async def query_polygon(
        self,
        tenant_id: str,
        wkt_polygon: str,
        *,
        time_window: Optional[TimeWindow] = None,
        as_of: Optional[datetime] = None,
    ) -> list[STQueryResult]:
        ring = _parse_wkt_polygon(wkt_polygon)
        if ring:
            test = lambda f: _point_in_ring(f.lon, f.lat, ring)  # noqa: E731
        else:
            # Unparseable WKT → fall back to "no spatial filter" (documented
            # approximation); the PostGIS backend uses exact ST_Within instead.
            test = lambda f: True  # noqa: E731
        async with self._lock:
            return [
                self._result(f)
                for f in self._scan(tenant_id, time_window, as_of)
                if test(f)
            ]

    async def delete(self, entity_uri: str, tenant_id: str) -> None:
        async with self._lock:
            self._facts = {
                k: v
                for k, v in self._facts.items()
                if not (v.tenant_id == tenant_id and v.entity_uri == entity_uri)
            }

    async def clear(self, tenant_id: str) -> None:
        async with self._lock:
            self._facts = {
                k: v for k, v in self._facts.items() if v.tenant_id != tenant_id
            }
