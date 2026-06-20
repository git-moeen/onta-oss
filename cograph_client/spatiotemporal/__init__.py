"""Spatio-temporal secondary index for entity URIs (COG-103).

Neptune has no spatial/temporal index; this subsystem adds a swappable secondary
index that answers geo + time queries and returns entity URIs plus denormalized
display ``attrs`` (single-hop, no Neptune round-trip).

Public surface:

* models: :class:`SpatioTemporalFact`, :class:`STQueryResult`, ``TimeWindow``
* protocol: :class:`SpatioTemporalIndex`
* backends: :class:`InMemorySpatioTemporalIndex` (default),
  :class:`PostGISSpatioTemporalIndex` (durable, generic Postgres DSN)
* selection: :func:`make_spatiotemporal_index`, :func:`get_spatiotemporal_index`,
  :func:`register_spatiotemporal_index`, :func:`reset_spatiotemporal_index`

``PostGISSpatioTemporalIndex`` is imported lazily (it pulls asyncpg) so importing
this package never requires Postgres.
"""

from __future__ import annotations

from cograph_client.spatiotemporal.memory import InMemorySpatioTemporalIndex
from cograph_client.spatiotemporal.protocol import (
    STQueryResult,
    SpatioTemporalFact,
    SpatioTemporalIndex,
    TimeWindow,
)
from cograph_client.spatiotemporal.registry import (
    get_spatiotemporal_index,
    make_spatiotemporal_index,
    register_spatiotemporal_index,
    reset_spatiotemporal_index,
)

__all__ = [
    "SpatioTemporalFact",
    "STQueryResult",
    "TimeWindow",
    "SpatioTemporalIndex",
    "InMemorySpatioTemporalIndex",
    "PostGISSpatioTemporalIndex",
    "make_spatiotemporal_index",
    "get_spatiotemporal_index",
    "register_spatiotemporal_index",
    "reset_spatiotemporal_index",
]


def __getattr__(name: str):
    # Lazy re-export so `from cograph_client.spatiotemporal import
    # PostGISSpatioTemporalIndex` works without importing asyncpg eagerly.
    if name == "PostGISSpatioTemporalIndex":
        from cograph_client.spatiotemporal.postgis import (
            PostGISSpatioTemporalIndex as _PG,
        )

        return _PG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
