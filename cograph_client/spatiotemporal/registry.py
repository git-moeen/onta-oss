"""Backend selection + plugin registration for the spatio-temporal index (COG-103).

Mirrors the ``register_job_backend`` / ``job_backend`` and ``make_job_store`` pattern:

* :func:`make_spatiotemporal_index` is the **factory** — it selects a backend from
  configuration: a :class:`PostGISSpatioTemporalIndex` when ``settings.database_url``
  is set (durable, shared across tasks), else the zero-config
  :class:`InMemorySpatioTemporalIndex`. The PostGIS store creates its pool/table
  lazily, so calling the factory never touches the network.
* :func:`register_spatiotemporal_index` lets a premium/alternate backend override the
  process-wide instance (same plugin style as ``register_governance_panel`` /
  ``register_adapter``). Pass ``None`` to clear it.
* :func:`get_spatiotemporal_index` returns the registered instance if any, else lazily
  builds one via the factory and caches it.
"""

from __future__ import annotations

from typing import Optional

import structlog

from cograph_client.config import settings
from cograph_client.spatiotemporal.memory import InMemorySpatioTemporalIndex
from cograph_client.spatiotemporal.protocol import SpatioTemporalIndex

logger = structlog.stdlib.get_logger("cograph.spatiotemporal.registry")

_registered: Optional[SpatioTemporalIndex] = None
_default: Optional[SpatioTemporalIndex] = None


def make_spatiotemporal_index() -> SpatioTemporalIndex:
    """Select the spatio-temporal index backend from configuration.

    Returns a :class:`PostGISSpatioTemporalIndex` when ``settings.database_url`` is
    set, else an :class:`InMemorySpatioTemporalIndex`. Never touches the network.
    """
    if settings.database_url:
        # Imported lazily so OSS installs without a DSN never import asyncpg paths.
        from cograph_client.spatiotemporal.postgis import PostGISSpatioTemporalIndex

        return PostGISSpatioTemporalIndex()
    return InMemorySpatioTemporalIndex()


def register_spatiotemporal_index(index: Optional[SpatioTemporalIndex]) -> None:
    """Register (or clear, with ``None``) the process-wide spatio-temporal index.

    A premium/alternate backend calls this at startup; OSS deployments never do and
    fall back to whatever :func:`make_spatiotemporal_index` selects.
    """
    global _registered
    _registered = index
    logger.info(
        "spatiotemporal_index_registered",
        backend=type(index).__name__ if index is not None else None,
    )


def get_spatiotemporal_index() -> SpatioTemporalIndex:
    """The registered index, else a lazily-built (and cached) factory default."""
    global _default
    if _registered is not None:
        return _registered
    if _default is None:
        _default = make_spatiotemporal_index()
    return _default


def reset_spatiotemporal_index() -> None:
    """Test helper — clear both the registered override and the cached default."""
    global _registered, _default
    _registered = None
    _default = None
