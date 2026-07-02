"""Backend selection + plugin registration for the semantic index (ONTA-175).

Mirrors ``spatiotemporal/registry.py`` (itself the ``register_job_backend`` /
``make_job_store`` pattern):

* :func:`make_semantic_index` is the **factory** — it selects a backend from
  configuration: the durable pgvector-backed
  :class:`~cograph_client.semantic.postgres.PostgresSemanticIndex` when
  ``settings.database_url`` is set (ONTA-176), else the zero-config
  :class:`InMemorySemanticIndex`.
* :func:`register_semantic_index` lets a premium/alternate backend override the
  process-wide instance (same plugin style as ``register_governance_panel`` /
  ``register_adapter``). Pass ``None`` to clear it.
* :func:`get_semantic_index` returns the registered instance if any, else
  lazily builds one via the factory and caches it.
"""

from __future__ import annotations

from typing import Optional

import structlog

from cograph_client.config import settings
from cograph_client.semantic.memory import InMemorySemanticIndex
from cograph_client.semantic.protocol import SemanticIndex

logger = structlog.stdlib.get_logger("cograph.semantic.registry")

_registered: Optional[SemanticIndex] = None
_default: Optional[SemanticIndex] = None


def make_semantic_index() -> SemanticIndex:
    """Select the semantic index backend from configuration.

    Returns a :class:`~cograph_client.semantic.postgres.PostgresSemanticIndex`
    when ``settings.database_url`` is set (durable, shared across tasks —
    pool/DDL created lazily on first use), else the zero-config
    :class:`InMemorySemanticIndex`. Never touches the network.
    """
    if settings.database_url:
        # Imported lazily so OSS installs without a DSN never import
        # asyncpg/pgvector paths (mirrors make_spatiotemporal_index).
        from cograph_client.semantic.postgres import PostgresSemanticIndex

        return PostgresSemanticIndex()
    return InMemorySemanticIndex()


def register_semantic_index(index: Optional[SemanticIndex]) -> None:
    """Register (or clear, with ``None``) the process-wide semantic index.

    A premium/alternate backend calls this at startup; OSS deployments never do
    and fall back to whatever :func:`make_semantic_index` selects.
    """
    global _registered
    _registered = index
    logger.info(
        "semantic_index_registered",
        backend=type(index).__name__ if index is not None else None,
    )


def get_semantic_index() -> SemanticIndex:
    """The registered index, else a lazily-built (and cached) factory default."""
    global _default
    if _registered is not None:
        return _registered
    if _default is None:
        _default = make_semantic_index()
    return _default


def reset_semantic_index() -> None:
    """Test helper — clear both the registered override and the cached default."""
    global _registered, _default
    _registered = None
    _default = None
