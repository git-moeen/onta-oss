"""Semantic instance index over marked free-text attributes (ONTA-175).

Neptune has no full-text or vector index; this subsystem adds a swappable
secondary index that answers hybrid lexical + vector queries over the values of
marked predicates (description, bio, notes, …) and returns entity URIs plus
denormalized display ``attrs`` and a matching snippet (single-hop, no Neptune
round-trip). See ``protocol.py`` for the full consistency model (write hook +
claim-based reconciler, ONTA-173/ONTA-181).

Public surface:

* models: :class:`SemanticChunk`, :class:`SemanticHit`,
  :class:`SemanticSearchResult`, ``ChunkKey``
* protocol: :class:`SemanticIndex`
* backend: :class:`InMemorySemanticIndex` (zero-config default; the durable
  pgvector adapter is ONTA-176)
* extraction: :func:`extract_semantic_chunks`, :func:`chunk_text`,
  :func:`canonicalize_values`, :func:`content_hash`,
  ``MAX_CHUNKS_PER_ENTITY``
* selection: :func:`make_semantic_index`, :func:`get_semantic_index`,
  :func:`register_semantic_index`, :func:`reset_semantic_index`
* maintenance (ONTA-181): :func:`semantic_index_enabled` (the master env gate),
  :func:`run_embed_fill_sweep`, :func:`reconcile_kg` — the claim-based
  reconciler duties (full design record in ``reconciler.py``)
"""

from __future__ import annotations

from cograph_client.semantic.extract import (
    MAX_CHUNKS_PER_ENTITY,
    canonicalize_values,
    chunk_text,
    content_hash,
    extract_semantic_chunks,
)
from cograph_client.semantic.memory import InMemorySemanticIndex
from cograph_client.semantic.reconciler import (
    reconcile_kg,
    run_embed_fill_sweep,
    semantic_index_enabled,
)
from cograph_client.semantic.protocol import (
    ChunkKey,
    SemanticChunk,
    SemanticHit,
    SemanticIndex,
    SemanticSearchResult,
)
from cograph_client.semantic.registry import (
    get_semantic_index,
    make_semantic_index,
    register_semantic_index,
    reset_semantic_index,
)

__all__ = [
    "ChunkKey",
    "SemanticChunk",
    "SemanticHit",
    "SemanticSearchResult",
    "SemanticIndex",
    "InMemorySemanticIndex",
    "MAX_CHUNKS_PER_ENTITY",
    "canonicalize_values",
    "chunk_text",
    "content_hash",
    "extract_semantic_chunks",
    "make_semantic_index",
    "get_semantic_index",
    "register_semantic_index",
    "reset_semantic_index",
    "semantic_index_enabled",
    "run_embed_fill_sweep",
    "reconcile_kg",
]
