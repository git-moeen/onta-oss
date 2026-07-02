"""Marker-driven extraction of semantic chunks from instance triples (ONTA-175).

The write path hands us the exact RDF triples it just inserted (the same input
``kg_writer._index_spatiotemporal`` receives); this module turns the values of
**marked free-text predicates** into :class:`SemanticChunk` rows for the
semantic instance index. Unlike the spatio-temporal extractor (datatype-driven:
a ``geo:wktLiteral`` *is* the signal), free text has no distinguishing
datatype — ``description`` and ``sku`` are both plain literals — so indexing is
**opt-in via a marker set**: the caller (the ONTA-181 write hook, consulting
its marker map; or the reconciler's backfill scan) says which predicates are
semantic-candidate attributes, and only those are extracted.

Canonicalization (the change-detection contract)
------------------------------------------------

All values of one marked attribute on one entity form ONE document:

1. values are stripped, empties dropped, exact duplicates removed;
2. multi-valued attributes are **sorted** and joined with a blank line —
   BEFORE hashing/chunking, so the doc (and therefore ``content_hash``) is
   deterministic regardless of triple order. This is load-bearing: the write
   hook skips unchanged hashes and the reconciler upserts by hash, so a
   spurious hash change (same values, different order) would re-chunk and
   re-embed a doc that didn't change.

``content_hash`` is the sha256 hex digest of that canonical doc, computed once
per (entity, attr); every chunk row of the doc carries the same hash.

Chunking (no tokenizer dependency)
----------------------------------

Chunks target ~256–512 tokens estimated at ~4 chars/token → 1024–2048 chars,
preferring to break on paragraph, then sentence, then whitespace boundaries
(hard mid-word cut only for pathological unbroken text). Edge cases are
explicit and tested: empty/whitespace docs → 0 chunks; anything short → exactly
1 chunk; identical docs within one entity are deduplicated (first attribute
wins); and a runaway entity is capped at :data:`MAX_CHUNKS_PER_ENTITY` chunks —
logged, never silent.

Object encoding: the write path emits URIs bare (``https://…``) and typed
literals as ``"<lexical>^^<type-uri>"`` — exactly what
:func:`cograph_client.graph.queries._escape_value` consumes. We mirror its
URI-vs-literal decision (URI objects are entity references, never text) and
strip the ``^^`` datatype tail the way ``spatiotemporal/extract.py`` does. The
small helpers are duplicated from there deliberately, so this leaf module stays
importable on its own without reaching into a sibling subsystem.
"""

from __future__ import annotations

import hashlib
import re
from typing import Collection, Iterable, Optional

import structlog

from cograph_client.semantic.protocol import SemanticChunk

logger = structlog.stdlib.get_logger("cograph.semantic.extract")

Triple = tuple[str, str, str]

#: Chunk size bounds in characters, from the ~4 chars/token heuristic:
#: 256 tokens ≈ 1024 chars (the earliest a natural break is accepted) and
#: 512 tokens ≈ 2048 chars (the hard ceiling). No tokenizer dependency —
#: the pgvector backend and the embed model tolerate the estimation slack.
MIN_CHUNK_CHARS = 1024
MAX_CHUNK_CHARS = 2048

#: Hard cap on chunks per ENTITY (across all its marked attributes). A single
#: pathological entity (a 1 MB scraped page in ``notes``) must not swamp the
#: index or the embed-fill sweep. Overflow is truncated and logged — never
#: silent — so the cap is observable in ops before anyone wonders why an
#: entity's tail text doesn't match.
MAX_CHUNKS_PER_ENTITY = 200

#: Deterministic separator between the sorted values of a multi-valued
#: attribute. A blank line, so the chunker's paragraph-preference naturally
#: avoids splitting mid-value.
VALUE_SEPARATOR = "\n\n"

# Standard RDF predicates used only to denormalize small display fields onto
# the chunk (so a hit renders with no Neptune round-trip). Never used to decide
# whether to index — that is the marker set's job. Mirrors spatiotemporal/extract.
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
_LABEL_LOCALS = {"label", "name", "title"}

_SENTENCE_END_RE = re.compile(r"[.!?][)\"”']*(?:\s|$)")


def _local_name(uri: str, *, lower: bool = True) -> str:
    """Last path/fragment segment of a URI (``…/types/Event/description`` →
    ``description``). Lower-cased by default for case-insensitive predicate
    matching; returns the input unchanged when it is not a URI. (Duplicated
    from ``spatiotemporal/extract.py`` — leaf-module independence.)"""
    if not isinstance(uri, str):
        return ""
    tail = uri.rsplit("#", 1)[-1]
    tail = tail.rsplit("/", 1)[-1]
    return tail.lower() if lower else tail


def _split_typed(obj: str) -> tuple[str, Optional[str]]:
    """Split ``"<lexical>^^<type-uri>"`` into ``(lexical, type_uri)``.

    Only treats the tail as a datatype when it is an ``http`` URI, so a plain
    string literal that happens to contain ``^^`` is left intact (type ``None``).
    (Duplicated from ``spatiotemporal/extract.py`` — leaf-module independence.)"""
    if isinstance(obj, str) and "^^" in obj:
        lexical, type_uri = obj.rsplit("^^", 1)
        if type_uri.startswith("http"):
            return lexical, type_uri
    return obj, None


def _is_uri_object(obj: str) -> bool:
    """True when the write path would emit ``obj`` as a URI, not a literal.

    Mirrors :func:`cograph_client.graph.queries._escape_value`'s decision
    exactly: a bare ``http(s)://…`` or an already-wrapped ``<…>`` is an entity
    reference / URI — never free text, even under a marked predicate (a marked
    local name can collide with a relation predicate on another type)."""
    if not isinstance(obj, str):
        return True
    return obj.startswith(("http://", "https://")) or (
        obj.startswith("<") and obj.endswith(">")
    )


def canonicalize_values(values: Iterable[str]) -> str:
    """Canonicalize an attribute's values into ONE deterministic document.

    Strip each value, drop empty/whitespace-only ones, remove exact
    duplicates, **sort**, and join with :data:`VALUE_SEPARATOR`. Sorting before
    hashing/chunking is the whole point: triple order is not stable across
    writers/replays, and ``content_hash`` must only change when content does.
    Returns ``""`` for no (usable) values.
    """
    cleaned = sorted({v.strip() for v in values if isinstance(v, str) and v.strip()})
    return VALUE_SEPARATOR.join(cleaned)


def content_hash(canonical_doc: str) -> str:
    """sha256 hex digest of a canonicalized (entity, attr) document — the
    change-detection currency shared by the write hook, the reconciler, and
    the embed-fill sweep (see the protocol module docstring)."""
    return hashlib.sha256(canonical_doc.encode("utf-8")).hexdigest()


def _best_break(window: str, min_break: int) -> int:
    """Best split position in ``window`` (which is exactly the max chunk size).

    Preference order, each accepted only at/after ``min_break`` so chunks never
    degenerate into slivers: paragraph boundary (blank line) → sentence end →
    any newline → any space → hard cut at the window end (pathological
    unbroken text — the only case a word may be split).
    """
    pos = window.rfind("\n\n", min_break)
    if pos != -1:
        return pos
    last_sentence = -1
    for m in _SENTENCE_END_RE.finditer(window, min_break):
        last_sentence = m.end()
    if last_sentence != -1:
        return last_sentence
    pos = window.rfind("\n", min_break)
    if pos != -1:
        return pos
    pos = window.rfind(" ", min_break)
    if pos != -1:
        return pos
    return len(window)


def chunk_text(
    text: str,
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    min_break: int = MIN_CHUNK_CHARS,
) -> list[str]:
    """Split ``text`` into chunks of at most ``max_chars`` characters.

    * empty / whitespace-only → ``[]`` (0 chunks);
    * anything up to ``max_chars`` → exactly 1 chunk;
    * longer text is split greedily at the best natural boundary in each
      ``max_chars`` window (see :func:`_best_break`), so chunks land in the
      ``[min_break, max_chars]`` range ≈ 256–512 estimated tokens.

    Chunk edges are whitespace-stripped (the boundary whitespace carries no
    content and would only perturb embeddings).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    rest = text
    while rest:
        if len(rest) <= max_chars:
            chunks.append(rest)
            break
        cut = _best_break(rest[:max_chars], min_break)
        piece = rest[:cut].rstrip()
        if piece:
            chunks.append(piece)
        rest = rest[cut:].lstrip()
    return chunks


class _EntityAccumulator:
    """Per-subject scratch state collected in a single pass over the triples."""

    __slots__ = ("values", "label", "type_name")

    def __init__(self) -> None:
        # attr local name -> raw values, both in first-seen order (output
        # ordering only — canonicalize_values sorts before hashing).
        self.values: dict[str, list[str]] = {}
        self.label: Optional[str] = None
        self.type_name: Optional[str] = None


def extract_semantic_chunks(
    triples: list[Triple],
    *,
    tenant_id: str,
    kg_name: str,
    marked_predicates: Collection[str],
) -> list[SemanticChunk]:
    """Build :class:`SemanticChunk` rows for every marked free-text attribute
    among ``triples``. Pure and side-effect free (the cap warning is a log,
    not a mutation); deterministic for a given input set.

    ``marked_predicates`` is the caller's marker set/map (a ``dict``'s keys
    work — membership is all we test): an entry may be a full predicate URI
    (exact match) or a bare attribute local name, matched case-insensitively
    against the predicate's local name. NOTE the deliberate conflation: a
    local-name entry (``"description"``) marks that attribute on EVERY type —
    the marker map's granularity (ONTA-181) is the attribute name, matching
    how the ontology names attributes tenant-wide.

    Per (entity, attr): values are canonicalized (:func:`canonicalize_values`),
    hashed (:func:`content_hash`), chunked (:func:`chunk_text`). Edge cases:

    * empty/whitespace-only doc → 0 chunks (the ONTA-181 hook turns "had
      chunks before, has 0 now" into a ``delete(..., attr=…)``);
    * identical canonical docs within one entity (e.g. ``summary`` mirroring
      ``description``) are emitted ONCE — first attribute in triple order wins
      (intra-entity dedup: double-indexing the same text would double-count it
      in every ranking);
    * at most :data:`MAX_CHUNKS_PER_ENTITY` chunks per entity across all its
      attributes — overflow truncated + logged (never silent).
    """
    # Normalize the marker set once: exact entries as given, plus each entry's
    # lowered local name (a non-URI entry's local name is itself).
    marked_exact = {m for m in marked_predicates if isinstance(m, str)}
    marked_locals = {_local_name(m) for m in marked_exact}

    acc: dict[str, _EntityAccumulator] = {}
    order: list[str] = []

    for s, p, o in triples:
        if not isinstance(s, str) or not isinstance(p, str):
            continue
        ent = acc.get(s)
        if ent is None:
            ent = acc[s] = _EntityAccumulator()
            order.append(s)

        # rdf:type -> denormalized type display name (no effect on indexing).
        if p == _RDF_TYPE:
            if ent.type_name is None:
                ent.type_name = _local_name(o, lower=False)
            continue

        if not isinstance(o, str):
            continue
        lexical, type_uri = _split_typed(o)

        # Label / name for denormalized display (plain literals only).
        if ent.label is None and (p == _RDFS_LABEL or _local_name(p) in _LABEL_LOCALS):
            if lexical and type_uri is None and not _is_uri_object(o):
                ent.label = lexical
            # NOT `continue`: a label predicate may itself be marked (e.g.
            # `title` on an Article) — it still contributes text below.

        if p not in marked_exact and _local_name(p) not in marked_locals:
            continue
        if _is_uri_object(o):
            continue  # entity reference, not free text — never index a URI
        # Any datatype's lexical form is accepted: marking is the gate, and a
        # marked predicate is free text by declaration.
        ent.values.setdefault(_local_name(p), []).append(lexical)

    chunks: list[SemanticChunk] = []
    for uri in order:
        ent = acc[uri]
        if not ent.values:
            continue
        display: dict[str, str] = {}
        if ent.label:
            display["label"] = ent.label
        if ent.type_name:
            display["type"] = ent.type_name

        seen_hashes: set[str] = set()  # intra-entity doc dedup
        entity_chunk_count = 0
        for attr, values in ent.values.items():
            doc = canonicalize_values(values)
            if not doc:
                continue  # empty/whitespace-only -> 0 chunks
            doc_hash = content_hash(doc)
            if doc_hash in seen_hashes:
                logger.debug(
                    "semantic_extract_duplicate_doc",
                    entity_uri=uri,
                    attr=attr,
                    content_hash=doc_hash,
                )
                continue
            seen_hashes.add(doc_hash)

            pieces = chunk_text(doc)
            budget = MAX_CHUNKS_PER_ENTITY - entity_chunk_count
            if len(pieces) > budget:
                logger.warning(
                    "semantic_extract_chunk_cap",
                    entity_uri=uri,
                    attr=attr,
                    cap=MAX_CHUNKS_PER_ENTITY,
                    produced=len(pieces),
                    dropped=len(pieces) - budget,
                )
                pieces = pieces[:budget]
            for ix, piece in enumerate(pieces):
                chunks.append(
                    SemanticChunk(
                        tenant_id=tenant_id,
                        kg_name=kg_name,
                        entity_uri=uri,
                        attr=attr,
                        chunk_ix=ix,
                        chunk_text=piece,
                        content_hash=doc_hash,
                        attrs=dict(display),
                    )
                )
            entity_chunk_count += len(pieces)
    return chunks
