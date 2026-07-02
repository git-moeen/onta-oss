"""Marker-driven semantic chunk extraction — canonicalization, hashing,
chunking, and every edge case in the ONTA-175 contract (empty/whitespace → 0
chunks, short → 1, multi-value sort-before-hash, intra-entity dedup, the
per-entity chunk cap logged-never-silent).
"""

from __future__ import annotations

import hashlib

import structlog

from cograph_client.semantic.extract import (
    MAX_CHUNK_CHARS,
    MAX_CHUNKS_PER_ENTITY,
    MIN_CHUNK_CHARS,
    VALUE_SEPARATOR,
    canonicalize_values,
    chunk_text,
    content_hash,
    extract_semantic_chunks,
)

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
XSD_STRING = "http://www.w3.org/2001/XMLSchema#string"

TENANT = "demo-tenant"
KG = "EventsSF"
MARKED = {"description", "notes", "bio"}


def _desc(uri: str, text: str, attr: str = "description") -> tuple:
    return (uri, f"https://cograph.tech/types/Event/{attr}", text)


def _extract(triples, marked=MARKED):
    return extract_semantic_chunks(
        triples, tenant_id=TENANT, kg_name=KG, marked_predicates=marked
    )


# ---------------------------------------------------------------------------
# chunk_text — the char-estimation chunker
# ---------------------------------------------------------------------------


def test_chunk_empty_and_whitespace_yield_zero_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  \n") == []


def test_chunk_short_text_is_exactly_one_chunk():
    assert chunk_text("A short description.") == ["A short description."]
    # Right at the ceiling still fits in one chunk.
    exact = "x" * MAX_CHUNK_CHARS
    assert chunk_text(exact) == [exact]


def test_chunk_long_text_splits_within_bounds():
    text = " ".join(f"Sentence number {i} says something useful." for i in range(200))
    chunks = chunk_text(text)
    assert len(chunks) > 1
    for c in chunks:
        assert 0 < len(c) <= MAX_CHUNK_CHARS
    # Natural (sentence/whitespace) breaks: no word is ever split, so the
    # multiset of words survives the round-trip.
    assert " ".join(chunks).split() == text.split()


def test_chunk_prefers_paragraph_boundaries():
    # Two paragraphs, each individually under the max but jointly over it: the
    # split must land on the blank line, not mid-paragraph.
    para1 = "First paragraph. " * 80  # ~1360 chars
    para2 = "Second paragraph. " * 80
    chunks = chunk_text(f"{para1.strip()}\n\n{para2.strip()}")
    assert len(chunks) == 2
    assert chunks[0] == para1.strip()
    assert chunks[1] == para2.strip()


def test_chunk_unbroken_text_hard_cuts():
    """Pathological text with no whitespace at all must still terminate, via
    hard cuts at the window edge (the only case a 'word' is split)."""
    blob = "a" * (MAX_CHUNK_CHARS * 3 + 100)
    chunks = chunk_text(blob)
    assert "".join(chunks) == blob
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)


def test_chunk_no_sliver_chunks():
    """Natural breaks are only accepted at/after MIN_CHUNK_CHARS, so no chunk
    except the final remainder can be a tiny sliver."""
    text = "Word. " * 2000
    chunks = chunk_text(text)
    assert all(len(c) >= MIN_CHUNK_CHARS for c in chunks[:-1])


# ---------------------------------------------------------------------------
# canonicalize_values / content_hash
# ---------------------------------------------------------------------------


def test_canonicalize_sorts_dedups_and_strips():
    assert canonicalize_values(["  b value ", "a value", "b value"]) == (
        f"a value{VALUE_SEPARATOR}b value"
    )
    assert canonicalize_values(["", "   ", "\n"]) == ""


def test_content_hash_is_sha256_of_canonical_doc():
    doc = "a value"
    assert content_hash(doc) == hashlib.sha256(doc.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# extract_semantic_chunks — basics
# ---------------------------------------------------------------------------


def test_extracts_marked_predicate_into_keyed_chunk():
    chunks = _extract(
        [
            ("e:1", RDF_TYPE, "https://cograph.tech/types/Event"),
            ("e:1", RDFS_LABEL, "Solar Expo"),
            _desc("e:1", "An expo about solar panels."),
        ]
    )
    assert len(chunks) == 1
    c = chunks[0]
    assert c.key() == (TENANT, KG, "e:1", "description", 0)
    assert c.chunk_text == "An expo about solar panels."
    assert c.content_hash == content_hash("An expo about solar panels.")
    # Denormalized display attrs, mirroring the spatio-temporal facts.
    assert c.attrs == {"label": "Solar Expo", "type": "Event"}
    # Fresh rows are always pending-embed: the NULL embedding IS the queue.
    assert c.embedding is None and c.embed_model is None
    assert c.attempt_count == 0 and c.last_error is None


def test_unmarked_predicates_are_ignored():
    chunks = _extract(
        [
            ("e:1", "https://cograph.tech/types/Event/sku", "ABC-123"),
            ("e:1", "https://cograph.tech/types/Event/venue_name", "Moscone"),
        ]
    )
    assert chunks == []


def test_marker_matches_full_uri_and_local_name():
    pred = "https://cograph.tech/types/Event/description"
    by_uri = _extract([("e:1", pred, "Some text.")], marked={pred})
    by_local = _extract([("e:1", pred, "Some text.")], marked={"description"})
    by_local_cased = _extract([("e:1", pred, "Some text.")], marked={"Description"})
    assert len(by_uri) == len(by_local) == len(by_local_cased) == 1
    # attr is always the (lowered) local name regardless of marker form.
    assert {c.attr for c in by_uri + by_local + by_local_cased} == {"description"}


def test_uri_objects_are_never_indexed():
    """A relation predicate sharing a marked local name must not index the
    target URI as text (mirrors _escape_value's URI-vs-literal decision)."""
    chunks = _extract(
        [
            ("e:1", "https://cograph.tech/types/Event/description", "https://cograph.tech/entities/e2"),
            ("e:1", "https://cograph.tech/types/Event/description", "<https://cograph.tech/entities/e3>"),
        ]
    )
    assert chunks == []


def test_typed_literal_datatype_is_stripped():
    chunks = _extract([_desc("e:1", f"Plain text value.^^{XSD_STRING}")])
    assert len(chunks) == 1
    assert chunks[0].chunk_text == "Plain text value."
    assert chunks[0].content_hash == content_hash("Plain text value.")


# ---------------------------------------------------------------------------
# edge cases from the ONTA-175 contract
# ---------------------------------------------------------------------------


def test_empty_and_whitespace_values_yield_zero_chunks():
    assert _extract([_desc("e:1", "")]) == []
    assert _extract([_desc("e:1", "   \n\t ")]) == []


def test_short_value_yields_exactly_one_chunk():
    chunks = _extract([_desc("e:1", "Short.")])
    assert [c.chunk_ix for c in chunks] == [0]


def test_long_value_yields_contiguous_chunks_sharing_one_hash():
    text = " ".join(f"Sentence {i} of the very long description." for i in range(150))
    chunks = _extract([_desc("e:1", text)])
    assert len(chunks) > 1
    assert [c.chunk_ix for c in chunks] == list(range(len(chunks)))
    # Every chunk of the (entity, attr) doc carries the SAME doc-level hash.
    assert {c.content_hash for c in chunks} == {content_hash(text)}


def test_multivalued_attribute_is_sorted_before_hashing():
    """Triple order must not change the doc or its hash (sort-before-hash)."""
    forward = _extract([_desc("e:1", "beta value"), _desc("e:1", "alpha value")])
    backward = _extract([_desc("e:1", "alpha value"), _desc("e:1", "beta value")])
    expected_doc = f"alpha value{VALUE_SEPARATOR}beta value"
    assert forward[0].chunk_text == expected_doc
    assert [c.content_hash for c in forward] == [c.content_hash for c in backward]
    assert forward[0].content_hash == content_hash(expected_doc)


def test_intra_entity_duplicate_values_dedup():
    """The same value repeated (duplicate triples) hashes like a single value."""
    chunks = _extract([_desc("e:1", "same text"), _desc("e:1", "same text")])
    assert len(chunks) == 1
    assert chunks[0].content_hash == content_hash("same text")


def test_intra_entity_duplicate_docs_across_attrs_dedup():
    """Two marked attrs carrying the identical doc index ONCE (first attr in
    triple order wins) — double-indexing would double-count in every ranking."""
    chunks = _extract(
        [
            _desc("e:1", "mirrored text", attr="description"),
            _desc("e:1", "mirrored text", attr="notes"),
        ]
    )
    assert len(chunks) == 1
    assert chunks[0].attr == "description"
    # The same doc on ANOTHER entity is not deduped — the dedup is per entity.
    chunks2 = _extract(
        [
            _desc("e:1", "mirrored text", attr="description"),
            _desc("e:2", "mirrored text", attr="notes"),
        ]
    )
    assert {(c.entity_uri, c.attr) for c in chunks2} == {
        ("e:1", "description"),
        ("e:2", "notes"),
    }


def test_chunk_cap_truncates_and_logs_never_silent():
    # ~120k tiny sentences -> far beyond 200 chunks' worth of text.
    big = " ".join(f"word{i}." for i in range(120_000))
    with structlog.testing.capture_logs() as logs:
        chunks = _extract([_desc("e:1", big)])
    assert len(chunks) == MAX_CHUNKS_PER_ENTITY
    assert [c.chunk_ix for c in chunks] == list(range(MAX_CHUNKS_PER_ENTITY))
    cap_events = [l for l in logs if l["event"] == "semantic_extract_chunk_cap"]
    assert cap_events and cap_events[0]["log_level"] == "warning"
    assert cap_events[0]["entity_uri"] == "e:1"
    assert cap_events[0]["dropped"] > 0


def test_chunk_cap_spans_all_attrs_of_one_entity():
    """The cap is per ENTITY: a second attr only gets the remaining budget."""
    big = " ".join(f"word{i}." for i in range(120_000))
    with structlog.testing.capture_logs():
        chunks = _extract(
            [_desc("e:1", big, attr="description"), _desc("e:1", "tiny", attr="notes")]
        )
    assert len(chunks) == MAX_CHUNKS_PER_ENTITY
    assert all(c.attr == "description" for c in chunks)  # notes got budget 0


def test_cap_does_not_leak_across_entities():
    big = " ".join(f"word{i}." for i in range(120_000))
    with structlog.testing.capture_logs():
        chunks = _extract([_desc("e:1", big), _desc("e:2", "small doc")])
    by_entity = {c.entity_uri for c in chunks}
    assert by_entity == {"e:1", "e:2"}
    assert sum(1 for c in chunks if c.entity_uri == "e:2") == 1


# ---------------------------------------------------------------------------
# denormalized attrs + multi-entity behavior
# ---------------------------------------------------------------------------


def test_attrs_empty_when_no_label_or_type():
    chunks = _extract([_desc("e:1", "No display fields here.")])
    assert chunks[0].attrs == {}


def test_label_from_name_local_and_first_type_wins():
    chunks = _extract(
        [
            ("e:1", "https://cograph.tech/types/Person/name", "Ada Lovelace"),
            ("e:1", RDF_TYPE, "https://cograph.tech/types/Person"),
            ("e:1", RDF_TYPE, "https://cograph.tech/types/Author"),
            ("e:1", "https://cograph.tech/types/Person/bio", "Wrote the first program."),
        ]
    )
    assert chunks[0].attrs == {"label": "Ada Lovelace", "type": "Person"}


def test_marked_label_predicate_contributes_text_and_display():
    """A predicate can be BOTH the display label and a marked text attr
    (e.g. `title` on an Article) — it must serve both roles."""
    chunks = _extract(
        [("e:1", "https://cograph.tech/types/Article/title", "A Grand Title")],
        marked={"title"},
    )
    assert len(chunks) == 1
    assert chunks[0].attr == "title"
    assert chunks[0].chunk_text == "A Grand Title"
    assert chunks[0].attrs == {"label": "A Grand Title"}


def test_multiple_entities_extracted_independently():
    chunks = _extract(
        [
            _desc("e:1", "First entity text."),
            _desc("e:2", "Second entity text."),
        ]
    )
    assert {(c.entity_uri, c.chunk_ix) for c in chunks} == {("e:1", 0), ("e:2", 0)}
    assert all(c.tenant_id == TENANT and c.kg_name == KG for c in chunks)


def test_marker_map_keys_work_as_marker_set():
    """The ONTA-181 hook holds a marker MAP; membership over its keys must be
    enough (no dedicated map handling in the extractor)."""
    marker_map = {"description": {"marked": True, "source": "auto"}}
    chunks = _extract([_desc("e:1", "Map-marked text.")], marked=marker_map)
    assert len(chunks) == 1


def test_extract_is_deterministic():
    triples = [
        ("e:1", RDF_TYPE, "https://cograph.tech/types/Event"),
        _desc("e:1", "beta"),
        _desc("e:1", "alpha"),
        _desc("e:2", "other"),
    ]
    a = _extract(triples)
    b = _extract(triples)
    assert [(c.key(), c.chunk_text, c.content_hash) for c in a] == [
        (c.key(), c.chunk_text, c.content_hash) for c in b
    ]
