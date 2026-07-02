"""Semantic index protocol conformance, registry wiring, and the InMemory
backend end-to-end (ONTA-175): replace-per-doc upsert semantics, hybrid
lexical+vector search with RRF fusion and the ``degraded`` signal, tenant/KG
isolation, and the embed-fill sweep seam (fetch_pending / fill_embeddings /
mark_embed_failed with the content_hash concurrency guard).
"""

from __future__ import annotations

import pytest

from cograph_client.semantic import (
    InMemorySemanticIndex,
    SemanticChunk,
    SemanticIndex,
    SemanticSearchResult,
    content_hash,
    get_semantic_index,
    make_semantic_index,
    register_semantic_index,
    reset_semantic_index,
)

TENANT = "demo-tenant"
OTHER_TENANT = "spider-bench"
KG = "kg1"
OTHER_KG = "kg2"
#: The "current" embed model for these tests: the vector leg (both backends)
#: only trusts vectors produced by the index's current model.
FAKE_MODEL = "fake-embed-model"


def _chunk(
    uri: str,
    text: str,
    *,
    ix: int = 0,
    attr: str = "description",
    tenant: str = TENANT,
    kg: str = KG,
    doc_text: str | None = None,
    embedding: list[float] | None = None,
    embed_model: str | None = None,
    attrs: dict | None = None,
) -> SemanticChunk:
    """A chunk row; ``doc_text`` overrides the hashed doc for multi-chunk docs
    (all chunks of one doc share the doc-level hash)."""
    return SemanticChunk(
        tenant_id=tenant,
        kg_name=kg,
        entity_uri=uri,
        attr=attr,
        chunk_ix=ix,
        chunk_text=text,
        content_hash=content_hash(doc_text if doc_text is not None else text),
        embedding=embedding,
        embed_model=embed_model
        if embed_model is not None
        else (FAKE_MODEL if embedding is not None else None),
        attrs=attrs if attrs is not None else {"label": uri},
    )


@pytest.fixture
def idx() -> InMemorySemanticIndex:
    return InMemorySemanticIndex(embed_model=FAKE_MODEL)


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_semantic_index()
    yield
    reset_semantic_index()


def _uris(result: SemanticSearchResult) -> set[str]:
    return {h.entity_uri for h in result.hits}


# ---------------------------------------------------------------------------
# Protocol + registry
# ---------------------------------------------------------------------------


def test_protocol_conformance(idx):
    assert isinstance(idx, SemanticIndex)


def test_factory_returns_inmemory(monkeypatch):
    # Unconditional for now — the pgvector branch is the ONTA-176 seam.
    from cograph_client import config

    monkeypatch.setattr(config.settings, "database_url", "", raising=False)
    assert isinstance(make_semantic_index(), InMemorySemanticIndex)


def test_register_and_get_roundtrip(idx):
    register_semantic_index(idx)
    assert get_semantic_index() is idx
    register_semantic_index(None)
    # Falls back to a lazily-built (cached) default.
    default = get_semantic_index()
    assert isinstance(default, InMemorySemanticIndex)
    assert get_semantic_index() is default  # cached


def test_reset_clears_cached_default():
    default = get_semantic_index()
    reset_semantic_index()
    assert get_semantic_index() is not default


# ---------------------------------------------------------------------------
# upsert semantics (replace-per-doc)
# ---------------------------------------------------------------------------


async def test_upsert_idempotent_by_pk(idx):
    await idx.upsert_chunks([_chunk("e:1", "solar panels for homes")])
    await idx.upsert_chunks([_chunk("e:1", "solar panels for homes")])
    res = await idx.search(TENANT, "solar panels")
    assert len(res.hits) == 1


async def test_upsert_unchanged_hash_preserves_filled_embedding(idx):
    """Replaying an unchanged doc must NOT reset a filled embedding to NULL
    (that would re-queue embed work on every replay)."""
    await idx.upsert_chunks([_chunk("e:1", "some text")])
    [pending] = await idx.fetch_pending()
    assert await idx.fill_embeddings([pending], [[1.0, 0.0]], embed_model="m1") == 1
    # Replay the identical chunk (fresh extract → embedding=None on the input).
    await idx.upsert_chunks([_chunk("e:1", "some text")])
    assert await idx.fetch_pending() == []  # still embedded, nothing re-queued


async def test_upsert_changed_hash_replaces_and_requeues(idx):
    await idx.upsert_chunks([_chunk("e:1", "old text")])
    [pending] = await idx.fetch_pending()
    await idx.fill_embeddings([pending], [[1.0, 0.0]], embed_model="m1")
    await idx.upsert_chunks([_chunk("e:1", "new text")])
    requeued = await idx.fetch_pending()
    assert [c.chunk_text for c in requeued] == ["new text"]
    assert requeued[0].embedding is None


async def test_upsert_shrunken_doc_deletes_stale_tail(idx):
    """A doc that re-chunks to fewer pieces must not leave ghost tail chunks
    (the ONTA-181 fewer-chunks-than-before case)."""
    await idx.upsert_chunks(
        [
            _chunk("e:1", "part one about quasars", ix=0, doc_text="v1"),
            _chunk("e:1", "part two about pulsars", ix=1, doc_text="v1"),
            _chunk("e:1", "part three about nebulae", ix=2, doc_text="v1"),
        ]
    )
    await idx.upsert_chunks([_chunk("e:1", "just quasars now", ix=0, doc_text="v2")])
    res = await idx.search(TENANT, "pulsars nebulae")
    assert res.hits == []  # tail chunks are gone
    res = await idx.search(TENANT, "quasars")
    assert _uris(res) == {"e:1"}


async def test_upsert_tail_delete_is_scoped_to_the_doc(idx):
    """Shrinking one (entity, attr) doc must not touch a sibling attr's chunks
    or another entity's."""
    await idx.upsert_chunks(
        [
            _chunk("e:1", "description part one", ix=0, doc_text="d1"),
            _chunk("e:1", "description part two", ix=1, doc_text="d1"),
            _chunk("e:1", "the notes text", ix=1, attr="notes", doc_text="n1"),
            _chunk("e:2", "other entity text", ix=1, doc_text="o1"),
        ]
    )
    await idx.upsert_chunks([_chunk("e:1", "short description", ix=0, doc_text="d2")])
    assert _uris(await idx.search(TENANT, "notes")) == {"e:1"}
    assert _uris(await idx.search(TENANT, "other entity")) == {"e:2"}
    assert (await idx.search(TENANT, "part two")).hits == []


# ---------------------------------------------------------------------------
# search — lexical, hybrid, degraded
# ---------------------------------------------------------------------------


async def test_lexical_search_finds_and_ranks(idx):
    await idx.upsert_chunks(
        [
            _chunk("e:solar", "installation of solar panels on residential roofs"),
            _chunk("e:wind", "wind turbine maintenance schedule"),
            _chunk("e:misc", "quarterly financial report"),
        ]
    )
    res = await idx.search(TENANT, "solar panels")
    assert res.hits[0].entity_uri == "e:solar"
    assert "e:misc" not in _uris(res)


async def test_search_without_embedding_is_degraded(idx):
    await idx.upsert_chunks([_chunk("e:1", "anything at all")])
    res = await idx.search(TENANT, "anything")
    assert res.degraded is True
    res = await idx.search(TENANT, "anything", query_embedding=[1.0, 0.0])
    assert res.degraded is False


async def test_hybrid_vector_leg_finds_lexical_misses(idx):
    """A chunk with zero token overlap but a near-identical embedding must be
    reachable through the vector leg (the whole point of hybrid)."""
    await idx.upsert_chunks(
        [
            _chunk("e:vec", "cardiac arrhythmia treatment", embedding=[1.0, 0.0, 0.0]),
            _chunk("e:lex", "heart rhythm disorder care"),
        ]
    )
    res = await idx.search(
        TENANT, "heart rhythm disorder", query_embedding=[0.99, 0.1, 0.0]
    )
    assert _uris(res) == {"e:vec", "e:lex"}


async def test_hybrid_rrf_prefers_dual_leg_matches(idx):
    """An entity ranked in BOTH legs outscores one ranked in a single leg
    (RRF sums 1/(k+rank) across legs)."""
    await idx.upsert_chunks(
        [
            _chunk("e:both", "solar panels efficiency", embedding=[1.0, 0.0]),
            _chunk("e:lexonly", "solar panels efficiency"),  # no embedding
        ]
    )
    res = await idx.search(TENANT, "solar panels", query_embedding=[1.0, 0.0])
    assert res.hits[0].entity_uri == "e:both"
    assert res.hits[0].score > res.hits[1].score


async def test_hit_carries_attrs_snippet_and_attr(idx):
    await idx.upsert_chunks(
        [
            _chunk(
                "e:1",
                "A long description about solar panel efficiency in cold climates.",
                attrs={"label": "Solar Report", "type": "Report"},
            )
        ]
    )
    res = await idx.search(TENANT, "solar efficiency")
    hit = res.hits[0]
    assert hit.attrs == {"label": "Solar Report", "type": "Report"}
    assert hit.attr == "description"
    assert hit.snippet.startswith("A long description")


async def test_snippet_is_truncated_for_long_chunks(idx):
    long_text = "solar " + "filler words appear here " * 40
    await idx.upsert_chunks([_chunk("e:1", long_text)])
    res = await idx.search(TENANT, "solar")
    assert len(res.hits[0].snippet) < len(long_text)
    assert res.hits[0].snippet.endswith("…")


async def test_chunks_group_into_one_entity_hit(idx):
    await idx.upsert_chunks(
        [
            _chunk("e:1", "solar panels part one", ix=0, doc_text="d"),
            _chunk("e:1", "solar panels part two", ix=1, doc_text="d"),
        ]
    )
    res = await idx.search(TENANT, "solar panels")
    assert len(res.hits) == 1
    assert res.hits[0].entity_uri == "e:1"


async def test_top_k_limits_entities(idx):
    await idx.upsert_chunks(
        [_chunk(f"e:{i}", f"solar panels variant {i}") for i in range(10)]
    )
    res = await idx.search(TENANT, "solar panels", top_k=3)
    assert len(res.hits) == 3


async def test_no_match_returns_empty(idx):
    await idx.upsert_chunks([_chunk("e:1", "wind turbines")])
    res = await idx.search(TENANT, "quantum entanglement")
    assert res.hits == []


# ---------------------------------------------------------------------------
# isolation: tenant, KG, type filter
# ---------------------------------------------------------------------------


async def test_tenant_isolation(idx):
    await idx.upsert_chunks(
        [
            _chunk("e:a", "shared query text", tenant=TENANT),
            _chunk("e:b", "shared query text", tenant=OTHER_TENANT),
        ]
    )
    assert _uris(await idx.search(TENANT, "shared query")) == {"e:a"}
    assert _uris(await idx.search(OTHER_TENANT, "shared query")) == {"e:b"}


async def test_kg_narrowing_on_search(idx):
    await idx.upsert_chunks(
        [
            _chunk("e:a", "common topic text", kg=KG),
            _chunk("e:b", "common topic text", kg=OTHER_KG),
        ]
    )
    # No kg_name → every KG in the tenant.
    assert _uris(await idx.search(TENANT, "common topic")) == {"e:a", "e:b"}
    assert _uris(await idx.search(TENANT, "common topic", kg_name=KG)) == {"e:a"}


async def test_type_filter_matches_denormalized_type(idx):
    await idx.upsert_chunks(
        [
            _chunk("e:ev", "annual gathering", attrs={"type": "Event"}),
            _chunk("e:org", "annual gathering", attrs={"type": "Organization"}),
        ]
    )
    res = await idx.search(TENANT, "annual gathering", type_filter="Event")
    assert _uris(res) == {"e:ev"}


async def test_delete_entity_and_attr_scoping(idx):
    await idx.upsert_chunks(
        [
            _chunk("e:1", "delete me description"),
            _chunk("e:1", "delete me notes", attr="notes"),
            _chunk("e:2", "delete me too"),
        ]
    )
    # attr-scoped delete: the reconciler's ghost-deletion primitive.
    await idx.delete("e:1", TENANT, attr="notes")
    assert _uris(await idx.search(TENANT, "delete")) == {"e:1", "e:2"}
    # entity-wide delete removes remaining docs.
    await idx.delete("e:1", TENANT)
    assert _uris(await idx.search(TENANT, "delete")) == {"e:2"}


async def test_delete_scoped_to_kg(idx):
    await idx.upsert_chunks(
        [
            _chunk("e:shared", "same entity two kgs", kg=KG),
            _chunk("e:shared", "same entity two kgs", kg=OTHER_KG),
        ]
    )
    await idx.delete("e:shared", TENANT, kg_name=KG)
    assert _uris(await idx.search(TENANT, "same entity", kg_name=OTHER_KG)) == {
        "e:shared"
    }
    assert (await idx.search(TENANT, "same entity", kg_name=KG)).hits == []


async def test_clear_one_kg_leaves_sibling(idx):
    """The crux of the kg_name dimension (KG delete): dropping one KG must not
    wipe a sibling KG's chunks."""
    await idx.upsert_chunks(
        [
            _chunk("e:a", "alpha content", kg=KG),
            _chunk("e:b", "alpha content", kg=OTHER_KG),
        ]
    )
    await idx.clear(TENANT, kg_name=KG)
    assert _uris(await idx.search(TENANT, "alpha content")) == {"e:b"}
    await idx.clear(TENANT)  # tenant-wide clear removes the rest
    assert (await idx.search(TENANT, "alpha content")).hits == []


async def test_clear_per_tenant(idx):
    await idx.upsert_chunks(
        [
            _chunk("e:a", "text here", tenant=TENANT),
            _chunk("e:b", "text here", tenant=OTHER_TENANT),
        ]
    )
    await idx.clear(TENANT)
    assert (await idx.search(TENANT, "text here")).hits == []
    assert _uris(await idx.search(OTHER_TENANT, "text here")) == {"e:b"}


# ---------------------------------------------------------------------------
# list_docs (the reconciler's ghost-diff enumeration)
# ---------------------------------------------------------------------------


async def test_list_docs_empty_index(idx):
    assert await idx.list_docs(TENANT) == []


async def test_list_docs_is_doc_granular_not_chunk_granular(idx):
    """One row per (entity, attr) DOC: a multi-chunk doc collapses to its
    chunk-0 row (every chunk carries the same doc-level hash by construction,
    and attrs are written per doc) — the granularity the reconciler's ghost
    diff and attrs-repair pass operate on."""
    await idx.upsert_chunks(
        [
            _chunk("e:1", "part one about quasars", ix=0, doc_text="d1"),
            _chunk("e:1", "part two about pulsars", ix=1, doc_text="d1"),
            _chunk("e:1", "the notes text", attr="notes"),
            _chunk("e:2", "other entity text"),
        ]
    )
    assert await idx.list_docs(TENANT) == [
        ("e:1", "description", content_hash("d1"), {"label": "e:1"}),
        ("e:1", "notes", content_hash("the notes text"), {"label": "e:1"}),
        ("e:2", "description", content_hash("other entity text"), {"label": "e:2"}),
    ]  # sorted (deterministic), one row per doc despite the two d1 chunks


async def test_list_docs_scoping(idx):
    """Identical scoping semantics to search/clear: tenant mandatory (never
    cross tenants), kg_name None = every KG in the tenant."""
    await idx.upsert_chunks(
        [
            _chunk("e:a", "alpha text", tenant=TENANT, kg=KG),
            _chunk("e:b", "beta text", tenant=TENANT, kg=OTHER_KG),
            _chunk("e:c", "gamma text", tenant=OTHER_TENANT, kg=KG),
        ]
    )
    assert [row[0] for row in await idx.list_docs(TENANT)] == ["e:a", "e:b"]
    assert [row[0] for row in await idx.list_docs(TENANT, kg_name=KG)] == ["e:a"]
    assert [row[0] for row in await idx.list_docs(OTHER_TENANT)] == ["e:c"]
    assert await idx.list_docs("no-such-tenant") == []


async def test_list_docs_tracks_hash_changes_and_deletes(idx):
    """The listing is the reconciler's change-detection currency: a replaced
    doc surfaces its NEW hash (still one row), and a deleted doc vanishes."""
    await idx.upsert_chunks([_chunk("e:1", "version one")])
    [(_, _, h1, _attrs)] = await idx.list_docs(TENANT)
    assert h1 == content_hash("version one")
    await idx.upsert_chunks([_chunk("e:1", "version two")])
    [(_, _, h2, _attrs)] = await idx.list_docs(TENANT)  # still exactly one row
    assert h2 == content_hash("version two") != h1
    await idx.delete("e:1", TENANT, attr="description")
    assert await idx.list_docs(TENANT) == []


async def test_unchanged_hash_upsert_repairs_attrs_and_keeps_embedding(idx):
    """The attrs half of the upsert contract: replaying a doc with the SAME
    text but different denormalized attrs (chunk born attrs={}, later writes
    carry the type/label) must refresh attrs WITHOUT resetting the filled
    embedding — the enrichment-shaped repair the reconciler relies on."""
    await idx.upsert_chunks([_chunk("e:1", "same text", attrs={})])
    [pending] = await idx.fetch_pending()
    assert await idx.fill_embeddings([pending], [[1.0, 0.0]], embed_model=FAKE_MODEL) == 1

    await idx.upsert_chunks(
        [_chunk("e:1", "same text", attrs={"label": "Fixed", "type": "Report"})]
    )
    assert await idx.fetch_pending() == []  # embedding survived, not re-queued
    [(_, _, h, attrs)] = await idx.list_docs(TENANT)
    assert h == content_hash("same text")  # text state untouched
    assert attrs == {"label": "Fixed", "type": "Report"}  # attrs repaired
    # The repaired attrs are live on hits (and for the type_filter).
    res = await idx.search(TENANT, "same text", type_filter="Report")
    assert _uris(res) == {"e:1"}
    assert res.hits[0].attrs == {"label": "Fixed", "type": "Report"}


async def test_delete_docs_batches_and_scopes(idx):
    """delete_docs removes exactly the (entity, attr) pairs in the given KG —
    the reconciler's batched ghost-deletion primitive."""
    await idx.upsert_chunks(
        [
            _chunk("e:1", "ghost description"),
            _chunk("e:1", "living notes", attr="notes"),
            _chunk("e:2", "ghost too"),
            _chunk("e:1", "sibling kg twin", kg=OTHER_KG),
            _chunk("e:3", "other tenant twin", tenant=OTHER_TENANT),
        ]
    )
    await idx.delete_docs(
        [("e:1", "description"), ("e:2", "description")], TENANT, kg_name=KG
    )
    assert [(row[0], row[1]) for row in await idx.list_docs(TENANT, kg_name=KG)] == [
        ("e:1", "notes")
    ]
    # The sibling KG and the other tenant were untouched.
    assert [row[0] for row in await idx.list_docs(TENANT, kg_name=OTHER_KG)] == ["e:1"]
    assert [row[0] for row in await idx.list_docs(OTHER_TENANT)] == ["e:3"]
    # Empty pairs is a no-op.
    await idx.delete_docs([], TENANT, kg_name=KG)
    assert len(await idx.list_docs(TENANT, kg_name=KG)) == 1


# ---------------------------------------------------------------------------
# embed-fill sweep seam (the NULL embedding IS the queue)
# ---------------------------------------------------------------------------


async def test_fetch_pending_returns_null_embedding_rows_in_order(idx):
    await idx.upsert_chunks(
        [
            _chunk("e:2", "beta text"),
            _chunk("e:1", "alpha text"),
            _chunk("e:3", "gamma text", embedding=[0.5, 0.5]),  # already embedded
        ]
    )
    pending = await idx.fetch_pending()
    assert [c.entity_uri for c in pending] == ["e:1", "e:2"]  # deterministic (PK order)
    assert all(c.embedding is None for c in pending)


async def test_fetch_pending_respects_limit_and_scoping(idx):
    await idx.upsert_chunks(
        [
            _chunk("e:1", "t1", tenant=TENANT, kg=KG),
            _chunk("e:2", "t2", tenant=TENANT, kg=OTHER_KG),
            _chunk("e:3", "t3", tenant=OTHER_TENANT),
        ]
    )
    assert len(await idx.fetch_pending(limit=2)) == 2
    assert len(await idx.fetch_pending()) == 3  # maintenance: all tenants
    assert {c.entity_uri for c in await idx.fetch_pending(tenant_id=TENANT)} == {
        "e:1",
        "e:2",
    }
    assert [c.entity_uri for c in await idx.fetch_pending(tenant_id=TENANT, kg_name=OTHER_KG)] == ["e:2"]


async def test_fill_embeddings_stamps_model_and_drains_queue(idx):
    await idx.upsert_chunks([_chunk("e:1", "embed me")])
    pending = await idx.fetch_pending()
    assert await idx.fill_embeddings(pending, [[0.1, 0.2]], embed_model=FAKE_MODEL) == 1
    assert await idx.fetch_pending() == []
    # The filled vector is live for the vector leg (stamped with the current
    # model, so it passes the leg's embed_model filter).
    res = await idx.search(TENANT, "zzz nolexicalmatch", query_embedding=[0.1, 0.2])
    assert _uris(res) == {"e:1"}


async def test_vector_leg_ignores_stale_model_embeddings(idx):
    """The vector leg only trusts vectors from the CURRENT embed model — a
    chunk embedded under an older model must be invisible to a pure-vector
    probe (the same rule as the pgvector ANN leg's ``embed_model =`` filter),
    without flipping ``degraded``."""
    await idx.upsert_chunks(
        [_chunk("e:old", "no lexical overlap here", embedding=[1.0, 0.0], embed_model="old-model")]
    )
    res = await idx.search(TENANT, "zzz nolexicalmatch", query_embedding=[1.0, 0.0])
    assert res.hits == []
    assert res.degraded is False  # the leg ran; it just (rightly) found nothing


async def test_dim_mismatch_query_embedding_degrades_loudly(idx):
    """A query embedding whose dimension matches NO current-model chunk must
    run lexical-only with degraded=True + a warning (mirrors the pgvector
    backend's dim-mismatch degrade) — never a silent all-zero vector leg."""
    import structlog

    await idx.upsert_chunks(
        [_chunk("e:1", "resilient text about pumps", embedding=[1.0, 0.0, 0.0, 0.0])]
    )
    with structlog.testing.capture_logs() as logs:
        res = await idx.search(TENANT, "resilient text", query_embedding=[1.0, 0.0])
    assert res.degraded is True
    assert _uris(res) == {"e:1"}  # lexical leg still works
    assert any(
        e["event"] == "semantic_query_embedding_dim_mismatch" for e in logs
    )


async def test_fill_embeddings_stale_hash_does_not_apply(idx):
    """The content_hash concurrency guard: if the doc changed between fetch and
    fill, the stale vector must NOT land on the new text."""
    await idx.upsert_chunks([_chunk("e:1", "version one")])
    [stale] = await idx.fetch_pending()
    await idx.upsert_chunks([_chunk("e:1", "version two")])  # replaced mid-flight
    assert await idx.fill_embeddings([stale], [[1.0, 0.0]], embed_model="m") == 0
    [still_pending] = await idx.fetch_pending()
    assert still_pending.chunk_text == "version two"
    assert still_pending.embedding is None


async def test_fill_embeddings_length_mismatch_raises(idx):
    await idx.upsert_chunks([_chunk("e:1", "text")])
    pending = await idx.fetch_pending()
    with pytest.raises(ValueError):
        await idx.fill_embeddings(pending, [], embed_model="m")


async def test_mark_embed_failed_tracks_attempts_and_dead_letters(idx):
    await idx.upsert_chunks([_chunk("e:1", "flaky text")])
    [pending] = await idx.fetch_pending()
    assert await idx.mark_embed_failed([pending], error="429 rate limited") == 1
    [row] = await idx.fetch_pending()
    assert row.attempt_count == 1
    assert row.last_error == "429 rate limited"
    # A max_attempts cutoff dead-letters the row (skipped, not deleted).
    assert await idx.fetch_pending(max_attempts=1) == []
    assert len(await idx.fetch_pending(max_attempts=2)) == 1
    # A successful fill clears the error.
    assert await idx.fill_embeddings([row], [[0.3, 0.4]], embed_model="m") == 1
    assert await idx.fetch_pending() == []


async def test_mark_embed_failed_stale_hash_is_ignored(idx):
    await idx.upsert_chunks([_chunk("e:1", "v1")])
    [stale] = await idx.fetch_pending()
    await idx.upsert_chunks([_chunk("e:1", "v2")])
    assert await idx.mark_embed_failed([stale], error="boom") == 0
    [row] = await idx.fetch_pending()
    assert row.attempt_count == 0 and row.last_error is None
