"""Relevance golden-queries for the semantic instance index (ONTA-178).

This suite is the ONLY guard on retrieval *quality* knobs — chunk size
(``extract.MIN/MAX_CHUNK_CHARS``), RRF fusion (k=60, 50-per-leg), and the
Postgres ``regconfig`` ('simple') — none of which any unit test constrains.
Change one of them and these hit@k assertions are what tells you whether
retrieval got better or worse. The corpus is a checked-in fixture of real-ish
parliamentary speeches (``tests/fixtures/semantic_golden_speeches.json``):
12 (entity, transcript) docs across two KGs and two types, three of them long
enough that the chunker actually splits them.

Determinism / CI-safety — NO network:

* Embeddings are hand-built **topic-axis vectors**: each doc carries a
  ``topics`` weight map over 8 named axes; each query carries its own. The
  test converts both to plain 8-dim vectors. Paraphrase queries reuse their
  target's topic mix (near in cosine space); unrelated docs sit on disjoint
  axes (far). Exact-quote queries use the reserved ``offtopic`` axis that no
  doc touches — orthogonal to the entire corpus — so their hits can only be
  carried by the FTS leg, which is exactly what that class guards.
* Both backends run: InMemory always; Postgres+pgvector when
  ``OMNIX_DATABASE_URL`` is set (mirroring ``test_semantic_parity.py`` — CI's
  DSN-gated step runs this file against the pgvector service container). The
  vector dimension is 8, matching the ``vector(8)`` table the other DSN-gated
  semantic tests create in the shared CI database.
* Seeding goes through the REAL production path: triples →
  ``extract_semantic_chunks`` (marker-driven, chunked) → ``upsert_chunks`` →
  ``fetch_pending``/``fill_embeddings`` (the embed-queue seam), so the gate
  also covers extraction + queue plumbing, not just ``search``.

Documented hit@k per query class (entity-level; rationale in the fixture):

* ``exact``      — k=1. FTS must win alone: the query embedding is orthogonal
  to every doc, so on the memory backend the vector leg is empty and on
  Postgres it contributes only noise ranks that RRF cannot elevate above an
  FTS rank-1.
* ``paraphrase`` — k=3. The dense leg must win; k=3 absorbs RRF interference
  from incidental lexical matches of common words (an FTS rank-1 competitor
  plus an ANN rank-2 can fuse above a pure ANN rank-1 — measured, not
  hypothetical).
* ``filtered``   — k=2, plus a strict exclusion assertion: no hit may come
  from outside the requested kg/type scope.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from cograph_client.semantic import InMemorySemanticIndex, SemanticChunk
from cograph_client.semantic.extract import (
    MAX_CHUNK_CHARS,
    extract_semantic_chunks,
)
from cograph_client.semantic.postgres import PostgresSemanticIndex

DSN = os.environ.get("OMNIX_DATABASE_URL", "")
needs_pg_reason = "OMNIX_DATABASE_URL not set; needs live Postgres with pgvector"

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "semantic_golden_speeches.json"
_FIXTURE = json.loads(FIXTURE_PATH.read_text())

AXES: list[str] = _FIXTURE["axes"]
DOCS: list[dict] = _FIXTURE["docs"]
QUERIES: list[dict] = _FIXTURE["queries"]
QUERY_IDS = [q["id"] for q in QUERIES]
QUERY_BY_ID = {q["id"]: q for q in QUERIES}

#: Documented hit@k per class — see the module docstring for the rationale.
HIT_AT_K = {"exact": 1, "paraphrase": 3, "filtered": 2}

DIM = len(AXES)  # 8 — matches the vector(8) CI table (parity/postgres tests)
GOLDEN_MODEL = "golden-fake-embed"

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

_URI_BY_KG: dict[str, set[str]] = {}
for _d in DOCS:
    _URI_BY_KG.setdefault(_d["kg"], set()).add(_d["entity_uri"])


def _vec(topics: dict[str, float]) -> list[float]:
    """Topic-weight map → fixed 8-dim vector (the deterministic fake embed)."""
    unknown = set(topics) - set(AXES)
    assert not unknown, f"fixture topics reference unknown axes: {unknown}"
    return [float(topics.get(axis, 0.0)) for axis in AXES]


def _doc_triples(doc: dict) -> list[tuple[str, str, str]]:
    """The triples ingestion would write for one speech entity — typed, with a
    name for the denormalized label and the marked ``transcript`` text."""
    type_uri = f"https://cograph.tech/types/{doc['type']}"
    attr_base = f"{type_uri}/attrs"
    return [
        (doc["entity_uri"], RDF_TYPE, type_uri),
        (doc["entity_uri"], f"{attr_base}/name", doc["label"]),
        (doc["entity_uri"], f"{attr_base}/transcript", doc["text"]),
    ]


def _corpus_chunks(tenant: str) -> list[SemanticChunk]:
    """The whole fixture corpus through the REAL extractor (marker-driven +
    chunked), per KG. Deterministic for a given tenant."""
    chunks: list[SemanticChunk] = []
    for doc in DOCS:
        chunks.extend(
            extract_semantic_chunks(
                _doc_triples(doc),
                tenant_id=tenant,
                kg_name=doc["kg"],
                marked_predicates={"transcript"},
            )
        )
    return chunks


async def _seed(backend, tenant: str) -> list[SemanticChunk]:
    """Upsert the corpus, then fill embeddings through the embed-queue seam
    (fetch_pending → fill_embeddings) — every chunk of a doc gets the doc's
    topic vector, exactly how the ONTA-181 sweep would fill real vectors."""
    chunks = _corpus_chunks(tenant)
    await backend.upsert_chunks(chunks)
    doc_vec = {d["entity_uri"]: _vec(d["topics"]) for d in DOCS}
    pending = await backend.fetch_pending(limit=1_000, tenant_id=tenant)
    filled = await backend.fill_embeddings(
        pending,
        [doc_vec[c.entity_uri] for c in pending],
        embed_model=GOLDEN_MODEL,
    )
    assert filled == len(pending) == len(chunks), "corpus did not fully embed"
    return chunks


async def _run(backend, tenant: str, q: dict, *, degraded: bool = False):
    return await backend.search(
        tenant,
        q["query"],
        query_embedding=None if degraded else _vec(q["topics"]),
        kg_name=q.get("kg"),
        type_filter=q.get("type"),
        top_k=10,
    )


def _t() -> str:
    return f"golden-{uuid.uuid4().hex[:10]}"


@pytest.fixture(params=["memory", "postgres"])
async def backend(request):
    """Same golden corpus + assertions against both backends; the postgres
    param skips without a DSN (mirrors test_semantic_parity.py). Both use
    GOLDEN_MODEL as the current embed model, matching what _seed stamps —
    each backend's vector leg only trusts current-model vectors."""
    if request.param == "memory":
        yield InMemorySemanticIndex(embed_model=GOLDEN_MODEL)
        return
    if not DSN:
        pytest.skip(needs_pg_reason)
    from cograph_client.db.pool import close_pg_pools, reset_pg_pools

    reset_pg_pools()
    idx = PostgresSemanticIndex(dsn=DSN, embed_model=GOLDEN_MODEL, embed_dim=DIM)
    yield idx
    await close_pg_pools()
    reset_pg_pools()


# ---------------------------------------------------------------------------
# Fixture self-checks: the corpus really exercises the chunker (the guard on
# MIN/MAX_CHUNK_CHARS — if chunk sizing changes, THESE say whether the fixture
# still covers the multi-chunk path).
# ---------------------------------------------------------------------------


def test_fixture_long_docs_actually_split():
    long_docs = [d for d in DOCS if len(d["text"]) > MAX_CHUNK_CHARS]
    assert len(long_docs) >= 3, (
        "the fixture must keep >=3 docs longer than MAX_CHUNK_CHARS so the "
        "golden queries exercise multi-chunk documents"
    )
    chunks = _corpus_chunks("chk")
    assert len(chunks) > len(DOCS), "no doc split — chunking is not exercised"
    per_entity: dict[str, int] = {}
    for c in chunks:
        per_entity[c.entity_uri] = per_entity.get(c.entity_uri, 0) + 1
    for d in long_docs:
        assert per_entity[d["entity_uri"]] >= 2, (
            f"long doc {d['id']} ({len(d['text'])} chars) did not split"
        )


def test_second_chunk_quotes_really_live_past_the_first_chunk():
    """The two ``from_second_chunk`` exact quotes must sit in chunk_ix >= 1 —
    otherwise their golden assertion silently stops proving that LATER chunks
    of a long doc are searchable."""
    chunks = _corpus_chunks("chk")
    probes = [q for q in QUERIES if q.get("from_second_chunk")]
    assert probes, "fixture lost its second-chunk probe queries"
    for q in probes:
        holders = [
            c
            for c in chunks
            if c.entity_uri == q["expect"]
            and q["query"].lower() in c.chunk_text.lower()
        ]
        assert holders, f"{q['id']}: quote not found verbatim in any chunk"
        assert all(c.chunk_ix >= 1 for c in holders), (
            f"{q['id']}: quote landed in chunk 0 — lengthen the doc so the "
            "quote stays past the first chunk boundary"
        )


def test_dense_only_queries_share_no_token_with_their_target():
    """``dense_only`` queries must stay lexically invisible to their target BY
    CONSTRUCTION (zero shared tokens, stopwords included) — that property is
    what makes the 'dense must win' negative assertion below meaningful."""
    import re

    tok = re.compile(r"[a-z0-9]+")
    doc_by_uri = {d["entity_uri"]: d for d in DOCS}
    for q in QUERIES:
        if not q.get("dense_only"):
            continue
        target_tokens = set(tok.findall(doc_by_uri[q["expect"]]["text"].lower()))
        query_tokens = set(tok.findall(q["query"].lower()))
        overlap = query_tokens & target_tokens
        assert not overlap, f"{q['id']}: shares tokens {overlap} with its target"


# ---------------------------------------------------------------------------
# The golden gate: hit@k per query, per backend.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query_id", QUERY_IDS)
async def test_golden_query_hit_at_k(backend, query_id):
    q = QUERY_BY_ID[query_id]
    tenant = _t()
    await _seed(backend, tenant)

    res = await _run(backend, tenant, q)
    assert res.degraded is False

    uris = [h.entity_uri for h in res.hits]
    k = HIT_AT_K[q["class"]]
    assert q["expect"] in uris[:k], (
        f"{q['id']} [{q['class']}] on {type(backend).__name__}: expected "
        f"{q['expect']} in top-{k}, got {uris}"
    )

    # Filtered queries: nothing outside the requested scope may leak in.
    if q.get("type"):
        assert all(h.attrs.get("type") == q["type"] for h in res.hits), (
            f"{q['id']}: a hit escaped the type filter: "
            f"{[(h.entity_uri, h.attrs.get('type')) for h in res.hits]}"
        )
    if q.get("kg"):
        allowed = _URI_BY_KG[q["kg"]]
        assert set(uris) <= allowed, (
            f"{q['id']}: hits leaked across the kg filter: {set(uris) - allowed}"
        )


# ---------------------------------------------------------------------------
# Class-level properties: WHICH leg carries each class (the reason the three
# classes exist — a regression that flips a leg off would still pass hybrid
# hit@k for the other class, so each leg gets its own witness).
# ---------------------------------------------------------------------------


async def test_exact_quotes_are_carried_by_fts_alone(backend):
    """Exact quotes must hit@1 even LEXICAL-ONLY (no query embedding): the FTS
    leg alone finds the verbatim text. Combined with the orthogonal embedding
    in the hybrid run, this pins the class to the lexical leg entirely."""
    tenant = _t()
    await _seed(backend, tenant)
    for q in (q for q in QUERIES if q["class"] == "exact"):
        res = await _run(backend, tenant, q, degraded=True)
        assert res.degraded is True
        uris = [h.entity_uri for h in res.hits]
        assert uris and uris[0] == q["expect"], (
            f"{q['id']}: FTS alone should rank the quoted doc first on "
            f"{type(backend).__name__}, got {uris}"
        )


async def test_dense_only_paraphrases_need_the_vector_leg(backend):
    """The 'dense must win' witness: paraphrases sharing NO token with their
    target are unreachable lexically (absent from degraded results entirely)
    but hit@k once the query embedding is supplied. This is the assertion
    that fails if the vector leg (or the RRF fusion of it) regresses."""
    tenant = _t()
    await _seed(backend, tenant)
    dense_only = [q for q in QUERIES if q.get("dense_only")]
    assert len(dense_only) >= 3, "fixture lost its dense-only probes"
    for q in dense_only:
        lexical = await _run(backend, tenant, q, degraded=True)
        assert q["expect"] not in {h.entity_uri for h in lexical.hits}, (
            f"{q['id']}: target was reachable lexically — the dense-only "
            "probe no longer proves anything; fix the fixture wording"
        )
        hybrid = await _run(backend, tenant, q)
        uris = [h.entity_uri for h in hybrid.hits]
        k = HIT_AT_K["paraphrase"]
        assert q["expect"] in uris[:k], (
            f"{q['id']}: dense leg failed to lift the target into top-{k} on "
            f"{type(backend).__name__}, got {uris}"
        )
