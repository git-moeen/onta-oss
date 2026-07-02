"""Smoke-parity suite: InMemory vs Postgres semantic index (ONTA-175/176).

The two backends deliberately use DIFFERENT leg scorers (naive term-frequency
vs ts_rank_cd; numpy cosine vs pgvector ``<=>``), so parity is **loose by
design**: the same scenario corpus runs against both and we assert *result-set
overlap* — the must-hit entity present, top-1 agreement where the corpus makes
it deterministic — never ranking equality.

The memory half runs everywhere (no DSN required); the Postgres half is gated
on ``OMNIX_DATABASE_URL`` and skips without it (CI provides a
``pgvector/pgvector:pg16`` service container). Both halves also carry the
security-grade tenant-isolation checks, RRF determinism with fake embeddings,
and the NULL-embedding-rows-are-FTS-only property.
"""

from __future__ import annotations

import os
import uuid

import pytest

from cograph_client.semantic import (
    InMemorySemanticIndex,
    SemanticChunk,
    content_hash,
)
from cograph_client.semantic.postgres import PostgresSemanticIndex

DSN = os.environ.get("OMNIX_DATABASE_URL", "")

KG = "kg1"
OTHER_KG = "kg2"
FAKE_MODEL = "fake-embed-model"
DIM = 8


def _vec(hot: int, *, lead: float = 1.0) -> list[float]:
    v = [0.0] * DIM
    v[hot] = lead
    return v


V_SOLAR = _vec(0)
V_WIND = _vec(1)
V_MED = _vec(2)
V_FIN = _vec(3)
V_MISC = _vec(4)


def _chunk(
    uri: str,
    text: str,
    *,
    tenant: str,
    ix: int = 0,
    attr: str = "description",
    kg: str = KG,
    embedding: list[float] | None = None,
    attrs: dict | None = None,
) -> SemanticChunk:
    return SemanticChunk(
        tenant_id=tenant,
        kg_name=kg,
        entity_uri=uri,
        attr=attr,
        chunk_ix=ix,
        chunk_text=text,
        content_hash=content_hash(text),
        embedding=embedding,
        # The Postgres ANN leg only trusts vectors from the current model.
        embed_model=FAKE_MODEL if embedding is not None else None,
        attrs=attrs if attrs is not None else {"label": uri},
    )


def _corpus(tenant: str) -> list[SemanticChunk]:
    """The shared scenario corpus: four embedded entities with distinct
    topics + one NULL-embedding (queued) entity that only the FTS leg can
    reach. Every scenario query below uses terms that ALL appear in its
    must-hit doc, because Postgres ``websearch_to_tsquery`` ANDs terms while
    the memory scorer ORs them — parity needs the intersection semantics."""
    return [
        _chunk(
            "e:solar",
            "Rooftop solar panel installation for residential homes; "
            "photovoltaic efficiency in cold climates.",
            tenant=tenant,
            embedding=V_SOLAR,
            attrs={"label": "Solar", "type": "Report"},
        ),
        _chunk(
            "e:wind",
            "Wind turbine maintenance schedule and blade inspection "
            "procedures for offshore farms.",
            tenant=tenant,
            embedding=V_WIND,
            attrs={"label": "Wind", "type": "Report"},
        ),
        _chunk(
            "e:med",
            "Cardiac arrhythmia treatment guidelines and heart rhythm "
            "disorder care.",
            tenant=tenant,
            embedding=V_MED,
            attrs={"label": "Med", "type": "Article"},
        ),
        _chunk(
            "e:fin",
            "Quarterly financial report covering revenue growth and "
            "operating margins.",
            tenant=tenant,
            embedding=V_FIN,
            attrs={"label": "Fin", "type": "Article"},
        ),
        _chunk(
            "e:pending",
            "Supply chain resilience strategies for semiconductor shortages.",
            tenant=tenant,
            embedding=None,  # queued: reachable through the FTS leg only
            attrs={"label": "Pending", "type": "Report"},
        ),
    ]


#: (query_text, query_embedding, must_hit, expect_top1)
#: expect_top1=False for e:pending: on the SQL backend a NULL-embedding
#: entity's FTS-rank-1 score (1/61) exactly ties an ANN-rank-1 noise entity
#: (pgvector ranks zero-similarity rows; the memory backend drops them), so
#: only membership — not the top spot — is portable. Loose parity by design.
SCENARIOS = [
    ("solar panel efficiency", [0.9, 0.1, 0, 0, 0, 0, 0, 0], "e:solar", True),
    ("wind turbine blade inspection", V_WIND, "e:wind", True),
    ("heart rhythm disorder", V_MED, "e:med", True),
    ("revenue growth margins", V_FIN, "e:fin", True),
    ("supply chain resilience semiconductor", V_MISC, "e:pending", False),
]


def _t() -> str:
    return f"t-{uuid.uuid4().hex[:10]}"


needs_pg_reason = "OMNIX_DATABASE_URL not set; needs live Postgres with pgvector"


@pytest.fixture(params=["memory", "postgres"])
async def backend(request):
    """The SAME tests run against both backends; the postgres param skips
    without a DSN (so the memory half is the no-DSN unit layer)."""
    if request.param == "memory":
        yield InMemorySemanticIndex()
        return
    if not DSN:
        pytest.skip(needs_pg_reason)
    from cograph_client.db.pool import close_pg_pools, reset_pg_pools

    reset_pg_pools()
    idx = PostgresSemanticIndex(dsn=DSN, embed_model=FAKE_MODEL, embed_dim=DIM)
    yield idx
    await close_pg_pools()
    reset_pg_pools()


def _uris(result) -> set[str]:
    return {h.entity_uri for h in result.hits}


# ---------------------------------------------------------------------------
# Per-backend: the scenario corpus behaves (runs on memory without a DSN)
# ---------------------------------------------------------------------------


async def test_scenarios_hit_per_backend(backend):
    tenant = _t()
    await backend.upsert_chunks(_corpus(tenant))
    for query, emb, must_hit, expect_top1 in SCENARIOS:
        res = await backend.search(tenant, query, query_embedding=emb, top_k=5)
        assert res.degraded is False
        uris = [h.entity_uri for h in res.hits]
        assert must_hit in uris, f"{query!r} missed {must_hit} on {type(backend).__name__}"
        if expect_top1:
            assert uris[0] == must_hit, f"{query!r} top-1 on {type(backend).__name__}"


async def test_degraded_lexical_only_per_backend(backend):
    tenant = _t()
    await backend.upsert_chunks(_corpus(tenant))
    res = await backend.search(tenant, "solar panel efficiency")
    assert res.degraded is True
    assert "e:solar" in _uris(res)
    # An embedding restores full hybrid mode.
    res = await backend.search(
        tenant, "solar panel efficiency", query_embedding=V_SOLAR
    )
    assert res.degraded is False


async def test_null_embedding_rows_fts_only_per_backend(backend):
    tenant = _t()
    await backend.upsert_chunks(_corpus(tenant))
    # Lexically reachable while queued…
    res = await backend.search(tenant, "supply chain resilience semiconductor")
    assert "e:pending" in _uris(res)
    # …but invisible to a pure-vector probe (no lexical overlap, any vector).
    res = await backend.search(tenant, "zzz qqq nothing", query_embedding=V_MISC)
    assert "e:pending" not in _uris(res)


async def test_tenant_isolation_security_per_backend(backend):
    """Security-grade: tenant A must NEVER see tenant B's rows, even when the
    text (and embedding) are byte-identical."""
    tenant_a, tenant_b = _t(), _t()
    await backend.upsert_chunks(
        [
            _chunk("e:a", "confidential shared wording", tenant=tenant_a, embedding=V_SOLAR),
            _chunk("e:b", "confidential shared wording", tenant=tenant_b, embedding=V_SOLAR),
        ]
    )
    res_a = await backend.search(
        tenant_a, "confidential shared wording", query_embedding=V_SOLAR
    )
    res_b = await backend.search(
        tenant_b, "confidential shared wording", query_embedding=V_SOLAR
    )
    assert _uris(res_a) == {"e:a"}
    assert _uris(res_b) == {"e:b"}
    # The queue seam is scoped too.
    await backend.upsert_chunks(
        [_chunk("e:queued-a", "only tenant a pending", tenant=tenant_a)]
    )
    pending_b = await backend.fetch_pending(tenant_id=tenant_b)
    assert all(c.tenant_id == tenant_b for c in pending_b)


async def test_kg_isolation_per_backend(backend):
    tenant = _t()
    await backend.upsert_chunks(
        [
            _chunk("e:a", "common topic text", tenant=tenant, kg=KG),
            _chunk("e:b", "common topic text", tenant=tenant, kg=OTHER_KG),
        ]
    )
    assert _uris(await backend.search(tenant, "common topic")) == {"e:a", "e:b"}
    assert _uris(await backend.search(tenant, "common topic", kg_name=KG)) == {"e:a"}
    await backend.clear(tenant, kg_name=KG)
    assert _uris(await backend.search(tenant, "common topic")) == {"e:b"}


async def test_type_filter_per_backend(backend):
    tenant = _t()
    await backend.upsert_chunks(_corpus(tenant))
    res = await backend.search(
        tenant, "heart rhythm disorder", query_embedding=V_MED, type_filter="Article"
    )
    assert "e:med" in _uris(res)
    res = await backend.search(
        tenant, "heart rhythm disorder", query_embedding=V_MED, type_filter="Report"
    )
    assert "e:med" not in _uris(res)


async def test_rrf_determinism_with_fake_embeddings_per_backend(backend):
    """Identical text + identical embeddings on two entities: both backends
    must (a) return identical output across repeated runs and (b) break the
    tie the same way — leg ranks tie-break on the PK, so the lower entity_uri
    ranks first and wins the higher fused score."""
    tenant = _t()
    await backend.upsert_chunks(
        [
            _chunk("e:tie:a", "duplicate ranking text", tenant=tenant, embedding=V_SOLAR),
            _chunk("e:tie:b", "duplicate ranking text", tenant=tenant, embedding=V_SOLAR),
        ]
    )
    runs = []
    for _ in range(3):
        res = await backend.search(
            tenant, "duplicate ranking text", query_embedding=V_SOLAR, top_k=5
        )
        runs.append([(h.entity_uri, round(h.score, 12)) for h in res.hits])
    assert runs[0] == runs[1] == runs[2]
    assert [uri for uri, _s in runs[0]] == ["e:tie:a", "e:tie:b"]
    assert runs[0][0][1] > runs[0][1][1]  # rank-1 fused score strictly higher


async def test_top_k_and_grouping_per_backend(backend):
    tenant = _t()
    # Two chunks of one doc must collapse into ONE entity hit.
    doc = "solar panels part one\n\nsolar panels part two"
    await backend.upsert_chunks(
        [
            _chunk("e:multi", "solar panels part one", tenant=tenant, ix=0),
            SemanticChunk(
                tenant_id=tenant,
                kg_name=KG,
                entity_uri="e:multi",
                attr="description",
                chunk_ix=1,
                chunk_text="solar panels part two",
                content_hash=content_hash(doc),
                attrs={"label": "e:multi"},
            ),
        ]
    )
    await backend.upsert_chunks(
        [_chunk(f"e:{i}", f"solar panels variant {i}", tenant=tenant) for i in range(6)]
    )
    res = await backend.search(tenant, "solar panels", top_k=3)
    assert len(res.hits) == 3
    all_res = await backend.search(tenant, "solar panels", top_k=50)
    assert len([h for h in all_res.hits if h.entity_uri == "e:multi"]) == 1


# ---------------------------------------------------------------------------
# Cross-backend smoke parity (needs the DSN — compares the two directly)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not DSN, reason=needs_pg_reason)
@pytest.mark.integration
async def test_smoke_parity_memory_vs_postgres():
    """Loose top-k overlap between the backends on the SAME corpus: the
    must-hit entity in both, top-1 agreement where deterministic, and a
    non-empty intersection of the top-3 sets. Never ranking equality."""
    from cograph_client.db.pool import close_pg_pools, reset_pg_pools

    reset_pg_pools()
    mem = InMemorySemanticIndex()
    pg = PostgresSemanticIndex(dsn=DSN, embed_model=FAKE_MODEL, embed_dim=DIM)
    try:
        tenant = _t()
        corpus = _corpus(tenant)
        await mem.upsert_chunks(corpus)
        await pg.upsert_chunks(corpus)
        for query, emb, must_hit, expect_top1 in SCENARIOS:
            res_mem = await mem.search(tenant, query, query_embedding=emb, top_k=5)
            res_pg = await pg.search(tenant, query, query_embedding=emb, top_k=5)
            mem_uris = [h.entity_uri for h in res_mem.hits]
            pg_uris = [h.entity_uri for h in res_pg.hits]
            assert must_hit in mem_uris, f"memory missed {must_hit} for {query!r}"
            assert must_hit in pg_uris, f"postgres missed {must_hit} for {query!r}"
            if expect_top1:
                assert mem_uris[0] == pg_uris[0] == must_hit, query
            assert set(mem_uris[:3]) & set(pg_uris[:3]), (
                f"no top-3 overlap for {query!r}: {mem_uris[:3]} vs {pg_uris[:3]}"
            )
    finally:
        await close_pg_pools()
        reset_pg_pools()
