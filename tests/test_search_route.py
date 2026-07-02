"""Route-level tests for the canonical semantic search (ONTA-178).

``POST /graphs/{tenant}/search`` is the ONE search surface every client
(webapp / CLI / MCP / SDK) rides, so this file locks the documented HTTP
contract end-to-end over the InMemory backend with fake embeddings:

* auth — same ``get_tenant`` dependency as every KG route: 401 without a key,
  403 for a multi-tenant key requesting an unowned path tenant, and the
  legacy single-tenant-key path-scoping behavior (results come from the KEY's
  tenant, never the foreign tenant named in the path);
* validation — blank query → 400; non-integer top_k → 422; out-of-range
  top_k clamped to [1, 50] with the effective value echoed back;
* unknown kg_name → 200 + empty hits (never a 404/500 — see the route docs);
* 503 with the ``COGRAPH_SEMANTIC_INDEX_ENABLED`` hint when the master gate
  is off (mirrors the reindex route);
* degraded shape — no embed key, or an embed failure, yields ``degraded=true``
  lexical-only results, never a 500;
* happy path — the route embeds the query via the shared embed client and
  passes the vector to the index (the index NEVER calls an embedding API —
  the locked ONTA-176 contract), and hits carry
  entity_uri / attrs / snippet / attr / score.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from cograph_client.auth.api_keys import register_external_verifier
from cograph_client.config import settings
from cograph_client.semantic.extract import content_hash
from cograph_client.semantic.memory import InMemorySemanticIndex
from cograph_client.semantic.protocol import SemanticChunk
from cograph_client.semantic.registry import (
    register_semantic_index,
    reset_semantic_index,
)

TENANT = "test-tenant"  # conftest's static-key tenant
KG = "kg1"
DIM = 4

V_SOLAR = [1.0, 0.0, 0.0, 0.0]
V_WIND = [0.0, 1.0, 0.0, 0.0]


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    # The gate is on for every test unless a test disables it explicitly.
    monkeypatch.setenv("COGRAPH_SEMANTIC_INDEX_ENABLED", "true")
    # No embed key by default — individual tests opt in to the embed path.
    monkeypatch.setattr(settings, "openrouter_api_key", "")
    reset_semantic_index()
    yield
    reset_semantic_index()
    register_external_verifier(None)


def _chunk(
    uri: str,
    text: str,
    *,
    kg: str = KG,
    tenant: str = TENANT,
    embedding: Optional[list[float]] = None,
    attrs: Optional[dict] = None,
) -> SemanticChunk:
    return SemanticChunk(
        tenant_id=tenant,
        kg_name=kg,
        entity_uri=uri,
        attr="description",
        chunk_ix=0,
        chunk_text=text,
        content_hash=content_hash(text),
        embedding=embedding,
        embed_model="fake-embed-model" if embedding is not None else None,
        attrs=attrs if attrs is not None else {"label": uri, "type": "Report"},
    )


def _seed(*chunks: SemanticChunk) -> InMemorySemanticIndex:
    index = InMemorySemanticIndex()
    register_semantic_index(index)
    asyncio.run(index.upsert_chunks(list(chunks)))
    return index


def _corpus() -> list[SemanticChunk]:
    return [
        _chunk(
            "e:solar",
            "Rooftop solar panel installation subsidies for residential homes.",
            embedding=V_SOLAR,
            attrs={"label": "Solar", "type": "Report"},
        ),
        _chunk(
            "e:wind",
            "Offshore wind turbine blade maintenance schedule.",
            embedding=V_WIND,
            attrs={"label": "Wind", "type": "Article"},
        ),
    ]


def _search(client, payload: dict, headers=None, tenant: str = TENANT):
    return client.post(f"/graphs/{tenant}/search", json=payload, headers=headers)


# --- auth ---------------------------------------------------------------------


def test_search_requires_auth(client):
    resp = _search(client, {"query": "solar"})
    assert resp.status_code in (401, 403)


def test_search_multi_tenant_key_403_for_unowned_tenant(client):
    """The cross-tenant 403 path, exactly as other routes get it from
    get_tenant: a user-scoped key granting [alpha, beta] must NOT search a
    tenant outside its grant — the key is valid, the tenant grant is not."""
    register_external_verifier(lambda key: ["alpha", "beta"])
    resp = _search(
        client,
        {"query": "solar"},
        headers={"X-API-Key": "multi-key"},  # not in the static map → verifier
        tenant="other-tenant",
    )
    assert resp.status_code == 403
    assert "other-tenant" in resp.json()["detail"]


def test_search_scopes_to_the_keys_tenant(client, auth_headers):
    """Legacy single-tenant static keys route to THEIR tenant regardless of the
    path (documented get_tenant behavior, same as the reindex route test): the
    search must run against the KEY's tenant — a foreign tenant named in the
    path must never leak its rows."""
    _seed(
        _chunk("e:mine", "confidential solar subsidies report", tenant=TENANT),
        _chunk("e:theirs", "confidential solar subsidies report", tenant="other-tenant"),
    )
    resp = _search(client, {"query": "confidential solar subsidies"},
                   headers=auth_headers, tenant="other-tenant")
    assert resp.status_code == 200
    uris = [h["entity_uri"] for h in resp.json()["hits"]]
    assert uris == ["e:mine"]


# --- gate + validation ----------------------------------------------------------


def test_search_503_when_semantic_index_disabled(monkeypatch, client, auth_headers):
    monkeypatch.delenv("COGRAPH_SEMANTIC_INDEX_ENABLED", raising=False)
    resp = _search(client, {"query": "solar"}, headers=auth_headers)
    assert resp.status_code == 503
    assert "COGRAPH_SEMANTIC_INDEX_ENABLED" in resp.json()["detail"]


@pytest.mark.parametrize("bad_query", ["", "   ", "\n\t "])
def test_search_blank_query_is_400(client, auth_headers, bad_query):
    """Documented choice: a blank query is a caller bug → 400, not empty 200."""
    _seed(*_corpus())
    resp = _search(client, {"query": bad_query}, headers=auth_headers)
    assert resp.status_code == 400
    assert "query" in resp.json()["detail"]


def test_search_missing_query_is_422(client, auth_headers):
    resp = _search(client, {}, headers=auth_headers)
    assert resp.status_code == 422


def test_search_non_integer_top_k_is_422(client, auth_headers):
    _seed(*_corpus())
    resp = _search(client, {"query": "solar", "top_k": "lots"}, headers=auth_headers)
    assert resp.status_code == 422


def test_search_top_k_clamped_to_cap(client, auth_headers):
    """Out-of-range top_k is clamped to [1, 50] and the EFFECTIVE value is
    echoed back (the documented, observable clamp)."""
    _seed(*_corpus())
    resp = _search(client, {"query": "solar", "top_k": 9999}, headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["top_k"] == 50

    resp = _search(client, {"query": "solar", "top_k": -3}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["top_k"] == 1
    assert len(body["hits"]) <= 1


def test_search_top_k_limits_hits(client, auth_headers):
    _seed(
        *[
            _chunk(f"e:{i}", f"solar panels variant {i}")
            for i in range(5)
        ]
    )
    resp = _search(client, {"query": "solar panels", "top_k": 2}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["hits"]) == 2
    assert body["count"] == 2
    assert body["top_k"] == 2


# --- kg / type scoping -----------------------------------------------------------


def test_search_unknown_kg_returns_empty_not_error(client, auth_headers):
    """Documented choice: an unknown KG is indistinguishable from an unindexed
    one — empty results, never a 404/500 (no Neptune existence round-trip)."""
    _seed(*_corpus())
    resp = _search(
        client, {"query": "solar", "kg_name": "no-such-kg"}, headers=auth_headers
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["hits"] == []
    assert body["count"] == 0


def test_search_empty_kg_name_means_all_kgs(client, auth_headers):
    """'' normalizes to None (all KGs) — a blank form field must not filter to
    a KG literally named the empty string."""
    _seed(_chunk("e:a", "solar text", kg="kga"), _chunk("e:b", "solar text", kg="kgb"))
    resp = _search(client, {"query": "solar", "kg_name": ""}, headers=auth_headers)
    assert {h["entity_uri"] for h in resp.json()["hits"]} == {"e:a", "e:b"}


def test_search_type_filter(client, auth_headers):
    _seed(*_corpus())
    resp = _search(
        client, {"query": "solar wind", "type": "Article"}, headers=auth_headers
    )
    assert resp.status_code == 200
    uris = {h["entity_uri"] for h in resp.json()["hits"]}
    assert uris == {"e:wind"}


# --- degraded shape ---------------------------------------------------------------


def test_search_degraded_without_embed_key(client, auth_headers):
    """No OpenRouter key → the route passes query_embedding=None → the backend
    runs lexical-only and the response says so (degraded=true, never silent)."""
    _seed(*_corpus())
    resp = _search(client, {"query": "solar panel subsidies"}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["degraded"] is True
    assert body["hits"][0]["entity_uri"] == "e:solar"


def test_search_degraded_on_embed_failure(monkeypatch, client, auth_headers):
    """Embedding outage degrades recall, it must never 500 the search."""
    _seed(*_corpus())
    monkeypatch.setattr(settings, "openrouter_api_key", "some-key")

    async def boom(texts, *, api_key, timeout=30):
        raise RuntimeError("embed service down")

    monkeypatch.setattr("cograph_client.nlp.embed_client.embed_texts", boom)
    resp = _search(client, {"query": "solar panel subsidies"}, headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["degraded"] is True
    assert body["hits"][0]["entity_uri"] == "e:solar"


# --- happy path -------------------------------------------------------------------


class _RecordingIndex(InMemorySemanticIndex):
    """InMemory index that records search kwargs — proves the route passed the
    query embedding DOWN (the index never embeds on its own; locked contract)."""

    def __init__(self) -> None:
        super().__init__()
        self.search_calls: list[dict] = []

    async def search(self, tenant_id, query_text, **kwargs):
        self.search_calls.append({"tenant_id": tenant_id, "query_text": query_text, **kwargs})
        return await super().search(tenant_id, query_text, **kwargs)


def test_search_happy_path_hybrid(monkeypatch, client, auth_headers):
    index = _RecordingIndex()
    register_semantic_index(index)
    asyncio.run(index.upsert_chunks(_corpus()))

    monkeypatch.setattr(settings, "openrouter_api_key", "some-key")
    embed_calls: list[list[str]] = []

    async def fake_embed(texts: list[str], *, api_key: str, timeout: float = 30):
        embed_calls.append(texts)
        return [V_SOLAR for _ in texts]

    monkeypatch.setattr("cograph_client.nlp.embed_client.embed_texts", fake_embed)

    resp = _search(
        client, {"query": "  solar panel subsidies  ", "top_k": 5}, headers=auth_headers
    )
    assert resp.status_code == 200
    body = resp.json()

    # Full hybrid: not degraded, best entity first, count/top_k echoed.
    assert body["degraded"] is False
    assert body["count"] == len(body["hits"]) >= 1
    assert body["top_k"] == 5
    top = body["hits"][0]
    assert top["entity_uri"] == "e:solar"
    assert top["attrs"] == {"label": "Solar", "type": "Report"}
    assert "solar" in top["snippet"].lower()
    assert top["attr"] == "description"
    assert top["score"] > 0

    # The ROUTE embedded the (stripped) query once via the shared client and
    # handed the vector to the index — the index never called an embedding API.
    assert embed_calls == [["solar panel subsidies"]]
    assert len(index.search_calls) == 1
    call = index.search_calls[0]
    assert call["tenant_id"] == TENANT
    assert call["query_text"] == "solar panel subsidies"
    assert call["query_embedding"] == V_SOLAR
    assert call["top_k"] == 5
