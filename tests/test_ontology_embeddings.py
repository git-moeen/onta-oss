"""Tests for the ontology embedding service."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from cograph_client.nlp.ontology_embeddings import (
    EmbeddingError,
    OntologyEmbeddingService,
    TenantEmbeddingStore,
    TypeChunk,
    _cosine_similarity,
    _extract_tenant_id,
    _parse_ontology_bindings,
)


GRAPH_URI = "https://cograph.tech/graphs/test-tenant"


def _make_fake_embedding(dim: int = 1536, seed: int = 0) -> list[float]:
    """Create a deterministic fake embedding vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def _make_ontology_bindings_raw(types_info: dict[str, list[str]]) -> list[dict]:
    """Create raw SPARQL bindings (pre-parse_sparql_results format)."""
    bindings = []
    for type_name, attrs in types_info.items():
        for attr in attrs:
            bindings.append({
                "type": {"type": "uri", "value": f"https://cograph.tech/types/{type_name}"},
                "typeLabel": {"type": "literal", "value": type_name},
                "attrLabel": {"type": "literal", "value": attr},
                "range": {"type": "uri", "value": "http://www.w3.org/2001/XMLSchema#string"},
            })
        if not attrs:
            bindings.append({
                "type": {"type": "uri", "value": f"https://cograph.tech/types/{type_name}"},
                "typeLabel": {"type": "literal", "value": type_name},
            })
    return bindings


def _make_ontology_bindings(types_info: dict[str, list[str]]) -> list[dict]:
    """Create parsed SPARQL bindings (post-parse_sparql_results format, plain string values)."""
    bindings = []
    for type_name, attrs in types_info.items():
        for attr in attrs:
            bindings.append({
                "type": f"https://cograph.tech/types/{type_name}",
                "typeLabel": type_name,
                "attrLabel": attr,
                "range": "http://www.w3.org/2001/XMLSchema#string",
            })
        if not attrs:
            bindings.append({
                "type": f"https://cograph.tech/types/{type_name}",
                "typeLabel": type_name,
            })
    return bindings


def _make_service() -> OntologyEmbeddingService:
    return OntologyEmbeddingService(openrouter_api_key="fake-key", s3_bucket="", s3_prefix="test")


def _make_embedding_response(texts: list[str], seed_offset: int = 0) -> dict:
    """Create a fake OpenRouter embedding API response."""
    return {
        "data": [
            {"embedding": _make_fake_embedding(seed=i + seed_offset)}
            for i in range(len(texts))
        ]
    }


def _mock_httpx_post(embedding_response: dict):
    """Create a mock for httpx.AsyncClient.post that returns embedding response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = embedding_response
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


# ── Unit tests for helpers ────────────────────────────────────────────


class TestHelpers:
    def test_cosine_similarity(self):
        q = np.array([1.0, 0.0, 0.0])
        m = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])
        sims = _cosine_similarity(q, m)
        assert sims[0] == pytest.approx(1.0)
        assert sims[1] == pytest.approx(0.0)
        assert sims[2] > 0

    def test_cosine_similarity_zero_query(self):
        q = np.array([0.0, 0.0, 0.0])
        m = np.array([[1.0, 0.0, 0.0]])
        sims = _cosine_similarity(q, m)
        assert sims[0] == 0.0

    def test_extract_tenant_id(self):
        assert _extract_tenant_id("https://cograph.tech/graphs/my-tenant") == "my-tenant"
        assert _extract_tenant_id("https://cograph.tech/graphs/t1/kg/zillow") == "t1"

    def test_parse_ontology_bindings(self):
        bindings = _make_ontology_bindings({"Property": ["price", "address"], "City": ["name"]})
        types = _parse_ontology_bindings(bindings)
        assert "Property" in types
        assert "City" in types
        assert len(types["Property"]["attributes"]) == 2
        assert len(types["City"]["attributes"]) == 1


# ── Core service tests ────────────────────────────────────────────────


class TestBuildFromOntology:
    @pytest.mark.asyncio
    async def test_build_from_ontology(self):
        """Full build: mock Neptune + embedding API, verify store populated."""
        svc = _make_service()
        mock_neptune = AsyncMock()
        mock_neptune.query.return_value = {
            "head": {"vars": ["type", "typeLabel", "attrLabel", "range"]},
            "results": {"bindings": _make_ontology_bindings_raw({"Property": ["price", "beds"], "City": ["name"]})},
        }

        mock_client = _mock_httpx_post(_make_embedding_response(["chunk1", "chunk2"]))
        with patch("cograph_client.nlp.embed_client.httpx.AsyncClient", return_value=mock_client):
            count = await svc.build_from_ontology(GRAPH_URI, mock_neptune)

        assert count == 2
        assert GRAPH_URI in svc._stores
        assert "Property" in svc._stores[GRAPH_URI].chunks
        assert "City" in svc._stores[GRAPH_URI].chunks
        assert svc._stores[GRAPH_URI].chunks["Property"].embedding.shape == (1536,)

    @pytest.mark.asyncio
    async def test_build_empty_ontology(self):
        """Empty ontology returns 0."""
        svc = _make_service()
        mock_neptune = AsyncMock()
        mock_neptune.query.return_value = {
            "head": {"vars": []},
            "results": {"bindings": []},
        }
        count = await svc.build_from_ontology(GRAPH_URI, mock_neptune)
        assert count == 0


class TestRetrieve:
    def _make_store_with_types(self, svc: OntologyEmbeddingService, type_configs: dict):
        """Populate a store with types that have specific embedding directions.

        type_configs: {"TypeName": {"seed": N, "targets": [...]}}
        """
        store = TenantEmbeddingStore()
        for tn, config in type_configs.items():
            emb = np.array(_make_fake_embedding(seed=config.get("seed", 0)), dtype=np.float32)
            store.chunks[tn] = TypeChunk(
                type_name=tn,
                chunk_text=f"Type: {tn}",
                embedding=emb,
                attributes=config.get("attributes", []),
                relationship_targets=config.get("targets", []),
            )
        svc._stores[GRAPH_URI] = store

    @pytest.mark.asyncio
    async def test_retrieve_top_k(self):
        """Verify top-K types returned by similarity."""
        svc = _make_service()
        # Create types with known embeddings
        self._make_store_with_types(svc, {
            "PropertyListing": {"seed": 1},
            "Broker": {"seed": 2},
            "City": {"seed": 3},
            "Vehicle": {"seed": 100},
            "Invoice": {"seed": 200},
        })

        # Mock question embedding to be close to PropertyListing (seed=1)
        question_emb = _make_fake_embedding(seed=1)  # same as PropertyListing
        mock_client = _mock_httpx_post({"data": [{"embedding": question_emb}]})

        with patch("cograph_client.nlp.embed_client.httpx.AsyncClient", return_value=mock_client):
            result = await svc.retrieve(GRAPH_URI, "How many properties?", top_k=2)

        assert result is not None
        assert "PropertyListing" in result
        # Should not include all 5 types
        lines = result.strip().split("\n")
        type_count = sum(1 for l in lines if l.startswith("Type:"))
        assert type_count <= 4  # top-2 + possible expansion

    @pytest.mark.asyncio
    async def test_retrieve_1hop_expansion(self):
        """Type A relates to B, query matches A, verify B included."""
        svc = _make_service()
        self._make_store_with_types(svc, {
            "PropertyListing": {"seed": 1, "targets": ["Broker"]},
            "Broker": {"seed": 50},
            "UnrelatedType": {"seed": 200},
        })

        question_emb = _make_fake_embedding(seed=1)
        mock_client = _mock_httpx_post({"data": [{"embedding": question_emb}]})

        with patch("cograph_client.nlp.embed_client.httpx.AsyncClient", return_value=mock_client):
            result = await svc.retrieve(GRAPH_URI, "properties", top_k=1)

        assert result is not None
        assert "PropertyListing" in result
        assert "Broker" in result  # 1-hop expansion

    @pytest.mark.asyncio
    async def test_retrieve_no_embeddings_returns_none(self):
        """No store, no S3, returns None."""
        svc = _make_service()
        result = await svc.retrieve(GRAPH_URI, "anything")
        assert result is None

    @pytest.mark.asyncio
    async def test_retrieve_cold_start_s3(self):
        """No in-memory store, load from S3."""
        svc = OntologyEmbeddingService(openrouter_api_key="fake-key", s3_bucket="my-bucket", s3_prefix="test")

        embedding = _make_fake_embedding(seed=1)
        s3_data = json.dumps({
            "Property": {
                "chunk_text": "Type: Property",
                "embedding": embedding,
                "attributes": [],
                "relationship_targets": [],
            }
        }).encode()

        question_emb = _make_fake_embedding(seed=1)
        mock_client = _mock_httpx_post({"data": [{"embedding": question_emb}]})

        with patch.object(svc, "_s3_get", return_value=s3_data):
            with patch("cograph_client.nlp.embed_client.httpx.AsyncClient", return_value=mock_client):
                result = await svc.retrieve(GRAPH_URI, "property", top_k=5)

        assert result is not None
        assert "Property" in result


class TestEmbedTypesIncremental:
    @pytest.mark.asyncio
    async def test_embed_types_incremental(self):
        """Build with A,B then add C. Verify store has all three."""
        svc = _make_service()

        # Pre-populate store with A, B
        store = TenantEmbeddingStore()
        for tn, seed in [("TypeA", 1), ("TypeB", 2)]:
            store.chunks[tn] = TypeChunk(
                type_name=tn,
                chunk_text=f"Type: {tn}",
                embedding=np.array(_make_fake_embedding(seed=seed), dtype=np.float32),
            )
        svc._stores[GRAPH_URI] = store

        # Mock Neptune to return all types including C
        mock_neptune = AsyncMock()
        mock_neptune.query.return_value = {
            "head": {"vars": ["type", "typeLabel", "attrLabel", "range"]},
            "results": {"bindings": _make_ontology_bindings_raw({
                "TypeA": ["a1"], "TypeB": ["b1"], "TypeC": ["c1"],
            })},
        }

        mock_client = _mock_httpx_post(_make_embedding_response(["TypeC"], seed_offset=10))
        with patch("cograph_client.nlp.embed_client.httpx.AsyncClient", return_value=mock_client):
            await svc.embed_types(GRAPH_URI, ["TypeC"], mock_neptune)

        assert "TypeA" in svc._stores[GRAPH_URI].chunks
        assert "TypeB" in svc._stores[GRAPH_URI].chunks
        assert "TypeC" in svc._stores[GRAPH_URI].chunks


class TestSafetyValve:
    @pytest.mark.asyncio
    async def test_safety_valve_large_type(self):
        """Type with 250 attrs gets filtered."""
        svc = _make_service()

        # Create type with 250 attributes
        attrs = [f"attr_{i} (string)" for i in range(250)]
        store = TenantEmbeddingStore()
        store.chunks["BigType"] = TypeChunk(
            type_name="BigType",
            chunk_text="Type: BigType\n  Attributes: " + ", ".join(attrs),
            embedding=np.array(_make_fake_embedding(seed=1), dtype=np.float32),
            attributes=attrs,
        )
        svc._stores[GRAPH_URI] = store

        # Mock: question embedding close to BigType, then attr embeddings
        question_emb = _make_fake_embedding(seed=1)
        attr_embs = [_make_fake_embedding(seed=i) for i in range(250)]
        call_count = 0

        def make_response(*args, **kwargs):
            nonlocal call_count
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            if call_count == 0:
                mock_resp.json.return_value = {"data": [{"embedding": question_emb}]}
            else:
                # Return embeddings for attributes (may be batched)
                body = kwargs.get("json", args[1] if len(args) > 1 else {})
                n = len(body.get("input", []))
                mock_resp.json.return_value = {
                    "data": [{"embedding": _make_fake_embedding(seed=i)} for i in range(n)]
                }
            call_count += 1
            return mock_resp

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=make_response)

        with patch("cograph_client.nlp.embed_client.httpx.AsyncClient", return_value=mock_client):
            result = await svc.retrieve(GRAPH_URI, "test query", top_k=5)

        assert result is not None
        # Result should have filtered attributes, not all 250
        attr_count = result.count("attr_")
        assert attr_count <= 50


class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_embedding_api_failure(self):
        """API error raises EmbeddingError."""
        svc = _make_service()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("cograph_client.nlp.embed_client.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(EmbeddingError, match="500"):
                await svc._embed_texts(["test"])

    @pytest.mark.asyncio
    async def test_s3_save_failure_non_blocking(self):
        """S3 failure during save doesn't crash; embeddings stay in memory."""
        svc = OntologyEmbeddingService(openrouter_api_key="fake-key", s3_bucket="my-bucket", s3_prefix="test")

        # Build store
        mock_neptune = AsyncMock()
        mock_neptune.query.return_value = {
            "head": {"vars": ["type", "typeLabel", "attrLabel", "range"]},
            "results": {"bindings": _make_ontology_bindings_raw({"TypeA": ["a1"]})},
        }
        mock_client = _mock_httpx_post(_make_embedding_response(["TypeA"]))

        with patch("cograph_client.nlp.embed_client.httpx.AsyncClient", return_value=mock_client):
            with patch.object(svc, "_s3_put", side_effect=Exception("S3 down")):
                count = await svc.build_from_ontology(GRAPH_URI, mock_neptune)

        # Should succeed despite S3 failure
        assert count == 1
        assert "TypeA" in svc._stores[GRAPH_URI].chunks


class TestInvalidate:
    def test_invalidate(self):
        """Invalidate clears the store."""
        svc = _make_service()
        svc._stores[GRAPH_URI] = TenantEmbeddingStore()
        svc._stores[GRAPH_URI].chunks["A"] = TypeChunk(
            type_name="A",
            chunk_text="Type: A",
            embedding=np.zeros(1536, dtype=np.float32),
        )
        svc.invalidate(GRAPH_URI)
        assert GRAPH_URI not in svc._stores
