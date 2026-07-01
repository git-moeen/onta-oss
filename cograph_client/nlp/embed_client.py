"""Shared OpenRouter embeddings client — the ONE place texts become vectors.

Extracted from the two byte-identical private copies that lived in
``ontology_embeddings.py`` (type chunks) and ``example_bank.py`` ((question,
SPARQL) pairs) so every embedding consumer — those two plus the semantic
instance index (ONTA-173) — shares a single model / batch-size / error
contract. Duplicated embed calls are the same drift risk the converged write
path (``graph/kg_writer.py``) exists to prevent: the moment one copy changes
model or batching, "similar" means different things in different subsystems.

Deliberately behavior-identical to the originals (ONTA-174 / PR 0): batches of
:data:`EMBEDDING_BATCH_SIZE`, a fresh ``httpx.AsyncClient`` per batch, 30s
timeout, no retries. Retry/backoff policy, if ever added, belongs HERE so all
consumers inherit it at once.
"""

from __future__ import annotations

import httpx
import numpy as np

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIM = 1536
EMBEDDING_BATCH_SIZE = 100


class EmbeddingError(RuntimeError):
    """Raised when the embedding API call fails.

    Subclasses ``RuntimeError`` for backward compatibility: ``example_bank``
    historically raised a bare ``RuntimeError`` here, so any caller catching
    that still works after the consolidation.
    """


async def embed_texts(
    texts: list[str], *, api_key: str, timeout: float = 30
) -> list[list[float]]:
    """Embed ``texts`` via the OpenRouter embeddings API, in batches.

    Returns one ``EMBEDDING_DIM``-length vector per input text, in order.
    Raises :class:`EmbeddingError` on any non-200 response.
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i : i + EMBEDDING_BATCH_SIZE]
        async with httpx.AsyncClient(timeout=timeout) as client:
            res = await client.post(
                OPENROUTER_EMBEDDINGS_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": EMBEDDING_MODEL, "input": batch},
            )
            if res.status_code != 200:
                raise EmbeddingError(
                    f"Embedding API returned {res.status_code}: {res.text}"
                )
            data = res.json()
            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)

    return all_embeddings


def cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between a query vector and a matrix of vectors."""
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(matrix.shape[0])
    matrix_norms = np.linalg.norm(matrix, axis=1)
    matrix_norms = np.where(matrix_norms == 0, 1, matrix_norms)
    return np.dot(matrix, query) / (matrix_norms * query_norm)
