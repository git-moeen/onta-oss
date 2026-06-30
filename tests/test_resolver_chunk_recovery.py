"""FIX 1 — extraction truncation must not silently drop a whole chunk.

A 50-record chunk's reified JSON output can exceed the LLM ``max_tokens``, get
truncated, fail to parse, and return an EMPTY :class:`ExtractionResult` — so the
WHOLE batch vanishes, logged only as a warning, with no row-conservation
accounting. The fix makes a failed/empty JSON chunk RECOVER: split its array in
half and retry each half (down to a floor), accounting for any record still lost.

These tests drive ``SchemaResolver.ingest`` with a mocked ``_extract`` that fails
for a dense chunk but succeeds once the chunk is small enough, and assert:
  * NO records are lost (every record lands as an entity),
  * splitting actually occurred (the dense chunk was retried in halves),
  * a chunk that can NEVER be extracted is accounted for in IngestResult
    (rows_in / rows_dropped), not silently presented as complete.

Harness mirrors tests/test_multityping_retail.py: a bare AsyncMock Neptune with
``_extract`` / ``_fetch_ontology`` patched, no network.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from cograph_client.resolver.schema_resolver import SchemaResolver
from cograph_client.resolver.models import (
    ExtractedAttribute,
    ExtractedEntity,
    ExtractionResult,
)
from cograph_client.resolver.verdict_cache import JsonVerdictCache


@pytest.fixture
def mock_neptune():
    client = AsyncMock()
    client.query.return_value = {"head": {"vars": []}, "results": {"bindings": []}}
    client.update.return_value = None
    client.batch_exists.return_value = set()
    return client


@pytest.fixture
def mock_cache(tmp_path):
    return JsonVerdictCache(tmp_path / "cache.json")


def _entity_for(record: dict) -> ExtractedEntity:
    """One Model entity per record, keyed by the record's id."""
    return ExtractedEntity(
        type_name="Model",
        id=str(record["id"]),
        attributes=[ExtractedAttribute(name="name", value=record["name"], datatype="string")],
    )


def _make_records(n: int) -> list[dict]:
    return [{"id": i, "name": f"model_{i}"} for i in range(n)]


def _fake_extract_factory(success_max: int, calls: list[int]):
    """Build a fake ``_extract`` that records each chunk's record count and only
    SUCCEEDS when the chunk holds ``<= success_max`` records — emulating
    truncation: a denser chunk overflows the token cap and returns EMPTY."""

    async def fake_extract(content, content_type, existing_types=None):
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = []
        n = len(data) if isinstance(data, list) else 0
        calls.append(n)
        if 0 < n <= success_max:
            return ExtractionResult(entities=[_entity_for(r) for r in data], relationships=[])
        # Too dense → truncated → empty extraction (the silent-loss path).
        return ExtractionResult(entities=[], relationships=[])

    return fake_extract


@pytest.mark.asyncio
async def test_dense_chunk_recovers_by_splitting_no_records_lost(mock_neptune, mock_cache):
    """50 records → 2 chunks of 25 (default batch). Each 25-chunk fails, but a
    half (~12-13) succeeds: every record must still land, and a split must have
    occurred (a 25-record extraction attempt followed by smaller ones)."""
    resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)
    records = _make_records(50)
    content = json.dumps(records)

    calls: list[int] = []
    # Succeeds at <=13 records: the 25-record chunk fails, each ~12/13 half wins.
    fake_extract = _fake_extract_factory(success_max=13, calls=calls)

    with patch.object(resolver, "_extract", side_effect=fake_extract):
        with patch.object(resolver, "_fetch_ontology", return_value=({}, {})):
            result = await resolver.ingest(content, "test-tenant", content_type="json")

    # No record lost: all 50 records produced an entity.
    assert result.rows_in == 50
    assert result.rows_dropped == 0
    assert result.entities_extracted == 50
    assert result.entities_resolved == 50

    # Splitting actually occurred: at least one chunk was attempted at full size
    # (25) and then re-attempted at a smaller size (the halves).
    assert 25 in calls, calls
    assert any(0 < c < 25 for c in calls), calls


@pytest.mark.asyncio
async def test_unrecoverable_chunk_is_accounted_not_silently_dropped(mock_neptune, mock_cache):
    """If a chunk can NEVER be extracted (fails even at the minimum size), its
    records must surface in rows_dropped — the run is not presented as complete
    with the loss hidden."""
    resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)
    records = _make_records(50)
    content = json.dumps(records)

    calls: list[int] = []
    # success_max=0 → every extraction returns empty, even single-record floors.
    fake_extract = _fake_extract_factory(success_max=0, calls=calls)

    with patch.object(resolver, "_extract", side_effect=fake_extract):
        with patch.object(resolver, "_fetch_ontology", return_value=({}, {})):
            result = await resolver.ingest(content, "test-tenant", content_type="json")

    # Every record is accounted for as a drop — nothing landed, nothing hidden.
    assert result.rows_in == 50
    assert result.rows_dropped == 50
    assert result.entities_extracted == 0
    # Recursion bottomed out at the floor (chunks of <= 3 were attempted).
    assert any(0 < c <= SchemaResolver._RECOVERY_MIN_RECORDS for c in calls), calls


@pytest.mark.asyncio
async def test_healthy_chunks_do_not_split(mock_neptune, mock_cache):
    """When extraction succeeds at full chunk size, no splitting happens and the
    behavior is the plain per-chunk path (no spurious extra _extract calls)."""
    resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)
    records = _make_records(50)
    content = json.dumps(records)

    calls: list[int] = []
    fake_extract = _fake_extract_factory(success_max=25, calls=calls)

    with patch.object(resolver, "_extract", side_effect=fake_extract):
        with patch.object(resolver, "_fetch_ontology", return_value=({}, {})):
            result = await resolver.ingest(content, "test-tenant", content_type="json")

    assert result.rows_in == 50
    assert result.rows_dropped == 0
    assert result.entities_resolved == 50
    # Exactly two chunks attempted, both at full size — no recovery splits.
    assert calls == [25, 25], calls
