"""FIX 3 + FIX 4 — subtype_description belongs ONLY on subtype branches, and is
written idempotently.

``ExtractedEntity.subtype_description`` defines a NEW SUBTYPE (models.py): it must
become the new type's ``rdfs:comment`` ONLY when the type is minted as a subtype.
The bug passed it into ``insert_type`` on the DIFFERENT / FLAGGED / same_as-rejected
(top-level) branches too. FIX 3 restricts it to the subtype branches; FIX 4 writes
it via ``upsert_type`` (single-valued comment REPLACE) so re-minting a type across
ingests can't accumulate duplicate comments.

Harness mirrors tests/test_resolver_relationships.py: bare AsyncMock Neptune with
a fake type-matcher and ``_extract`` / ``_fetch_ontology`` patched. We assert on
the SPARQL strings sent to ``neptune.update``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from cograph_client.resolver.schema_resolver import SchemaResolver
from cograph_client.resolver.models import (
    ExtractedEntity,
    ExtractionResult,
    MatchVerdict,
    TypeMatch,
)
from cograph_client.resolver.verdict_cache import JsonVerdictCache


DESC = "a score measuring how human a generated voice sounds"
RDFS_COMMENT = "http://www.w3.org/2000/01/rdf-schema#comment"


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


class FakeTypeMatcher:
    """Returns one canned TypeMatch for every proposed type."""

    def __init__(self, verdict: MatchVerdict, parent_type: str | None = None):
        self._verdict = verdict
        self._parent_type = parent_type

    async def match(self, proposed_type, proposed_description, existing_types):
        return TypeMatch(
            proposed=proposed_type,
            resolved=proposed_type,
            verdict=self._verdict,
            confidence=0.9,
            is_new=self._verdict != MatchVerdict.SAME,
            parent_type=self._parent_type,
        )


def _update_strings(mock_neptune) -> list[str]:
    return [str(c.args[0]) if c.args else str(c) for c in mock_neptune.update.call_args_list]


def _comment_writes_for(mock_neptune, type_name: str, description: str) -> list[str]:
    """Every update string that writes ``description`` as the rdfs:comment of
    ``types/<type_name>``."""
    needle_type = f"https://cograph.tech/types/{type_name}>"
    out = []
    for s in _update_strings(mock_neptune):
        if RDFS_COMMENT in s and description in s and needle_type in s:
            out.append(s)
    return out


async def _ingest_one(resolver, entity: ExtractedEntity, existing_types=None):
    existing_types = existing_types or {}
    existing_attrs = {t: {} for t in existing_types}
    extraction = ExtractionResult(entities=[entity], relationships=[])
    with patch.object(resolver, "_extract", return_value=extraction):
        with patch.object(
            resolver, "_fetch_ontology", return_value=(dict(existing_types), existing_attrs)
        ):
            return await resolver.ingest("data", "test-tenant")


# ---------------------------------------------------------------------------
# Subtype branches WRITE the description (via upsert)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subtype_branch_writes_description(mock_neptune, mock_cache):
    """match.verdict == SUBTYPE → the description IS written as the new type's
    rdfs:comment."""
    resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)
    resolver._type_matcher = FakeTypeMatcher(MatchVerdict.SUBTYPE, parent_type="Score")

    entity = ExtractedEntity(
        type_name="HumannessIndex", id="hi-1", subtype_description=DESC,
    )
    result = await _ingest_one(resolver, entity, existing_types={"Score": ""})

    assert "HumannessIndex" in result.types_created
    writes = _comment_writes_for(mock_neptune, "HumannessIndex", DESC)
    assert writes, "subtype branch must write the subtype_description as rdfs:comment"


@pytest.mark.asyncio
async def test_subtype_branch_uses_upsert_not_blind_insert(mock_neptune, mock_cache):
    """FIX 4: the comment write is an UPSERT (DELETE-then-INSERT the single-valued
    comment), not a blind INSERT DATA — so re-ingest can't accumulate duplicates."""
    resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)
    resolver._type_matcher = FakeTypeMatcher(MatchVerdict.SUBTYPE, parent_type="Score")

    entity = ExtractedEntity(type_name="HumannessIndex", id="hi-1", subtype_description=DESC)
    await _ingest_one(resolver, entity, existing_types={"Score": ""})

    writes = _comment_writes_for(mock_neptune, "HumannessIndex", DESC)
    assert writes
    # upsert_type emits a DELETE ... INSERT ... WHERE for the single-valued
    # comment; a blind insert_type would be "INSERT DATA { ... rdfs:comment ... }".
    assert any("DELETE" in w and "WHERE" in w for w in writes), (
        "subtype description must be written with upsert (DELETE/INSERT/WHERE) "
        "semantics for idempotency"
    )


@pytest.mark.asyncio
async def test_brand_new_lineage_via_parent_chain_writes_description(mock_neptune, mock_cache):
    """A DIFFERENT verdict but the entity carries a parent_chain → _link_parent
    mints it as a subtype, so the description IS written (there, via upsert)."""
    resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)
    resolver._type_matcher = FakeTypeMatcher(MatchVerdict.DIFFERENT)

    entity = ExtractedEntity(
        type_name="Condo", id="c-1",
        parent_chain=["Property", "Asset"],
        subtype_description="a privately owned unit in a multi-unit building",
    )
    result = await _ingest_one(resolver, entity)

    assert "Condo" in result.types_created
    writes = _comment_writes_for(
        mock_neptune, "Condo", "a privately owned unit in a multi-unit building"
    )
    assert writes, "a type linked into a lineage via parent_chain must carry its description"
    assert any("DELETE" in w and "WHERE" in w for w in writes), "must be upsert (idempotent)"


# ---------------------------------------------------------------------------
# Top-level branches must NOT write the description
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_top_level_different_does_not_write_description(mock_neptune, mock_cache):
    """A genuinely-new TOP-LEVEL type (DIFFERENT, no parent_chain) must NOT write
    subtype_description — the field only describes a subtype (FIX 3)."""
    resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)
    resolver._type_matcher = FakeTypeMatcher(MatchVerdict.DIFFERENT)

    # subtype_description present but this is a top-level type → must be ignored.
    entity = ExtractedEntity(type_name="Spaceship", id="ss-1", subtype_description=DESC)
    result = await _ingest_one(resolver, entity)

    assert "Spaceship" in result.types_created
    assert _comment_writes_for(mock_neptune, "Spaceship", DESC) == [], (
        "top-level DIFFERENT branch must not write subtype_description"
    )


@pytest.mark.asyncio
async def test_flagged_top_level_does_not_write_description(mock_neptune, mock_cache):
    """A FLAGGED type with no parent linkage must NOT write subtype_description."""
    resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)
    resolver._type_matcher = FakeTypeMatcher(MatchVerdict.FLAGGED)

    entity = ExtractedEntity(type_name="Widget", id="w-1", subtype_description=DESC)
    result = await _ingest_one(resolver, entity)

    assert "Widget" in result.flagged_types
    assert _comment_writes_for(mock_neptune, "Widget", DESC) == [], (
        "FLAGGED top-level branch must not write subtype_description"
    )


@pytest.mark.asyncio
async def test_same_as_rejected_does_not_write_description(mock_neptune, mock_cache):
    """same_as claimed but REJECTED (verdict DIFFERENT) → a genuine top-level
    type; subtype_description must not be written (FIX 3)."""
    resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)
    # entity.same_as names an existing type, but the matcher rejects the claim.
    resolver._type_matcher = FakeTypeMatcher(MatchVerdict.DIFFERENT)

    entity = ExtractedEntity(
        type_name="Gadget", id="g-1", same_as="Widget", subtype_description=DESC,
    )
    result = await _ingest_one(resolver, entity, existing_types={"Widget": ""})

    assert "Gadget" in result.types_created
    assert _comment_writes_for(mock_neptune, "Gadget", DESC) == [], (
        "same_as-rejected (top-level) branch must not write subtype_description"
    )
