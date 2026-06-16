"""Tests for relationship registration during ingestion and type placement.

Covers:
1. Relationships between entities register object properties in the ontology
2. same_as maps to existing types instead of creating duplicates
3. parent_type creates subtype relationships
4. Extraction prompt includes existing types
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from cograph_client.resolver.schema_resolver import (
    SchemaResolver,
    EXTRACTION_SYSTEM,
    EXTRACTION_USER_TEMPLATE,
)
from cograph_client.resolver.models import (
    ExtractedEntity,
    ExtractedAttribute,
    ExtractedRelationship,
    ExtractionResult,
    IngestResult,
)
from cograph_client.resolver.verdict_cache import JsonVerdictCache
from cograph_client.graph.ontology_queries import type_uri


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_neptune():
    client = AsyncMock()
    client.query.return_value = {
        "head": {"vars": []},
        "results": {"bindings": []},
    }
    client.update.return_value = None
    return client


@pytest.fixture
def mock_cache(tmp_path):
    return JsonVerdictCache(tmp_path / "cache.json")


# ---------------------------------------------------------------------------
# 1. Extraction prompt includes existing types
# ---------------------------------------------------------------------------

class TestExtractionPrompt:
    def test_system_prompt_has_entity_first_principle(self):
        assert "Entity-first principle" in EXTRACTION_SYSTEM

    def test_system_prompt_has_type_placement(self):
        assert "same_as" in EXTRACTION_SYSTEM
        assert "parent_type" in EXTRACTION_SYSTEM

    def test_user_template_has_existing_types_placeholder(self):
        assert "{existing_types}" in EXTRACTION_USER_TEMPLATE

    def test_user_template_has_same_as_field(self):
        assert "same_as" in EXTRACTION_USER_TEMPLATE

    def test_user_template_has_parent_type_field(self):
        assert "parent_type" in EXTRACTION_USER_TEMPLATE


# ---------------------------------------------------------------------------
# 2. Type placement: same_as
# ---------------------------------------------------------------------------

class TestTypePlacementSameAs:
    @pytest.mark.asyncio
    async def test_same_as_maps_to_existing_type(self, mock_neptune, mock_cache):
        """When LLM sets same_as, entity uses the existing type name."""
        resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)

        extraction = ExtractionResult(
            entities=[
                ExtractedEntity(
                    type_name="Home",
                    id="123-main",
                    same_as="Property",
                    attributes=[ExtractedAttribute(name="price", value="500000", datatype="integer")],
                ),
            ],
            relationships=[],
        )

        existing_types = {"Property": ""}
        existing_attrs = {"Property": {}}

        with patch.object(resolver, "_extract", return_value=extraction):
            with patch.object(resolver, "_fetch_ontology", return_value=(existing_types, existing_attrs)):
                result = await resolver.ingest("A home at 123 Main", "test-tenant")

        # Should NOT have created a new "Home" type
        assert "Home" not in result.types_created
        # Entity URI should use "Property" not "Home"
        update_calls = [str(c) for c in mock_neptune.update.call_args_list]
        insert_calls = " ".join(update_calls)
        assert "entities/Property/" in insert_calls


# ---------------------------------------------------------------------------
# 3. Type placement: parent_type
# ---------------------------------------------------------------------------

class TestTypePlacementParentType:
    @pytest.mark.asyncio
    async def test_parent_type_creates_subtype(self, mock_neptune, mock_cache):
        """When LLM sets parent_type, a subClassOf triple is created."""
        resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)

        extraction = ExtractionResult(
            entities=[
                ExtractedEntity(
                    type_name="Condo",
                    id="456-oak",
                    parent_type="Property",
                    attributes=[ExtractedAttribute(name="hoa_fee", value="450", datatype="integer")],
                ),
            ],
            relationships=[],
        )

        existing_types = {"Property": ""}
        existing_attrs = {"Property": {}}

        with patch.object(resolver, "_extract", return_value=extraction):
            with patch.object(resolver, "_fetch_ontology", return_value=(existing_types, existing_attrs)):
                result = await resolver.ingest("A condo at 456 Oak", "test-tenant")

        assert "Condo" in result.types_created
        # Verify subClassOf was inserted
        update_calls = [str(c) for c in mock_neptune.update.call_args_list]
        insert_calls = " ".join(update_calls)
        assert "subClassOf" in insert_calls

    @pytest.mark.asyncio
    async def test_parent_type_ignored_if_not_in_ontology(self, mock_neptune, mock_cache):
        """If parent_type references a type not in ontology, skip subtype creation."""
        resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)

        extraction = ExtractionResult(
            entities=[
                ExtractedEntity(
                    type_name="Spaceship",
                    id="falcon-9",
                    parent_type="Vehicle",  # Vehicle doesn't exist
                    attributes=[],
                ),
            ],
            relationships=[],
        )

        existing_types = {"Property": ""}
        existing_attrs = {"Property": {}}

        with patch.object(resolver, "_extract", return_value=extraction):
            with patch.object(resolver, "_fetch_ontology", return_value=(existing_types, existing_attrs)):
                result = await resolver.ingest("A spaceship", "test-tenant")

        assert "Spaceship" in result.types_created
        # No subClassOf since Vehicle doesn't exist
        update_calls = [str(c) for c in mock_neptune.update.call_args_list]
        insert_calls = " ".join(update_calls)
        assert "subClassOf" not in insert_calls


# ---------------------------------------------------------------------------
# 4. Relationship registration as object properties
# ---------------------------------------------------------------------------

class TestRelationshipRegistration:
    @pytest.mark.asyncio
    async def test_relationship_registers_ontology_attribute(self, mock_neptune, mock_cache):
        """Relationships between entities should register as object properties in the ontology."""
        resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)

        extraction = ExtractionResult(
            entities=[
                ExtractedEntity(type_name="Person", id="john", attributes=[
                    ExtractedAttribute(name="name", value="John", datatype="string"),
                ]),
                ExtractedEntity(type_name="City", id="sf", attributes=[
                    ExtractedAttribute(name="name", value="San Francisco", datatype="string"),
                ]),
            ],
            relationships=[
                ExtractedRelationship(source_id="john", predicate="lives_in", target_id="sf"),
            ],
        )

        existing_types = {"Person": "", "City": ""}
        existing_attrs = {"Person": {}, "City": {}}

        with patch.object(resolver, "_extract", return_value=extraction):
            with patch.object(resolver, "_fetch_ontology", return_value=(existing_types, existing_attrs)):
                result = await resolver.ingest("John lives in SF", "test-tenant")

        # The relationship should be registered as an ontology attribute
        assert "Person.lives_in" in result.attributes_added

        # Verify insert_attribute was called with City as the datatype (range)
        update_calls = [str(c) for c in mock_neptune.update.call_args_list]
        insert_calls = " ".join(update_calls)
        assert "Person/attrs/lives_in" in insert_calls
        assert "cograph.tech/types/City" in insert_calls

    @pytest.mark.asyncio
    async def test_relationship_not_duplicated(self, mock_neptune, mock_cache):
        """If the relationship attribute already exists, don't re-register it."""
        resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)

        from cograph_client.resolver.attribute_resolver import AttributeSchema

        extraction = ExtractionResult(
            entities=[
                ExtractedEntity(type_name="Person", id="john", attributes=[]),
                ExtractedEntity(type_name="City", id="sf", attributes=[]),
            ],
            relationships=[
                ExtractedRelationship(source_id="john", predicate="lives_in", target_id="sf"),
            ],
        )

        existing_types = {"Person": "", "City": ""}
        existing_attrs = {
            "Person": {"lives_in": AttributeSchema("lives_in", "City")},
            "City": {},
        }

        with patch.object(resolver, "_extract", return_value=extraction):
            with patch.object(resolver, "_fetch_ontology", return_value=(existing_types, existing_attrs)):
                result = await resolver.ingest("John lives in SF", "test-tenant")

        # Should NOT re-register the attribute
        assert "Person.lives_in" not in result.attributes_added

    @pytest.mark.asyncio
    async def test_relationship_upgrades_primitive_attribute(self, mock_neptune, mock_cache):
        """A predicate first seen as a primitive attribute, then carrying an
        entity object, must have its ontology range UPGRADED to the target type.

        Regression: without the upgrade the predicate keeps its ``xsd:string``
        range, so the schema-only Explorer overview can't draw the edge even
        though the per-type detail view shows it from instance data. The two
        views disagreed (RetailerSKU → Product line missing in the overview).
        """
        resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)

        from cograph_client.resolver.attribute_resolver import AttributeSchema

        extraction = ExtractionResult(
            entities=[
                ExtractedEntity(type_name="Person", id="john", attributes=[]),
                ExtractedEntity(type_name="City", id="sf", attributes=[]),
            ],
            relationships=[
                ExtractedRelationship(source_id="john", predicate="lives_in", target_id="sf"),
            ],
        )

        existing_types = {"Person": "", "City": ""}
        # `lives_in` was previously registered as a primitive (string) attribute.
        existing_attrs = {
            "Person": {"lives_in": AttributeSchema("lives_in", "string")},
            "City": {},
        }

        with patch.object(resolver, "_extract", return_value=extraction):
            with patch.object(resolver, "_fetch_ontology", return_value=(existing_types, existing_attrs)):
                await resolver.ingest("John lives in SF", "test-tenant")

        update_calls = " ".join(str(c) for c in mock_neptune.update.call_args_list)
        # The range was re-pointed at the City type (delete-then-insert).
        assert "DELETE" in update_calls and "INSERT" in update_calls
        assert "Person/attrs/lives_in" in update_calls
        assert "cograph.tech/types/City" in update_calls

    @pytest.mark.asyncio
    async def test_instance_triple_always_inserted(self, mock_neptune, mock_cache):
        """Instance relationship triples should always be inserted regardless of ontology state."""
        resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)

        extraction = ExtractionResult(
            entities=[
                ExtractedEntity(type_name="Person", id="john", attributes=[]),
                ExtractedEntity(type_name="City", id="sf", attributes=[]),
            ],
            relationships=[
                ExtractedRelationship(source_id="john", predicate="lives_in", target_id="sf"),
            ],
        )

        existing_types = {"Person": "", "City": ""}
        existing_attrs = {"Person": {}, "City": {}}

        with patch.object(resolver, "_extract", return_value=extraction):
            with patch.object(resolver, "_fetch_ontology", return_value=(existing_types, existing_attrs)):
                result = await resolver.ingest("John lives in SF", "test-tenant")

        assert result.triples_inserted > 0
        update_calls = [str(c) for c in mock_neptune.update.call_args_list]
        insert_calls = " ".join(update_calls)
        assert "onto/lives_in" in insert_calls
