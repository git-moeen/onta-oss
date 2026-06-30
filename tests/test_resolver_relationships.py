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

    def test_system_prompt_has_domain_modeling_guidance(self):
        """The new domain-modeling blocks must be present (Cause 2): reify
        measurements, lift providers/orgs, subtypes with a description."""
        assert "Reify measurements" in EXTRACTION_SYSTEM
        assert "Lift providers / organizations" in EXTRACTION_SYSTEM
        assert "Subtypes with a description" in EXTRACTION_SYSTEM
        # The guidance names the concrete signals the resolver downstream relies on.
        assert "subtype_description" in EXTRACTION_SYSTEM
        assert "Organization" in EXTRACTION_SYSTEM

    def test_user_template_has_subtype_description_field(self):
        assert "subtype_description" in EXTRACTION_USER_TEMPLATE


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


# ---------------------------------------------------------------------------
# 5. Domain modeling: reified measurements, lifted orgs, described subtypes
# ---------------------------------------------------------------------------


class TestDomainModeling:
    """The extraction pipeline (Cause 2) must normalize a web/document ingest into
    a richer ontology — a Model, a lifted Organization, a Score REIFIED as its own
    entity, and a HumannessIndex SUBTYPE of Score carrying an rdfs:comment — rather
    than flattening everything onto one type. _extract is mocked, so this asserts
    the RESOLVE→INSERT behavior the strengthened prompt is meant to drive."""

    @staticmethod
    def _new_type_matcher(monkeypatch, resolver):
        """Force every proposed type to resolve as genuinely-new (DIFFERENT),
        hermetically — no LLM, no embeddings. Real behavior for brand-new types;
        subtyping is then driven by parent_chain in _link_parent."""
        from cograph_client.resolver.models import MatchVerdict, TypeMatch

        async def fake_match(proposed_type, proposed_description, existing_types):
            return TypeMatch(
                proposed=proposed_type, resolved=proposed_type,
                verdict=MatchVerdict.DIFFERENT, confidence=1.0, is_new=True,
            )

        monkeypatch.setattr(resolver._type_matcher, "match", fake_match)

    @pytest.mark.asyncio
    async def test_json_ingest_reifies_score_lifts_org_and_describes_subtype(
        self, mock_neptune, mock_cache, monkeypatch,
    ):
        resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)
        self._new_type_matcher(monkeypatch, resolver)

        DESC = "a score measuring how human a generated voice sounds"
        # The normalized shape the strengthened extractor is meant to produce from
        # a "TTS models and their scores" leaderboard ingest.
        extraction = ExtractionResult(
            entities=[
                ExtractedEntity(
                    type_name="Model", id="eleven-v3",
                    attributes=[ExtractedAttribute(name="modality", value="tts")],
                ),
                ExtractedEntity(
                    type_name="Organization", id="ElevenLabs",
                    attributes=[ExtractedAttribute(name="homepage", value="elevenlabs.io")],
                ),
                ExtractedEntity(
                    type_name="HumannessIndex", id="eleven-v3-humanness",
                    parent_chain=["Score"],
                    subtype_description=DESC,
                    attributes=[
                        ExtractedAttribute(name="value", value="87.5", datatype="float"),
                        ExtractedAttribute(
                            name="timestamp", value="2026-06-01T00:00:00Z",
                            datatype="datetime",
                        ),
                    ],
                ),
            ],
            relationships=[
                ExtractedRelationship(
                    source_id="eleven-v3", predicate="has_score",
                    target_id="eleven-v3-humanness",
                ),
                ExtractedRelationship(
                    source_id="eleven-v3-humanness", predicate="provided_by",
                    target_id="ElevenLabs",
                ),
            ],
        )

        content = json.dumps([{"name": "Eleven v3", "humanness": "87.5"}])
        with patch.object(resolver, "_extract", return_value=extraction):
            with patch.object(resolver, "_fetch_ontology", return_value=({}, {})):
                result = await resolver.ingest(content, "test-tenant", content_type="json")

        update_calls = " ".join(str(c) for c in mock_neptune.update.call_args_list)

        # --- types: Model, Organization, Score (synthesized parent) + the
        #     HumannessIndex subtype were all created.
        for t in ("Model", "Organization", "Score", "HumannessIndex"):
            assert t in result.types_created, f"{t} not created: {result.types_created}"

        # --- the subtype edge HumannessIndex subClassOf Score.
        assert (
            f"<{type_uri('HumannessIndex')}> "
            "<http://www.w3.org/2000/01/rdf-schema#subClassOf> "
            f"<{type_uri('Score')}>"
        ) in update_calls

        # --- HumannessIndex carries the description as an rdfs:comment, threaded
        #     through insert_type(graph, name, subtype_description). Score (the
        #     synthesized ancestor) gets NO comment — only the minted subtype does.
        assert (
            f"<{type_uri('HumannessIndex')}> "
            f'<http://www.w3.org/2000/01/rdf-schema#comment> "{DESC}"'
        ) in update_calls
        assert (
            f"<{type_uri('Score')}> "
            "<http://www.w3.org/2000/01/rdf-schema#comment>"
        ) not in update_calls

        # --- relationships became object properties (entity→entity edges), not
        #     scalar attributes: has_score (Model→HumannessIndex) and provided_by
        #     (HumannessIndex→Organization).
        assert "Model.has_score" in result.attributes_added
        assert "HumannessIndex.provided_by" in result.attributes_added
        assert "onto/has_score" in update_calls
        assert "onto/provided_by" in update_calls
        # provided_by's ontology range points at the Organization type.
        assert (
            f"HumannessIndex/attrs/provided_by" in update_calls
            and type_uri("Organization") in update_calls
        )

        # --- the reified Score's measurement values are TYPED literals on the
        #     HumannessIndex entity (value:float, timestamp:datetime) — the
        #     measurement is an entity with its own attributes, not a bare scalar.
        assert (
            'HumannessIndex/attrs/value> "87.5"'
            "^^<http://www.w3.org/2001/XMLSchema#float>"
        ) in update_calls
        # (the validator coerces the datetime to xsd form, dropping the 'Z').
        assert (
            "HumannessIndex/attrs/timestamp> "
            '"2026-06-01T00:00:00"'
            "^^<http://www.w3.org/2001/XMLSchema#dateTime>"
        ) in update_calls

        # --- all three entities resolved (Model, Organization, HumannessIndex).
        assert result.entities_resolved == 3

    @pytest.mark.asyncio
    async def test_plain_json_ingest_without_domain_signals_still_works(
        self, mock_neptune, mock_cache, monkeypatch,
    ):
        """Regression: an ordinary ingest whose extraction sets NO
        subtype_description / parent_chain / reified measurement still ingests
        cleanly — the new field + prompt must not break the common case."""
        resolver = SchemaResolver(mock_neptune, "fake-key", mock_cache)
        self._new_type_matcher(monkeypatch, resolver)

        extraction = ExtractionResult(
            entities=[
                ExtractedEntity(
                    type_name="Article", id="a1",
                    attributes=[ExtractedAttribute(name="title", value="Hello")],
                ),
            ],
            relationships=[],
        )

        content = json.dumps([{"title": "Hello"}])
        with patch.object(resolver, "_extract", return_value=extraction):
            with patch.object(resolver, "_fetch_ontology", return_value=({}, {})):
                result = await resolver.ingest(content, "test-tenant", content_type="json")

        assert "Article" in result.types_created
        assert result.entities_resolved == 1
        # The minted type carries NO comment (subtype_description defaulted None →
        # insert_type got "" → no rdfs:comment triple).
        update_calls = " ".join(str(c) for c in mock_neptune.update.call_args_list)
        assert (
            f"<{type_uri('Article')}> "
            "<http://www.w3.org/2000/01/rdf-schema#comment>"
        ) not in update_calls


# ---------------------------------------------------------------------------
# 6. ExtractedEntity model round-trips the new subtype_description field
# ---------------------------------------------------------------------------


class TestExtractedEntityModel:
    def test_subtype_description_round_trips(self):
        desc = "a score measuring how human a generated voice sounds"
        e = ExtractedEntity(
            type_name="HumannessIndex",
            id="x",
            parent_chain=["Score"],
            subtype_description=desc,
            attributes=[ExtractedAttribute(name="value", value="9.1", datatype="float")],
        )
        dumped = e.model_dump()
        assert dumped["subtype_description"] == desc
        # Round-trips back through validation unchanged.
        again = ExtractedEntity(**dumped)
        assert again.subtype_description == desc
        assert again.parent_chain == ["Score"]

    def test_subtype_description_defaults_none(self):
        e = ExtractedEntity(type_name="Article", id="a1")
        assert e.subtype_description is None
        # Parses from a payload that omits the field entirely (back-compat).
        from_payload = ExtractedEntity(**{"type_name": "Article", "id": "a1"})
        assert from_payload.subtype_description is None
