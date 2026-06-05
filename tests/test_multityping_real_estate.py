"""Multi-typing tests for the real_estate domain: Condo < Property.

Covers two ADR-mandated behaviours:

A) INGESTION
   - A Condo instance is stamped with its leaf type (Condo), not the ancestor.
   - Ancestor synthesis creates Property in the ontology when Condo is first seen.
   - ER fires on Condo because config_for_with_hierarchy walks the chain and finds
     DEFAULT_PROPERTY_CONFIG on Property — two Condos at the same address merge
     (REVIEW or AUTO_MERGE) rather than minting two distinct URIs.
   - If ER were still flat (config_for('Condo') == None), the merge would not happen;
     asserting it DOES happen proves the chain-walk is load-bearing.

B) QUERYING (subclass closure seam)
   - rewrite_type_predicate_to_closure turns a query for Property into one that
     covers Condo instances (rdfs:subClassOf* in the predicate path).
   - with_subclass_closure returns the correct property-path string.
   - parent_map_query generates valid SPARQL for fetching the hierarchy.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch

# ---------------------------------------------------------------------------
# Helpers / imports under test
# ---------------------------------------------------------------------------

from cograph_client.resolver.er.types import (
    DEFAULT_PROPERTY_CONFIG,
    ERConfig,
    MergeAction,
    MergeDecision,
    MatchScore,
    NormalizedSignals,
    ancestor_chain,
    config_for,
    config_for_with_hierarchy,
    primary_type,
)
from cograph_client.graph.ontology_queries import (
    parent_map_query,
    rewrite_type_predicate_to_closure,
    type_uri,
    with_subclass_closure,
)
from cograph_client.resolver.models import ExtractedAttribute, ExtractedEntity, IngestResult

GRAPH_URI = "https://cograph.tech/graphs/test-tenant"
TYPES_URI = "https://cograph.tech/types/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUB = "http://www.w3.org/2000/01/rdf-schema#subClassOf"


# ---------------------------------------------------------------------------
# Pure-function tests for the hierarchy helpers (no I/O needed)
# ---------------------------------------------------------------------------


class TestAncestorChain:
    def test_condo_inherits_property(self):
        parent_of = {"Condo": "Property"}
        chain = ancestor_chain("Condo", parent_of)
        assert chain == ["Condo", "Property"]

    def test_leaf_only_when_no_hierarchy(self):
        chain = ancestor_chain("Condo", {})
        assert chain == ["Condo"]

    def test_deeper_chain(self):
        parent_of = {"LuxuryCondo": "Condo", "Condo": "Property"}
        chain = ancestor_chain("LuxuryCondo", parent_of)
        assert chain == ["LuxuryCondo", "Condo", "Property"]


class TestConfigForWithHierarchy:
    """Proves the load-bearing ER fix: Condo inherits Property's config."""

    def test_flat_config_for_returns_none_for_condo(self):
        # Condo is NOT in DEFAULTS_BY_TYPE — flat lookup returns None.
        assert config_for("Condo") is None

    def test_hierarchy_lookup_finds_property_config(self):
        parent_of = {"Condo": "Property"}
        cfg = config_for_with_hierarchy("Condo", parent_of)
        # Must resolve to the exact DEFAULT_PROPERTY_CONFIG object.
        assert cfg is DEFAULT_PROPERTY_CONFIG

    def test_hierarchy_lookup_returns_none_without_parent_map(self):
        cfg = config_for_with_hierarchy("Condo", {})
        assert cfg is None

    def test_property_config_resolves_directly(self):
        cfg = config_for_with_hierarchy("Property", {})
        assert cfg is DEFAULT_PROPERTY_CONFIG

    def test_deep_chain_inherits_property_config(self):
        parent_of = {"LuxuryCondo": "Condo", "Condo": "Property"}
        cfg = config_for_with_hierarchy("LuxuryCondo", parent_of)
        assert cfg is DEFAULT_PROPERTY_CONFIG


class TestPrimaryType:
    def test_condo_is_leaf_over_property(self):
        parent_of = {"Condo": "Property"}
        # When both are asserted, Condo dominates Property (it's more specific).
        pt = primary_type(["Property", "Condo"], parent_of)
        assert pt == "Condo"

    def test_single_type_returned_as_is(self):
        assert primary_type(["Condo"], {}) == "Condo"

    def test_empty_returns_none(self):
        assert primary_type([], {}) is None


# ---------------------------------------------------------------------------
# Querying / subclass-closure seam tests
# ---------------------------------------------------------------------------


class TestSubclassClosure:
    def test_with_subclass_closure_returns_property_path(self):
        path = with_subclass_closure("Property")
        # Must contain rdf:type and rdfs:subClassOf* in the correct property-path form.
        assert "subClassOf>*" in path
        assert "rdf-syntax-ns#type" in path or "#type" in path

    def test_rewrite_form_a_bare_a(self):
        sparql = "SELECT ?x WHERE { ?x a <https://cograph.tech/types/Property> . }"
        rewritten = rewrite_type_predicate_to_closure(sparql)
        assert "subClassOf>*" in rewritten
        assert "<https://cograph.tech/types/Property>" in rewritten
        # The bare `a` predicate should be gone (replaced by closure path).
        # Check closure path appears exactly once.
        assert rewritten.count("subClassOf>*") == 1

    def test_rewrite_form_b_full_rdf_type(self):
        sparql = (
            "SELECT ?x WHERE { "
            f"?x <{RDF_TYPE}> <https://cograph.tech/types/Property> . "
            "}"
        )
        rewritten = rewrite_type_predicate_to_closure(sparql)
        assert "subClassOf>*" in rewritten
        assert "<https://cograph.tech/types/Property>" in rewritten

    def test_rewrite_covers_condo_query_for_property_ancestor(self):
        """A query for Property type must become a closure path so Condo instances
        (which are asserted as Condo, not Property) are returned by Neptune."""
        sparql = "SELECT ?unit WHERE { ?unit a <https://cograph.tech/types/Property> . }"
        rewritten = rewrite_type_predicate_to_closure(sparql)
        # The object URI is preserved verbatim.
        assert "<https://cograph.tech/types/Property>" in rewritten
        # The predicate becomes the closure path.
        assert "subClassOf>*" in rewritten

    def test_rewrite_idempotent(self):
        sparql = "SELECT ?x WHERE { ?x a <https://cograph.tech/types/Property> . }"
        once = rewrite_type_predicate_to_closure(sparql)
        twice = rewrite_type_predicate_to_closure(once)
        assert once == twice

    def test_parent_map_query_contains_subclassof(self):
        q = parent_map_query(GRAPH_URI)
        assert "subClassOf" in q
        assert GRAPH_URI in q
        assert "?child" in q
        assert "?parent" in q

    def test_rewrite_does_not_touch_non_cograph_type_triples(self):
        """Triples whose object is NOT under cograph.tech/types/ must be left alone."""
        sparql = "SELECT ?x WHERE { ?x a <https://schema.org/Person> . }"
        rewritten = rewrite_type_predicate_to_closure(sparql)
        assert rewritten == sparql  # unchanged


# ---------------------------------------------------------------------------
# Ingestion tests — mocked Neptune, SchemaResolver._resolve_type path
# ---------------------------------------------------------------------------


def _make_neptune_mock():
    """Return an AsyncMock Neptune client with sensible defaults."""
    client = AsyncMock()
    client.health.return_value = True
    client.query.return_value = {
        "head": {"vars": []},
        "results": {"bindings": []},
    }
    client.update.return_value = None
    client.batch_exists = AsyncMock(return_value=set())
    return client


def _condo_entity(address: str, unit: str | None = None) -> ExtractedEntity:
    attrs = [
        ExtractedAttribute(name="address", value=address, datatype="string"),
        ExtractedAttribute(name="unit_number", value=unit or "1A", datatype="string"),
        ExtractedAttribute(name="price", value="450000", datatype="integer"),
    ]
    return ExtractedEntity(
        type_name="Condo",
        id=address,
        same_as=None,
        parent_type="Property",
        attributes=attrs,
    )


@pytest.fixture
def mock_neptune():
    return _make_neptune_mock()


# ---------------------------------------------------------------------------
# A-1: Leaf type asserted — rdf:type triple points to Condo, not Property
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ingestion_stamps_leaf_type_condo(mock_neptune):
    """The rdf:type triple written to Neptune must reference Condo, not Property."""
    from cograph_client.resolver.schema_resolver import SchemaResolver
    from cograph_client.resolver.verdict_cache import JsonVerdictCache

    cache = JsonVerdictCache.__new__(JsonVerdictCache)
    cache._path = None
    cache._cache = {}

    resolver = SchemaResolver(mock_neptune, "fake-key", cache)
    resolver._er_enabled = False  # isolate; ER tested separately

    entity = _condo_entity("123 Main St San Francisco CA 94105")

    # Simulate: Property already exists, Condo is new (subtype path)
    existing_types = {"Property": ""}
    existing_attrs = {"Property": {}}
    result = IngestResult(entities_extracted=1)

    with patch.object(resolver._type_matcher, "match") as mock_match:
        from cograph_client.resolver.models import MatchVerdict, TypeMatch
        mock_match.return_value = TypeMatch(
            proposed="Condo",
            resolved="Condo",
            verdict=MatchVerdict.SUBTYPE,
            confidence=0.95,
            is_new=True,
            parent_type="Property",
        )
        resolved_type = await resolver._resolve_type(
            entity, GRAPH_URI, existing_types, existing_attrs, result
        )

    # The resolved type must be the LEAF (Condo), not the ancestor.
    assert resolved_type == "Condo"


# ---------------------------------------------------------------------------
# A-2: Ancestor synthesis inserts Property into ontology
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ancestor_synthesis_creates_parent_type(mock_neptune):
    """_synthesize_ancestors must call neptune.update for any missing ancestor."""
    from cograph_client.resolver.schema_resolver import SchemaResolver
    from cograph_client.resolver.verdict_cache import JsonVerdictCache

    cache = JsonVerdictCache.__new__(JsonVerdictCache)
    cache._path = None
    cache._cache = {}

    resolver = SchemaResolver(mock_neptune, "fake-key", cache)

    # Property does NOT exist yet — synthesis must create it.
    existing_types: dict = {}
    existing_attrs: dict = {}
    result = IngestResult(entities_extracted=1)

    await resolver._synthesize_ancestors(
        "Condo", "Property", GRAPH_URI, existing_types, existing_attrs, result
    )

    # neptune.update must have been called (to insert Property type).
    assert mock_neptune.update.call_count >= 1

    # Property must appear in existing_types (resolver registers it).
    assert "Property" in existing_types

    # Property must be in result.types_created.
    assert "Property" in result.types_created


@pytest.mark.asyncio
async def test_ancestor_synthesis_idempotent_when_parent_exists(mock_neptune):
    """If Property is already in existing_types, no redundant Neptune writes."""
    from cograph_client.resolver.schema_resolver import SchemaResolver
    from cograph_client.resolver.verdict_cache import JsonVerdictCache

    cache = JsonVerdictCache.__new__(JsonVerdictCache)
    cache._path = None
    cache._cache = {}

    resolver = SchemaResolver(mock_neptune, "fake-key", cache)

    existing_types = {"Property": ""}
    existing_attrs = {"Property": {}}
    result = IngestResult(entities_extracted=1)

    await resolver._synthesize_ancestors(
        "Condo", "Property", GRAPH_URI, existing_types, existing_attrs, result
    )

    # Property already present → synthesis should NOT re-insert it.
    # update() may still be called for the subtype edge itself (Condo->Property),
    # but Property type INSERT must not appear again.
    # The key check: Property must NOT appear in result.types_created.
    assert "Property" not in result.types_created


# ---------------------------------------------------------------------------
# A-3: ER chain-walk — two Condos at the same address merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_er_fires_for_condo_via_property_config():
    """config_for_with_hierarchy('Condo', {'Condo':'Property'}) must return a config.

    This is the precondition for ER to fire on Condo. If it were still flat,
    config_for('Condo') == None and ER would be skipped — two identical Condos
    would get distinct URIs, violating deduplication.
    """
    parent_of = {"Condo": "Property"}
    cfg = config_for_with_hierarchy("Condo", parent_of)
    assert cfg is not None, (
        "config_for_with_hierarchy must return DEFAULT_PROPERTY_CONFIG for Condo "
        "when parent_of={'Condo':'Property'}; flat config_for('Condo') is None "
        "which would silently skip ER."
    )
    assert cfg is DEFAULT_PROPERTY_CONFIG


@pytest.mark.asyncio
async def test_er_pipeline_merges_duplicate_condos(mock_neptune):
    """Two Condo records at the same normalized address must merge (REVIEW or AUTO_MERGE).

    The test bypasses the full resolver and calls ERPipeline.find_match directly
    so we can control the blocker response (returning the first Condo as a candidate).
    Without chain-walk this test would see SKIP (no config found for Condo).
    """
    from cograph_client.resolver.er.engine import ERPipeline
    from cograph_client.resolver.er.normalize import DefaultNormalizer
    from cograph_client.resolver.er.types import NormalizedSignals

    address = "123 main st san francisco ca 94105"

    entity1 = _condo_entity("123 Main St San Francisco CA 94105", unit="1A")
    entity2 = _condo_entity("123 Main St San Francisco CA 94105", unit="1A")

    canonical_uri = f"https://cograph.tech/entities/Condo/123_Main_St"

    # Normalize the address the same way the normalizer would so signals match.
    normalizer = DefaultNormalizer()
    from cograph_client.resolver.er.engine import extract_signals
    raw1 = extract_signals(entity1)
    norm1 = normalizer.normalize(raw1)

    # Build a candidate mock that returns entity1's signals for entity2's lookup.
    mock_neptune_er = _make_neptune_mock()
    pipeline = ERPipeline(mock_neptune_er)

    # Patch candidates_with_signals to simulate entity1 already in the graph.
    pipeline._blocker.candidates_with_signals = AsyncMock(
        return_value={canonical_uri: norm1}
    )

    parent_of = {"Condo": "Property"}
    cfg = config_for_with_hierarchy("Condo", parent_of)
    assert cfg is not None  # guard: chain-walk must produce a config

    type_uri_str = f"{TYPES_URI}Condo"
    decision = await pipeline.find_match(
        entity2, "Condo", type_uri_str, GRAPH_URI,
        config=cfg,
        parent_of=parent_of,
    )

    # With address as the only decisive-free signal, the score will land in
    # the REVIEW band (>= review_threshold=0.80) or AUTO_MERGE (>= 0.95).
    # Either proves ER fired — a flat config_for would have returned SKIP.
    assert decision.action in (MergeAction.REVIEW, MergeAction.AUTO_MERGE), (
        f"Expected REVIEW or AUTO_MERGE but got {decision.action}. "
        f"Best score: {decision.best_match.score if decision.best_match else 'N/A'}. "
        "If SKIP: ER did not fire, meaning config_for_with_hierarchy returned None."
    )


@pytest.mark.asyncio
async def test_er_skips_condo_without_parent_map():
    """Contrast test: without the parent map, config is None and ER is skipped."""
    from cograph_client.resolver.er.engine import ERPipeline

    entity = _condo_entity("123 Main St San Francisco CA 94105")
    mock_neptune_er = _make_neptune_mock()
    pipeline = ERPipeline(mock_neptune_er)

    # No parent_of — flat config lookup, Condo not in DEFAULTS_BY_TYPE.
    decision = await pipeline.find_match(
        entity, "Condo",
        f"{TYPES_URI}Condo",
        GRAPH_URI,
        config=None,
        parent_of={},  # empty map → flat behavior → config_for('Condo') == None
    )
    assert decision.action == MergeAction.SKIP, (
        "Without parent_of, config_for_with_hierarchy('Condo', {}) is None "
        "and ER must return SKIP — proving the chain-walk is what enables merging."
    )


# ---------------------------------------------------------------------------
# B: Querying — subclass closure on generated SPARQL
# ---------------------------------------------------------------------------


class TestQueryClosureForRealEstate:
    """Prove the query-time closure seam works for the Property/Condo hierarchy."""

    def test_property_query_rewritten_to_include_subtypes(self):
        """A raw SPARQL query for Property instances gets rewritten so Condo
        instances (typed as Condo, not Property) are also returned by Neptune."""
        raw_sparql = (
            "SELECT ?unit ?price WHERE {\n"
            "  ?unit a <https://cograph.tech/types/Property> .\n"
            "  ?unit <https://cograph.tech/types/Property/attrs/price> ?price .\n"
            "}"
        )
        rewritten = rewrite_type_predicate_to_closure(raw_sparql)

        # The predicate must become the closure path.
        assert "subClassOf>*" in rewritten, (
            "Expected subclass-closure path in rewritten SPARQL; "
            "a plain rdf:type predicate would miss Condo instances."
        )
        # The object (Property URI) must be preserved.
        assert "<https://cograph.tech/types/Property>" in rewritten

        # The bare `a` predicate should not remain as-is in predicate position
        # next to a cograph.tech/types/ object (it was replaced).
        import re
        bare_a = re.search(
            r'\?\w+\s+a\s+<https://cograph\.tech/types/Property>',
            rewritten
        )
        assert bare_a is None, "Bare `a` predicate should have been rewritten to closure path."

    def test_closure_path_string_is_correct(self):
        """with_subclass_closure returns the exact property-path used in Neptune."""
        path = with_subclass_closure("Property")
        rdf_ns = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        rdfs_sub = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
        assert f"<{rdf_ns}>" in path
        assert f"<{rdfs_sub}>*" in path

    def test_multiple_type_triples_all_rewritten(self):
        """If a query has two type assertions they both get rewritten."""
        sparql = (
            "SELECT ?x ?y WHERE {\n"
            "  ?x a <https://cograph.tech/types/Property> .\n"
            "  ?y a <https://cograph.tech/types/Property> .\n"
            "}"
        )
        rewritten = rewrite_type_predicate_to_closure(sparql)
        assert rewritten.count("subClassOf>*") == 2
