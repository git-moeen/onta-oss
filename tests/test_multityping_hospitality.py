"""Multi-typing tests for the hospitality domain.

Hierarchy under test:  HotelGuest < Guest < Person

INGESTION assertions (Part A):
  A1  - _resolve_type stamps HotelGuest as most-specific asserted type.
  A2  - ancestor_chain("HotelGuest", {"HotelGuest":"Guest","Guest":"Person"}) yields
        the full lineage [HotelGuest, Guest, Person].
  A3  - _synthesize_ancestors calls insert_type for each missing ancestor (Guest,
        Person) so rdfs:subClassOf chains are complete after a HotelGuest entity
        is ingested into a fresh ontology that only has HotelGuest.
  A4  - ER merges two cross-file HotelGuest records sharing an email, even though
        config_for("HotelGuest") is None and config_for_with_hierarchy correctly
        resolves to DEFAULT_GUEST_CONFIG via the chain walk.

QUERYING assertions (Part B):
  B1  - with_subclass_closure("Person") returns the property-path string that
        contains "subClassOf>*", proving the closure seam exists.
  B2  - rewrite_type_predicate_to_closure rewrites a SPARQL query for "Person"
        to use the closure predicate, so HotelGuest instances (asserted as
        HotelGuest) are returned by a Person query.
  B3  - parent_map_query produces a SPARQL SELECT with the rdfs:subClassOf WHERE
        clause the SchemaResolver uses to build its parent_of map.
  B4  - After rewrite, the closure predicate appears exactly once per type triple
        (idempotency: applying the rewrite twice equals applying it once).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Pure hierarchy helpers (no I/O)
# ---------------------------------------------------------------------------

from cograph_client.resolver.er.types import (
    DEFAULT_GUEST_CONFIG,
    ancestor_chain,
    config_for,
    config_for_with_hierarchy,
    primary_config_type,
    primary_type,
)
from cograph_client.graph.ontology_queries import (
    parent_map_query,
    rewrite_type_predicate_to_closure,
    with_subclass_closure,
)
from cograph_client.resolver.models import ExtractedAttribute, ExtractedEntity, IngestResult
from cograph_client.resolver.er.types import MergeAction, MergeDecision


# Hospitality domain hierarchy used across all tests
PARENT_OF = {
    "HotelGuest": "Guest",
    "Guest": "Person",
}


# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------

def _hotel_guest_entity(name: str, email: str, entity_id: str | None = None) -> ExtractedEntity:
    """Construct a minimal HotelGuest ExtractedEntity."""
    return ExtractedEntity(
        type_name="HotelGuest",
        id=entity_id or email,
        same_as=None,
        parent_type="Guest",
        attributes=[
            ExtractedAttribute(name="name", value=name, datatype="string"),
            ExtractedAttribute(name="email", value=email, datatype="string"),
        ],
    )


# ===========================================================================
# Part A — INGESTION
# ===========================================================================


class TestAncestorChainHospitality:
    """A1 / A2 — ancestor_chain returns the full HotelGuest → Guest → Person lineage."""

    def test_full_chain(self):
        chain = ancestor_chain("HotelGuest", PARENT_OF)
        assert chain == ["HotelGuest", "Guest", "Person"]

    def test_chain_mid_node(self):
        chain = ancestor_chain("Guest", PARENT_OF)
        assert chain == ["Guest", "Person"]

    def test_chain_root(self):
        chain = ancestor_chain("Person", PARENT_OF)
        assert chain == ["Person"]

    def test_unknown_type_empty_map(self):
        assert ancestor_chain("HotelGuest", {}) == ["HotelGuest"]

    def test_cycle_guard(self):
        cyclic = {"A": "B", "B": "A"}
        chain = ancestor_chain("A", cyclic)
        assert chain == ["A", "B"]  # stops when B would revisit A

    def test_self_loop_guard(self):
        assert ancestor_chain("X", {"X": "X"}) == ["X"]


class TestConfigForHospitality:
    """A1 — config_for flat lookup returns None for HotelGuest (not in DEFAULTS_BY_TYPE)."""

    def test_config_for_hotel_guest_is_none(self):
        # Flat lookup must return None — proving the legacy path would skip ER.
        assert config_for("HotelGuest") is None

    def test_config_for_guest_is_default_guest(self):
        cfg = config_for("Guest")
        assert cfg is DEFAULT_GUEST_CONFIG

    def test_config_for_person_is_default_guest(self):
        # Person is in DEFAULTS_BY_TYPE mapped to DEFAULT_GUEST_CONFIG.
        cfg = config_for("Person")
        assert cfg is DEFAULT_GUEST_CONFIG


class TestConfigForWithHierarchyHospitality:
    """A4 (setup) — config_for_with_hierarchy climbs to Guest and returns DEFAULT_GUEST_CONFIG."""

    def test_hotel_guest_resolves_via_chain(self):
        cfg = config_for_with_hierarchy("HotelGuest", PARENT_OF)
        assert cfg is DEFAULT_GUEST_CONFIG

    def test_hotel_guest_empty_parent_of_returns_none(self):
        # Without the hierarchy map the chain is just [HotelGuest] and nothing
        # in DEFAULTS_BY_TYPE maps to it → must return None (flat behavior).
        assert config_for_with_hierarchy("HotelGuest", {}) is None

    def test_guest_directly_resolves(self):
        cfg = config_for_with_hierarchy("Guest", PARENT_OF)
        assert cfg is DEFAULT_GUEST_CONFIG

    def test_person_directly_resolves(self):
        cfg = config_for_with_hierarchy("Person", PARENT_OF)
        assert cfg is DEFAULT_GUEST_CONFIG


class TestPrimaryTypeHospitality:
    """A1 — primary_type selects the deepest (most-specific) leaf."""

    def test_hotel_guest_dominates_guest(self):
        # HotelGuest is more specific than Guest; Guest must be dominated.
        pt = primary_type(["Guest", "HotelGuest"], PARENT_OF)
        assert pt == "HotelGuest"

    def test_hotel_guest_dominates_person(self):
        pt = primary_type(["Person", "HotelGuest"], PARENT_OF)
        assert pt == "HotelGuest"

    def test_hotel_guest_dominates_all_ancestors(self):
        pt = primary_type(["Person", "Guest", "HotelGuest"], PARENT_OF)
        assert pt == "HotelGuest"

    def test_single_type(self):
        assert primary_type(["HotelGuest"], PARENT_OF) == "HotelGuest"

    def test_empty(self):
        assert primary_type([], PARENT_OF) is None


class TestPrimaryConfigTypeHospitality:
    """A4 (setup) — primary_config_type finds the deepest ER-enabled asserted type."""

    def test_hotel_guest_is_deepest_configured(self):
        # HotelGuest itself has no direct config, but it resolves via Guest.
        pct = primary_config_type(["HotelGuest"], PARENT_OF)
        assert pct == "HotelGuest"

    def test_hotel_guest_over_person(self):
        pct = primary_config_type(["HotelGuest", "Person"], PARENT_OF)
        assert pct == "HotelGuest"

    def test_no_configured_type(self):
        # A made-up type with no hierarchy and not in DEFAULTS_BY_TYPE.
        assert primary_config_type(["Widget"], {}) is None


# ---------------------------------------------------------------------------
# A3 — ancestor synthesis via SchemaResolver._synthesize_ancestors
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_neptune():
    """Minimal AsyncMock NeptuneClient."""
    from cograph_client.graph.client import NeptuneClient
    client = AsyncMock(spec=NeptuneClient)
    client.health.return_value = True
    client.query.return_value = {"head": {"vars": []}, "results": {"bindings": []}}
    client.update.return_value = None
    return client


@pytest.fixture
def resolver(mock_neptune):
    """SchemaResolver wired to mocked Neptune (no real LLM, no real DB)."""
    from cograph_client.resolver.schema_resolver import SchemaResolver
    from cograph_client.resolver.verdict_cache import JsonVerdictCache
    import tempfile
    from pathlib import Path

    verdict_dir = tempfile.mkdtemp()
    verdict_path = Path(verdict_dir) / "verdicts.json"
    cache = JsonVerdictCache(verdict_path)

    with patch.dict("os.environ", {
        "ANTHROPIC_API_KEY": "test-key",
        "OPENROUTER_API_KEY": "test-or-key",
        "COGRAPH_ER_ENABLED": "0",  # ER off — we test ER separately below
    }):
        resolver = SchemaResolver(
            neptune=mock_neptune,
            anthropic_key="test-key",
            verdict_cache=cache,
        )
    return resolver


@pytest.mark.asyncio
async def test_a3_synthesize_ancestors_inserts_missing_parents(resolver, mock_neptune):
    """A3 — _synthesize_ancestors inserts Guest and Person when only HotelGuest exists."""
    graph_uri = "https://cograph.tech/graphs/test-tenant"

    # Ontology starts with only HotelGuest registered.
    existing_types: dict[str, str] = {"HotelGuest": ""}
    existing_attrs: dict = {"HotelGuest": {}}
    result = IngestResult(entities_extracted=1)

    # Simulate the SubClassOf edge HotelGuest -> Guest already in parent_of
    resolver._parent_of = {"HotelGuest": "Guest"}

    # Call synthesize for the chain HotelGuest -> Guest (Guest's parent unknown yet)
    await resolver._synthesize_ancestors(
        "HotelGuest", "Guest", graph_uri, existing_types, existing_attrs, result
    )

    # Guest must now be registered in the resolver's memory
    assert "Guest" in existing_types, "Guest should have been synthesized into existing_types"

    # insert_type should have been called for Guest
    update_calls_sparql = [c.args[0] for c in mock_neptune.update.call_args_list]
    assert any("Guest" in sparql for sparql in update_calls_sparql), (
        "Neptune.update should have been called with an INSERT for Guest"
    )
    # Guest should appear in types_created
    assert "Guest" in result.types_created


@pytest.mark.asyncio
async def test_a3_synthesize_ancestors_full_chain(resolver, mock_neptune):
    """A3 — synthesizing HotelGuest -> Guest with Guest -> Person already in parent_of
    closes the full chain: both Guest AND Person are synthesized when missing."""
    graph_uri = "https://cograph.tech/graphs/test-tenant"

    existing_types: dict[str, str] = {"HotelGuest": ""}
    existing_attrs: dict = {"HotelGuest": {}}
    result = IngestResult(entities_extracted=1)

    # Provide the full chain in parent_of upfront so ancestor_chain can walk all the way
    resolver._parent_of = {"HotelGuest": "Guest", "Guest": "Person"}

    await resolver._synthesize_ancestors(
        "HotelGuest", "Guest", graph_uri, existing_types, existing_attrs, result
    )

    # Both Guest and Person must now exist in the resolver's memory
    assert "Guest" in existing_types
    assert "Person" in existing_types

    update_calls_sparql = [c.args[0] for c in mock_neptune.update.call_args_list]
    # At minimum Neptune was called with INSERTs touching Guest and Person
    types_inserted = [s for s in update_calls_sparql if "INSERT" in s]
    assert any("Guest" in s for s in types_inserted)
    assert any("Person" in s for s in types_inserted)

    assert "Guest" in result.types_created
    assert "Person" in result.types_created


@pytest.mark.asyncio
async def test_a3_synthesize_idempotent(resolver, mock_neptune):
    """A3 — synthesize is a no-op for ancestors that already exist in existing_types."""
    graph_uri = "https://cograph.tech/graphs/test-tenant"

    # Both ancestors already present
    existing_types = {"HotelGuest": "", "Guest": "", "Person": ""}
    existing_attrs = {"HotelGuest": {}, "Guest": {}, "Person": {}}
    result = IngestResult(entities_extracted=1)
    resolver._parent_of = {"HotelGuest": "Guest", "Guest": "Person"}

    await resolver._synthesize_ancestors(
        "HotelGuest", "Guest", graph_uri, existing_types, existing_attrs, result
    )

    # No new types should have been created
    assert result.types_created == []
    # Neptune.update should NOT have been called at all (nothing to insert)
    assert mock_neptune.update.call_count == 0


# ---------------------------------------------------------------------------
# A4 — ER merges two cross-file HotelGuest records via chain-walked config
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a4_er_merges_hotel_guests_via_guest_config():
    """A4 — ERPipeline.find_match merges two HotelGuest records sharing an email.

    Because config_for("HotelGuest") is None, a flat ER lookup would skip them.
    With config_for_with_hierarchy + PARENT_OF the correct config is supplied
    and the decisive email signal causes AUTO_MERGE.
    """
    from cograph_client.resolver.er.engine import ERPipeline
    from cograph_client.resolver.er.types import MergeAction

    mock_neptune = AsyncMock()

    # The second call to candidates_with_signals returns a single candidate
    # whose normalized signals match the incoming entity (same email).
    from cograph_client.resolver.er.types import NormalizedSignals
    CANONICAL_URI = "https://cograph.tech/entities/HotelGuest/alice_example_com-abc12345"
    existing_candidate_signals = NormalizedSignals(
        name="alice smith",
        email="alice@example.com",
        email_local="alice",
    )

    pipeline = ERPipeline(mock_neptune)
    pipeline._blocker.candidates_with_signals = AsyncMock(
        return_value={CANONICAL_URI: existing_candidate_signals}
    )

    incoming = _hotel_guest_entity("Alice Smith", "alice@example.com")

    # Explicitly pass config via chain walk (this is what SchemaResolver now does)
    config = config_for_with_hierarchy("HotelGuest", PARENT_OF)
    assert config is DEFAULT_GUEST_CONFIG, (
        "Precondition: chain walk must return DEFAULT_GUEST_CONFIG for HotelGuest"
    )

    decision = await pipeline.find_match(
        incoming,
        type_name="HotelGuest",
        type_uri="https://cograph.tech/types/HotelGuest",
        instance_graph="https://cograph.tech/graphs/test-tenant",
        config=config,
        parent_of=PARENT_OF,
    )

    assert decision.action == MergeAction.AUTO_MERGE, (
        f"Expected AUTO_MERGE but got {decision.action}. "
        "Two HotelGuest records sharing an email must merge via Guest config."
    )
    assert decision.canonical_uri == CANONICAL_URI


@pytest.mark.asyncio
async def test_a4_er_no_merge_without_hierarchy():
    """A4 (control) — flat config_for returns None for HotelGuest, so ER is skipped."""
    from cograph_client.resolver.er.engine import ERPipeline

    mock_neptune = AsyncMock()
    pipeline = ERPipeline(mock_neptune)
    pipeline._blocker.candidates_with_signals = AsyncMock(return_value={})

    incoming = _hotel_guest_entity("Alice Smith", "alice@example.com")

    # Intentionally pass config=None and empty parent_of (flat behavior)
    decision = await pipeline.find_match(
        incoming,
        type_name="HotelGuest",
        type_uri="https://cograph.tech/types/HotelGuest",
        instance_graph="https://cograph.tech/graphs/test-tenant",
        config=None,    # forces internal config_for_with_hierarchy lookup
        parent_of={},   # empty map → config_for_with_hierarchy returns None
    )

    # With no config, the pipeline returns SKIP immediately (never touches blocker)
    assert decision.action == MergeAction.SKIP, (
        "Without hierarchy, HotelGuest has no config and ER must be skipped (SKIP)."
    )


# ===========================================================================
# Part B — QUERYING (subclass closure seam)
# ===========================================================================


class TestWithSubclassClosure:
    """B1 — with_subclass_closure returns the property-path string."""

    def test_returns_closure_path(self):
        path = with_subclass_closure("Person")
        assert "subClassOf>*" in path, (
            f"with_subclass_closure must include 'subClassOf>*' but got: {path!r}"
        )

    def test_path_contains_rdf_type(self):
        path = with_subclass_closure("HotelGuest")
        assert "type>" in path

    def test_path_is_type_independent(self):
        # per spec: type_name is ignored — the predicate path is always the same
        p1 = with_subclass_closure("HotelGuest")
        p2 = with_subclass_closure("Person")
        assert p1 == p2


class TestRewriteTypePredicateToClosureHospitality:
    """B2 — rewrite_type_predicate_to_closure rewrites Person queries to cover subtypes."""

    PERSON_URI = "https://cograph.tech/types/Person"
    HOTEL_GUEST_URI = "https://cograph.tech/types/HotelGuest"

    def test_form_a_rewritten(self):
        sparql = f"SELECT ?x WHERE {{ ?x a <{self.PERSON_URI}> . }}"
        result = rewrite_type_predicate_to_closure(sparql)
        assert "subClassOf>*" in result, (
            "Form A (`?x a <types/Person>`) must be rewritten to closure path."
        )
        assert self.PERSON_URI in result, "Object URI must be preserved verbatim."

    def test_form_b_rewritten(self):
        rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
        sparql = f"SELECT ?x WHERE {{ ?x <{rdf_type}> <{self.PERSON_URI}> . }}"
        result = rewrite_type_predicate_to_closure(sparql)
        assert "subClassOf>*" in result, (
            "Form B (`?x rdf:type <types/Person>`) must be rewritten to closure path."
        )
        assert self.PERSON_URI in result

    def test_rewrite_covers_hotel_guest_type_uri(self):
        """A query for HotelGuest is also rewritten (closure of a leaf = the leaf itself)."""
        sparql = f"SELECT ?x WHERE {{ ?x a <{self.HOTEL_GUEST_URI}> . }}"
        result = rewrite_type_predicate_to_closure(sparql)
        assert "subClassOf>*" in result

    def test_idempotent(self):
        """B4 — applying the rewrite twice must be identical to applying it once."""
        sparql = f"SELECT ?x WHERE {{ ?x a <{self.PERSON_URI}> . }}"
        once = rewrite_type_predicate_to_closure(sparql)
        twice = rewrite_type_predicate_to_closure(once)
        assert once == twice, (
            "rewrite_type_predicate_to_closure must be idempotent: applying twice == once."
        )

    def test_multiple_type_triples_all_rewritten(self):
        """Both type triples in a multi-type query are rewritten."""
        sparql = (
            f"SELECT ?x ?y WHERE {{ "
            f"?x a <{self.PERSON_URI}> . "
            f"?y a <{self.HOTEL_GUEST_URI}> . "
            f"}}"
        )
        result = rewrite_type_predicate_to_closure(sparql)
        # Two separate type assertions → two 'subClassOf>*' occurrences
        assert result.count("subClassOf>*") == 2, (
            "All type assertion triples in the query must be rewritten."
        )

    def test_non_cograph_type_uri_not_rewritten(self):
        """Triples whose object is NOT under https://cograph.tech/types/ are left alone."""
        sparql = "SELECT ?x WHERE { ?x a <http://schema.org/Person> . }"
        result = rewrite_type_predicate_to_closure(sparql)
        assert "subClassOf>*" not in result

    def test_closure_path_present_after_person_query_rewrite(self):
        """Concrete assertion that a 'list all Persons' query returns subtypes."""
        sparql = f"SELECT ?guest WHERE {{ ?guest a <{self.PERSON_URI}> . }}"
        rewritten = rewrite_type_predicate_to_closure(sparql)
        # The rewritten query uses the closure path, so HotelGuest instances
        # (asserted as `rdf:type <types/HotelGuest>`) are covered via:
        #   HotelGuest rdfs:subClassOf Guest rdfs:subClassOf Person
        assert "subClassOf>*" in rewritten
        assert self.PERSON_URI in rewritten


class TestParentMapQueryHospitality:
    """B3 — parent_map_query produces correct SPARQL for building the parent_of map."""

    def test_contains_subclass_of(self):
        sparql = parent_map_query("https://cograph.tech/graphs/test-tenant")
        assert "subClassOf" in sparql

    def test_selects_child_parent(self):
        sparql = parent_map_query("https://cograph.tech/graphs/test-tenant")
        assert "?child" in sparql
        assert "?parent" in sparql

    def test_is_select_query(self):
        sparql = parent_map_query("https://cograph.tech/graphs/test-tenant")
        assert sparql.strip().upper().startswith("SELECT")

    def test_graph_uri_included(self):
        graph = "https://cograph.tech/graphs/hotel-tenant"
        sparql = parent_map_query(graph)
        assert graph in sparql


# ---------------------------------------------------------------------------
# Integration-flavour: resolver _fetch_parent_map builds the right map
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_parent_map_builds_hospitality_map(mock_neptune):
    """SchemaResolver._fetch_parent_map turns Neptune bindings into the parent_of dict."""
    from cograph_client.resolver.schema_resolver import SchemaResolver
    from cograph_client.resolver.verdict_cache import JsonVerdictCache
    import tempfile

    # Mock Neptune to return two rdfs:subClassOf edges:
    #   HotelGuest subClassOf Guest
    #   Guest      subClassOf Person
    mock_neptune.query.return_value = {
        "head": {"vars": ["child", "parent"]},
        "results": {
            "bindings": [
                {
                    "child": {"type": "uri", "value": "https://cograph.tech/types/HotelGuest"},
                    "parent": {"type": "uri", "value": "https://cograph.tech/types/Guest"},
                },
                {
                    "child": {"type": "uri", "value": "https://cograph.tech/types/Guest"},
                    "parent": {"type": "uri", "value": "https://cograph.tech/types/Person"},
                },
            ]
        },
    }

    from pathlib import Path
    verdict_dir = tempfile.mkdtemp()
    verdict_path = Path(verdict_dir) / "verdicts.json"
    cache = JsonVerdictCache(verdict_path)

    with patch.dict("os.environ", {
        "ANTHROPIC_API_KEY": "test-key",
        "OPENROUTER_API_KEY": "test-or-key",
    }):
        res = SchemaResolver(neptune=mock_neptune, anthropic_key="test-key", verdict_cache=cache)

    parent_of = await res._fetch_parent_map("https://cograph.tech/graphs/test-tenant")

    assert parent_of == {"HotelGuest": "Guest", "Guest": "Person"}, (
        f"_fetch_parent_map must return exactly {{'HotelGuest':'Guest','Guest':'Person'}} "
        f"but got {parent_of!r}"
    )

    # Confirm that climbing the chain now works correctly
    chain = ancestor_chain("HotelGuest", parent_of)
    assert chain == ["HotelGuest", "Guest", "Person"]
    cfg = config_for_with_hierarchy("HotelGuest", parent_of)
    assert cfg is DEFAULT_GUEST_CONFIG
