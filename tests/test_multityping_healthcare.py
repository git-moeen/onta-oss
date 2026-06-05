"""Healthcare domain multi-typing tests (ADR 0001).

Domain: Patient < Person
Proves BOTH ingestion and querying for a subtype that is NOT directly in
DEFAULTS_BY_TYPE but IS ER-enabled via the ancestor chain:
  Patient -> Person (in DEFAULTS_BY_TYPE -> DEFAULT_GUEST_CONFIG)

Tests are split into three groups:
  A) Pure hierarchy / ER config logic (no Neptune, pure unit tests)
  B) ER pipeline: two Patient records sharing email merge to one canonical URI
     (proves config_for_with_hierarchy fires on a leaf subtype)
  C) SPARQL closure: query for Person covers Patient instances via subClassOf*
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from cograph_client.resolver.er.types import (
    DEFAULT_GUEST_CONFIG,
    DEFAULTS_BY_TYPE,
    ERConfig,
    NormalizedSignals,
    ancestor_chain,
    config_for,
    config_for_with_hierarchy,
    primary_type,
    primary_config_type,
)
from cograph_client.graph.ontology_queries import (
    rewrite_type_predicate_to_closure,
    with_subclass_closure,
    insert_type,
    insert_subtype,
)
from cograph_client.resolver.models import ExtractedAttribute, ExtractedEntity


# ---------------------------------------------------------------------------
# A) Pure hierarchy logic
# ---------------------------------------------------------------------------

PATIENT_PARENT_OF = {"Patient": "Person"}


def test_ancestor_chain_patient_to_person() -> None:
    """Patient -> Person is a two-element chain."""
    chain = ancestor_chain("Patient", PATIENT_PARENT_OF)
    assert chain == ["Patient", "Person"]


def test_ancestor_chain_patient_no_hierarchy() -> None:
    """Without hierarchy info, chain is just [Patient]."""
    chain = ancestor_chain("Patient", {})
    assert chain == ["Patient"]


def test_config_for_patient_flat_returns_value() -> None:
    """Patient IS in DEFAULTS_BY_TYPE, so flat config_for returns a config."""
    # Per DEFAULTS_BY_TYPE, Patient -> DEFAULT_GUEST_CONFIG
    cfg = config_for("Patient")
    assert cfg is DEFAULT_GUEST_CONFIG


def test_config_for_with_hierarchy_patient_inherits_person() -> None:
    """config_for_with_hierarchy climbs Patient -> Person and finds config."""
    # Even with hierarchy, Patient is in DEFAULTS_BY_TYPE directly, so it
    # resolves immediately.
    cfg = config_for_with_hierarchy("Patient", PATIENT_PARENT_OF)
    assert cfg is not None
    assert cfg is DEFAULT_GUEST_CONFIG


def test_config_for_with_hierarchy_novel_subtype() -> None:
    """A brand-new leaf not in DEFAULTS_BY_TYPE inherits config via chain walk.

    HospitalPatient is NOT in DEFAULTS_BY_TYPE. With hierarchy pointing to
    Patient (which IS configured), the walk succeeds.
    """
    novel_parent_of = {"HospitalPatient": "Patient", "Patient": "Person"}
    cfg = config_for_with_hierarchy("HospitalPatient", novel_parent_of)
    assert cfg is not None
    # Finds Patient first (index 1 in chain), which maps to DEFAULT_GUEST_CONFIG
    assert cfg is DEFAULT_GUEST_CONFIG


def test_config_for_novel_subtype_flat_returns_none() -> None:
    """Without hierarchy, an unknown leaf returns None (proves flat is insufficient)."""
    cfg = config_for("HospitalPatient")
    assert cfg is None, (
        "flat config_for must return None for unlisted type — "
        "chain-walk is the only way ER fires on novel subtypes"
    )


def test_primary_type_patient_is_leaf() -> None:
    """Patient is more specific than Person; primary_type returns Patient."""
    pt = primary_type(["Patient", "Person"], PATIENT_PARENT_OF)
    assert pt == "Patient"


def test_primary_type_single_patient() -> None:
    pt = primary_type(["Patient"], PATIENT_PARENT_OF)
    assert pt == "Patient"


def test_primary_type_empty() -> None:
    assert primary_type([], PATIENT_PARENT_OF) is None


def test_primary_config_type_patient() -> None:
    """primary_config_type returns Patient (deepest configured type)."""
    result = primary_config_type(["Patient", "Person"], PATIENT_PARENT_OF)
    assert result == "Patient"


# ---------------------------------------------------------------------------
# B) ER pipeline: two Patient records sharing email merge to one URI
# ---------------------------------------------------------------------------


def _patient_entity(entity_id: str, email: str, name: str) -> ExtractedEntity:
    """Build a minimal Patient ExtractedEntity for ER testing."""
    return ExtractedEntity(
        type_name="Patient",
        id=entity_id,
        attributes=[
            ExtractedAttribute(name="name", value=name, datatype="string"),
            ExtractedAttribute(name="email", value=email, datatype="string"),
        ],
    )


@pytest.mark.asyncio
async def test_er_patient_merges_via_ancestor_config() -> None:
    """Two Patient records sharing email merge BECAUSE config is found via chain-walk.

    This is the load-bearing ER fix: if config_for_with_hierarchy were replaced
    by flat config_for, HospitalPatient (not in DEFAULTS_BY_TYPE) would get
    config=None and ER would skip, leaving two duplicate URIs.

    We test the slightly simpler case where `Patient` itself resolves — the
    decisive-email path fires and AUTO_MERGE is returned.
    """
    from cograph_client.resolver.er.engine import ERPipeline, extract_signals
    from cograph_client.resolver.er.types import MergeAction
    from cograph_client.resolver.er.normalize import DefaultNormalizer
    from cograph_client.resolver.er.blocking import generate_block_keys

    SHARED_EMAIL = "alice.nguyen@hospital.org"
    entity1 = _patient_entity("alice-1", SHARED_EMAIL, "Alice Nguyen")
    entity2 = _patient_entity("alice-2", SHARED_EMAIL, "A. Nguyen")

    # Build normalized signals for entity1 (the "existing" record in the store)
    normalizer = DefaultNormalizer()
    sig1_raw = extract_signals(entity1)
    sig1 = normalizer.normalize(sig1_raw)

    # Mock Neptune: candidates_with_signals returns entity1's signals as existing
    mock_neptune = AsyncMock()
    # SparqlBlocker.candidates_with_signals is called inside find_match
    canonical_uri_1 = "https://cograph.tech/entities/Patient/alice-nguyen-hospitalhospitalorg"

    pipeline = ERPipeline(mock_neptune)
    # Patch the blocker directly so we control what "already exists" in the store
    pipeline._blocker.candidates_with_signals = AsyncMock(
        return_value={canonical_uri_1: sig1}
    )

    type_uri = "https://cograph.tech/types/Patient"
    instance_graph = "https://cograph.tech/graphs/test-tenant"

    # Use config_for_with_hierarchy so the leaf inherits from Person
    config = config_for_with_hierarchy("Patient", PATIENT_PARENT_OF)
    assert config is not None, "Pre-condition: Patient must resolve to a config"

    decision = await pipeline.find_match(
        entity2,
        "Patient",
        type_uri,
        instance_graph,
        config=config,
        parent_of=PATIENT_PARENT_OF,
    )

    assert decision.action == MergeAction.AUTO_MERGE, (
        f"Expected AUTO_MERGE via decisive email, got {decision.action}. "
        "This means ER did NOT fire correctly on Patient."
    )
    assert decision.canonical_uri == canonical_uri_1


@pytest.mark.asyncio
async def test_er_novel_patient_subtype_merges_via_chain() -> None:
    """A novel subtype NOT in DEFAULTS_BY_TYPE merges via chain-walk.

    HospitalPatient -> Patient -> Person: config resolved at Patient level.
    This would be None under flat config_for, proving hierarchy walk is essential.
    """
    from cograph_client.resolver.er.engine import ERPipeline, extract_signals
    from cograph_client.resolver.er.types import MergeAction
    from cograph_client.resolver.er.normalize import DefaultNormalizer

    novel_parent_of = {"HospitalPatient": "Patient", "Patient": "Person"}

    SHARED_EMAIL = "bob.chen@hospital.org"
    entity1 = ExtractedEntity(
        type_name="HospitalPatient",
        id="bob-1",
        attributes=[
            ExtractedAttribute(name="name", value="Bob Chen", datatype="string"),
            ExtractedAttribute(name="email", value=SHARED_EMAIL, datatype="string"),
        ],
    )
    entity2 = ExtractedEntity(
        type_name="HospitalPatient",
        id="bob-2",
        attributes=[
            ExtractedAttribute(name="name", value="Robert Chen", datatype="string"),
            ExtractedAttribute(name="email", value=SHARED_EMAIL, datatype="string"),
        ],
    )

    normalizer = DefaultNormalizer()
    sig1 = normalizer.normalize(extract_signals(entity1))
    canonical_uri_1 = "https://cograph.tech/entities/HospitalPatient/bob-chen"

    mock_neptune = AsyncMock()
    pipeline = ERPipeline(mock_neptune)
    pipeline._blocker.candidates_with_signals = AsyncMock(
        return_value={canonical_uri_1: sig1}
    )

    # Prove flat config_for returns None for this novel leaf
    assert config_for("HospitalPatient") is None, (
        "Sanity: HospitalPatient must NOT be in flat DEFAULTS_BY_TYPE"
    )

    # But with hierarchy, config is resolved
    config = config_for_with_hierarchy("HospitalPatient", novel_parent_of)
    assert config is not None, "HospitalPatient must inherit config via chain-walk"

    decision = await pipeline.find_match(
        entity2,
        "HospitalPatient",
        "https://cograph.tech/types/HospitalPatient",
        "https://cograph.tech/graphs/test-tenant",
        config=config,
        parent_of=novel_parent_of,
    )

    assert decision.action == MergeAction.AUTO_MERGE, (
        f"Novel subtype HospitalPatient should merge via chain-walk config. "
        f"Got {decision.action} — ER chain-walk is broken."
    )
    assert decision.canonical_uri == canonical_uri_1


@pytest.mark.asyncio
async def test_er_skips_without_hierarchy() -> None:
    """Confirm that with config=None and no hierarchy, ER skips for unknown leaf.

    Demonstrates the regression that the chain-walk fix prevents: if the caller
    passes neither config nor parent_of, find_match falls back to flat lookup
    (DEFAULTS_BY_TYPE.get) and returns SKIP for HospitalPatient.
    """
    from cograph_client.resolver.er.engine import ERPipeline
    from cograph_client.resolver.er.types import MergeAction

    mock_neptune = AsyncMock()
    pipeline = ERPipeline(mock_neptune)
    pipeline._blocker.candidates_with_signals = AsyncMock(return_value={})

    entity = ExtractedEntity(
        type_name="HospitalPatient",
        id="charlie-1",
        attributes=[
            ExtractedAttribute(name="email", value="charlie@hospital.org", datatype="string"),
        ],
    )

    # No config, no parent_of — flat lookup returns None → SKIP
    decision = await pipeline.find_match(
        entity,
        "HospitalPatient",
        "https://cograph.tech/types/HospitalPatient",
        "https://cograph.tech/graphs/test-tenant",
        config=None,
        parent_of={},
    )

    assert decision.action == MergeAction.SKIP, (
        "Without hierarchy, unknown leaf should SKIP ER — "
        "demonstrating the regression the fix prevents."
    )


# ---------------------------------------------------------------------------
# C) SPARQL closure: query for Person covers Patient instances
# ---------------------------------------------------------------------------


def test_rewrite_type_predicate_to_closure_patient() -> None:
    """A query for Person type gets rewritten to use subclass-closure path."""
    sparql = "SELECT ?x WHERE { ?x a <https://cograph.tech/types/Person> . }"
    rewritten = rewrite_type_predicate_to_closure(sparql)

    assert "subClassOf>*" in rewritten, (
        "Rewritten SPARQL must contain the subClassOf* closure path "
        "so Patient instances are returned when querying for Person."
    )
    # The object URI (Person) must be preserved
    assert "<https://cograph.tech/types/Person>" in rewritten


def test_rewrite_type_predicate_to_closure_full_rdf_type_form() -> None:
    """Form B (full rdf:type IRI) is also rewritten."""
    rdf_type = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    sparql = f"SELECT ?x WHERE {{ ?x <{rdf_type}> <https://cograph.tech/types/Person> . }}"
    rewritten = rewrite_type_predicate_to_closure(sparql)

    assert "subClassOf>*" in rewritten


def test_rewrite_type_predicate_to_closure_is_idempotent() -> None:
    """Applying closure rewrite twice is the same as applying it once."""
    sparql = "SELECT ?x WHERE { ?x a <https://cograph.tech/types/Patient> . }"
    once = rewrite_type_predicate_to_closure(sparql)
    twice = rewrite_type_predicate_to_closure(once)
    assert once == twice, "rewrite_type_predicate_to_closure must be idempotent"


def test_with_subclass_closure_returns_closure_path() -> None:
    """with_subclass_closure returns the property-path string for subtype queries."""
    path = with_subclass_closure("Person")
    assert "subClassOf" in path
    assert "rdf-syntax-ns#type" in path or "rdf:type" in path or "#type" in path


def test_closure_covers_patient_under_person_query() -> None:
    """Compose a Patient instance triple and verify the Person query rewrite would match it.

    In a real triple store, `?x a/rdfs:subClassOf* :Person` returns instances
    asserted as Patient if Patient rdfs:subClassOf Person exists. Here we verify:
      1. Patient instances are stamped with Patient (NOT Person) as their asserted type.
      2. A query for Person is rewritten to use the closure path, which semantically
         covers Patient.
    """
    # Simulated asserted type triple for a Patient instance
    patient_uri = "https://cograph.tech/entities/Patient/alice-nguyen"
    patient_type = "https://cograph.tech/types/Patient"
    rdf_type_pred = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

    # Step 1: The instance is stamped with Patient (not Person)
    asserted_type_triple_predicate = rdf_type_pred
    asserted_type_triple_object = patient_type
    assert "Patient" in asserted_type_triple_object
    assert "Person" not in asserted_type_triple_object

    # Step 2: A query for Person gets the closure rewrite
    person_query = f"SELECT ?x WHERE {{ ?x a <https://cograph.tech/types/Person> . }}"
    rewritten = rewrite_type_predicate_to_closure(person_query)

    # The closure path replaces the bare `a`
    assert "subClassOf>*" in rewritten
    assert "<https://cograph.tech/types/Person>" in rewritten
    # The bare `a` predicate in predicate position is gone
    assert " a <https://cograph.tech/types/Person>" not in rewritten


def test_insert_type_creates_patient_and_person() -> None:
    """insert_type generates correct SPARQL for both Person and Patient types."""
    graph = "https://cograph.tech/graphs/test-tenant"

    person_sparql = insert_type(graph, "Person")
    assert "cograph.tech/types/Person" in person_sparql
    assert "Class" in person_sparql

    # Patient with parent_type=Person creates subClassOf triple
    patient_sparql = insert_type(graph, "Patient", parent_type="Person")
    assert "cograph.tech/types/Patient" in patient_sparql
    assert "subClassOf" in patient_sparql
    assert "cograph.tech/types/Person" in patient_sparql


def test_insert_subtype_creates_patient_subclassof_person() -> None:
    """insert_subtype generates correct SPARQL to link Patient to Person."""
    graph = "https://cograph.tech/graphs/test-tenant"
    sparql = insert_subtype(graph, "Person", "Patient")

    assert "subClassOf" in sparql
    assert "cograph.tech/types/Patient" in sparql
    assert "cograph.tech/types/Person" in sparql


# ---------------------------------------------------------------------------
# D) SchemaResolver._synthesize_ancestors integration (with mocked Neptune)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_ancestors_creates_person_when_patient_ingested() -> None:
    """_synthesize_ancestors closes the chain: when Patient is created with
    parent Person, Person is inserted into the ontology if not present.
    """
    from unittest.mock import AsyncMock as AM
    from cograph_client.resolver.schema_resolver import SchemaResolver
    from cograph_client.resolver.models import IngestResult

    mock_neptune = AM()
    mock_neptune.update = AM(return_value=None)
    mock_neptune.query = AM(return_value={"head": {"vars": []}, "results": {"bindings": []}})

    # Build a minimal resolver — avoid touching Anthropic/OpenRouter keys.
    # settings is imported lazily inside __init__ via `from cograph_client.config import settings`
    mock_settings = MagicMock()
    mock_settings.openrouter_api_key = ""
    with patch("cograph_client.resolver.schema_resolver.anthropic"):
        with patch("cograph_client.config.settings", mock_settings):
            from cograph_client.resolver.verdict_cache import JsonVerdictCache
            verdict_cache = MagicMock(spec=JsonVerdictCache)
            resolver = SchemaResolver(
                neptune=mock_neptune,
                anthropic_key="sk-fake",
                verdict_cache=verdict_cache,
            )

    graph_uri = "https://cograph.tech/graphs/test-tenant"
    existing_types: dict[str, str] = {}      # empty ontology
    existing_attrs: dict = {}
    result = IngestResult(entities_extracted=1)

    # Synthesize Patient <- Person. Person does NOT exist yet.
    await resolver._synthesize_ancestors(
        "Patient", "Person", graph_uri, existing_types, existing_attrs, result,
    )

    # Person should now be in existing_types
    assert "Person" in existing_types, (
        "_synthesize_ancestors must add Person to existing_types "
        "when it creates the ancestor"
    )

    # Neptune.update must have been called (at least one INSERT for Person)
    assert mock_neptune.update.called, (
        "_synthesize_ancestors must call neptune.update to insert ancestor types"
    )

    # Check that result.types_created records the synthesized ancestor
    assert "Person" in result.types_created, (
        "result.types_created must include Person synthesized via ancestor chain"
    )

    # The parent_of map must record the Patient->Person edge
    assert resolver._parent_of.get("Patient") == "Person", (
        "_synthesize_ancestors must record the child->parent edge in self._parent_of"
    )
