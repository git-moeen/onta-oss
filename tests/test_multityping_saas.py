"""Multi-typing tests for the SaaS domain.

Hierarchy: TrialSubscriber < Subscriber < Person

A) INGESTION
   - Instance is stamped with its MOST-SPECIFIC asserted type (TrialSubscriber).
   - Ancestor synthesis creates the missing parent types (Subscriber, Person) in
     the ontology so rdfs:subClassOf chains are complete.
   - ER fires on TrialSubscriber because config_for_with_hierarchy climbs the
     chain to Subscriber (which maps to DEFAULT_GUEST_CONFIG).  Two cross-file
     signups sharing an email therefore merge to one canonical URI.
     With the old flat config_for(TrialSubscriber) the result would be None and
     ER would NOT fire — they would get distinct URIs.

B) QUERYING
   - rewrite_type_predicate_to_closure rewrites a SPARQL type triple for the
     ANCESTOR (Person) into a property-path closure so instances asserted as
     the leaf subtype (TrialSubscriber) are also returned.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _fake_api_keys(monkeypatch):
    """Set fake API keys via monkeypatch so they are auto-restored after each
    test. Using os.environ.setdefault here would leak OPENROUTER_API_KEY into the
    rest of the process and make unrelated pipeline tests (test_ask_*) attempt a
    real OpenRouter call → HTTPStatusError.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key")


# ---------------------------------------------------------------------------
# A) INGESTION — pure-logic tier (no Neptune needed)
# ---------------------------------------------------------------------------


class TestSaaSTypeHierarchyPure:
    """Verify ancestor_chain, config_for_with_hierarchy, and primary_type
    for the SaaS hierarchy TrialSubscriber < Subscriber < Person without
    touching any async / Neptune code.
    """

    SAAS_PARENT_OF = {
        "TrialSubscriber": "Subscriber",
        "Subscriber": "Person",
    }

    # ------------------------------------------------------------------ chain
    def test_ancestor_chain_leaf(self):
        from cograph_client.resolver.er.types import ancestor_chain
        chain = ancestor_chain("TrialSubscriber", self.SAAS_PARENT_OF)
        assert chain == ["TrialSubscriber", "Subscriber", "Person"]

    def test_ancestor_chain_mid(self):
        from cograph_client.resolver.er.types import ancestor_chain
        chain = ancestor_chain("Subscriber", self.SAAS_PARENT_OF)
        assert chain == ["Subscriber", "Person"]

    def test_ancestor_chain_root(self):
        from cograph_client.resolver.er.types import ancestor_chain
        chain = ancestor_chain("Person", self.SAAS_PARENT_OF)
        assert chain == ["Person"]

    # -------------------------------------------------------- config_for_with_hierarchy
    def test_subscriber_resolves_to_guest_config(self):
        """Subscriber is directly in DEFAULTS_BY_TYPE → Guest config."""
        from cograph_client.resolver.er.types import (
            DEFAULT_GUEST_CONFIG,
            config_for_with_hierarchy,
        )
        cfg = config_for_with_hierarchy("Subscriber", self.SAAS_PARENT_OF)
        assert cfg is DEFAULT_GUEST_CONFIG

    def test_trial_subscriber_inherits_guest_config(self):
        """TrialSubscriber is NOT in DEFAULTS_BY_TYPE, but its ancestor
        Subscriber IS → config_for_with_hierarchy must climb and return
        DEFAULT_GUEST_CONFIG (identity check).
        """
        from cograph_client.resolver.er.types import (
            DEFAULT_GUEST_CONFIG,
            config_for_with_hierarchy,
        )
        cfg = config_for_with_hierarchy("TrialSubscriber", self.SAAS_PARENT_OF)
        assert cfg is DEFAULT_GUEST_CONFIG

    def test_flat_config_for_returns_none_for_leaf(self):
        """Flat config_for(TrialSubscriber) must be None — proving the old
        behavior was broken for granular subtypes.  This is the regressed path
        that the hierarchy-aware config fixes.
        """
        from cograph_client.resolver.er.types import config_for
        cfg = config_for("TrialSubscriber")
        assert cfg is None, (
            "config_for is intentionally flat; TrialSubscriber is not in "
            "DEFAULTS_BY_TYPE.  The hierarchy-aware path is config_for_with_hierarchy."
        )

    # --------------------------------------------------------- primary_type
    def test_primary_type_picks_leaf(self):
        from cograph_client.resolver.er.types import primary_type
        # When all three are asserted, TrialSubscriber dominates.
        pt = primary_type(
            ["Person", "Subscriber", "TrialSubscriber"],
            self.SAAS_PARENT_OF,
        )
        assert pt == "TrialSubscriber"

    def test_primary_type_single(self):
        from cograph_client.resolver.er.types import primary_type
        pt = primary_type(["TrialSubscriber"], self.SAAS_PARENT_OF)
        assert pt == "TrialSubscriber"

    def test_primary_config_type_leaf(self):
        """primary_config_type must return TrialSubscriber even though its
        config is inherited — the leaf IS the most-specific configured type.
        """
        from cograph_client.resolver.er.types import primary_config_type
        pct = primary_config_type(
            ["TrialSubscriber", "Subscriber", "Person"],
            self.SAAS_PARENT_OF,
        )
        assert pct == "TrialSubscriber"


# ---------------------------------------------------------------------------
# B) QUERYING — closure rewrite (pure SPARQL string seam, no Neptune)
# ---------------------------------------------------------------------------


TYPES_NS = "https://cograph.tech/types/"
RDF_TYPE_FULL = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"


class TestSubclassClosureRewrite:
    """Assert that rewrite_type_predicate_to_closure rewrites ancestor-type
    queries so leaf-typed instances are covered at query time.
    """

    PERSON_SPARQL_FORM_A = (
        f"SELECT ?s WHERE {{ ?s a <{TYPES_NS}Person> . }}"
    )
    PERSON_SPARQL_FORM_B = (
        f"SELECT ?s WHERE {{ ?s <{RDF_TYPE_FULL}> <{TYPES_NS}Person> . }}"
    )

    def test_form_a_rewritten_to_closure(self):
        from cograph_client.graph.ontology_queries import rewrite_type_predicate_to_closure
        out = rewrite_type_predicate_to_closure(self.PERSON_SPARQL_FORM_A)
        assert f"{RDFS_SUBCLASS}>*" in out, (
            "Expected subclass-closure path in rewritten SPARQL"
        )
        # The object type URI must be preserved
        assert f"<{TYPES_NS}Person>" in out

    def test_form_b_rewritten_to_closure(self):
        from cograph_client.graph.ontology_queries import rewrite_type_predicate_to_closure
        out = rewrite_type_predicate_to_closure(self.PERSON_SPARQL_FORM_B)
        assert f"{RDFS_SUBCLASS}>*" in out
        assert f"<{TYPES_NS}Person>" in out

    def test_idempotent(self):
        from cograph_client.graph.ontology_queries import rewrite_type_predicate_to_closure
        once = rewrite_type_predicate_to_closure(self.PERSON_SPARQL_FORM_A)
        twice = rewrite_type_predicate_to_closure(once)
        assert once == twice, "rewrite_type_predicate_to_closure must be idempotent"

    def test_with_subclass_closure_seam(self):
        """with_subclass_closure() must return the expected property-path string
        independent of the type name passed in.
        """
        from cograph_client.graph.ontology_queries import with_subclass_closure
        path = with_subclass_closure("Person")
        assert f"{RDFS_SUBCLASS}>" in path
        # The path must include the '*' quantifier for transitive closure.
        assert "*" in path

    def test_leaf_type_rewrite_covers_leaf_instances(self):
        """A query for TrialSubscriber rewritten with the closure path is
        semantically correct: the closure over the leaf is set-equal to the
        leaf itself (no narrowing possible), so applying the rewrite is safe.
        """
        from cograph_client.graph.ontology_queries import rewrite_type_predicate_to_closure
        leaf_sparql = (
            f"SELECT ?s WHERE {{ ?s a <{TYPES_NS}TrialSubscriber> . }}"
        )
        out = rewrite_type_predicate_to_closure(leaf_sparql)
        assert f"<{TYPES_NS}TrialSubscriber>" in out
        assert f"{RDFS_SUBCLASS}>*" in out

    def test_non_type_uri_not_rewritten(self):
        """Only objects under the cograph.tech/types/ namespace trigger rewriting.
        Other rdf:type usages (e.g., rdfs:Class, owl:Thing) must be left alone.
        """
        from cograph_client.graph.ontology_queries import rewrite_type_predicate_to_closure
        schema_sparql = (
            "SELECT ?s WHERE { "
            "?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> "
            "<http://www.w3.org/2000/01/rdf-schema#Class> . "
            "}"
        )
        out = rewrite_type_predicate_to_closure(schema_sparql)
        # Must NOT inject the closure path for non-types/ objects.
        assert f"{RDFS_SUBCLASS}>*" not in out


# ---------------------------------------------------------------------------
# C) INGESTION — ER merge via hierarchy walk (async, mocked Neptune)
# ---------------------------------------------------------------------------


def _make_subscriber(email: str, name: str = "Alice") -> "ExtractedEntity":
    from cograph_client.resolver.models import ExtractedAttribute, ExtractedEntity
    return ExtractedEntity(
        type_name="TrialSubscriber",
        id=email,
        parent_type="Subscriber",
        attributes=[
            ExtractedAttribute(name="email", value=email, datatype="string"),
            ExtractedAttribute(name="name", value=name, datatype="string"),
        ],
    )


class TestERMergeViaHierarchy:
    """ER fires on TrialSubscriber because config_for_with_hierarchy(
    'TrialSubscriber', {'TrialSubscriber':'Subscriber', 'Subscriber':'Person'})
    returns DEFAULT_GUEST_CONFIG.  Two signups with the same email should
    AUTO_MERGE to one canonical URI.  Without hierarchy-aware config, they
    would get distinct URIs (config_for('TrialSubscriber') is None).
    """

    SAAS_PARENT_OF = {
        "TrialSubscriber": "Subscriber",
        "Subscriber": "Person",
    }

    @pytest.mark.asyncio
    async def test_two_trial_subscribers_merge_on_email(self, mock_neptune):
        """Feed two TrialSubscriber entities with the same email through the
        ER pipeline.  The second entity should find the first as a candidate
        and AUTO_MERGE because email is decisive in DEFAULT_GUEST_CONFIG.

        We mock Neptune so that:
          - _blocker.candidates_with_signals returns the FIRST entity's URI
            with its stored signals for the SECOND call.
          - The first call gets no candidates (entity is new).
        """
        from unittest.mock import AsyncMock, patch
        from cograph_client.resolver.er.engine import ERPipeline
        from cograph_client.resolver.er.types import (
            DEFAULT_GUEST_CONFIG,
            MergeAction,
            config_for_with_hierarchy,
        )
        from cograph_client.resolver.er.normalize import DefaultNormalizer
        from cograph_client.resolver.er.blocking import generate_block_keys

        SHARED_EMAIL = "alice@startup.io"
        entity1 = _make_subscriber(SHARED_EMAIL, "Alice")
        entity2 = _make_subscriber(SHARED_EMAIL, "Alice K.")

        type_name = "TrialSubscriber"
        type_uri_str = f"https://cograph.tech/types/{type_name}"
        instance_graph = "https://cograph.tech/graphs/test-tenant"
        entity1_uri = f"https://cograph.tech/entities/{type_name}/alice_startup_io"

        # Verify the config is reachable via hierarchy.
        cfg = config_for_with_hierarchy(type_name, self.SAAS_PARENT_OF)
        assert cfg is DEFAULT_GUEST_CONFIG, (
            "Pre-condition: config_for_with_hierarchy must return DEFAULT_GUEST_CONFIG "
            "for TrialSubscriber — if this fails, the whole ER test is moot"
        )

        # Compute normalized signals for entity1 so we can simulate the blocker
        # returning entity1 as a candidate when entity2 is processed.
        from cograph_client.resolver.er.engine import extract_signals
        normalizer = DefaultNormalizer()
        raw1 = extract_signals(entity1)
        norm1 = normalizer.normalize(raw1)

        pipeline = ERPipeline(mock_neptune)

        # -- First call: no candidates → SKIP (entity1 is new) --
        with patch.object(
            pipeline._blocker, "candidates_with_signals",
            new=AsyncMock(return_value={}),
        ):
            decision1 = await pipeline.find_match(
                entity1, type_name, type_uri_str, instance_graph,
                config=cfg, parent_of=self.SAAS_PARENT_OF,
            )
        assert decision1.action == MergeAction.SKIP, (
            "First entity has no candidates yet → should be new (SKIP)"
        )

        # -- Second call: entity1 is a candidate with identical signals → AUTO_MERGE --
        with patch.object(
            pipeline._blocker, "candidates_with_signals",
            new=AsyncMock(return_value={entity1_uri: norm1}),
        ):
            decision2 = await pipeline.find_match(
                entity2, type_name, type_uri_str, instance_graph,
                config=cfg, parent_of=self.SAAS_PARENT_OF,
            )
        assert decision2.action == MergeAction.AUTO_MERGE, (
            "Second entity shares email with entity1 → email is decisive → AUTO_MERGE"
        )
        assert decision2.canonical_uri == entity1_uri

    @pytest.mark.asyncio
    async def test_no_er_without_hierarchy_info(self, mock_neptune):
        """Proves the DEFECT that existed before the hierarchy-aware fix:
        flat config_for('TrialSubscriber') is None → ER would SKIP → two
        identical signups get distinct URIs.

        With the fix, only passing parent_of={} (no hierarchy) reproduces
        the flat-config behavior; passing the real SAAS_PARENT_OF enables ER.
        This test verifies the OLD (broken) behavior is reproducible via
        config_for_with_hierarchy('TrialSubscriber', {}) == None.
        """
        from cograph_client.resolver.er.types import config_for_with_hierarchy
        # No hierarchy → no config for the leaf.
        cfg_no_hierarchy = config_for_with_hierarchy("TrialSubscriber", {})
        assert cfg_no_hierarchy is None, (
            "With empty parent_of, TrialSubscriber has no ERConfig — "
            "ER would SKIP and two signups would NOT merge"
        )
        # With hierarchy → config inherited from Subscriber.
        cfg_with_hierarchy = config_for_with_hierarchy(
            "TrialSubscriber", self.SAAS_PARENT_OF
        )
        assert cfg_with_hierarchy is not None, (
            "With SAAS_PARENT_OF, TrialSubscriber inherits Subscriber's ERConfig"
        )


# ---------------------------------------------------------------------------
# D) INGESTION — ancestor synthesis (mocked SchemaResolver internals)
# ---------------------------------------------------------------------------


class TestAncestorSynthesis:
    """_synthesize_ancestors must create missing ancestors (Subscriber, Person)
    when a TrialSubscriber is ingested with parent_type='Subscriber' and
    Subscriber is not yet in existing_types.
    """

    def _make_resolver(self, mock_neptune):
        """Construct a SchemaResolver with all I/O mocked out."""
        from unittest.mock import MagicMock, patch, AsyncMock
        from cograph_client.resolver.schema_resolver import SchemaResolver

        # Mock the verdict cache so it doesn't touch the filesystem.
        mock_cache = MagicMock()
        mock_cache.get = AsyncMock(return_value=None)
        mock_cache.put = AsyncMock(return_value=None)
        mock_cache.get_all_for_proposed = AsyncMock(return_value=[])

        with patch("anthropic.AsyncAnthropic"):
            resolver = SchemaResolver(
                neptune=mock_neptune,
                anthropic_key="fake-key",
                verdict_cache=mock_cache,
            )
        return resolver

    @pytest.mark.asyncio
    async def test_synthesize_creates_missing_ancestors(self, mock_neptune):
        """When TrialSubscriber is ingested with parent_type='Subscriber' and
        neither Subscriber nor Person exist in the ontology, _synthesize_ancestors
        should create them by calling neptune.update() for insert_type and
        insert_subtype on each missing ancestor.
        """
        from cograph_client.resolver.models import IngestResult

        resolver = self._make_resolver(mock_neptune)
        # Seed parent_of with the known chain so synthesis has something to walk.
        resolver._parent_of = {
            "TrialSubscriber": "Subscriber",
            "Subscriber": "Person",
        }

        # Only 'Person' is pre-existing; Subscriber is missing.
        existing_types: dict[str, str] = {"Person": ""}
        existing_attrs: dict = {"Person": {}}
        result = IngestResult(entities_extracted=1)

        await resolver._synthesize_ancestors(
            child_type="TrialSubscriber",
            parent_type="Subscriber",
            graph_uri="https://cograph.tech/graphs/test-tenant",
            existing_types=existing_types,
            existing_attrs=existing_attrs,
            result=result,
        )

        # Subscriber should now be in existing_types (created by synthesis)
        assert "Subscriber" in existing_types, (
            "_synthesize_ancestors must add the missing Subscriber ancestor to existing_types"
        )

        # Neptune.update must have been called at least once to insert the type
        assert mock_neptune.update.called, (
            "_synthesize_ancestors must call neptune.update to insert missing ancestor types"
        )

        # Check the SPARQL calls include 'Subscriber' in type insertions
        sparql_calls = [str(call) for call in mock_neptune.update.call_args_list]
        subscriber_calls = [c for c in sparql_calls if "Subscriber" in c]
        assert subscriber_calls, (
            "Expected at least one neptune.update call mentioning 'Subscriber' "
            f"but got calls: {sparql_calls}"
        )

    @pytest.mark.asyncio
    async def test_synthesize_idempotent_when_ancestors_exist(self, mock_neptune):
        """If Subscriber already exists in existing_types, synthesis skips it
        and does NOT issue redundant Neptune writes for that ancestor.
        """
        from cograph_client.resolver.models import IngestResult

        resolver = self._make_resolver(mock_neptune)
        resolver._parent_of = {
            "TrialSubscriber": "Subscriber",
            "Subscriber": "Person",
        }

        # Both Subscriber AND Person already exist.
        existing_types: dict[str, str] = {"Person": "", "Subscriber": ""}
        existing_attrs: dict = {"Person": {}, "Subscriber": {}}
        result = IngestResult(entities_extracted=1)
        call_count_before = mock_neptune.update.call_count

        await resolver._synthesize_ancestors(
            child_type="TrialSubscriber",
            parent_type="Subscriber",
            graph_uri="https://cograph.tech/graphs/test-tenant",
            existing_types=existing_types,
            existing_attrs=existing_attrs,
            result=result,
        )

        # No new types should have been added to existing_types
        assert set(existing_types.keys()) == {"Person", "Subscriber"}, (
            "No new ancestor types should be added when all ancestors already exist"
        )
        # No NEW Neptune writes should have been issued
        assert mock_neptune.update.call_count == call_count_before, (
            "_synthesize_ancestors must not re-insert ancestors that already exist"
        )
