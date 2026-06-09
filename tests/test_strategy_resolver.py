"""Strategy-bundle resolver tests (ADR 0002 §3, COG-39).

Covers the generalized chain-walking resolver in cograph_client/resolver/strategy.py:

  - chain-walk inheritance for a non-ER entry (sensitivity tags)
  - precedence across ordered registries (tenant overrides public for the
    same type; type specificity outranks registry order)
  - ER wrapper equivalence: config_for_with_hierarchy through the new
    resolver matches the old DEFAULTS_BY_TYPE chain walk exactly, on the
    real DEFAULTS_BY_TYPE (backward-compat regression)
  - unknown type / unknown entry -> None
  - register_strategy seam

All pure — no Neptune, no LLM, no env. Registration tests use explicit
registries so the module-level default registry is never mutated.
"""

from __future__ import annotations

from cograph_client.resolver.er.types import (
    DEFAULT_CUSTOMER_CONFIG,
    DEFAULT_GUEST_CONFIG,
    DEFAULTS_BY_TYPE,
    ancestor_chain,
    config_for,
    config_for_with_hierarchy,
)
from cograph_client.resolver.strategy import (
    WELL_KNOWN_ENTRIES,
    default_registry,
    register_strategy,
    resolve_entry,
)

PARENT_OF = {"HotelGuest": "Guest", "Guest": "Person"}


# ---------------------------------------------------------------------------
# Chain-walk inheritance for a non-ER entry
# ---------------------------------------------------------------------------


class TestChainWalkInheritance:
    def test_leaf_inherits_ancestor_sensitivity(self):
        """HotelGuest has no 'sensitivity' entry; Person does — the chain walk
        climbs HotelGuest -> Guest -> Person and returns Person's tags."""
        registry = {"Person": {"sensitivity": ("pii",)}}
        tags = resolve_entry("HotelGuest", "sensitivity", PARENT_OF, [registry])
        assert tags == ("pii",)

    def test_nearest_ancestor_wins(self):
        """An entry on Guest shadows the one on Person for HotelGuest —
        nearest configured ancestor, not the root."""
        registry = {
            "Guest": {"enrich": "guest-enrich"},
            "Person": {"enrich": "person-enrich"},
        }
        assert resolve_entry("HotelGuest", "enrich", PARENT_OF, [registry]) == "guest-enrich"
        # Person itself still resolves to its own entry.
        assert resolve_entry("Person", "enrich", PARENT_OF, [registry]) == "person-enrich"

    def test_entries_resolve_independently(self):
        """Bundle entries inherit independently: 'verify' from Guest,
        'sensitivity' from Person — not all-or-nothing per bundle."""
        registry = {
            "Guest": {"verify": "guest-verify"},
            "Person": {"sensitivity": ("pii",)},
        }
        assert resolve_entry("HotelGuest", "verify", PARENT_OF, [registry]) == "guest-verify"
        assert resolve_entry("HotelGuest", "sensitivity", PARENT_OF, [registry]) == ("pii",)

    def test_empty_parent_of_is_flat(self):
        """With no hierarchy the chain is just [type_name] — flat lookup."""
        registry = {"Person": {"sensitivity": ("pii",)}}
        assert resolve_entry("HotelGuest", "sensitivity", {}, [registry]) is None
        assert resolve_entry("Person", "sensitivity", {}, [registry]) == ("pii",)

    def test_cyclic_parent_of_terminates(self):
        """Malformed subClassOf cycles can't spin forever (ancestor_chain guard)."""
        cyclic = {"A": "B", "B": "A"}
        assert resolve_entry("A", "search", cyclic, [{}]) is None


# ---------------------------------------------------------------------------
# Precedence across ordered registries (tenant > enhanced > public)
# ---------------------------------------------------------------------------


class TestRegistryPrecedence:
    def test_tenant_overrides_public_for_same_type(self):
        """First registry in the list wins for the same type — mirroring
        LayerStack shadowing (tenant first, public last)."""
        tenant = {"Guest": {"enrich": "tenant-enrich"}}
        public = {"Guest": {"enrich": "public-enrich"}}
        assert resolve_entry("Guest", "enrich", {}, [tenant, public]) == "tenant-enrich"
        # Public alone still resolves.
        assert resolve_entry("Guest", "enrich", {}, [public]) == "public-enrich"

    def test_type_specificity_outranks_registry_order(self):
        """At each type in the chain, registries are checked in order BEFORE
        climbing: public's entry on Guest itself beats tenant's entry on the
        Person ancestor."""
        tenant = {"Person": {"enrich": "tenant-person-enrich"}}
        public = {"Guest": {"enrich": "public-guest-enrich"}}
        assert (
            resolve_entry("HotelGuest", "enrich", PARENT_OF, [tenant, public])
            == "public-guest-enrich"
        )

    def test_missing_entry_falls_through_to_next_registry(self):
        """A bundle that defines OTHER keys doesn't block resolution of the
        requested key from a lower registry."""
        tenant = {"Guest": {"verify": "tenant-verify"}}
        public = {"Guest": {"enrich": "public-enrich"}}
        assert resolve_entry("Guest", "enrich", {}, [tenant, public]) == "public-enrich"

    def test_no_registries_is_none(self):
        assert resolve_entry("Guest", "enrich", PARENT_OF, []) is None


# ---------------------------------------------------------------------------
# ER wrapper equivalence (backward-compat regression, real DEFAULTS_BY_TYPE)
# ---------------------------------------------------------------------------


def _old_config_for_with_hierarchy(type_name, parent_of):
    """The pre-COG-39 implementation, verbatim: first ancestor in
    DEFAULTS_BY_TYPE wins."""
    for ancestor in ancestor_chain(type_name, parent_of):
        cfg = DEFAULTS_BY_TYPE.get(ancestor)
        if cfg is not None:
            return cfg
    return None


class TestERWrapperEquivalence:
    def test_equivalence_on_real_defaults(self):
        """For every configured type, plus novel subtypes and unknowns, the
        wrapper returns the IDENTICAL object the old implementation did."""
        parent_of = {
            **PARENT_OF,
            "LoyaltyCustomer": "Customer",
            "HospitalPatient": "Patient",
            "Condo": "Property",
            "Widget": "Gadget",  # chain with no configured ancestor
        }
        probes = list(DEFAULTS_BY_TYPE) + [
            "HotelGuest", "LoyaltyCustomer", "HospitalPatient", "Condo",
            "Widget", "Nonexistent",
        ]
        for type_name in probes:
            assert config_for_with_hierarchy(type_name, parent_of) is (
                _old_config_for_with_hierarchy(type_name, parent_of)
            ), f"wrapper diverged from old behavior for {type_name}"
            # And flat (empty parent_of), as config_for uses.
            assert config_for_with_hierarchy(type_name, {}) is (
                DEFAULTS_BY_TYPE.get(type_name)
            ), f"flat wrapper diverged for {type_name}"

    def test_leaf_inherits_guest_config(self):
        """The load-bearing ER case: HotelGuest (not in DEFAULTS_BY_TYPE)
        inherits DEFAULT_GUEST_CONFIG via the chain."""
        assert config_for("HotelGuest") is None
        assert config_for_with_hierarchy("HotelGuest", PARENT_OF) is DEFAULT_GUEST_CONFIG

    def test_default_registry_mirrors_defaults_by_type(self):
        """The migrated registry carries exactly DEFAULTS_BY_TYPE as 'er'
        entries — same keys, same config objects."""
        registry = default_registry()
        assert set(registry) == set(DEFAULTS_BY_TYPE)
        for type_name, cfg in DEFAULTS_BY_TYPE.items():
            assert registry[type_name]["er"] is cfg


# ---------------------------------------------------------------------------
# Unknown type / unknown entry
# ---------------------------------------------------------------------------


class TestUnknowns:
    def test_unknown_type_is_none(self):
        assert resolve_entry("Nonexistent", "er", PARENT_OF, [default_registry()]) is None

    def test_unknown_entry_is_none(self):
        """A type with a bundle but no such entry: 'verify' is well-known but
        the default registry only carries 'er'."""
        assert "verify" in WELL_KNOWN_ENTRIES
        assert resolve_entry("Guest", "verify", PARENT_OF, [default_registry()]) is None

    def test_unknown_type_and_entry_is_none(self):
        assert resolve_entry("Nonexistent", "no-such-key", {}, [default_registry()]) is None


# ---------------------------------------------------------------------------
# register_strategy seam
# ---------------------------------------------------------------------------


class TestRegisterStrategy:
    def test_register_into_explicit_registry(self):
        """Premium-style usage: a separate registry passed ahead of the
        default, attaching a non-ER strategy that the leaf inherits."""
        premium: dict[str, dict] = {}
        register_strategy("Guest", "verify", "premium-verifier", registry=premium)
        assert (
            resolve_entry("HotelGuest", "verify", PARENT_OF, [premium, default_registry()])
            == "premium-verifier"
        )
        # The default registry is untouched.
        assert "verify" not in default_registry().get("Guest", {})

    def test_register_overwrites_same_key(self):
        registry: dict[str, dict] = {}
        register_strategy("Guest", "search", "v1", registry=registry)
        register_strategy("Guest", "search", "v2", registry=registry)
        assert resolve_entry("Guest", "search", {}, [registry]) == "v2"

    def test_register_preserves_other_entries(self):
        registry = {"Guest": {"er": "existing-er"}}
        register_strategy("Guest", "enrich", "new-enrich", registry=registry)
        assert registry["Guest"] == {"er": "existing-er", "enrich": "new-enrich"}

    def test_register_er_override_flows_through_wrapper(self):
        """An 'er' entry registered on the DEFAULT registry is what
        config_for_with_hierarchy resolves — the seam premium ER tuning uses.
        Restores the registry afterwards so no state leaks across tests."""
        registry = default_registry()
        assert "Condo" not in registry
        try:
            register_strategy("Condo", "er", DEFAULT_CUSTOMER_CONFIG)
            assert config_for_with_hierarchy("Condo", {}) is DEFAULT_CUSTOMER_CONFIG
        finally:
            registry.pop("Condo", None)
        assert config_for_with_hierarchy("Condo", {}) is None
