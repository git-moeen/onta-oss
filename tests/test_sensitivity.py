"""Sensitivity tags + v1 enforcement tests (ADR 0002 §5, COG-42).

Covers cograph_client/resolver/sensitivity.py:

  - rule 1: guard_enrichment_payload strips sensitive attrs entirely
  - rule 2: filter_response_attrs strips unless entitled
  - rule 3: redact_for_log keeps keys, replaces sensitive values
  - inheritance: maps merge down the subclass chain; child re-tags override
  - default-public: untagged attrs always pass through
  - registry precedence: first registry defining 'sensitivity' wins per type
  - empty registry / shipped default registry => exact no-ops (backward-compat
    regression: the built-in OSS registry carries no sensitivity entries)

All pure — no Neptune, no LLM, no env. Tests use explicit registries so the
module-level default registry is never mutated.
"""

from __future__ import annotations

from cograph_client.resolver.sensitivity import (
    PII,
    PUBLIC,
    REDACTED,
    SECRET,
    SENSITIVITY_ENTRY,
    filter_response_attrs,
    guard_enrichment_payload,
    is_sensitive,
    redact_for_log,
    resolve_sensitivity_map,
)
from cograph_client.resolver.strategy import WELL_KNOWN_ENTRIES, default_registry

PARENT_OF = {"HotelGuest": "Guest", "Guest": "Person"}

# Person tags two attrs; everything else defaults to 'public'.
PERSON_REGISTRY = {
    "Person": {SENSITIVITY_ENTRY: {"email": PII, "api_token": SECRET}},
}

RECORD = {"name": "Ada", "email": "ada@example.com", "api_token": "tok-123", "city": "SF"}


# ---------------------------------------------------------------------------
# Tag vocabulary
# ---------------------------------------------------------------------------


class TestVocabulary:
    def test_pii_and_secret_are_sensitive(self):
        assert is_sensitive(PII)
        assert is_sensitive(SECRET)

    def test_public_and_unknown_are_not(self):
        assert not is_sensitive(PUBLIC)
        assert not is_sensitive("internal")  # unknown tags are not sensitive

    def test_sensitivity_is_a_well_known_bundle_entry(self):
        assert SENSITIVITY_ENTRY in WELL_KNOWN_ENTRIES


# ---------------------------------------------------------------------------
# Rule 1: guard_enrichment_payload
# ---------------------------------------------------------------------------


class TestGuardEnrichmentPayload:
    def test_strips_pii_and_secret_keys(self):
        guarded = guard_enrichment_payload(RECORD, "Person", {}, [PERSON_REGISTRY])
        assert guarded == {"name": "Ada", "city": "SF"}

    def test_inherited_tags_apply_to_leaf(self):
        """HotelGuest has no map of its own — Person's tags still guard it."""
        guarded = guard_enrichment_payload(RECORD, "HotelGuest", PARENT_OF, [PERSON_REGISTRY])
        assert "email" not in guarded
        assert "api_token" not in guarded

    def test_input_dict_is_not_mutated(self):
        original = dict(RECORD)
        guard_enrichment_payload(RECORD, "Person", {}, [PERSON_REGISTRY])
        assert RECORD == original


# ---------------------------------------------------------------------------
# Rule 2: filter_response_attrs
# ---------------------------------------------------------------------------


class TestFilterResponseAttrs:
    def test_non_entitled_strips_sensitive(self):
        attrs = filter_response_attrs(RECORD, "Person", {}, [PERSON_REGISTRY], entitled=False)
        assert attrs == {"name": "Ada", "city": "SF"}

    def test_entitled_sees_everything(self):
        attrs = filter_response_attrs(RECORD, "Person", {}, [PERSON_REGISTRY], entitled=True)
        assert attrs == RECORD
        # New dict, not the caller's record.
        assert attrs is not RECORD

    def test_non_entitled_leaf_inherits_tags(self):
        attrs = filter_response_attrs(
            RECORD, "HotelGuest", PARENT_OF, [PERSON_REGISTRY], entitled=False
        )
        assert attrs == {"name": "Ada", "city": "SF"}


# ---------------------------------------------------------------------------
# Rule 3: redact_for_log
# ---------------------------------------------------------------------------


class TestRedactForLog:
    def test_values_replaced_keys_kept(self):
        redacted = redact_for_log(RECORD, "Person", {}, [PERSON_REGISTRY])
        assert redacted == {
            "name": "Ada",
            "email": REDACTED,
            "api_token": REDACTED,
            "city": "SF",
        }

    def test_inherited_tags_redact_leaf_record(self):
        redacted = redact_for_log(RECORD, "HotelGuest", PARENT_OF, [PERSON_REGISTRY])
        assert redacted["email"] == REDACTED
        assert redacted["name"] == "Ada"

    def test_input_dict_is_not_mutated(self):
        original = dict(RECORD)
        redact_for_log(RECORD, "Person", {}, [PERSON_REGISTRY])
        assert RECORD == original


# ---------------------------------------------------------------------------
# Inheritance: merge down the chain, child re-tag overrides parent
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_child_overrides_single_attr_keeps_rest(self):
        """Guest re-tags email back to public; Person's api_token tag still
        applies — merge, not nearest-ancestor-wins-wholesale."""
        registry = {
            "Person": {SENSITIVITY_ENTRY: {"email": PII, "api_token": SECRET}},
            "Guest": {SENSITIVITY_ENTRY: {"email": PUBLIC}},
        }
        smap = resolve_sensitivity_map("HotelGuest", PARENT_OF, [registry])
        assert smap == {"email": PUBLIC, "api_token": SECRET}
        guarded = guard_enrichment_payload(RECORD, "HotelGuest", PARENT_OF, [registry])
        assert "email" in guarded
        assert "api_token" not in guarded

    def test_child_can_tighten_a_public_attr(self):
        """The reverse override: child re-tags an inherited-public attr as pii."""
        registry = {
            "Person": {SENSITIVITY_ENTRY: {"city": PUBLIC}},
            "Guest": {SENSITIVITY_ENTRY: {"city": PII}},
        }
        assert resolve_sensitivity_map("HotelGuest", PARENT_OF, [registry]) == {"city": PII}
        assert resolve_sensitivity_map("Person", PARENT_OF, [registry]) == {"city": PUBLIC}

    def test_registry_precedence_shadows_per_type(self):
        """For the SAME type, the first registry's map wins outright (tenant
        shadows public, mirroring resolve_entry) — maps do not merge across
        registries at one type."""
        tenant = {"Guest": {SENSITIVITY_ENTRY: {"email": PUBLIC}}}
        public = {"Guest": {SENSITIVITY_ENTRY: {"email": PII, "phone": PII}}}
        smap = resolve_sensitivity_map("Guest", PARENT_OF, [tenant, public])
        assert smap == {"email": PUBLIC}

    def test_cyclic_parent_of_terminates(self):
        """Malformed subClassOf cycles can't spin forever (ancestor_chain guard)."""
        cyclic = {"A": "B", "B": "A"}
        registry = {"B": {SENSITIVITY_ENTRY: {"x": PII}}}
        assert resolve_sensitivity_map("A", cyclic, [registry]) == {"x": PII}


# ---------------------------------------------------------------------------
# Default-public
# ---------------------------------------------------------------------------


class TestDefaultPublic:
    def test_untagged_attr_passes_every_rule(self):
        """'city' is absent from Person's map — defaults to 'public' and
        survives all three rules."""
        assert "city" in guard_enrichment_payload(RECORD, "Person", {}, [PERSON_REGISTRY])
        assert "city" in filter_response_attrs(
            RECORD, "Person", {}, [PERSON_REGISTRY], entitled=False
        )
        assert redact_for_log(RECORD, "Person", {}, [PERSON_REGISTRY])["city"] == "SF"

    def test_explicit_public_tag_behaves_like_absent(self):
        registry = {"Person": {SENSITIVITY_ENTRY: {"city": PUBLIC}}}
        assert guard_enrichment_payload(RECORD, "Person", {}, [registry]) == RECORD


# ---------------------------------------------------------------------------
# Empty registry / shipped defaults => no-ops (backward-compat regression)
# ---------------------------------------------------------------------------


class TestNoOpRegression:
    def test_no_registries_is_a_noop(self):
        assert guard_enrichment_payload(RECORD, "Person", PARENT_OF, []) == RECORD
        assert filter_response_attrs(RECORD, "Person", PARENT_OF, [], entitled=False) == RECORD
        assert redact_for_log(RECORD, "Person", PARENT_OF, []) == RECORD

    def test_empty_registry_is_a_noop(self):
        assert guard_enrichment_payload(RECORD, "Person", PARENT_OF, [{}]) == RECORD
        assert filter_response_attrs(RECORD, "Person", PARENT_OF, [{}], entitled=False) == RECORD
        assert redact_for_log(RECORD, "Person", PARENT_OF, [{}]) == RECORD

    def test_shipped_default_registry_is_a_noop(self):
        """The built-in OSS registry carries only 'er' entries — no type ships
        sensitivity tags, so default behavior is untouched."""
        registry = default_registry()
        assert not any(SENSITIVITY_ENTRY in bundle for bundle in registry.values())
        assert guard_enrichment_payload(RECORD, "Guest", PARENT_OF, [registry]) == RECORD
        assert redact_for_log(RECORD, "Guest", PARENT_OF, [registry]) == RECORD

    def test_unknown_type_is_a_noop(self):
        assert resolve_sensitivity_map("Nonexistent", PARENT_OF, [PERSON_REGISTRY]) == {}
        assert guard_enrichment_payload(RECORD, "Nonexistent", PARENT_OF, [PERSON_REGISTRY]) == RECORD
