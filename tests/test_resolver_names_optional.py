"""A ``name``/label is OPTIONAL — the schema-inference prompts must keep telling
the agent NOT to fabricate a name for entities that have none (reified
measurements, untitled events, dependent/association entities).

Like ``test_resolver_org_guard``, this is a *prompt* guard: the real effect is on
non-deterministic LLM inference and can't be unit-asserted. What we lock down is
that the guidance stays PRESENT in the inference prompts, so a future edit can't
silently drop it and regress the "made-up name for a nameless entity" failure
(e.g. a HumannessIndexScore surfacing a "<model> humanness <n>" pseudo-name that
is redundant with its measured-model + value). Asserts are intentionally
concept-level so benign rewording doesn't break them; removing the guidance does.
"""

from __future__ import annotations

from cograph_client.resolver.csv_resolver import COMPLETE_SYSTEM, REASON_SYSTEM
from cograph_client.resolver.schema_resolver import EXTRACTION_SYSTEM


def _names_optional_block() -> str:
    """The 'Names are optional' section of EXTRACTION_SYSTEM, lowercased.

    Anchored on the section HEADER ("Names are optional:" — the trailing colon
    disambiguates it from the '(see "Names are optional" below)' cross-ref
    earlier in the prompt) and bounded by the next section header. BOTH anchors
    are hard-asserted, so a rename fails loudly rather than silently slicing the
    wrong span.
    """
    text = EXTRACTION_SYSTEM
    start = text.find("Names are optional:")
    assert start != -1, (
        "EXTRACTION_SYSTEM no longer has a 'Names are optional:' section header "
        "— the no-fabricated-name guidance may have been moved or removed"
    )
    end = text.find("Lift providers / organizations", start)
    assert end != -1, (
        "EXTRACTION_SYSTEM: the 'Lift providers / organizations' section that "
        "bounds 'Names are optional' is gone — re-anchor this slice"
    )
    return text[start:end].lower()


def test_extraction_says_name_is_optional_for_nameless_entities():
    block = _names_optional_block()
    # Names the nameless entity kinds...
    assert "reified" in block or "measurement" in block
    assert "score" in block and "rating" in block
    # ...and forbids fabricating a name for them.
    assert "no proper name" in block or "not fabricate" in block


def test_extraction_keeps_named_entity_whitelist():
    """The rule must still allow a name for genuinely-named entities, so the
    model doesn't over-correct and drop names from people/orgs/products."""
    block = _names_optional_block()
    assert "person" in block and "organization" in block


def test_extraction_id_rule_is_a_handle_not_a_display_label():
    """The load-bearing edit: an id is a handle, not a display label — so a
    nameless entity's id isn't defaulted to a descriptive phrase (which would
    surface as its rdfs:label)."""
    text = EXTRACTION_SYSTEM.lower()
    assert "not a display label" in text or "stable handle" in text
    assert "structural id" in text or "structural" in text


def test_csv_reason_pass_keeps_names_optional_rule():
    text = REASON_SYSTEM.lower()
    assert "name" in text and "optional" in text
    # A reified/dependent entity must not be keyed/named on a descriptive column.
    assert "reified" in text or "measurement" in text


def test_csv_complete_pass_forbids_name_core_slot_for_promoted_types():
    text = COMPLETE_SYSTEM.lower()
    assert "name" in text
    # Promoted/dependent/measurement types get no name of their own.
    assert "no name" in text
