"""Tests for the ontology evolution resolver (COG-84).

Deterministic and OFFLINE: the single intent-parse LLM call is monkeypatched,
and the TypeMatcher cascade is replaced with a fake that returns canned
TypeMatch verdicts keyed by phrase — so no network, no Neptune, no embeddings.
"""

import pytest

from cograph_client.models.ontology import ResolutionResult, ResolvedChange
from cograph_client.resolver.models import MatchVerdict, TypeMatch
from cograph_client.resolver.ontology_resolver import (
    OntologyResolver,
    TypeInventory,
    build_inventory,
)


class FakeTypeMatcher:
    """Returns a canned TypeMatch per proposed phrase.

    ``verdicts`` maps a lowercased phrase → (resolved_name, verdict, confidence).
    Any phrase not in the map is treated as a brand-new DIFFERENT type.
    """

    def __init__(self, verdicts: dict[str, tuple[str, MatchVerdict, float]]):
        self._verdicts = verdicts

    async def match(self, proposed_type, proposed_description, existing_types):
        key = proposed_type.strip().lower()
        if key in self._verdicts:
            resolved, verdict, conf = self._verdicts[key]
            return TypeMatch(
                proposed=proposed_type,
                resolved=resolved,
                verdict=verdict,
                confidence=conf,
                is_new=verdict != MatchVerdict.SAME,
                parent_type=resolved if verdict == MatchVerdict.SUBTYPE else None,
            )
        # Unknown phrase → new type.
        return TypeMatch(
            proposed=proposed_type,
            resolved=proposed_type,
            verdict=MatchVerdict.DIFFERENT,
            confidence=0.9,
            is_new=True,
        )


def _make_resolver(verdicts, intents, backbone_links=None):
    """Build a resolver whose intent-parse returns ``intents`` and whose type
    matcher returns the canned ``verdicts``. The backbone-scaffold call is
    stubbed too (default: no links) so resolving stays offline."""
    resolver = OntologyResolver(
        openrouter_key="test-key",
        type_matcher=FakeTypeMatcher(verdicts),
    )

    async def fake_intent_llm(_user_content):
        return {"intents": intents}

    async def fake_backbone_llm(_user_content):
        return {"links": backbone_links or []}

    resolver._call_intent_llm = fake_intent_llm  # type: ignore[method-assign]
    resolver._call_backbone_llm = fake_backbone_llm  # type: ignore[method-assign]
    return resolver


# --- existing ontology snapshots used across tests ---------------------------

PERSON = TypeInventory(
    name="Person",
    description="A human being",
    attributes={"name": "string", "birth_date": "datetime"},
    relationships={"works_for": "Company"},
)
COMPANY = TypeInventory(name="Company", description="An organization")

ONTOLOGY = {"Person": PERSON, "Company": COMPANY}


@pytest.mark.asyncio
async def test_high_confidence_reuse_lands_in_applied():
    """An existing attribute on an existing SAME type → reuse, in `applied`."""
    resolver = _make_resolver(
        verdicts={"person": ("Person", MatchVerdict.SAME, 0.98)},
        intents=[{
            "subject_phrase": "person",
            "kind": "attribute",
            "name_phrase": "birth_date",
            "datatype_hint": "datetime",
        }],
    )

    result = await resolver.resolve_with_inventory("track a person's birth date", "g", ONTOLOGY)

    assert isinstance(result, ResolutionResult)
    assert len(result.applied) == 1
    assert not result.proposals
    change = result.applied[0]
    assert change.action == "reuse"
    assert change.kind == "attribute"
    assert change.subject_type == "Person"
    assert change.name == "birth_date"
    assert change.datatype_or_target == "datetime"


@pytest.mark.asyncio
async def test_extend_existing_type_lands_in_applied():
    """A NEW attribute on an existing SAME type → extend, in `applied`."""
    resolver = _make_resolver(
        verdicts={"person": ("Person", MatchVerdict.SAME, 0.97)},
        intents=[{
            "subject_phrase": "person",
            "kind": "attribute",
            "name_phrase": "nickname",
            "datatype_hint": "string",
        }],
    )

    result = await resolver.resolve_with_inventory("track a person's nickname", "g", ONTOLOGY)

    assert len(result.applied) == 1
    assert not result.proposals
    change = result.applied[0]
    assert change.action == "extend"
    assert change.subject_type == "Person"
    assert change.name == "nickname"
    assert change.datatype_or_target == "string"


@pytest.mark.asyncio
async def test_brand_new_type_lands_in_proposals_with_create():
    """An ask about a type that doesn't exist → create, in `proposals`."""
    resolver = _make_resolver(
        verdicts={},  # "spaceship" is unknown → new type
        intents=[{
            "subject_phrase": "spaceship",
            "kind": "attribute",
            "name_phrase": "max_speed",
            "datatype_hint": "float",
        }],
    )

    result = await resolver.resolve_with_inventory("track a spaceship's max speed", "g", ONTOLOGY)

    assert not result.applied
    assert len(result.proposals) == 1
    change = result.proposals[0]
    assert change.action == "create"
    assert change.subject_type == "spaceship"
    assert change.kind == "attribute"
    assert change.datatype_or_target == "float"


@pytest.mark.asyncio
async def test_relationship_to_existing_target_type():
    """datatype/target = an existing type name → kind 'relationship', range = target."""
    resolver = _make_resolver(
        verdicts={
            "person": ("Person", MatchVerdict.SAME, 0.98),
            "company": ("Company", MatchVerdict.SAME, 0.95),
        },
        intents=[{
            "subject_phrase": "person",
            "kind": "relationship",
            "name_phrase": "employed by",
            "target_phrase": "company",
        }],
    )

    result = await resolver.resolve_with_inventory(
        "track which company a person works for", "g", ONTOLOGY
    )

    # "works_for" already exists on Person (normalize_predicate fuzzy-matches
    # "employed_by" → existing predicate? it won't, so this is an extend) — the
    # key assertions are kind + range pointing at the existing target type, and
    # that an existing-target relationship is auto-appliable.
    all_changes = result.applied + result.proposals
    assert len(all_changes) == 1
    change = all_changes[0]
    assert change.kind == "relationship"
    assert change.subject_type == "Person"
    assert change.datatype_or_target == "Company"
    # Both subject and target are existing SAME types → no creation needed.
    assert change.action in ("reuse", "extend")
    assert change in result.applied


@pytest.mark.asyncio
async def test_relationship_reuses_existing_predicate():
    """An existing predicate on the subject type → reuse."""
    resolver = _make_resolver(
        verdicts={
            "person": ("Person", MatchVerdict.SAME, 0.98),
            "company": ("Company", MatchVerdict.SAME, 0.95),
        },
        intents=[{
            "subject_phrase": "person",
            "kind": "relationship",
            "name_phrase": "works_for",
            "target_phrase": "company",
        }],
    )

    result = await resolver.resolve_with_inventory(
        "track which company a person works for", "g", ONTOLOGY
    )

    assert len(result.applied) == 1
    change = result.applied[0]
    assert change.action == "reuse"
    assert change.name == "works_for"
    assert change.datatype_or_target == "Company"


@pytest.mark.asyncio
async def test_relationship_to_new_target_type_is_proposed():
    """A relationship whose target type is brand-new → create, in `proposals`."""
    resolver = _make_resolver(
        verdicts={
            "person": ("Person", MatchVerdict.SAME, 0.98),
            # "spaceship" target is unknown → new type
        },
        intents=[{
            "subject_phrase": "person",
            "kind": "relationship",
            "name_phrase": "owns",
            "target_phrase": "spaceship",
        }],
    )

    result = await resolver.resolve_with_inventory("a person owns a spaceship", "g", ONTOLOGY)

    assert not result.applied
    assert len(result.proposals) == 1
    change = result.proposals[0]
    assert change.action == "create"
    assert change.kind == "relationship"
    assert change.datatype_or_target == "spaceship"


@pytest.mark.asyncio
async def test_attribute_primitive_datatype_branch():
    """A primitive datatype_hint with no target → kind 'attribute'."""
    resolver = _make_resolver(
        verdicts={"company": ("Company", MatchVerdict.SAME, 0.96)},
        intents=[{
            "subject_phrase": "company",
            "kind": "attribute",
            "name_phrase": "employee_count",
            "datatype_hint": "integer",
        }],
    )

    result = await resolver.resolve_with_inventory(
        "track a company's employee count", "g", ONTOLOGY
    )

    all_changes = result.applied + result.proposals
    assert len(all_changes) == 1
    change = all_changes[0]
    assert change.kind == "attribute"
    assert change.datatype_or_target == "integer"
    assert change.action == "extend"  # new attr on existing Company


@pytest.mark.asyncio
async def test_non_primitive_datatype_hint_is_treated_as_relationship():
    """A datatype_hint that names another type (not a primitive) → relationship."""
    resolver = _make_resolver(
        verdicts={
            "person": ("Person", MatchVerdict.SAME, 0.98),
            "company": ("Company", MatchVerdict.SAME, 0.95),
        },
        intents=[{
            "subject_phrase": "person",
            "kind": "attribute",          # model mislabeled it
            "name_phrase": "employer",
            "datatype_hint": "Company",   # but the "datatype" is a type
        }],
    )

    result = await resolver.resolve_with_inventory("a person's employer", "g", ONTOLOGY)

    all_changes = result.applied + result.proposals
    assert len(all_changes) == 1
    change = all_changes[0]
    assert change.kind == "relationship"
    assert change.datatype_or_target == "Company"


@pytest.mark.asyncio
async def test_mid_band_confidence_same_match_is_proposed():
    """A SAME subject match below the apply floor → create (ambiguous), proposed."""
    resolver = _make_resolver(
        verdicts={"person": ("Person", MatchVerdict.SAME, 0.55)},  # below 0.70 floor
        intents=[{
            "subject_phrase": "person",
            "kind": "attribute",
            "name_phrase": "nickname",
            "datatype_hint": "string",
        }],
    )

    result = await resolver.resolve_with_inventory("track a person's nickname", "g", ONTOLOGY)

    assert not result.applied
    assert len(result.proposals) == 1
    assert result.proposals[0].action == "create"


@pytest.mark.asyncio
async def test_subtype_subject_is_proposed_as_create():
    """A SUBTYPE subject match needs a new (sub)type → create, proposed."""
    resolver = _make_resolver(
        verdicts={"engineer": ("Engineer", MatchVerdict.SUBTYPE, 0.85)},
        intents=[{
            "subject_phrase": "engineer",
            "kind": "attribute",
            "name_phrase": "specialty",
            "datatype_hint": "string",
        }],
    )

    result = await resolver.resolve_with_inventory("track an engineer's specialty", "g", ONTOLOGY)

    assert not result.applied
    assert len(result.proposals) == 1
    assert result.proposals[0].action == "create"
    assert result.proposals[0].subject_type == "Engineer"


@pytest.mark.asyncio
async def test_empty_intents_yields_empty_plan():
    resolver = _make_resolver(verdicts={}, intents=[])
    result = await resolver.resolve_with_inventory("do nothing", "g", ONTOLOGY)
    assert result.applied == []
    assert result.proposals == []


def test_build_inventory_separates_attributes_and_relationships():
    bindings = [
        {"typeLabel": "Person", "attrLabel": "name",
         "range": "http://www.w3.org/2001/XMLSchema#string"},
        {"typeLabel": "Person", "attrLabel": "works_for",
         "range": "https://cograph.tech/types/Company"},
        {"typeLabel": "Company"},
    ]
    inv = build_inventory(bindings)
    assert set(inv) == {"Person", "Company"}
    assert inv["Person"].attributes == {"name": "string"}
    assert inv["Person"].relationships == {"works_for": "Company"}
    # adapters produce the shapes the OSS primitives expect
    assert inv["Person"].relationship_predicates() == {"works_for"}
    schemas = inv["Person"].attribute_schemas()
    assert schemas["name"].datatype == "string"


@pytest.mark.asyncio
async def test_backbone_scaffolds_new_type_into_existing_types():
    """Introducing a NEW entity type also proposes wiring it into the existing
    ontology backbone (new Neighborhood → existing ZipCode / City), so the graph
    stays connected. The backbone links come back as 'create' proposals."""
    inventory = {
        "PropertyListing": TypeInventory(name="PropertyListing", description="A listing"),
        "ZipCode": TypeInventory(name="ZipCode"),
        "City": TypeInventory(name="City"),
    }
    resolver = _make_resolver(
        verdicts={"property listing": ("PropertyListing", MatchVerdict.SAME, 0.95)},
        intents=[{
            "subject_phrase": "property listing", "kind": "relationship",
            "name_phrase": "located in neighborhood", "target_phrase": "neighborhood",
        }],
        backbone_links=[
            {"subject_type": "neighborhood", "predicate": "in zip code", "target_type": "ZipCode"},
            {"subject_type": "neighborhood", "predicate": "in city", "target_type": "City"},
            # hallucinated target must be dropped
            {"subject_type": "neighborhood", "predicate": "in state", "target_type": "State"},
        ],
    )

    result = await resolver.resolve_with_inventory(
        "I wanna know which neighborhood each property listing is at", "g", inventory
    )

    rels = {(c.subject_type, c.datatype_or_target) for c in result.proposals}
    assert ("PropertyListing", "neighborhood") in rels      # the asked relationship
    assert ("neighborhood", "ZipCode") in rels              # backbone link
    assert ("neighborhood", "City") in rels                 # backbone link
    assert ("neighborhood", "State") not in rels            # hallucinated target dropped
    backbone = [c for c in result.proposals if c.subject_type == "neighborhood"]
    assert all(c.action == "create" for c in backbone)


def test_resolved_change_is_json_serializable():
    change = ResolvedChange(
        kind="relationship",
        subject_type="Person",
        name="works_for",
        datatype_or_target="Company",
        action="reuse",
        confidence=0.98,
        reason="already exists",
    )
    dumped = change.model_dump_json()
    assert '"works_for"' in dumped
    assert ResolvedChange.model_validate_json(dumped) == change
