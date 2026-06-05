"""Gap-closure tests for ADR 0001 multi-typing (COG-33 follow-up).

Covers the two pieces deferred from the initial implementation:

  Gap 2 — full parent_chain lineage closure: a brand-new MULTI-LEVEL hierarchy
          (Condo < Property < Asset, none pre-existing) closes in one ingest row
          via ExtractedEntity.parent_chain.

  Gap 1 — multi-type write: ExtractedEntity.also_types produces an additional
          asserted rdf:type per genuine independent co-classification, while
          same-lineage "co-types" (an ancestor/descendant) are skipped.

All mocked — no live Neptune, no LLM. An empty ontology makes TypeMatcher
short-circuit (no model call), so _resolve_type can be exercised end-to-end.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.ontology_queries import type_uri
from cograph_client.resolver.models import ExtractedAttribute, ExtractedEntity, IngestResult
from cograph_client.resolver.schema_resolver import SchemaResolver


RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"


@pytest.fixture
def mock_neptune():
    client = AsyncMock(spec=NeptuneClient)
    client.health.return_value = True
    client.query.return_value = {"head": {"vars": []}, "results": {"bindings": []}}
    client.update.return_value = None
    return client


@pytest.fixture
def resolver(mock_neptune):
    verdict_path = Path(tempfile.mkdtemp()) / "verdicts.json"
    from cograph_client.resolver.verdict_cache import JsonVerdictCache

    with patch.dict("os.environ", {
        "ANTHROPIC_API_KEY": "test-key",
        "OPENROUTER_API_KEY": "test-or-key",
        "COGRAPH_ER_ENABLED": "0",
    }):
        return SchemaResolver(
            neptune=mock_neptune,
            anthropic_key="test-key",
            verdict_cache=JsonVerdictCache(verdict_path),
        )


def _update_sparql(mock_neptune) -> list[str]:
    return [c.args[0] for c in mock_neptune.update.call_args_list]


# ---------------------------------------------------------------------------
# Gap 2 — full parent_chain closes a brand-new multi-level lineage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_brand_new_lineage_closes_via_parent_chain(resolver, mock_neptune):
    """Condo < Property < Asset, none pre-existing. Ingesting one Condo with
    parent_chain=['Property','Asset'] must create all three types and both
    subClassOf edges — the case the old single-hint synthesis could not close.
    """
    entity = ExtractedEntity(
        type_name="Condo", id="unit-5B", parent_chain=["Property", "Asset"],
    )
    existing_types: dict[str, str] = {}
    existing_attrs: dict[str, dict] = {}
    result = IngestResult(entities_extracted=1)

    resolved = await resolver._resolve_type(
        entity, "g", existing_types, existing_attrs, result,
    )
    assert resolved == "Condo"  # leaf stays most-specific

    sparql = " || ".join(_update_sparql(mock_neptune))
    # All three types created.
    for t in ("Condo", "Property", "Asset"):
        assert type_uri(t) in sparql, f"{t} type was not created"
        assert t in result.types_created, f"{t} missing from types_created"
    # Both subClassOf edges present (child->parent for each consecutive pair).
    assert resolver._parent_of["Condo"] == "Property"
    assert resolver._parent_of["Property"] == "Asset"


@pytest.mark.asyncio
async def test_existing_parent_plus_deeper_new_chain(resolver, mock_neptune):
    """Property already exists; ingest Condo with an immediate existing parent
    AND a deeper new ancestor (Asset). Condo links to existing Property and
    Asset is synthesized above it. Tested via _link_parent to avoid the LLM
    path that a non-empty ontology would trigger in _resolve_type.
    """
    entity = ExtractedEntity(
        type_name="Condo", id="unit-5B",
        parent_type="Property", parent_chain=["Property", "Asset"],
    )
    existing_types = {"Property": ""}
    existing_attrs: dict[str, dict] = {"Property": {}}
    result = IngestResult(entities_extracted=1)

    await resolver._link_parent(entity, "g", existing_types, existing_attrs, result)

    sparql = " || ".join(_update_sparql(mock_neptune))
    assert type_uri("Asset") in sparql and "Asset" in result.types_created
    assert resolver._parent_of["Condo"] == "Property"
    assert resolver._parent_of["Property"] == "Asset"
    # Property already existed — must NOT be recreated.
    assert "Property" not in result.types_created


# ---------------------------------------------------------------------------
# Gap 1 — also_types: multi-type write + same-lineage guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_also_types_emit_extra_rdf_type(resolver):
    """A genuine co-classification (Employee that is also a Guest) yields two
    asserted rdf:type triples on the same instance URI."""
    entity = ExtractedEntity(type_name="Employee", id="emp-1")
    collected: list[tuple[str, str, str]] = []
    result = IngestResult(entities_extracted=1)
    uri = "https://cograph.tech/entities/Employee/emp-1"

    await resolver._resolve_and_insert_entity(
        entity, "Employee", uri, is_duplicate=False,
        graph_uri="g", existing_types={"Employee": "", "Guest": ""},
        existing_attrs={"Employee": {}, "Guest": {}}, source="", result=result,
        _collect_triples=collected, also_types=["Guest"],
    )

    type_triples = [(s, o) for (s, p, o) in collected if p == RDF_TYPE]
    assert (uri, type_uri("Employee")) in type_triples
    assert (uri, type_uri("Guest")) in type_triples, "co-type rdf:type not asserted"


@pytest.mark.asyncio
async def test_resolve_also_types_skips_ancestor(resolver):
    """A 'co-type' that is actually an ancestor of the primary (Guest above
    HotelGuest) is recovered by closure, not asserted — so it's skipped."""
    resolver._parent_of = {"HotelGuest": "Guest"}
    entity = ExtractedEntity(type_name="HotelGuest", id="g1", also_types=["Guest"])
    result = IngestResult(entities_extracted=1)

    got = await resolver._resolve_also_types(
        entity, "HotelGuest", "g",
        {"HotelGuest": "", "Guest": ""}, {"HotelGuest": {}, "Guest": {}}, result,
    )
    assert got == []


@pytest.mark.asyncio
async def test_resolve_also_types_keeps_independent(resolver):
    """A genuinely independent co-type (Guest, unrelated to Employee in the
    hierarchy) is kept."""
    resolver._parent_of = {}
    entity = ExtractedEntity(type_name="Employee", id="e1", also_types=["Guest"])
    result = IngestResult(entities_extracted=1)

    got = await resolver._resolve_also_types(
        entity, "Employee", "g",
        {"Employee": "", "Guest": ""}, {"Employee": {}, "Guest": {}}, result,
    )
    assert got == ["Guest"]


@pytest.mark.asyncio
async def test_single_type_unchanged_no_extra_triples(resolver):
    """Regression guard: the common single-type path (no also_types) still emits
    exactly one rdf:type triple."""
    entity = ExtractedEntity(type_name="Guest", id="g9")
    collected: list[tuple[str, str, str]] = []
    result = IngestResult(entities_extracted=1)
    uri = "https://cograph.tech/entities/Guest/g9"

    await resolver._resolve_and_insert_entity(
        entity, "Guest", uri, is_duplicate=False,
        graph_uri="g", existing_types={"Guest": ""}, existing_attrs={"Guest": {}},
        source="", result=result, _collect_triples=collected, also_types=None,
    )
    type_triples = [t for t in collected if t[1] == RDF_TYPE]
    assert len(type_triples) == 1 and type_triples[0][2] == type_uri("Guest")


# ---------------------------------------------------------------------------
# Gap 3 — query-time subclass-closure rewriter (Form A/B/C + idempotency)
# ---------------------------------------------------------------------------


CLOSURE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>/<http://www.w3.org/2000/01/rdf-schema#subClassOf>*"
PERSON = "<https://cograph.tech/types/Person>"


@pytest.mark.parametrize("query_fragment", [
    f"?p a {PERSON}",                                                      # Form A: `a`
    f"?p <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> {PERSON}",      # Form B: full IRI
    f"?p rdf:type {PERSON}",                                               # Form C: prefixed
])
def test_closure_rewrite_all_forms(query_fragment):
    from cograph_client.graph.ontology_queries import rewrite_type_predicate_to_closure
    out = rewrite_type_predicate_to_closure(f"SELECT ?p WHERE {{ {query_fragment} }}")
    assert f"?p {CLOSURE} {PERSON}" in out


def test_closure_rewrite_is_idempotent():
    from cograph_client.graph.ontology_queries import rewrite_type_predicate_to_closure
    once = rewrite_type_predicate_to_closure(f"SELECT ?p WHERE {{ ?p a {PERSON} }}")
    twice = rewrite_type_predicate_to_closure(once)
    assert once == twice
    assert twice.count("subClassOf>*") == 1


def test_closure_rewrite_leaves_non_type_triples_alone():
    from cograph_client.graph.ontology_queries import rewrite_type_predicate_to_closure
    # A normal predicate whose object is a types URI must NOT be rewritten.
    q = "SELECT ?p WHERE { ?p <https://cograph.tech/onto/works_at> ?c }"
    assert rewrite_type_predicate_to_closure(q) == q
