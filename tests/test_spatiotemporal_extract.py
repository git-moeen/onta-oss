"""Datatype-driven extraction + automatic write-path population of the
spatio-temporal index.

Covers :func:`extract_spatiotemporal_facts` (triples → facts) and the auto-index
hook inside :func:`cograph_client.graph.kg_writer.insert_facts` (every converged
writer indexes its geometry-bearing entities, scoped per-KG, best-effort).
"""

from __future__ import annotations

from datetime import datetime, timezone

from cograph_client.graph.kg_writer import insert_facts
from cograph_client.graph.queries import kg_graph_uri, parse_kg_graph_uri
from cograph_client.spatiotemporal.extract import extract_spatiotemporal_facts
from cograph_client.spatiotemporal.registry import (
    get_spatiotemporal_index,
    register_spatiotemporal_index,
    reset_spatiotemporal_index,
)

import pytest

GEO = "http://www.opengis.net/ont/geosparql#wktLiteral"
DT = "http://www.w3.org/2001/XMLSchema#dateTime"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

TENANT = "demo-tenant"
KG = "EventsSF"


def _dt(y: int, m: int = 1, d: int = 1) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _geom(uri: str, lon: float, lat: float) -> tuple:
    return (uri, f"https://cograph.tech/types/T/loc", f"POINT({lon} {lat})^^{GEO}")


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_spatiotemporal_index()
    yield
    reset_spatiotemporal_index()


# ---------------------------------------------------------------------------
# extract_spatiotemporal_facts
# ---------------------------------------------------------------------------


def test_extracts_point_from_wkt_literal():
    facts = extract_spatiotemporal_facts(
        [_geom("e:1", 2.29, 48.85)], tenant_id=TENANT, kg_name=KG
    )
    assert len(facts) == 1
    f = facts[0]
    assert (f.lon, f.lat) == (2.29, 48.85)
    assert f.tenant_id == TENANT and f.kg_name == KG
    assert f.valid_from is None and f.valid_to is None


def test_entity_without_geometry_is_skipped():
    triples = [
        ("e:person", RDF_TYPE, "https://cograph.tech/types/Person"),
        ("e:person", "https://cograph.tech/types/Person/birth_date", f"1970-01-01T00:00:00^^{DT}"),
    ]
    assert extract_spatiotemporal_facts(triples, tenant_id=TENANT, kg_name=KG) == []


def test_lone_date_is_not_validity():
    """A single non-validity date must NOT become valid_time (we don't guess)."""
    triples = [
        _geom("e:place", 2.29, 48.85),
        ("e:place", "https://cograph.tech/types/Place/founded", f"1889-03-31T00:00:00^^{DT}"),
    ]
    f = extract_spatiotemporal_facts(triples, tenant_id=TENANT, kg_name=KG)[0]
    assert f.valid_from is None and f.valid_to is None


def test_start_end_pair_becomes_validity():
    triples = [
        _geom("e:expo", 2.30, 48.86),
        ("e:expo", "https://cograph.tech/types/Event/start_date", f"2024-06-01T00:00:00^^{DT}"),
        ("e:expo", "https://cograph.tech/types/Event/end_date", f"2024-06-10T00:00:00^^{DT}"),
    ]
    f = extract_spatiotemporal_facts(triples, tenant_id=TENANT, kg_name=KG)[0]
    assert f.valid_from == _dt(2024, 6, 1) and f.valid_to == _dt(2024, 6, 10)


def test_inverted_start_end_pair_opens_validity():
    """from > to would make PostGIS tstzrange raise and drop the write batch — the
    extractor discards an inverted range to open validity instead (entity still
    indexed by its geometry)."""
    triples = [
        _geom("e:bad", 2.30, 48.86),
        ("e:bad", "https://cograph.tech/types/Event/start_date", f"2024-06-10T00:00:00^^{DT}"),
        ("e:bad", "https://cograph.tech/types/Event/end_date", f"2024-06-01T00:00:00^^{DT}"),
    ]
    facts = extract_spatiotemporal_facts(triples, tenant_id=TENANT, kg_name=KG)
    assert len(facts) == 1  # still indexed
    assert facts[0].valid_from is None and facts[0].valid_to is None


def test_parse_kg_graph_uri_rejects_companion_graph():
    """A provenance/companion graph (extra path segment) must NOT greedily parse to
    kg_name='<kg>/provenance' — it returns None so only true per-KG graphs route."""
    assert parse_kg_graph_uri(kg_graph_uri(TENANT, KG)) == (TENANT, KG)
    assert parse_kg_graph_uri(kg_graph_uri(TENANT, KG) + "/provenance") is None
    assert parse_kg_graph_uri("https://cograph.tech/graphs/demo-tenant") is None


def test_explicit_valid_from_only_open_ended():
    triples = [
        _geom("e:site", 2.29, 48.85),
        ("e:site", "https://cograph.tech/types/Site/valid_from", f"2020-01-01T00:00:00^^{DT}"),
    ]
    f = extract_spatiotemporal_facts(triples, tenant_id=TENANT, kg_name=KG)[0]
    assert f.valid_from == _dt(2020) and f.valid_to is None


def test_denormalizes_label_and_type():
    triples = [
        ("e:1", RDF_TYPE, "https://cograph.tech/types/Venue"),
        ("e:1", RDFS_LABEL, "Ferry Building"),
        _geom("e:1", -122.39, 37.79),
    ]
    f = extract_spatiotemporal_facts(triples, tenant_id=TENANT, kg_name=KG)[0]
    assert f.attrs == {"label": "Ferry Building", "type": "Venue"}  # PascalCase type kept


def test_out_of_range_point_ignored():
    bad = ("e:bad", "https://cograph.tech/types/T/loc", f"POINT(999 999)^^{GEO}")
    assert extract_spatiotemporal_facts([bad], tenant_id=TENANT, kg_name=KG) == []


def test_order_preserved_and_multiple_entities():
    triples = [_geom("e:a", 1.0, 1.0), _geom("e:b", 2.0, 2.0)]
    facts = extract_spatiotemporal_facts(triples, tenant_id=TENANT, kg_name=KG)
    assert [f.entity_uri for f in facts] == ["e:a", "e:b"]


def test_plain_string_with_caret_not_mistyped():
    """A plain string literal containing '^^' (no http tail) is not a typed value."""
    triples = [
        _geom("e:1", 2.29, 48.85),
        ("e:1", "https://cograph.tech/types/T/note", "a^^b weird value"),
    ]
    facts = extract_spatiotemporal_facts(triples, tenant_id=TENANT, kg_name=KG)
    assert len(facts) == 1  # the note neither breaks parsing nor adds a fact


# ---------------------------------------------------------------------------
# insert_facts auto-population
# ---------------------------------------------------------------------------


class _FakeNeptune:
    def __init__(self) -> None:
        self.updates: list[str] = []

    async def update(self, sparql: str) -> None:
        self.updates.append(sparql)


async def test_insert_facts_populates_index_scoped_to_kg():
    neptune = _FakeNeptune()
    graph = kg_graph_uri(TENANT, KG)
    triples = [
        ("e:venue", RDF_TYPE, "https://cograph.tech/types/Venue"),
        _geom("e:venue", -122.4194, 37.7749),
        ("e:noband", "https://cograph.tech/types/Band/name", "no geo here"),
    ]
    await insert_facts(neptune, graph, triples)
    assert neptune.updates  # the primary write still happened

    idx = get_spatiotemporal_index()
    hit = await idx.query_radius(TENANT, -122.4194, 37.7749, 1_000, kg_name=KG)
    assert {r.entity_uri for r in hit} == {"e:venue"}
    # Nothing leaks into a different KG.
    assert await idx.query_radius(TENANT, -122.4194, 37.7749, 1_000, kg_name="Other") == []


async def test_insert_facts_skips_non_kg_graph():
    """Writing to the tenant ontology graph (not a per-KG graph) indexes nothing."""
    neptune = _FakeNeptune()
    onto_graph = "https://cograph.tech/graphs/demo-tenant"  # no /kg/ segment
    await insert_facts(neptune, onto_graph, [_geom("e:x", 1.0, 1.0)])
    idx = get_spatiotemporal_index()
    assert await idx.query_radius(TENANT, 1.0, 1.0, 1_000) == []


async def test_index_failure_does_not_fail_write():
    """A derived-index error must never propagate out of the primary KG write."""

    class _BoomIndex:
        async def upsert_many(self, facts):
            raise RuntimeError("index down")

    register_spatiotemporal_index(_BoomIndex())
    neptune = _FakeNeptune()
    graph = kg_graph_uri(TENANT, KG)
    # Must not raise despite the index blowing up.
    await insert_facts(neptune, graph, [_geom("e:venue", -122.4, 37.7)])
    assert neptune.updates  # write went through
