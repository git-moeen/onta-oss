"""Read-side spatial routing (ONTA-157 Phase 2).

Pure helpers (intent parse, gate, formatting) + the gated fast-path orchestration
on NLQueryPipeline, exercised with a mocked intent LLM, a registered in-memory
index, and a fake Neptune. No live LLM / Neptune / Postgres.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from cograph_client.graph.queries import kg_graph_uri
from cograph_client.nlp.pipeline import NLQueryPipeline
from cograph_client.spatiotemporal.protocol import STQueryResult, SpatioTemporalFact
from cograph_client.spatiotemporal.registry import (
    register_spatiotemporal_index,
    reset_spatiotemporal_index,
)
from cograph_client.spatiotemporal.memory import InMemorySpatioTemporalIndex
from cograph_client.spatiotemporal.routing import (
    STQueryIntent,
    SpatialAnchor,
    filter_by_type,
    format_spatial_answer,
    looks_spatial,
    parse_spatial_intent,
)

TENANT = "demo-tenant"
KG = "EventsSF"
SF_FERRY = (-122.3933, 37.7956)
SF_CITY_HALL = (-122.4194, 37.7793)


def _intent_json(**over):
    base = {
        "is_spatial": True,
        "kind": "radius",
        "anchor_lon": None,
        "anchor_lat": None,
        "anchor_description": None,
        "radius_m": 5000,
        "bbox": None,
        "target_type": None,
        "as_of": None,
        "time_from": None,
        "time_to": None,
    }
    base.update(over)
    return base


# --------------------------------------------------------------------------- pure
class TestLooksSpatial:
    @pytest.mark.parametrize(
        "q",
        [
            "restaurants within 2km of the Ferry Building",
            "venues near City Hall",
            "what is closest to downtown",
            "places within a 500 meter radius",
        ],
    )
    def test_positive(self, q):
        assert looks_spatial(q) is True

    @pytest.mark.parametrize(
        "q",
        ["how many bands played", "list all venues", "the difference between A and B"],
    )
    def test_negative(self, q):
        assert looks_spatial(q) is False


class TestParseIntent:
    def test_radius_with_description(self):
        i = parse_spatial_intent(
            _intent_json(anchor_description="the Ferry Building", radius_m=2000, target_type="Restaurant")
        )
        assert i.kind == "radius" and i.radius_m == 2000
        assert i.anchor.entity_description == "the Ferry Building"
        assert i.target_type == "Restaurant"

    def test_radius_with_coords(self):
        i = parse_spatial_intent(_intent_json(anchor_lon=-122.4, anchor_lat=37.8))
        assert i.anchor.has_coords()

    def test_bbox(self):
        i = parse_spatial_intent(
            _intent_json(kind="bbox", radius_m=None, bbox=[-122.6, 37.6, -122.3, 37.9])
        )
        assert i.kind == "bbox" and i.bbox == (-122.6, 37.6, -122.3, 37.9)

    def test_not_spatial_is_none(self):
        assert parse_spatial_intent(_intent_json(is_spatial=False)) is None

    def test_radius_missing_radius_is_none(self):
        assert parse_spatial_intent(_intent_json(anchor_lon=1, anchor_lat=2, radius_m=None)) is None

    def test_radius_missing_anchor_is_none(self):
        assert parse_spatial_intent(_intent_json(radius_m=1000)) is None  # no coords, no desc

    def test_bad_bbox_is_none(self):
        assert parse_spatial_intent(_intent_json(kind="bbox", radius_m=None, bbox=[1, 2, 3])) is None

    def test_unknown_kind_is_none(self):
        assert parse_spatial_intent(_intent_json(kind="polygon")) is None


class TestFilterAndFormat:
    def _hits(self):
        return [
            STQueryResult(entity_uri="e:1", attrs={"label": "Slanted Door", "type": "Restaurant"}),
            STQueryResult(entity_uri="e:2", attrs={"label": "Pier 1", "type": "Pier"}),
        ]

    def test_filter_by_type(self):
        assert [h.entity_uri for h in filter_by_type(self._hits(), "Restaurant")] == ["e:1"]

    def test_filter_no_target_passthrough(self):
        assert len(filter_by_type(self._hits(), None)) == 2

    def test_format_lists_hits(self):
        i = STQueryIntent(kind="radius", anchor=SpatialAnchor(entity_description="X"), radius_m=2000)
        out = format_spatial_answer(self._hits(), i)
        assert "Found 2" in out and "Slanted Door (Restaurant)" in out

    def test_format_empty(self):
        i = STQueryIntent(kind="radius", anchor=SpatialAnchor(entity_description="X"), radius_m=2000)
        assert format_spatial_answer([], i).startswith("No entities found")


# ------------------------------------------------------------------- orchestration
class FakeNeptune:
    """Returns a single ?wkt row for the anchor-resolution query; else empty."""

    def __init__(self, wkt: str | None = None):
        self.wkt = wkt
        self.queries: list[str] = []

    async def query(self, sparql: str):
        self.queries.append(sparql)
        if self.wkt is not None and "wktLiteral" in sparql:
            return {"head": {"vars": ["wkt"]}, "results": {"bindings": [{"wkt": {"value": self.wkt}}]}}
        return {"head": {"vars": []}, "results": {"bindings": []}}


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_spatiotemporal_index()
    yield
    reset_spatiotemporal_index()


def _pipeline(neptune):
    return NLQueryPipeline(neptune, anthropic_key="dummy-key")


def _index_with_facts(facts):
    idx = InMemorySpatioTemporalIndex()
    register_spatiotemporal_index(idx)
    return idx


def _fact(uri, lon, lat, *, kg=KG, attrs=None, vf=None, vt=None):
    return SpatioTemporalFact(
        entity_uri=uri, tenant_id=TENANT, kg_name=kg, lon=lon, lat=lat,
        valid_from=vf, valid_to=vt, attrs=attrs or {"label": uri},
    )


async def _run(pipe, question, intent_json, data_graph=None):
    """Drive the fast path with a canned intent (no real LLM)."""
    async def fake_detect(q, onto):
        return intent_json
    pipe._detect_spatial_intent = fake_detect  # type: ignore[assignment]
    dg = data_graph or kg_graph_uri(TENANT, KG)
    return await pipe._try_spatial_fast_path(question, "onto", dg, {}, time.time())


async def test_radius_explicit_coords_answers_from_index():
    idx = _index_with_facts(None)
    await idx.upsert_many([
        _fact("e:ferry", *SF_FERRY, attrs={"label": "Ferry Building", "type": "Venue"}),
        _fact("e:hall", *SF_CITY_HALL, attrs={"label": "City Hall", "type": "Venue"}),
    ])
    pipe = _pipeline(FakeNeptune())
    res = await _run(pipe, "venues within 1km of here",
                     _intent_json(anchor_lon=SF_FERRY[0], anchor_lat=SF_FERRY[1], radius_m=1000))
    assert res is not None
    assert res.sparql == ""  # no SPARQL — answered from the index
    assert "Ferry Building" in res.answer and "City Hall" not in res.answer
    assert res.timing.get("spatial_routed") == "true"


async def test_radius_anchor_resolved_via_neptune():
    idx = _index_with_facts(None)
    await idx.upsert_many([
        _fact("e:slanted", -122.3932, 37.7955, attrs={"label": "Slanted Door", "type": "Restaurant"}),
    ])
    # Neptune resolves "the Ferry Building" to a point next to the restaurant.
    pipe = _pipeline(FakeNeptune(wkt=f"POINT({SF_FERRY[0]} {SF_FERRY[1]})"))
    res = await _run(pipe, "restaurants within 1km of the Ferry Building",
                     _intent_json(anchor_description="the Ferry Building", radius_m=1000, target_type="Restaurant"))
    assert res is not None
    assert "Slanted Door" in res.answer
    assert any("wktLiteral" in q for q in pipe.neptune.queries)  # anchor lookup happened


async def test_target_type_filters_results():
    idx = _index_with_facts(None)
    await idx.upsert_many([
        _fact("e:r", *SF_FERRY, attrs={"label": "Resto", "type": "Restaurant"}),
        _fact("e:p", *SF_FERRY, attrs={"label": "Pier", "type": "Pier"}),
    ])
    pipe = _pipeline(FakeNeptune())
    res = await _run(pipe, "restaurants within 1km of here",
                     _intent_json(anchor_lon=SF_FERRY[0], anchor_lat=SF_FERRY[1], radius_m=1000, target_type="Restaurant"))
    assert "Resto" in res.answer and "Pier" not in res.answer


async def test_temporal_as_of_excludes_out_of_range():
    idx = _index_with_facts(None)
    d = lambda y: datetime(y, 1, 1, tzinfo=timezone.utc)
    await idx.upsert_many([
        _fact("e:old", *SF_FERRY, attrs={"label": "Old", "type": "Venue"}, vf=d(2010), vt=d(2012)),
        _fact("e:now", *SF_FERRY, attrs={"label": "Now", "type": "Venue"}, vf=d(2024), vt=d(2030)),
    ])
    pipe = _pipeline(FakeNeptune())
    res = await _run(pipe, "venues within 1km of here in 2026",
                     _intent_json(anchor_lon=SF_FERRY[0], anchor_lat=SF_FERRY[1], radius_m=1000, as_of="2026-01-01"))
    assert "Now" in res.answer and "Old" not in res.answer


async def test_bbox_path():
    idx = _index_with_facts(None)
    await idx.upsert_many([
        _fact("e:in", *SF_FERRY, attrs={"label": "In", "type": "Venue"}),
        _fact("e:out", -73.9855, 40.7580, attrs={"label": "NYC", "type": "Venue"}),
    ])
    pipe = _pipeline(FakeNeptune())
    res = await _run(pipe, "venues in this bounding box",
                     _intent_json(kind="bbox", radius_m=None, bbox=[-122.6, 37.6, -122.3, 37.9]))
    assert "In" in res.answer and "NYC" not in res.answer


async def test_non_spatial_intent_falls_through():
    _index_with_facts(None)
    pipe = _pipeline(FakeNeptune())
    res = await _run(pipe, "how many venues", _intent_json(is_spatial=False))
    assert res is None


async def test_non_kg_graph_falls_through():
    _index_with_facts(None)
    pipe = _pipeline(FakeNeptune())
    # The tenant ontology graph is not a per-KG instance graph → no routing.
    res = await _run(pipe, "venues within 1km of here",
                     _intent_json(anchor_lon=1, anchor_lat=2, radius_m=1000),
                     data_graph="https://cograph.tech/graphs/demo-tenant")
    assert res is None


async def test_unresolved_anchor_falls_through():
    _index_with_facts(None)
    pipe = _pipeline(FakeNeptune(wkt=None))  # neptune finds no matching entity
    res = await _run(pipe, "restaurants within 1km of Nowhereville",
                     _intent_json(anchor_description="Nowhereville", radius_m=1000))
    assert res is None


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("COGRAPH_SPATIAL_ROUTING_ENABLED", raising=False)
    assert _pipeline(FakeNeptune())._spatial_routing_enabled is False


def test_flag_on_when_set(monkeypatch):
    monkeypatch.setenv("COGRAPH_SPATIAL_ROUTING_ENABLED", "1")
    assert _pipeline(FakeNeptune())._spatial_routing_enabled is True
