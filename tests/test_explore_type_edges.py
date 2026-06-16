"""The Explorer overview's type→type edges come from instance data.

Regression for: the overview graph (type-edges) was derived purely from the
ontology's declared ``rdfs:range`` while the per-type detail view counted any
predicate with entity-valued objects. So a relationship present in the data but
whose ontology range was never a ``types/`` URI (e.g. ``RetailerSKU identifies
Product``) showed in the detail view but was MISSING from the overview. The
endpoint now reads the same instance-derived ``targetType`` the summary uses.
"""
import os

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

os.environ["OMNIX_API_KEYS"] = '{"test-key": "test-tenant"}'
os.environ["OMNIX_NEPTUNE_ENDPOINT"] = "http://fake-neptune:8182"

from cograph_client.api.app import create_app
from cograph_client.graph.client import NeptuneClient

TENANT = "test-tenant"
KG = "test"
TYPES = "https://cograph.tech/types/"
ENTITIES = "https://cograph.tech/entities/"


@pytest.fixture
def mock_neptune():
    client = AsyncMock(spec=NeptuneClient)
    client.health.return_value = True
    client.update.return_value = None
    return client


@pytest.fixture
def client(mock_neptune):
    app = create_app()
    app.state.neptune_client = mock_neptune
    return TestClient(app)


@pytest.fixture
def auth_headers():
    return {"X-API-Key": "test-key"}


def _rows(*binding_dicts):
    # The parser only reads variables declared in head.vars, so collect them.
    variables = []
    for b in binding_dicts:
        for k in b:
            if k not in variables:
                variables.append(k)
    return {
        "head": {"vars": variables},
        "results": {"bindings": [
            {k: {"value": v} for k, v in b.items()} for b in binding_dicts
        ]},
    }


def test_edges_from_stats(client, mock_neptune, auth_headers):
    # Stats graph carries instance-derived targetType. Both directions of the
    # same pair must collapse to a single undirected edge.
    def route(sparql, *a, **k):
        if "targetType" in sparql and "forType" in sparql:
            return _rows(
                {"src": TYPES + "RetailerSKU", "tgt": TYPES + "Product"},
                {"src": TYPES + "Product", "tgt": TYPES + "RetailerSKU"},
                {"src": TYPES + "RetailerSKU", "tgt": TYPES + "Retailer"},
            )
        return _rows()

    mock_neptune.query.side_effect = route

    resp = client.get(f"/graphs/{TENANT}/explore/kgs/{KG}/type-edges", headers=auth_headers)
    assert resp.status_code == 200
    edges = resp.json()

    pairs = {tuple(sorted((e["source"], e["target"]))) for e in edges}
    assert pairs == {("Product", "RetailerSKU"), ("Retailer", "RetailerSKU")}
    # The RetailerSKU↔Product edge — the one the detail view shows — is present.
    assert ("Product", "RetailerSKU") in pairs
    assert all(e["weight"] == 70 for e in edges)


def test_edges_live_scan_fallback(client, mock_neptune, auth_headers):
    # No stats materialized → fall back to a live instance scan. Target type is
    # read from the object entity URI leaf.
    def route(sparql, *a, **k):
        if "targetType" in sparql and "forType" in sparql:
            return _rows()  # stats absent → triggers fallback
        if "?e ?p ?o" in sparql:
            return _rows(
                {"type": TYPES + "RetailerSKU", "o": ENTITIES + "Product/p1"},
                {"type": TYPES + "RetailerSKU", "o": ENTITIES + "Product/p2"},
            )
        return _rows()

    mock_neptune.query.side_effect = route

    resp = client.get(f"/graphs/{TENANT}/explore/kgs/{KG}/type-edges", headers=auth_headers)
    assert resp.status_code == 200
    edges = resp.json()
    assert {tuple(sorted((e["source"], e["target"]))) for e in edges} == {("Product", "RetailerSKU")}


def test_edges_empty_kg(client, mock_neptune, auth_headers):
    # No stats and no instance edges → an empty list, not an error.
    mock_neptune.query.return_value = _rows()
    resp = client.get(f"/graphs/{TENANT}/explore/kgs/{KG}/type-edges", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []
