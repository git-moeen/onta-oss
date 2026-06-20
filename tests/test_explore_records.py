"""Tests for the paged per-type records endpoint (COG-100).

GET /graphs/{tenant}/explore/kgs/{kg}/types/{type}/records
  ?limit=<int>  &cursor=<last-entity-uri>

Follows the mock harness established in test_explore_type_edges.py:
  - TestClient(create_app()) with mock_neptune injected via app.state
  - mock_neptune.query.side_effect routes SPARQL strings to fixture data
  - _rows(*dicts) builds the SPARQL JSON wire format the parser expects

Scenarios covered:
  1. Page of rows with attribute columns (resolved via ontology)
  2. Pagination: cursor advances next_cursor; next_cursor is null at final page
  3. Empty type → empty sentinel, no error
  4. System predicates (rdfs:label used as name, ingested_at etc. excluded from cols)
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
KG = "movies"
TYPE = "Movie"

ENTITIES = "https://cograph.tech/entities/"
TYPES = "https://cograph.tech/types/"
ONTO = "https://cograph.tech/onto/"
RDFS = "http://www.w3.org/2000/01/rdf-schema"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
STATS_ENTITY_COUNT = "https://cograph.tech/stats/entityCount"

E1 = ENTITIES + "Movie/m1"
E2 = ENTITIES + "Movie/m2"
E3 = ENTITIES + "Movie/m3"

TITLE_PRED = ONTO + "title"
YEAR_PRED = ONTO + "year"
INGESTED_AT_PRED = "https://cograph.tech/onto/ingested_at"
SOURCE_PRED = "https://cograph.tech/onto/source"
LABEL_PRED = RDFS + "#label"
TYPE_URI = TYPES + TYPE


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
    """Build a SPARQL JSON result the parser can consume."""
    variables: list[str] = []
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


def _empty():
    return {"head": {"vars": []}, "results": {"bindings": []}}


# ---------------------------------------------------------------------------
# 1. Happy path: page of rows with ontology-resolved attribute columns
# ---------------------------------------------------------------------------

def test_records_basic_page(client, mock_neptune, auth_headers):
    """Two entities are returned with title and year columns resolved from the ontology."""

    def route(sparql, *a, **k):
        # Ontology attr-def query
        if "attrLabel" in sparql:
            return _rows(
                {"attr": ONTO + "types/Movie/attrs/title", "attrLabel": "title", "range": ""},
                {"attr": ONTO + "types/Movie/attrs/year", "attrLabel": "year", "range": ""},
            )
        # Entity page query (DISTINCT ?e)
        if "DISTINCT ?e" in sparql and "ORDER BY ?e" in sparql:
            return _rows({"e": E1}, {"e": E2})
        # Stats entity count
        if "entityCount" in sparql:
            return _rows({"ec": "10"})
        # Attribute values for the page
        if "VALUES ?e" in sparql:
            return _rows(
                {"e": E1, "p": LABEL_PRED, "o": "The Matrix"},
                {"e": E1, "p": TITLE_PRED, "o": "The Matrix"},
                {"e": E1, "p": YEAR_PRED, "o": "1999"},
                {"e": E2, "p": LABEL_PRED, "o": "Inception"},
                {"e": E2, "p": TITLE_PRED, "o": "Inception"},
                {"e": E2, "p": YEAR_PRED, "o": "2010"},
            )
        return _empty()

    mock_neptune.query.side_effect = route

    resp = client.get(
        f"/graphs/{TENANT}/explore/kgs/{KG}/types/{TYPE}/records",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()

    # columns: name always first, then attribute names
    assert data["columns"][0] == "name"
    assert "title" in data["columns"]
    assert "year" in data["columns"]

    # rows
    assert len(data["rows"]) == 2
    names = {r["name"] for r in data["rows"]}
    assert "The Matrix" in names
    assert "Inception" in names

    ids = {r["id"] for r in data["rows"]}
    assert E1 in ids and E2 in ids

    # total from stats
    assert data["total"] == 10


# ---------------------------------------------------------------------------
# 2. Pagination: cursor advances; next_cursor set on a full page, null at end
# ---------------------------------------------------------------------------

def test_records_pagination_full_page(client, mock_neptune, auth_headers):
    """When a full page is returned next_cursor is the last entity URI."""

    def route(sparql, *a, **k):
        if "attrLabel" in sparql:
            return _empty()
        if "DISTINCT ?e" in sparql and "ORDER BY ?e" in sparql:
            return _rows({"e": E1}, {"e": E2})
        if "entityCount" in sparql:
            return _rows({"ec": "5"})
        if "VALUES ?e" in sparql:
            return _rows(
                {"e": E1, "p": TITLE_PRED, "o": "Movie A"},
                {"e": E2, "p": TITLE_PRED, "o": "Movie B"},
            )
        return _empty()

    mock_neptune.query.side_effect = route

    resp = client.get(
        f"/graphs/{TENANT}/explore/kgs/{KG}/types/{TYPE}/records",
        params={"limit": 2},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    # Full page (2 of 2 requested) → next_cursor is last entity URI
    assert data["next_cursor"] == E2


def test_records_pagination_last_page(client, mock_neptune, auth_headers):
    """When fewer than limit entities are returned next_cursor is null."""

    def route(sparql, *a, **k):
        if "attrLabel" in sparql:
            return _empty()
        if "DISTINCT ?e" in sparql and "ORDER BY ?e" in sparql:
            return _rows({"e": E3})   # only 1 result, limit=2 → last page
        if "entityCount" in sparql:
            return _rows({"ec": "3"})
        if "VALUES ?e" in sparql:
            return _rows({"e": E3, "p": TITLE_PRED, "o": "Movie C"})
        return _empty()

    mock_neptune.query.side_effect = route

    resp = client.get(
        f"/graphs/{TENANT}/explore/kgs/{KG}/types/{TYPE}/records",
        params={"limit": 2, "cursor": E2},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["next_cursor"] is None
    assert len(data["rows"]) == 1


def test_records_cursor_filter_in_sparql(client, mock_neptune, auth_headers):
    """The cursor value appears in the entities SPARQL as a keyset filter."""
    captured: list[str] = []

    def route(sparql, *a, **k):
        captured.append(sparql)
        if "attrLabel" in sparql:
            return _empty()
        if "DISTINCT ?e" in sparql:
            return _rows({"e": E3})
        if "entityCount" in sparql:
            return _rows({"ec": "3"})
        if "VALUES ?e" in sparql:
            return _empty()
        return _empty()

    mock_neptune.query.side_effect = route

    resp = client.get(
        f"/graphs/{TENANT}/explore/kgs/{KG}/types/{TYPE}/records",
        params={"cursor": E2},
        headers=auth_headers,
    )
    assert resp.status_code == 200
    entity_queries = [s for s in captured if "DISTINCT ?e" in s and "ORDER BY ?e" in s]
    assert entity_queries, "expected an entity page query"
    assert E2 in entity_queries[0], "cursor URI should appear in the entity page SPARQL"


# ---------------------------------------------------------------------------
# 3. Empty type → empty sentinel (no error)
# ---------------------------------------------------------------------------

def test_records_empty_type(client, mock_neptune, auth_headers):
    """A type with no instances returns the empty sentinel, never an error."""

    def route(sparql, *a, **k):
        if "attrLabel" in sparql:
            return _empty()
        if "DISTINCT ?e" in sparql:
            return _empty()   # no entities
        if "entityCount" in sparql:
            return _empty()   # no stats either
        if "COUNT" in sparql:
            return _rows({"n": "0"})
        return _empty()

    mock_neptune.query.side_effect = route

    resp = client.get(
        f"/graphs/{TENANT}/explore/kgs/{KG}/types/{TYPE}/records",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data == {"columns": ["name"], "rows": [], "total": 0, "next_cursor": None}


# ---------------------------------------------------------------------------
# 4. System predicates excluded; rdfs:label used as the row name
# ---------------------------------------------------------------------------

def test_records_system_predicates_excluded(client, mock_neptune, auth_headers):
    """ingested_at and source are excluded from columns; label becomes the name."""

    def route(sparql, *a, **k):
        if "attrLabel" in sparql:
            return _empty()
        if "DISTINCT ?e" in sparql and "ORDER BY ?e" in sparql:
            return _rows({"e": E1})
        if "entityCount" in sparql:
            return _rows({"ec": "1"})
        if "VALUES ?e" in sparql:
            return _rows(
                {"e": E1, "p": LABEL_PRED, "o": "Named Movie"},
                {"e": E1, "p": INGESTED_AT_PRED, "o": "2024-01-01"},
                {"e": E1, "p": SOURCE_PRED, "o": "import"},
                {"e": E1, "p": TITLE_PRED, "o": "Named Movie"},
            )
        return _empty()

    mock_neptune.query.side_effect = route

    resp = client.get(
        f"/graphs/{TENANT}/explore/kgs/{KG}/types/{TYPE}/records",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    columns = data["columns"]

    # system predicates must NOT appear as columns
    assert "ingested_at" not in columns
    assert "source" not in columns

    # rdfs:label must NOT appear as a column (used for name only)
    assert "label" not in columns

    # name should be the label value
    assert data["rows"][0]["name"] == "Named Movie"


def test_records_name_falls_back_to_id_leaf(client, mock_neptune, auth_headers):
    """When no rdfs:label, name comes from the last URI segment."""

    def route(sparql, *a, **k):
        if "attrLabel" in sparql:
            return _empty()
        if "DISTINCT ?e" in sparql and "ORDER BY ?e" in sparql:
            return _rows({"e": E1})
        if "entityCount" in sparql:
            return _rows({"ec": "1"})
        if "VALUES ?e" in sparql:
            return _rows({"e": E1, "p": TITLE_PRED, "o": "Some Title"})
        return _empty()

    mock_neptune.query.side_effect = route

    resp = client.get(
        f"/graphs/{TENANT}/explore/kgs/{KG}/types/{TYPE}/records",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    # E1 = ".../Movie/m1" → leaf is "m1"
    assert data["rows"][0]["name"] == "m1"


# ---------------------------------------------------------------------------
# 5. Total falls back to COUNT query when stats are absent
# ---------------------------------------------------------------------------

def test_records_total_fallback_count(client, mock_neptune, auth_headers):
    """When the stats graph has no entityCount, a COUNT query is used for total."""

    def route(sparql, *a, **k):
        if "attrLabel" in sparql:
            return _empty()
        if "DISTINCT ?e" in sparql and "ORDER BY ?e" in sparql:
            return _rows({"e": E1})
        if "entityCount" in sparql:
            return _empty()   # no stats
        if "VALUES ?e" in sparql:
            return _rows({"e": E1, "p": TITLE_PRED, "o": "Film"})
        if "COUNT" in sparql:
            return _rows({"n": "42"})
        return _empty()

    mock_neptune.query.side_effect = route

    resp = client.get(
        f"/graphs/{TENANT}/explore/kgs/{KG}/types/{TYPE}/records",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 42
