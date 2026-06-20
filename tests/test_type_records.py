"""Paged per-type records endpoint (COG-100).

GET /graphs/{tenant}/explore/kgs/{kg}/types/{type}/records returns instances of
a type as table rows: a `uri` + `label` column followed by one column per
discovered attribute local-name, plus the total instance count and Load-more
pagination (`has_more` / `next_cursor`).

The handler issues three SPARQL queries — a COUNT, an attribute-predicate
discovery, and the paged rows query — so the mock is driven with `side_effect`
to return distinct fake SPARQL JSON for each.
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
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
# Must match the handler's GROUP_CONCAT delimiter (ASCII unit separator).
DELIM = "\x1f"


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
    """Build SPARQL Results JSON; collects vars from the binding keys."""
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


def _count(n):
    return _rows({"n": str(n)})


def _preds(*pred_uris):
    return _rows(*[{"p": p} for p in pred_uris])


def _url(limit=None, cursor=None):
    base = f"/graphs/{TENANT}/explore/kgs/{KG}/types/Product/records"
    params = []
    if limit is not None:
        params.append(f"limit={limit}")
    if cursor is not None:
        params.append(f"cursor={cursor}")
    return base + ("?" + "&".join(params) if params else "")


def test_records_basic_shape(client, mock_neptune, auth_headers):
    # COUNT → predicate discovery → paged rows.
    name_p = TYPES + "Product/attrs/name"
    gtin_p = TYPES + "Product/attrs/gtin"
    mock_neptune.query.side_effect = [
        _count(2),
        _preds(name_p, gtin_p),
        _rows(
            {"e": ENTITIES + "Product/p1", "label": "Widget", "v0": "Widget", "v1": "012345"},
            {"e": ENTITIES + "Product/p2", "label": "Gadget", "v0": "Gadget", "v1": "067890"},
        ),
    ]

    resp = client.get(_url(limit=50), headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()

    assert body["columns"] == ["uri", "label", "name", "gtin"]
    assert body["total"] == 2
    assert body["has_more"] is False
    assert body["next_cursor"] is None
    assert len(body["rows"]) == 2

    r0 = body["rows"][0]
    assert r0["uri"] == ENTITIES + "Product/p1"
    assert r0["label"] == "Widget"
    # Attributes collapse to a (sorted, distinct) list — single value → 1-elem list.
    assert r0["name"] == ["Widget"]
    assert r0["gtin"] == ["012345"]


def test_records_pagination_has_more_and_cursor(client, mock_neptune, auth_headers):
    # limit=2 → handler over-fetches 3 (LIMIT 3). Three rows back means has_more.
    name_p = TYPES + "Product/attrs/name"
    mock_neptune.query.side_effect = [
        _count(10),
        _preds(name_p),
        _rows(
            {"e": ENTITIES + "Product/p1", "v0": "A"},
            {"e": ENTITIES + "Product/p2", "v0": "B"},
            {"e": ENTITIES + "Product/p3", "v0": "C"},  # the +1 over-fetch
        ),
    ]

    resp = client.get(_url(limit=2), headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()

    assert body["total"] == 10
    assert body["has_more"] is True
    # Over-fetched row is trimmed; only `limit` rows returned.
    assert len(body["rows"]) == 2
    assert [r["uri"] for r in body["rows"]] == [
        ENTITIES + "Product/p1", ENTITIES + "Product/p2",
    ]
    # next_cursor advances by the page size from offset 0.
    assert body["next_cursor"] == "2"

    # The rows query must carry LIMIT 3 (limit+1) and OFFSET 0.
    rows_sparql = mock_neptune.query.call_args_list[2].args[0]
    assert "LIMIT 3" in rows_sparql
    assert "OFFSET 0" in rows_sparql


def test_records_cursor_offset_honored(client, mock_neptune, auth_headers):
    # A cursor of "50" must become OFFSET 50 in the rows query.
    name_p = TYPES + "Product/attrs/name"
    mock_neptune.query.side_effect = [
        _count(120),
        _preds(name_p),
        _rows({"e": ENTITIES + "Product/p51", "v0": "X"}),
    ]

    resp = client.get(_url(limit=50, cursor="50"), headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_more"] is False
    assert body["next_cursor"] is None

    rows_sparql = mock_neptune.query.call_args_list[2].args[0]
    assert "OFFSET 50" in rows_sparql
    assert "LIMIT 51" in rows_sparql


def test_records_malformed_cursor_defaults_to_zero(client, mock_neptune, auth_headers):
    name_p = TYPES + "Product/attrs/name"
    mock_neptune.query.side_effect = [
        _count(1),
        _preds(name_p),
        _rows({"e": ENTITIES + "Product/p1", "v0": "X"}),
    ]

    resp = client.get(_url(cursor="not-an-int"), headers=auth_headers)
    assert resp.status_code == 200
    rows_sparql = mock_neptune.query.call_args_list[2].args[0]
    assert "OFFSET 0" in rows_sparql


def test_records_limit_clamped_to_max(client, mock_neptune, auth_headers):
    # limit above the 200 ceiling is rejected by the Query validator (422).
    resp = client.get(_url(limit=5000), headers=auth_headers)
    assert resp.status_code == 422


def test_records_multivalued_collapsed_to_sorted_distinct_list(client, mock_neptune, auth_headers):
    # GROUP_CONCAT delimits multiple values; the handler splits + dedupes + sorts.
    color_p = TYPES + "Product/attrs/color"
    mock_neptune.query.side_effect = [
        _count(1),
        _preds(color_p),
        _rows({
            "e": ENTITIES + "Product/p1",
            # red, blue, red → distinct + sorted → ["blue", "red"]
            "v0": DELIM.join(["red", "blue", "red"]),
        }),
    ]

    resp = client.get(_url(), headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["columns"] == ["uri", "label", "color"]
    assert body["rows"][0]["color"] == ["blue", "red"]


def test_records_missing_attribute_omitted_from_row(client, mock_neptune, auth_headers):
    # An entity with no value for a column simply omits that key (no null).
    name_p = TYPES + "Product/attrs/name"
    gtin_p = TYPES + "Product/attrs/gtin"
    mock_neptune.query.side_effect = [
        _count(1),
        _preds(name_p, gtin_p),
        _rows({"e": ENTITIES + "Product/p1", "v0": "Widget"}),  # no v1 (gtin)
    ]

    resp = client.get(_url(), headers=auth_headers)
    assert resp.status_code == 200
    row = resp.json()["rows"][0]
    assert row["name"] == ["Widget"]
    assert "gtin" not in row


def test_records_system_predicates_excluded_from_columns(client, mock_neptune, auth_headers):
    # ingested_at / source / rdfs:label must not become attribute columns.
    name_p = TYPES + "Product/attrs/name"
    mock_neptune.query.side_effect = [
        _count(1),
        _preds(
            name_p,
            "https://cograph.tech/onto/ingested_at",
            "https://cograph.tech/onto/source",
            RDFS_LABEL,
        ),
        _rows({"e": ENTITIES + "Product/p1", "v0": "Widget"}),
    ]

    resp = client.get(_url(), headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["columns"] == ["uri", "label", "name"]


def test_records_empty_type(client, mock_neptune, auth_headers):
    # No instances → total 0, no rows, just the uri/label columns.
    mock_neptune.query.side_effect = [
        _count(0),
        _preds(),
        _rows(),
    ]

    resp = client.get(_url(), headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["rows"] == []
    assert body["columns"] == ["uri", "label"]
    assert body["has_more"] is False
    assert body["next_cursor"] is None
