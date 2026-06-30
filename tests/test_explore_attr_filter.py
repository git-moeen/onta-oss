"""Explorer Attributes panel must hide internal/housekeeping predicates.

Regression for the display bug where the per-type summary listed INTERNAL
predicates as if they were user-facing domain attributes (with a coverage %).
For a web-ingested ``TTSModel`` the panel showed ``batch_id`` (100%),
``blockKey`` (82%), ``erSignal_name`` (82%) alongside the real ``score``.

Internal triples confirmed on live instances:
  * ``…/onto/batch_id``, ``…/onto/ingested_at``, ``…/onto/source`` (housekeeping)
  * ``…/er/blockKey``, ``…/er/erSignal_name``                      (ER internals)
Only ``…/types/<T>/attrs/<a>`` predicates (and real relationships to ``…/entities/…``)
are user-facing.

The fix adds ``_is_internal_predicate`` and applies it at summary assembly
(``_assemble_summary``) AND at scan/recompute time (``_live_scan``,
``recompute_kg_stats``). These tests assert the assembled summary's attributes
contain ONLY the real domain attribute.

Mock harness mirrors test_explore_records.py / test_explore_type_edges.py:
TestClient(create_app()) with a mock Neptune injected on app.state and
``query.side_effect`` routing SPARQL strings to fixture rows.
"""
import os

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

os.environ["OMNIX_API_KEYS"] = '{"test-key": "test-tenant"}'
os.environ["OMNIX_NEPTUNE_ENDPOINT"] = "http://fake-neptune:8182"

from cograph_client.api.app import create_app
from cograph_client.api.routes.explore import (
    RDF_TYPE,
    _assemble_summary,
    _is_internal_predicate,
)
from cograph_client.graph.client import NeptuneClient

TENANT = "test-tenant"
KG = "web"
TYPE = "TTSModel"

TYPES = "https://cograph.tech/types/"
ENTITIES = "https://cograph.tech/entities/"
ONTO = "https://cograph.tech/onto/"
ER = "https://cograph.tech/er/"
RDFS = "http://www.w3.org/2000/01/rdf-schema"

# Real domain attribute: instance predicate …/onto/score ← attrs/score
SCORE_PRED = ONTO + "score"
# Internal predicates that the bug surfaced as attributes
BATCH_ID_PRED = ONTO + "batch_id"
INGESTED_AT_PRED = ONTO + "ingested_at"
SOURCE_PRED = ONTO + "source"
BLOCK_KEY_PRED = ER + "blockKey"
ER_SIGNAL_NAME_PRED = ER + "erSignal_name"

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
# 1. End-to-end: /summary via the live-scan path drops internal predicates
# ---------------------------------------------------------------------------

def test_summary_excludes_internal_predicates(client, mock_neptune, auth_headers):
    """A TTSModel whose entities carry onto/batch_id, er/blockKey,
    er/erSignal_name AND a real onto/score must report ONLY `score` as an
    attribute — the internal predicates never appear in the panel."""

    def route(sparql, *a, **k):
        # Ontology label/comment/parent lookup
        if "?label" in sparql and "subClassOf" in sparql:
            return _rows({"label": TYPE})
        # Ontology attr-def query (declares `score`)
        if "attrLabel" in sparql:
            return _rows(
                {"attr": TYPES + "TTSModel/attrs/score", "attrLabel": "score", "range": ""},
            )
        # Precomputed stats: none materialized → forces the live scan fallback
        if "entityCount" in sparql:
            return _empty()
        if "forType" in sparql:
            return _empty()
        # Live instance scan: rdf:type row (entity count) + one row per predicate
        if "GROUP BY ?p" in sparql:
            return _rows(
                {"p": RDF_TYPE, "cnt": "11", "rel": "0", "sample": ENTITIES + "TTSModel/a"},
                {"p": SCORE_PRED, "cnt": "11", "rel": "0", "sample": "0.9"},
                {"p": BATCH_ID_PRED, "cnt": "11", "rel": "0", "sample": "batch-xyz"},
                {"p": BLOCK_KEY_PRED, "cnt": "9", "rel": "0", "sample": "T520"},
                {"p": ER_SIGNAL_NAME_PRED, "cnt": "9", "rel": "0", "sample": "tts"},
                {"p": INGESTED_AT_PRED, "cnt": "11", "rel": "0", "sample": "2026-01-01"},
                {"p": SOURCE_PRED, "cnt": "11", "rel": "0", "sample": "web"},
            )
        return _empty()

    mock_neptune.query.side_effect = route

    resp = client.get(
        f"/graphs/{TENANT}/explore/kgs/{KG}/types/{TYPE}/summary",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()

    attr_names = {a["name"] for a in data["attributes"]}
    # ONLY the real domain attribute survives.
    assert attr_names == {"score"}, attr_names
    # Explicitly assert each internal predicate is gone.
    assert "batch_id" not in attr_names
    assert "blockKey" not in attr_names
    assert "erSignal_name" not in attr_names
    assert "ingested_at" not in attr_names
    assert "source" not in attr_names
    # No internal predicate leaked into relationships either.
    rel_names = {r["name"] for r in data["relationships"]}
    assert rel_names == set(), rel_names


# ---------------------------------------------------------------------------
# 2. _assemble_summary filters already-materialized stats (the backstop path)
# ---------------------------------------------------------------------------

def test_assemble_summary_filters_internal_records():
    """Even if internal predicates were already materialized into the stats
    graph (KG recomputed before the fix), assembly drops them — so stale stats
    don't need a recompute to render a clean panel."""
    pred_records = [
        {"p": SCORE_PRED, "cnt": 11, "rel": 0, "target": None},
        {"p": BATCH_ID_PRED, "cnt": 11, "rel": 0, "target": None},
        {"p": BLOCK_KEY_PRED, "cnt": 9, "rel": 0, "target": None},
        {"p": ER_SIGNAL_NAME_PRED, "cnt": 9, "rel": 0, "target": None},
        {"p": RDFS + "#label", "cnt": 11, "rel": 0, "target": None},
    ]
    attr_defs = {TYPES + "TTSModel/attrs/score": {"name": "score", "range": ""}}

    summary = _assemble_summary(
        type_name=TYPE,
        onto_row={},
        parent_type=None,
        entity_count=11,
        pred_records=pred_records,
        attr_defs=attr_defs,
    )

    assert {a["name"] for a in summary["attributes"]} == {"score"}
    assert summary["relationships"] == []
    # The surviving attribute keeps its real coverage (11/11 = 100%).
    score = next(a for a in summary["attributes"] if a["name"] == "score")
    assert score["coverage_pct"] == 100.0


# ---------------------------------------------------------------------------
# 3. _is_internal_predicate unit table — root-cause helper, namespace-based
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p_uri", [
    BATCH_ID_PRED,
    INGESTED_AT_PRED,
    SOURCE_PRED,
    ONTO + "coreSlot",
    ONTO + "aliasOf",
    ONTO + "lambda_refreshed_at",
    ONTO + "norm/price",              # whole normalization namespace
    BLOCK_KEY_PRED,
    ER_SIGNAL_NAME_PRED,
    ER + "anythingElse",             # whole ER namespace
    RDF_TYPE,
    RDFS + "#label",
    RDFS + "#comment",               # any rdfs:* term
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property",  # any rdf:* term
    "",                               # empty → internal
])
def test_is_internal_predicate_true(p_uri):
    assert _is_internal_predicate(p_uri) is True


@pytest.mark.parametrize("p_uri", [
    SCORE_PRED,                        # real attribute predicate …/onto/score
    ONTO + "listed_by",               # real relationship predicate …/onto/<pred>
    ONTO + "company_name",            # real attribute on …/onto/
    TYPES + "TTSModel/attrs/score",   # ontology attr URI
])
def test_is_internal_predicate_false(p_uri):
    """Real domain predicates — including relationships/attributes that
    legitimately live under …/onto/ — must NOT be classified internal."""
    assert _is_internal_predicate(p_uri) is False


# ---------------------------------------------------------------------------
# 4. FIX 2 — a RELATIONSHIP named like a housekeeping marker is NOT hidden
# ---------------------------------------------------------------------------
# The curated markers (onto/source, onto/batch_id, …) are ALWAYS literal-valued
# housekeeping. A real relationship predicate that happens to share a marker leaf
# name — e.g. a reified measurement minted with predicate …/onto/source pointing
# at an Organization entity — must show in the Relationships panel, while the
# literal housekeeping …/onto/source (the ingest "source" string) stays hidden.

ORG_ENTITY = ENTITIES + "Organization/openrouter"


@pytest.mark.parametrize("marker_pred", [
    SOURCE_PRED,                       # the exact collision the reification prompt nudged
    BATCH_ID_PRED,
    INGESTED_AT_PRED,
])
def test_marker_predicate_kept_when_relationship(marker_pred):
    """A predicate sharing a housekeeping leaf name is EXEMPT from the marker
    check when it's a relationship (object is an entity IRI)."""
    # As a literal-valued housekeeping marker → hidden.
    assert _is_internal_predicate(marker_pred, is_relationship=False) is True
    # As a relationship (entity-valued object) → shown.
    assert _is_internal_predicate(marker_pred, is_relationship=True) is False


def test_er_namespace_relationship_still_hidden():
    """The whole-namespace exclusions (ER, normalization, rdf/rdfs) apply to
    relationships too — they are NEVER legitimate domain edges."""
    assert _is_internal_predicate(BLOCK_KEY_PRED, is_relationship=True) is True
    assert _is_internal_predicate(ONTO + "norm/x", is_relationship=True) is True
    assert _is_internal_predicate(RDF_TYPE, is_relationship=True) is True


def test_assemble_summary_shows_real_source_relationship_hides_literal_source():
    """Two predicates both named ``source``:

    * a REAL relationship (rel > 0, object → an Organization entity) → appears in
      ``relationships`` as ``source`` with the right target type;
    * a housekeeping LITERAL ``onto/source`` (rel == 0) → hidden entirely.

    Plus the real ``score`` attribute survives. This is the user-reported
    collision: the reification prompt nudges a measurement→provider relationship,
    and naming it ``source`` must not get it filtered like the ingest marker.
    """
    pred_records = [
        # Real attribute survives.
        {"p": SCORE_PRED, "cnt": 11, "rel": 0, "target": None},
        # Real relationship named "source": rel > 0, target is Organization.
        {"p": SOURCE_PRED, "cnt": 11, "rel": 11, "target": "Organization"},
        # Housekeeping LITERAL source on the SAME leaf name: rel == 0 → hidden.
        # (In real data these are distinct triples; here a second record with the
        # same predicate URI but no entity objects models the literal marker.)
        {"p": BATCH_ID_PRED, "cnt": 11, "rel": 0, "target": None},
        {"p": INGESTED_AT_PRED, "cnt": 11, "rel": 0, "target": None},
    ]
    attr_defs = {TYPES + "TTSModel/attrs/score": {"name": "score", "range": ""}}

    summary = _assemble_summary(
        type_name=TYPE,
        onto_row={},
        parent_type=None,
        entity_count=11,
        pred_records=pred_records,
        attr_defs=attr_defs,
    )

    # The real "source" relationship is SHOWN, pointed at Organization.
    rel_by_name = {r["name"]: r for r in summary["relationships"]}
    assert "source" in rel_by_name, summary["relationships"]
    assert rel_by_name["source"]["target_type"] == "Organization"

    # The literal housekeeping markers are HIDDEN (never attributes either).
    attr_names = {a["name"] for a in summary["attributes"]}
    assert attr_names == {"score"}, attr_names
    assert "source" not in attr_names
    assert "batch_id" not in attr_names
    assert "ingested_at" not in attr_names


def test_live_scan_keeps_entity_valued_source_relationship(client, mock_neptune, auth_headers):
    """End-to-end via the /summary live-scan path: a TTSModel whose entities have
    a real ``onto/source`` edge to an Organization AND a housekeeping literal
    ``onto/batch_id`` → ``source`` shows as a relationship, batch_id is hidden."""
    # The /summary endpoint memoizes per (tenant, kg, type) in a module cache;
    # another test in this file warms the same key, so clear it to read fresh.
    from cograph_client.api.routes import explore as _explore_mod
    _explore_mod._summary_cache.clear()

    def route(sparql, *a, **k):
        if "?label" in sparql and "subClassOf" in sparql:
            return _rows({"label": TYPE})
        if "attrLabel" in sparql:
            return _rows(
                {"attr": TYPES + "TTSModel/attrs/score", "attrLabel": "score", "range": ""},
            )
        if "entityCount" in sparql:
            return _empty()
        if "forType" in sparql:
            return _empty()
        if "GROUP BY ?p" in sparql:
            return _rows(
                {"p": RDF_TYPE, "cnt": "11", "rel": "0", "sample": ENTITIES + "TTSModel/a"},
                {"p": SCORE_PRED, "cnt": "11", "rel": "0", "sample": "0.9"},
                # Real relationship named "source": rel > 0, sample is an entity IRI.
                {"p": SOURCE_PRED, "cnt": "11", "rel": "11", "sample": ORG_ENTITY},
                # Housekeeping literal marker: rel == 0 → hidden.
                {"p": BATCH_ID_PRED, "cnt": "11", "rel": "0", "sample": "batch-xyz"},
            )
        return _empty()

    mock_neptune.query.side_effect = route

    resp = client.get(
        f"/graphs/{TENANT}/explore/kgs/{KG}/types/{TYPE}/summary",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    data = resp.json()

    rel_names = {r["name"] for r in data["relationships"]}
    assert "source" in rel_names, data["relationships"]
    # target type recovered from the sample entity IRI leaf.
    src_rel = next(r for r in data["relationships"] if r["name"] == "source")
    assert src_rel["target_type"] == "Organization"

    attr_names = {a["name"] for a in data["attributes"]}
    assert attr_names == {"score"}, attr_names
    assert "batch_id" not in attr_names
    assert "source" not in attr_names
