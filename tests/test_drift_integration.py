"""ADR 0004 drift-control wired into the Explorer endpoints (flag-gated).

The FOUNDATION (``cograph_client/resolver/drift_control.py``) is the pure
decision logic; these tests cover the INTEGRATION — the support floor reaching
the Explorer's overview edges and the recompute drift report. Everything is
gated behind ``OMNIX_DRIFT_CONTROL``: with the flag OFF the endpoints behave
byte-identically to before (a 6%-coverage drift edge is still drawn, recompute
returns no ``drift`` key); with it ON the drift edge is excluded while a 100%
edge and a core-slot edge survive.

The flag is read live from ``os.environ`` at call time
(``drift_control.drift_control_enabled()``), so ``monkeypatch.setenv`` takes
effect inside the request with no reload — that is the mechanism under test.

Reference truth (ADR 0004, calibrated): ``ManufacturerPartNumber.issuedby ->
Retailer`` at 41/685 (6%) QUARANTINES; ``RetailerSKU.issuedby -> Retailer`` at
604/604 DECLARES; ``SKU.issued_by`` is a core slot (0/604) and is EXEMPT.
"""
import os

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

os.environ["OMNIX_API_KEYS"] = '{"test-key": "test-tenant"}'
os.environ["OMNIX_NEPTUNE_ENDPOINT"] = "http://fake-neptune:8182"

from cograph_client.api.app import create_app
from cograph_client.api.routes import explore
from cograph_client.graph.client import NeptuneClient

TENANT = "test-tenant"
KG = "test"
TYPES = "https://cograph.tech/types/"
ENTITIES = "https://cograph.tech/entities/"
RDF_TYPE = explore.RDF_TYPE
CORE_SLOT = explore._CORE_SLOT_PRED

# Predicate (attribute) URIs for the three reference edges.
MPN_ISSUEDBY = TYPES + "MPN/attrs/issuedby"          # 41/685 -> 6% -> quarantine
RSKU_ISSUEDBY = TYPES + "RetailerSKU/attrs/issuedby"  # 604/604 -> 100% -> declare
SKU_ISSUED_BY = TYPES + "SKU/attrs/issued_by"          # core slot, 0/604 -> exempt


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
    """Build a SPARQL JSON result from plain {var: value} dicts (parser-shaped)."""
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


# Three edges as they appear in the drift-aware stats read: source type, target
# type (Retailer for all three), the predicate URI, support (rel), and the
# source type's entity count. MPN is the 6% drift edge; RetailerSKU is 100%;
# SKU.issued_by carries support 0 but is a core slot (exempt).
_EDGE_ROWS = [
    {"src": TYPES + "MPN", "tgt": TYPES + "Retailer",
     "pred": MPN_ISSUEDBY, "rel": "41", "ec": "685"},
    {"src": TYPES + "RetailerSKU", "tgt": TYPES + "Retailer",
     "pred": RSKU_ISSUEDBY, "rel": "604", "ec": "604"},
    {"src": TYPES + "SKU", "tgt": TYPES + "Retailer",
     "pred": SKU_ISSUED_BY, "rel": "0", "ec": "604"},
]


def _edges_router(*, core_slots=()):
    """Route the type-edges endpoint reads for both flag states.

    - The OFF read selects ``forType`` + ``targetType`` (no ``forPred``) and
      carries only src/tgt; every edge passes through unfiltered.
    - The ON read additionally selects ``forPred`` (+ rel + entityCount) so the
      floor can be applied; the core-slot query is keyed on ``coreSlot``.
    """
    def route(sparql, *a, **k):
        if "coreSlot" in sparql:
            return _rows(*({"attr": c} for c in core_slots))
        if "targetType" in sparql and "forPred" in sparql:
            # Drift-aware (flag ON) edge read.
            return _rows(*_EDGE_ROWS)
        if "targetType" in sparql and "forType" in sparql:
            # Plain (flag OFF) edge read — only src/tgt are projected.
            return _rows(*({"src": r["src"], "tgt": r["tgt"]} for r in _EDGE_ROWS))
        return _rows()

    return route


def test_flag_off_keeps_low_coverage_edge(client, mock_neptune, auth_headers, monkeypatch):
    """Flag OFF: the overview returns every stats edge — no floor applied.

    The 6%-coverage MPN->Retailer drift edge must still be present (today's
    behavior is preserved exactly when the flag is unset).
    """
    monkeypatch.delenv("OMNIX_DRIFT_CONTROL", raising=False)
    mock_neptune.query.side_effect = _edges_router()

    resp = client.get(f"/graphs/{TENANT}/explore/kgs/{KG}/type-edges", headers=auth_headers)
    assert resp.status_code == 200
    pairs = {tuple(sorted((e["source"], e["target"]))) for e in resp.json()}

    assert ("MPN", "Retailer") in pairs           # the 6% drift edge — kept when OFF
    assert ("Retailer", "RetailerSKU") in pairs    # the 100% edge
    assert ("Retailer", "SKU") in pairs            # the core-slot edge


def test_flag_on_excludes_low_coverage_edge(client, mock_neptune, auth_headers, monkeypatch):
    """Flag ON: the 6% MPN->Retailer edge is dropped; 100% + core-slot survive.

    MPN->Retailer (41/685) falls below the 20% floor -> excluded. RetailerSKU
    (604/604) clears it. SKU.issued_by has 0 support but is a core slot, so it
    is exempt and stays drawn.
    """
    monkeypatch.setenv("OMNIX_DRIFT_CONTROL", "1")
    mock_neptune.query.side_effect = _edges_router(core_slots=(SKU_ISSUED_BY,))

    resp = client.get(f"/graphs/{TENANT}/explore/kgs/{KG}/type-edges", headers=auth_headers)
    assert resp.status_code == 200
    pairs = {tuple(sorted((e["source"], e["target"]))) for e in resp.json()}

    assert ("MPN", "Retailer") not in pairs        # 6% drift edge EXCLUDED
    assert ("Retailer", "RetailerSKU") in pairs     # 100% edge kept
    assert ("Retailer", "SKU") in pairs             # core slot exempt -> kept


def test_flag_on_without_core_marker_excludes_zero_support(client, mock_neptune, auth_headers, monkeypatch):
    """Flag ON but SKU.issued_by NOT marked core: its 0-support edge is dropped.

    Confirms the core-slot exemption is what saves the SKU edge, not some
    accidental pass-through — without the marker a 0-support edge quarantines.
    """
    monkeypatch.setenv("OMNIX_DRIFT_CONTROL", "1")
    mock_neptune.query.side_effect = _edges_router(core_slots=())  # nothing marked core

    resp = client.get(f"/graphs/{TENANT}/explore/kgs/{KG}/type-edges", headers=auth_headers)
    assert resp.status_code == 200
    pairs = {tuple(sorted((e["source"], e["target"]))) for e in resp.json()}

    assert ("MPN", "Retailer") not in pairs        # 6% drift edge excluded
    assert ("Retailer", "SKU") not in pairs         # 0 support, no marker -> excluded
    assert ("Retailer", "RetailerSKU") in pairs     # 100% edge still kept


# --- recompute drift report ---------------------------------------------------

def _recompute_router(*, core_slots=()):
    """Route the whole-KG recompute scan + the core-slot lookup.

    The scan rows reproduce the reference cases: MPN (685 instances, issuedby
    41), RetailerSKU (604, issuedby 604). The drift report is built from the
    relationship declarations (predicates with rel > 0).
    """
    def route(sparql, *a, **k):
        if "coreSlot" in sparql:
            return _rows(*({"attr": c} for c in core_slots))
        if "?e ?p ?o" in sparql and "GROUP BY" in sparql:
            return _rows(
                {"type": TYPES + "MPN", "p": RDF_TYPE, "cnt": "685", "rel": "0"},
                {"type": TYPES + "MPN", "p": MPN_ISSUEDBY, "cnt": "41", "rel": "41",
                 "sample": ENTITIES + "Retailer/r1"},
                {"type": TYPES + "RetailerSKU", "p": RDF_TYPE, "cnt": "604", "rel": "0"},
                {"type": TYPES + "RetailerSKU", "p": RSKU_ISSUEDBY, "cnt": "604", "rel": "604",
                 "sample": ENTITIES + "Retailer/r2"},
            )
        return _rows()

    return route


@pytest.mark.asyncio
async def test_recompute_no_drift_key_when_flag_off(mock_neptune, monkeypatch):
    """Flag OFF: ``recompute_kg_stats`` returns the unchanged dict (no ``drift``)."""
    monkeypatch.delenv("OMNIX_DRIFT_CONTROL", raising=False)
    mock_neptune.query.side_effect = _recompute_router()

    out = await explore.recompute_kg_stats(mock_neptune, TENANT, KG)
    assert "drift" not in out
    assert set(out) == {"types", "predicate_rows"}


@pytest.mark.asyncio
async def test_recompute_drift_report_when_flag_on(mock_neptune, monkeypatch):
    """Flag ON: the returned dict carries a ``drift`` report splitting the edges.

    MPN.issuedby (41/685, 6%) quarantines; RetailerSKU.issuedby (604/604)
    declares. The report shape matches ``drift_control.drift_report``.
    """
    monkeypatch.setenv("OMNIX_DRIFT_CONTROL", "1")
    mock_neptune.query.side_effect = _recompute_router()

    out = await explore.recompute_kg_stats(mock_neptune, TENANT, KG)
    assert "drift" in out
    report = out["drift"]
    assert report["floor_cov"] == 20.0
    assert report["floor_count"] == 5
    assert report["kept"] == 1          # RetailerSKU.issuedby
    assert report["quarantined"] == 1   # MPN.issuedby
    quarantined_keys = {q["key"] for q in report["quarantine"]}
    assert quarantined_keys == {"MPN.issuedby"}
    assert report["quarantine"][0]["support"] == 41
    assert report["quarantine"][0]["coverage"] == 5.99
