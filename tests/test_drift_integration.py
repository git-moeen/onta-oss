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


def test_observe_only_does_not_filter_overview(client, mock_neptune, auth_headers, monkeypatch):
    """Flag ON + OBSERVE_ONLY: the overview is NOT filtered (acts like flag OFF).

    Observe-only collects the coverage distribution via the recompute report
    without acting on the overview — so the 6% MPN->Retailer drift edge must
    still be drawn, exactly as when the feature is off. This is the de-risking
    mode: measure before you filter.
    """
    monkeypatch.setenv("OMNIX_DRIFT_CONTROL", "1")
    monkeypatch.setenv("OMNIX_DRIFT_OBSERVE_ONLY", "1")
    mock_neptune.query.side_effect = _edges_router(core_slots=(SKU_ISSUED_BY,))

    resp = client.get(f"/graphs/{TENANT}/explore/kgs/{KG}/type-edges", headers=auth_headers)
    assert resp.status_code == 200
    pairs = {tuple(sorted((e["source"], e["target"]))) for e in resp.json()}

    assert ("MPN", "Retailer") in pairs            # 6% drift edge NOT filtered in observe-only
    assert ("Retailer", "RetailerSKU") in pairs
    assert ("Retailer", "SKU") in pairs


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


# --- live-scan fallback (legacy KG with NO materialized stats) ----------------
# Reference: a KG ingested before stats existed. The stats-drift read returns no
# bindings (=> _read_edges_from_stats_drift returns None), so the endpoint falls
# back to the live instance scan, which must apply the SAME floor when the flag
# is ON. Instance predicates are minted as `…/onto/<predName>`; the matching
# core-slot marker lives at `…/types/<srcLeaf>/attrs/<predName>`.
ONTO = "https://cograph.tech/onto/"
MPN_ISSUEDBY_ONTO = ONTO + "issuedby"          # MPN instances: 41/685 -> 6%
RSKU_ISSUEDBY_ONTO = ONTO + "issuedby"          # RetailerSKU: 600/600 -> 100%
SKU_ISSUED_BY_ONTO = ONTO + "issued_by"         # SKU core slot, 0/604 -> exempt
# Core-slot attr URI (what _live_edge_scan_drift joins on): attr_uri(src, pred).
SKU_ISSUED_BY_ATTR = TYPES + "SKU/attrs/issued_by"


def _live_scan_router(*, core_slots=()):
    """Route the legacy-KG live-scan reads (flag ON, NO stats materialized).

    - The drift stats read (``targetType`` + ``forPred``) returns NO bindings so
      ``_read_edges_from_stats_drift`` returns None and the endpoint falls back
      to the live instance scan.
    - The live-scan edge aggregation groups by ``?type ?p ?tgt`` (``COUNT(DISTINCT``
      over the KG graph). One LOW edge (MPN.issuedby 41/685 = 6%) and one HIGH
      edge (RetailerSKU.issuedby 600/600 = 100%), plus a 0-support SKU core slot.
    - The per-type entity-count query groups by ``?type`` only.
    - The core-slot lookup is keyed on ``coreSlot``.
    """
    def route(sparql, *a, **k):
        if "coreSlot" in sparql:
            return _rows(*({"attr": c} for c in core_slots))
        # Drift stats read — empty so the endpoint falls to the live scan.
        if "targetType" in sparql:
            return _rows()
        # Live-scan edge aggregation: groups by source type, predicate, target.
        if "GROUP BY ?type ?p ?tgt" in sparql:
            return _rows(
                {"type": TYPES + "MPN", "p": MPN_ISSUEDBY_ONTO,
                 "tgt": "Retailer", "support": "41"},        # 41/685 -> 6%
                {"type": TYPES + "RetailerSKU", "p": RSKU_ISSUEDBY_ONTO,
                 "tgt": "Retailer", "support": "600"},        # 600/600 -> 100%
                {"type": TYPES + "SKU", "p": SKU_ISSUED_BY_ONTO,
                 "tgt": "Retailer", "support": "0"},          # core slot, exempt
            )
        # Per-type entity counts (source_count).
        if "GROUP BY ?type" in sparql:
            return _rows(
                {"type": TYPES + "MPN", "ec": "685"},
                {"type": TYPES + "RetailerSKU", "ec": "600"},
                {"type": TYPES + "SKU", "ec": "604"},
            )
        return _rows()

    return route


def test_flag_on_live_scan_excludes_low_coverage_edge(client, mock_neptune, auth_headers, monkeypatch):
    """Flag ON, legacy KG (NO stats): the live-scan path applies the floor.

    The stats-drift read returns None, so the endpoint falls back to the live
    instance scan. The 6% MPN->Retailer edge (41/685) must be EXCLUDED there too
    — the production gap this fixes — while the 100% RetailerSKU edge (600/600)
    is kept.
    """
    monkeypatch.setenv("OMNIX_DRIFT_CONTROL", "1")
    mock_neptune.query.side_effect = _live_scan_router()

    resp = client.get(f"/graphs/{TENANT}/explore/kgs/{KG}/type-edges", headers=auth_headers)
    assert resp.status_code == 200
    pairs = {tuple(sorted((e["source"], e["target"]))) for e in resp.json()}

    assert ("MPN", "Retailer") not in pairs        # 6% drift edge EXCLUDED on live scan
    assert ("Retailer", "RetailerSKU") in pairs     # 100% edge kept


def test_flag_on_live_scan_core_slot_exempt(client, mock_neptune, auth_headers, monkeypatch):
    """Flag ON, legacy KG: a 0-support core slot is exempt on the live-scan path.

    SKU.issued_by carries 0 support (0/604) but is marked a core slot, so it is
    declared even on the live scan. Confirms the live scan reads the ontology
    core-slot marker (keyed by attr_uri) the same way the stats path does.
    """
    monkeypatch.setenv("OMNIX_DRIFT_CONTROL", "1")
    mock_neptune.query.side_effect = _live_scan_router(core_slots=(SKU_ISSUED_BY_ATTR,))

    resp = client.get(f"/graphs/{TENANT}/explore/kgs/{KG}/type-edges", headers=auth_headers)
    assert resp.status_code == 200
    pairs = {tuple(sorted((e["source"], e["target"]))) for e in resp.json()}

    assert ("MPN", "Retailer") not in pairs        # 6% drift edge still excluded
    assert ("Retailer", "SKU") in pairs             # core slot exempt -> kept
    assert ("Retailer", "RetailerSKU") in pairs     # 100% edge kept


def test_flag_off_live_scan_keeps_low_coverage_edge(client, mock_neptune, auth_headers, monkeypatch):
    """Flag OFF, legacy KG: the unfiltered live scan keeps the 6% drift edge.

    With the flag OFF the endpoint must take the original unfiltered
    ``_live_edge_scan`` (byte-identical to before): the 6% MPN->Retailer edge is
    still drawn. Routed via the plain instance scan (``?e ?p ?o`` over the KG,
    no aggregation), which the OFF path uses when stats are absent.
    """
    monkeypatch.delenv("OMNIX_DRIFT_CONTROL", raising=False)

    def route(sparql, *a, **k):
        # Plain stats read (forType + targetType, no forPred) — empty so the
        # endpoint falls back to the unfiltered live scan.
        if "targetType" in sparql:
            return _rows()
        # Unfiltered live scan: distinct (?type, ?o) instance pairs, no GROUP BY.
        if "?e ?p ?o" in sparql and "GROUP BY" not in sparql:
            return _rows(
                {"type": TYPES + "MPN", "o": ENTITIES + "Retailer/r1"},
                {"type": TYPES + "RetailerSKU", "o": ENTITIES + "Retailer/r2"},
            )
        return _rows()

    mock_neptune.query.side_effect = route

    resp = client.get(f"/graphs/{TENANT}/explore/kgs/{KG}/type-edges", headers=auth_headers)
    assert resp.status_code == 200
    pairs = {tuple(sorted((e["source"], e["target"]))) for e in resp.json()}

    assert ("MPN", "Retailer") in pairs            # 6% drift edge kept when OFF
    assert ("Retailer", "RetailerSKU") in pairs     # 100% edge kept


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


@pytest.mark.asyncio
async def test_recompute_drift_report_in_observe_only(mock_neptune, monkeypatch):
    """Observe-only STILL produces the report + the full coverage distribution.

    The whole point of observe-only is to collect the real spread of
    relationship coverages, so recompute must compute and return the report (the
    overview just doesn't act on it). The report carries `coverages` for EVERY
    relationship (kept + quarantined), the histogram source for setting the floor
    from real data.
    """
    monkeypatch.setenv("OMNIX_DRIFT_CONTROL", "1")
    monkeypatch.setenv("OMNIX_DRIFT_OBSERVE_ONLY", "1")
    mock_neptune.query.side_effect = _recompute_router()

    out = await explore.recompute_kg_stats(mock_neptune, TENANT, KG)
    assert "drift" in out                       # report still produced in observe-only
    report = out["drift"]
    cov_by_key = {c["key"]: c for c in report["coverages"]}
    assert set(cov_by_key) == {"MPN.issuedby", "RetailerSKU.issuedby"}  # ALL, not just quarantined
    assert cov_by_key["MPN.issuedby"]["coverage"] == 5.99
    assert cov_by_key["MPN.issuedby"]["kept"] is False
    assert cov_by_key["RetailerSKU.issuedby"]["coverage"] == 100.0
    assert cov_by_key["RetailerSKU.issuedby"]["kept"] is True


# --- core-slot exemption: attr_uri match, not raw predicate URI ---------------
# These prove the bug fix in BOTH the stats-drift read and _build_drift_report:
# ?pred/pred_uri there is the INSTANCE predicate URI (…/onto/<pred>), while the
# core-slot query returns ontology ATTR URIs (…/types/<Type>/attrs/<pred>). The
# old `pred in core_slots` comparison could NEVER match, so a sparse core slot
# was wrongly quarantined. The fix joins on attr_uri(<srcLeaf>, <predLeaf>).
#
# A sparse edge: MPNcore.issuedby at 41/685 (6%) — below the 20% floor, so it is
# kept ONLY if its core-slot marker is honored. The core marker lives at the
# attr URI; the stats/recompute rows carry the …/onto/ instance predicate.
MPNCORE_ISSUEDBY_ONTO = ONTO + "issuedby"                  # instance predicate URI
MPNCORE_ISSUEDBY_ATTR = TYPES + "MPNcore/attrs/issuedby"   # ontology attr URI (core marker)


def _edges_drift_core_router(*, core_slots=()):
    """Drift-aware stats read (flag ON) with a SPARSE edge on an …/onto/ predicate.

    The single edge MPNcore->Retailer is sparse (41/685 = 6%, below floor). Its
    ``?pred`` is the INSTANCE predicate URI (…/onto/issuedby) exactly as the real
    stats graph stores it — so the old ``pred in core_slots`` comparison (which
    expects an attr URI) can never match it, and the fix's attr_uri join is what
    exercises the core-slot exemption.
    """
    def route(sparql, *a, **k):
        if "coreSlot" in sparql:
            return _rows(*({"attr": c} for c in core_slots))
        if "targetType" in sparql and "forPred" in sparql:
            return _rows(
                {"src": TYPES + "MPNcore", "tgt": TYPES + "Retailer",
                 "pred": MPNCORE_ISSUEDBY_ONTO, "rel": "41", "ec": "685"},
            )
        return _rows()

    return route


def test_flag_on_stats_sparse_core_slot_kept_via_attr_uri(client, mock_neptune, auth_headers, monkeypatch):
    """Flag ON, stats materialized: a sparse edge on a CORE slot is KEPT (exempt).

    MPNcore.issuedby at 41/685 (6%) is below the 20% floor, but its attr URI is
    marked a core slot, so it must survive. ?pred in the stats read is the
    instance URI (…/onto/issuedby), so this only passes with the attr_uri join —
    the old ``pred in core_slots`` comparison would wrongly drop it.
    """
    monkeypatch.setenv("OMNIX_DRIFT_CONTROL", "1")
    mock_neptune.query.side_effect = _edges_drift_core_router(core_slots=(MPNCORE_ISSUEDBY_ATTR,))

    resp = client.get(f"/graphs/{TENANT}/explore/kgs/{KG}/type-edges", headers=auth_headers)
    assert resp.status_code == 200
    pairs = {tuple(sorted((e["source"], e["target"]))) for e in resp.json()}

    assert ("MPNcore", "Retailer") in pairs   # sparse but core-slot exempt -> kept


def test_flag_on_stats_sparse_non_core_slot_excluded(client, mock_neptune, auth_headers, monkeypatch):
    """Flag ON, stats materialized: the SAME sparse edge WITHOUT a core marker is dropped.

    The companion to the test above — confirms 6% coverage is genuinely below the
    floor, so the core-slot exemption (not some pass-through) is what saved it.
    """
    monkeypatch.setenv("OMNIX_DRIFT_CONTROL", "1")
    mock_neptune.query.side_effect = _edges_drift_core_router(core_slots=())  # nothing marked core

    resp = client.get(f"/graphs/{TENANT}/explore/kgs/{KG}/type-edges", headers=auth_headers)
    assert resp.status_code == 200
    pairs = {tuple(sorted((e["source"], e["target"]))) for e in resp.json()}

    assert ("MPNcore", "Retailer") not in pairs   # sparse, no core marker -> excluded


def _recompute_core_router(*, core_slots=()):
    """Whole-KG recompute scan where the sparse predicate is an …/onto/ instance URI.

    MPNcore has 685 instances and a sparse issuedby relationship (41/685 = 6%).
    The scan's ``?p`` is the INSTANCE predicate URI (…/onto/issuedby), as the real
    recompute scan reads it — so _build_drift_report's old ``pred_uri in
    core_slots`` (attr URIs) never matched, wrongly quarantining the core slot.
    The fix joins on attr_uri(type_leaf, pred_leaf).
    """
    def route(sparql, *a, **k):
        if "coreSlot" in sparql:
            return _rows(*({"attr": c} for c in core_slots))
        if "?e ?p ?o" in sparql and "GROUP BY" in sparql:
            return _rows(
                {"type": TYPES + "MPNcore", "p": RDF_TYPE, "cnt": "685", "rel": "0"},
                {"type": TYPES + "MPNcore", "p": MPNCORE_ISSUEDBY_ONTO,
                 "cnt": "41", "rel": "41", "sample": ENTITIES + "Retailer/r1"},
            )
        return _rows()

    return route


@pytest.mark.asyncio
async def test_recompute_sparse_core_slot_kept_not_quarantined(mock_neptune, monkeypatch):
    """Flag ON recompute: a sparse CORE-slot relationship is reported as KEPT.

    MPNcore.issuedby (41/685, 6%) is below the floor but is a core slot, so it
    must NOT be quarantined. The recompute scan's predicate is the …/onto/
    instance URI, so this passes only with the attr_uri join in
    _build_drift_report — the old ``pred_uri in core_slots`` would quarantine it.
    """
    monkeypatch.setenv("OMNIX_DRIFT_CONTROL", "1")
    mock_neptune.query.side_effect = _recompute_core_router(core_slots=(MPNCORE_ISSUEDBY_ATTR,))

    out = await explore.recompute_kg_stats(mock_neptune, TENANT, KG)
    report = out["drift"]
    assert report["kept"] == 1          # MPNcore.issuedby exempt -> kept
    assert report["quarantined"] == 0   # core slot must NOT be quarantined
    assert report["quarantine"] == []


@pytest.mark.asyncio
async def test_recompute_sparse_non_core_slot_quarantined(mock_neptune, monkeypatch):
    """Flag ON recompute: the SAME sparse relationship WITHOUT a core marker quarantines.

    Companion to the test above: 41/685 (6%) is genuinely below the floor, so
    without the core marker it is held for review — proving the exemption (not a
    pass-through) is what kept the core case.
    """
    monkeypatch.setenv("OMNIX_DRIFT_CONTROL", "1")
    mock_neptune.query.side_effect = _recompute_core_router(core_slots=())  # nothing marked core

    out = await explore.recompute_kg_stats(mock_neptune, TENANT, KG)
    report = out["drift"]
    assert report["kept"] == 0
    assert report["quarantined"] == 1
    assert {q["key"] for q in report["quarantine"]} == {"MPNcore.issuedby"}
