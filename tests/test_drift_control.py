"""ADR 0004 drift-control decision-logic tests.

Pure-math tests for ``cograph_client.resolver.drift_control``. They pin the
calibrated rule (floor=20% + support>=5 + core-slot exemption) against the four
REFERENCE TRUTH cases from the ADR / ``scripts/adr4_drift_threshold_experiment``,
plus env-override behavior, the reconcile split, drift-report shape, and the
core-slot exemption. No Neptune, no LLM, no flag dependency — this module is
importable and correct regardless of ``OMNIX_DRIFT_CONTROL``.
"""
from __future__ import annotations

import pytest

from cograph_client.resolver import drift_control as dc


# --- coverage math ------------------------------------------------------------
def test_coverage_basic():
    # 41 of 685 == ~6% (the MPN->Retailer drift signal).
    assert dc.coverage(41, 685) == pytest.approx(5.9854, abs=1e-3)


def test_coverage_full():
    assert dc.coverage(604, 604) == 100.0


def test_coverage_source_count_zero_is_not_divide_by_zero():
    # An empty source type => 0 coverage, never a ZeroDivisionError.
    assert dc.coverage(0, 0) == 0.0
    assert dc.coverage(5, 0) == 0.0


def test_coverage_gray_zone():
    # 11 of 34 == ~32% (the genuinely-sparse-but-real billing edge).
    assert dc.coverage(11, 34) == pytest.approx(32.35, abs=1e-2)


# --- should_declare on ALL FOUR reference truth cases (default floor) ---------
def test_reference_mpn_issuedby_retailer_quarantined():
    # MPN issuedby->Retailer 41/685 (6%) => QUARANTINE (mis-domained drift).
    assert dc.should_declare(41, 685, is_core_slot=False) is False


def test_reference_retailersku_issuedby_retailer_declared():
    # RetailerSKU issuedby->Retailer 604/604 (100%) => DECLARE.
    assert dc.should_declare(604, 604, is_core_slot=False) is True


def test_reference_sku_issued_by_core_slot_declared_exempt():
    # SKU.issued_by core slot 0/604 (0%) => DECLARE (core-slot exemption).
    assert dc.should_declare(0, 604, is_core_slot=True) is True


def test_reference_gray_zone_declared_at_floor_20():
    # gray-zone 11/34 (32%) => DECLARE at floor 20 (above the floor, support>=5).
    assert dc.should_declare(11, 34, is_core_slot=False) is True


# --- floor boundary / count guard ---------------------------------------------
def test_count_guard_blocks_tiny_high_coverage_type():
    # 2 of 3 == 66% coverage but only 2 instances < FLOOR_COUNT(5) => QUARANTINE.
    assert dc.should_declare(2, 3, is_core_slot=False) is False


def test_at_exact_floor_declares():
    # coverage == 20.0 and support == 5: both at-floor (>=) => DECLARE.
    assert dc.should_declare(5, 25, is_core_slot=False) is True


# --- env override via monkeypatch (lower floor lets the 6% case through) ------
def test_env_floor_cov_override_lets_six_percent_through(monkeypatch):
    monkeypatch.setenv(dc.ENV_FLOOR_COV, "5")
    # At a 5% floor the 41/685 (~6%) edge now clears the coverage floor, and
    # support 41 >= 5, so it DECLARES — proving the env value is read at call time.
    assert dc.env_floor_cov() == 5.0
    assert dc.should_declare(41, 685, is_core_slot=False) is True


def test_env_floor_count_override(monkeypatch):
    monkeypatch.setenv(dc.ENV_FLOOR_COUNT, "50")
    assert dc.env_floor_count() == 50
    # 41 support now below the raised count floor of 50 => QUARANTINE again.
    assert dc.should_declare(41, 685, is_core_slot=False) is False


def test_explicit_floor_arg_overrides_env(monkeypatch):
    monkeypatch.setenv(dc.ENV_FLOOR_COV, "5")
    # Explicit floor_cov arg wins over the env: 6% < 20 => QUARANTINE.
    assert dc.should_declare(41, 685, is_core_slot=False, floor_cov=20.0) is False


def test_malformed_env_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv(dc.ENV_FLOOR_COV, "not-a-number")
    monkeypatch.setenv(dc.ENV_FLOOR_COUNT, "xyz")
    assert dc.env_floor_cov() == dc.DEFAULT_FLOOR_COV
    assert dc.env_floor_count() == dc.DEFAULT_FLOOR_COUNT


def test_drift_control_enabled_flag(monkeypatch):
    monkeypatch.delenv(dc.ENV_ENABLED, raising=False)
    assert dc.drift_control_enabled() is False
    monkeypatch.setenv(dc.ENV_ENABLED, "0")
    assert dc.drift_control_enabled() is False
    monkeypatch.setenv(dc.ENV_ENABLED, "true")  # only "1" enables
    assert dc.drift_control_enabled() is False
    monkeypatch.setenv(dc.ENV_ENABLED, "1")
    assert dc.drift_control_enabled() is True


def test_observe_only_flag(monkeypatch):
    monkeypatch.delenv(dc.ENV_OBSERVE_ONLY, raising=False)
    assert dc.observe_only() is False
    monkeypatch.setenv(dc.ENV_OBSERVE_ONLY, "0")
    assert dc.observe_only() is False
    monkeypatch.setenv(dc.ENV_OBSERVE_ONLY, "1")
    assert dc.observe_only() is True
    # Independent of the feature flag — `act` = enabled AND NOT observe_only is
    # composed by the caller (explore.get_type_edges), not here.
    monkeypatch.delenv(dc.ENV_ENABLED, raising=False)
    assert dc.observe_only() is True


# --- reconcile split ----------------------------------------------------------
def _reference_declarations() -> list[dict]:
    """The measured june-15-test/test retail declarations (ADR 0004 evidence)."""
    return [
        {"key": "RetailerSKU.identifies->Product", "support": 604, "source_count": 604},
        {"key": "RetailerSKU.issuedby->Retailer", "support": 604, "source_count": 604},
        {"key": "ManufacturerPartNumber.identifies->Product", "support": 685, "source_count": 685},
        {"key": "ManufacturerPartNumber.issuedby->Retailer", "support": 41, "source_count": 685},
        {"key": "Product.manufacturedby->Manufacturer", "support": 591, "source_count": 895},
        {"key": "SKU.issued_by->Supplier", "support": 0, "source_count": 604, "is_core_slot": True},
    ]


def test_reconcile_split():
    result = dc.reconcile(_reference_declarations())
    kept_keys = {d["key"] for d in result.keep}
    quarantined_keys = {d["key"] for d in result.quarantine}
    # Only the 6% mis-domained edge is held; everything real + the core slot kept.
    assert quarantined_keys == {"ManufacturerPartNumber.issuedby->Retailer"}
    assert "SKU.issued_by->Supplier" in kept_keys  # core-slot exemption
    assert "RetailerSKU.issuedby->Retailer" in kept_keys
    assert len(result.keep) == 5
    assert len(result.quarantine) == 1


def test_reconcile_does_not_mutate_input():
    decls = _reference_declarations()
    snapshot = [dict(d) for d in decls]
    dc.reconcile(decls)
    assert decls == snapshot  # pure: input untouched


def test_reconcile_default_is_core_slot_false():
    # A declaration with no is_core_slot key is treated as non-core (0% => held).
    result = dc.reconcile([{"key": "X.empty->Y", "support": 0, "source_count": 100}])
    assert len(result.quarantine) == 1
    assert len(result.keep) == 0


# --- drift_report shape / counts ----------------------------------------------
def test_drift_report_shape_and_counts():
    decls = _reference_declarations()
    report = dc.drift_report(decls)
    assert set(report.keys()) == {
        "floor_cov", "floor_count", "kept", "quarantined", "quarantine", "coverages",
    }
    assert report["floor_cov"] == 20.0
    assert report["floor_count"] == 5
    assert report["kept"] == 5
    assert report["quarantined"] == 1
    assert len(report["quarantine"]) == 1
    held = report["quarantine"][0]
    assert set(held.keys()) == {"key", "coverage", "support"}
    assert held["key"] == "ManufacturerPartNumber.issuedby->Retailer"
    assert held["support"] == 41
    assert held["coverage"] == pytest.approx(5.99, abs=1e-2)
    # `coverages` is the observe-only distribution: EVERY declaration, with its
    # kept verdict — the histogram source for setting the floor from real data.
    assert len(report["coverages"]) == len(decls)
    by_key = {c["key"]: c for c in report["coverages"]}
    assert set(by_key["ManufacturerPartNumber.issuedby->Retailer"].keys()) == {
        "key", "coverage", "support", "source_count", "is_core_slot", "kept",
    }
    assert by_key["ManufacturerPartNumber.issuedby->Retailer"]["kept"] is False


def test_drift_report_reflects_floor_override():
    # At floor 5%, the 6% edge clears and the quarantine empties.
    report = dc.drift_report(_reference_declarations(), floor_cov=5.0)
    assert report["floor_cov"] == 5.0
    assert report["quarantined"] == 0
    assert report["kept"] == 6
    assert report["quarantine"] == []


# --- core-slot exemption (standalone) -----------------------------------------
def test_core_slot_exemption_overrides_floor():
    # 0/604 below every floor, but core => DECLARE; flip the flag => QUARANTINE.
    assert dc.should_declare(0, 604, is_core_slot=True) is True
    assert dc.should_declare(0, 604, is_core_slot=False) is False
