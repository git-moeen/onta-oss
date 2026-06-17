"""Tests for the single-pass strict-JSON extraction stage (ADR-0005 §6).

All tests run fully offline: the model call is injected via the ``extractor``
argument or exercises the deterministic default extractor. No network/LLM.
"""

from __future__ import annotations

from cograph_client.enrichment.extraction import (
    CALIBRATION_METHOD,
    EXTRACTION_METHOD,
    _calibrate,
    default_extractor,
    extract_value,
)
from cograph_client.enrichment.models import Verdict


# ---------------------------------------------------------------------------
# Successful extraction populates the new Verdict provenance fields
# ---------------------------------------------------------------------------


def test_successful_extraction_populates_fields():
    def fake_extractor(raw_text, attribute, entity_label):
        # Honours the strict-JSON contract shape.
        return {"value": "Robert Bosch GmbH", "confidence": 0.8}

    v = extract_value(
        "Bosch is a German engineering company...",
        "manufacturer",
        "Bosch",
        source="exa",
        raw_confidence=0.42,
        extractor=fake_extractor,
    )
    assert isinstance(v, Verdict)
    assert v.value == "Robert Bosch GmbH"
    assert v.source == "exa"
    assert v.extraction_method == EXTRACTION_METHOD == "single_pass_json"
    assert v.calibration_method == CALIBRATION_METHOD
    # raw_confidence is stored verbatim.
    assert v.raw_confidence == 0.42
    # retrieved_at provenance is stamped.
    assert v.retrieved_at is not None
    # Calibrated confidence is a real number in range.
    assert 0.0 <= v.confidence <= 1.0


# ---------------------------------------------------------------------------
# Blank / empty input returns None
# ---------------------------------------------------------------------------


def test_blank_input_returns_none():
    assert extract_value("", "manufacturer", "Bosch", source="exa") is None
    assert extract_value("   \n  ", "manufacturer", "Bosch", source="exa") is None


def test_no_plausible_value_returns_none():
    # Extractor yields a nullish value → nothing extractable.
    def null_extractor(raw_text, attribute, entity_label):
        return {"value": None, "confidence": 0.0}

    assert (
        extract_value(
            "some text", "manufacturer", "Bosch", source="exa", extractor=null_extractor
        )
        is None
    )

    def empty_extractor(raw_text, attribute, entity_label):
        return None

    assert (
        extract_value(
            "some text", "manufacturer", "Bosch", source="exa", extractor=empty_extractor
        )
        is None
    )


# ---------------------------------------------------------------------------
# raw_confidence passes through; calibrated confidence differs from raw
# ---------------------------------------------------------------------------


def test_raw_confidence_passthrough_and_calibration_differs():
    def fake_extractor(raw_text, attribute, entity_label):
        # No model confidence → calibration falls back to raw_confidence basis.
        return {"value": "Germany"}

    raw = 0.9
    v = extract_value(
        "Germany",
        "country",
        "Bosch",
        source="exa",
        raw_confidence=raw,
        extractor=fake_extractor,
    )
    assert v is not None
    # Stored verbatim.
    assert v.raw_confidence == raw
    # Calibrated value must NOT echo the raw signal.
    assert v.confidence != raw
    # And it should be conservative (shrunk toward the prior / capped).
    assert v.confidence < raw


def test_calibration_is_monotonic_and_never_identity():
    # Different raw inputs map to different (ordered) calibrated outputs, none
    # equal to the input.
    low = _calibrate(0.2)
    high = _calibrate(0.95)
    assert low != 0.2
    assert high != 0.95
    assert low < high
    # Ceiling cap holds.
    assert _calibrate(1.0) <= 0.9


# ---------------------------------------------------------------------------
# Default (offline) extractor heuristics
# ---------------------------------------------------------------------------


def test_default_extractor_parses_embedded_json():
    out = default_extractor('{"value": "Germany", "confidence": 0.77}', "country", "Bosch")
    assert out == {"value": "Germany", "confidence": 0.77}


def test_default_extractor_parses_attribute_line():
    out = default_extractor("manufacturer: Robert Bosch GmbH", "manufacturer", "Bosch")
    assert out is not None
    assert out["value"] == "Robert Bosch GmbH"


def test_default_extractor_blank_returns_none():
    assert default_extractor("", "x", "y") is None
    assert default_extractor("   ", "x", "y") is None


def test_extract_value_end_to_end_with_default_extractor():
    # No injected extractor → uses the deterministic default, fully offline.
    v = extract_value(
        '{"value": "Germany", "confidence": 0.85}',
        "country",
        "Bosch",
        source="web",
    )
    assert v is not None
    assert v.value == "Germany"
    assert v.source == "web"
    assert v.extraction_method == "single_pass_json"
    # Model confidence 0.85 is calibrated, not echoed.
    assert v.confidence != 0.85
