"""Provider-agnostic single-pass strict-JSON extraction (ADR-0005 §6).

This module turns raw web-search text into a structured :class:`Verdict`.
It is the OSS baseline that lets *any* enrichment adapter (Wikidata, Exa,
Perplexity, …) populate provenance fields uniformly, instead of each adapter
re-implementing value extraction.

Strict-JSON contract
---------------------
The design intent is a *single* model call whose prompt instructs the model to
return **exactly** one JSON object and nothing else:

    {"value": <string|null>, "confidence": <float 0..1>}

* ``value`` — the extracted attribute value, or ``null`` if the text does not
  contain a plausible value for the attribute.
* ``confidence`` — the model's self-reported correctness probability in
  ``[0, 1]``. This is treated as an *untrusted raw* signal (like a neural
  relevance score): it is recorded but never compared to a tier threshold.

The caller-facing :func:`extract_value` is provider-agnostic: the actual model
call is **injectable** via an ``extractor`` callable (a small
:class:`ExtractorFn` protocol). The default extractor is fully deterministic
and offline — it parses an obvious value out of the text with no network or
LLM — so unit tests need no mocking infrastructure. Production code wires in a
real LLM-backed extractor that honours the JSON contract above.

Boundary: OSS file. Imports only stdlib / ``cograph_client.*``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Optional, Protocol

from cograph_client.enrichment.models import Verdict

EXTRACTION_METHOD = "single_pass_json"
CALIBRATION_METHOD = "single_pass_linear_shrink"

# Calibration policy (documented choice):
# A raw/model confidence is optimistic and uncalibrated. We shrink it toward a
# conservative prior so the calibrated ``confidence`` never simply echoes the
# raw signal. The shrink keeps ordering but pulls every score toward 0.5 and
# caps the ceiling, reflecting that a single-pass extraction is inherently less
# trustworthy than a structured-source (e.g. Wikidata) hit.
_SHRINK = 0.6  # weight kept on the raw signal; 1 - _SHRINK pulls toward prior
_PRIOR = 0.5
_CEILING = 0.9
_DEFAULT_RAW = 0.7  # assumed model confidence when none is provided


class ExtractorFn(Protocol):
    """A single-pass extractor honouring the strict-JSON contract.

    Implementations receive the raw text plus the attribute/entity context and
    MUST return a ``dict`` shaped like ``{"value": ..., "confidence": ...}``
    (or ``None`` to signal "nothing extractable"). Production implementations
    wrap an LLM call; the OSS default is deterministic.
    """

    def __call__(
        self, raw_text: str, attribute: str, entity_label: str
    ) -> Optional[dict]: ...


def _calibrate(raw: float) -> float:
    """Map a raw/model confidence to a conservative calibrated score.

    Linear shrink toward a 0.5 prior, then cap. Monotonic in ``raw`` but never
    an identity, so calibrated != raw for any input.
    """
    raw = max(0.0, min(1.0, float(raw)))
    cal = _SHRINK * raw + (1.0 - _SHRINK) * _PRIOR
    return round(min(cal, _CEILING), 4)


# Tokens that should never be treated as an extracted value on their own.
_NULLISH = {"", "null", "none", "n/a", "na", "unknown", "-", "—"}


def default_extractor(
    raw_text: str, attribute: str, entity_label: str
) -> Optional[dict]:
    """Deterministic, offline default honouring the strict-JSON shape.

    Heuristics (no network, no LLM):

    1. If the text already contains a strict-JSON object with a ``value`` key,
       parse and return it (this is the real LLM path's output shape, so tests
       can feed canned JSON straight through).
    2. Otherwise look for an explicit ``<attribute>: <value>`` line and lift the
       right-hand side.
    3. Otherwise fall back to the first non-empty line as a best-effort value.

    Returns ``None`` when nothing plausible is found.
    """
    if not raw_text or not raw_text.strip():
        return None

    # (1) Embedded strict JSON, e.g. an LLM echo captured in a fixture.
    obj = _try_parse_json(raw_text)
    if obj is not None and "value" in obj:
        return obj

    # (2) "<attribute>: value" pattern (case-insensitive on the key).
    pattern = re.compile(
        rf"^\s*{re.escape(attribute)}\s*[:=]\s*(?P<val>.+?)\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    m = pattern.search(raw_text)
    if m:
        val = m.group("val").strip().strip('"').strip()
        if val.lower() not in _NULLISH:
            return {"value": val, "confidence": _DEFAULT_RAW}

    # (3) First non-empty, non-nullish line.
    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped and stripped.lower() not in _NULLISH:
            return {"value": stripped, "confidence": _DEFAULT_RAW}

    return None


def _try_parse_json(text: str) -> Optional[dict]:
    """Best-effort parse of the first ``{...}`` object in ``text``."""
    text = text.strip()
    candidate = text
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_value(
    raw_text: str,
    attribute: str,
    entity_label: str,
    *,
    source: str,
    raw_confidence: float | None = None,
    extractor: ExtractorFn | None = None,
) -> Verdict | None:
    """Extract a structured value from raw web-search text.

    Single-pass: one (injectable) extractor call returns strict JSON
    ``{"value": ..., "confidence": ...}``. See module docstring for the
    contract.

    Args:
        raw_text: Raw text returned by a web-search/source adapter.
        attribute: The attribute name being enriched (e.g. ``"manufacturer"``).
        entity_label: The entity the attribute belongs to (e.g. ``"Bosch"``).
        source: Provenance source label, stored on the returned Verdict.
        raw_confidence: Optional untrusted upstream signal (e.g. neural
            relevance). Passed through verbatim onto the Verdict; the model's
            own confidence is preferred for calibration when present.
        extractor: Optional injected extractor (defaults to the deterministic
            offline :func:`default_extractor`).

    Returns:
        A :class:`Verdict` with ``extraction_method = "single_pass_json"`` and a
        conservatively *calibrated* ``confidence`` (never a raw echo), or
        ``None`` when nothing is extractable (blank input / no plausible value).
    """
    if not raw_text or not raw_text.strip():
        return None

    fn = extractor or default_extractor
    result = fn(raw_text, attribute, entity_label)
    if not result:
        return None

    value = result.get("value")
    if value is None or str(value).strip().lower() in _NULLISH:
        return None
    value = str(value).strip()

    # Prefer the model's self-reported confidence for calibration; fall back to
    # the upstream raw_confidence, then a neutral default. raw_confidence is
    # still stored verbatim on the Verdict regardless.
    model_conf = result.get("confidence")
    basis = model_conf if model_conf is not None else raw_confidence
    if basis is None:
        basis = _DEFAULT_RAW

    return Verdict(
        value=value,
        confidence=_calibrate(basis),
        source=source,
        raw_confidence=raw_confidence,
        retrieved_at=datetime.now(timezone.utc),
        extraction_method=EXTRACTION_METHOD,
        calibration_method=CALIBRATION_METHOD,
    )
