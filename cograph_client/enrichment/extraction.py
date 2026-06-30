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


# ---------------------------------------------------------------------------
# URL-valued attributes
# ---------------------------------------------------------------------------
# Enriching a "website" / "homepage" / "*_url" attribute is special: the answer
# is a URL, and the canonical URL the adapter resolved is the verdict's own
# ``source_url`` citation — NOT a value lifted from the page body. Running the
# single-pass extractor over the page text otherwise returns page chrome ("Skip
# to content", "Platform") or the entity name as the "value". So for URL-valued
# attributes we prefer a URL: keep the extracted value when it is itself
# URL-shaped (e.g. Wikidata's official-website P856, where the value IS the site
# and ``source_url`` is only the Wikidata provenance page), and fall back to
# ``source_url`` only when the extracted value is not a URL.

# Any URL-typed attribute (broad) — gates whether coercion runs at all and
# whether an already-URL value is normalized.
_URL_ATTR_NAMES = frozenset({
    "website", "url", "homepage", "home_page", "webpage", "web_page",
    "web_site", "website_url", "homepage_url", "web_address", "weburl",
})
_URL_ATTR_SUFFIXES = ("_url", "_uri", "_website", "_homepage")
_URL_DATATYPES = frozenset({"uri", "url", "anyuri"})

# The entity's-OWN-website family (narrow) — only these may recover from the
# resolved ``source_url`` citation, because for them the page the adapter
# resolved IS the answer. For a generic ``image_url`` / ``redirect_uri`` /
# ``logo_uri`` the citation page is the WRONG value, so those never fall back.
_WEBSITE_ATTR_NAMES = frozenset({
    "website", "url", "homepage", "home_page", "webpage", "web_page",
    "web_site", "website_url", "homepage_url", "web_address", "weburl",
    "site", "web", "domain",
})

# File extensions a bare dotted token may end in — a "value" like ``file.tar.gz``
# or ``report.docx`` is a filename, NOT a domain, and must not be treated as a
# URL (otherwise junk gets kept-and-https-prefixed instead of recovering the
# real citation).
_FILE_EXTS = frozenset({
    "gz", "tar", "zip", "rar", "7z", "pdf", "doc", "docx", "xls", "xlsx",
    "ppt", "pptx", "csv", "tsv", "txt", "md", "rtf", "png", "jpg", "jpeg",
    "gif", "svg", "webp", "bmp", "ico", "mp3", "mp4", "mov", "avi", "mkv",
    "wav", "json", "xml", "yaml", "yml", "css", "exe", "dmg", "pkg",
})

_URL_SCHEME_RE = re.compile(r"^https?://\S+$", re.IGNORECASE)
# A bare domain: unicode-friendly labels + a >=2-letter TLD, optional
# path/port/query/fragment. (IDN raw-unicode is accepted so a non-ASCII website
# value isn't dropped; ``¡-￿`` covers the BMP letter range.)
_BARE_DOMAIN_RE = re.compile(
    r"^(?:[a-z0-9¡-￿-]+\.)+(?P<tld>[a-z¡-￿]{2,})(?:[/:?#]\S*)?$",
    re.IGNORECASE,
)


def is_url_attribute(attribute: str, datatype: str | None = None) -> bool:
    """True when an attribute's value should be a URL (``website``, ``*_url``,
    or a declared ``uri`` datatype)."""
    if datatype and str(datatype).strip().lower() in _URL_DATATYPES:
        return True
    name = (attribute or "").strip().lower().replace(" ", "_").replace("-", "_")
    if name in _URL_ATTR_NAMES:
        return True
    return name.endswith(_URL_ATTR_SUFFIXES)


def is_website_attribute(attribute: str, datatype: str | None = None) -> bool:
    """True for the entity's-own-website family — the only URL attributes that
    may recover their value from the resolved ``source_url`` citation."""
    name = (attribute or "").strip().lower().replace(" ", "_").replace("-", "_")
    if name in _WEBSITE_ATTR_NAMES:
        return True
    return name.endswith(("_website", "_homepage"))


def looks_like_url(text: str | None) -> bool:
    """Loose check that ``text`` is a single URL or bare domain (no whitespace).

    Rejects filenames (``file.tar.gz``), version-ish tokens (``v2.5`` — TLD must
    be >=2 letters), and abbreviations (``e.g`` / ``U.S.A`` — TLD too short).
    """
    if not text:
        return False
    t = text.strip()
    if not t or any(c.isspace() for c in t):
        return False
    if _URL_SCHEME_RE.match(t):
        return True
    m = _BARE_DOMAIN_RE.match(t)
    if not m:
        return False
    # A bare token whose final label is a file extension is a filename, not a host.
    return m.group("tld").lower() not in _FILE_EXTS


def normalize_url(text: str) -> str:
    """Ensure an ``https://`` scheme and trim a path-terminating trailing slash.

    The path is preserved (never reduced to origin — a profile/sub-page URL is
    still the answer for a ``*_url`` attribute). A trailing slash is stripped
    ONLY when it terminates the path — never inside a ``?query`` or ``#fragment``
    (where a ``/`` can be a significant character)."""
    t = (text or "").strip()
    if not t:
        return t
    if not re.match(r"^https?://", t, re.IGNORECASE):
        t = "https://" + t
    if "?" not in t and "#" not in t:
        t = t.rstrip("/")
    return t


def coerce_url_attribute_value(
    attribute: str, verdict: Verdict, datatype: str | None = None
) -> Verdict:
    """For a URL-valued attribute, ensure the verdict's ``value`` is a URL.

    * non-URL attribute → returned unchanged.
    * value already URL-shaped → normalized (keeps Wikidata's official-website
      value — the site itself — rather than overwriting it with the Wikidata
      provenance page).
    * value not URL-shaped (page chrome / entity name) → replaced with the
      resolved ``source_url`` citation ONLY for the entity's-own-website family
      (where the citation IS the site); other ``*_url`` attributes keep their
      value rather than inherit a wrong provenance page.
    """
    if not is_url_attribute(attribute, datatype):
        return verdict
    val = (verdict.value or "").strip()
    if looks_like_url(val):
        norm = normalize_url(val)
        return verdict if norm == verdict.value else verdict.model_copy(update={"value": norm})
    if is_website_attribute(attribute, datatype):
        src = (getattr(verdict, "source_url", None) or "").strip()
        if looks_like_url(src):
            return verdict.model_copy(update={"value": normalize_url(src)})
    return verdict
