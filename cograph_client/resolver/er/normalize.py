"""Default Normalizer implementation for cross-file entity resolution.

Pure-stdlib normalization: diacritic stripping, honorific/suffix removal,
nickname expansion, gmail-dot canonicalization, E.164 phone shaping,
USPS-style address abbreviation, and best-effort DOB ISO parsing.

This module intentionally avoids any external dependency (no phonenumbers,
no usaddress, no dateutil) so it can run inside the OSS client without
extra wheels. Edge cases (international phone CC inference, non-Latin name
scripts, non-USPS address vocab) are punted to a future proprietary
normalizer that can satisfy the same Protocol.
"""

from __future__ import annotations

import re
import unicodedata

from .nickname import NICKNAME_TO_CANONICAL
from .types import EntitySignals, NormalizedSignals

# ---------------------------------------------------------------------------
# Name helpers
# ---------------------------------------------------------------------------

_HONORIFICS: frozenset[str] = frozenset({"mr", "mrs", "ms", "miss", "dr", "sir", "dame"})
_SUFFIXES: frozenset[str] = frozenset({"jr", "sr", "ii", "iii", "iv", "phd", "md"})

# Keep letters, hyphen, apostrophe, whitespace; nuke periods/commas/etc.
_NAME_KEEP_RE = re.compile(r"[^a-z\s'\-]")
_WS_RE = re.compile(r"\s+")


def _strip_diacritics(s: str) -> str:
    decomposed = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _normalize_name(raw: str | None) -> tuple[str | None, tuple[str, ...]]:
    if not raw:
        return None, ()
    s = _strip_diacritics(raw).lower().strip()
    if not s:
        return None, ()
    # strip periods etc. before tokenizing so "mr." -> "mr"
    s = _NAME_KEEP_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return None, ()
    tokens = [t for t in s.split(" ") if t]
    # drop honorifics / suffixes (token-level)
    tokens = [t for t in tokens if t not in _HONORIFICS and t not in _SUFFIXES]
    if not tokens:
        return None, ()
    # expand first-token nickname
    first = tokens[0]
    if first in NICKNAME_TO_CANONICAL:
        tokens[0] = NICKNAME_TO_CANONICAL[first]
    cleaned = " ".join(tokens)
    return cleaned, tuple(sorted(tokens))


# ---------------------------------------------------------------------------
# Email helpers
# ---------------------------------------------------------------------------

_GMAIL_DOMAINS: frozenset[str] = frozenset({"gmail.com", "googlemail.com"})


def _normalize_email(raw: str | None) -> tuple[str | None, str | None]:
    if not raw:
        return None, None
    s = raw.strip().lower()
    if not s:
        return None, None
    if "@" in s:
        local, _, domain = s.rpartition("@")
        domain = domain.strip()
    else:
        local, domain = s, ""
    # drop +tag
    if "+" in local:
        local = local.split("+", 1)[0]
    # gmail: drop all dots in local part
    if domain in _GMAIL_DOMAINS:
        local = local.replace(".", "")
    if not local:
        return None, None
    if domain:
        return f"{local}@{domain}", local
    return local, local


# ---------------------------------------------------------------------------
# Phone helpers
# ---------------------------------------------------------------------------

_PHONE_KEEP_RE = re.compile(r"[^\d+]")


def _normalize_phone(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    has_plus = s.lstrip().startswith("+")
    digits = _PHONE_KEEP_RE.sub("", s).lstrip("+")
    if not digits:
        return None
    if has_plus:
        return f"+{digits}"
    if len(digits) == 10:
        return f"+1{digits}"
    if len(digits) == 11 and digits.startswith("1"):
        return f"+{digits}"
    return f"+{digits}"


# ---------------------------------------------------------------------------
# Address helpers
# ---------------------------------------------------------------------------

_ADDR_ABBREV: dict[str, str] = {
    "street": "st",
    "avenue": "ave",
    "road": "rd",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "court": "ct",
    "place": "pl",
    "suite": "ste",
    "apartment": "apt",
    "north": "n",
    "south": "s",
    "east": "e",
    "west": "w",
}

# matches "apt 4b", "ste. 200", "unit 12", "# 7" and everything after.
_UNIT_TAIL_RE = re.compile(r"\b(apt|ste|unit)\b\.?\s*\S*.*$", re.IGNORECASE)
_HASH_TAIL_RE = re.compile(r"#.*$")
_ADDR_PUNCT_RE = re.compile(r"[.,;]")


def _normalize_address(raw: str | None) -> tuple[str | None, tuple[str, ...]]:
    if not raw:
        return None, ()
    s = raw.lower().strip()
    if not s:
        return None, ()
    s = _ADDR_PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    # strip unit tails before abbreviating (so "apartment 4" -> dropped, not "apt 4" kept)
    s = _UNIT_TAIL_RE.sub("", s)
    s = _HASH_TAIL_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    if not s:
        return None, ()
    tokens = s.split(" ")
    tokens = [_ADDR_ABBREV.get(t, t) for t in tokens if t]
    cleaned = " ".join(tokens).strip()
    if not cleaned:
        return None, ()
    return cleaned, tuple(sorted(tokens))


# ---------------------------------------------------------------------------
# DOB helpers
# ---------------------------------------------------------------------------

_ISO_RE = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")
_SLASH_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_DASH_DMY_RE = re.compile(r"^(\d{1,2})-(\d{1,2})-(\d{4})$")


def _valid_ymd(y: int, m: int, d: int) -> bool:
    if not (1 <= m <= 12 and 1 <= d <= 31):
        return False
    # rough month-length check
    month_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return d <= month_days[m - 1]


def _normalize_dob(raw: str | None) -> str | None:
    if not raw:
        return None
    s = raw.strip()
    if not s:
        return None
    # ISO YYYY-MM-DD
    m = _ISO_RE.match(s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if _valid_ymd(y, mo, d):
            return f"{y:04d}-{mo:02d}-{d:02d}"
        return None
    # Slash form — default to US (MM/DD/YYYY). If the month value is > 12,
    # fall back to DD/MM/YYYY (European).
    m = _SLASH_RE.match(s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # try US first
        if 1 <= a <= 12 and _valid_ymd(y, a, b):
            return f"{y:04d}-{a:02d}-{b:02d}"
        if 1 <= b <= 12 and _valid_ymd(y, b, a):
            return f"{y:04d}-{b:02d}-{a:02d}"
        return None
    # Dash form like 01-02-2000 — sentinel "-" in the spec means treat as
    # ISO-style, which for a 3-part dash-separated date means DD-MM-YYYY
    # is ambiguous; we mirror slash logic but prefer DD-MM-YYYY since the
    # spec says "default to ISO interpretation if a sentinel `-` is in
    # the string" (i.e. don't assume US).
    m = _DASH_DMY_RE.match(s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # prefer DD-MM-YYYY when dashes are used
        if 1 <= b <= 12 and _valid_ymd(y, b, a):
            return f"{y:04d}-{b:02d}-{a:02d}"
        if 1 <= a <= 12 and _valid_ymd(y, a, b):
            return f"{y:04d}-{a:02d}-{b:02d}"
        return None
    return None


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------


class DefaultNormalizer:
    """Stdlib-only Normalizer. Implements the Normalizer Protocol."""

    def normalize(self, signals: EntitySignals) -> NormalizedSignals:
        name, name_tokens = _normalize_name(signals.name)
        email, email_local = _normalize_email(signals.email)
        phone_e164 = _normalize_phone(signals.phone)
        address, address_tokens = _normalize_address(signals.address)
        dob_iso = _normalize_dob(signals.dob)
        # Normalize alias emails, dedup against primary
        alias_full: list[str] = []
        alias_local: list[str] = []
        if email_local:
            alias_local.append(email_local)
        for raw in signals.email_aliases:
            a_full, a_local = _normalize_email(raw)
            if a_full and a_full != email and a_full not in alias_full:
                alias_full.append(a_full)
            if a_local and a_local not in alias_local:
                alias_local.append(a_local)
        return NormalizedSignals(
            name=name,
            name_tokens=name_tokens,
            email=email,
            email_local=email_local,
            email_aliases=tuple(alias_full),
            email_locals=tuple(alias_local),
            phone_e164=phone_e164,
            address=address,
            address_tokens=address_tokens,
            dob_iso=dob_iso,
        )
