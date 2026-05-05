"""Wikidata source adapter for the lite enrichment tier.

Two-step lookup:
  1. wbsearchentities (label → Q-id)
  2. wbgetentities (Q-id → claims for the requested property)

Designed to be defensive: any HTTP error, rate-limit, or missing data
returns []. Network calls have a 10s timeout.
"""

from __future__ import annotations

import re
from typing import Optional

import httpx
import structlog

from cograph_client.enrichment.models import Verdict

logger = structlog.stdlib.get_logger("cograph.enrichment")


WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_BASE = "https://www.wikidata.org/wiki/"
TIMEOUT_S = 10.0


_CLEAN_CACHE: dict[str, list[str]] = {}


def _is_sku_token(tok: str) -> bool:
    """A token is SKU/model-code-ish if it's:
    - all digits, or
    - alphanumeric mix containing at least one digit (e.g. M3, WH-1000XM5), or
    - contains a hyphen with digits (e.g. 33-2304).
    """
    if not tok:
        return False
    if tok.isdigit():
        return True
    has_digit = any(c.isdigit() for c in tok)
    if not has_digit:
        return False
    # Strip hyphens for the alphanumeric check; if there are digits anywhere
    # and it's not purely letters, treat as SKU-ish.
    cleaned = tok.replace("-", "")
    if cleaned.isalnum() and has_digit:
        return True
    if "-" in tok and has_digit:
        return True
    return False


def _clean_label_candidates(label: str) -> list[str]:
    """Generate progressively cleaner label candidates for Wikidata search.

    Layered approach (each step makes the label fuzzier):
      0. Original label.
      A. Strip trailing SKU-ish tokens until a wordy (mostly letters) token.
      B. If A is still > 3 words, take just the first 2 tokens.
      C. First token only (last resort, often resolves to brand alone).

    Returns a deduped, order-preserving list. Cached on `label`.
    """
    if label in _CLEAN_CACHE:
        return list(_CLEAN_CACHE[label])

    candidates: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        s = s.strip()
        if s and s not in seen:
            seen.add(s)
            candidates.append(s)

    _add(label)

    tokens = re.split(r"\s+", label.strip()) if label.strip() else []

    # Candidate A: strip trailing SKU-ish tokens until a wordy token.
    a_tokens = list(tokens)
    while a_tokens and _is_sku_token(a_tokens[-1]):
        a_tokens.pop()
    if a_tokens and a_tokens != tokens:
        _add(" ".join(a_tokens))

    # Candidate B: if A is still > 3 words (>= 3 tokens), first 2 tokens.
    # Spec example: "Apple MacBook Pro" (3 tokens) collapses to "Apple MacBook".
    base_for_b = a_tokens if a_tokens else tokens
    if len(base_for_b) >= 3:
        _add(" ".join(base_for_b[:2]))

    # Candidate C: first token only.
    if tokens:
        _add(tokens[0])

    _CLEAN_CACHE[label] = list(candidates)
    return candidates


WIKIDATA_PROPS: dict[str, str] = {
    "manufacturer": "P176",
    "brand": "P1716",
    "mpn": "P528",  # catalog code
    "country": "P17",
    "industry": "P452",
    "instance_of": "P31",
    "founder": "P112",
    "headquarters": "P159",
    "ceo": "P169",
    "isbn": "P212",
    "duration": "P2047",
    "release_date": "P577",
    "director": "P57",
    "genre": "P136",
}


class WikidataAdapter:
    name = "wikidata"

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        self._client = client

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            # Wikimedia's UA policy 403s the default python-httpx UA; identify
            # ourselves with project + contact so the request is honored.
            # See https://meta.wikimedia.org/wiki/User-Agent_policy
            self._client = httpx.AsyncClient(
                timeout=TIMEOUT_S,
                headers={
                    "User-Agent": (
                        "cograph-enrichment/0.1 "
                        "(+https://github.com/git-moeen/cograph-oss; ops@cograph.tech)"
                    ),
                    "Accept": "application/json",
                },
            )
        return self._client

    async def lookup(
        self, entity_label: str, attribute: str, context: dict
    ) -> list[Verdict]:
        prop = WIKIDATA_PROPS.get(attribute)
        if not prop:
            return []
        if not entity_label:
            return []

        try:
            qid, penalty = await self._search_entity_with_fallback(entity_label)
            if not qid:
                return []
            verdicts = await self._fetch_claims(qid, attribute, prop)
            if penalty > 0.0 and verdicts:
                verdicts = [
                    Verdict(
                        value=v.value,
                        confidence=max(0.0, round(v.confidence - penalty, 4)),
                        source=v.source,
                        source_url=v.source_url,
                    )
                    for v in verdicts
                ]
            return verdicts
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            logger.warning(
                "wikidata_lookup_failed",
                entity=entity_label,
                attribute=attribute,
                error=str(e),
            )
            return []
        except Exception as e:  # noqa: BLE001 — defensive boundary
            logger.warning(
                "wikidata_lookup_error",
                entity=entity_label,
                attribute=attribute,
                error=str(e),
            )
            return []

    async def _search_entity_with_fallback(
        self, label: str
    ) -> tuple[Optional[str], float]:
        """Layered label resolution.

        wbsearchentities is a strict prefix/term matcher and won't find labels
        with SKU/model-code suffixes (e.g. "Apple MacBook Pro M3"). We try the
        original first, then progressively cleaner candidates produced by
        `_clean_label_candidates`. Each fallback step costs 0.05 in confidence
        so direct hits beat fuzzy ones. Capped at 4 search calls total
        (original + 3 candidates) to avoid runaway.
        """
        candidates = _clean_label_candidates(label)[:4]
        for idx, cand in enumerate(candidates):
            qid = await self._search_entity(cand)
            if qid:
                return qid, idx * 0.05
        return None, 0.0

    async def _search_entity(self, label: str) -> Optional[str]:
        client = await self._get_client()
        resp = await client.get(
            WIKIDATA_API,
            params={
                "action": "wbsearchentities",
                "search": label,
                "language": "en",
                "format": "json",
                "limit": 1,
                "type": "item",
            },
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        results = data.get("search", [])
        if not results:
            return None
        return results[0].get("id")

    async def _fetch_claims(
        self, qid: str, attribute: str, prop: str
    ) -> list[Verdict]:
        client = await self._get_client()
        resp = await client.get(
            WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "ids": qid,
                "props": "claims",
                "format": "json",
            },
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        entities = data.get("entities", {}) or {}
        entity = entities.get(qid) or {}
        claims = (entity.get("claims") or {}).get(prop, [])
        if not claims:
            return []

        # Cache label resolution within a single call.
        label_cache: dict[str, str] = {}

        verdicts: list[Verdict] = []
        for idx, claim in enumerate(claims):
            value = await self._resolve_claim_value(claim, label_cache)
            if value is None:
                continue
            # First claim is treated as canonical (highest confidence).
            confidence = 0.95 if idx == 0 else 0.9
            verdicts.append(
                Verdict(
                    value=value,
                    confidence=confidence,
                    source="wikidata",
                    source_url=f"{WIKIDATA_ENTITY_BASE}{qid}",
                )
            )
        return verdicts

    async def _resolve_claim_value(
        self, claim: dict, label_cache: dict[str, str]
    ) -> Optional[str]:
        mainsnak = claim.get("mainsnak") or {}
        datavalue = mainsnak.get("datavalue") or {}
        dvtype = datavalue.get("type")
        value = datavalue.get("value")
        if value is None:
            return None

        if dvtype == "string":
            return str(value)
        if dvtype == "wikibase-entityid":
            target_qid = value.get("id")
            if not target_qid:
                return None
            if target_qid in label_cache:
                return label_cache[target_qid]
            label = await self._fetch_label(target_qid)
            if label:
                label_cache[target_qid] = label
                return label
            return target_qid
        if dvtype == "time":
            return value.get("time", "") if isinstance(value, dict) else str(value)
        if dvtype == "quantity":
            return str(value.get("amount", "")) if isinstance(value, dict) else str(value)
        if dvtype == "monolingualtext":
            return value.get("text", "") if isinstance(value, dict) else str(value)
        if dvtype == "globecoordinate":
            if isinstance(value, dict):
                lat = value.get("latitude")
                lon = value.get("longitude")
                if lat is not None and lon is not None:
                    return f"{lat},{lon}"
            return None
        return None

    async def _fetch_label(self, qid: str) -> Optional[str]:
        client = await self._get_client()
        try:
            resp = await client.get(
                WIKIDATA_API,
                params={
                    "action": "wbgetentities",
                    "ids": qid,
                    "props": "labels",
                    "languages": "en",
                    "format": "json",
                },
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            entity = (data.get("entities") or {}).get(qid) or {}
            labels = entity.get("labels") or {}
            en = labels.get("en") or {}
            return en.get("value")
        except (httpx.HTTPError, httpx.TimeoutException):
            return None

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
