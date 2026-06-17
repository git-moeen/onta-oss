"""In-memory verdict cache for enrichment lookups.

Cache key (ADR-0005 §2): a normalization of
`(entity_type, normalized_label, attribute, strategy_version, source)`.

Including `strategy_version` means a tier-chain/strategy change auto-invalidates
the cache (a new version yields different keys, hence a clean miss). The label is
normalized (lowercased, trimmed, internal whitespace collapsed) so cosmetic label
variants dedup to one entry — this is the dedup-cost win the ADR depends on.

This is the OSS keying/dedup pattern. A persistent (DynamoDB) backing table is
premium and out of scope here.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional

from cograph_client.enrichment.models import Verdict

_WHITESPACE_RE = re.compile(r"\s+")

# Backward-reasonable defaults so legacy 3-arg call sites still produce a stable
# key. New call sites should thread entity_type + strategy_version explicitly.
_DEFAULT_ENTITY_TYPE = ""
_DEFAULT_STRATEGY_VERSION = "v1"

CacheKey = tuple[str, str, str, str, str]


def _normalize_label(label: str) -> str:
    """Normalize an entity label for cache keying.

    Lowercase, strip leading/trailing whitespace, and collapse internal
    whitespace runs to a single space. So "City", "city", and "  City  " all
    map to "city".

    TODO(ADR-0005 §2): alias-folding. If/when an alias map is threaded through
    to the cache layer, fold known aliases here so e.g. "KN" and "K&N" share a
    key. Not done now to avoid reaching into other agents' files for aliases.
    """
    if not label:
        return ""
    return _WHITESPACE_RE.sub(" ", label.strip().lower())


def _key(
    entity_type: str,
    entity_label: str,
    attribute: str,
    strategy_version: str,
    source: str,
) -> CacheKey:
    return (
        (entity_type or "").lower(),
        _normalize_label(entity_label),
        attribute,
        str(strategy_version),
        source,
    )


class EnrichmentCache:
    def __init__(self) -> None:
        self._cache: dict[CacheKey, list[Verdict]] = {}
        self._lock = asyncio.Lock()

    async def get(
        self,
        entity_label: str,
        attribute: str,
        source: str,
        entity_type: str = _DEFAULT_ENTITY_TYPE,
        strategy_version: str = _DEFAULT_STRATEGY_VERSION,
    ) -> Optional[list[Verdict]]:
        async with self._lock:
            value = self._cache.get(
                _key(entity_type, entity_label, attribute, strategy_version, source)
            )
            return [v.model_copy(deep=True) for v in value] if value is not None else None

    async def put(
        self,
        entity_label: str,
        attribute: str,
        source: str,
        verdicts: list[Verdict],
        entity_type: str = _DEFAULT_ENTITY_TYPE,
        strategy_version: str = _DEFAULT_STRATEGY_VERSION,
    ) -> None:
        async with self._lock:
            self._cache[
                _key(entity_type, entity_label, attribute, strategy_version, source)
            ] = [v.model_copy(deep=True) for v in verdicts]


_cache: Optional[EnrichmentCache] = None


def get_enrichment_cache() -> EnrichmentCache:
    global _cache
    if _cache is None:
        _cache = EnrichmentCache()
    return _cache


def reset_enrichment_cache() -> None:
    global _cache
    _cache = None
