"""In-memory verdict cache for enrichment lookups."""

from __future__ import annotations

import asyncio
from typing import Optional

from cograph_client.enrichment.models import Verdict


def _key(entity_label: str, attribute: str, source: str) -> tuple[str, str, str]:
    return (entity_label.lower(), attribute, source)


class EnrichmentCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str], list[Verdict]] = {}
        self._lock = asyncio.Lock()

    async def get(
        self, entity_label: str, attribute: str, source: str
    ) -> Optional[list[Verdict]]:
        async with self._lock:
            value = self._cache.get(_key(entity_label, attribute, source))
            return [v.model_copy(deep=True) for v in value] if value is not None else None

    async def put(
        self,
        entity_label: str,
        attribute: str,
        source: str,
        verdicts: list[Verdict],
    ) -> None:
        async with self._lock:
            self._cache[_key(entity_label, attribute, source)] = [
                v.model_copy(deep=True) for v in verdicts
            ]


_cache: Optional[EnrichmentCache] = None


def get_enrichment_cache() -> EnrichmentCache:
    global _cache
    if _cache is None:
        _cache = EnrichmentCache()
    return _cache


def reset_enrichment_cache() -> None:
    global _cache
    _cache = None
