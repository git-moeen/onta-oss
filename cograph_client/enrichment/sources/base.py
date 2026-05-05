"""Source adapter protocol and registry."""

from __future__ import annotations

from typing import Optional, Protocol

from cograph_client.enrichment.models import Verdict


class SourceAdapter(Protocol):
    name: str

    async def lookup(
        self, entity_label: str, attribute: str, context: dict
    ) -> list[Verdict]: ...


_adapters: dict[str, SourceAdapter] = {}


def register_adapter(adapter: SourceAdapter) -> None:
    _adapters[adapter.name] = adapter


def get_adapter(name: str) -> Optional[SourceAdapter]:
    return _adapters.get(name)


def list_adapters() -> list[str]:
    return list(_adapters.keys())
