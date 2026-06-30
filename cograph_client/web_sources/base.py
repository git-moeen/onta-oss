"""Web-source provider protocol and registry.

A *web-source provider* turns a natural-language discovery query
("a list of models offered by OpenRouter") into a uniform table of records —
the same ``list[dict]`` shape CSV ingest produces — which the web-discovery
capability then commits through the standard ingest pipeline
(:meth:`cograph_client.resolver.schema_resolver.SchemaResolver.ingest_mapped_records`).

This is the discovery counterpart to the enrichment adapter protocol
(:mod:`cograph_client.enrichment.sources.base`). The split matters: enrichment
fills a missing ``(entity, attribute)`` cell on entities that ALREADY exist;
discovery CREATES a whole set of new entities from a query. Different I/O shape
(there is no ``entity_label`` when the rows don't exist yet), so it gets its own
protocol — but the same plugin pattern: OSS defines the seam, a downstream
(proprietary) deployment registers a paid provider at boot.

Providers self-describe their COST the same generic way adapters do (COG-123):
the planner reads ``is_paid`` / ``cost_per_call`` via :func:`provider_cost`
(defensive ``getattr`` with free defaults), so the OSS cost model never hardcodes
the name of any specific paid provider.

OSS ships with NO provider registered. The web-discovery capability degrades
gracefully when :func:`get_web_source` returns ``None`` (the same no-op pattern
the ``suggest-relationships`` action uses without a registered recommender).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


@dataclass
class DiscoverResult:
    """The output of a discovery run — a table of records, like CSV rows.

    ``rows`` are uniform string-keyed dicts (one record each), ready to feed the
    SAME schema-inference / ``apply_mapping`` path CSV ingest uses. ``provenance``
    maps each row to the source URL it was drawn from, so every committed entity
    can carry a per-record ``source_url`` citation (the discovery counterpart to
    enrichment's ``<attr>_source_url``). The map is keyed by the row's natural
    name, falling back to its index as a string — i.e. ``{r.get("name", str(i)):
    url}``, the convention all bundled adapters + the stub follow. The web-ingest
    capability resolves a row's URL by that same key (name, then positional
    index — see ``web_ingest_cap._row_source_url``), so an index-keyed provider
    resolves too; populate it however your source allows. ``sources`` is the
    distinct set of sources consulted (for the plan preview). ``is_partial`` is
    True when the provider truncated at ``max_rows``; ``estimated_total`` is the
    provider's best guess at the full result size (used only to label the
    plan-time cost estimate, never to drive writes).
    """

    rows: list[dict[str, str]] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    is_partial: bool = False
    estimated_total: Optional[int] = None


@runtime_checkable
class WebSourceProvider(Protocol):
    """Protocol for a web-discovery provider.

    REQUIRED: ``name`` and ``discover``.

    OPTIONAL: ``is_paid`` / ``cost_per_call`` — the OSS cost signal (COG-123),
    read generically via :func:`provider_cost`. A provider that declares neither
    is treated as FREE. A paid provider opts in by setting ``is_paid = True``
    and/or a positive ``cost_per_call`` (the per-discovery-call USD cost). Either
    signal alone marks the provider paid.

    OPTIONAL (URL-targeted extraction): two boolean attributes, both read
    DEFENSIVELY elsewhere via ``getattr(provider, ..., False)`` so a provider that
    declares neither stays a plain query-discovery provider:

    - ``supports_urls: bool`` — the provider can extract records from explicit
      URLs (passed to :meth:`discover` via the ``urls`` kwarg) instead of (or in
      addition to) web-searching for ``query``. :func:`get_web_source` with
      ``for_urls=True`` selects the first provider that sets this.
    - ``url_only: bool`` — the provider ONLY does URL-targeted extraction and is
      never used for plain query discovery. Such a provider is SKIPPED when
      :func:`get_web_source` picks a default query provider, so registering it
      alongside a query provider leaves the no-arg query default unaffected.
    """

    name: str
    # Optional cost signal — declared for typing/documentation; defaulted to free
    # in :func:`provider_cost`.
    is_paid: bool
    cost_per_call: float
    # Optional URL-extraction capability flags — declared for typing/docs; read
    # defensively (default False) wherever a provider is selected/invoked.
    supports_urls: bool
    url_only: bool

    async def discover(
        self,
        query: str,
        *,
        sample: bool,
        max_rows: int,
        hint_columns: Optional[list[str]],
        context: dict,
        urls: Optional[list[str]] = None,
    ) -> DiscoverResult:
        """Find records on the web matching ``query``.

        ``sample=True`` asks for a small, representative slice (a handful of rows)
        cheap enough to drive the plan-time preview + schema inference; the full
        pull (``sample=False``) must be drawn the SAME way so the previewed schema
        matches the committed one. ``max_rows`` caps the result; ``hint_columns``
        are optional desired fields the user named; ``context`` carries
        tenant/kg/type hints the provider may use.

        ``urls`` is an OPTIONAL list of explicit pages to extract records FROM. When
        non-empty, a URL-capable provider (``supports_urls=True``) EXTRACTS records
        from those pages instead of web-searching for ``query`` (``query`` may
        still carry "what to pull from these pages"). In URL mode the returned
        :class:`DiscoverResult` shape is UNCHANGED, but ``sources`` is the input
        URLs and ``provenance`` maps each row's natural key to the URL it came
        from. A query-only provider may ignore ``urls``.
        """
        ...


# Module-level registry — same shape as register_adapter / register_capability.
_providers: dict[str, WebSourceProvider] = {}


def register_web_source(provider: WebSourceProvider) -> None:
    """Register (or replace) a web-source provider by name. Idempotent."""
    _providers[provider.name] = provider


def get_web_source(
    name: Optional[str] = None, *, for_urls: bool = False
) -> Optional[WebSourceProvider]:
    """Return a provider by ``name``, or select one for the requested mode.

    With ``name`` given, returns that provider (or ``None``). Otherwise selection
    tolerates TWO registered providers — a plain query provider and a URL-only one:

    - ``for_urls=True`` → the first provider that declares ``supports_urls`` (the
      URL-targeted extractor), or ``None`` if none does.
    - query mode (default) → the sole provider that is NOT ``url_only``; if no
      such single provider exists it falls back to the lone registered provider
      (the backward-compatible single-provider convenience).

    The no-name conveniences keep the capability decoupled from provider names:
    OSS registers none (returns ``None`` → graceful degradation), a deployment
    registers exactly one query provider and it is selected automatically; adding
    a ``url_only`` extractor alongside it does not disturb the query default.
    """
    if name is not None:
        return _providers.get(name)
    candidates = list(_providers.values())
    if for_urls:
        for p in candidates:
            if getattr(p, "supports_urls", False):
                return p
        return None
    # Query mode: ignore url_only providers.
    q = [p for p in candidates if not getattr(p, "url_only", False)]
    if len(q) == 1:
        return q[0]
    if len(_providers) == 1:
        return next(iter(_providers.values()))
    return None


def list_web_sources() -> list[str]:
    return list(_providers.keys())


def reset_web_sources() -> None:
    """Clear the registry. For tests."""
    _providers.clear()


def provider_cost(provider: WebSourceProvider) -> tuple[bool, float]:
    """Read a provider's declared cost signal generically (COG-123).

    Returns ``(is_paid, cost_per_call)``. Defensive ``getattr`` with free
    defaults, so a provider that declares neither attribute is treated as free.
    Paid if it sets ``is_paid = True`` OR a positive ``cost_per_call``. Never
    raises on a malformed/non-numeric ``cost_per_call``; coerces to 0.0.
    """
    try:
        cost = float(getattr(provider, "cost_per_call", 0.0) or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    if cost < 0.0:
        cost = 0.0
    is_paid = bool(getattr(provider, "is_paid", False)) or cost > 0.0
    return is_paid, cost


__all__ = [
    "DiscoverResult",
    "WebSourceProvider",
    "get_web_source",
    "list_web_sources",
    "provider_cost",
    "register_web_source",
    "reset_web_sources",
]
