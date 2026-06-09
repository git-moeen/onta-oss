"""Strategy-bundle resolver (ADR 0002 §3).

ONE chain-walking resolver for everything a type carries: ER/dedupe config,
verification strategy, enrichment strategy, search/indexing strategy,
sensitivity tags, functions. A type with no entry inherits the nearest
configured ancestor's, walked up rdfs:subClassOf — exactly what
config_for_with_hierarchy in resolver/er/types.py does for ER today (and
that function is now a thin wrapper over resolve_entry).

A StrategyRegistry maps type_name -> bundle dict keyed by WELL_KNOWN_ENTRIES.
resolve_entry takes an ORDERED list of registries (tenant first, then
enhanced, then public — precedence by position, mirroring LayerStack
shadowing in graph/layers.py): at each type in the subclass chain, registries
are checked in order BEFORE climbing to the parent, so an entry on the type
itself in ANY visible registry beats an entry on an ancestor.

register_strategy is the OSS plugin seam (same style as register_adapter /
register_external_verifier): premium code attaches its curated strategy
library to its own registry and passes that registry ahead of the default.
The curated library itself is NOT here — only the mechanism is OSS.
"""

from __future__ import annotations

from typing import Any

from cograph_client.resolver.er.types import DEFAULTS_BY_TYPE, ancestor_chain

# Well-known bundle entry keys. Not enforced — premium layers may carry
# additional keys — but everything in ADR 0002 §3 resolves through these.
WELL_KNOWN_ENTRIES = ("er", "verify", "enrich", "search", "sensitivity", "functions")

# type_name -> bundle dict (entry_key -> strategy value).
StrategyRegistry = dict[str, dict[str, Any]]

# The OSS default registry: today's per-type ER configs, migrated from
# DEFAULTS_BY_TYPE into 'er' bundle entries. Resolution through this registry
# is behavior-identical to the old DEFAULTS_BY_TYPE.get chain walk.
_DEFAULT_REGISTRY: StrategyRegistry = {
    type_name: {"er": cfg} for type_name, cfg in DEFAULTS_BY_TYPE.items()
}


def default_registry() -> StrategyRegistry:
    """The built-in OSS registry (ER defaults). Lowest precedence by convention."""
    return _DEFAULT_REGISTRY


def register_strategy(
    type_name: str,
    entry_key: str,
    value: Any,
    registry: StrategyRegistry | None = None,
) -> None:
    """Attach a strategy entry to a type's bundle.

    Plugin seam for premium / tenant code (register_adapter style): pass your
    own `registry` to keep layers separate, or omit it to extend the default
    registry in place. Overwrites an existing entry for the same key.
    """
    target = registry if registry is not None else _DEFAULT_REGISTRY
    target.setdefault(type_name, {})[entry_key] = value


def resolve_entry(
    type_name: str,
    entry_key: str,
    parent_of: dict[str, str],
    registries: list[StrategyRegistry],
) -> Any | None:
    """Resolve one bundle entry for a type, climbing the subclass chain.

    Walks ancestor_chain(type_name, parent_of); at each type, checks
    `registries` in order (precedence by position — tenant first, then
    enhanced, then public) and returns the first bundle defining `entry_key`.
    Type specificity outranks registry precedence: a lower registry's entry on
    the type itself beats a higher registry's entry on an ancestor.

    Returns None if no registry defines the entry anywhere in the chain —
    callers treat that as "no strategy, use the legacy path" (for 'er' that
    means ER is skipped, exactly as before).

    Pure: no Neptune, no I/O. Cycle-guarded via ancestor_chain.
    """
    for ancestor in ancestor_chain(type_name, parent_of):
        for registry in registries:
            bundle = registry.get(ancestor)
            if bundle is not None and entry_key in bundle:
                return bundle[entry_key]
    return None
