"""Sensitivity tags + v1 enforcement (ADR 0002 §5).

Attributes carry schema-level sensitivity tags ('pii' | 'secret' | 'public'),
stored as the 'sensitivity' entry of a type's strategy bundle
(resolver/strategy.py): a dict of attr_name -> tag. An attribute absent from
the map defaults to 'public'; 'pii' and 'secret' both count as sensitive.

Inheritance differs from resolve_entry's nearest-ancestor-wins: sensitivity
maps MERGE down the subclass chain, child-overrides-parent, so a subtype can
re-tag a single attribute without redeclaring its ancestors' whole map
(resolve_sensitivity_map). At each type in the chain, registries are still
checked in precedence order (tenant > enhanced > public) and the first one
defining 'sensitivity' for that type wins for that type — the same shadowing
resolve_entry applies.

v1 enforces exactly three rules, each a pure reusable utility:

  1. guard_enrichment_payload — never send a sensitive value to an external
     enrichment service.
  2. filter_response_attrs   — never return a sensitive value without
     entitlement.
  3. redact_for_log          — replace sensitive values with '[REDACTED]'
     before logging.

Nothing in the OSS resolver currently logs attribute values with the type +
parent-map context these utilities need (resolve_attribute's
attr_type_mismatch warning is a pure function with no ontology context), so
the pipeline is not wired here — callers with that context (the API response
layer for rule 2, enrichment adapters/executor for rule 1, any structured log
of attribute records for rule 3) route through these utilities. With no
'sensitivity' entries registered anywhere, all three are exact no-ops.

Value-level detection (per-instance sensitivity) is explicitly deferred by
the ADR; the tag vocabulary leaves room for it without reworking this module.
"""

from __future__ import annotations

from typing import Any

from cograph_client.resolver.er.types import ancestor_chain
from cograph_client.resolver.strategy import StrategyRegistry

# Tag vocabulary (ADR 0002 §5).
PII = "pii"
SECRET = "secret"
PUBLIC = "public"
SENSITIVE_TAGS = frozenset({PII, SECRET})

# The strategy-bundle entry key sensitivity maps live under.
SENSITIVITY_ENTRY = "sensitivity"

REDACTED = "[REDACTED]"


def is_sensitive(tag: str) -> bool:
    """True for 'pii' and 'secret'; 'public' (and anything unknown) is not."""
    return tag in SENSITIVE_TAGS


def resolve_sensitivity_map(
    type_name: str,
    parent_of: dict[str, str],
    registries: list[StrategyRegistry],
) -> dict[str, str]:
    """Effective attr_name -> tag map for a type.

    Walks ancestor_chain(type_name, parent_of) root-first and merges each
    type's 'sensitivity' entry over its ancestors', so a child re-tagging an
    attribute overrides the parent while inheriting everything else. At each
    type, registries are checked in precedence order and the first bundle
    defining 'sensitivity' wins for that type (resolve_entry shadowing).

    Pure: no Neptune, no I/O. Cycle-guarded via ancestor_chain. No entries
    anywhere in the chain => empty map => everything defaults to 'public'.
    """
    merged: dict[str, str] = {}
    for ancestor in reversed(ancestor_chain(type_name, parent_of)):
        for registry in registries:
            bundle = registry.get(ancestor)
            if bundle is not None and SENSITIVITY_ENTRY in bundle:
                merged.update(bundle[SENSITIVITY_ENTRY])
                break
    return merged


def guard_enrichment_payload(
    payload: dict[str, Any],
    type_name: str,
    parent_of: dict[str, str],
    registries: list[StrategyRegistry],
) -> dict[str, Any]:
    """Rule 1: never send a sensitive attribute's value to external enrichment.

    Returns a NEW dict with pii/secret attributes removed entirely (key and
    value — an external service should not learn the attribute exists on this
    record). Untagged attributes default to 'public' and pass through, so an
    empty registry list is an exact no-op copy.
    """
    smap = resolve_sensitivity_map(type_name, parent_of, registries)
    return {k: v for k, v in payload.items() if not is_sensitive(smap.get(k, PUBLIC))}


def filter_response_attrs(
    attrs: dict[str, Any],
    type_name: str,
    parent_of: dict[str, str],
    registries: list[StrategyRegistry],
    entitled: bool,
) -> dict[str, Any]:
    """Rule 2: never return a sensitive attribute without entitlement.

    Entitled callers get an unchanged copy; non-entitled callers get the
    payload with pii/secret attributes removed. Always returns a new dict so
    callers can mutate the result without touching the source record.
    """
    if entitled:
        return dict(attrs)
    smap = resolve_sensitivity_map(type_name, parent_of, registries)
    return {k: v for k, v in attrs.items() if not is_sensitive(smap.get(k, PUBLIC))}


def redact_for_log(
    record: dict[str, Any],
    type_name: str,
    parent_of: dict[str, str],
    registries: list[StrategyRegistry],
) -> dict[str, Any]:
    """Rule 3: redact sensitive VALUES before logging.

    Unlike rules 1-2 the keys are kept — log records keep a stable shape and
    show that a redaction happened — but every pii/secret value is replaced
    with '[REDACTED]'. Returns a new dict; the source record is untouched.
    """
    smap = resolve_sensitivity_map(type_name, parent_of, registries)
    return {
        k: REDACTED if is_sensitive(smap.get(k, PUBLIC)) else v
        for k, v in record.items()
    }
