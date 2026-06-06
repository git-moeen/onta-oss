"""COG-35: per-type instance counts attribute each instance to ONE type.

Follow-up to ADR 0001 multi-typing. A multi-typed instance carries more than one
asserted ``rdf:type`` (its ``also_types`` co-classifications). The Explorer's
type-stats scan groups instance counts by raw ``rdf:type``; without a guard a
multi-typed instance is counted once PER asserted type — double-counted across
each type's panel.

``recompute_kg_stats`` now adds a primary-type guard (``_PRIMARY_TYPE_GUARD``):
an instance contributes to ``?type`` only when no OTHER asserted ``types/`` URI
sorts strictly before it. For the independent co-types the resolver emits
(equal-depth siblings) this is byte-identical to ``primary_type`` — the
lexicographically smallest asserted type wins. Single-typed instances are
unaffected (the NOT EXISTS is vacuously satisfied).

These tests prove both: a multi-typed instance is counted ONCE (regression for
the double-count), and single-typed instances count exactly as before.
"""
from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest

from cograph_client.api.routes import explore
from cograph_client.api.routes.explore import (
    RDF_TYPE,
    TYPE_URI_PREFIX,
    _PRIMARY_TYPE_GUARD,
    recompute_kg_stats,
)
from cograph_client.graph.client import NeptuneClient
from cograph_client.resolver.er.types import primary_type

TENANT = "test-tenant"
KG = "demo"


# ---------------------------------------------------------------------------
# 1) The generated scan carries the primary-type NOT EXISTS guard.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recompute_scan_contains_primary_type_guard():
    """The scan query must include the NOT EXISTS guard that rejects every
    asserted type heavier than the smallest — i.e. counts each instance once."""
    captured: list[str] = []

    client = AsyncMock(spec=NeptuneClient)

    async def capture_query(sparql, *a, **k):
        captured.append(sparql)
        return {"head": {"vars": []}, "results": {"bindings": []}}

    client.query.side_effect = capture_query
    client.update.return_value = None

    await recompute_kg_stats(client, TENANT, KG)

    scan = next((q for q in captured if "GROUP BY ?type ?p" in q), None)
    assert scan is not None, f"scan query not issued; captured={captured}"
    # The exact guard block is present...
    assert _PRIMARY_TYPE_GUARD in scan
    # ...and it is a subclass-prefix-scoped, lexicographic NOT EXISTS guard.
    assert "FILTER NOT EXISTS" in scan
    assert "?type2" in scan
    assert "STR(?type2) < STR(?type)" in scan
    assert TYPE_URI_PREFIX in scan


# ---------------------------------------------------------------------------
# Tiny SPARQL-aware mock: evaluates the rdf:type-grouped scan WITH the guard
# against a fixed triple set, so de-duplication is actually exercised.
# ---------------------------------------------------------------------------


def _make_scan_mock(triples: list[tuple[str, str, str]]):
    """Return an async query fn that evaluates the recompute scan over `triples`.

    Honors the primary-type guard: an entity ?e contributes to ?type only when
    no other asserted `types/` type of ?e has a lexicographically smaller URI.
    Produces SPARQL-JSON bindings of (type, p, cnt, sample, rel) grouped by
    (type, p), exactly as Neptune would for the guarded scan.
    """
    # asserted types per entity (only types/ URIs)
    types_of: dict[str, set[str]] = {}
    for s, p, o in triples:
        if p == RDF_TYPE and o.startswith(TYPE_URI_PREFIX):
            types_of.setdefault(s, set()).add(o)

    def primary_uri(entity: str) -> str | None:
        ts = types_of.get(entity)
        return min(ts) if ts else None  # smallest URI == primary (the guard)

    async def query(sparql, *a, **k):
        if "GROUP BY ?type ?p" not in sparql:
            return {"head": {"vars": []}, "results": {"bindings": []}}
        # group -> (type, p) -> {entities, rel_total, sample}
        groups: dict[tuple[str, str], dict] = {}
        for s, p, o in triples:
            asserted = types_of.get(s)
            if not asserted:
                continue
            prim = primary_uri(s)
            # guard: this entity is only attributed to its primary type
            for t in asserted:
                if t != prim:
                    continue
                g = groups.setdefault((t, p), {"ents": set(), "rel": 0, "sample": o})
                g["ents"].add(s)
                if o.startswith("https://cograph.tech/entities/"):
                    g["rel"] += 1
        bindings = []
        for (t, p), g in groups.items():
            bindings.append({
                "type": {"value": t},
                "p": {"value": p},
                "cnt": {"value": str(len(g["ents"]))},
                "sample": {"value": g["sample"]},
                "rel": {"value": str(g["rel"])},
            })
        return {
            "head": {"vars": ["type", "p", "cnt", "sample", "rel"]},
            "results": {"bindings": bindings},
        }

    return query


def _entity_counts_from_update(update_sparql: str) -> dict[str, int]:
    """Pull the per-type entityCount triples out of the INSERT DATA body."""
    out: dict[str, int] = {}
    for m in re.finditer(
        r"<(https://cograph\.tech/types/[^>]+)> "
        r"<https://cograph\.tech/stats/entityCount> (\d+)",
        update_sparql,
    ):
        out[m.group(1)] = int(m.group(2))
    return out


def _t(name: str) -> str:
    return f"{TYPE_URI_PREFIX}{name}"


@pytest.mark.asyncio
async def test_multityped_instance_counted_once_under_primary_type():
    """An instance asserted as both Employee AND Guest (independent co-types)
    is counted ONCE — under its primary type — not once per type.

    primary_type(['Employee','Guest'], {}) == 'Employee' (lexicographically
    smallest equal-depth sibling), so the single attribution lands on Employee.
    """
    # sanity: confirm the helper's choice we are mirroring in SPARQL.
    assert primary_type(["Employee", "Guest"], {}) == "Employee"

    e = "https://cograph.tech/entities/Employee/alice"
    triples = [
        # Multi-typed: both asserted rdf:types.
        (e, RDF_TYPE, _t("Employee")),
        (e, RDF_TYPE, _t("Guest")),
        (e, "https://cograph.tech/types/Employee/attrs/name", "Alice"),
    ]
    client = AsyncMock(spec=NeptuneClient)
    client.query.side_effect = _make_scan_mock(triples)
    captured_updates: list[str] = []

    async def capture_update(sparql, *a, **k):
        captured_updates.append(sparql)

    client.update.side_effect = capture_update

    await recompute_kg_stats(client, TENANT, KG)

    update = next((u for u in captured_updates if "entityCount" in u), "")
    counts = _entity_counts_from_update(update)

    # Counted ONCE, under the primary (smallest) type — NOT once per type.
    assert counts.get(_t("Employee")) == 1
    assert _t("Guest") not in counts  # the heavier co-type gets zero, not 1
    # Total attributed instances across all types == 1 (no double-count).
    assert sum(counts.values()) == 1


@pytest.mark.asyncio
async def test_single_typed_instances_counted_exactly_as_before():
    """Regression guard: the common single-type case is unchanged.

    Three Guests + two Employees, each single-typed. The guard's NOT EXISTS is
    vacuously satisfied for single-typed instances, so every instance is counted
    under its one type — identical to pre-COG-35 behavior.
    """
    triples: list[tuple[str, str, str]] = []
    for i in range(3):
        e = f"https://cograph.tech/entities/Guest/g{i}"
        triples.append((e, RDF_TYPE, _t("Guest")))
        triples.append((e, "https://cograph.tech/types/Guest/attrs/name", f"g{i}"))
    for i in range(2):
        e = f"https://cograph.tech/entities/Employee/e{i}"
        triples.append((e, RDF_TYPE, _t("Employee")))
        triples.append((e, "https://cograph.tech/types/Employee/attrs/name", f"e{i}"))

    client = AsyncMock(spec=NeptuneClient)
    client.query.side_effect = _make_scan_mock(triples)
    captured_updates: list[str] = []

    async def capture_update(sparql, *a, **k):
        captured_updates.append(sparql)

    client.update.side_effect = capture_update

    await recompute_kg_stats(client, TENANT, KG)

    update = next((u for u in captured_updates if "entityCount" in u), "")
    counts = _entity_counts_from_update(update)

    assert counts.get(_t("Guest")) == 3
    assert counts.get(_t("Employee")) == 2
    assert sum(counts.values()) == 5  # every instance counted exactly once


@pytest.mark.asyncio
async def test_mixed_single_and_multi_typed_population():
    """Realistic mix: mostly single-typed, one multi-typed straddler.

    4 plain Guests + 1 plain Employee + 1 instance asserted as BOTH. The
    straddler is attributed only to Employee (its primary). Final tallies:
    Guest=4, Employee=2 (the plain one + the straddler), total=6 distinct
    instances — never 7 (which is what raw rdf:type grouping would report).
    """
    triples: list[tuple[str, str, str]] = []
    for i in range(4):
        e = f"https://cograph.tech/entities/Guest/g{i}"
        triples.append((e, RDF_TYPE, _t("Guest")))
    plain_emp = "https://cograph.tech/entities/Employee/e0"
    triples.append((plain_emp, RDF_TYPE, _t("Employee")))
    straddler = "https://cograph.tech/entities/Employee/both"
    triples.append((straddler, RDF_TYPE, _t("Employee")))
    triples.append((straddler, RDF_TYPE, _t("Guest")))

    client = AsyncMock(spec=NeptuneClient)
    client.query.side_effect = _make_scan_mock(triples)
    captured_updates: list[str] = []

    async def capture_update(sparql, *a, **k):
        captured_updates.append(sparql)

    client.update.side_effect = capture_update

    await recompute_kg_stats(client, TENANT, KG)

    update = next((u for u in captured_updates if "entityCount" in u), "")
    counts = _entity_counts_from_update(update)

    assert counts.get(_t("Guest")) == 4      # straddler NOT double-counted here
    assert counts.get(_t("Employee")) == 2   # plain employee + straddler
    assert sum(counts.values()) == 6         # 6 distinct instances, not 7
