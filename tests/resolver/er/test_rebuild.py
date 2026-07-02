"""Tests for the second-pass `er rebuild` (MOE-22).

The cluster computation is pure (no graph), so most of this exercises
compute_clusters / choose_canonical directly. A couple of async tests drive
rebuild_type with a fake Neptune client to cover the orchestration + idempotency
end-to-end without a real store. Since ADR 0007, the merge SPARQL lives in
`kg_writer.rewrite_subject` (via `queries.rewrite_subject_update`), so the
two-direction move is asserted against that builder here.
"""

from __future__ import annotations

import pytest

from cograph_client.graph.queries import rewrite_subject_update
from cograph_client.resolver.er.rebuild import (
    choose_canonical,
    compute_clusters,
    rebuild_type,
)
from cograph_client.resolver.er.types import DEFAULT_GUEST_CONFIG, NormalizedSignals


def _ns(**kwargs) -> NormalizedSignals:
    return NormalizedSignals(**kwargs)


# Three fragments of ONE human: "John Smith", "Jon Smith", "J. Smith". They
# share an email (decisive) and/or a name-block. Distinct PMS rows that ingest
# minted as separate URIs because they couldn't see each other mid-batch.
JOHN_A = _ns(name="john smith", name_tokens=("john", "smith"),
             email="john.smith0@gmail.com", email_local="johnsmith0",
             phone_e164="+442258595506")
JOHN_B = _ns(name="jon smith", name_tokens=("jon", "smith"),
             email="john.smith0@gmail.com", email_local="johnsmith0",
             phone_e164="+442258595506")
JOHN_C = _ns(name="john smith", name_tokens=("john", "smith"),
             phone_e164="+442258595506")  # no email, but shares lastname+phone block

# A DIFFERENT human who also happens to be named John Smith — different email,
# different phone. Must NEVER merge with the cluster above.
OTHER_JOHN = _ns(name="john smith", name_tokens=("john", "smith"),
                 email="jsmith99@yahoo.com", email_local="jsmith99",
                 phone_e164="+15551234000")


def test_collapses_fragments_of_one_human() -> None:
    entities = {
        "uri:johnA": JOHN_A,
        "uri:johnB": JOHN_B,
        "uri:johnC": JOHN_C,
    }
    clusters = compute_clusters(entities, DEFAULT_GUEST_CONFIG)
    assert len(clusters) == 1
    assert set(clusters[0]) == {"uri:johnA", "uri:johnB", "uri:johnC"}


def test_zero_false_merge_across_distinct_humans() -> None:
    # Two unrelated John Smiths in the population alongside the real cluster.
    entities = {
        "uri:johnA": JOHN_A,
        "uri:johnB": JOHN_B,
        "uri:other": OTHER_JOHN,
    }
    clusters = compute_clusters(entities, DEFAULT_GUEST_CONFIG)
    # Exactly one cluster (the real John), and it must not include the other.
    assert len(clusters) == 1
    assert "uri:other" not in clusters[0]
    assert set(clusters[0]) == {"uri:johnA", "uri:johnB"}


def test_no_clusters_when_all_distinct() -> None:
    entities = {
        "uri:a": _ns(name="alice brown", name_tokens=("alice", "brown"),
                     email="alice@x.com", email_local="alice"),
        "uri:b": _ns(name="bob green", name_tokens=("bob", "green"),
                     email="bob@y.com", email_local="bob"),
        "uri:other": OTHER_JOHN,
    }
    assert compute_clusters(entities, DEFAULT_GUEST_CONFIG) == []


def test_singletons_are_omitted() -> None:
    entities = {"uri:only": JOHN_A}
    assert compute_clusters(entities, DEFAULT_GUEST_CONFIG) == []


def test_choose_canonical_prefers_richest_then_stable() -> None:
    entities = {
        "uri:rich": JOHN_A,   # name + email + phone
        "uri:poor": JOHN_C,   # name + phone only
    }
    # The signal-richer entity wins regardless of URI ordering.
    assert choose_canonical(["uri:poor", "uri:rich"], entities) == "uri:rich"
    # Deterministic tie-break: equal richness → smallest URI.
    tie = {"uri:zzz": JOHN_A, "uri:aaa": JOHN_A}
    assert choose_canonical(["uri:zzz", "uri:aaa"], tie) == "uri:aaa"


def test_rewrite_subject_update_moves_both_directions() -> None:
    op = rewrite_subject_update("graph:hotel", "uri:loser", "uri:canon")
    # Outgoing edges of the loser move to the canonical...
    assert "DELETE { <uri:loser> ?p ?o }" in op
    assert "INSERT { <uri:canon> ?p ?o }" in op
    # ...and incoming references to the loser repoint to the canonical.
    assert "DELETE { ?s ?p <uri:loser> }" in op
    assert "INSERT { ?s ?p <uri:canon> }" in op
    assert "WITH <graph:hotel>" in op


# ---------------------------------------------------------------------------
# Async orchestration with a fake store
# ---------------------------------------------------------------------------


class _FakeNeptune:
    """Minimal stand-in: serves a fixed signal population, records updates."""

    def __init__(self, signal_rows: list[dict]):
        self._rows = signal_rows
        self.updates: list[str] = []

    async def query(self, sparql: str) -> dict:
        return {"results": {"bindings": self._rows}}

    async def update(self, sparql: str) -> None:
        self.updates.append(sparql)


def _sig_rows(entity: str, **signals) -> list[dict]:
    rows = []
    for name, value in signals.items():
        rows.append({
            "entity": {"value": entity},
            "p": {"value": f"https://cograph.tech/er/erSignal_{name}"},
            "o": {"value": value},
        })
    return rows


@pytest.mark.asyncio
async def test_rebuild_type_merges_and_reports() -> None:
    rows = (
        _sig_rows("uri:johnA", name="john smith", email="john.smith0@gmail.com",
                  email_local="johnsmith0", phone_e164="+442258595506")
        + _sig_rows("uri:johnB", name="jon smith", email="john.smith0@gmail.com",
                    email_local="johnsmith0", phone_e164="+442258595506")
        + _sig_rows("uri:other", name="john smith", email="jsmith99@yahoo.com",
                    email_local="jsmith99", phone_e164="+15551234000")
    )
    client = _FakeNeptune(rows)
    report = await rebuild_type(
        client, "graph:hotel", "Person",
        "https://cograph.tech/types/Person", DEFAULT_GUEST_CONFIG,
    )
    assert report["entities_before"] == 3
    assert report["entities_after"] == 2          # johnA + johnB collapse; other stays
    assert report["clusters_merged"] == 1
    assert report["fragments_absorbed"] == 1
    assert len(client.updates) == 1               # one chunked update issued
    # The "other" John must not appear as a merge loser.
    assert "uri:other" not in client.updates[0]


@pytest.mark.asyncio
async def test_rebuild_type_idempotent_on_distinct_population() -> None:
    rows = (
        _sig_rows("uri:a", name="alice brown", email="alice@x.com", email_local="alice")
        + _sig_rows("uri:b", name="bob green", email="bob@y.com", email_local="bob")
    )
    client = _FakeNeptune(rows)
    report = await rebuild_type(
        client, "graph:hotel", "Person",
        "https://cograph.tech/types/Person", DEFAULT_GUEST_CONFIG,
    )
    assert report["fragments_absorbed"] == 0
    assert report["entities_after"] == 2
    assert client.updates == []                   # nothing to merge → no writes
