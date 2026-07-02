"""Second-pass entity resolution — `er rebuild` (MOE-22).

During ingest, ER lookups within a single batch can't see each other's index
triples (the block-index writes land at the *end* of the batch), so same-entity
rows from one source file don't auto-collapse. Cross-batch merges work (CRM →
existing PMS, Loyalty → existing PMS); intra-batch fragments don't. Net effect
on the hotel demo: ~162 Person entities for ~45 real humans.

This module re-runs resolution over an **already-ingested** KG. Every entity's
signals are already indexed, so blocking now finds the fragments. We:

    1. load every entity of each ER-enabled type with its stored signals
    2. re-block and score all in-block pairs with the SAME scorer + config +
       auto_merge_threshold that ingest uses
    3. union-find the auto-merge pairs into clusters
    4. merge each cluster onto one canonical URI (all triples flow in)

Reusing ingest's exact merge criterion is the key safety property: the rebuild
can never merge more aggressively than ingest already would — it only catches
the fragments ingest *would* have merged had the index been visible mid-batch.
That structurally preserves the "zero false cross-bucket merges" guarantee the
ER test suite asserts.

Idempotent: a second run sees singleton clusters (nothing left to merge).

The cluster computation (`compute_clusters`, `choose_canonical`) is pure and
unit-tested without a graph; only `rebuild_type` / `rebuild_kg` touch Neptune,
and they fold each merge through the shared `kg_writer.rewrite_subject` primitive
(ADR 0007) so a merged-away URI never lingers in a derived secondary index.
"""

from __future__ import annotations

import structlog

from cograph_client.graph.kg_writer import refresh_after_write, rewrite_subject
from cograph_client.graph.queries import parse_kg_graph_uri
from cograph_client.resolver.er.blocking import SparqlBlocker, generate_block_keys
from cograph_client.resolver.er.scoring import DefaultScorer
from cograph_client.resolver.er.types import (
    ERConfig,
    NormalizedSignals,
    Scorer,
    config_for,
)

logger = structlog.stdlib.get_logger("cograph.resolver.er.rebuild")

TYPE_URI_PREFIX = "https://cograph.tech/types/"


# ---------------------------------------------------------------------------
# Union-Find
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, items: list[str]) -> None:
        self._parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        # path compression
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def clusters(self) -> list[list[str]]:
        groups: dict[str, list[str]] = {}
        for x in self._parent:
            groups.setdefault(self.find(x), []).append(x)
        return [sorted(g) for g in groups.values()]


# ---------------------------------------------------------------------------
# Pure cluster computation
# ---------------------------------------------------------------------------


def compute_clusters(
    entities: dict[str, NormalizedSignals],
    config: ERConfig,
    scorer: Scorer | None = None,
) -> list[list[str]]:
    """Return the clusters (size >= 2) of entity URIs that should be merged.

    Blocks every entity by the same strategies ingest uses, scores all distinct
    in-block pairs, and unions any pair scoring >= ``config.auto_merge_threshold``.
    Singleton clusters (nothing to merge) are omitted.
    """
    scorer = scorer or DefaultScorer()

    # Inverted block index: block-key string -> entity URIs sharing it.
    block_index: dict[str, set[str]] = {}
    for uri, signals in entities.items():
        for key in generate_block_keys(signals):
            block_index.setdefault(f"{key.kind}:{key.value}", set()).add(uri)

    uf = _UnionFind(list(entities.keys()))

    # Score each distinct pair that co-occurs in at least one block. A pair may
    # share several block keys; score it once.
    scored_pairs: set[tuple[str, str]] = set()
    for uris in block_index.values():
        if len(uris) < 2:
            continue
        members = sorted(uris)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair = (members[i], members[j])
                if pair in scored_pairs:
                    continue
                scored_pairs.add(pair)
                a, b = entities[pair[0]], entities[pair[1]]
                if scorer.score(a, b, config).score >= config.auto_merge_threshold:
                    uf.union(pair[0], pair[1])

    return [c for c in uf.clusters() if len(c) >= 2]


def _signal_richness(s: NormalizedSignals) -> int:
    return sum(bool(x) for x in (s.name, s.email, s.phone_e164, s.address, s.dob_iso)) + len(
        s.email_aliases
    )


def choose_canonical(cluster: list[str], entities: dict[str, NormalizedSignals]) -> str:
    """Pick the survivor URI for a cluster: the signal-richest entity, with the
    lexicographically-smallest URI as a deterministic tie-break."""
    return min(
        cluster,
        key=lambda u: (-_signal_richness(entities.get(u, NormalizedSignals())), u),
    )


# ---------------------------------------------------------------------------
# Graph-touching orchestration
# ---------------------------------------------------------------------------


async def rebuild_type(
    client,
    instance_graph: str,
    type_name: str,
    type_uri: str,
    config: ERConfig,
    scorer: Scorer | None = None,
) -> dict:
    """Re-resolve one ER-enabled type in one KG. Returns a small stat dict."""
    blocker = SparqlBlocker(client)
    entities = await blocker.all_entities_with_signals(instance_graph, type_uri)
    before = len(entities)
    if before < 2:
        return {
            "type": type_name,
            "entities_before": before,
            "entities_after": before,
            "clusters_merged": 0,
            "fragments_absorbed": 0,
        }

    clusters = compute_clusters(entities, config, scorer)

    # Collect (canonical, loser) pairs.
    merges: list[tuple[str, str]] = []
    for cluster in clusters:
        canonical = choose_canonical(cluster, entities)
        for uri in cluster:
            if uri != canonical:
                merges.append((canonical, uri))

    # Fold each loser into its canonical through the shared rewrite primitive
    # (ADR 0007): one batched URI rewrite per (loser -> canonical). A rewrite is a
    # single semantic event, so the derived indexes re-key (cheap) rather than
    # evict-and-recompute — done once, below, for the whole batch.
    for canonical, loser in merges:
        await rewrite_subject(
            client,
            instance_graph,
            loser,
            canonical,
            touched_types=[type_name],
            reason="er-merge rebuild",
        )

    # One post-write refresh per rebuild batch (NOT per entity): re-key the merged
    # subjects in the derived indexes and recompute type-stats. Scope comes from
    # the instance-graph URI; a non-KG graph (e.g. a test stub) yields no scope, so
    # the refresh is skipped — the rewrites themselves already landed in Neptune.
    if merges:
        scope = parse_kg_graph_uri(instance_graph)
        if scope is not None:
            r_tenant, r_kg = scope
            await refresh_after_write(
                client,
                tenant_id=r_tenant,
                kg_name=r_kg,
                affected_types=(),
                rewritten_subjects={loser: canonical for canonical, loser in merges},
            )

    after = before - len(merges)
    logger.info(
        "er_rebuild_type",
        type=type_name,
        entities_before=before,
        entities_after=after,
        clusters_merged=len(clusters),
        fragments_absorbed=len(merges),
    )
    return {
        "type": type_name,
        "entities_before": before,
        "entities_after": after,
        "clusters_merged": len(clusters),
        "fragments_absorbed": len(merges),
    }


async def _types_in_graph(client, instance_graph: str) -> list[str]:
    """Distinct rdf:type URIs present in the instance graph (cograph types only)."""
    sparql = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT DISTINCT ?t
FROM <{instance_graph}>
WHERE {{
  ?e rdf:type ?t .
  FILTER(STRSTARTS(STR(?t), "{TYPE_URI_PREFIX}"))
}}
"""
    data = await client.query(sparql)
    rows = data.get("results", {}).get("bindings", [])
    return [r["t"]["value"] for r in rows if r.get("t")]


async def rebuild_kg(client, instance_graph: str) -> dict:
    """Run the second-pass rebuild over every ER-enabled type present in a KG.

    `instance_graph` is the KG's instance graph URI (from kg_graph_uri).
    Returns a report with per-type before/after counts.
    """
    type_uris = await _types_in_graph(client, instance_graph)
    per_type: list[dict] = []
    for type_uri in type_uris:
        type_name = type_uri[len(TYPE_URI_PREFIX):].rstrip("/")
        config = config_for(type_name)
        if config is None:
            continue  # not ER-enabled — skip
        per_type.append(
            await rebuild_type(client, instance_graph, type_name, type_uri, config)
        )

    total_absorbed = sum(t["fragments_absorbed"] for t in per_type)
    logger.info("er_rebuild_kg_complete", types=len(per_type), fragments_absorbed=total_absorbed)
    return {"types": per_type, "fragments_absorbed_total": total_absorbed}
