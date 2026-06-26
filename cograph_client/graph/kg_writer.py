"""The single insertion + post-write housekeeping path for the knowledge graph.

**Convergence rule (do not bypass).** Every process that writes instance data
into a KG — CSV/JSON ingestion AND enrichment (search results) — MUST go through
these two functions. They are the one place that decides *how* facts are written
and *what* must be refreshed afterwards, so the two paths can never drift. The
moment a writer hand-rolls its own ``insert_triples`` + housekeeping, the paths
diverge silently (the exact bug this module exists to prevent: enrichment used to
write un-batched and never re-embedded or invalidated the NL-planning cache, so a
freshly-enriched attribute served stale embeddings and stale query plans while an
ingested one did not).

Split into two composable steps because the two writers differ in *what facts
they produce* (ingest mints entities; enrichment fills attributes on existing
ones) but are identical in *how those facts get written and refreshed*:

- :func:`insert_facts` — batched instance-triple write, plus the optional
  canonical companion-provenance graph (ADR 0002 §4). Always batched so a large
  write can never blow past Neptune's per-statement size limit.
- :func:`refresh_after_write` — invalidate the NL-planning ontology cache,
  re-embed the affected types (so semantic retrieval never serves a stale schema
  embedding after a new attribute lands), and schedule the Explorer type-stats
  recompute. Every successful write calls this with the types it touched.

Layering note: this module sits in ``graph/`` and must stay importable without
pulling in ``nlp`` or the API routes, so the embedding-service / ontology-cache /
stats-recompute dependencies are imported lazily inside
:func:`refresh_after_write` (they live in higher layers). Housekeeping is
best-effort — embedding/stats failures are logged, never raised — matching the
non-blocking behavior the ingest routes already had.
"""

from __future__ import annotations

from typing import Iterable, Optional

import structlog

from cograph_client.graph.provenance import provenance_graph_uri
from cograph_client.graph.queries import batched_insert_triples, tenant_graph_uri

logger = structlog.stdlib.get_logger("cograph.graph.kg_writer")

Triple = tuple[str, str, str]


async def insert_facts(
    neptune,
    instance_graph: str,
    instance_triples: list[Triple],
    *,
    provenance_triples: Optional[list[Triple]] = None,
) -> None:
    """Write instance triples (and optional canonical provenance) to the KG.

    The ONE insertion primitive for both ingest and enrichment. Always batched
    (``batched_insert_triples``) so a large write is chunked into multiple
    ``INSERT DATA`` statements rather than one statement that can exceed
    Neptune's size limit.

    ``provenance_triples`` (already built via
    :func:`cograph_client.graph.provenance.build_provenance_triples`) are written
    to the data graph's companion provenance graph
    (``provenance_graph_uri(instance_graph)``), exactly as the ingest path does.
    Pass ``None``/empty to skip (callers that surface provenance as ordinary
    instance attributes — e.g. enrichment's ``*_source_url`` citations — include
    those in ``instance_triples`` and need no separate provenance graph write).
    """
    if instance_triples:
        for sparql in batched_insert_triples(instance_graph, instance_triples):
            await neptune.update(sparql)
    if provenance_triples:
        prov_graph = provenance_graph_uri(instance_graph)
        for sparql in batched_insert_triples(prov_graph, provenance_triples):
            await neptune.update(sparql)


async def refresh_after_write(
    neptune,
    *,
    tenant_id: str,
    kg_name: Optional[str],
    affected_types: Iterable[str] = (),
    recompute_stats: bool = True,
) -> None:
    """Post-write housekeeping shared by ingest + enrichment.

    Runs the three refreshes a write can invalidate, in order:

    1. **Invalidate the NL-planning ontology cache** for the tenant graph, so a
       newly-declared type/attribute is visible to query planning immediately
       instead of after a TTL.
    2. **Re-embed the affected types** (``affected_types`` — types whose schema
       changed: new types, or types that gained an attribute) so semantic
       retrieval never serves a stale schema embedding. No-op when the embedding
       service is unconfigured.
    3. **Schedule the Explorer type-stats recompute** for the KG (coverage %,
       counts) when ``kg_name`` is given and ``recompute_stats`` is set.

    Best-effort: embedding and stats failures are logged and swallowed (a write
    must not fail because a downstream refresh hiccuped), matching the ingest
    routes' existing non-blocking behavior. Imports are lazy to keep ``graph/``
    free of ``nlp`` / API-route import cycles.
    """
    onto_graph = tenant_graph_uri(tenant_id)

    # 1. NL-planning ontology cache.
    try:
        from cograph_client.nlp.pipeline import NLQueryPipeline

        NLQueryPipeline.invalidate_cache(onto_graph)
    except Exception:  # noqa: BLE001 — never fail a write on a cache hiccup
        logger.warning("ontology_cache_invalidate_failed", exc_info=True)

    # 2. Re-embed affected types (dedup, order-preserving).
    types = list(dict.fromkeys(t for t in affected_types if t))
    if types:
        try:
            from cograph_client.nlp.pipeline import get_embedding_service

            svc = get_embedding_service()
            if svc is not None:
                await svc.embed_types(onto_graph, types, neptune)
        except Exception:  # noqa: BLE001 — non-blocking, mirrors the ingest routes
            logger.warning("embed_types_failed", types=types, exc_info=True)

    # 3. Explorer type-stats recompute (background, best-effort).
    if recompute_stats and kg_name:
        try:
            from cograph_client.api.routes.explore import schedule_recompute

            schedule_recompute(neptune, tenant_id, kg_name)
        except Exception:  # noqa: BLE001
            logger.warning("schedule_recompute_failed", exc_info=True)
