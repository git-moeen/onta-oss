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
  canonical companion-provenance graph (ADR 0002 §4) AND the spatio-temporal
  secondary index (another companion store derived from the same facts: every
  geometry-bearing entity is auto-indexed for geo/time queries, best-effort).
  Always batched so a large write can never blow past Neptune's per-statement
  size limit.
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

import asyncio
import os
import re
from typing import Iterable, Optional

import structlog

from cograph_client.graph.provenance import provenance_graph_uri
from cograph_client.graph.queries import (
    _escape_literal,
    batched_insert_triples,
    parse_kg_graph_uri,
    tenant_graph_uri,
)

logger = structlog.stdlib.get_logger("cograph.graph.kg_writer")

Triple = tuple[str, str, str]

# Hard cap on the synchronous spatio-temporal index upsert inside insert_facts.
# The index is a DERIVED, eventually-consistent companion store; Neptune (the
# source of truth) is already written by the time we reach it. Catching exceptions
# isn't enough — a hung/partitioned Postgres (pool exhaustion, Aurora failover)
# would otherwise block the KG-write request on this await with no exception. The
# timeout converts a hang into a caught TimeoutError → logged, index skipped, the
# write proceeds. Env-overridable for ops.
_INDEX_UPSERT_TIMEOUT_S = float(
    os.environ.get("COGRAPH_SPATIOTEMPORAL_UPSERT_TIMEOUT_S", "10")
)


def _semantic_upsert_timeout_s() -> float:
    """Timeout for the semantic-index write hook (ONTA-181) — the same hang-to-
    TimeoutError conversion as ``_INDEX_UPSERT_TIMEOUT_S``, with its own knob
    because the semantic hook does strictly more work per write (marker-map
    read + chunk upsert + empty-doc deletes). Read per call so tests/ops can
    tune it without re-importing the module."""
    return float(os.environ.get("COGRAPH_SEMANTIC_UPSERT_TIMEOUT_S", "10"))

# KG-registration triple shape — the `<kg_uri> <onto/kg_name> "name"` record in
# the tenant metadata graph that ``list_kgs`` reads to populate the Explorer
# dropdown. Kept here (not imported from the API route) so this module stays in
# the ``graph/`` layer with no dependency on ``api.routes``. Must match the
# shape ``api/routes/knowledge_graphs.py`` writes/reads (``OMNIX_ONTO/kg_name``
# and the ``_kg_meta_uri`` URI).
_OMNIX_ONTO = "https://cograph.tech/onto"
_KG_NAME_PRED = f"{_OMNIX_ONTO}/kg_name"

# A KG name that can legally be created via the Explorer ("New KG" button). Must
# match ``KGCreate.name``'s pattern in ``api/routes/knowledge_graphs.py`` — a
# name that can't be created via the UI must not be allowed to silently corrupt
# the registration URI (``<{kg_uri}>`` interpolates the raw name, so a `>` or
# whitespace would break the URI even when the literal is escaped).
_KG_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def _kg_meta_uri(tenant_id: str, kg_name: str) -> str:
    return f"https://cograph.tech/kgs/{tenant_id}/{kg_name}"


async def ensure_kg_registered(neptune, tenant_id: str, kg_name: str) -> None:
    """Idempotently register a KG in the tenant metadata graph.

    Writes the ``<kg_uri> <onto/kg_name> "name"`` record that ``list_kgs`` reads
    to populate the Explorer dropdown — but ONLY if the KG is not already
    registered. Historically this record was written in exactly one place
    (``create_kg``, the Explorer's "New KG" button), so any non-UI writer (agent
    web-discovery, CLI, MCP) that ingested into a brand-new ``kg_name`` left the
    KG invisible. Folding registration into the shared write path fixes that for
    every writer at once.

    Idempotent + non-clobbering by construction: a single
    ``INSERT … WHERE { FILTER NOT EXISTS { <kg_uri> <kg_name> ?n } }`` so it (a)
    never duplicates the registration triple and (b) never overwrites an existing
    registration or its ``kg_description`` (the whole INSERT is skipped when any
    ``kg_name`` already exists for this KG URI).

    Deliberately does NOT write ``kg_triple_count 0``: data has already been
    ingested by the time the shared write path registers, so a literal ``0`` would
    be stale-on-arrival and ``list_kgs`` only live-counts when the count is
    *absent*. Leaving it absent lets ``list_kgs`` lazily compute + persist the
    real count on first read.

    Safety: the literal is escaped via the canonical ``_escape_literal`` (no
    SPARQL-literal breakout on a name containing ``"`` / ``\\`` / newline), and the
    name is validated against the same ``^[a-zA-Z0-9_-]+$`` pattern the UI
    enforces before it's interpolated into the registration URI — a name that
    couldn't be created via the UI is skipped rather than allowed to corrupt the
    URI. Best-effort overall: a failure is logged, never raised, matching the rest
    of the post-write housekeeping.
    """
    if not kg_name:
        return
    if not _KG_NAME_RE.match(kg_name):
        # A name with URI-breaking characters (``>``, whitespace, …) can't be a
        # real KG (the UI rejects it), so don't risk corrupting the metadata
        # graph — log and skip rather than emit a malformed registration.
        logger.warning("ensure_kg_registered_invalid_name", kg_name=kg_name)
        return
    base = tenant_graph_uri(tenant_id)
    kg_uri = _kg_meta_uri(tenant_id, kg_name)
    sparql = (
        f"WITH <{base}>\n"
        f"INSERT {{\n"
        f'  <{kg_uri}> <{_KG_NAME_PRED}> "{_escape_literal(kg_name)}" .\n'
        f"}}\n"
        f"WHERE {{\n"
        f"  FILTER NOT EXISTS {{ <{kg_uri}> <{_KG_NAME_PRED}> ?n }}\n"
        f"}}"
    )
    try:
        await neptune.update(sparql)
    except Exception:  # noqa: BLE001 — never fail a write on a registration hiccup
        logger.warning("ensure_kg_registered_failed", kg_name=kg_name, exc_info=True)


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
    if instance_triples:
        await _index_spatiotemporal(instance_graph, instance_triples)
        await _index_semantic(neptune, instance_graph, instance_triples)


async def _index_spatiotemporal(
    instance_graph: str, instance_triples: list[Triple]
) -> None:
    """Populate the spatio-temporal secondary index from the just-written triples.

    A companion store derived from the same facts (like the provenance graph) —
    kept HERE in the single insertion primitive so EVERY converged writer (ingest,
    enrichment, normalization, dedupe, …) auto-indexes its geometry-bearing
    entities with no per-caller wiring. Datatype-driven: ``extract_spatiotemporal_facts``
    only emits a fact for an entity carrying a ``geo:wktLiteral``, so a write with
    no coordinates does ~no work and pays only a list scan.

    Best-effort and fully isolated: a derived-index hiccup must NEVER fail the
    primary KG write (Neptune is the source of truth; this index is eventually
    consistent). Skips non-KG graphs (the URI doesn't parse to a tenant/KG).
    """
    scope = parse_kg_graph_uri(instance_graph)
    if scope is None:
        return  # not a per-KG instance graph → nothing to scope an index row to
    tenant_id, kg_name = scope
    try:
        from cograph_client.spatiotemporal.extract import extract_spatiotemporal_facts
        from cograph_client.spatiotemporal.registry import get_spatiotemporal_index

        facts = extract_spatiotemporal_facts(
            instance_triples, tenant_id=tenant_id, kg_name=kg_name
        )
        if facts:
            # Time-bounded so a hung backend can't block the write (see the
            # _INDEX_UPSERT_TIMEOUT_S note); TimeoutError is caught below.
            await asyncio.wait_for(
                get_spatiotemporal_index().upsert_many(facts),
                timeout=_INDEX_UPSERT_TIMEOUT_S,
            )
    except Exception:  # noqa: BLE001 — never fail a KG write on a derived-index hiccup
        logger.warning(
            "spatiotemporal_index_update_failed",
            instance_graph=instance_graph,
            exc_info=True,
        )


async def _index_semantic(
    neptune, instance_graph: str, instance_triples: list[Triple]
) -> None:
    """Populate the semantic instance index from the just-written triples (ONTA-181).

    The FRESHNESS half of the ONTA-173 consistency model (the claim-based
    reconciler in ``semantic/reconciler.py`` is the correctness half): chunks
    of marked free-text attributes land in the same request that wrote Neptune,
    with ``embedding=NULL`` — the store-side generated tsvector makes them
    lexically searchable instantly; vector recall follows within one embed-fill
    sweep. Kept HERE in the single insertion primitive (like the
    spatio-temporal hook above) so EVERY converged writer auto-indexes with no
    per-caller wiring.

    Env-gated OFF by default (``COGRAPH_SEMANTIC_INDEX_ENABLED`` — cost/rollout
    control: indexing implies embedding spend and index growth). Marker-driven:
    only predicates the tenant's textKind map (``graph/text_markers.py``) marks
    ``free_text`` are extracted — free text has no distinguishing datatype, so
    unlike the spatio-temporal hook this one needs the ``neptune`` handle to
    consult the (TTL-cached) marker map.

    Empty-doc contract: a marked attr present in THIS WRITE whose canonicalized
    doc came out empty (or was deduped away because it mirrors another attr's
    doc) gets ``delete(entity, tenant, kg_name=…, attr=…)`` — per the ONTA-175
    upsert contract an empty doc has no chunk rows to carry its key, so the
    hook must issue the delete explicitly.

    Best-effort and time-bounded exactly like ``_index_spatiotemporal``: the
    whole body (marker read + upsert + deletes + schedule ensure) runs under
    one ``asyncio.wait_for`` so a hung index backend can't block the KG write,
    and ANY failure is logged, never raised — the KG write must NEVER fail on
    an index hiccup (Neptune is already the source of truth at this point).
    """
    scope = parse_kg_graph_uri(instance_graph)
    if scope is None:
        return  # not a per-KG instance graph → nothing to scope an index row to
    tenant_id, kg_name = scope
    try:
        from cograph_client.semantic.reconciler import semantic_index_enabled

        if not semantic_index_enabled():
            return
        await asyncio.wait_for(
            _index_semantic_inner(neptune, tenant_id, kg_name, instance_triples),
            timeout=_semantic_upsert_timeout_s(),
        )
    except Exception:  # noqa: BLE001 — never fail a KG write on a derived-index hiccup
        logger.warning(
            "semantic_index_update_failed",
            instance_graph=instance_graph,
            exc_info=True,
        )


async def _index_semantic_inner(
    neptune, tenant_id: str, kg_name: str, instance_triples: list[Triple]
) -> None:
    """The unguarded body of :func:`_index_semantic` (wrapped in one timeout).

    Imports are lazy so ``graph/`` stays importable without pulling in the
    semantic subsystem (mirrors the spatio-temporal hook's lazy imports).
    """
    from cograph_client.graph.text_markers import get_free_text_map
    from cograph_client.semantic.extract import extract_semantic_chunks
    from cograph_client.semantic.reconciler import (
        ensure_reconcile_schedule_from_hook,
        marked_doc_keys,
    )
    from cograph_client.semantic.registry import get_semantic_index

    marker_map = await get_free_text_map(neptune, tenant_id)
    marked = {uri for uri, is_free_text in marker_map.items() if is_free_text}
    if marked:
        index = get_semantic_index()
        chunks = extract_semantic_chunks(
            instance_triples,
            tenant_id=tenant_id,
            kg_name=kg_name,
            marked_predicates=marked,
        )
        if chunks:
            await index.upsert_chunks(chunks)
        # Empty-doc deletes: marked (entity, attr) docs touched by THIS write
        # that produced no chunks of their own (emptied or deduped — see the
        # docstring above). Only docs in this write are considered — full ghost
        # repair (deleted entities, marker flips) is the reconciler's job.
        emitted = {(c.entity_uri, c.attr) for c in chunks}
        emptied = marked_doc_keys(instance_triples, marked) - emitted
        for entity_uri, attr in sorted(emptied):
            await index.delete(entity_uri, tenant_id, kg_name=kg_name, attr=attr)
        logger.info(
            "semantic_index_hook",
            tenant_id=tenant_id,
            kg_name=kg_name,
            chunks_written=len(chunks),
            docs_deleted=len(emptied),
        )
    # Ensure the KG's recurring reconcile schedule exists (memoized — one
    # store round-trip per (tenant, kg) per process), even when nothing is
    # marked yet: the reconciler's default candidacy heuristic may mark
    # attributes this hook can't (client-mapped CSV rows, enrichment-minted).
    await ensure_reconcile_schedule_from_hook(tenant_id, kg_name)


async def refresh_after_write(
    neptune,
    *,
    tenant_id: str,
    kg_name: Optional[str],
    affected_types: Iterable[str] = (),
    recompute_stats: bool = True,
) -> None:
    """Post-write housekeeping shared by ingest + enrichment.

    Runs the refreshes a write can invalidate, in order:

    1. **Invalidate the NL-planning ontology cache** for the tenant graph, so a
       newly-declared type/attribute is visible to query planning immediately
       instead of after a TTL.
    2. **Re-embed the affected types** (``affected_types`` — types whose schema
       changed: new types, or types that gained an attribute) so semantic
       retrieval never serves a stale schema embedding. No-op when the embedding
       service is unconfigured.
    3. **Register the KG** in the tenant metadata graph (idempotently) when
       ``kg_name`` is given, so a non-UI writer (web-discovery, CLI, MCP) that
       ingested into a brand-new KG still shows up in the Explorer dropdown —
       not just KGs created via the "New KG" button (ONTA-153).
    4. **Schedule the Explorer type-stats recompute** for the KG (coverage %,
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

    # 1b. Free-text marker map (ONTA-177): a schema pass may have written
    #     `<attr> <onto/textKind> "free_text"` markers with this write, so drop
    #     the tenant's cached {predicate -> is_free_text} map — query-side
    #     consumers (semantic-index routing, ONTA-176) must see fresh verdicts
    #     immediately, not after the TTL (which remains the multi-task backstop).
    try:
        from cograph_client.graph.text_markers import invalidate as invalidate_text_markers

        invalidate_text_markers(tenant_id)
    except Exception:  # noqa: BLE001 — never fail a write on a cache hiccup
        logger.warning("text_marker_cache_invalidate_failed", exc_info=True)

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

    # 3. Register the KG in the tenant metadata graph (idempotent, best-effort)
    #    so non-UI writers don't leave it invisible to list_kgs (ONTA-153).
    if kg_name:
        await ensure_kg_registered(neptune, tenant_id, kg_name)

    # 4. Explorer type-stats recompute (background, best-effort).
    if recompute_stats and kg_name:
        try:
            from cograph_client.api.routes.explore import schedule_recompute

            schedule_recompute(neptune, tenant_id, kg_name)
        except Exception:  # noqa: BLE001
            logger.warning("schedule_recompute_failed", exc_info=True)
