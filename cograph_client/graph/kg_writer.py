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

Removals join the same path (ADR 0007). A fact *leaving* the graph or a subject
being *renamed* carries the identical fan-out obligation as an insert, so:

- :func:`delete_facts` — the one removal primitive (batched whole-subject or
  triple deletes) + a provenance *tombstone*.
- :func:`rewrite_subject` — the one URI-rewrite primitive (ER merge) + a
  provenance *rewrite* event; expressed as a single re-key event, NOT
  delete-then-insert, so derived indexes re-key cheaply instead of recomputing.
- :func:`refresh_after_write` grows ``deleted_subjects`` / ``rewritten_subjects``
  kwargs (no sibling ``refresh_after_delete`` — that fork is the banned drift):
  the same housekeeping pass evicts deleted subjects and re-keys renamed ones
  from every derived secondary index.

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
from datetime import datetime, timezone
from typing import Iterable, Optional

import structlog

from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.provenance import (
    build_rewrite_triples,
    build_tombstone_triples,
    provenance_graph_uri,
)
from cograph_client.graph.queries import (
    _escape_literal,
    batched_delete_triples,
    batched_insert_triples,
    count_subject_predicates_query,
    count_subjects_query,
    delete_subject_predicates_query,
    delete_subjects_query,
    parse_kg_graph_uri,
    rewrite_subject_update,
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


def _provenance_enabled() -> bool:
    """Whether removal/rename primitives write companion-graph provenance events.

    Gated by the same ``COGRAPH_PROVENANCE_ENABLED`` env var the ingest path uses
    for assertion provenance (default OFF), so tombstone/rewrite events only land
    when governance/undo is switched on.
    """
    return os.environ.get("COGRAPH_PROVENANCE_ENABLED", "0") == "1"


def _chunk(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


async def _count_matching(neptune, count_sparql: str) -> int:
    """Best-effort ``SELECT (COUNT(*) AS ?n)`` → int (0 on any failure).

    Used by :func:`delete_facts` to return an accurate removed-triple count for
    the pattern-based (subject / predicate-scoped) removals, whose count can't be
    known up front the way a concrete-triple list's can. Best-effort because the
    count is informational — a hiccup here must never fail the delete."""
    try:
        _, rows = parse_sparql_results(await neptune.query(count_sparql))
        return int(rows[0].get("n", 0)) if rows else 0
    except Exception:  # noqa: BLE001 — the count is informational, never load-bearing
        return 0

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


async def delete_facts(
    neptune,
    instance_graph: str,
    *,
    subjects: Optional[list[str]] = None,
    triples: Optional[list[Triple]] = None,
    touched_types: Iterable[str] = (),
    reason: str = "",
) -> int:
    """Remove instance triples from the KG — the single removal primitive (ADR 0007).

    The mirror of :func:`insert_facts`: a fact *leaving* the graph must fan out to
    the same places an arriving one does (a provenance record here; derived-index
    eviction via :func:`refresh_after_write`), regardless of which writer produced
    the removal. Two removal shapes, both **batched with the same discipline as**
    ``insert_facts`` so a large removal never emits one oversized statement:

    * ``subjects`` — delete EVERY triple whose subject is one of these URIs
      (whole-entity removal: a normalization orphan sweep, an ER-merge loser, a
      metadata upsert's clear-before-write).
    * ``triples`` — delete specific ``(s, p, o)`` triples. An entry whose object
      is ``None`` is a **predicate-scoped** delete: every object of that
      ``(s, p)`` is removed — the "clear this attribute before writing the new
      value" case (an attribute update = delete-old + insert-new), which avoids a
      fragile literal round-trip on the old value.

    Writes a ``tombstone`` event to the companion provenance graph
    (:func:`cograph_client.graph.provenance.build_tombstone_triples`, gated by
    ``COGRAPH_PROVENANCE_ENABLED`` exactly like assertion provenance) so
    governance/undo can see removals, not just assertions. Returns the
    removed-triple count.

    Does NOT itself touch derived secondary indexes: call
    :func:`refresh_after_write` with ``deleted_subjects`` once per operation so a
    single housekeeping pass evicts them (batched refresh, not per-delete).
    """
    subjects = [s for s in (subjects or []) if s]
    all_triples = list(triples or [])
    concrete = [(s, p, o) for (s, p, o) in all_triples if o is not None and s and p]
    sp_pairs = [(s, p) for (s, p, o) in all_triples if o is None and s and p]

    removed = 0
    # 1. Concrete-triple deletes (batched DELETE DATA) — exact count.
    if concrete:
        for sparql in batched_delete_triples(instance_graph, concrete):
            await neptune.update(sparql)
        removed += len(concrete)
    # 2. Predicate-scoped deletes (every object of each (s, p)) — count per chunk.
    for chunk in _chunk(sp_pairs, 500):
        removed += await _count_matching(
            neptune, count_subject_predicates_query(instance_graph, chunk)
        )
        await neptune.update(delete_subject_predicates_query(instance_graph, chunk))
    # 3. Whole-subject deletes — count per chunk.
    for chunk in _chunk(subjects, 500):
        removed += await _count_matching(
            neptune, count_subjects_query(instance_graph, chunk)
        )
        await neptune.update(delete_subjects_query(instance_graph, chunk))

    # 4. Provenance tombstone (gated + best-effort — never fail a write on it).
    if (subjects or all_triples) and _provenance_enabled():
        try:
            prov = build_tombstone_triples(
                subjects=subjects,
                triples=all_triples,
                graph_uri=instance_graph,
                reason=reason,
                timestamp=datetime.now(timezone.utc),
                touched_types=touched_types,
            )
            if prov:
                prov_graph = provenance_graph_uri(instance_graph)
                for sparql in batched_insert_triples(prov_graph, prov):
                    await neptune.update(sparql)
        except Exception:  # noqa: BLE001 — provenance is governance metadata, not the write
            logger.warning(
                "delete_facts_provenance_failed",
                instance_graph=instance_graph,
                exc_info=True,
            )
    return removed


async def rewrite_subject(
    neptune,
    instance_graph: str,
    old_uri: str,
    new_uri: str,
    *,
    touched_types: Iterable[str] = (),
    reason: str = "",
) -> None:
    """Rename a subject in place — the single URI-rewrite primitive (ADR 0007).

    Moves every triple referencing ``old_uri`` (as subject AND as object) onto
    ``new_uri`` in one batched update (``rewrite_subject_update``). Deliberately
    **not** ``delete_facts`` + ``insert_facts``: a rename is ONE semantic event,
    so derived indexes re-key (cheap) instead of evict-and-recompute (an embedding
    recompute per merged entity is the full enrichment-embed cost for zero
    information gain). Records a ``rewrite`` provenance event (old → new), gated by
    ``COGRAPH_PROVENANCE_ENABLED``.

    Does NOT itself touch derived secondary indexes: call
    :func:`refresh_after_write` with ``rewritten_subjects={old: new}`` once per
    rebuild batch so a single housekeeping pass re-keys them.
    """
    if not old_uri or not new_uri or old_uri == new_uri:
        return
    await neptune.update(rewrite_subject_update(instance_graph, old_uri, new_uri))
    if _provenance_enabled():
        try:
            prov = build_rewrite_triples(
                old_uri,
                new_uri,
                graph_uri=instance_graph,
                reason=reason,
                timestamp=datetime.now(timezone.utc),
                touched_types=touched_types,
            )
            prov_graph = provenance_graph_uri(instance_graph)
            for sparql in batched_insert_triples(prov_graph, prov):
                await neptune.update(sparql)
        except Exception:  # noqa: BLE001 — provenance is governance metadata, not the write
            logger.warning(
                "rewrite_subject_provenance_failed",
                instance_graph=instance_graph,
                exc_info=True,
            )


async def refresh_after_write(
    neptune,
    *,
    tenant_id: str,
    kg_name: Optional[str],
    affected_types: Iterable[str] = (),
    recompute_stats: bool = True,
    deleted_subjects: Iterable[str] = (),
    rewritten_subjects: Optional[dict[str, str]] = None,
) -> None:
    """Post-write housekeeping shared by ingest + enrichment + removals (ADR 0007).

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
    5. **Evict / re-key derived secondary indexes** for removals and renames:
       ``deleted_subjects`` are dropped from the spatiotemporal index (and the
       upcoming semantic index); ``rewritten_subjects`` (old → new, from an ER
       merge) are re-keyed rather than evicted. Both default empty so every
       existing call site is untouched. This is the removal-side mirror of
       ``insert_facts``'s ``_index_spatiotemporal`` — a sibling
       ``refresh_after_delete`` is deliberately NOT added (a forked refresh is the
       exact drift this convergence bans; an attribute *update* is a delete +
       insert and must run one refresh, not two).

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

    # 5. Derived secondary-index maintenance for removals / renames.
    await _deindex_secondary(
        tenant_id, kg_name, list(deleted_subjects), rewritten_subjects or {}
    )


async def _deindex_secondary(
    tenant_id: str,
    kg_name: Optional[str],
    deleted_subjects: list[str],
    rewritten_subjects: dict[str, str],
) -> None:
    """Evict deleted subjects and re-key renamed ones from derived secondary indexes.

    The removal-side mirror of :func:`_index_spatiotemporal`: when a fact LEAVES
    the graph (delete) or a subject is RENAMED (ER merge), every derived index
    keyed by subject URI must drop the ghost row (delete) or move it to the new
    key (re-key), exactly as an insert upserts. Leaving this out is what let the
    spatiotemporal index accumulate ghost rows for merged-away / deleted subjects
    (ADR 0007). Best-effort + time-bounded, same isolation as the insert side —
    Neptune is the source of truth and this index is eventually consistent.

    SEMANTIC-INDEX SEAM (ONTA-173): the upcoming embeddings-keyed-by-node-URI
    index subscribes to the SAME three event kinds — insert (via
    ``_index_spatiotemporal``), delete (evict below), rewrite (re-key below). Add
    its ``delete`` / ``rekey`` calls right alongside the spatiotemporal ones here
    so it never inherits the ghost-row problem this function exists to prevent.
    """
    if not deleted_subjects and not rewritten_subjects:
        return
    try:
        from cograph_client.spatiotemporal.registry import get_spatiotemporal_index

        index = get_spatiotemporal_index()

        async def _work() -> None:
            for uri in deleted_subjects:
                await index.delete(uri, tenant_id, kg_name=kg_name)
            for old, new in rewritten_subjects.items():
                rekey = getattr(index, "rekey", None)
                if rekey is not None:
                    await rekey(old, new, tenant_id, kg_name=kg_name)
                else:
                    # A backend that predates re-key (an out-of-tree override):
                    # evict the stale key so a ghost row can't survive. Correctness
                    # (no ghost) over the re-key cost saving.
                    await index.delete(old, tenant_id, kg_name=kg_name)

        # Time-bounded so a hung backend can't block the write (see the
        # _INDEX_UPSERT_TIMEOUT_S note); TimeoutError is caught below.
        await asyncio.wait_for(_work(), timeout=_INDEX_UPSERT_TIMEOUT_S)
    except Exception:  # noqa: BLE001 — never fail a write on a derived-index hiccup
        logger.warning(
            "spatiotemporal_index_deindex_failed",
            tenant_id=tenant_id,
            kg_name=kg_name,
            exc_info=True,
        )
