"""Claim-based reconciler for the semantic instance index (ONTA-181).

The CORRECTNESS half of the ONTA-173 consistency model (see
``semantic/protocol.py`` for the full diagram). The write hook in
``graph.kg_writer._index_semantic`` gives FRESHNESS (chunks land in the same
request that wrote Neptune, ``embedding=NULL``); this module gives CORRECTNESS
via two recurring duties, both fired as ordinary :class:`~cograph_client.scheduling.models.Schedule`
rows through the existing :class:`~cograph_client.scheduling.runner.ScheduleRunner`:

1. **Embed-fill sweep** (:func:`run_embed_fill_sweep`, ~5 min cadence, ONE
   global schedule row): drains ``embedding IS NULL`` rows (the durable queue —
   no outbox table) via the ONE shared embed client (``nlp.embed_client``,
   ONTA-174) and stamps ``embed_model``. Failures increment ``attempt_count``
   (``mark_embed_failed``); rows past the attempt cutoff are dead-lettered by
   ``fetch_pending(max_attempts=…)`` so a poison row can never wedge the sweep.
2. **Neptune-scan reconcile** (:func:`reconcile_kg`, hourly + on-demand, one
   schedule row PER KG): re-reads every marked free-text attribute from the KG
   instance graph, re-extracts chunks, upserts by ``content_hash`` (unchanged
   docs keep their filled embeddings), DELETES ghosts (index docs whose
   (entity, attr) no longer exists in Neptune — ER merges
   (``resolver/er/rebuild.py``) and normalization deletes bypass the write
   hook), and applies candidacy flips. **The first run against a KG IS the
   backfill**: an already-ingested KG (the parliamentary-speeches scenario)
   gets indexed without re-ingest, because "reconcile an empty index" and
   "backfill" are the same computation.

Why Schedule rows + the existing runner (design record)
-------------------------------------------------------

Overlapping ECS tasks during a rolling deploy must not double-run a sweep. The
scheduling subsystem already solved exactly this: ``ScheduleRunner`` claims due
rows with ``SELECT … FOR UPDATE SKIP LOCKED`` in one transaction
(``_claim_due_postgres``) and advances ``next_run`` before dispatching, so two
replicas claim disjoint batches. Re-implementing a bare asyncio loop here would
fork that guarantee — so semantic maintenance is expressed as Schedule rows
(actions ``semantic-embed-fill`` / ``semantic-reconcile``) dispatched by the
SAME runner. Topology:

* **one global embed-fill row** (``id="semantic-embed-fill"``, sentinel tenant
  ``_system``) — the sweep spans all tenants via the protocol's
  maintenance-only ``fetch_pending(tenant_id=None)`` exception, so per-tenant
  rows would only multiply identical work;
* **one reconcile row per KG** (``id="semantic-reconcile:{tenant}:{kg}"``) —
  reconcile scans ONE instance graph, and per-KG rows double as the KG
  discovery mechanism (OSS has no tenant directory a worker could enumerate):
  the write hook ensures a KG's row on first write, and the on-demand reindex
  route (``POST /graphs/{t}/kgs/{k}/search/reindex``) ensures it for
  already-ingested KGs and pulls ``next_run`` forward to fire immediately.

No job rows are created for these dispatches — a 5-minute sweep would spam the
unified Jobs feed; observability is structlog counters (``chunks_written``,
``skipped_unchanged_hash``, ``attrs_repaired``, ``embeds_pending``,
``embeds_filled``, ``embed_failures``, ``ghosts_deleted``), logged every run,
never silently zero.

Ghost enumeration (the ``list_docs`` Protocol method)
-----------------------------------------------------

Diffing "docs in the index" against "docs in Neptune" uses
:meth:`~cograph_client.semantic.protocol.SemanticIndex.list_docs` —
``(entity_uri, attr, content_hash, attrs)``, one row per (entity, attr)
document. The snapshot is taken BEFORE the Neptune scan starts (docs the
write hook indexes mid-scan must never be misread as ghosts), the hash drives
the unchanged-skip, the attrs drive the attrs-repair pass (type/label drift
with unchanged text), and ghost deletion is batched through ``delete_docs``
— and is additionally SKIPPED whenever the scan was truncated at the page
cap (a partial expected set must never drive deletions). Both first-party
backends implement the seam (InMemory sorts its dict; the pgvector adapter
runs one ``DISTINCT ON`` per KG), so every default deployment ghost-repairs.
The method started life as an OPTIONAL duck-typed seam (the Protocol was
frozen while ONTA-181 was built), and the reconciler still looks it up with
``getattr``: a third-party backend registered via ``register_semantic_index``
that predates the Protocol method must DEGRADE, not crash. In that case ghost
deletion and the unchanged-hash skip are SKIPPED — loudly logged with
``doc_listing_supported=False`` — and everything else still converges
(upserts are hash-idempotent; shrunk docs still lose their stale tail via the
upsert contract). A legacy 3-tuple listing degrades one notch: hash-diffing
still works, only the attrs-repair pass is skipped.

Candidacy default heuristic (ONTA-177 hand-off)
-----------------------------------------------

Attributes with NO ``textKind`` verdict (client-mapped ``/ingest/csv/rows``,
enrichment-minted attrs the schema pass never saw) are classified
reconciler-side from SAMPLED VALUES via the shared, name-blind
:func:`~cograph_client.graph.text_markers.classify_text_candidacy`:
``FREE_TEXT`` → durable ``"free_text"`` marker; ``NOT_CANDIDATE`` → durable
``"not_text"`` (decided-no — absence stays "undecided"); ``AMBIGUOUS`` → left
undecided (the LLM REASON layer, the only layer allowed to read attribute
NAMES, is not available in a background worker). Verdicts are written via the
canonical ``upsert_attribute_text_kind`` so they are durable and visible to
every other consumer.

Env knobs (all raw environment variables, the ``COGRAPH_*`` convention used by
``kg_writer`` / ``text_markers`` / the schedule runner; read per call so tests
and ops can tune without re-import):

* ``COGRAPH_SEMANTIC_INDEX_ENABLED`` — master gate for the write hook AND this
  reconciler (default **false**: cost/rollout control — embedding spend and
  index growth are opt-in).
* ``COGRAPH_SEMANTIC_EMBED_FILL_INTERVAL_S`` — embed-fill cadence (default 300).
* ``COGRAPH_SEMANTIC_RECONCILE_INTERVAL_S`` — per-KG reconcile cadence
  (default 3600).
* ``COGRAPH_SEMANTIC_EMBED_MAX_ATTEMPTS`` — dead-letter cutoff for embed
  failures (default 5).
* ``COGRAPH_SEMANTIC_SCAN_PAGE_SIZE`` — Neptune scan page size (default 10000).
* ``COGRAPH_SEMANTIC_ENSURE_MEMO_TTL_S`` — TTL of the write hook's
  ensure-schedule memo (default 600; see
  :func:`ensure_reconcile_schedule_from_hook`).
* ``COGRAPH_SEMANTIC_UPSERT_TIMEOUT_S`` — the WRITE HOOK's timeout (read in
  ``graph/kg_writer.py``, listed here for one-stop docs).

Vendor-neutral by construction (OSS boundary): no cloud identifiers, ARNs, or
hostnames; the only configuration is the generic DSN already carried by the
scheduling store and the OpenRouter key already carried by ``settings``.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

import structlog

from cograph_client.semantic.extract import (
    _is_uri_object,
    _local_name,
    extract_semantic_chunks,
)
from cograph_client.semantic.protocol import SemanticChunk, SemanticIndex
from cograph_client.semantic.registry import get_semantic_index

logger = structlog.stdlib.get_logger("cograph.semantic.reconciler")

Triple = tuple[str, str, str]

# --- schedule vocabulary ------------------------------------------------------

#: Schedule actions dispatched to this module (also members of
#: ``scheduling.models.ScheduleAction``).
SEMANTIC_EMBED_FILL_ACTION = "semantic-embed-fill"
SEMANTIC_RECONCILE_ACTION = "semantic-reconcile"

#: Deterministic id of the single global embed-fill schedule row. Deterministic
#: ids make the ensure-* helpers idempotent across processes: N racing creates
#: converge on one row (the Postgres store's ``create`` is an UPSERT by id).
EMBED_FILL_SCHEDULE_ID = "semantic-embed-fill"

#: Sentinel tenant for the global sweep row. Never a real tenant (real tenant
#: slugs are user-facing); the tenant-scoped schedule CRUD routes can't list or
#: touch it, and the sweep itself spans tenants via ``fetch_pending``'s
#: maintenance-only ``tenant_id=None`` exception.
_SYSTEM_TENANT = "_system"
_GLOBAL_KG = "*"

#: ``https://cograph.tech/types/{Type}/attrs/{name}`` — the only predicate
#: shape the candidacy heuristic may classify (system predicates like
#: ``rdfs:label`` / ``onto/ingested_at`` never carry a textKind verdict).
_ATTR_URI_RE = re.compile(
    r"^https://cograph\.tech/types/(?P<type>[^/]+)/attrs/(?P<attr>[^/]+)$"
)

#: Durable decided-no verdict written by the default heuristic. Anything other
#: than ``free_text`` reads back as ``is_free_text=False`` in
#: ``get_free_text_map`` — the point is that the attribute is DECIDED (absence
#: would mean "undecided" and get re-sampled every reconcile).
TEXT_KIND_NOT_TEXT = "not_text"

_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
#: Local names treated as display-label sources by the extractor; their
#: predicates are included in the reconcile scan so reconciler-written rows
#: carry the same denormalized ``attrs`` a hook-written row would.
_LABEL_LOCALS = {"label", "name", "title"}

#: Hard bounds on one reconcile run — a runaway KG must exhaust the page cap,
#: not the worker's memory. Truncation is logged, never silent.
_MAX_SCAN_PAGES = 200
#: Cap on attributes the default heuristic samples per run (each is one
#: bounded SPARQL sample query; the rest wait for the next hourly run).
_MAX_CANDIDACY_ATTRS_PER_RUN = 50
_CANDIDACY_SAMPLE_SIZE = 100
#: Chunk-upsert batch budget. Batches are packed on DOC boundaries — the
#: protocol's complete-document contract forbids splitting one (entity, attr)
#: doc across calls — so a batch may exceed this by one doc's tail.
_UPSERT_BATCH_CHUNKS = 500
#: Defensive bound on sweep iterations (each drains up to ``limit`` rows).
_MAX_SWEEP_ITERATIONS = 1000


# --- env knobs (read per call — tests/ops tune without re-import) -------------


def semantic_index_enabled() -> bool:
    """Master gate for the semantic index write path + reconciler (ONTA-181).

    Default **false**: indexing costs embedding spend and index growth, so a
    deployment opts in explicitly. Gates the ``kg_writer`` hook, both
    reconciler duties, the schedule seeding, and the reindex route.
    """
    raw = os.environ.get("COGRAPH_SEMANTIC_INDEX_ENABLED", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        val = int(float(raw))
    except ValueError:
        return default
    return val if val >= minimum else default


def embed_fill_interval_s() -> int:
    return _int_env("COGRAPH_SEMANTIC_EMBED_FILL_INTERVAL_S", 300)


def reconcile_interval_s() -> int:
    return _int_env("COGRAPH_SEMANTIC_RECONCILE_INTERVAL_S", 3600)


def embed_max_attempts() -> int:
    return _int_env("COGRAPH_SEMANTIC_EMBED_MAX_ATTEMPTS", 5)


def _scan_page_size() -> int:
    return _int_env("COGRAPH_SEMANTIC_SCAN_PAGE_SIZE", 10000)


def _ensure_memo_ttl_s() -> int:
    return _int_env("COGRAPH_SEMANTIC_ENSURE_MEMO_TTL_S", 600)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _now_monotonic() -> float:
    """Seam for the memo clock (monkeypatched in tests — no sleeps)."""
    return time.monotonic()


# --- schedule-row management ---------------------------------------------------


def reconcile_schedule_id(tenant_id: str, kg_name: str) -> str:
    """Deterministic per-KG reconcile schedule id (idempotent ensure/remove)."""
    return f"semantic-reconcile:{tenant_id}:{kg_name}"


async def ensure_embed_fill_schedule(store: Any):
    """Idempotently ensure the single global embed-fill schedule row.

    Called from app startup when the feature is enabled. ``get``-then-``create``
    (not blind create) so a restart never resets a live row's ``next_run``;
    a changed cadence knob is applied in place. Races between replicas converge
    because the id is deterministic and the durable store upserts by id.
    """
    from cograph_client.enrichment.models import JobCategory
    from cograph_client.scheduling.models import Schedule

    interval = embed_fill_interval_s()
    existing = await store.get(EMBED_FILL_SCHEDULE_ID)
    if existing is not None:
        if existing.interval_seconds != interval:
            existing.interval_seconds = interval
            existing.cron = None
            await store.update(existing)
            logger.info(
                "semantic_embed_fill_schedule_retuned", interval_seconds=interval
            )
        return existing
    now = _now()
    schedule = Schedule(
        id=EMBED_FILL_SCHEDULE_ID,
        tenant_id=_SYSTEM_TENANT,
        kg_name=_GLOBAL_KG,
        category=JobCategory.reconciliation,
        action=SEMANTIC_EMBED_FILL_ACTION,
        interval_seconds=interval,
        enabled=True,
        next_run=now,  # first sweep fires on the next runner poll
        created_at=now,
    )
    await store.create(schedule)
    logger.info("semantic_embed_fill_schedule_created", interval_seconds=interval)
    return schedule


async def ensure_reconcile_schedule(
    store: Any, tenant_id: str, kg_name: str, *, due_now: bool = False
):
    """Idempotently ensure the per-KG reconcile schedule row.

    A fresh row is seeded ``next_run=now`` — the FIRST reconcile of a KG is the
    backfill, so it should fire on the next runner poll, then settle onto the
    hourly cadence. ``due_now=True`` (the on-demand reindex route) pulls an
    EXISTING row's ``next_run`` forward to now instead of waiting out the hour.
    """
    from cograph_client.enrichment.models import JobCategory
    from cograph_client.scheduling.models import Schedule

    sid = reconcile_schedule_id(tenant_id, kg_name)
    now = _now()
    existing = await store.get(sid)
    if existing is not None:
        if due_now and (existing.next_run is None or existing.next_run > now):
            existing.next_run = now
            existing.enabled = True
            await store.update(existing)
            logger.info(
                "semantic_reconcile_pulled_forward",
                tenant_id=tenant_id,
                kg_name=kg_name,
            )
        return existing
    schedule = Schedule(
        id=sid,
        tenant_id=tenant_id,
        kg_name=kg_name,
        category=JobCategory.reconciliation,
        action=SEMANTIC_RECONCILE_ACTION,
        interval_seconds=reconcile_interval_s(),
        enabled=True,
        next_run=now,  # first run = the backfill
        created_at=now,
    )
    await store.create(schedule)
    logger.info(
        "semantic_reconcile_schedule_created", tenant_id=tenant_id, kg_name=kg_name
    )
    return schedule


async def remove_reconcile_schedule(store: Any, tenant_id: str, kg_name: str) -> None:
    """Drop the per-KG reconcile row (the KG-delete path) + the hook's memo, so
    a same-named KG recreated later in this process re-ensures a fresh row."""
    await store.delete(reconcile_schedule_id(tenant_id, kg_name))
    _ensured_reconcile.pop((tenant_id, kg_name), None)


# The write hook's ensure path. A module-level store (same selection logic as
# the runner's: Postgres when a DSN is configured — same table, so rows are
# shared — else the process-wide in-memory singleton) plus a TTL memo so the
# hook pays the ensure round-trip once per (tenant, kg) per TTL window
# (COGRAPH_SEMANTIC_ENSURE_MEMO_TTL_S, default 600s), not once per write.
#
# Why a TTL and not a process-lifetime memo: the schedules CRUD routes can
# DELETE the auto-created reconcile row without this module ever hearing about
# it (only the KG-delete path calls remove_reconcile_schedule). A permanent
# memo would then "poison" this process — it would never re-ensure the row and
# the KG would silently stop reconciling until a restart. With the TTL, a
# CRUD-deleted row is re-created within one TTL window by the next write.
_hook_store: Any = None
#: ``(tenant, kg) -> monotonic deadline``; entries past their deadline are
#: re-ensured on the next hook write.
_ensured_reconcile: dict[tuple[str, str], float] = {}


async def ensure_reconcile_schedule_from_hook(tenant_id: str, kg_name: str) -> None:
    """Best-effort, TTL-memoized ensure used by ``kg_writer._index_semantic``.

    This is how a KG that receives writes while the feature is enabled gets its
    recurring reconcile row without any operator action (already-ingested,
    write-quiet KGs use the reindex route instead). Exceptions propagate to the
    hook's catch-all (the memo is only set on success, so the next write
    retries).

    **Deleting the auto-created schedule row is NOT a durable opt-out.** As
    long as the feature gate (``COGRAPH_SEMANTIC_INDEX_ENABLED``) is on and the
    KG keeps receiving writes, this hook resurrects a deleted
    ``semantic-reconcile:{tenant}:{kg}`` row within one memo TTL
    (``COGRAPH_SEMANTIC_ENSURE_MEMO_TTL_S``, default 600s) — by design, so a
    stray CRUD delete can't silently disable correctness maintenance for a
    live KG. The durable off-switch is the env gate itself (flip it off and
    stale rows become logged no-ops — see :func:`dispatch_semantic_schedule`).
    """
    key = (tenant_id, kg_name)
    deadline = _ensured_reconcile.get(key)
    if deadline is not None and _now_monotonic() < deadline:
        return
    global _hook_store
    if _hook_store is None:
        from cograph_client.scheduling.store import make_schedule_store

        _hook_store = make_schedule_store()
    await ensure_reconcile_schedule(_hook_store, tenant_id, kg_name)
    _ensured_reconcile[key] = _now_monotonic() + _ensure_memo_ttl_s()


def reset_for_tests() -> None:
    """Test hook: restore pristine module state between tests."""
    global _hook_store
    _hook_store = None
    _ensured_reconcile.clear()
    _bg_tasks.clear()


# --- runner dispatch glue -------------------------------------------------------


async def dispatch_semantic_schedule(schedule: Any, *, client: Any) -> None:
    """Route a claimed semantic Schedule row to its duty.

    Called from ``api.routes.actions.dispatch_scheduled_action`` (the same
    dispatch seam every other scheduled action uses, so claim exclusivity is
    inherited from the runner). Deliberately creates NO job rows — a 5-minute
    sweep would flood the unified Jobs feed; the structlog counters emitted by
    the duties are the observability surface. Gated on the master env knob so
    stale rows left over from a disable are cheap no-ops, not surprise spend.
    """
    if not semantic_index_enabled():
        logger.info(
            "semantic_schedule_skipped_disabled",
            schedule_id=schedule.id,
            action=schedule.action,
        )
        return
    if schedule.action == SEMANTIC_EMBED_FILL_ACTION:
        await run_embed_fill_sweep()
    elif schedule.action == SEMANTIC_RECONCILE_ACTION:
        await reconcile_kg(client, schedule.tenant_id, schedule.kg_name)
    else:  # defensive: only semantic actions are routed here
        raise ValueError(f"not a semantic schedule action: {schedule.action!r}")


# Fire-and-forget reconciles for deployments with NO runner (zero-config OSS:
# no DSN, scheduler off). Strong refs, mirroring explore.schedule_recompute —
# a bare create_task result is only weakly referenced by the loop and can be
# GC'd mid-flight.
_bg_tasks: set = set()


def schedule_reconcile_task(neptune: Any, tenant_id: str, kg_name: str) -> None:
    """Fire-and-forget one reconcile (the reindex route's no-runner fallback)."""

    async def _safe() -> None:
        try:
            await reconcile_kg(neptune, tenant_id, kg_name)
        except Exception:  # noqa: BLE001 — background task must not crash the loop
            logger.warning(
                "semantic_reconcile_task_failed",
                tenant_id=tenant_id,
                kg_name=kg_name,
                exc_info=True,
            )

    task = asyncio.create_task(_safe())
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


# --- duty (a): embed-fill sweep --------------------------------------------------


async def run_embed_fill_sweep(
    *,
    index: Optional[SemanticIndex] = None,
    api_key: Optional[str] = None,
    limit: int = 100,
    max_attempts: Optional[int] = None,
) -> dict[str, int]:
    """Drain ``embedding IS NULL`` rows through the shared embed client.

    The NULL embedding column IS the durable queue (protocol docstring), which
    is what makes this sweep crash-safe: a deploy kill mid-fill loses nothing —
    rows already filled stay filled (``fill_embeddings`` is guarded by the
    ``content_hash`` optimistic-concurrency token), the rest are still NULL and
    drain on the next sweep.

    Poison handling: a failed batch is ``mark_embed_failed`` (attempt_count++)
    and the sweep MOVES ON — an in-sweep ``seen`` set prevents re-fetching the
    same rows in this run, and ``fetch_pending(max_attempts=…)`` dead-letters
    rows past the cutoff on later runs (they stay inspectable via a higher
    ``max_attempts``, never silently vanish). Backoff = one sweep interval per
    attempt (the sweep cadence is the spacing; no per-row timer state).

    Counters (structlog, emitted every run — no silent zeros):
    ``embeds_pending`` (rows found queued), ``embeds_filled``,
    ``embed_failures``.
    """
    from cograph_client.config import settings
    from cograph_client.nlp.embed_client import EMBEDDING_MODEL, embed_texts

    counters = {"embeds_pending": 0, "embeds_filled": 0, "embed_failures": 0}
    if not semantic_index_enabled():
        logger.info("semantic_embed_fill_skipped_disabled")
        return counters

    idx = index if index is not None else get_semantic_index()
    key = api_key if api_key is not None else settings.openrouter_api_key
    cutoff = max_attempts if max_attempts is not None else embed_max_attempts()

    # Keys of rows that FAILED this sweep. Only failures stay pending (a
    # successful fill drains the row from the queue), so only failures need
    # the fetch window widened to slide past them — adding every processed row
    # here would make the window (and the backend's scan) grow linearly with
    # progress, i.e. a quadratic sweep over a large healthy queue.
    seen: set[tuple] = set()
    for _ in range(_MAX_SWEEP_ITERATIONS):
        # fetch_pending drains in deterministic (PK) order, so a row that just
        # FAILED is still at the head of the queue. Widening the window by the
        # rows already failed this sweep lets the fetch slide past them —
        # otherwise a poison row at the head would wedge the whole sweep at
        # ``limit=len(failed)`` (exactly the failure mode ONTA-181 forbids).
        batch = await idx.fetch_pending(limit=limit + len(seen), max_attempts=cutoff)
        fresh = [c for c in batch if c.key() not in seen][:limit]
        if not fresh:
            break
        counters["embeds_pending"] += len(fresh)
        if not key:
            # Lexical-only deployment (no OpenRouter key): rows stay queued —
            # search still works degraded (generated tsvector), and the queue
            # drains the moment a key is configured. Loud, not silent.
            logger.warning(
                "semantic_embed_fill_no_api_key", pending=counters["embeds_pending"]
            )
            break
        try:
            vectors = await embed_texts([c.chunk_text for c in fresh], api_key=key)
            counters["embeds_filled"] += await idx.fill_embeddings(
                fresh, vectors, embed_model=EMBEDDING_MODEL
            )
        except Exception as exc:  # noqa: BLE001 — a bad batch must not wedge the sweep
            seen.update(c.key() for c in fresh)  # failed rows stay pending: slide past
            counters["embed_failures"] += len(fresh)
            await idx.mark_embed_failed(fresh, error=str(exc)[:500])
            logger.warning(
                "semantic_embed_fill_batch_failed",
                batch_size=len(fresh),
                error=str(exc)[:200],
            )
    logger.info("semantic_embed_fill_sweep", **counters)
    return counters


# --- duty (b): Neptune-scan reconcile --------------------------------------------


def marked_doc_keys(
    triples: Sequence[Triple], marked_predicates: set[str]
) -> set[tuple[str, str]]:
    """(entity_uri, attr-local-name) pairs carrying a marked LITERAL value.

    The write hook diffs this against what ``extract_semantic_chunks`` actually
    emitted to find docs that must be DELETED for this write: a marked attr
    whose canonicalized doc came out empty (values all blank), or one deduped
    away because it mirrors another attr's doc (extract indexes identical docs
    once). Matching mirrors the extractor exactly — exact predicate URI OR
    lower-cased local name — via the extractor's own helpers, so the two can
    never disagree about what "marked" means.
    """
    marked_locals = {_local_name(m) for m in marked_predicates}
    keys: set[tuple[str, str]] = set()
    for s, p, o in triples:
        if not isinstance(s, str) or not isinstance(p, str) or not isinstance(o, str):
            continue
        if p not in marked_predicates and _local_name(p) not in marked_locals:
            continue
        if _is_uri_object(o):
            continue  # entity reference, never a text doc
        keys.add((s, _local_name(p)))
    return keys


async def _fetch_marker_map(neptune: Any, tenant_id: str) -> dict[str, bool]:
    """Uncached ``{attr URI -> is_free_text}`` fetch that RAISES on failure.

    Deliberately NOT :func:`~cograph_client.graph.text_markers.get_free_text_map`:
    that request-path helper is best-effort (returns ``{}`` on a Neptune
    hiccup), which is right for query routing but catastrophic here — an empty
    map is indistinguishable from "no markers", and reconciling against it
    would ghost-delete the whole KG's index and let the heuristic overwrite
    REASON-layer verdicts. A correctness worker must abort (the runner retries
    on the next cadence) rather than act on a maybe-empty map. Same query
    builder + constant as the shared helper, so the map semantics can't drift.
    """
    from cograph_client.graph.ontology_queries import (
        TEXT_KIND_FREE_TEXT,
        text_kind_map_query,
    )
    from cograph_client.graph.parser import parse_sparql_results
    from cograph_client.graph.queries import tenant_graph_uri

    raw = await neptune.query(text_kind_map_query(tenant_graph_uri(tenant_id)))
    _, bindings = parse_sparql_results(raw)
    return {
        row["attr"]: row.get("kind") == TEXT_KIND_FREE_TEXT
        for row in bindings
        if row.get("attr")
    }


async def _distinct_literal_predicates(neptune: Any, kg_graph: str) -> list[str]:
    """Every predicate carrying at least one literal object in the KG graph."""
    from cograph_client.graph.parser import parse_sparql_results

    sparql = (
        f"SELECT DISTINCT ?p FROM <{kg_graph}> WHERE {{\n"
        f"  ?e ?p ?o .\n"
        f"  FILTER(isLiteral(?o))\n"
        f"}}"
    )
    _, bindings = parse_sparql_results(await neptune.query(sparql))
    return [row["p"] for row in bindings if row.get("p")]


async def _apply_default_candidacy(
    neptune: Any,
    tenant_id: str,
    kg_graph: str,
    literal_predicates: list[str],
    marker_map: dict[str, bool],
) -> dict[str, int]:
    """Write durable textKind verdicts for attributes with NO verdict yet.

    The ONTA-177 hand-off: schema-pass-mapped attributes arrive with a marker,
    but client-mapped CSV rows and enrichment-minted attributes never met the
    schema pass — this is their (name-blind) candidacy path. Only predicates in
    the canonical ``types/{T}/attrs/{a}`` shape are considered; verdicts go
    through ``upsert_attribute_text_kind`` (single-valued, idempotent) and the
    tenant's marker cache is invalidated so query-side consumers see them.
    """
    from cograph_client.graph.ontology_queries import (
        TEXT_KIND_FREE_TEXT,
        upsert_attribute_text_kind,
    )
    from cograph_client.graph.parser import parse_sparql_results
    from cograph_client.graph.queries import tenant_graph_uri
    from cograph_client.graph.text_markers import (
        TextCandidacy,
        classify_text_candidacy,
        invalidate,
    )

    # Local names already marked free_text: the extractor's documented
    # conflation marks that local name on EVERY type, so a same-named attr on
    # another type is already covered — re-classifying it as "undecided" would
    # fight the existing verdict.
    marked_locals = {_local_name(u) for u, ft in marker_map.items() if ft}

    undecided: list[tuple[str, str, str]] = []  # (pred_uri, type_name, attr_name)
    for pred in literal_predicates:
        if pred in marker_map:
            continue  # decided (free_text or decided-no)
        m = _ATTR_URI_RE.match(pred)
        if m is None:
            continue  # system/foreign predicate — never carries a verdict
        if _local_name(pred) in marked_locals:
            continue
        undecided.append((pred, m.group("type"), m.group("attr")))

    counters = {"attrs_marked_free_text": 0, "attrs_marked_not_text": 0}
    if not undecided:
        return counters
    if len(undecided) > _MAX_CANDIDACY_ATTRS_PER_RUN:
        logger.info(
            "semantic_candidacy_capped",
            undecided=len(undecided),
            cap=_MAX_CANDIDACY_ATTRS_PER_RUN,
        )
        # Fairness guard: a deterministic prefix would starve everything past
        # the cap FOREVER when more than the cap stay perpetually AMBIGUOUS
        # (heuristic verdicts are durable, but AMBIGUOUS attrs are re-sampled
        # every run, so a stable order re-samples the same head each time).
        # Randomizing before truncation gives every undecided attr a chance of
        # being sampled on each run, so all of them are eventually classified.
        random.shuffle(undecided)
        undecided = undecided[:_MAX_CANDIDACY_ATTRS_PER_RUN]

    onto_graph = tenant_graph_uri(tenant_id)
    wrote = False
    for pred, type_name, attr_name in undecided:
        sample_sparql = (
            f"SELECT ?o FROM <{kg_graph}> WHERE {{\n"
            f"  ?e <{pred}> ?o .\n"
            f"  FILTER(isLiteral(?o))\n"
            f"}} LIMIT {_CANDIDACY_SAMPLE_SIZE}"
        )
        _, rows = parse_sparql_results(await neptune.query(sample_sparql))
        verdict = classify_text_candidacy([row.get("o", "") for row in rows])
        if verdict is TextCandidacy.AMBIGUOUS:
            # Needs the LLM REASON layer (name-aware) — not available in a
            # background worker; stays undecided and is re-sampled next run.
            continue
        kind = (
            TEXT_KIND_FREE_TEXT
            if verdict is TextCandidacy.FREE_TEXT
            else TEXT_KIND_NOT_TEXT
        )
        await neptune.update(
            upsert_attribute_text_kind(onto_graph, type_name, attr_name, kind)
        )
        wrote = True
        if verdict is TextCandidacy.FREE_TEXT:
            counters["attrs_marked_free_text"] += 1
        else:
            counters["attrs_marked_not_text"] += 1
    if wrote:
        # Make the fresh verdicts visible to the request path immediately (the
        # TTL remains the cross-process backstop).
        invalidate(tenant_id)
    return counters


def _sparql_string_literal(value: str) -> str:
    """Escape a Python string for embedding in a double-quoted SPARQL literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _scan_query(
    kg_graph: str,
    predicates: Sequence[str],
    limit: int,
    after_entity: Optional[str] = None,
) -> str:
    """One keyset-paginated scan page: entities strictly after
    ``after_entity`` (the last COMPLETELY-scanned entity of the previous page),
    in stable ``?e ?p ?o`` order. Keyset instead of OFFSET so page N costs the
    same as page 1 — OFFSET makes the store re-walk (and re-sort) every
    already-scanned row, an O(pages²) total scan."""
    values = " ".join(f"<{p}>" for p in predicates)
    entity_filter = (
        f'  FILTER(STR(?e) > "{_sparql_string_literal(after_entity)}")\n'
        if after_entity
        else ""
    )
    return (
        f"SELECT ?e ?p ?o FROM <{kg_graph}> WHERE {{\n"
        f"  VALUES ?p {{ {values} }}\n"
        f"  ?e ?p ?o .\n"
        f"{entity_filter}"
        f"}} ORDER BY ?e ?p ?o LIMIT {limit}"
    )


async def _scan_triples(
    neptune: Any, kg_graph: str, predicates: Sequence[str]
) -> tuple[list[Triple], bool]:
    """Scan ``?e ?p ?o`` for the given predicates via keyset pagination by
    entity, bounded by the page cap.

    Each full page holds back its trailing entity group — the page boundary
    may have cut that entity's rows mid-group — and the next page re-fetches
    from ``FILTER(STR(?e) > "<last complete entity>")``, so every entity's
    rows arrive CONTIGUOUS AND COMPLETE. That grouping is load-bearing for
    ``extract_semantic_chunks``: its intra-entity doc dedup and per-entity
    chunk cap assume they see all of an entity's values together.

    Returns ``(triples, truncated)``. ``truncated=True`` means the page cap
    (:data:`_MAX_SCAN_PAGES`) was exhausted and the scan is PARTIAL — logged
    here, and the caller must NOT ghost-delete against it (a partial expected
    set would mass-delete perfectly healthy docs).
    """
    from cograph_client.graph.parser import parse_sparql_results

    page = _scan_page_size()
    triples: list[Triple] = []
    after: Optional[str] = None  # last completely-scanned entity
    for _page_ix in range(_MAX_SCAN_PAGES):
        sparql = _scan_query(kg_graph, predicates, page, after_entity=after)
        _, rows = parse_sparql_results(await neptune.query(sparql))
        page_triples: list[Triple] = []
        for row in rows:
            e, p = row.get("e", ""), row.get("p", "")
            if e and p:
                page_triples.append((e, p, row.get("o", "")))
        if len(rows) < page:
            # Final page: nothing after it, so every entity here is complete.
            triples.extend(page_triples)
            return triples, False
        if not page_triples:
            continue  # defensive: a full page of unusable bindings — bounded by the cap
        last_entity = page_triples[-1][0]
        complete = [t for t in page_triples if t[0] != last_entity]
        if complete:
            # Hold the trailing (possibly partial) entity group back; the next
            # page re-fetches it from its first row.
            triples.extend(complete)
            after = complete[-1][0]
        else:
            # The whole page is ONE entity: entity-level keyset cannot page
            # inside it, so keep what we have (its first `page` rows — far
            # beyond extract's per-entity chunk cap at any sane page size) and
            # step past it. Loud, never silent: rows beyond the page are lost.
            logger.warning(
                "semantic_scan_entity_exceeds_page",
                kg_graph=kg_graph,
                entity_uri=last_entity,
                page_size=page,
            )
            triples.extend(page_triples)
            after = last_entity
    logger.warning(
        "semantic_scan_truncated",
        kg_graph=kg_graph,
        pages=_MAX_SCAN_PAGES,
        page_size=page,
    )
    return triples, True


async def _upsert_in_doc_batches(idx: SemanticIndex, chunks: list[SemanticChunk]) -> None:
    """Upsert in batches packed on DOC boundaries (complete-document contract:
    all chunks of one (entity, attr) doc must travel in one call). Batched so a
    crash mid-reconcile persists partial progress — the rerun skips unchanged
    hashes and converges (the partial-resume property)."""
    batch: list[SemanticChunk] = []
    current_doc: Optional[tuple] = None
    for chunk in chunks:
        if (
            batch
            and len(batch) >= _UPSERT_BATCH_CHUNKS
            and chunk.doc_key() != current_doc
        ):
            await idx.upsert_chunks(batch)
            batch = []
        batch.append(chunk)
        current_doc = chunk.doc_key()
    if batch:
        await idx.upsert_chunks(batch)


async def reconcile_kg(
    neptune: Any,
    tenant_id: str,
    kg_name: str,
    *,
    index: Optional[SemanticIndex] = None,
) -> dict[str, int]:
    """Reconcile ONE KG's semantic index against Neptune (source of truth).

    Steps (idempotent end-to-end — an interrupted run leaves the index strictly
    closer to converged, and the rerun finishes the job):

    1. fetch the marker map (uncached, raising — see :func:`_fetch_marker_map`);
    2. apply the default candidacy heuristic to undecided attributes and fold
       any new ``free_text`` verdicts into this run's marker set;
    3. snapshot the index's doc listing (``list_docs``) — taken **FIRST,
       before every other Neptune read of the run** (marker fetch, candidacy
       sampling, the scan). The ordering is load-bearing: a doc the write hook
       indexes anywhere inside the run's read window (including during the
       candidacy pass, whose marker writes can extend the scan-predicate set
       after it was derived) would otherwise be in the listing but absent from
       the expected set and get ghost-deleted. Snapshot-first means any doc
       indexed after the snapshot simply isn't a ghost candidate this run (its
       Neptune triples land before its index rows, so the NEXT run's scan sees
       it), and any doc in the snapshot already has its triples committed and
       IS seen by this scan;
    4. scan the instance graph for marked predicates (plus ``rdf:type`` /
       label predicates for display parity with hook-written rows) — keyset
       pagination by entity (whole-entity groups; see :func:`_scan_triples`) —
       matching the extractor's exact-URI-or-local-name semantics so
       hook-written docs are never mistaken for ghosts;
    5. re-extract chunks and upsert by ``content_hash``: docs whose hash AND
       denormalized attrs are unchanged are skipped entirely when the backend
       supports doc listing (``skipped_unchanged_hash``); docs whose text is
       unchanged but whose attrs drifted (type/label changed without the
       marked text changing) ARE re-upserted (``attrs_repaired``) — the
       backend contract keeps their filled embeddings while refreshing attrs;
    6. DELETE ghosts: snapshot docs absent from the expected set (covers
       ER-merged entities, normalization deletes, emptied values, AND
       marker-removed attributes in one diff), batched through the backend's
       ``delete_docs`` (per-doc ``delete`` fallback for third-party backends).
       SKIPPED — loudly — when the backend predates ``list_docs`` (module
       docstring) **or when the scan was truncated at the page cap**: a
       partial expected set must never drive deletions (it would mass-delete
       every doc past the truncation point).

    Raises on Neptune failures — the schedule runner logs and retries on the
    next cadence; acting on partial reads would be worse than waiting.
    """
    from cograph_client.graph.queries import kg_graph_uri

    counters = {
        "chunks_written": 0,
        "skipped_unchanged_hash": 0,
        "attrs_repaired": 0,
        "ghosts_deleted": 0,
        "attrs_marked_free_text": 0,
        "attrs_marked_not_text": 0,
    }
    if not semantic_index_enabled():
        logger.info(
            "semantic_reconcile_skipped_disabled",
            tenant_id=tenant_id,
            kg_name=kg_name,
        )
        return counters

    idx = index if index is not None else get_semantic_index()
    kg_graph = kg_graph_uri(tenant_id, kg_name)

    # 0. Snapshot the index FIRST — before ANY of this run's Neptune reads
    # (marker fetch, candidacy sampling, the scan). The ordering is
    # load-bearing beyond just "before the scan": the candidacy pass below can
    # issue up to _MAX_CANDIDACY_ATTRS_PER_RUN sample queries and even write
    # new markers, so a doc the write hook indexes during that window (for an
    # attr this run's scan-predicate set was derived too early to include)
    # would land in `current` with no chance of appearing in `expected` — and
    # be ghost-deleted. Snapshot-first closes the whole window: any doc
    # indexed after this point simply isn't a ghost candidate this run (its
    # Neptune triples land before its index rows, so the NEXT run claims it).
    #
    # getattr, not a direct call: list_docs is a Protocol method now, but a
    # third-party backend compiled against the pre-list_docs Protocol must
    # degrade gracefully, never crash (module docstring). Both first-party
    # backends implement it, so the default paths always take this branch.
    lister = getattr(idx, "list_docs", None)
    doc_listing_supported = callable(lister)
    current: dict[tuple[str, str], str] = {}
    # Doc key -> stored denormalized attrs; None when the backend's listing
    # predates the 4-tuple form (legacy 3-tuples: hash-diff only, no attrs
    # repair — degrade, don't crash).
    current_attrs: dict[tuple[str, str], Optional[dict[str, Any]]] = {}
    if doc_listing_supported:
        for row in await lister(tenant_id, kg_name=kg_name):
            key = (row[0], row[1])
            current[key] = row[2]
            current_attrs[key] = (
                dict(row[3]) if len(row) > 3 and row[3] is not None else None
            )

    # 1+2. Markers, then the default heuristic for undecided attributes.
    marker_map = await _fetch_marker_map(neptune, tenant_id)
    literal_preds = await _distinct_literal_predicates(neptune, kg_graph)
    candidacy = await _apply_default_candidacy(
        neptune, tenant_id, kg_graph, literal_preds, marker_map
    )
    counters.update(candidacy)
    if candidacy["attrs_marked_free_text"] or candidacy["attrs_marked_not_text"]:
        marker_map = await _fetch_marker_map(neptune, tenant_id)

    marked = {uri for uri, is_ft in marker_map.items() if is_ft}
    marked_locals = {_local_name(u) for u in marked}

    # 3. Scan: marked predicates by exact URI OR local name (the extractor's
    # conflation — a marked local name covers same-named attrs on every type),
    # plus rdf:type / label predicates for the denormalized display attrs.
    scan_preds: set[str] = set(marked)
    for pred in literal_preds:
        local = _local_name(pred)
        if local in marked_locals or local in _LABEL_LOCALS:
            scan_preds.add(pred)
    scan_preds.add(_RDF_TYPE)
    scan_preds.add(_RDFS_LABEL)

    # 4. Scan (keyset-paginated, whole-entity groups).
    triples: list[Triple] = []
    scan_truncated = False
    if marked:
        triples, scan_truncated = await _scan_triples(
            neptune, kg_graph, sorted(scan_preds)
        )

    # Client-side re-sort for strict parity with the write hook, which sorts
    # its re-read triples in Python (kg_writer._fetch_touched_entity_triples).
    # The server's ORDER BY ?e ?p ?o may order heterogeneous literal values
    # differently than Python's codepoint sort on the rendered strings; the
    # extractor's first-qualifying-wins choices (labels, cross-attr dedup)
    # must not depend on which side built the doc.
    triples.sort()

    chunks = extract_semantic_chunks(
        triples, tenant_id=tenant_id, kg_name=kg_name, marked_predicates=marked
    )

    # 5. Diff against the snapshot, upsert changes: a doc is written when its
    # hash changed OR (hash unchanged but) its denormalized attrs drifted —
    # the attrs-repair pass; the backend contract preserves the filled
    # embedding on unchanged-hash rewrites while refreshing attrs.
    expected: dict[tuple[str, str], str] = {}
    for c in chunks:
        expected[(c.entity_uri, c.attr)] = c.content_hash

    if doc_listing_supported:
        to_write = []
        repaired: set[tuple[str, str]] = set()
        for c in chunks:
            key = (c.entity_uri, c.attr)
            if current.get(key) != c.content_hash:
                to_write.append(c)
                continue
            stored_attrs = current_attrs.get(key)
            if stored_attrs is not None and stored_attrs != c.attrs:
                to_write.append(c)
                repaired.add(key)
        counters["skipped_unchanged_hash"] = len(chunks) - len(to_write)
        counters["attrs_repaired"] = len(repaired)
    else:
        # No listing: upsert everything — the backend's upsert contract still
        # keeps unchanged-hash rows (and their embeddings) as-is.
        to_write = chunks

    if to_write:
        await _upsert_in_doc_batches(idx, to_write)
        counters["chunks_written"] = len(to_write)

    # 6. Ghosts: snapshot docs not in the expected set. One diff covers every
    # bypass class — ER merges, normalization deletes, emptied docs, unmarked
    # (decided-no) attributes. NEVER against a truncated scan: a partial
    # expected set would mass-delete every healthy doc past the cutoff.
    if not doc_listing_supported:
        logger.warning(
            "semantic_reconcile_ghost_scan_skipped",
            tenant_id=tenant_id,
            kg_name=kg_name,
            reason="backend predates the Protocol's list_docs method",
        )
    elif scan_truncated:
        logger.warning(
            "semantic_reconcile_ghosts_skipped_scan_truncated",
            tenant_id=tenant_id,
            kg_name=kg_name,
            reason=(
                "the Neptune scan hit the page cap; the expected set is "
                "partial, so ghost deletion is skipped this run (raise "
                "COGRAPH_SEMANTIC_SCAN_PAGE_SIZE for KGs this large)"
            ),
        )
    else:
        ghosts = sorted(set(current) - set(expected))
        if ghosts:
            deleter = getattr(idx, "delete_docs", None)
            if callable(deleter):
                # One batched round trip for the whole ghost set.
                await deleter(ghosts, tenant_id, kg_name=kg_name)
            else:
                # Third-party backend predating delete_docs: per-doc fallback.
                for entity_uri, attr in ghosts:
                    await idx.delete(
                        entity_uri, tenant_id, kg_name=kg_name, attr=attr
                    )
        counters["ghosts_deleted"] = len(ghosts)

    logger.info(
        "semantic_reconcile",
        tenant_id=tenant_id,
        kg_name=kg_name,
        marked_attrs=len(marked),
        doc_listing_supported=doc_listing_supported,
        **counters,
    )
    return counters
