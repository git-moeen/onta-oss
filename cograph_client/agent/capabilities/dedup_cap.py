"""Dedup capability — find & merge duplicate entities through the agent (COG-122).

Reuses the EXISTING entity-resolution engine end-to-end (no reimplementation,
no new matching code):

* ``plan`` proposes ONE :class:`PlanStep` (``capability="dedup"``,
  ``action="run_dedup"``) describing a second-pass entity-resolution rebuild of
  the active KG. It grounds the preview in the KG's real ER-enabled types (the
  same ``config_for`` selection :func:`rebuild_kg` itself uses) when they can be
  enumerated cheaply, and degrades gracefully to a generic preview when Neptune
  is unavailable — the plan must never fail. Dedup is COMPUTE, not paid web
  calls, so the cost dict is ``{paid_calls: 0, estimated_usd: 0.0, ...}``. No
  writes happen at plan time.

* ``execute`` drives the existing ER engine as a TRACKED background job, mirroring
  the ``actions.py`` find-merge-duplicates worker exactly: it creates an
  :class:`EnrichJob` tagged ``category=dedupe`` in the agent context's job store
  and spawns :func:`cograph_client.resolver.er.rebuild.rebuild_kg` — the SAME
  entry point the explore ``er-rebuild`` route and the dedupe action use. On
  completion it records the merge volume into the job and schedules a type-stats
  recompute (dedup collapses fragments → per-type counts change). Returns an ack
  consistent with the enrich capability so the chat ``result`` rendering works.

Boundary (docs/oss_proprietary_boundary.md): the OSS ER engine CORE — normalize,
block, score, merge, decisive signals — is what this wraps. Advanced ER tooling
(review-queue UI, embedding matchers, active learning, bulk remerge) is PREMIUM
and is NOT pulled in here. The merge criterion is ingest's own
``auto_merge_threshold`` (per-type :class:`ERConfig`), so a rebuild can never
merge more aggressively than ingest already would. A downstream deployment that
wants a curated matcher/verifier plugs it in through the existing ER plugin
protocol (e.g. ``register_external_verifier``) — never by editing this OSS file.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import structlog

from cograph_client.agent.registry import AgentContext, PlanStep
from cograph_client.enrichment.models import (
    ConflictPolicy,
    EnrichJob,
    EnrichmentTier,
    JobCategory,
    JobStatus,
    JobTrigger,
)
from cograph_client.graph.queries import kg_graph_uri
from cograph_client.resolver.er.rebuild import TYPE_URI_PREFIX
from cograph_client.resolver.er.types import config_for

logger = structlog.stdlib.get_logger("cograph.agent.dedup")

# Strong refs to background rebuild tasks (mirrors enrich_cap / normalize_cap /
# actions.py): a bare create_task() is only weakly held by CPython and can be
# GC'd at its first await once the request returns, silently stranding the job.
_bg_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


class DedupCapability:
    """Find-and-merge-duplicates capability behind the single agent endpoint."""

    name = "dedup"

    def describe(self) -> str:
        return (
            "Find and merge duplicate entities (entity resolution): collapse "
            "records that refer to the same real-world thing into one. Use for "
            "'dedupe', 'find duplicates', 'merge duplicate <type>', 'collapse "
            "duplicate records' requests. Operates over the whole KG."
        )

    async def plan(self, ctx: AgentContext, instruction: str) -> list[PlanStep]:
        """Propose ONE dedup step — a second-pass ER rebuild over the active KG.

        Dedup is KG-scoped: :func:`rebuild_kg` auto-discovers every ER-enabled
        type and re-blocks/scores/merges fragments using ingest's exact
        per-type ``auto_merge_threshold``. We don't extract attributes/values
        from the NL (there are none to extract) — we ground the preview in the
        KG's REAL ER-enabled types so the user sees what would actually run.
        """
        if not ctx.kg_name:
            return []

        er_types = await self._er_enabled_types(ctx)
        scope_phrase = (
            f"types: {', '.join(er_types)}" if er_types else "all ER-enabled types"
        )

        return [
            PlanStep(
                capability=self.name,
                action="run_dedup",
                params={
                    "kg_name": ctx.kg_name,
                    # The ER engine selects types itself (config_for); we record
                    # the previewed ER-enabled types for transparency only — the
                    # rebuild is NOT restricted to this list.
                    "er_types": er_types,
                    # Surface (read-only) the merge criterion the engine uses so
                    # the user knows merges follow ingest's own thresholds, not a
                    # looser one. These are per-type ERConfig values, not knobs
                    # this capability overrides.
                    "merge_criterion": "ingest auto_merge_threshold (per-type ERConfig)",
                },
                rationale=(
                    "Run a second-pass entity-resolution rebuild over "
                    f"{ctx.kg_name}: re-block and score every entity of each "
                    "ER-enabled type and merge same-entity fragments using the "
                    "same threshold ingest applies (so it can never over-merge)."
                ),
                confidence=0.85,
                preview={
                    "summary": (
                        f"Find and merge duplicate entities across {scope_phrase} "
                        f"in {ctx.kg_name}. Merges use ingest's own "
                        "auto-merge threshold, so no entity is merged more "
                        "aggressively than during ingestion. Results will be "
                        "summarized as 'N duplicate fragments merged'."
                    ),
                    "kg_name": ctx.kg_name,
                    "er_types": er_types,
                    "merge_criterion": (
                        "ingest auto_merge_threshold (per-type ERConfig)"
                    ),
                },
                # Dedup is COMPUTE (re-blocking + scoring + SPARQL merges), not
                # paid web lookups — no per-call external cost. Keys mirror the
                # enrich plan's cost contract the web reads (estimated_usd /
                # paid_calls) so the cost badge renders consistently.
                cost={
                    "paid_calls": 0,
                    "estimated_usd": 0.0,
                    "note": (
                        "Entity resolution is compute-only (no paid external "
                        "calls)."
                    ),
                },
            )
        ]

    async def _er_enabled_types(self, ctx: AgentContext) -> list[str]:
        """Best-effort list of the KG's ER-enabled type names for the preview.

        Queries the distinct ``rdf:type`` URIs present in the KG's instance graph
        (the same cograph-type filter :func:`rebuild_kg` uses) and keeps those
        that resolve to an :class:`ERConfig` via :func:`config_for` — i.e. the
        exact set the rebuild will act on. Defensive: ANY Neptune/parse error
        degrades to ``[]`` so the plan still proposes a (generic-preview) dedup
        step rather than failing on a preview.
        """
        instance_graph = kg_graph_uri(ctx.tenant_id, ctx.kg_name)
        sparql = (
            "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
            "SELECT DISTINCT ?t\n"
            f"FROM <{instance_graph}>\n"
            "WHERE {\n"
            "  ?e rdf:type ?t .\n"
            f'  FILTER(STRSTARTS(STR(?t), "{TYPE_URI_PREFIX}"))\n'
            "}"
        )
        try:
            data = await ctx.neptune.query(sparql)
        except Exception:  # noqa: BLE001 — preview must never fail the plan
            logger.warning("agent_dedup_type_enum_failed", exc_info=True)
            return []
        rows = (data or {}).get("results", {}).get("bindings", [])
        names: list[str] = []
        seen: set[str] = set()
        for r in rows:
            t = (r.get("t") or {}).get("value")
            if not t or not t.startswith(TYPE_URI_PREFIX):
                continue
            type_name = t[len(TYPE_URI_PREFIX):].rstrip("/")
            # Only ER-enabled types are touched by rebuild_kg (config_for None →
            # skipped). Climbing the subclass chain is done inside the rebuild;
            # the flat config_for here is a faithful-enough preview filter.
            if type_name and config_for(type_name) is not None and (
                type_name.lower() not in seen
            ):
                seen.add(type_name.lower())
                names.append(type_name)
        return sorted(names)

    async def execute(self, ctx: AgentContext, step: PlanStep) -> dict:
        """Run the second-pass ER rebuild as a tracked background job.

        Mirrors ``actions.py``'s find-merge-duplicates worker: create an
        EnrichJob (category=dedupe) in the agent's job store and spawn the
        rebuild against the existing engine. Returns an ack immediately; the UI
        polls the job for the merge volume.
        """
        job_store = ctx.extras.get("enrichment_job_store")
        if job_store is None:
            raise RuntimeError(
                "enrichment job_store not available in agent context"
            )
        kg_name = step.params.get("kg_name") or ctx.kg_name
        job = EnrichJob(
            id=str(uuid.uuid4()),
            tenant_id=ctx.tenant_id,
            kg_name=kg_name,
            # Dedup is KG-wide (the engine selects ER-enabled types itself), so
            # there is no single type_name; "" keeps the job model happy and
            # matches the actions.py dedupe job shape.
            type_name="",
            attributes=[],
            tier=EnrichmentTier.lite,
            status=JobStatus.queued,
            created_at=datetime.now(timezone.utc),
            conflict_policy=ConflictPolicy.stage,
            category=JobCategory.dedupe,
            trigger=JobTrigger.manual,
        )
        await job_store.create(job)
        _spawn(_run_dedup(ctx.neptune, job_store, job.id, ctx.tenant_id, kg_name))
        return {
            "kind": "ack",
            "capability": self.name,
            "action": step.action,
            "job_id": job.id,
            "job_status": job.status.value,
            "message": (
                f"Finding and merging duplicate entities in {kg_name} in the "
                "background; the job will report how many fragments were merged."
            ),
        }


async def _run_dedup(
    neptune,
    job_store,
    job_id: str,
    tenant_id: str,
    kg_name: str,
) -> None:
    """Background worker: run second-pass ER over a KG and record the report.

    Reuses :func:`cograph_client.resolver.er.rebuild.rebuild_kg` directly — the
    SAME primitive the explore ``er-rebuild`` route and the ``actions.py``
    find-merge-duplicates worker use. Records the merge volume into the job's
    progress counters + flips status to applied (or failed) + last_run. On
    success it also schedules a type-stats recompute (a dedup collapses
    fragments and changes per-type counts, so the Explorer's precomputed stats
    are stale until recomputed) — mirroring both existing dedup paths. Detached:
    errors are logged and recorded on the job, never raised.
    """
    # Imported lazily inside the worker (not at module import) to keep the OSS
    # capability importable without the route module, and to avoid any import
    # cycle through explore.py at agent-package import time.
    from cograph_client.api.routes.explore import schedule_recompute
    from cograph_client.resolver.er.rebuild import rebuild_kg

    job = await job_store.get(job_id)
    if job is None:
        return
    job.status = JobStatus.running
    job.started_at = datetime.now(timezone.utc)
    await job_store.update(job)

    try:
        report = await rebuild_kg(neptune, kg_graph_uri(tenant_id, kg_name))
        job = await job_store.get(job_id) or job
        absorbed = int(report.get("fragments_absorbed_total", 0))
        # Record before/after merge volume into the job's progress counters so
        # the UI can show "N duplicates merged" without a bespoke field (same as
        # the actions.py dedupe worker).
        job.progress.total = absorbed
        job.progress.processed = absorbed
        job.error = (
            f"merged {absorbed} duplicate fragment(s) across "
            f"{len(report.get('types', []))} type(s)"
        )
        job.status = JobStatus.applied
        # Merge volume changed per-type counts → refresh the Explorer's
        # precomputed type-stats in the background (best-effort).
        schedule_recompute(neptune, tenant_id, kg_name)
    except Exception as exc:  # noqa: BLE001 — detached worker, never raise
        logger.warning(
            "agent_dedup_failed", tenant=tenant_id, kg=kg_name, error=str(exc)
        )
        job = await job_store.get(job_id) or job
        job.status = JobStatus.failed
        job.error = f"dedup failed: {exc}"
    finally:
        now = datetime.now(timezone.utc)
        job.completed_at = now
        job.last_run = now
        await job_store.update(job)
