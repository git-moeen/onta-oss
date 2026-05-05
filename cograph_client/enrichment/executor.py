"""Async executor for enrichment jobs.

Reads entities from Neptune, runs them through the source funnel
(lite tier = wikidata, with cache), and either stages results for
review or applies them directly based on conflict_policy.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import structlog

from cograph_client.enrichment.cache import EnrichmentCache
from cograph_client.enrichment.job_store import JobStore
from cograph_client.enrichment.models import (
    ConflictPolicy,
    ConflictReview,
    EnrichJob,
    JobStatus,
    RowResult,
    Verdict,
)
from cograph_client.enrichment.sources.base import SourceAdapter
from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import insert_triples, kg_graph_uri

logger = structlog.stdlib.get_logger("cograph.enrichment")


TYPE_URI_PREFIX = "https://cograph.tech/types/"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
NAME_FALLBACK_ATTRS = ["name", "title", "headline"]
WORKER_POOL_SIZE = 8
PROGRESS_FLUSH_EVERY = 10


def _type_uri(type_name: str) -> str:
    return f"{TYPE_URI_PREFIX}{type_name}"


def _attr_uri(type_name: str, attr: str) -> str:
    return f"{TYPE_URI_PREFIX}{type_name}/attrs/{attr}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _build_select_query(
    graph_uri: str, type_name: str, attributes: list[str], limit: Optional[int]
) -> str:
    type_uri = _type_uri(type_name)
    attr_uris = [_attr_uri(type_name, a) for a in attributes]
    fallback_uris = [_attr_uri(type_name, a) for a in NAME_FALLBACK_ATTRS]

    in_list = ", ".join(f"<{u}>" for u in attr_uris) if attr_uris else "<urn:none>"
    fallback_in = ", ".join(f"<{u}>" for u in fallback_uris)

    limit_clause = f"\nLIMIT {int(limit)}" if limit else ""

    # GROUP_CONCAT predicate::value for all matching attribute triples.
    # Also pull a label / name fallback for entity_label.
    return (
        f"SELECT ?e ?label ?nameAttr\n"
        f'  (GROUP_CONCAT(DISTINCT CONCAT(STR(?p), "::", STR(?o)); separator="||") AS ?vals)\n'
        f"FROM <{graph_uri}> WHERE {{\n"
        f"  ?e a <{type_uri}> .\n"
        f"  OPTIONAL {{ ?e <{RDFS_LABEL}> ?label }}\n"
        f"  OPTIONAL {{ ?e ?fp ?nameAttr . FILTER(?fp IN ({fallback_in})) }}\n"
        f"  OPTIONAL {{ ?e ?p ?o . FILTER(?p IN ({in_list})) }}\n"
        f"}} GROUP BY ?e ?label ?nameAttr"
        f"{limit_clause}"
    )


def _parse_vals(vals_field: str) -> dict[str, str]:
    """Parse ?vals (predicate::value pairs joined by '||') into a dict.

    If the same predicate appears multiple times, the first one wins.
    """
    out: dict[str, str] = {}
    if not vals_field:
        return out
    for chunk in vals_field.split("||"):
        if "::" not in chunk:
            continue
        p, _, v = chunk.partition("::")
        if p and p not in out:
            out[p] = v
    return out


def _values_match(existing: str, candidate: str) -> bool:
    """Loose match: case-insensitive substring or exact equality."""
    if not existing or not candidate:
        return False
    a = existing.strip().lower()
    b = candidate.strip().lower()
    if a == b:
        return True
    return a in b or b in a


class EnrichmentExecutor:
    def __init__(
        self,
        neptune_client: NeptuneClient,
        job_store: JobStore,
        cache: EnrichmentCache,
        wikidata_adapter: SourceAdapter,
    ) -> None:
        self._neptune = neptune_client
        self._jobs = job_store
        self._cache = cache
        self._wikidata = wikidata_adapter

    async def count_entities(self, tenant_id: str, kg_name: str, type_name: str) -> int:
        graph_uri = kg_graph_uri(tenant_id, kg_name)
        query = (
            f"SELECT (COUNT(DISTINCT ?e) AS ?n) FROM <{graph_uri}> WHERE {{\n"
            f"  ?e a <{_type_uri(type_name)}> .\n"
            f"}}"
        )
        raw = await self._neptune.query(query)
        _, bindings = parse_sparql_results(raw)
        if not bindings:
            return 0
        try:
            return int(bindings[0].get("n", "0"))
        except (TypeError, ValueError):
            return 0

    async def run(self, job: EnrichJob, tenant_id: str) -> None:
        try:
            job.status = JobStatus.running
            job.started_at = _now()
            await self._jobs.update(job)

            graph_uri = kg_graph_uri(tenant_id, job.kg_name)
            sel = _build_select_query(
                graph_uri, job.type_name, job.attributes, job.limit
            )
            raw = await self._neptune.query(sel)
            _, bindings = parse_sparql_results(raw)

            entities: list[dict] = []
            for row in bindings:
                e_uri = row.get("e", "")
                if not e_uri:
                    continue
                label = row.get("label") or row.get("nameAttr") or _slug_from_uri(e_uri)
                vals = _parse_vals(row.get("vals", ""))
                entities.append({"uri": e_uri, "label": label, "vals": vals})

            job.progress.total = len(entities) * len(job.attributes)
            await self._jobs.update(job)

            sem = asyncio.Semaphore(WORKER_POOL_SIZE)
            counter = {"n": 0}
            counter_lock = asyncio.Lock()

            async def process_entity(ent: dict) -> list[RowResult]:
                results: list[RowResult] = []
                async with sem:
                    for attribute in job.attributes:
                        # Cooperative cancellation
                        latest = await self._jobs.get(job.id)
                        if latest and latest.status == JobStatus.cancelled:
                            return results

                        existing = ent["vals"].get(_attr_uri(job.type_name, attribute))
                        verdicts = await self._lookup(
                            ent["label"], attribute, job, cache_hit_inc=True
                        )
                        best = self._pick_best(verdicts, job.confidence_min)

                        action: str
                        if best is None:
                            action = "no_match"
                        elif existing is None or existing == "":
                            action = "filled"
                        elif _values_match(existing, best.value):
                            action = "verified"
                        else:
                            action = "conflict"

                        results.append(
                            RowResult(
                                entity_uri=ent["uri"],
                                attribute=attribute,
                                existing_value=existing,
                                verdict=best,
                                action=action,  # type: ignore[arg-type]
                            )
                        )

                        async with counter_lock:
                            counter["n"] += 1
                            if action == "filled":
                                job.progress.filled += 1
                            elif action == "verified":
                                job.progress.verified += 1
                            elif action == "conflict":
                                job.progress.conflicts += 1
                            elif action == "skipped":
                                job.progress.skipped += 1
                            job.progress.processed = counter["n"]
                            if counter["n"] % PROGRESS_FLUSH_EVERY == 0:
                                await self._jobs.update(job)
                return results

            tasks = [asyncio.create_task(process_entity(e)) for e in entities]
            all_rows: list[RowResult] = []
            for t in tasks:
                rows = await t
                all_rows.extend(rows)

            # Re-check cancellation after work loop.
            latest = await self._jobs.get(job.id)
            if latest and latest.status == JobStatus.cancelled:
                job.status = JobStatus.cancelled
                job.completed_at = _now()
                await self._jobs.update(job)
                return

            # Keep only conflicts in long-term results (per spec).
            job.results = [r for r in all_rows if r.action == "conflict"]

            # Apply phase
            policy = job.conflict_policy
            if policy == ConflictPolicy.stage:
                job.status = JobStatus.review
                job.completed_at = _now()
                await self._jobs.update(job)
                return

            triples = self._select_triples_for_policy(all_rows, job.type_name, policy)
            if triples:
                await self._neptune.update(insert_triples(graph_uri, triples))
            job.status = JobStatus.applied
            job.completed_at = _now()
            await self._jobs.update(job)

        except Exception as exc:  # noqa: BLE001
            logger.exception("enrichment_job_failed", job_id=job.id, error=str(exc))
            job.status = JobStatus.failed
            job.error = str(exc)
            job.completed_at = _now()
            try:
                await self._jobs.update(job)
            except Exception:  # noqa: BLE001
                pass

    async def _lookup(
        self, entity_label: str, attribute: str, job: EnrichJob, cache_hit_inc: bool
    ) -> list[Verdict]:
        source = self._wikidata.name
        cached = await self._cache.get(entity_label, attribute, source)
        if cached is not None:
            if cache_hit_inc:
                job.progress.cache_hits += 1
            return cached
        verdicts = await self._wikidata.lookup(entity_label, attribute, {})
        await self._cache.put(entity_label, attribute, source, verdicts)
        return verdicts

    def _pick_best(
        self, verdicts: list[Verdict], confidence_min: float
    ) -> Optional[Verdict]:
        eligible = [v for v in verdicts if v.confidence >= confidence_min]
        if not eligible:
            return None
        return max(eligible, key=lambda v: v.confidence)

    def _select_triples_for_policy(
        self, rows: list[RowResult], type_name: str, policy: ConflictPolicy
    ) -> list[tuple[str, str, str]]:
        triples: list[tuple[str, str, str]] = []
        for r in rows:
            if r.verdict is None:
                continue
            p = _attr_uri(type_name, r.attribute)
            if policy == ConflictPolicy.overwrite:
                if r.action in ("filled", "conflict", "verified"):
                    triples.append((r.entity_uri, p, r.verdict.value))
            elif policy == ConflictPolicy.verify:
                if r.action == "filled":
                    triples.append((r.entity_uri, p, r.verdict.value))
            elif policy == ConflictPolicy.skip:
                if r.action == "filled":
                    triples.append((r.entity_uri, p, r.verdict.value))
        return triples

    async def apply_decisions(
        self, job_id: str, decisions: list[ConflictReview]
    ) -> int:
        job = await self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        graph_uri = kg_graph_uri(job.tenant_id, job.kg_name)
        triples: list[tuple[str, str, str]] = []
        for d in decisions:
            if d.decision != "accept":
                continue
            p = _attr_uri(job.type_name, d.attribute)
            triples.append((d.entity_uri, p, d.proposed.value))
        if triples:
            await self._neptune.update(insert_triples(graph_uri, triples))
        job.status = JobStatus.applied
        job.completed_at = _now()
        await self._jobs.update(job)
        return len(triples)


def _slug_from_uri(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1]
