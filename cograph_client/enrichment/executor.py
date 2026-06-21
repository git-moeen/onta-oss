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
from cograph_client.enrichment.canonicalize import apply_canonicalizer
from cograph_client.enrichment.job_store import JobStore
from cograph_client.enrichment.models import (
    ConflictPolicy,
    ConflictReview,
    EnrichJob,
    JobStatus,
    RowResult,
    Verdict,
)
from cograph_client.enrichment.sources.base import (
    SourceAdapter,
    get_adapter,
    register_adapter,
)
from cograph_client.enrichment.strategy import (
    AttributeStrategy,
    TypeStrategy,
    load_strategy,
)
from cograph_client.enrichment.tiers import get_chain
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


def _values_match_with_strategy(
    existing: str, candidate: str, attr_strategy: AttributeStrategy | None
) -> bool:
    """Apply canonicalizer + aliases to the existing value before matching."""
    if attr_strategy is None:
        return _values_match(existing, candidate)
    transformed = existing
    if attr_strategy.canonicalizer:
        transformed = apply_canonicalizer(attr_strategy.canonicalizer, transformed)
    # Alias dictionary: literal lookup AND match against the transformed form.
    if attr_strategy.aliases:
        if existing in attr_strategy.aliases:
            transformed = attr_strategy.aliases[existing]
        elif transformed in attr_strategy.aliases:
            transformed = attr_strategy.aliases[transformed]
    return _values_match(transformed, candidate)


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
        # Register the wikidata adapter into the global adapter registry so
        # chain-based lookups can resolve it by name. Idempotent.
        try:
            register_adapter(wikidata_adapter)
        except Exception:  # noqa: BLE001
            pass

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

            # Load ontology-driven strategy. Always returns a TypeStrategy.
            strategy = await load_strategy(self._neptune, tenant_id, job.type_name)
            # Cache-key version for this strategy. A change here auto-invalidates
            # the cache (different key -> clean miss). TODO(ADR-0005 §2): the ADR
            # wants a real strategy_version field on TypeStrategy/AttributeStrategy;
            # derive a stable string until that lands.
            strategy_version = str(getattr(strategy, "version", "v1"))
            # Track which adapter names were missing so we warn once per job.
            missing_adapter_names: set[str] = set()

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
                        attr_strategy = strategy.attributes.get(attribute)

                        # Strategy merge: request value wins; ontology fills gaps.
                        # confidence_min: if ontology specifies one and the
                        # request is at the default (0.85), take the ontology
                        # value. Pragmatic heuristic since EnrichRequest has no
                        # "unset" sentinel.
                        effective_confidence = job.confidence_min
                        if attr_strategy and attr_strategy.confidence_min is not None:
                            if abs(job.confidence_min - 0.85) < 1e-9:
                                effective_confidence = attr_strategy.confidence_min

                        # Adapter chain: per-attribute sources (if any) override
                        # the tier chain.
                        if attr_strategy and attr_strategy.sources:
                            chain = list(attr_strategy.sources)
                        else:
                            chain = get_chain(job.tier)

                        verdicts = await self._lookup_chain(
                            ent["label"],
                            attribute,
                            chain,
                            job,
                            missing_adapter_names,
                            effective_confidence,
                            strategy_version,
                        )
                        best = self._pick_best(verdicts, effective_confidence)

                        action: str
                        if best is None:
                            action = "no_match"
                        elif existing is None or existing == "":
                            action = "filled"
                        elif _values_match_with_strategy(
                            existing, best.value, attr_strategy
                        ):
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

            # Keep conflicts AND fills/verifications in results so the cited
            # verdict (value + source_url + provenance) is retrievable via the
            # job API, not just conflicts. Skips/no-matches carry no verdict.
            job.results = [r for r in all_rows if r.action in ("conflict", "filled", "verified")]

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
                # New attribute values were written → the Explorer's precomputed
                # type-stats (coverage %, counts) are now stale. Recompute them
                # in the background so the panels refresh without waiting for the
                # summary-cache TTL. Only fire when something was actually applied.
                self._schedule_stats_recompute(tenant_id, job.kg_name)
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
        self,
        entity_label: str,
        attribute: str,
        job: EnrichJob,
        cache_hit_inc: bool,
        strategy_version: str = "v1",
    ) -> list[Verdict]:
        source = self._wikidata.name
        cached = await self._cache.get(
            entity_label, attribute, source, job.type_name, strategy_version
        )
        if cached is not None:
            if cache_hit_inc:
                job.progress.cache_hits += 1
            return cached
        verdicts = await self._wikidata.lookup(entity_label, attribute, {})
        await self._cache.put(
            entity_label, attribute, source, verdicts, job.type_name, strategy_version
        )
        return verdicts

    async def _lookup_chain(
        self,
        entity_label: str,
        attribute: str,
        chain: list[str],
        job: EnrichJob,
        missing: set[str],
        confidence_min: float,
        strategy_version: str = "v1",
    ) -> list[Verdict]:
        """Walk an adapter chain, returning verdicts from the first adapter
        that yields one with confidence >= confidence_min.

        - "cache" entries in the chain are skipped (cache is a layer wrapped
          around each adapter call, not an adapter itself).
        - Unregistered adapter names are skipped with a one-shot warning per
          job, never fail the job.
        """
        cache_hit_counted = False
        for name in chain:
            if name == "cache":
                # Cache is a layer, not an adapter.
                continue
            adapter = get_adapter(name)
            if adapter is None:
                if name not in missing:
                    missing.add(name)
                    logger.warning(
                        "enrichment_adapter_missing",
                        adapter=name,
                        job_id=job.id,
                        tier=job.tier.value if hasattr(job.tier, "value") else str(job.tier),
                    )
                continue
            cached = await self._cache.get(
                entity_label, attribute, adapter.name, job.type_name, strategy_version
            )
            if cached is not None:
                if not cache_hit_counted:
                    job.progress.cache_hits += 1
                    cache_hit_counted = True
                verdicts = cached
            else:
                try:
                    verdicts = await adapter.lookup(entity_label, attribute, {})
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "enrichment_adapter_error",
                        adapter=name,
                        job_id=job.id,
                        error=str(exc),
                    )
                    verdicts = []
                await self._cache.put(
                    entity_label,
                    attribute,
                    adapter.name,
                    verdicts,
                    job.type_name,
                    strategy_version,
                )
            # Stop at first sufficient-confidence verdict.
            if any(v.confidence >= confidence_min for v in verdicts):
                return verdicts
        # No adapter yielded a sufficiently-confident verdict; return last (may
        # be empty). For simplicity return [] so caller treats as no_match.
        return []

    def _pick_best(
        self, verdicts: list[Verdict], confidence_min: float
    ) -> Optional[Verdict]:
        eligible = [v for v in verdicts if v.confidence >= confidence_min]
        if not eligible:
            return None
        return max(eligible, key=lambda v: v.confidence)

    @staticmethod
    def _provenance_triples(
        entity_uri: str, type_name: str, attribute: str, verdict
    ) -> list[tuple[str, str, str]]:
        """Persist where + when an enriched value came from, as queryable
        attributes (`<attr>_source_url`, `<attr>_provenance`) on the entity — so
        the citation is visible through /ask and the Explorer, not just in the
        adapter. Audit-friendly: every enriched fact carries its source."""
        out: list[tuple[str, str, str]] = []
        if getattr(verdict, "source_url", None):
            out.append((entity_uri, _attr_uri(type_name, f"{attribute}_source_url"), verdict.source_url))
        prov = (verdict.source or "")
        if getattr(verdict, "reasoning", None):
            prov = f"{prov} ({verdict.reasoning})" if prov else verdict.reasoning
        if prov:
            out.append((entity_uri, _attr_uri(type_name, f"{attribute}_provenance"), prov))
        return out

    def _select_triples_for_policy(
        self, rows: list[RowResult], type_name: str, policy: ConflictPolicy
    ) -> list[tuple[str, str, str]]:
        triples: list[tuple[str, str, str]] = []
        for r in rows:
            if r.verdict is None:
                continue
            p = _attr_uri(type_name, r.attribute)
            applied = False
            if policy == ConflictPolicy.overwrite:
                applied = r.action in ("filled", "conflict", "verified")
            elif policy in (ConflictPolicy.verify, ConflictPolicy.skip):
                applied = r.action == "filled"
            if applied:
                triples.append((r.entity_uri, p, r.verdict.value))
                triples.extend(self._provenance_triples(r.entity_uri, type_name, r.attribute, r.verdict))
        return triples

    async def apply_decisions(
        self, job_id: str, decisions: list[ConflictReview]
    ) -> int:
        job = await self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        graph_uri = kg_graph_uri(job.tenant_id, job.kg_name)
        triples: list[tuple[str, str, str]] = []
        applied = 0  # number of accepted facts (provenance triples don't count)
        for d in decisions:
            if d.decision != "accept":
                continue
            p = _attr_uri(job.type_name, d.attribute)
            triples.append((d.entity_uri, p, d.proposed.value))
            triples.extend(self._provenance_triples(d.entity_uri, job.type_name, d.attribute, d.proposed))
            applied += 1
        if triples:
            await self._neptune.update(insert_triples(graph_uri, triples))
            # Accepted facts were written → refresh the Explorer's precomputed
            # type-stats in the background (mirrors the auto-apply path in run()).
            self._schedule_stats_recompute(job.tenant_id, job.kg_name)
        job.status = JobStatus.applied
        job.completed_at = _now()
        await self._jobs.update(job)
        return applied

    def _schedule_stats_recompute(self, tenant_id: str, kg_name: str) -> None:
        """Fire-and-forget a type-stats recompute after an enrichment write.

        Lazy-imported from the explore route to avoid a module-load import cycle
        (the API route modules cross-reference one another; ingest.py uses the
        same lazy pattern). ``schedule_recompute`` is best-effort — it swallows
        Neptune errors internally — so this never affects the job's outcome.
        """
        from cograph_client.api.routes.explore import schedule_recompute

        schedule_recompute(self._neptune, tenant_id, kg_name)


def _slug_from_uri(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1]
