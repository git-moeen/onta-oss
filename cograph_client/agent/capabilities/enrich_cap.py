"""Enrichment capability — with clean-before-enrich composition.

Reuses the existing enrichment engine (no reimplementation):

* ``plan`` parses the NL instruction into the existing :class:`EnrichRequest`
  shape (attributes + optional scope ``predicate=value`` + tier + confidence).
  THEN it detects a prerequisite: if the **scope predicate's target values are
  composite** (un-normalized — a delimiter shows up in the sampled target
  labels), scoping by ``value`` would MISS the rows packed inside a composite
  cell (e.g. scope ``speaks=Persian`` misses an entity whose ``speaks`` points
  at ``English__Persian``). In that case it emits a NORMALIZE step FIRST (reusing
  :class:`NormalizeCapability.plan` so the cleanup logic isn't duplicated) and
  sets the enrich step's ``depends_on`` to it. Returns ``[normalize_step?,
  enrich_step]``. No writes.

* ``execute`` runs the enrichment as a background job, building the EXACT same
  :class:`EnrichJob` + ``EnrichmentExecutor.run`` the ``/enrich/jobs`` route
  builds (strong-ref ``_spawn`` so the task can't be GC'd). Returns an ack.

The agent never calls the ``/enrich`` HTTP route — it drives the executor + job
store directly via the same primitives.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog

from cograph_client.agent.capabilities.normalize_cap import NormalizeCapability
from cograph_client.agent.registry import AgentContext, PlanStep
from cograph_client.enrichment.models import (
    EnrichJob,
    EnrichScope,
    EnrichmentTier,
    JobStatus,
)
from cograph_client.enrichment.sources.base import adapter_cost, get_adapter
from cograph_client.enrichment.tiers import get_chain
from cograph_client.graph.ontology_queries import list_types_query
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import kg_graph_uri, tenant_graph_uri
from cograph_client.normalization.inference import (
    list_type_schema,
    sample_predicate_values,
)
from cograph_client.resolver.llm_router import PRIMARY_MODEL, openrouter_chat

logger = structlog.stdlib.get_logger("cograph.agent.enrich")

_bg_tasks: set[asyncio.Task] = set()

# Conservative default cap so a large/expensive enrich is BOUNDED by default
# (COG-123). It is written into the plan ``params`` (and the EnrichJob.limit at
# execute time) and surfaced in the preview; the user can override it. 200 keeps
# a first paid run small enough to inspect cheaply while still covering most
# scoped subsets in one pass.
_DEFAULT_PLAN_LIMIT = 200

# Outer safety cap on a resolved subset ("top N", "those", an explicit list) so a
# missing/over-broad LIMIT in the generated subset SPARQL can't fan a paid enrich
# out to thousands of entities. The subset's own N (when given) still applies; this
# only bounds the worst case.
_SUBSET_MAX = 500

# Functional confidence floor the AGENT'S PLAN uses for web-sourced enrichments
# (COG-121). Web adapters (Exa/Parallel/…) return verdicts at a low, conservative
# prior, so the global EnrichRequest default of 0.85 silently filters ALL of them
# → 0 values written. This floor is conservative (still rejects junk) but low
# enough that calibrated web verdicts actually land. It is applied ONLY in the
# agent's plan and is overridable; the global 0.85 default is unchanged, so
# direct API callers keep the safe default.
_WEB_CONFIDENCE_MIN = 0.4
# The global EnrichRequest default; the marker for "user did not ask for a
# specific confidence" so we only override an UNSET (default) confidence.
_DEFAULT_CONFIDENCE_MIN = 0.85


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


# Delimiters that signal a composite (un-normalized) target value. "__" is the
# slugified list separator the ingest produces; the rest are raw list delimiters.
_COMPOSITE_DELIMS = ["__", ", ", "; ", " / ", " | "]


class EnrichCapability:
    name = "enrich"

    def __init__(self, normalize: NormalizeCapability | None = None) -> None:
        # Reuse the normalize capability to BUILD the prerequisite step so the
        # clean-before-enrich logic lives in exactly one place.
        self._normalize = normalize or NormalizeCapability()

    def describe(self) -> str:
        return (
            "Fill in or verify missing attributes on a type by looking them up "
            "from external sources (enrichment). Use for 'enrich', 'fill in', "
            "'look up', 'find the <attribute> for <type>' requests, optionally "
            "scoped (e.g. 'for managers', 'who speak Persian')."
        )

    async def plan(
        self,
        ctx: AgentContext,
        instruction: str,
        parsed: dict | None = None,
    ) -> list[PlanStep]:
        """Build [normalize_step?, enrich_step] from the instruction.

        ``parsed`` (optional) lets the planner pass an already-parsed request
        (attributes/scope/tier/confidence). When absent we ground the extraction
        in the type's REAL schema: we fetch the active type's attribute +
        relationship names from the ontology and feed them to the LLM so an NL
        phrase like "current company" maps to the ``company`` attribute (and the
        tier is chosen with web-fact guidance), instead of the model guessing a
        stray word ("current") and the planner bailing to clarify.

        The target TYPE is resolved from the instruction first, NOT from the
        Explorer's current selection: "enrich brokers with their websites"
        enriches Broker even when PropertyListing is the type selected in the UI.
        ``ctx.type_name`` (the selection) is only a fallback for when the message
        names no known type (see :func:`_resolve_target_type`).
        """
        known_types = await _list_types(ctx)
        type_name = _resolve_target_type(instruction, known_types, ctx.type_name)
        if not type_name:
            return []
        schema = await list_type_schema(ctx.neptune, ctx.tenant_id, type_name)
        req = parsed or await _extract_enrich_request(
            ctx, instruction, type_name, schema
        )
        attributes: list[str] = req.get("attributes") or []
        if not attributes:
            return []
        tier = _coerce_tier(req.get("tier"))
        requested_confidence = float(
            req.get("confidence_min", _DEFAULT_CONFIDENCE_MIN)
            or _DEFAULT_CONFIDENCE_MIN
        )
        scope = req.get("scope")  # {"predicate":..., "value":...} | None

        # Ranked / specific subset ("the top 5 brokers by listing count", "those",
        # an explicit list). A field=value scope CANNOT express a ranked aggregate,
        # so when the extractor flags a subset we resolve it to the CONCRETE entity
        # IRIs via the shared NL→SPARQL pipeline and enrich exactly those
        # (``entity_uris`` wins over scope in the executor). Fail CLOSED: if the
        # user explicitly named a subset we could not resolve, do NOT silently
        # enrich the whole type — return no plan so the turn clarifies instead.
        subset = req.get("subset")  # {"description": str, "limit": int|None} | None
        entity_uris: list[str] | None = None
        if subset and subset.get("description"):
            entity_uris = await self._resolve_subset_uris(ctx, type_name, subset)
            if not entity_uris:
                return []
            scope = None  # the explicit entity set supersedes any value-scope

        # Resolve the tier's adapter chain ONCE and derive (a) whether it is a
        # paid/web chain and (b) the per-entity paid cost — both driven by
        # adapter-declared metadata, never adapter names (COG-123/COG-121 boundary).
        per_entity_cost, paid_adapters, has_paid = _resolve_chain_cost(tier)

        # COG-121: for a WEB-sourced enrichment (the resolved chain has a paid/web
        # adapter) lower the plan's confidence_min to a functional floor so the
        # low-prior web verdicts aren't all silently filtered → 0 writes. Only
        # override an UNSET (default 0.85) confidence: if the user explicitly asked
        # for a stricter/looser value we respect it. Overridable downstream.
        confidence_min = requested_confidence
        confidence_lowered = False
        user_set_confidence = abs(requested_confidence - _DEFAULT_CONFIDENCE_MIN) > 1e-9
        if has_paid and not user_set_confidence:
            # NOTE (interaction): the executor's per-attribute ontology-confidence
            # override only fires when confidence_min == _DEFAULT_CONFIDENCE_MIN
            # (0.85), i.e. the "unset" sentinel. Lowering to the web floor here is
            # INTENTIONAL and relaxes BOTH the global 0.85 default AND any stricter
            # per-attribute ontology threshold for these web-sourced facts: without
            # the floor the low-prior web verdicts are all filtered → 0 writes. A
            # user who wants per-attribute thresholds honored sets confidence_min
            # explicitly (which keeps user_set_confidence True and skips this floor).
            confidence_min = _WEB_CONFIDENCE_MIN
            confidence_lowered = True

        steps: list[PlanStep] = []
        depends_on: list[str] = []

        # clean-before-enrich: if a scope predicate's target is composite,
        # normalize it FIRST so the scope actually matches the packed rows.
        if scope and scope.get("predicate"):
            samples, _kind = await sample_predicate_values(
                ctx.neptune,
                ctx.tenant_id,
                ctx.kg_name,
                type_name,
                scope["predicate"],
            )
            if _looks_composite(samples):
                norm_steps = await self._normalize.plan(
                    ctx, instruction, predicate_leaves=[scope["predicate"]]
                )
                if norm_steps:
                    norm = norm_steps[0]
                    norm.rationale = (
                        f"Clean '{scope['predicate']}' before enrichment: its "
                        f"values are composite, so scoping by "
                        f"{scope.get('value')!r} would miss packed rows."
                    )
                    steps.append(norm)
                    depends_on = [norm.id]

        # Bound the job + estimate how many entities it will touch. For an explicit
        # entity set the user already chose the size, so there is NO cap and the
        # matched count is exact (= the resolved IRIs). Otherwise apply the
        # conservative default cap (COG-123) and estimate the matched count via the
        # executor's existing index-efficient COUNT — no new query engine. The
        # executor calls the adapter chain once per (entity, attribute) pair
        # (executor.process_entity loops over job.attributes around _lookup_chain),
        # so a paid lookup runs entities × len(attributes) times; cost ≈
        # per-entity-paid-cost × that paid-call count. When the count can't be
        # computed cheaply we fall back to a clearly-labeled estimate (the cap).
        if entity_uris is not None:
            limit = None
            matched, matched_exact = len(entity_uris), True
        else:
            limit = _DEFAULT_PLAN_LIMIT
            matched, matched_exact = await self._estimate_matched(
                ctx, type_name, scope, attributes
            )
        cost = _estimate_cost(
            tier=tier,
            per_entity_cost=per_entity_cost,
            paid_adapters=paid_adapters,
            has_paid=has_paid,
            matched=matched,
            matched_exact=matched_exact,
            limit=limit,
            n_attributes=len(attributes),
        )

        subset_desc = subset.get("description") if subset else None
        n_entities = len(entity_uris) if entity_uris is not None else None
        if n_entities is not None:
            target_phrase = (
                f"the {n_entities} selected {type_name} "
                f"{'entity' if n_entities == 1 else 'entities'}"
            )
        else:
            target_phrase = f"matched {type_name} entities (capped at {limit})"
        enrich_step = PlanStep(
            capability=self.name,
            action="run_enrichment",
            params={
                "type_name": type_name,
                "attributes": attributes,
                "tier": tier.value,
                "confidence_min": confidence_min,
                "scope": scope,
                "limit": limit,
                "entity_uris": entity_uris,
            },
            rationale=(
                f"Enrich {', '.join(attributes)} on {type_name}"
                + (
                    f" for {subset_desc}" if subset_desc
                    else (
                        f" scoped to {scope['predicate']}={scope['value']}"
                        if scope else ""
                    )
                )
                + f" via the {tier.value} tier."
            ),
            confidence=0.8,
            preview={
                "summary": (
                    f"Look up {', '.join(attributes)} for {target_phrase} "
                    "and stage the results for review."
                ),
                "scope": scope,
                "tier": tier.value,
                "limit": limit,
                "entity_count": n_entities,
                "confidence_min": confidence_min,
                "confidence_note": _confidence_note(
                    confidence_min, confidence_lowered
                ),
                "cost_estimate": cost.get("note", ""),
            },
            cost=cost,
            depends_on=depends_on,
        )
        steps.append(enrich_step)
        return steps

    async def _estimate_matched(
        self,
        ctx: AgentContext,
        type_name: str,
        scope: dict | None,
        attributes: list[str],
    ) -> tuple[Optional[int], bool]:
        """Estimate how many entities the enrich job will match.

        Reuses the executor's existing index-efficient ``count_entities`` (the
        same SELECT/COUNT path COG-112 built — no new query engine). Returns
        ``(count, exact)``: ``exact=True`` when the COUNT actually ran, else
        ``(None, False)`` so the caller falls back to a labeled estimate rather
        than reporting a misleading 0. Defensive: any executor/Neptune error or a
        missing executor degrades to ``(None, False)`` — the plan must never fail
        on a cost estimate.
        """
        executor = ctx.extras.get("enrichment_executor")
        if executor is None or not hasattr(executor, "count_entities"):
            return None, False
        enrich_scope = None
        if scope and scope.get("predicate") and scope.get("value"):
            try:
                enrich_scope = EnrichScope(
                    predicate=scope["predicate"], value=scope["value"]
                )
            except Exception:  # noqa: BLE001 — a bad scope just means "no count"
                return None, False
        try:
            n = await executor.count_entities(
                ctx.tenant_id,
                ctx.kg_name,
                type_name,
                scope=enrich_scope,
            )
            return int(n), True
        except Exception:  # noqa: BLE001
            logger.warning("agent_enrich_count_failed", exc_info=True)
            return None, False

    async def _resolve_subset_uris(
        self, ctx: AgentContext, type_name: str, subset: dict
    ) -> list[str]:
        """Resolve a ranked/specific subset to the concrete entity IRIs it names.

        Reuses the shared NL→SPARQL engine (:meth:`NLQueryPipeline.select_entity_uris`,
        the same pipeline the question capability/``/ask`` route use) so "the 5
        brokers with the most listings" becomes those 5 IRIs — no new query engine,
        no client-side ranking. The subset's own LIMIT is honored by the generated
        SPARQL; ``_SUBSET_MAX`` is an outer safety cap so a runaway/unbounded subset
        can't fan out to thousands of paid calls. Returns ``[]`` on any failure —
        the caller fails closed rather than enriching the whole type by accident.
        """
        description = str(subset.get("description") or "").strip()
        if not description:
            return []
        raw_limit = subset.get("limit")
        lim = (
            int(raw_limit)
            if isinstance(raw_limit, (int, float))
            and not isinstance(raw_limit, bool)
            and raw_limit > 0
            else None
        )
        lim = min(lim, _SUBSET_MAX) if lim else _SUBSET_MAX

        # Lazy import: keep the heavy NL pipeline (and its anthropic client) out of
        # agent-registry import time, mirroring QueryCapability._build_pipeline.
        from cograph_client.nlp.pipeline import NLQueryPipeline

        pipeline = NLQueryPipeline(ctx.neptune, ctx.anthropic_key)
        onto_graph = tenant_graph_uri(ctx.tenant_id)
        instance_graph = (
            kg_graph_uri(ctx.tenant_id, ctx.kg_name) if ctx.kg_name else onto_graph
        )
        try:
            return await pipeline.select_entity_uris(
                description, type_name, onto_graph, instance_graph, lim
            )
        except Exception:  # noqa: BLE001 — resolution must never crash planning
            logger.warning("agent_enrich_subset_resolve_failed", exc_info=True)
            return []

    async def execute(self, ctx: AgentContext, step: PlanStep) -> dict:
        """Create + run an EnrichJob in the background (same as /enrich/jobs)."""
        p = step.params
        executor = ctx.extras.get("enrichment_executor")
        job_store = ctx.extras.get("enrichment_job_store")
        if executor is None or job_store is None:
            raise RuntimeError(
                "enrichment executor/job_store not available in agent context"
            )
        scope = None
        if p.get("scope") and p["scope"].get("predicate"):
            scope = EnrichScope(
                predicate=p["scope"]["predicate"], value=p["scope"]["value"]
            )
        # Explicit entity set (resolved from a ranked/specific subset at plan time);
        # the executor uses a VALUES block and lets it win over scope.
        entity_uris = p.get("entity_uris") or None
        limit = p.get("limit")
        job = EnrichJob(
            id=str(uuid.uuid4()),
            tenant_id=ctx.tenant_id,
            kg_name=ctx.kg_name,
            type_name=p["type_name"],
            attributes=p["attributes"],
            tier=_coerce_tier(p.get("tier")),
            status=JobStatus.queued,
            created_at=datetime.now(timezone.utc),
            conflict_policy=_default_conflict_policy(),
            confidence_min=float(
                p.get("confidence_min", _DEFAULT_CONFIDENCE_MIN)
                or _DEFAULT_CONFIDENCE_MIN
            ),
            scope=scope,
            entity_uris=entity_uris,
            # Carry the plan's proposed cap so the job actually honors the bound
            # surfaced to the user at plan time (COG-123). int() guards a stray
            # non-int; None leaves whole-subset behavior unchanged. bool is a
            # subclass of int, so exclude it explicitly — a stray True/False must
            # not be coerced to a 1/0 limit.
            limit=int(limit)
            if isinstance(limit, (int, float)) and not isinstance(limit, bool) and limit
            else None,
        )
        await job_store.create(job)
        _spawn(executor.run(job, ctx.tenant_id))
        return {
            "kind": "ack",
            "capability": self.name,
            "action": step.action,
            "job_id": job.id,
            "job_status": job.status.value,
            "message": (
                f"Enriching {', '.join(job.attributes)} on {job.type_name} "
                "in the background; results will be staged for review."
            ),
        }


# --- target-type resolution: prefer the type NAMED in the instruction --------- #
# The Explorer sends the currently-selected type as ``ctx.type_name``. That
# selection must NEVER override a type the user actually names in their message:
# "enrich brokers with their websites" enriches Broker even when PropertyListing
# is the selected type. We resolve the target type from the instruction text
# (case-insensitive, CamelCase- and plural-tolerant) and fall back to the
# selection ONLY when the message names no known type — so a missing/wrong UI
# selection no longer bails the plan to "couldn't determine the specifics".

_TYPE_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*")


async def _list_types(ctx: AgentContext) -> list[str]:
    """The tenant's declared type names, for resolving the target type from text.

    Reuses the SAME ontology query the ``/ontology/types`` route and the ontology
    capability use (:func:`list_types_query`) — a bounded single round-trip read,
    never an instance scan. Defensive: any read error degrades to ``[]`` so type
    resolution falls back to the selected type rather than failing the plan.
    """
    try:
        onto_graph = tenant_graph_uri(ctx.tenant_id)
        _, rows = parse_sparql_results(
            await ctx.neptune.query(list_types_query(onto_graph))
        )
    except Exception:  # noqa: BLE001 — a type-list read must never break planning
        logger.warning("agent_enrich_list_types_failed", exc_info=True)
        return []
    seen: set[str] = set()
    names: list[str] = []
    for r in rows:
        label = (r.get("label") or "").strip()
        if label and label not in seen:
            seen.add(label)
            names.append(label)
    return names


def _singularize(word: str) -> str:
    """Tiny dependency-free English singularizer — for MATCHING only, not display."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"  # companies -> company, agencies -> agency
    if len(w) > 4 and w.endswith(("ses", "xes", "zes", "ches", "shes")):
        return w[:-2]  # addresses -> address, boxes -> box
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]  # brokers -> broker, listings -> listing
    return w


def _type_tokens(text: str) -> list[str]:
    return [w.lower() for w in _TYPE_WORD_RE.findall(text or "")]


def _camel_words(type_name: str) -> list[str]:
    """Split a type name into lowercase words: ``PropertyListing`` -> ['property',
    'listing'], ``URL`` -> ['url'], ``real_estate_agent`` -> ['real', 'estate',
    'agent']. Lets a multi-word type be phrase-matched against the instruction."""
    parts: list[str] = []
    for chunk in re.split(r"[\s_\-]+", type_name or ""):
        parts.extend(
            re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", chunk)
        )
    return [p.lower() for p in parts if p]


def _phrase_in_tokens(words: list[str], tokens: list[str]) -> bool:
    """True if ``words`` appears as a contiguous run in ``tokens`` (each compared
    singularized, so "property listings" matches the type ``PropertyListing``)."""
    if not words:
        return False
    w = [_singularize(x) for x in words]
    t = [_singularize(x) for x in tokens]
    span = len(w)
    return any(t[i : i + span] == w for i in range(len(t) - span + 1))


def _match_type_in_text(text: str, known_types: list[str]) -> str | None:
    """Return the known type NAMED in ``text``, or None.

    A single-word type matches a (singularized) token; a multi-word (CamelCase)
    type matches the full phrase in order. When several types match, the LONGEST
    name wins so a specific ``PropertyListing`` beats a bare ``Property``.
    """
    tokens = _type_tokens(text)
    if not tokens or not known_types:
        return None
    singles = {_singularize(t) for t in tokens}
    best: str | None = None
    best_len = -1
    for name in known_types:
        words = _camel_words(name)
        if not words:
            continue
        if len(words) == 1:
            hit = _singularize(words[0]) in singles
        else:
            hit = _phrase_in_tokens(words, tokens)
        if hit and len(name) > best_len:
            best, best_len = name, len(name)
    return best


def _resolve_target_type(
    instruction: str, known_types: list[str], selected: str | None
) -> str | None:
    """Pick the type to enrich, PREFERRING one named in the instruction.

    Order:
      1. a known type named in the message wins — the user said it explicitly;
      2. else the selected (UI) type, when it is a real KG type OR when we
         couldn't list types at all (preserve the legacy selection behavior);
      3. else, when the KG has exactly one type, that type;
      4. else None — the caller asks which type to enrich.
    """
    named = _match_type_in_text(instruction, known_types)
    if named:
        return named
    if selected and (not known_types or selected in known_types):
        return selected
    if len(known_types) == 1:
        return known_types[0]
    return None


def _default_conflict_policy():
    from cograph_client.enrichment.models import ConflictPolicy

    return ConflictPolicy.stage


def _looks_composite(samples: list[str]) -> bool:
    """Cheap composite check: any sampled target value carries a list delimiter."""
    for v in samples:
        for d in _COMPOSITE_DELIMS:
            if d in v:
                return True
    return False


def _coerce_tier(tier) -> EnrichmentTier:
    if isinstance(tier, EnrichmentTier):
        return tier
    try:
        return EnrichmentTier(str(tier))
    except ValueError:
        return EnrichmentTier.lite


def _resolve_chain_cost(tier: EnrichmentTier) -> tuple[float, int, bool]:
    """Per-entity paid cost for a tier, derived GENERICALLY from adapter metadata.

    Resolves the tier's adapter chain (:func:`get_chain`), looks up each adapter
    in the global registry, and sums the declared ``cost_per_call`` of the PAID
    adapters (:func:`adapter_cost`). Returns
    ``(per_entity_paid_cost, paid_adapter_count, has_paid)``.

    Boundary-clean (COG-123): "paid" and "how much" come ONLY from what an
    adapter declares about itself — never a hardcoded adapter name. The OSS
    Wikidata adapter declares free, so the OSS-only ``lite`` chain costs 0; a
    downstream deployment that registers a paid adapter (Exa/Parallel/…) with
    ``is_paid``/``cost_per_call`` gets a non-zero estimate with no OSS change.
    The "cache" pseudo-entry in a chain is not an adapter and is skipped, mirroring
    the executor's :meth:`_lookup_chain`. An unregistered adapter name contributes
    nothing (it can't run, so it can't cost) — same as the executor skipping it.
    """
    per_entity_cost = 0.0
    paid_adapters = 0
    for name in get_chain(tier):
        if name == "cache":
            continue
        adapter = get_adapter(name)
        if adapter is None:
            continue
        is_paid, cost = adapter_cost(adapter)
        if is_paid:
            paid_adapters += 1
            per_entity_cost += cost
    return per_entity_cost, paid_adapters, paid_adapters > 0


def _estimate_cost(
    tier: EnrichmentTier,
    per_entity_cost: float,
    paid_adapters: int,
    has_paid: bool,
    matched: Optional[int],
    matched_exact: bool,
    limit: Optional[int],
    n_attributes: int = 1,
) -> dict:
    """Honest plan-time cost estimate (COG-123).

    Cost ≈ per-entity-paid-cost × min(matched, limit) × ``n_attributes``. The
    executor calls the adapter chain once per (entity, attribute) pair (see
    ``EnrichmentExecutor.process_entity`` looping over ``job.attributes`` around
    ``_lookup_chain``), so a multi-attribute enrich multiplies the paid-call
    count — quoting only by entities under-counts by ``n_attributes×``. The
    per-entity cost and the paid/free decision are driven by adapter-declared
    metadata (see :func:`_resolve_chain_cost`), so this never special-cases an
    adapter by name.

    - **All-free chain** (no paid adapter — e.g. the OSS ``lite`` Wikidata-only
      tier): ``paid_calls=0`` and an explicit "no paid calls" note.
    - **Paid chain**: report the estimated paid-call count (= entities to process,
      capped at ``limit``, times ``n_attributes``) and the dollar estimate. When
      the matched count was computed exactly we say ``N``; when it couldn't be
      computed cheaply we fall back to the ``limit`` as a clearly-labeled
      UPPER-BOUND estimate ("up to N") — NEVER a silent 0 for a paid tier.
    """
    if not has_paid:
        return {
            "paid_calls": 0,
            # Key names match the web plan-step cost contract EXACTLY
            # (``step.cost.estimated_usd`` / ``step.cost.paid_calls`` —
            # web/app/components/explore/useAgentChat.ts AgentStepCost +
            # AgentChat.tsx PlanStepRow). Do NOT rename without updating both.
            "estimated_usd": 0.0,
            "per_entity_cost_usd": 0.0,
            "note": f"{tier.value} tier — no paid calls (all sources are free).",
        }

    # Number of ENTITIES the paid adapters will be called for, capped at limit.
    if matched_exact and matched is not None:
        entities = matched if limit is None else min(matched, limit)
        estimated = True
    else:
        # Couldn't compute the matched count cheaply — bound by the proposed
        # limit and label it an upper bound rather than reporting a bogus 0.
        entities = limit if limit is not None else 0
        estimated = False

    # The chain runs once per (entity, attribute) pair, so the paid-call count
    # (and dollar cost) scales by the number of attributes being enriched.
    n_attributes = max(int(n_attributes), 1)
    paid_calls = entities * n_attributes
    estimated_cost = round(per_entity_cost * paid_calls, 4)

    entity_phrase = f"{entities}" if estimated else (
        f"up to {entities}" if entities else "an unknown number of"
    )
    matched_clause = (
        f"~{matched} matched" if (matched_exact and matched is not None)
        else "matched count unavailable (using the cap as an upper bound)"
    )
    if n_attributes > 1:
        # Multi-attribute: state the basis so the entities × attributes = calls
        # arithmetic is transparent.
        note = (
            f"{tier.value} tier (paid): ≈ {entity_phrase} entities × "
            f"{n_attributes} attributes = {paid_calls} paid lookups "
            f"(${per_entity_cost:.4f}/call) ≈ ${estimated_cost:.2f} "
            f"[{matched_clause}]."
        )
    else:
        note = (
            f"{tier.value} tier (paid): {entity_phrase} paid lookups "
            f"(${per_entity_cost:.4f}/entity × {entities}) ≈ ${estimated_cost:.2f} "
            f"[{matched_clause}]."
        )
    return {
        "paid_calls": paid_calls,
        "paid_calls_estimated": not estimated,  # True = upper-bound, not exact
        "paid_adapters": paid_adapters,
        "attributes": n_attributes,
        "per_entity_cost_usd": round(per_entity_cost, 4),
        # Key names match the web plan-step cost contract EXACTLY
        # (``step.cost.estimated_usd`` / ``step.cost.paid_calls`` —
        # web/app/components/explore/useAgentChat.ts AgentStepCost +
        # AgentChat.tsx PlanStepRow). Do NOT rename without updating both.
        "estimated_usd": estimated_cost,
        "matched_entities": matched if matched_exact else None,
        "limit": limit,
        "note": note,
    }


def _confidence_note(confidence_min: float, lowered: bool) -> str:
    """Human-facing explanation of the chosen ``confidence_min`` (COG-121)."""
    if lowered:
        return (
            f"Web-sourced facts: confidence_min lowered to {confidence_min:g} so "
            f"low-prior web verdicts are written instead of all being filtered out "
            f"(the strict {_DEFAULT_CONFIDENCE_MIN:g} default would write nothing). "
            f"Overridable."
        )
    return f"confidence_min = {confidence_min:g}."


# --- LLM extraction grounded in the type's real schema ----------------------- #

# Open-web / person / company facts the FREE Wikidata tier usually can't answer
# well — these should default to the paid web ``core`` tier (Parallel/Exa). Used
# only as a deterministic backstop when the LLM omits a tier.
_WEB_FACT_HINTS = {
    "company", "employer", "organization", "organisation", "website", "url",
    "homepage", "description", "bio", "summary", "reviews", "rating", "founder",
    "headquarters", "hq", "location", "address", "email", "phone", "title",
    "role", "position", "industry", "revenue", "funding", "ceo", "linkedin",
}

_EXTRACT_SYSTEM = """\
You extract an enrichment request from a user's instruction, GROUNDED in the \
active type's real schema. You are given the type's actual ATTRIBUTE names and \
RELATIONSHIP names (with their target types). Map the natural-language phrases \
in the instruction onto those real predicate names — never invent a stray word.

Return STRICT JSON only (no markdown):
{
  "attributes": ["<attribute name(s) to enrich>"],
  "scope": {"predicate": "<an attribute OR relationship name>", "value": "<v>"} \
or null,
  "subset": {"description": "<self-contained description of WHICH entities>", \
"limit": <int or null>} or null,
  "tier": "lite" | "base" | "core" | "pro",
  "confidence_min": 0.85
}

RULES:
- "attributes" are the field(s) to FILL IN / look up. Map the noun in the \
instruction to the nearest existing ATTRIBUTE name. Examples: "current company" \
/ "employer" -> "company"; "the website" -> "website"; "their bio" -> \
"description". If NO existing attribute fits but the user clearly names a new \
fact to add, propose a clean lowercase singular noun for it (e.g. "company") — \
NEVER emit a modifier word like "current", "their", "the", "missing".
- "scope" restricts WHICH entities to enrich by a simple FIELD=VALUE match ("for \
managers", "who speak Persian"). Its "predicate" MUST be one of the given \
attribute or relationship names. "languages" / "what they speak" -> the "speaks" \
relationship; "level" / "who are managers" -> the level attribute/relationship. \
If there is no such filter, return null.
- "subset" pins enrichment to a RANKED or SPECIFIC set of entities that a simple \
field=value "scope" CANNOT express — "the top 5 <type> by <metric>", "the 10 \
most recent ...", "those"/"them"/"these" (entities referenced earlier in the \
conversation), or an explicit named list. Write "description" as a SELF-CONTAINED \
phrase naming exactly which entities (resolve pronouns using the whole \
conversation, e.g. turn "those" into "the 5 brokers with the most property \
listings"), and "limit" = the count if the user gave one (else null). Use \
"subset" ONLY for ranked/specific sets; for "all <type>" or a plain field=value \
filter leave it null. "scope" and "subset" are mutually exclusive — prefer \
"subset" when the request is ranked or refers to specific earlier entities.
- "tier" selects the data source. Choose "core" (paid web search: \
Parallel/Exa) for OPEN-WEB facts about people or companies — employer, company, \
website, description, bio, reviews, founder, headquarters, email, role, title, \
industry, etc. Wikidata (the free "lite" tier) does NOT have these. Use "lite" \
ONLY for structured, catalogued identifiers Wikidata reliably holds (e.g. a \
country's ISO code, a film's release year, a well-known org's founding date). \
When unsure for a web-lookup attribute, default to "core".
- "confidence_min" defaults to 0.85 unless the user asks for stricter/looser."""

_EXTRACT_USER_TEMPLATE = """\
Type: {type_name}
Attributes: {attributes}
Relationships: {relationships}

Instruction: {instruction}

Extract the enrichment request as strict JSON."""


async def _extract_enrich_request(
    ctx: AgentContext,
    instruction: str,
    type_name: str,
    schema: dict,
) -> dict:
    """LLM-extract {attributes, scope, tier, confidence_min}, schema-grounded.

    Falls back to the deterministic regex parser when there is no key or the LLM
    errors, so the agent never 500s on extraction. The extracted attributes /
    scope predicate are validated against the type's real schema; the tier is
    backstopped from the web-fact heuristic when the model omits it.
    """
    attr_names = [a for a in schema.get("attributes", []) if a]
    rel_names = [r.get("name") for r in schema.get("relationships", []) if r.get("name")]
    parsed: dict | None = None
    if ctx.openrouter_key:
        rels_block = ", ".join(
            f"{r['name']} (-> {r.get('target_type') or '?'})"
            for r in schema.get("relationships", [])
            if r.get("name")
        ) or "(none)"
        user = _EXTRACT_USER_TEMPLATE.format(
            type_name=type_name,
            attributes=", ".join(attr_names) or "(none)",
            relationships=rels_block,
            instruction=instruction,
        )
        try:
            text = await openrouter_chat(
                ctx.openrouter_key,
                _EXTRACT_SYSTEM,
                user,
                model=PRIMARY_MODEL,
                temperature=0,
                max_tokens=400,
                timeout=30,
            )
            parsed = _parse_json_object(text)
        except Exception:
            logger.warning("agent_enrich_extract_failed", exc_info=True)
            parsed = None
    if not parsed:
        parsed = _parse_enrich_instruction(instruction)
    return _validate_enrich_request(parsed, attr_names, rel_names)


def _validate_enrich_request(
    parsed: dict, attr_names: list[str], rel_names: list[str]
) -> dict:
    """Sanitize an extracted request against the type's real schema.

    - attributes: dropped if they are stray modifier words; otherwise normalized
      (matched case-insensitively to an existing attribute, else kept as a
      proposed new attribute name).
    - scope.predicate: kept only if it resolves to a real attribute/relationship
      (case-insensitively); otherwise the scope is dropped (a bad scope would
      match nothing).
    - tier: web-fact backstop applied when missing/invalid.
    """
    known = {n.lower(): n for n in (*attr_names, *rel_names)}
    attr_lookup = {n.lower(): n for n in attr_names}

    raw_attrs = parsed.get("attributes") or []
    if isinstance(raw_attrs, str):
        raw_attrs = [raw_attrs]
    attributes: list[str] = []
    for a in raw_attrs:
        norm = _normalize_attr(a)
        if not norm:
            continue
        attributes.append(attr_lookup.get(norm.lower(), norm))
    # De-dupe preserving order.
    seen: set[str] = set()
    attributes = [a for a in attributes if not (a.lower() in seen or seen.add(a.lower()))]

    scope = parsed.get("scope")
    if isinstance(scope, dict) and scope.get("predicate") and scope.get("value"):
        pred = str(scope["predicate"]).strip()
        # Resolve against the real schema. When the schema is EMPTY (no ontology
        # available — e.g. a brand-new/uningested type) we can't validate, so we
        # keep the extracted predicate rather than silently dropping a valid scope.
        resolved = known.get(pred.lower(), pred if not known else None)
        scope = (
            {"predicate": resolved, "value": str(scope["value"]).strip()}
            if resolved
            else None
        )
    else:
        scope = None

    # Ranked/specific subset → a self-contained description + optional positive
    # int limit. Kept independent of the type schema (it is resolved later via a
    # SPARQL select, not validated against predicate names). A subset supersedes a
    # value-scope, so drop the scope when a subset is present.
    subset = parsed.get("subset")
    if isinstance(subset, dict) and str(subset.get("description") or "").strip():
        raw_limit = subset.get("limit")
        s_limit = (
            int(raw_limit)
            if isinstance(raw_limit, (int, float))
            and not isinstance(raw_limit, bool)
            and raw_limit > 0
            else None
        )
        subset = {"description": str(subset["description"]).strip(), "limit": s_limit}
        scope = None
    else:
        subset = None

    tier = parsed.get("tier")
    if tier not in {t.value for t in EnrichmentTier}:
        tier = _tier_for_attributes(attributes)

    return {
        "attributes": attributes,
        "scope": scope,
        "subset": subset,
        "tier": tier,
        "confidence_min": parsed.get("confidence_min", 0.85),
    }


# Stray modifier / filler words an extractor must never emit as an attribute.
_STOPWORDS = {
    "current", "the", "a", "an", "their", "its", "his", "her", "missing",
    "this", "that", "these", "those", "all", "each", "every", "some", "new",
    "of", "for", "in", "on", "with",
}


def _normalize_attr(value) -> str:
    """Reduce an extracted attribute phrase to a clean predicate noun, or "".

    Strips a leading modifier ("current company" -> "company"), drops pure
    stopwords ("current" -> ""), and slugs spaces to underscores so the result
    is a usable attribute leaf name.
    """
    if not isinstance(value, str):
        return ""
    words = [w for w in re.split(r"\s+", value.strip()) if w]
    # Drop leading stopwords ("current company" -> "company").
    while words and words[0].lower() in _STOPWORDS:
        words.pop(0)
    # Stop at the first trailing stopword ("company for" -> "company").
    kept: list[str] = []
    for w in words:
        if w.lower() in _STOPWORDS:
            break
        kept.append(w)
    if not kept:
        return ""
    cleaned = "_".join(kept).strip("_-")
    return cleaned if cleaned and cleaned.lower() not in _STOPWORDS else ""


def _tier_for_attributes(attributes: list[str]) -> str:
    """Default tier: ``core`` (paid web) when any attribute is an open-web fact,
    else ``core`` anyway for safety — Wikidata-only ``lite`` is opt-in via the
    LLM (structured identifiers), not the silent default for a web lookup."""
    for a in attributes:
        if a.lower() in _WEB_FACT_HINTS:
            return EnrichmentTier.core.value
    # No clear structured-identifier signal → prefer the paid web tier so a
    # person/company lookup isn't silently downgraded to a Wikidata miss.
    return EnrichmentTier.core.value if attributes else EnrichmentTier.lite.value


def _parse_json_object(text: str) -> dict | None:
    """Best-effort parse of an LLM JSON object reply (tolerant of code fences)."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = "\n".join(
            l for l in stripped.split("\n") if not l.strip().startswith("```")
        )
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        stripped = stripped[start : end + 1]
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


# --- Deterministic fallback parser (no LLM key / LLM error) ------------------ #

_ATTR_TRIGGER = re.compile(
    r"\b(?:enrich|fill in|fill|look up|lookup|find|get|add)\s+(?:the\s+)?"
    r"([A-Za-z_][\w-]*(?:\s+[A-Za-z_][\w-]*)?)",
    re.IGNORECASE,
)
# Relationship scope: "<verb> <Value>" e.g. "speak Persian", "speaks French".
# group(1) = verb, group(2) = value. Verb is lemmatized to its predicate leaf.
_SCOPE_REL = re.compile(
    r"\b(speak|speaks|speaking|knows?|knowing|using|uses?)\s+"
    r"([A-Z][\w-]+)",
)


def _parse_enrich_instruction(instruction: str) -> dict:
    """Deterministic best-effort parse used only when the LLM is unavailable.

    Extracts attribute noun(s) after the enrich verb (dropping a leading
    modifier like "current") and an optional relationship scope. Tier is left
    unset so :func:`_validate_enrich_request` applies the web-fact default.

    Examples:
      "enrich the current company for managers"
        → attributes=["company"]   (the "current" modifier is dropped)
      "enrich company for mentors who speak Persian"
        → attributes=["company"], scope={"predicate":"speaks","value":"Persian"}
    """
    attributes: list[str] = []
    m = _ATTR_TRIGGER.search(instruction)
    if m:
        norm = _normalize_attr(m.group(1))
        if norm:
            attributes = [norm]

    scope = None
    rel = _SCOPE_REL.search(instruction)
    if rel:
        verb = rel.group(1).lower()
        pred = _SCOPE_VERB_LEMMA.get(verb, verb)
        scope = {"predicate": pred, "value": rel.group(2)}
    return {"attributes": attributes, "scope": scope, "tier": None}


# Map inflected scope verbs to their predicate leaf (the ontology stores the
# bare relationship name, e.g. "speaks").
_SCOPE_VERB_LEMMA = {
    "speak": "speaks",
    "speaks": "speaks",
    "speaking": "speaks",
    "know": "knows",
    "knows": "knows",
    "knowing": "knows",
    "use": "uses",
    "uses": "uses",
    "using": "uses",
}
