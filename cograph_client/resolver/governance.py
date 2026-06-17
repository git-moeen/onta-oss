"""Governance pipeline seam for Global ontology writes (ADR 0002 §2, COG-43).

When ingestion meets a brand-new type, it MAY be proposed for the shared
Global-Public layer: a reasoned TypeProposal goes to a judge panel
(independent LLM judges, majority vote — the same pattern as the
type-matcher's 3-judge fan-out). Approved types land in the Public layer
WITH governance provenance (proposer model, reasoning, votes, timestamp —
paralleling the COG-38 per-fact provenance encoding) plus an append-only
changelog entry (ADR 0002 §8). Rejected / tenant-specific types stay in the
tenant layer exactly as today.

Ingestion never blocks on governance: the tenant-layer write happens first
regardless of the verdict, panel failures degrade to the tenant layer, and
every Global write is reversible via revoke_type (the structural answer to
the spider-bench contamination class of bug).

This is the OSS seam only — the production judge service is premium and
plugs in through the JudgePanel protocol.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections import deque
from datetime import datetime, timezone
from typing import Literal, Protocol
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from cograph_client.graph.layers import Layer, layer_type_uri, public_graph_uri
from cograph_client.graph.ontology_queries import RDF, RDFS, XSD, entity_exists_query
from cograph_client.graph.provenance import provenance_graph_uri
from cograph_client.graph.queries import insert_triples
from cograph_client.resolver.models import CSVSchemaMapping, TypeExtension

logger = structlog.stdlib.get_logger("cograph.resolver.governance")

# Governance vocabulary — record nodes live in the Public graph's companion
# provenance graph (same encoding pattern as COG-38 statement metadata);
# changelog entries live in their own append-only named graph.
GOV_NS = "https://cograph.tech/gov/"
GOV_SUBJECT = f"{GOV_NS}subject"
GOV_PROPOSER_MODEL = f"{GOV_NS}proposer_model"
GOV_REASONING = f"{GOV_NS}reasoning"
GOV_TENANT = f"{GOV_NS}tenant"
GOV_APPROVED = f"{GOV_NS}approved"
GOV_VOTES = f"{GOV_NS}votes"
GOV_VOTE = f"{GOV_NS}vote"
GOV_ACTION = f"{GOV_NS}action"
GOV_TIMESTAMP = f"{GOV_NS}timestamp"
# Mapping-shape governance vocabulary (ADR 0003 §5, COG-56). Encoding is OSS
# mechanism (like the rest of GOV_*); the premium judge service writes with it.
GOV_KIND = f"{GOV_NS}kind"
GOV_CONFIDENCE = f"{GOV_NS}confidence"
GOV_DATASET_HINT = f"{GOV_NS}dataset_hint"
GOV_ALIGNED_TO = f"{GOV_NS}aligned_to"

_CHANGELOG_GRAPH_URI = "https://cograph.tech/graphs/global/changelog"


def changelog_graph_uri() -> str:
    """Append-only changelog graph for Global ontology changes (ADR 0002 §8)."""
    return _CHANGELOG_GRAPH_URI


class TypeProposal(BaseModel):
    """A brand-new type proposed for the Global-Public layer."""

    type_name: str
    parent_chain: list[str] = Field(default_factory=list)
    tenant_id: str
    reasoning: str = Field(description="Proposer's written case that the type is universal")
    proposer_model: str


class JudgeVerdict(BaseModel):
    """One judge's vote on a TypeProposal."""

    approve: bool
    reasoning: str = ""


class GovernanceDecision(BaseModel):
    """Outcome of propose_and_judge: where the type belongs, with the votes."""

    target_layer: Literal["public", "tenant"]
    votes: list[JudgeVerdict] = Field(default_factory=list)
    approved: bool


class JudgePanel(Protocol):
    """Pluggable judge panel — the premium governance service implements this."""

    async def judge(self, proposal: TypeProposal) -> list[JudgeVerdict]: ...


GOV_JUDGE_SYSTEM_PROMPT = """\
You are one of several independent judges on a knowledge-graph governance
panel. A new ontology type has been proposed for the shared Global-Public
layer, which EVERY tenant sees. Approve ONLY universal vocabulary — generic
concepts any organization in any industry could use (Person, Invoice, Hotel,
LoyaltyTier). Reject tenant-specific, project-specific, or benchmark
vocabulary (internal codenames, one-off dataset labels, contest categories).
When in doubt, reject: a rejected type still works for the proposing tenant
in its own layer, but a bad Global type pollutes everyone.

Respond with valid JSON only. No markdown, no explanation."""

GOV_JUDGE_USER_TEMPLATE = """\
Proposed type: "{type_name}"
Parent lineage: {parent_chain}
Proposing tenant: {tenant_id}
Proposer reasoning: {reasoning}

Should this type be admitted to the shared Global-Public ontology layer?

Return JSON:
{{
  "approve": true | false,
  "reasoning": "<one sentence>"
}}"""


# OSS governance judge model — env-overridable; default preserves prior
# behavior. The premium ShapeJudgePanel uses COGRAPH_GOV_JUDGE_MODEL (no default)
# and overrides this panel entirely when registered.
DEFAULT_GOV_JUDGE_MODEL = os.environ.get("OMNIX_GOV_JUDGE_MODEL", "claude-sonnet-4-6")


class LLMJudgePanel:
    """Default panel: N independent LLM judges, mirroring the type-matcher's
    3-judge fan-out (temperature 0.7 for diversity, asyncio.gather)."""

    def __init__(self, client, n_judges: int = 3, model: str = DEFAULT_GOV_JUDGE_MODEL):
        self._client = client
        self._n_judges = n_judges
        self._model = model

    async def judge(self, proposal: TypeProposal) -> list[JudgeVerdict]:
        async def single_judge() -> JudgeVerdict:
            msg = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                temperature=0.7,  # diversity between judges
                system=GOV_JUDGE_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": GOV_JUDGE_USER_TEMPLATE.format(
                        type_name=proposal.type_name,
                        parent_chain=proposal.parent_chain or "(top-level type)",
                        tenant_id=proposal.tenant_id,
                        reasoning=proposal.reasoning or "(none given)",
                    ),
                }],
            )
            try:
                data = json.loads(msg.content[0].text)
                return JudgeVerdict(
                    approve=bool(data.get("approve", False)),
                    reasoning=str(data.get("reasoning", "")),
                )
            except (json.JSONDecodeError, IndexError, TypeError, AttributeError):
                # Unparseable vote counts as a rejection — conservative by design.
                return JudgeVerdict(approve=False, reasoning="parse error")

        return list(await asyncio.gather(*(single_judge() for _ in range(self._n_judges))))


def governance_record_uri(type_uri: str, tenant_id: str) -> str:
    """Record node URI: one per (governed subject, proposing tenant) — keyed by
    sha1 like COG-38 statement nodes so re-judging the same proposal targets
    the same node, which governance_record_update then rewrites in place.

    Public (COG-56): the provenance-record ENCODING is OSS mechanism (ADR 0002
    boundary); the premium shape-governance writer reuses it so its Global
    writes carry the exact same record shape as OSS type governance.
    """
    rid = hashlib.sha1(f"{type_uri}|{tenant_id}".encode("utf-8")).hexdigest()
    return f"{GOV_NS}rec/{rid}"


def governance_record_update(
    subject_uri: str,
    tenant_id: str,
    proposer_model: str,
    reasoning: str,
    votes: list[JudgeVerdict],
    ts: str,
    extra: list[tuple[str, str]] | None = None,
) -> str:
    """One SPARQL update that REPLACES the governance record for
    (subject_uri, tenant_id): DELETE WHERE clears the record node's existing
    triples, then INSERT DATA writes the fresh ones — semicolon-joined so the
    rewrite is a single atomic update. Plain INSERTs would append, and
    re-judging the same proposal would accumulate stale GOV_VOTE /
    GOV_REASONING triples on the deterministic record node (COG-45).

    ``extra`` appends additional (predicate, object) pairs to the record node
    — the COG-56 seam for mapping-shape records (GOV_KIND, GOV_CONFIDENCE,
    GOV_DATASET_HINT, GOV_ALIGNED_TO) without forking the encoding.
    """
    record = governance_record_uri(subject_uri, tenant_id)
    approvals = sum(1 for v in votes if v.approve)
    triples = [
        (record, GOV_SUBJECT, subject_uri),
        (record, GOV_PROPOSER_MODEL, proposer_model),
        (record, GOV_REASONING, reasoning),
        (record, GOV_TENANT, tenant_id),
        (record, GOV_APPROVED, "true"),
        (record, GOV_VOTES, f"{approvals}/{len(votes)}"),
        (record, GOV_TIMESTAMP, f"{ts}^^{XSD}#dateTime"),
    ]
    for v in votes:
        triples.append(
            (record, GOV_VOTE, f"{'approve' if v.approve else 'reject'}: {v.reasoning}"),
        )
    for pred, obj in extra or []:
        triples.append((record, pred, obj))
    prov_g = provenance_graph_uri(public_graph_uri())
    return (
        f"DELETE WHERE {{ GRAPH <{prov_g}> {{ <{record}> ?p ?o }} }};\n"
        + insert_triples(prov_g, triples)
    )


def changelog_triples(
    action: str, subject_uri: str, tenant_id: str, ts: str,
) -> list[tuple[str, str, str]]:
    """One append-only changelog entry (fresh node — entries are never
    rewritten). Public (COG-56): reused by the premium shape-governance writer
    so Global shape writes land in the same changelog as type governance."""
    entry = f"{GOV_NS}log/{uuid4()}"
    triples = [
        (entry, GOV_ACTION, action),
        (entry, GOV_SUBJECT, subject_uri),
        (entry, GOV_TIMESTAMP, f"{ts}^^{XSD}#dateTime"),
    ]
    if tenant_id:
        triples.append((entry, GOV_TENANT, tenant_id))
    return triples


class GovernanceEngine:
    """Judges proposals and performs tagged, reversible Global-Public writes."""

    def __init__(self, neptune):
        self._neptune = neptune

    async def propose_and_judge(
        self, proposal: TypeProposal, panel: JudgePanel,
    ) -> GovernanceDecision:
        """Run the panel and decide the target layer by majority vote.

        Never raises: a failing panel degrades to the tenant layer (today's
        behavior), because ingestion must not block on governance.
        """
        try:
            votes = await panel.judge(proposal)
        except Exception as exc:
            logger.warning(
                "governance_panel_failed", type_name=proposal.type_name, error=str(exc),
            )
            votes = []
        # Strict majority of cast votes; an empty/failed panel never approves.
        approved = bool(votes) and sum(v.approve for v in votes) * 2 > len(votes)
        logger.info(
            "governance_decision",
            type_name=proposal.type_name,
            approved=approved,
            votes=[v.approve for v in votes],
        )
        return GovernanceDecision(
            target_layer="public" if approved else "tenant",
            votes=votes,
            approved=approved,
        )

    async def write_governed_type(
        self,
        proposal: TypeProposal,
        decision: GovernanceDecision,
        timestamp: datetime | None = None,
    ) -> str:
        """Insert the approved type into the Public layer, tagged and reversible.

        Three writes for the approved type: (1) the type itself in the Public
        graph — every triple's subject is the public type URI so revoke_type
        removes exactly what this created; (2) a governance-provenance record
        (proposer model, reasoning, votes, timestamp) in the Public graph's
        companion provenance graph, paralleling COG-38 — the record node is
        REPLACED, not appended to, so re-judging never accumulates stale
        votes; (3) an append-only changelog entry (ADR 0002 §8).

        If the proposal carries a parent_chain, any ancestor missing from the
        Public layer is synthesized into it as part of the same governed
        write (mirroring ADR 0001's ancestor synthesis), so the subClassOf
        edge this emits is never dangling (COG-45). Each synthesized ancestor
        gets the same trio: type triples, a provenance record whose reasoning
        marks it as ancestor synthesis for the approved child, and a
        changelog entry. No parent info — no edge. Returns the public type URI.
        """
        if not decision.approved:
            raise ValueError("write_governed_type requires an approved decision")
        ts = (timestamp or datetime.now(timezone.utc)).isoformat()
        pub_uri = layer_type_uri(Layer.PUBLIC, proposal.type_name)
        # Find what's missing from the Public lineage BEFORE writing anything,
        # so the checks never observe this write's own inserts.
        missing_ancestors = await self._missing_public_ancestors(proposal.parent_chain)

        type_triples = [
            (pub_uri, f"{RDF}#type", f"{RDFS}#Class"),
            (pub_uri, f"{RDFS}#label", proposal.type_name),
        ]
        if proposal.parent_chain:
            # Immediate parent in the Public namespace — never dangling: every
            # missing ancestor in the chain is synthesized below.
            type_triples.append(
                (pub_uri, f"{RDFS}#subClassOf", layer_type_uri(Layer.PUBLIC, proposal.parent_chain[0])),
            )
        await self._neptune.update(insert_triples(public_graph_uri(), type_triples))

        approvals = sum(1 for v in decision.votes if v.approve)
        await self._neptune.update(
            governance_record_update(
                pub_uri, proposal.tenant_id, proposal.proposer_model,
                proposal.reasoning, decision.votes, ts,
            ),
        )

        await self._neptune.update(
            insert_triples(
                changelog_graph_uri(),
                changelog_triples("add_type", pub_uri, proposal.tenant_id, ts),
            ),
        )

        # missing_ancestors is a prefix of parent_chain, so its indexes are
        # chain indexes — each synthesized ancestor links the next chain entry.
        for i in range(len(missing_ancestors)):
            await self._synthesize_ancestor(proposal, decision, i, ts)

        logger.info("governance_type_written", type_uri=pub_uri, votes=f"{approvals}/{len(decision.votes)}")
        return pub_uri

    async def _missing_public_ancestors(self, parent_chain: list[str]) -> list[str]:
        """The leading run of parent_chain names absent from the Public layer.

        Walks the chain root-ward and stops at the first ancestor that already
        exists — its own lineage was closed when IT was governed in, so
        nothing above it can be dangling. Missing names come back in chain
        order (immediate parent first), i.e. always a prefix of parent_chain.
        """
        missing: list[str] = []
        for name in parent_chain:
            uri = layer_type_uri(Layer.PUBLIC, name)
            if await self._neptune.ask(entity_exists_query(public_graph_uri(), uri)):
                break
            missing.append(name)
        return missing

    async def _synthesize_ancestor(
        self,
        proposal: TypeProposal,
        decision: GovernanceDecision,
        chain_index: int,
        ts: str,
    ) -> None:
        """Synthesize parent_chain[chain_index] into the Public layer.

        Same trio of writes as the approved child: type triples (with a
        subClassOf edge to the next chain entry's Public URI, when one
        exists — the root of the chain gets no edge), a provenance record
        whose reasoning marks the ancestor synthesis, and a changelog entry.
        The record reuses the child's panel votes: approval of the child is
        what admits its lineage.
        """
        name = proposal.parent_chain[chain_index]
        anc_uri = layer_type_uri(Layer.PUBLIC, name)
        anc_triples = [
            (anc_uri, f"{RDF}#type", f"{RDFS}#Class"),
            (anc_uri, f"{RDFS}#label", name),
        ]
        if chain_index + 1 < len(proposal.parent_chain):
            anc_triples.append(
                (anc_uri, f"{RDFS}#subClassOf",
                 layer_type_uri(Layer.PUBLIC, proposal.parent_chain[chain_index + 1])),
            )
        await self._neptune.update(insert_triples(public_graph_uri(), anc_triples))

        await self._neptune.update(
            governance_record_update(
                anc_uri, proposal.tenant_id, proposal.proposer_model,
                f"Ancestor synthesis: missing Public-layer ancestor of approved type "
                f"'{proposal.type_name}' (closes the subClassOf lineage, COG-45)",
                decision.votes, ts,
            ),
        )

        await self._neptune.update(
            insert_triples(
                changelog_graph_uri(),
                changelog_triples("add_type", anc_uri, proposal.tenant_id, ts),
            ),
        )
        logger.info(
            "governance_ancestor_synthesized", type_uri=anc_uri, child=proposal.type_name,
        )

    async def revoke_type(self, type_uri: str, timestamp: datetime | None = None) -> None:
        """Engine-shaped wrapper around the module-level revoke_type."""
        await revoke_type(self._neptune, type_uri, timestamp=timestamp)


async def revoke_type(neptune, type_uri: str, timestamp: datetime | None = None) -> None:
    """Rip a governed type out of the Public layer wholesale (reversibility,
    ADR 0002 §2).

    Removes (1) every Public-graph triple with the type as subject OR object
    (the type node plus any subClassOf edges pointing at it) and (2) its
    governance record(s). The changelog is append-only — the add_type entry
    stays for audit and a revoke_type entry is appended instead.
    """
    g = public_graph_uri()
    await neptune.update(
        f"DELETE {{ GRAPH <{g}> {{ ?s ?p ?o }} }}\n"
        f"WHERE {{ GRAPH <{g}> {{ ?s ?p ?o . "
        f"FILTER(?s = <{type_uri}> || ?o = <{type_uri}>) }} }}"
    )
    prov_g = provenance_graph_uri(g)
    await neptune.update(
        f"DELETE {{ GRAPH <{prov_g}> {{ ?node ?p ?o }} }}\n"
        f"WHERE {{ GRAPH <{prov_g}> {{ ?node <{GOV_SUBJECT}> <{type_uri}> . ?node ?p ?o }} }}"
    )
    ts = (timestamp or datetime.now(timezone.utc)).isoformat()
    await neptune.update(
        insert_triples(changelog_graph_uri(), changelog_triples("revoke_type", type_uri, "", ts)),
    )
    logger.info("governance_type_revoked", type_uri=type_uri)


# ---------------------------------------------------------------------------
# Mapping-shape governance (ADR 0003 §5, COG-56)
#
# ADR 0002 §2 gates TYPE NAMES entering shared layers; ADR 0003 extends the
# same pipeline to MAPPING SHAPE: dependent-entity promotions, core-slot
# proposals, dataset constants, and any low-confidence (<0.7) reason-pass
# decision are judge-panel material — held, voted, provenance-tagged,
# reversible; never auto-committed to shared layers.
#
# OSS owns the SEAM only: the proposal data structures, the extraction of
# proposals from a posted CSVSchemaMapping, the fire-and-forget enqueue off
# the /ingest/csv/rows path, and the registration protocol
# (register_governance_panel). The default panel is a no-op holder that just
# records pending proposals — tenant-layer-only behavior, exactly what the
# pre-registration already wrote. The judge-panel SERVICE, the Global-Public
# shape write path, alignment to approved canonical shapes, and entitlement
# gating are premium and plug in through ShapeGovernancePanel.
#
# Tenant-layer fallback keeps ingestion non-blocking (ADR 0002 §2): the
# tenant uses the shape immediately whatever the panel later decides; the
# panel only decides whether the shape is universal.
# ---------------------------------------------------------------------------

# Reason-pass decisions below this confidence are judge-panel material
# (mirrors the held_for_review threshold on TypeExtension/CoreSlot).
LOW_CONFIDENCE_THRESHOLD = 0.7

ShapeProposalKind = Literal[
    "promotion", "core_slot", "dataset_constant", "low_confidence_decision",
]


class MappingShapeProposal(BaseModel):
    """One mapping-shape decision submitted for judge-panel review.

    Carries the Pass D ``TypeExtension`` payload (when the decision is
    completion-pass output) plus source context: proposing tenant, dataset
    hint, the model's confidence and written reasoning, and — for promotions —
    the host type the attribute was promoted from (the alignment anchor for
    the canonical dependent-identifier shape's ``identifies`` edge).
    """

    kind: ShapeProposalKind
    subject: str = Field(
        description=(
            "What is proposed: the promoted type name, 'Type.slot' for "
            "core slots / dataset constants, or 'entity:<name>' / "
            "'column:<name>' for low-confidence reason-pass decisions"
        ),
    )
    tenant_id: str
    dataset_hint: str = Field(
        default="", description="ingest source / kg name — which dataset produced the proposal",
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    reasoning: str = Field(default="", description="the proposer model's written why")
    proposer_model: str = ""
    extension: TypeExtension | None = Field(
        default=None,
        description="full Pass D payload for promotion/core-slot/dataset-constant kinds",
    )
    host_type: str | None = Field(
        default=None,
        description="type owning the promoted-from attribute (promotions only)",
    )
    slot_name: str | None = Field(
        default=None, description="which core slot, for core_slot/dataset_constant kinds",
    )


class ShapeGovernancePanel(Protocol):
    """Pluggable mapping-shape panel — the premium judge service implements
    this and registers via register_governance_panel. submit() runs on a
    background task, never on the request path, and may take as long as it
    likes; exceptions are logged and swallowed by the seam."""

    async def submit(self, proposal: MappingShapeProposal) -> None: ...


class PendingShapeProposals:
    """OSS default panel: records pending proposals and judges nothing.

    Tenant-layer-only behavior — the tenant-layer writes already happened in
    the /ingest/csv/rows pre-registration; with no premium service registered
    the proposals simply accumulate here (bounded ring buffer) for inspection.
    """

    def __init__(self, max_pending: int = 1000):
        self._pending: deque[MappingShapeProposal] = deque(maxlen=max_pending)

    async def submit(self, proposal: MappingShapeProposal) -> None:
        self._pending.append(proposal)
        logger.info(
            "shape_proposal_pending",
            kind=proposal.kind,
            subject=proposal.subject,
            tenant=proposal.tenant_id,
            confidence=proposal.confidence,
        )

    def pending(self) -> list[MappingShapeProposal]:
        return list(self._pending)

    def clear(self) -> None:
        self._pending.clear()


#: Module singleton — the default holder proposals land in when no premium
#: panel is registered (and the place tests inspect pending proposals).
pending_shape_proposals = PendingShapeProposals()

_shape_panel: ShapeGovernancePanel | None = None


def register_governance_panel(panel: ShapeGovernancePanel | None) -> None:
    """Register (or clear, with None) the mapping-shape governance panel.

    Same plugin-protocol style as register_external_verifier /
    register_adapter: the premium judge-panel service calls this at app
    startup; OSS deployments never do and fall back to the pending-proposal
    holder (tenant-layer-only behavior).
    """
    global _shape_panel
    _shape_panel = panel
    logger.info(
        "shape_governance_panel_registered",
        panel=type(panel).__name__ if panel is not None else None,
    )


def governance_panel() -> ShapeGovernancePanel:
    """The registered panel, or the OSS default pending-proposal holder."""
    return _shape_panel if _shape_panel is not None else pending_shape_proposals


def _low_confidence(confidence: float | None) -> bool:
    return confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD


def mapping_shape_proposals(
    mapping: CSVSchemaMapping,
    tenant_id: str,
    *,
    dataset_hint: str = "",
    proposer_model: str = "",
) -> list[MappingShapeProposal]:
    """Extract every judge-panel-material decision from a posted mapping.

    - every dependent-entity PROMOTION (one proposal carrying the whole
      TypeExtension — the panel judges the shape as a unit);
    - every CORE SLOT added to a pre-existing (non-promoted) type;
    - every DATASET CONSTANT (its own judgement, whatever type it rides on);
    - every reason-pass decision (entity / column) with confidence < 0.7.

    High-confidence reason-pass decisions and legacy mappings (no v2 fields)
    yield nothing — auto-commit to the tenant layer stays exactly as is.
    """
    common = dict(
        tenant_id=tenant_id, dataset_hint=dataset_hint, proposer_model=proposer_model,
    )
    proposals: list[MappingShapeProposal] = []

    # Which type owns each column — the promotion's host type (the target of
    # the canonical shape's `identifies` edge) is the type that owned the
    # promoted-from attribute.
    entity_type_by_name = {e.name: e.type_name for e in (mapping.entities or [])}

    def _host_type(promoted_from: str) -> str | None:
        for col in mapping.columns:
            if (col.attribute_name or col.column_name) == promoted_from:
                if col.entity and col.entity in entity_type_by_name:
                    return entity_type_by_name[col.entity]
                return mapping.entity_type or None
        return mapping.entity_type or None

    for ext in (mapping.ontology_extensions.types if mapping.ontology_extensions else []):
        if ext.promoted_from_attribute:
            why = next((s.why for s in ext.core_slots if s.why), None)
            proposals.append(MappingShapeProposal(
                kind="promotion",
                subject=ext.type_name,
                confidence=ext.confidence,
                reasoning=why or (
                    f"Dependent entity promoted from attribute "
                    f"'{ext.promoted_from_attribute}' (ADR 0003 Pass D)"
                ),
                extension=ext,
                host_type=_host_type(ext.promoted_from_attribute),
                **common,
            ))
        else:
            for slot in ext.core_slots:
                proposals.append(MappingShapeProposal(
                    kind="core_slot",
                    subject=f"{ext.type_name}.{slot.name}",
                    confidence=slot.confidence,
                    reasoning=slot.why or "",
                    extension=ext,
                    slot_name=slot.name,
                    **common,
                ))
        for slot in ext.core_slots:
            if slot.dataset_constant is not None:
                proposals.append(MappingShapeProposal(
                    kind="dataset_constant",
                    subject=f"{ext.type_name}.{slot.name}",
                    confidence=slot.dataset_constant.confidence,
                    reasoning=(
                        f"Dataset context implies the single value "
                        f"'{slot.dataset_constant.value}' for this core slot"
                        + (f": {slot.why}" if slot.why else "")
                    ),
                    extension=ext,
                    slot_name=slot.name,
                    **common,
                ))

    for spec in mapping.entities or []:
        if _low_confidence(spec.confidence):
            proposals.append(MappingShapeProposal(
                kind="low_confidence_decision",
                subject=f"entity:{spec.name}",
                confidence=spec.confidence,
                reasoning=spec.why or "",
                **common,
            ))
    for col in mapping.columns:
        if _low_confidence(col.confidence):
            proposals.append(MappingShapeProposal(
                kind="low_confidence_decision",
                subject=f"column:{col.column_name}",
                confidence=col.confidence,
                reasoning=col.why or "",
                **common,
            ))
    return proposals


# Background submission tasks — referenced here so drain_shape_governance()
# can await them deterministically (same pattern as SchemaResolver's
# _governance_tasks, COG-46), and so the tasks are never garbage-collected
# mid-flight.
_shape_tasks: list[asyncio.Task] = []


def enqueue_shape_proposals(proposals: list[MappingShapeProposal]) -> int:
    """Fire-and-forget: schedule submission of proposals to the registered
    panel as ONE background task and return immediately.

    Called from the /ingest/csv/rows path AFTER the tenant-layer writes
    succeed — the request never awaits the panel (an LLM judge panel can take
    seconds; ingest latency must not change). Never raises: scheduling
    failures are logged and degrade to tenant-layer-only behavior. Returns
    the number of proposals scheduled (0 when there is nothing to submit or
    scheduling failed).
    """
    if not proposals:
        return 0
    try:
        _shape_tasks[:] = [t for t in _shape_tasks if not t.done()]
        _shape_tasks.append(
            asyncio.create_task(_submit_shape_proposals(list(proposals)))
        )
    except Exception:
        logger.warning("shape_governance_enqueue_failed", exc_info=True)
        return 0
    return len(proposals)


async def _submit_shape_proposals(proposals: list[MappingShapeProposal]) -> None:
    """Submit each proposal to the panel, inside the background task.

    Panel exceptions are logged and swallowed per proposal — a failing
    premium service must never crash the task or affect ingest.
    """
    panel = governance_panel()
    for proposal in proposals:
        try:
            await panel.submit(proposal)
        except Exception:
            logger.warning(
                "shape_proposal_submit_failed",
                kind=proposal.kind,
                subject=proposal.subject,
                exc_info=True,
            )


async def drain_shape_governance() -> None:
    """Await all pending background shape-governance tasks.

    Mapping-shape governance is eventually consistent (the panel runs after
    /ingest/csv/rows has returned); call this to deterministically wait for
    every scheduled submission — tests, and shutdown paths. Safe to call any
    time; task failures were logged in-task and are never re-raised.
    """
    tasks = list(_shape_tasks)
    _shape_tasks.clear()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
