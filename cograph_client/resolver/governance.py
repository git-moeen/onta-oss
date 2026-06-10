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
from datetime import datetime, timezone
from typing import Literal, Protocol
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from cograph_client.graph.layers import Layer, layer_type_uri, public_graph_uri
from cograph_client.graph.ontology_queries import RDF, RDFS, XSD, entity_exists_query
from cograph_client.graph.provenance import provenance_graph_uri
from cograph_client.graph.queries import insert_triples

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


class LLMJudgePanel:
    """Default panel: N independent LLM judges, mirroring the type-matcher's
    3-judge fan-out (temperature 0.7 for diversity, asyncio.gather)."""

    def __init__(self, client, n_judges: int = 3, model: str = "claude-sonnet-4-6"):
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


def _governance_record_uri(type_uri: str, tenant_id: str) -> str:
    """Record node URI: one per (governed type, proposing tenant) — keyed by
    sha1 like COG-38 statement nodes so re-judging the same proposal targets
    the same node, which _governance_record_update then rewrites in place."""
    rid = hashlib.sha1(f"{type_uri}|{tenant_id}".encode("utf-8")).hexdigest()
    return f"{GOV_NS}rec/{rid}"


def _governance_record_update(
    subject_uri: str,
    tenant_id: str,
    proposer_model: str,
    reasoning: str,
    votes: list[JudgeVerdict],
    ts: str,
) -> str:
    """One SPARQL update that REPLACES the governance record for
    (subject_uri, tenant_id): DELETE WHERE clears the record node's existing
    triples, then INSERT DATA writes the fresh ones — semicolon-joined so the
    rewrite is a single atomic update. Plain INSERTs would append, and
    re-judging the same proposal would accumulate stale GOV_VOTE /
    GOV_REASONING triples on the deterministic record node (COG-45)."""
    record = _governance_record_uri(subject_uri, tenant_id)
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
    prov_g = provenance_graph_uri(public_graph_uri())
    return (
        f"DELETE WHERE {{ GRAPH <{prov_g}> {{ <{record}> ?p ?o }} }};\n"
        + insert_triples(prov_g, triples)
    )


def _changelog_triples(
    action: str, subject_uri: str, tenant_id: str, ts: str,
) -> list[tuple[str, str, str]]:
    """One append-only changelog entry (fresh node — entries are never rewritten)."""
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
            _governance_record_update(
                pub_uri, proposal.tenant_id, proposal.proposer_model,
                proposal.reasoning, decision.votes, ts,
            ),
        )

        await self._neptune.update(
            insert_triples(
                changelog_graph_uri(),
                _changelog_triples("add_type", pub_uri, proposal.tenant_id, ts),
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
            _governance_record_update(
                anc_uri, proposal.tenant_id, proposal.proposer_model,
                f"Ancestor synthesis: missing Public-layer ancestor of approved type "
                f"'{proposal.type_name}' (closes the subClassOf lineage, COG-45)",
                decision.votes, ts,
            ),
        )

        await self._neptune.update(
            insert_triples(
                changelog_graph_uri(),
                _changelog_triples("add_type", anc_uri, proposal.tenant_id, ts),
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
        insert_triples(changelog_graph_uri(), _changelog_triples("revoke_type", type_uri, "", ts)),
    )
    logger.info("governance_type_revoked", type_uri=type_uri)
