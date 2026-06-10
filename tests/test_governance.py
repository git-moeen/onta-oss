"""Governance pipeline seam tests (ADR 0002 §2, COG-43).

Covers the judge panel (N independent mocked LLM judges, majority vote,
parse-error votes count as rejections), the GovernanceEngine decision +
tagged reversible Public-layer writes (type, provenance record, append-only
changelog), revoke reversibility, and the resolver wiring:
COGRAPH_GOVERNANCE_ENABLED off (the default) must be byte-identical to
today's new-type behavior; on, a majority-approved type ALSO lands in the
Public layer while a rejected one stays tenant-only — and governance
failures never block ingest.

All mocked — no live Neptune, no LLM, no network. Env is only touched via
patch.dict (auto-restored), never process-globally.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.layers import Layer, layer_type_uri, public_graph_uri
from cograph_client.graph.ontology_queries import type_uri
from cograph_client.graph.provenance import provenance_graph_uri
from cograph_client.resolver.governance import (
    GOV_NS,
    GovernanceDecision,
    GovernanceEngine,
    JudgeVerdict,
    LLMJudgePanel,
    TypeProposal,
    changelog_graph_uri,
    revoke_type,
)
from cograph_client.resolver.models import ExtractedEntity, IngestResult
from cograph_client.resolver.schema_resolver import SchemaResolver


FIXED_TS = datetime(2026, 6, 9, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def mock_neptune():
    client = AsyncMock(spec=NeptuneClient)
    client.health.return_value = True
    client.query.return_value = {"head": {"vars": []}, "results": {"bindings": []}}
    client.update.return_value = None
    return client


def _proposal(**overrides) -> TypeProposal:
    fields = dict(
        type_name="LoyaltyTier",
        parent_chain=["Tier"],
        tenant_id="acme",
        reasoning="Generic hospitality vocabulary",
        proposer_model="test-model",
    )
    fields.update(overrides)
    return TypeProposal(**fields)


def _verdicts(*approvals: bool) -> list[JudgeVerdict]:
    return [JudgeVerdict(approve=a, reasoning=f"vote-{i}") for i, a in enumerate(approvals)]


class StubPanel:
    """Minimal JudgePanel impl — also exercises the Protocol seam."""

    def __init__(self, verdicts):
        self._verdicts = verdicts

    async def judge(self, proposal: TypeProposal) -> list[JudgeVerdict]:
        if isinstance(self._verdicts, Exception):
            raise self._verdicts
        return self._verdicts


def _update_sparql(mock_neptune) -> list[str]:
    return [c.args[0] for c in mock_neptune.update.call_args_list]


# ---------------------------------------------------------------------------
# LLMJudgePanel — N independent judges, mocked client
# ---------------------------------------------------------------------------


def _mock_anthropic(reply: str):
    client = AsyncMock()
    msg = MagicMock()
    msg.content = [MagicMock(text=reply)]
    client.messages.create.return_value = msg
    return client


@pytest.mark.asyncio
async def test_llm_judge_panel_fans_out_n_independent_calls():
    client = _mock_anthropic('{"approve": true, "reasoning": "universal"}')
    panel = LLMJudgePanel(client, n_judges=3)

    verdicts = await panel.judge(_proposal())

    assert client.messages.create.call_count == 3
    assert len(verdicts) == 3
    assert all(v.approve for v in verdicts)
    assert all(v.reasoning == "universal" for v in verdicts)
    # The proposal's content reaches the judges.
    prompt = client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "LoyaltyTier" in prompt and "acme" in prompt


@pytest.mark.asyncio
async def test_llm_judge_panel_parse_error_counts_as_rejection():
    panel = LLMJudgePanel(_mock_anthropic("not json at all"), n_judges=3)
    verdicts = await panel.judge(_proposal())
    assert len(verdicts) == 3
    assert all(not v.approve for v in verdicts)


# ---------------------------------------------------------------------------
# propose_and_judge — majority vote, never raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_majority_approve_targets_public_layer(mock_neptune):
    engine = GovernanceEngine(mock_neptune)
    decision = await engine.propose_and_judge(_proposal(), StubPanel(_verdicts(True, True, False)))
    assert decision.approved is True
    assert decision.target_layer == "public"
    assert [v.approve for v in decision.votes] == [True, True, False]


@pytest.mark.asyncio
@pytest.mark.parametrize("votes", [
    (False, False, True),  # majority reject
    (True, False),         # tie is NOT a majority
    (),                    # empty panel never approves
])
async def test_no_majority_targets_tenant_layer(mock_neptune, votes):
    engine = GovernanceEngine(mock_neptune)
    decision = await engine.propose_and_judge(_proposal(), StubPanel(_verdicts(*votes)))
    assert decision.approved is False
    assert decision.target_layer == "tenant"


@pytest.mark.asyncio
async def test_panel_failure_degrades_to_tenant_never_raises(mock_neptune):
    engine = GovernanceEngine(mock_neptune)
    decision = await engine.propose_and_judge(
        _proposal(), StubPanel(RuntimeError("judge service down")),
    )
    assert decision.approved is False
    assert decision.target_layer == "tenant"
    assert decision.votes == []


# ---------------------------------------------------------------------------
# write_governed_type — Public-layer write with provenance + changelog
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_governed_type_emits_type_provenance_and_changelog(mock_neptune):
    engine = GovernanceEngine(mock_neptune)
    proposal = _proposal()
    decision = GovernanceDecision(
        target_layer="public", votes=_verdicts(True, True, False), approved=True,
    )

    pub_uri = await engine.write_governed_type(proposal, decision, timestamp=FIXED_TS)

    assert pub_uri == layer_type_uri(Layer.PUBLIC, "LoyaltyTier")
    calls = _update_sparql(mock_neptune)
    assert len(calls) == 3

    # 1) Type triples in the Public layer graph, Public namespace.
    type_sparql = calls[0]
    assert f"GRAPH <{public_graph_uri()}>" in type_sparql
    assert f"<{pub_uri}>" in type_sparql
    assert "#Class>" in type_sparql and '"LoyaltyTier"' in type_sparql
    # Immediate parent linked in the PUBLIC namespace, not the tenant one.
    assert f"<{layer_type_uri(Layer.PUBLIC, 'Tier')}>" in type_sparql
    assert f"<{type_uri('Tier')}>" not in type_sparql

    # 2) Governance provenance in the Public graph's companion provenance graph.
    gov_sparql = calls[1]
    assert f"GRAPH <{provenance_graph_uri(public_graph_uri())}>" in gov_sparql
    assert '"test-model"' in gov_sparql
    assert "Generic hospitality vocabulary" in gov_sparql
    assert '"2/3"' in gov_sparql
    assert "2026-06-09T12:00:00+00:00" in gov_sparql
    assert '"acme"' in gov_sparql
    # Per-judge votes recorded for audit.
    assert "approve: vote-0" in gov_sparql and "reject: vote-2" in gov_sparql

    # 3) Append-only changelog entry.
    log_sparql = calls[2]
    assert f"GRAPH <{changelog_graph_uri()}>" in log_sparql
    assert '"add_type"' in log_sparql and f"<{pub_uri}>" in log_sparql


@pytest.mark.asyncio
async def test_write_governed_type_rejects_unapproved_decision(mock_neptune):
    engine = GovernanceEngine(mock_neptune)
    decision = GovernanceDecision(target_layer="tenant", votes=_verdicts(False, False), approved=False)
    with pytest.raises(ValueError):
        await engine.write_governed_type(_proposal(), decision, timestamp=FIXED_TS)
    assert mock_neptune.update.call_count == 0


@pytest.mark.asyncio
async def test_top_level_proposal_writes_no_subclass_edge(mock_neptune):
    engine = GovernanceEngine(mock_neptune)
    decision = GovernanceDecision(target_layer="public", votes=_verdicts(True, True), approved=True)
    await engine.write_governed_type(_proposal(parent_chain=[]), decision, timestamp=FIXED_TS)
    assert "subClassOf" not in _update_sparql(mock_neptune)[0]


# ---------------------------------------------------------------------------
# revoke_type — reversibility: removes what write created, changelog stays
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_revoke_removes_what_write_created(mock_neptune):
    engine = GovernanceEngine(mock_neptune)
    decision = GovernanceDecision(target_layer="public", votes=_verdicts(True, True, False), approved=True)
    pub_uri = await engine.write_governed_type(_proposal(), decision, timestamp=FIXED_TS)
    mock_neptune.reset_mock()

    await revoke_type(mock_neptune, pub_uri, timestamp=FIXED_TS)

    calls = _update_sparql(mock_neptune)
    assert len(calls) == 3
    # 1) Every Public-graph triple with the type as subject OR object is gone —
    #    the write only created triples whose subject is pub_uri, so this
    #    covers exactly what the write created (plus dangling edges INTO it).
    assert calls[0].startswith("DELETE")
    assert f"GRAPH <{public_graph_uri()}>" in calls[0]
    assert f"?s = <{pub_uri}>" in calls[0] and f"?o = <{pub_uri}>" in calls[0]
    # 2) The governance record(s) keyed to this type are gone too.
    assert calls[1].startswith("DELETE")
    assert f"GRAPH <{provenance_graph_uri(public_graph_uri())}>" in calls[1]
    assert f"<{GOV_NS}subject> <{pub_uri}>" in calls[1]
    # 3) The changelog is append-only: a revoke entry is INSERTed, nothing deleted.
    assert calls[2].startswith("INSERT")
    assert f"GRAPH <{changelog_graph_uri()}>" in calls[2]
    assert '"revoke_type"' in calls[2] and f"<{pub_uri}>" in calls[2]


@pytest.mark.asyncio
async def test_engine_revoke_method_delegates(mock_neptune):
    engine = GovernanceEngine(mock_neptune)
    await engine.revoke_type(layer_type_uri(Layer.PUBLIC, "LoyaltyTier"), timestamp=FIXED_TS)
    assert mock_neptune.update.call_count == 3


# ---------------------------------------------------------------------------
# Resolver wiring — flag off (regression) / flag on
# ---------------------------------------------------------------------------


def _make_resolver(mock_neptune, governance: bool) -> SchemaResolver:
    verdict_path = Path(tempfile.mkdtemp()) / "verdicts.json"
    from cograph_client.resolver.verdict_cache import JsonVerdictCache

    env = {
        "ANTHROPIC_API_KEY": "test-key",
        "OPENROUTER_API_KEY": "test-or-key",
        "COGRAPH_ER_ENABLED": "0",
    }
    if governance:
        env["COGRAPH_GOVERNANCE_ENABLED"] = "1"
    with patch.dict("os.environ", env):
        return SchemaResolver(
            neptune=mock_neptune,
            anthropic_key="test-key",
            verdict_cache=JsonVerdictCache(verdict_path),
        )


def _new_entity() -> ExtractedEntity:
    # Empty ontology makes TypeMatcher short-circuit (no LLM call), so the
    # brand-new-type path of _resolve_type runs end-to-end.
    return ExtractedEntity(type_name="LoyaltyTier", id="gold")


TENANT_GRAPH = "https://cograph.tech/graphs/acme"


@pytest.mark.asyncio
async def test_flag_off_new_type_path_identical_regression(mock_neptune):
    """Default (flag unset): exactly one tenant-graph insert_type call — the
    pre-COG-43 behavior — and zero Public-layer / governance writes."""
    resolver = _make_resolver(mock_neptune, governance=False)
    assert resolver._governance_enabled is False
    result = IngestResult(entities_extracted=1)

    resolved = await resolver._resolve_type(_new_entity(), TENANT_GRAPH, {}, {}, result)

    assert resolved == "LoyaltyTier"
    calls = _update_sparql(mock_neptune)
    assert len(calls) == 1
    assert f"GRAPH <{TENANT_GRAPH}>" in calls[0]
    assert f"<{type_uri('LoyaltyTier')}>" in calls[0]
    assert public_graph_uri() not in calls[0] and GOV_NS not in calls[0]


@pytest.mark.asyncio
async def test_flag_on_majority_approve_also_writes_public_layer(mock_neptune):
    """Flag on + majority approve: tenant insert_type happens synchronously
    (today's behavior, ingestion never waits on governance for usability);
    the judge panel + governed Public-layer copy + provenance + changelog run
    as a BACKGROUND task (COG-46) awaited via drain_governance()."""
    resolver = _make_resolver(mock_neptune, governance=True)
    resolver._judge_panel = StubPanel(_verdicts(True, True, False))
    result = IngestResult(entities_extracted=1)

    resolved = await resolver._resolve_type(_new_entity(), TENANT_GRAPH, {}, {}, result)

    assert resolved == "LoyaltyTier"
    # The ingest path returned after the tenant write alone — governance is
    # scheduled, retained on the resolver, and not yet (necessarily) done.
    assert len(_update_sparql(mock_neptune)) == 1
    assert len(resolver._governance_tasks) == 1

    await resolver.drain_governance()

    assert resolver._governance_tasks == []
    calls = _update_sparql(mock_neptune)
    assert len(calls) == 4
    # Tenant write happens first and is unchanged.
    assert f"GRAPH <{TENANT_GRAPH}>" in calls[0]
    assert f"<{type_uri('LoyaltyTier')}>" in calls[0]
    # Then the governed Public-layer copy with provenance + changelog.
    assert f"GRAPH <{public_graph_uri()}>" in calls[1]
    assert f"<{layer_type_uri(Layer.PUBLIC, 'LoyaltyTier')}>" in calls[1]
    assert f"GRAPH <{provenance_graph_uri(public_graph_uri())}>" in calls[2]
    assert '"2/3"' in calls[2]
    # The proposal carries the tenant id (derived from the graph) and the
    # proposer model (the extract model).
    assert '"acme"' in calls[2]
    assert resolver.EXTRACT_MODEL in calls[2]
    assert f"GRAPH <{changelog_graph_uri()}>" in calls[3] and '"add_type"' in calls[3]


@pytest.mark.asyncio
async def test_flag_on_majority_reject_stays_tenant_only(mock_neptune):
    resolver = _make_resolver(mock_neptune, governance=True)
    resolver._judge_panel = StubPanel(_verdicts(False, False, True))
    result = IngestResult(entities_extracted=1)

    resolved = await resolver._resolve_type(_new_entity(), TENANT_GRAPH, {}, {}, result)
    await resolver.drain_governance()

    assert resolved == "LoyaltyTier"
    calls = _update_sparql(mock_neptune)
    assert len(calls) == 1, "rejected proposal must add NO writes beyond the tenant insert"
    assert f"GRAPH <{TENANT_GRAPH}>" in calls[0]


@pytest.mark.asyncio
async def test_flag_on_governance_write_failure_never_blocks_ingest(mock_neptune):
    """A Public-layer write blowing up in the background task is logged and
    swallowed — the type is still created in the tenant layer, _resolve_type
    returns normally, and drain_governance() never re-raises."""
    resolver = _make_resolver(mock_neptune, governance=True)
    resolver._judge_panel = StubPanel(_verdicts(True, True, True))
    resolver._governance.write_governed_type = AsyncMock(side_effect=RuntimeError("neptune down"))
    result = IngestResult(entities_extracted=1)

    resolved = await resolver._resolve_type(_new_entity(), TENANT_GRAPH, {}, {}, result)
    await resolver.drain_governance()

    assert resolved == "LoyaltyTier"
    assert "LoyaltyTier" in result.types_created
    assert mock_neptune.update.call_count == 1  # the tenant insert_type only


@pytest.mark.asyncio
async def test_drain_governance_safe_with_nothing_pending(mock_neptune):
    """drain_governance() is a no-op with no scheduled tasks — flag on or off."""
    for governance in (False, True):
        resolver = _make_resolver(mock_neptune, governance=governance)
        await resolver.drain_governance()
        assert resolver._governance_tasks == []
