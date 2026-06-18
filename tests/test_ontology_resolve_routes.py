"""Route tests for the NL ontology-evolution endpoints (COG-81).

Deterministic and offline: `OntologyResolver.resolve` is monkeypatched to return
a fixed plan, and the NeptuneClient is the `mock_neptune` AsyncMock fixture, so no
LLM / embedding / Neptune call is made. We assert what SPARQL the route writes.
"""

from unittest.mock import AsyncMock

import cograph_client.api.routes.ontology as onto_routes
from cograph_client.models.ontology import ResolutionResult, ResolvedChange


def _patch_resolver(monkeypatch, result: ResolutionResult):
    """Make the route build a resolver whose `.resolve` returns `result`."""
    fake = AsyncMock()
    fake.resolve = AsyncMock(return_value=result)
    monkeypatch.setattr(onto_routes, "_build_resolver", lambda graph_uri: fake)
    return fake


def test_resolve_auto_applies_confident_change(client, auth_headers, mock_neptune, monkeypatch):
    applied = ResolvedChange(
        kind="attribute",
        subject_type="Person",
        name="age",
        datatype_or_target="integer",
        action="extend",
        confidence=0.95,
        reason="clear extend on existing Person",
    )
    _patch_resolver(monkeypatch, ResolutionResult(applied=[applied], proposals=[], summary="1 applied"))

    resp = client.post(
        "/graphs/test-tenant/ontology/resolve",
        headers=auth_headers,
        json={"ask": "track how old a person is"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["applied"]) == 1
    assert data["applied"][0]["name"] == "age"
    assert data["proposals"] == []
    # An attribute extend writes exactly one upsert_attribute statement.
    assert mock_neptune.update.call_count == 1
    sent = mock_neptune.update.call_args[0][0]
    assert "age" in sent and "integer" in sent


def test_resolve_with_only_proposals_writes_nothing(client, auth_headers, mock_neptune, monkeypatch):
    proposal = ResolvedChange(
        kind="relationship",
        subject_type="Person",
        name="works_at",
        datatype_or_target="Company",
        action="create",
        confidence=0.4,
        reason="new target type Company",
    )
    _patch_resolver(monkeypatch, ResolutionResult(applied=[], proposals=[proposal], summary="1 proposal"))

    resp = client.post(
        "/graphs/test-tenant/ontology/resolve",
        headers=auth_headers,
        json={"ask": "track which company a person works for"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["applied"] == []
    assert len(data["proposals"]) == 1
    assert data["proposals"][0]["name"] == "works_at"
    # Proposals are NOT auto-applied — nothing is written.
    assert mock_neptune.update.call_count == 0


def test_resolve_dry_run_returns_everything_as_proposals_and_writes_nothing(
    client, auth_headers, mock_neptune, monkeypatch
):
    """dry_run=True: the would-be-applied change AND the proposals all come back
    under `proposals`, `applied` is empty, `dry_run` is echoed, and ZERO writes
    hit Neptune."""
    applied = ResolvedChange(
        kind="attribute",
        subject_type="Person",
        name="age",
        datatype_or_target="integer",
        action="extend",
        confidence=0.95,
        reason="clear extend on existing Person",
    )
    proposal = ResolvedChange(
        kind="relationship",
        subject_type="Person",
        name="works_at",
        datatype_or_target="Company",
        action="create",
        confidence=0.4,
        reason="new target type Company",
    )
    _patch_resolver(
        monkeypatch,
        ResolutionResult(applied=[applied], proposals=[proposal], summary="1 applied, 1 proposal"),
    )

    resp = client.post(
        "/graphs/test-tenant/ontology/resolve",
        headers=auth_headers,
        json={"ask": "track a person's age and employer", "dry_run": True},
    )

    assert resp.status_code == 200
    data = resp.json()
    # Plan-only: applied is empty, everything folded into proposals.
    assert data["applied"] == []
    assert data["dry_run"] is True
    names = {p["name"] for p in data["proposals"]}
    assert names == {"age", "works_at"}
    # Nothing was written to Neptune (zero update calls).
    assert mock_neptune.update.call_count == 0


def test_resolve_default_omits_dry_run_and_still_auto_applies(
    client, auth_headers, mock_neptune, monkeypatch
):
    """Default (dry_run unset) is byte-for-byte the prior behavior: the confident
    change auto-applies and `dry_run` defaults to False in the response."""
    applied = ResolvedChange(
        kind="attribute",
        subject_type="Person",
        name="age",
        datatype_or_target="integer",
        action="extend",
        confidence=0.95,
        reason="clear extend on existing Person",
    )
    _patch_resolver(monkeypatch, ResolutionResult(applied=[applied], proposals=[], summary="1 applied"))

    resp = client.post(
        "/graphs/test-tenant/ontology/resolve",
        headers=auth_headers,
        json={"ask": "track how old a person is"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["applied"]) == 1
    assert data["dry_run"] is False
    # The confident change is still auto-applied (one upsert write).
    assert mock_neptune.update.call_count == 1


def test_apply_create_relationship_mints_target_and_property(client, auth_headers, mock_neptune):
    proposal = {
        "kind": "relationship",
        "subject_type": "Person",
        "name": "works_at",
        "datatype_or_target": "Company",
        "action": "create",
        "confidence": 0.9,
        "reason": "confirmed by agent",
    }

    resp = client.post(
        "/graphs/test-tenant/ontology/apply",
        headers=auth_headers,
        json=proposal,
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["applied"]["name"] == "works_at"
    # create relationship → mint subject (create) + ensure target type + property.
    assert mock_neptune.update.call_count == 3
    all_sparql = " ".join(c[0][0] for c in mock_neptune.update.call_args_list)
    assert "Person" in all_sparql
    assert "Company" in all_sparql
    assert "works_at" in all_sparql


def test_apply_attribute_extend_writes_single_upsert(client, auth_headers, mock_neptune):
    proposal = {
        "kind": "attribute",
        "subject_type": "Person",
        "name": "email",
        "datatype_or_target": "string",
        "action": "extend",
        "confidence": 0.99,
        "reason": "confirmed",
    }

    resp = client.post(
        "/graphs/test-tenant/ontology/apply",
        headers=auth_headers,
        json=proposal,
    )

    assert resp.status_code == 200
    # extend attribute → only the upsert_attribute statement, no type minting.
    assert mock_neptune.update.call_count == 1
    sent = mock_neptune.update.call_args[0][0]
    assert "email" in sent
