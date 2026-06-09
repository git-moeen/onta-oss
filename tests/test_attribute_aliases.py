"""Attribute alias mechanism tests (COG-40, ADR 0002 §7).

Covers: alias registration/retirement SPARQL, alias-map chain flattening +
cycle guard, the conservative full-IRI query rewriter, the batched lazy
backfill, and the NL pipeline wiring — including the regression-critical
default path (feature OFF / zero aliases => zero behavior change).

All mocked — no live Neptune, no LLM, no network.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cograph_client.graph.aliases import (
    ALIAS_OF,
    backfill_aliases,
    fetch_alias_map,
    register_alias,
    retire_alias,
    rewrite_query_attrs,
)
from cograph_client.graph.client import NeptuneClient
from cograph_client.nlp.pipeline import NLQueryPipeline

ONTO_GRAPH = "https://cograph.tech/graphs/t-alias"
DATA_GRAPH = "https://cograph.tech/graphs/t-alias/kg/main"

PHONE_NUM = "https://cograph.tech/types/Guest/attrs/phone_num"
PHONE = "https://cograph.tech/types/Guest/attrs/phone"
CONTACT = "https://cograph.tech/types/Person/attrs/contact_phone"


@pytest.fixture
def mock_neptune():
    client = AsyncMock(spec=NeptuneClient)
    client.query.return_value = {"head": {"vars": []}, "results": {"bindings": []}}
    client.update.return_value = None
    return client


def _alias_bindings(*pairs: tuple[str, str]) -> dict:
    return {
        "head": {"vars": ["old", "new"]},
        "results": {
            "bindings": [
                {
                    "old": {"type": "uri", "value": old},
                    "new": {"type": "uri", "value": new},
                }
                for old, new in pairs
            ]
        },
    }


def _count_result(n: int) -> dict:
    return {
        "head": {"vars": ["n"]},
        "results": {"bindings": [{"n": {"type": "literal", "value": str(n)}}]},
    }


# ---------------------------------------------------------------------------
# register / retire
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_alias_writes_alias_triple(mock_neptune):
    await register_alias(mock_neptune, ONTO_GRAPH, PHONE_NUM, PHONE)
    sparql = mock_neptune.update.call_args.args[0]
    assert "INSERT DATA" in sparql
    assert f"GRAPH <{ONTO_GRAPH}>" in sparql
    assert f"<{PHONE_NUM}> <{ALIAS_OF}> <{PHONE}> ." in sparql


@pytest.mark.asyncio
async def test_register_alias_rejects_self_alias(mock_neptune):
    with pytest.raises(ValueError):
        await register_alias(mock_neptune, ONTO_GRAPH, PHONE, PHONE)
    mock_neptune.update.assert_not_called()


@pytest.mark.asyncio
async def test_retire_alias_deletes_alias_triple(mock_neptune):
    await retire_alias(mock_neptune, ONTO_GRAPH, PHONE_NUM)
    sparql = mock_neptune.update.call_args.args[0]
    assert "DELETE WHERE" in sparql
    assert f"<{PHONE_NUM}> <{ALIAS_OF}> ?new" in sparql


# ---------------------------------------------------------------------------
# fetch_alias_map — chain flattening + cycle guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_alias_map_flattens_chains(mock_neptune):
    """a -> b -> c resolves to {a: c, b: c} — every rewrite is one hop."""
    mock_neptune.query.return_value = _alias_bindings(
        (PHONE_NUM, PHONE), (PHONE, CONTACT),
    )
    got = await fetch_alias_map(mock_neptune, ONTO_GRAPH)
    assert got == {PHONE_NUM: CONTACT, PHONE: CONTACT}


@pytest.mark.asyncio
async def test_fetch_alias_map_drops_cycles(mock_neptune):
    """a -> b -> a is nonsensical alias data: both entries dropped, no hang."""
    mock_neptune.query.return_value = _alias_bindings(
        (PHONE_NUM, PHONE), (PHONE, PHONE_NUM),
    )
    got = await fetch_alias_map(mock_neptune, ONTO_GRAPH)
    assert got == {}


@pytest.mark.asyncio
async def test_fetch_alias_map_self_cycle_dropped_others_kept(mock_neptune):
    mock_neptune.query.return_value = _alias_bindings(
        (PHONE, PHONE),          # self-alias: dropped
        (PHONE_NUM, CONTACT),    # independent alias: kept
    )
    got = await fetch_alias_map(mock_neptune, ONTO_GRAPH)
    assert got == {PHONE_NUM: CONTACT}


# ---------------------------------------------------------------------------
# rewrite_query_attrs — full-IRI matches only
# ---------------------------------------------------------------------------


def test_rewrite_query_attrs_rewrites_full_iri():
    q = f"SELECT ?v WHERE {{ ?g <{PHONE_NUM}> ?v }}"
    out = rewrite_query_attrs(q, {PHONE_NUM: PHONE})
    assert f"<{PHONE}>" in out
    assert f"<{PHONE_NUM}>" not in out


def test_rewrite_query_attrs_ignores_prefix_overlap():
    """`<.../attrs/phone>` must NOT fire inside `<.../attrs/phone_num>`."""
    q = f"SELECT ?v WHERE {{ ?g <{PHONE_NUM}> ?v }}"
    assert rewrite_query_attrs(q, {PHONE: CONTACT}) == q


def test_rewrite_query_attrs_single_pass_no_rechaining():
    """An unflattened map (a->b, b->c) must not double-rewrite a to c."""
    q = f"SELECT ?v WHERE {{ ?g <{PHONE_NUM}> ?v }}"
    out = rewrite_query_attrs(q, {PHONE_NUM: PHONE, PHONE: CONTACT})
    assert f"<{PHONE}>" in out and f"<{CONTACT}>" not in out


def test_rewrite_query_attrs_empty_map_is_identity():
    q = f"SELECT ?v WHERE {{ ?g <{PHONE_NUM}> ?v }}"
    assert rewrite_query_attrs(q, {}) == q


# ---------------------------------------------------------------------------
# backfill_aliases — batched DELETE/INSERT WHERE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backfill_emits_batched_delete_insert(mock_neptune):
    """2500 triples at batch_size=1000 => 3 batched updates, count returned."""
    mock_neptune.query.return_value = _count_result(2500)

    rewritten = await backfill_aliases(
        mock_neptune, DATA_GRAPH, {PHONE_NUM: PHONE}, batch_size=1000,
    )

    assert rewritten == 2500
    updates = [c.args[0] for c in mock_neptune.update.call_args_list]
    assert len(updates) == 3
    for sparql in updates:
        assert f"DELETE {{ GRAPH <{DATA_GRAPH}> {{ ?s <{PHONE_NUM}> ?o }} }}" in sparql
        assert f"INSERT {{ GRAPH <{DATA_GRAPH}> {{ ?s <{PHONE}> ?o }} }}" in sparql
        assert "LIMIT 1000" in sparql
    # The count probe ran against the old predicate in the data graph.
    count_sparql = mock_neptune.query.call_args.args[0]
    assert "COUNT" in count_sparql and f"<{PHONE_NUM}>" in count_sparql


@pytest.mark.asyncio
async def test_backfill_zero_triples_no_updates(mock_neptune):
    mock_neptune.query.return_value = _count_result(0)
    rewritten = await backfill_aliases(mock_neptune, DATA_GRAPH, {PHONE_NUM: PHONE})
    assert rewritten == 0
    mock_neptune.update.assert_not_called()


# ---------------------------------------------------------------------------
# NL pipeline wiring (gated by COGRAPH_ALIASES_ENABLED)
# ---------------------------------------------------------------------------


def _llm_message(sparql: str) -> MagicMock:
    msg = MagicMock()
    msg.content = [MagicMock(text=json.dumps({
        "sparql": sparql,
        "explanation": "test",
        "functions_needed": [],
    }))]
    return msg


def _exec_result() -> dict:
    return {
        "head": {"vars": ["v"]},
        "results": {"bindings": [{"v": {"type": "literal", "value": "555-0100"}}]},
    }


@pytest.mark.asyncio
async def test_pipeline_default_off_no_alias_query_no_rewrite(mock_neptune, monkeypatch):
    """Regression guard: with the flag unset (the default), /ask issues NO
    alias-map query and the generated SPARQL passes through untouched."""
    monkeypatch.delenv("COGRAPH_ALIASES_ENABLED", raising=False)
    pipeline = NLQueryPipeline(mock_neptune, "fake-key")
    pipeline._openrouter_key = ""
    assert pipeline._aliases_enabled is False

    generated = f"SELECT ?v WHERE {{ ?g <{PHONE_NUM}> ?v }}"
    with patch("cograph_client.nlp.pipeline.get_embedding_service", return_value=None), \
         patch.object(pipeline, "_fetch_ontology", new=AsyncMock(return_value="Type: Guest")), \
         patch.object(pipeline.anthropic.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _llm_message(generated)
        mock_neptune.query.return_value = _exec_result()
        result = await pipeline.ask("guest phone?", f"{ONTO_GRAPH}-off")

    assert f"<{PHONE_NUM}>" in result.sparql  # not rewritten
    for call in mock_neptune.query.call_args_list:
        assert "aliasOf" not in call.args[0]


@pytest.mark.asyncio
async def test_pipeline_enabled_rewrites_aliased_attr(mock_neptune, monkeypatch):
    """Flag on + alias registered: the executed SPARQL uses the new IRI."""
    monkeypatch.setenv("COGRAPH_ALIASES_ENABLED", "1")
    pipeline = NLQueryPipeline(mock_neptune, "fake-key")
    pipeline._openrouter_key = ""
    assert pipeline._aliases_enabled is True

    generated = f"SELECT ?v WHERE {{ ?g <{PHONE_NUM}> ?v }}"
    mock_neptune.query.side_effect = [
        _alias_bindings((PHONE_NUM, PHONE)),  # alias-map fetch
        _exec_result(),                       # query execution
    ]
    with patch("cograph_client.nlp.pipeline.get_embedding_service", return_value=None), \
         patch.object(pipeline, "_fetch_ontology", new=AsyncMock(return_value="Type: Guest")), \
         patch.object(pipeline.anthropic.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _llm_message(generated)
        result = await pipeline.ask("guest phone?", f"{ONTO_GRAPH}-on")

    assert f"<{PHONE}>" in result.sparql
    assert f"<{PHONE_NUM}>" not in result.sparql
    executed = mock_neptune.query.call_args_list[1].args[0]
    assert f"<{PHONE}>" in executed


@pytest.mark.asyncio
async def test_pipeline_enabled_zero_aliases_unchanged(mock_neptune, monkeypatch):
    """Flag on but no aliases registered: empty map => query untouched."""
    monkeypatch.setenv("COGRAPH_ALIASES_ENABLED", "1")
    pipeline = NLQueryPipeline(mock_neptune, "fake-key")
    pipeline._openrouter_key = ""

    generated = f"SELECT ?v WHERE {{ ?g <{PHONE_NUM}> ?v }}"
    mock_neptune.query.side_effect = [
        {"head": {"vars": ["old", "new"]}, "results": {"bindings": []}},
        _exec_result(),
    ]
    with patch("cograph_client.nlp.pipeline.get_embedding_service", return_value=None), \
         patch.object(pipeline, "_fetch_ontology", new=AsyncMock(return_value="Type: Guest")), \
         patch.object(pipeline.anthropic.messages, "create", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = _llm_message(generated)
        result = await pipeline.ask("guest phone?", f"{ONTO_GRAPH}-zero")

    assert f"<{PHONE_NUM}>" in result.sparql


def test_fix_common_sparql_issues_default_signature_unchanged():
    """Backward-compat: the two-arg call (no alias_map) still works and the
    alias pass is a no-op."""
    q = f"SELECT ?v WHERE {{ ?g <{PHONE_NUM}> ?v }}"
    out = NLQueryPipeline._fix_common_sparql_issues(q, "")
    assert f"<{PHONE_NUM}>" in out
