"""Tests for ontology layers + precedence resolver (ADR 0002 §1, COG-36).

Covers shadowing order (Tenant > Enhanced > Public), graceful degradation for
non-entitled stacks (Enhanced excluded, never an error), per-layer namespace
separation, the per-layer type fetch helper, and a regression guard that the
existing tenant namespace / type_uri() behavior is byte-for-byte unchanged.

All mocked — no live Neptune, no LLM, no network.
"""

from unittest.mock import AsyncMock

import pytest

from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.layers import (
    Layer,
    LayerStack,
    enhanced_graph_uri,
    fetch_types_by_layer,
    layer_type_uri,
    public_graph_uri,
    type_namespace,
)
from cograph_client.graph.ontology_queries import type_uri


# ---------------------------------------------------------------------------
# Stack construction + visibility
# ---------------------------------------------------------------------------

TENANT_GRAPH = "https://cograph.tech/graphs/acme"


def test_entitled_stack_layers_in_precedence_order():
    stack = LayerStack(TENANT_GRAPH, entitled=True)
    assert stack.layers == (Layer.TENANT, Layer.ENHANCED, Layer.PUBLIC)
    assert stack.visible_graph_uris() == [
        TENANT_GRAPH, enhanced_graph_uri(), public_graph_uri(),
    ]


def test_non_entitled_stack_excludes_enhanced():
    stack = LayerStack(TENANT_GRAPH, entitled=False)
    assert stack.layers == (Layer.TENANT, Layer.PUBLIC)
    assert enhanced_graph_uri() not in stack.visible_graph_uris()
    assert stack.visible_graph_uris() == [TENANT_GRAPH, public_graph_uri()]


def test_tenant_graph_uri_is_whatever_caller_passes():
    # The tenant layer has no fixed graph — callers keep passing their own.
    stack = LayerStack("https://cograph.tech/graphs/other-tenant")
    assert stack.graph_uri_for(Layer.TENANT) == "https://cograph.tech/graphs/other-tenant"


# ---------------------------------------------------------------------------
# Shadowing: Tenant > Enhanced > Public
# ---------------------------------------------------------------------------


def test_tenant_definition_shadows_enhanced_and_public():
    stack = LayerStack(TENANT_GRAPH, entitled=True)
    types_by_layer = {
        Layer.TENANT: {"Hotel": "tenant def"},
        Layer.ENHANCED: {"Hotel": "enhanced def"},
        Layer.PUBLIC: {"Hotel": "public def"},
    }
    assert stack.resolve_type("Hotel", types_by_layer) == (Layer.TENANT, "tenant def")


def test_enhanced_definition_shadows_public():
    stack = LayerStack(TENANT_GRAPH, entitled=True)
    types_by_layer = {
        Layer.TENANT: {},
        Layer.ENHANCED: {"Hotel": "enhanced def"},
        Layer.PUBLIC: {"Hotel": "public def"},
    }
    assert stack.resolve_type("Hotel", types_by_layer) == (Layer.ENHANCED, "enhanced def")


def test_public_definition_used_when_nothing_shadows():
    stack = LayerStack(TENANT_GRAPH, entitled=True)
    types_by_layer = {Layer.PUBLIC: {"Hotel": "public def"}}
    assert stack.resolve_type("Hotel", types_by_layer) == (Layer.PUBLIC, "public def")


def test_resolve_type_returns_none_when_undefined():
    stack = LayerStack(TENANT_GRAPH, entitled=True)
    assert stack.resolve_type("Hotel", {Layer.TENANT: {}, Layer.PUBLIC: {}}) is None


def test_non_entitled_resolution_skips_enhanced_definitions():
    """Even if Enhanced data is present in the input map, a non-entitled stack
    must not see it — it degrades to Tenant > Public, never errors."""
    stack = LayerStack(TENANT_GRAPH, entitled=False)
    types_by_layer = {
        Layer.ENHANCED: {"Hotel": "enhanced def"},
        Layer.PUBLIC: {"Hotel": "public def"},
    }
    assert stack.resolve_type("Hotel", types_by_layer) == (Layer.PUBLIC, "public def")


# ---------------------------------------------------------------------------
# Namespaces: one per layer, tenant namespace unchanged (regression)
# ---------------------------------------------------------------------------


def test_layer_namespaces_are_distinct():
    namespaces = {type_namespace(layer) for layer in Layer}
    assert len(namespaces) == 3
    uris = {layer_type_uri(layer, "Hotel") for layer in Layer}
    assert len(uris) == 3


def test_enhanced_and_public_namespaces():
    assert layer_type_uri(Layer.PUBLIC, "Hotel") == "https://cograph.tech/types/public/Hotel"
    assert layer_type_uri(Layer.ENHANCED, "Hotel") == "https://cograph.tech/types/x/Hotel"


def test_tenant_type_uri_unchanged_regression():
    """Backward-compat: the existing namespace stays the tenant namespace and
    type_uri() output is byte-for-byte what it was before layers existed."""
    assert type_uri("Place") == "https://cograph.tech/types/Place"
    assert layer_type_uri(Layer.TENANT, "Place") == type_uri("Place")
    assert type_namespace(Layer.TENANT) == "https://cograph.tech/types/"


# ---------------------------------------------------------------------------
# fetch_types_by_layer: one query per visible graph, per-layer degradation
# ---------------------------------------------------------------------------


def _bindings(*labels: str) -> dict:
    return {
        "head": {"vars": ["type", "label", "comment", "parent"]},
        "results": {"bindings": [
            {"label": {"type": "literal", "value": label}} for label in labels
        ]},
    }


@pytest.fixture
def mock_neptune():
    client = AsyncMock(spec=NeptuneClient)
    client.health.return_value = True
    client.query.return_value = {"head": {"vars": []}, "results": {"bindings": []}}
    client.update.return_value = None
    return client


@pytest.mark.asyncio
async def test_fetch_types_by_layer_queries_each_visible_graph(mock_neptune):
    stack = LayerStack(TENANT_GRAPH, entitled=True)
    mock_neptune.query.side_effect = [
        _bindings("Condo"),            # tenant
        _bindings("Hotel"),            # enhanced
        _bindings("Hotel", "Person"),  # public
    ]

    got = await fetch_types_by_layer(mock_neptune, stack)

    assert got == {
        Layer.TENANT: {"Condo": ""},
        Layer.ENHANCED: {"Hotel": ""},
        Layer.PUBLIC: {"Hotel": "", "Person": ""},
    }
    queried = [c.args[0] for c in mock_neptune.query.call_args_list]
    assert len(queried) == 3
    for sparql, graph in zip(queried, stack.visible_graph_uris()):
        assert f"<{graph}>" in sparql
    # Output feeds resolve_type directly: enhanced shadows public for Hotel.
    assert stack.resolve_type("Hotel", got) == (Layer.ENHANCED, "")


@pytest.mark.asyncio
async def test_fetch_types_by_layer_non_entitled_skips_enhanced_graph(mock_neptune):
    stack = LayerStack(TENANT_GRAPH, entitled=False)

    got = await fetch_types_by_layer(mock_neptune, stack)

    assert set(got) == {Layer.TENANT, Layer.PUBLIC}
    queried = " || ".join(c.args[0] for c in mock_neptune.query.call_args_list)
    assert enhanced_graph_uri() not in queried


@pytest.mark.asyncio
async def test_fetch_types_by_layer_degrades_per_layer_on_error(mock_neptune):
    """A failing layer graph (e.g. Public not yet seeded) yields {} for that
    layer only — the call never raises and other layers are unaffected."""
    stack = LayerStack(TENANT_GRAPH, entitled=True)
    mock_neptune.query.side_effect = [
        _bindings("Condo"),
        RuntimeError("enhanced graph unreachable"),
        _bindings("Person"),
    ]

    got = await fetch_types_by_layer(mock_neptune, stack)

    assert got == {
        Layer.TENANT: {"Condo": ""},
        Layer.ENHANCED: {},
        Layer.PUBLIC: {"Person": ""},
    }
