"""Ontology layers + precedence resolver (ADR 0002 §1).

Three layers, each a named graph with its own type-URI namespace:

  Tenant > Global-Enhanced > Global-Public

Precedence is resolved by SHADOWING: the highest visible layer that defines a
type name wins for that tenant's queries; lower layers are never mutated.
Non-entitled tenants simply do not see the Enhanced layer — resolution
degrades gracefully to ``Tenant > Public``, never errors.

Namespaces (one per layer, so shadowing is explicit and collisions impossible):

  Tenant   — https://cograph.tech/types/          (the EXISTING namespace,
             unchanged — existing data keeps resolving via type_uri())
  Enhanced — https://cograph.tech/types/x/
  Public   — https://cograph.tech/types/public/

Everything here is additive and opt-in: default single-tenant, single-graph
behavior is untouched. The schema resolver can adopt fetch_types_by_layer +
LayerStack.resolve_type when the layer-aware closure work lands.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from .ontology_queries import list_types_query, type_uri
from .parser import parse_sparql_results

logger = structlog.stdlib.get_logger("cograph.graph.layers")


class Layer(str, Enum):
    """Ontology layers, see module docstring for precedence."""

    TENANT = "tenant"
    ENHANCED = "enhanced"
    PUBLIC = "public"


# Shared named graphs for the two Global layers. The tenant layer's graph URI
# stays whatever callers pass today (per-tenant, e.g. tenant_graph_uri()).
_PUBLIC_GRAPH_URI = "https://cograph.tech/graphs/global/public"
_ENHANCED_GRAPH_URI = "https://cograph.tech/graphs/global/enhanced"

# Per-layer type-URI namespaces. TENANT is the existing namespace — do not
# change it, or every URI already written to Neptune stops resolving.
_TYPE_NAMESPACES = {
    Layer.TENANT: "https://cograph.tech/types/",
    Layer.ENHANCED: "https://cograph.tech/types/x/",
    Layer.PUBLIC: "https://cograph.tech/types/public/",
}


def public_graph_uri() -> str:
    """Named graph holding the Global-Public ontology layer."""
    return _PUBLIC_GRAPH_URI


def enhanced_graph_uri() -> str:
    """Named graph holding the Global-Enhanced (premium delta) layer."""
    return _ENHANCED_GRAPH_URI


def type_namespace(layer: Layer) -> str:
    """Type-URI namespace prefix for a layer."""
    return _TYPE_NAMESPACES[layer]


def type_name_from_uri(uri: str) -> str | None:
    """Extract the bare type name from a type URI in ANY layer's namespace.

    Tries namespaces longest-first (the tenant namespace is a prefix of the
    other two, so order matters): `types/public/Person`, `types/x/Person`, and
    `types/Person` all yield "Person". Attribute URIs reduce to their type name
    (`types/Person/attrs/email` -> "Person"), matching the existing parent-map
    parsing. Returns None for URIs outside every layer namespace (e.g.
    rdfs:Resource), which callers skip.
    """
    for ns in sorted(_TYPE_NAMESPACES.values(), key=len, reverse=True):
        if uri.startswith(ns):
            name = uri[len(ns):].rstrip("/").split("/")[0]
            return name or None
    return None


def layer_type_uri(layer: Layer, type_name: str) -> str:
    """Type URI for `type_name` in `layer`'s namespace.

    For TENANT this delegates to the existing type_uri() so the two can never
    drift — tenant URIs are exactly what they have always been.
    """
    if layer is Layer.TENANT:
        return type_uri(type_name)
    return f"{_TYPE_NAMESPACES[layer]}{type_name}"


@dataclass(frozen=True)
class LayerStack:
    """The ordered set of ontology layers visible to one tenant.

    Built from (tenant_graph_uri, entitled). Entitled tenants see
    [TENANT, ENHANCED, PUBLIC]; non-entitled see [TENANT, PUBLIC] — the
    Enhanced layer is silently excluded, never an error.
    """

    tenant_graph_uri: str
    entitled: bool = False

    @property
    def layers(self) -> tuple[Layer, ...]:
        """Visible layers in precedence order (highest first)."""
        if self.entitled:
            return (Layer.TENANT, Layer.ENHANCED, Layer.PUBLIC)
        return (Layer.TENANT, Layer.PUBLIC)

    def graph_uri_for(self, layer: Layer) -> str:
        if layer is Layer.TENANT:
            return self.tenant_graph_uri
        if layer is Layer.ENHANCED:
            return _ENHANCED_GRAPH_URI
        return _PUBLIC_GRAPH_URI

    def visible_graph_uris(self) -> list[str]:
        """Graph URIs of the visible layers, in precedence order."""
        return [self.graph_uri_for(layer) for layer in self.layers]

    def resolve_type(
        self, name: str, types_by_layer: dict[Layer, dict[str, Any]]
    ) -> tuple[Layer, Any] | None:
        """Resolve `name` across the stack by shadowing.

        The first VISIBLE layer (in precedence order) that defines `name`
        wins; definitions in lower layers — or in layers not visible to this
        stack, e.g. ENHANCED for a non-entitled tenant — are ignored.
        Returns (layer, definition) or None if no visible layer defines it.
        """
        for layer in self.layers:
            defs = types_by_layer.get(layer)
            if defs is not None and name in defs:
                return layer, defs[name]
        return None


async def fetch_types_by_layer(neptune, stack: LayerStack) -> dict[Layer, dict[str, str]]:
    """Fetch existing types per visible layer (one list_types_query per graph).

    Returns {layer: {type_name: description}} for every layer in the stack —
    shaped to feed LayerStack.resolve_type. A layer whose graph is missing,
    empty, or errors yields {} (graceful degradation, mirroring
    SchemaResolver._fetch_ontology); other layers are unaffected.
    """
    types_by_layer: dict[Layer, dict[str, str]] = {}
    for layer in stack.layers:
        types: dict[str, str] = {}
        try:
            raw = await neptune.query(list_types_query(stack.graph_uri_for(layer)))
            _, bindings = parse_sparql_results(raw)
            for row in bindings:
                label = row.get("label", "")
                if label:
                    types[label] = row.get("comment", "")
        except Exception:
            # Degrade to an empty layer, never error (ADR 0002 §1).
            logger.warning("layer_types_fetch_failed", layer=layer.value, exc_info=True)
        types_by_layer[layer] = types
    return types_by_layer
