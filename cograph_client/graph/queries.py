import re


def tenant_graph_uri(tenant_id: str) -> str:
    """Base graph URI for a tenant. Used as the ontology graph."""
    return f"https://cograph.tech/graphs/{tenant_id}"


def kg_graph_uri(tenant_id: str, kg_name: str) -> str:
    """Named graph URI for a specific knowledge graph within a tenant."""
    return f"https://cograph.tech/graphs/{tenant_id}/kg/{kg_name}"


# The kg segment is anchored to a single path component ([^/]+, no slashes) so a
# COMPANION graph URI — e.g. a provenance graph ".../kg/<kg>/provenance" — does NOT
# greedily parse to kg_name="<kg>/provenance"; it correctly returns None (matching
# the docstring contract). KG names can't contain "/" (KGCreate enforces
# ^[a-zA-Z0-9_-]+$), so this never rejects a real KG.
_KG_GRAPH_RE = re.compile(
    r"^https://cograph\.tech/graphs/(?P<tenant>[^/]+)/kg/(?P<kg>[^/]+)$"
)


def parse_kg_graph_uri(graph_uri: str) -> tuple[str, str] | None:
    """Inverse of :func:`kg_graph_uri`: ``(tenant_id, kg_name)`` or ``None``.

    Returns ``None`` for anything that is not a per-KG instance-graph URI (e.g. the
    tenant ontology graph or a provenance companion graph), so callers can detect
    a non-KG graph and skip per-KG work rather than mis-deriving a scope.
    """
    if not isinstance(graph_uri, str):
        return None
    m = _KG_GRAPH_RE.match(graph_uri)
    if not m:
        return None
    return m.group("tenant"), m.group("kg")


def _escape_value(value: str) -> str:
    """Wrap a value as a URI (<...>), typed literal ("..."^^<xsd:type>), or plain literal ("...").

    Typed literal convention: "500000^^xsd:integer" → "500000"^^<xsd:integer>
    """
    if value.startswith("http://") or value.startswith("https://"):
        return f"<{value}>"
    if value.startswith("<") and value.endswith(">"):
        return value
    # Check for typed literal: value^^xsd:type
    if "^^" in value:
        literal, xsd_type = value.rsplit("^^", 1)
        return f'"{_escape_literal(literal)}"^^<{xsd_type}>'
    return f'"{_escape_literal(value)}"'


def _escape_literal(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def insert_triples(graph_uri: str, triples: list[tuple[str, str, str]]) -> str:
    triple_strs = []
    for s, p, o in triples:
        triple_strs.append(f"  {_escape_value(s)} {_escape_value(p)} {_escape_value(o)} .")
    body = "\n".join(triple_strs)
    return f"INSERT DATA {{\n  GRAPH <{graph_uri}> {{\n{body}\n  }}\n}}"


def batched_insert_triples(
    graph_uri: str, triples: list[tuple[str, str, str]], batch_size: int = 500,
) -> list[str]:
    """Split triples into batched SPARQL INSERT DATA statements."""
    if not triples:
        return []
    return [
        insert_triples(graph_uri, triples[i : i + batch_size])
        for i in range(0, len(triples), batch_size)
    ]


def delete_triples(graph_uri: str, triples: list[tuple[str, str, str]]) -> str:
    triple_strs = []
    for s, p, o in triples:
        triple_strs.append(f"  {_escape_value(s)} {_escape_value(p)} {_escape_value(o)} .")
    body = "\n".join(triple_strs)
    return f"DELETE DATA {{\n  GRAPH <{graph_uri}> {{\n{body}\n  }}\n}}"


def select_triples(
    graph_uri: str,
    subject: str | None = None,
    predicate: str | None = None,
    obj: str | None = None,
    limit: int = 100,
) -> str:
    s = _escape_value(subject) if subject else "?s"
    p = _escape_value(predicate) if predicate else "?p"
    o = _escape_value(obj) if obj else "?o"
    return (
        f"SELECT ?s ?p ?o FROM <{graph_uri}>\n"
        f"WHERE {{ ?s ?p ?o .\n"
        f"  FILTER(?s = {s} || {s} = ?s)\n"
        f"  FILTER(?p = {p} || {p} = ?p)\n"
        f"  FILTER(?o = {o} || {o} = ?o)\n"
        f"}}\nLIMIT {limit}"
    ) if any([subject, predicate, obj]) else (
        f"SELECT ?s ?p ?o FROM <{graph_uri}>\n"
        f"WHERE {{ ?s ?p ?o . }}\n"
        f"LIMIT {limit}"
    )


def scoped_query(graph_uri: str, sparql: str) -> str:
    """Wrap a user-provided SPARQL query to scope it to a tenant's named graph."""
    return f"# Scoped to tenant graph\n# FROM <{graph_uri}>\n{sparql}"


def register_function_triple(
    graph_uri: str,
    entity_type: str,
    function_name: str,
    endpoint_url: str,
    description: str = "",
) -> str:
    func_uri = f"https://cograph.tech/functions/{function_name}"
    type_uri = f"https://cograph.tech/types/{entity_type}"
    triples = [
        (func_uri, "https://cograph.tech/onto/attachedTo", type_uri),
        (func_uri, "https://cograph.tech/onto/endpointUrl", endpoint_url),
        (func_uri, "https://cograph.tech/onto/name", function_name),
    ]
    if description:
        triples.append((func_uri, "https://cograph.tech/onto/description", description))
    return insert_triples(graph_uri, triples)


BATCH_PREDICATE = "https://cograph.tech/onto/batch_id"


def delete_batch_query(graph_uri: str, batch_id: str) -> str:
    """Delete all triples whose subject belongs to a given batch.

    This removes: (1) the batch provenance triple itself, and
    (2) all other triples sharing the same subject.
    """
    return (
        f"DELETE {{\n"
        f"  GRAPH <{graph_uri}> {{ ?s ?p ?o }}\n"
        f"}} WHERE {{\n"
        f"  GRAPH <{graph_uri}> {{\n"
        f"    ?s <{BATCH_PREDICATE}> \"{_escape_literal(batch_id)}\" .\n"
        f"    ?s ?p ?o .\n"
        f"  }}\n"
        f"}}"
    )


def list_functions_query(graph_uri: str, entity_type: str | None = None) -> str:
    type_filter = ""
    if entity_type:
        type_uri = f"https://cograph.tech/types/{entity_type}"
        type_filter = f'  FILTER(?type = <{type_uri}>)\n'
    return (
        f"SELECT ?name ?type ?endpoint ?desc FROM <{graph_uri}>\n"
        f"WHERE {{\n"
        f"  ?func <https://cograph.tech/onto/name> ?name .\n"
        f"  ?func <https://cograph.tech/onto/attachedTo> ?type .\n"
        f"  ?func <https://cograph.tech/onto/endpointUrl> ?endpoint .\n"
        f"  OPTIONAL {{ ?func <https://cograph.tech/onto/description> ?desc }}\n"
        f"{type_filter}}}"
    )
