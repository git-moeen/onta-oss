"""SPARQL query builders for ontology management."""

OMNIX_ONTO = "https://cograph.tech/onto"
RDFS = "http://www.w3.org/2000/01/rdf-schema"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns"
XSD = "http://www.w3.org/2001/XMLSchema"


def type_uri(type_name: str) -> str:
    return f"https://cograph.tech/types/{type_name}"


def attr_uri(type_name: str, attr_name: str) -> str:
    return f"https://cograph.tech/types/{type_name}/attrs/{attr_name}"


def insert_type(graph_uri: str, name: str, description: str = "", parent_type: str | None = None) -> str:
    uri = type_uri(name)
    triples = [
        f'  <{uri}> <{RDF}#type> <{RDFS}#Class> .',
        f'  <{uri}> <{RDFS}#label> "{_esc(name)}" .',
    ]
    if description:
        triples.append(f'  <{uri}> <{RDFS}#comment> "{_esc(description)}" .')
    if parent_type:
        triples.append(f'  <{uri}> <{RDFS}#subClassOf> <{type_uri(parent_type)}> .')
    body = "\n".join(triples)
    return f"INSERT DATA {{\n  GRAPH <{graph_uri}> {{\n{body}\n  }}\n}}"


def insert_attribute(graph_uri: str, type_name: str, attr_name: str, description: str = "", datatype: str = "string") -> str:
    t_uri = type_uri(type_name)
    a_uri = attr_uri(type_name, attr_name)
    xsd_type = _datatype_to_xsd(datatype)
    triples = [
        f'  <{a_uri}> <{RDF}#type> <{RDF}#Property> .',
        f'  <{a_uri}> <{RDFS}#label> "{_esc(attr_name)}" .',
        f'  <{a_uri}> <{RDFS}#domain> <{t_uri}> .',
        f'  <{a_uri}> <{RDFS}#range> <{xsd_type}> .',
    ]
    if description:
        triples.append(f'  <{a_uri}> <{RDFS}#comment> "{_esc(description)}" .')
    body = "\n".join(triples)
    return f"INSERT DATA {{\n  GRAPH <{graph_uri}> {{\n{body}\n  }}\n}}"


def insert_subtype(graph_uri: str, parent_name: str, child_name: str) -> str:
    return (
        f"INSERT DATA {{\n"
        f"  GRAPH <{graph_uri}> {{\n"
        f"    <{type_uri(child_name)}> <{RDFS}#subClassOf> <{type_uri(parent_name)}> .\n"
        f"  }}\n"
        f"}}"
    )


def merge_predicates(graph_uri: str, old_predicate: str, new_predicate: str) -> str:
    """Generate SPARQL UPDATE to rename a predicate across all triples."""
    return (
        f"DELETE {{ GRAPH <{graph_uri}> {{ ?s <{old_predicate}> ?o }} }}\n"
        f"INSERT {{ GRAPH <{graph_uri}> {{ ?s <{new_predicate}> ?o }} }}\n"
        f"WHERE {{ GRAPH <{graph_uri}> {{ ?s <{old_predicate}> ?o }} }}"
    )


def list_types_query(graph_uri: str) -> str:
    return (
        f"SELECT ?type ?label ?comment ?parent FROM <{graph_uri}>\n"
        f"WHERE {{\n"
        f"  ?type <{RDF}#type> <{RDFS}#Class> .\n"
        f"  ?type <{RDFS}#label> ?label .\n"
        f"  OPTIONAL {{ ?type <{RDFS}#comment> ?comment }}\n"
        f"  OPTIONAL {{ ?type <{RDFS}#subClassOf> ?parent }}\n"
        f"}}"
    )


def get_type_detail_query(graph_uri: str, type_name: str) -> str:
    t_uri = type_uri(type_name)
    return (
        f"SELECT ?label ?comment ?parent FROM <{graph_uri}>\n"
        f"WHERE {{\n"
        f"  <{t_uri}> <{RDFS}#label> ?label .\n"
        f"  OPTIONAL {{ <{t_uri}> <{RDFS}#comment> ?comment }}\n"
        f"  OPTIONAL {{ <{t_uri}> <{RDFS}#subClassOf> ?parent }}\n"
        f"}}"
    )


def get_type_attributes_query(graph_uri: str, type_name: str) -> str:
    t_uri = type_uri(type_name)
    return (
        f"SELECT ?attr ?attrLabel ?attrComment ?range FROM <{graph_uri}>\n"
        f"WHERE {{\n"
        f"  ?attr <{RDF}#type> <{RDF}#Property> .\n"
        f"  ?attr <{RDFS}#domain> <{t_uri}> .\n"
        f"  ?attr <{RDFS}#label> ?attrLabel .\n"
        f"  OPTIONAL {{ ?attr <{RDFS}#comment> ?attrComment }}\n"
        f"  OPTIONAL {{ ?attr <{RDFS}#range> ?range }}\n"
        f"}}"
    )


def get_subtypes_query(graph_uri: str, type_name: str) -> str:
    t_uri = type_uri(type_name)
    return (
        f"SELECT ?sub ?label FROM <{graph_uri}>\n"
        f"WHERE {{\n"
        f"  ?sub <{RDFS}#subClassOf> <{t_uri}> .\n"
        f"  ?sub <{RDFS}#label> ?label .\n"
        f"}}"
    )


def get_type_functions_query(graph_uri: str, type_name: str) -> str:
    t_uri = type_uri(type_name)
    return (
        f"SELECT ?name ?endpoint ?desc FROM <{graph_uri}>\n"
        f"WHERE {{\n"
        f"  ?func <{OMNIX_ONTO}/attachedTo> <{t_uri}> .\n"
        f"  ?func <{OMNIX_ONTO}/name> ?name .\n"
        f"  OPTIONAL {{ ?func <{OMNIX_ONTO}/endpointUrl> ?endpoint }}\n"
        f"  OPTIONAL {{ ?func <{OMNIX_ONTO}/description> ?desc }}\n"
        f"}}"
    )


def parent_map_query(graph_uri: str) -> str:
    """Select every rdfs:subClassOf edge so a caller can build a child->parent map.

    Returns ?child ?parent for all `?child rdfs:subClassOf ?parent` triples.
    The caller turns these bindings into `parent_of: dict[str, str]` (keyed by
    the type *name*, i.e. the last URI path segment) for hierarchy walks used by
    config_for_with_hierarchy / primary_type / ancestor_chain.
    """
    return (
        f"SELECT ?child ?parent FROM <{graph_uri}>\n"
        f"WHERE {{\n"
        f"  ?child <{RDFS}#subClassOf> ?parent .\n"
        f"}}"
    )


def with_subclass_closure(type_name: str) -> str:
    """Return the SPARQL property-path predicate that matches `type_name` and any
    of its subtypes: `a/<RDFS#subClassOf>*`.

    Used in place of a bare `a`/rdf:type predicate so a query over a parent type
    returns subtype instances too (ADR rule 2 — query-time subclass closure).
    The trailing `<type_uri(type_name)>` object is supplied by the caller.
    """
    return f"<{RDF}#type>/<{RDFS}#subClassOf>*"


# Property path that turns a type-assertion predicate into its subclass closure.
_CLOSURE_PATH = f"<{RDF}#type>/<{RDFS}#subClassOf>*"
_TYPES_URI = "https://cograph.tech/types/"


def rewrite_type_predicate_to_closure(sparql: str) -> str:
    """Rewrite type-assertion triples to use subclass-closure property paths.

    Turns `?var a <types/X>` and `?var <rdf:type> <types/X>` into
    `?var <rdf:type>/<rdfs:subClassOf>* <types/X>`, so a query over a parent
    type returns all subtype instances (ADR rule 2).

    Deterministic and regex-based — no ontology lookup, no Neptune, no LLM:
      - Only matches type-assertion predicate position whose OBJECT is a
        `https://cograph.tech/types/...` URI (the only place rewriting is valid).
      - Closure over a leaf type is set-equal to the leaf itself, so the rewrite
        is safe to apply unconditionally.
      - Idempotent: a triple already using the closure path (`.../subClassOf>*`)
        is left untouched.

    Pure string transform — unit-testable with a plain SPARQL string.
    """
    import re

    rdf_type_full = f"<{RDF}#type>"
    types_obj = re.escape(_TYPES_URI)

    # Form A: `?var a <https://cograph.tech/types/X>`
    sparql = re.sub(
        rf'(\?\w+)\s+a\s+(<{types_obj}\w+>)',
        rf'\1 {_CLOSURE_PATH} \2',
        sparql,
    )

    # Form B: `?var <http://...#type> <https://cograph.tech/types/X>`
    # The negative-lookahead on the predicate guards idempotence: skip when the
    # predicate is already the closure path (which itself contains <...#type>).
    sparql = re.sub(
        rf'(\?\w+)\s+{re.escape(rdf_type_full)}(?!/)\s+(<{types_obj}\w+>)',
        rf'\1 {_CLOSURE_PATH} \2',
        sparql,
    )

    return sparql


def get_full_ontology_query(graph_uri: str) -> str:
    """Get all types, attributes, and functions in one query for the NL pipeline."""
    return (
        f"SELECT ?type ?typeLabel ?attr ?attrLabel ?range ?funcName FROM <{graph_uri}>\n"
        f"WHERE {{\n"
        f"  ?type <{RDF}#type> <{RDFS}#Class> .\n"
        f"  ?type <{RDFS}#label> ?typeLabel .\n"
        f"  OPTIONAL {{\n"
        f"    ?attr <{RDFS}#domain> ?type .\n"
        f"    ?attr <{RDFS}#label> ?attrLabel .\n"
        f"    OPTIONAL {{ ?attr <{RDFS}#range> ?range }}\n"
        f"  }}\n"
        f"  OPTIONAL {{\n"
        f"    ?func <{OMNIX_ONTO}/attachedTo> ?type .\n"
        f"    ?func <{OMNIX_ONTO}/name> ?funcName .\n"
        f"  }}\n"
        f"}}"
    )


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


PRIMITIVE_TYPES = {"string", "integer", "float", "boolean", "datetime", "uri"}


def entity_exists_query(graph_uri: str, entity_uri: str) -> str:
    """Check if an entity URI already exists in the graph."""
    return (
        f"ASK FROM <{graph_uri}> WHERE {{\n"
        f"  <{entity_uri}> ?p ?o .\n"
        f"}}"
    )


def batch_entity_exists_query(graph_uri: str, entity_uris: list[str]) -> str:
    """Check which entity URIs already exist in the graph.

    Uses SPARQL VALUES clause to batch-check up to 500 URIs at once.
    Returns entity URIs that have at least one triple.
    """
    values = " ".join(f"(<{uri}>)" for uri in entity_uris)
    return (
        f"SELECT DISTINCT ?entity FROM <{graph_uri}> WHERE {{\n"
        f"  VALUES (?entity) {{ {values} }}\n"
        f"  ?entity ?p ?o .\n"
        f"}}"
    )


def _datatype_to_xsd(datatype: str) -> str:
    mapping = {
        "string": f"{XSD}#string",
        "integer": f"{XSD}#integer",
        "float": f"{XSD}#float",
        "boolean": f"{XSD}#boolean",
        "datetime": f"{XSD}#dateTime",
        "uri": f"{RDFS}#Resource",
    }
    if datatype in mapping:
        return mapping[datatype]
    # Treat as a reference to another type
    return type_uri(datatype)
