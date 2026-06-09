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


def parent_map_query(graph_uri: str | list[str]) -> str:
    """Select every rdfs:subClassOf edge so a caller can build a child->parent map.

    Returns ?child ?parent for all `?child rdfs:subClassOf ?parent` triples.
    The caller turns these bindings into `parent_of: dict[str, str]` (keyed by
    the type *name*, i.e. the last URI path segment) for hierarchy walks used by
    config_for_with_hierarchy / primary_type / ancestor_chain.

    Layer-aware variant (ADR 0002 §1, COG-37): pass a LIST of graph URIs (a
    LayerStack's visible_graph_uris()) and the query reads the UNION of those
    graphs — subClassOf edges may span layers (a tenant leaf under a Public
    parent). Each UNION branch is a GRAPH pattern (the form Neptune handles
    cleanly) and BINDs its graph URI to ?graph so the caller can apply layer
    precedence (shadowing) when merging duplicate child edges.

    The single-graph (str) form is byte-identical to the pre-layer query —
    regression-critical for existing callers.
    """
    if isinstance(graph_uri, str):
        return (
            f"SELECT ?child ?parent FROM <{graph_uri}>\n"
            f"WHERE {{\n"
            f"  ?child <{RDFS}#subClassOf> ?parent .\n"
            f"}}"
        )
    branches = "\n  UNION\n".join(
        f"  {{ GRAPH <{g}> {{ ?child <{RDFS}#subClassOf> ?parent . }} BIND(<{g}> AS ?graph) }}"
        for g in graph_uri
    )
    return (
        f"SELECT ?child ?parent ?graph\n"
        f"WHERE {{\n"
        f"{branches}\n"
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

    Turns `?var a <types/X>`, `?var <rdf:type> <types/X>`, and the prefixed
    `?var rdf:type <types/X>` into `?var <rdf:type>/<rdfs:subClassOf>* <types/X>`,
    so a query over a parent type returns all subtype instances (ADR rule 2).

    Deterministic and regex-based — no ontology lookup, no Neptune, no LLM:
      - Only matches type-assertion predicate position whose OBJECT is a
        `https://cograph.tech/types/...` URI (the only place rewriting is valid).
      - Closure over a leaf type is set-equal to the leaf itself, so the rewrite
        is safe to apply unconditionally.
      - Idempotent: a triple already using the closure path (`.../subClassOf>*`)
        is left untouched.

    Beyond the three DIRECT forms above, the rewriter also closes the INDIRECT
    type-selection shapes the LLM sometimes emits (COG-34), where the type is
    bound to a VARIABLE in the rdf:type object position and that variable is
    constrained to a `types/...` URI elsewhere in the query:

      D) VALUES form: `VALUES ?t { <types/X> } ... ?x <rdf:type> ?t`
      E) FILTER equality: `?x <rdf:type> ?t . FILTER(?t = <types/X>)`
      F) FILTER IN: `?x <rdf:type> ?t . FILTER(?t IN (<types/X>, <types/Y>))`

    For these, the OBJECT variable is left in place and only the rdf:type
    PREDICATE of the matching triple is upgraded to the closure path. Closure
    over the constrained value still yields subtypes, because the constraint
    pins the variable to the named type(s) and `subClassOf*` walks down from
    there. A UNION of explicit `?x a <types/Ti>` branches needs no new code —
    each branch is already a Form A/B/C direct triple.

    Deterministic and regex-based — no ontology lookup, no Neptune, no LLM:
      - Only matches type-assertion predicate position whose OBJECT is a
        `https://cograph.tech/types/...` URI (the only place rewriting is valid).
      - Closure over a leaf type is set-equal to the leaf itself, so the rewrite
        is safe to apply unconditionally.
      - Idempotent: a triple already using the closure path (`.../subClassOf>*`)
        is left untouched.

    NOTE: best-effort safety net, NOT a SPARQL parser. The indirect-shape pass
    is intentionally narrow: it only fires when the SAME variable appears both as
    the bare object of an rdf:type triple AND in a VALUES / FILTER constraint that
    references a `types/...` URI. Shapes it deliberately does NOT cover (to avoid
    brittle rewrites): a type variable constrained only indirectly (e.g. via a
    join to another triple), VALUES blocks that mix type and non-type URIs for the
    same variable, and constraints that span subquery boundaries. In those cases
    the query returns only the named type — acceptable, since the NL prompt steers
    the model toward the direct `?x a <type>` form that closes reliably.

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

    # Form C: prefixed `?var rdf:type <https://cograph.tech/types/X>`. Common
    # when the model declares `PREFIX rdf:`. Negative-lookahead on `/` keeps it
    # idempotent against an already-rewritten `rdf:type/rdfs:subClassOf*`.
    sparql = re.sub(
        rf'(\?\w+)\s+rdf:type(?!/)\s+(<{types_obj}\w+>)',
        rf'\1 {_CLOSURE_PATH} \2',
        sparql,
    )

    # ---- Indirect forms (COG-34): type bound to a variable + a constraint ----
    sparql = _rewrite_indirect_type_constraints(sparql)

    return sparql


def _rewrite_indirect_type_constraints(sparql: str) -> str:
    """Close rdf:type triples whose OBJECT is a VARIABLE constrained to a type URI.

    Handles COG-34 forms D/E/F (VALUES, FILTER `=`, FILTER `IN`). For each
    candidate variable `?t` we (1) confirm it is the bare object of an rdf:type
    triple, (2) confirm it is constrained to at least one `https://cograph.tech/
    types/...` URI via VALUES or FILTER, then (3) upgrade ONLY that triple's
    rdf:type predicate to the closure path. The object variable is untouched, so
    the existing VALUES/FILTER constraint keeps pinning it to the named type(s)
    while `subClassOf*` walks down to subtypes.

    Narrow and idempotent by construction: it rewrites a predicate only when the
    predicate is a bare rdf:type (`a`, `<...#type>`, or `rdf:type`) immediately
    followed by the SAME variable, and the closure-path predicate already contains
    `<...#type>/` so it can never re-match.
    """
    import re

    # Predicate alternation for a *bare* rdf:type, in any of the three notations.
    # Each branch forbids a trailing `/` so an already-rewritten closure path
    # (`...#type>/<...subClassOf>*`) is never matched again -> idempotent.
    rdf_type_full = f"<{RDF}#type>"
    bare_type_pred = (
        rf'(?:a|rdf:type(?!/)|{re.escape(rdf_type_full)}(?!/))'
    )

    # 1) Find every variable used as the bare object of an rdf:type triple:
    #    `?subj <bare-type-pred> ?typevar` (object MUST be a variable here).
    type_obj_vars: set[str] = set()
    for m in re.finditer(
        rf'\?\w+\s+{bare_type_pred}\s+(\?\w+)',
        sparql,
    ):
        type_obj_vars.add(m.group(1))

    if not type_obj_vars:
        return sparql

    types_uri = re.escape(_TYPES_URI)

    def _is_constrained_to_type(var: str) -> bool:
        """True if `var` (e.g. '?t') is bound/constrained to a types URI via a
        VALUES block or a FILTER (= / IN) elsewhere in the query."""
        v = re.escape(var)

        # VALUES ?t { ... <types/X> ... }  (single-var form)
        for vm in re.finditer(
            rf'VALUES\s+{v}\s*\{{([^}}]*)\}}',
            sparql,
            flags=re.IGNORECASE,
        ):
            if re.search(rf'<{types_uri}\w+>', vm.group(1)):
                return True

        # FILTER(?t = <types/X>)  and  FILTER(?t IN (<types/X>, ...))
        # Scan each FILTER body that mentions the variable and references a type.
        for fm in re.finditer(r'FILTER\s*\((.*?)\)', sparql, flags=re.IGNORECASE | re.DOTALL):
            body = fm.group(1)
            if not re.search(rf'{v}\b', body):
                continue
            # `?t = <types/X>` or `?t IN (...types/X...)`
            if re.search(rf'{v}\s*=\s*<{types_uri}\w+>', body):
                return True
            if re.search(rf'{v}\s+IN\s*\(', body, flags=re.IGNORECASE) and re.search(
                rf'<{types_uri}\w+>', body
            ):
                return True
        return False

    constrained = {v for v in type_obj_vars if _is_constrained_to_type(v)}
    if not constrained:
        return sparql

    # 2) Upgrade ONLY the rdf:type predicate of triples whose object is a
    #    constrained variable. Leave the variable in place.
    for var in constrained:
        v = re.escape(var)
        sparql = re.sub(
            rf'(\?\w+)\s+{bare_type_pred}\s+({v})\b',
            rf'\1 {_CLOSURE_PATH} \2',
            sparql,
        )

    return sparql


def add_layer_from_clauses(sparql: str, graph_uris: list[str]) -> str:
    """Add FROM <g> clauses for layer graphs missing from a graph-scoped query.

    Generated NL queries are scoped to one data graph (`FROM <data-graph>`),
    but with ontology layers (ADR 0002 §1) the subClassOf edges that the
    closure path `rdf:type/rdfs:subClassOf*` walks may live in OTHER layer
    graphs (a tenant leaf under a Public parent). Multiple FROM clauses make
    the default graph the union of all of them, so the closure walk sees every
    visible layer.

    Pure string transform, idempotent: a graph already in a FROM clause is not
    added twice. Queries with no FROM and no WHERE (shapes we don't understand)
    are returned unchanged. With an empty `graph_uris` the input is untouched —
    the single-graph call path stays byte-identical.
    """
    import re

    missing = [
        g for g in graph_uris
        if not re.search(rf'FROM\s+<{re.escape(g)}>', sparql)
    ]
    if not missing:
        return sparql
    extra = " ".join(f"FROM <{g}>" for g in missing)

    # Insert after the last existing FROM clause, else just before WHERE.
    from_matches = list(re.finditer(r'FROM\s+<[^>]+>', sparql))
    if from_matches:
        end = from_matches[-1].end()
        return f"{sparql[:end]} {extra}{sparql[end:]}"
    where = re.search(r'\bWHERE\b', sparql, flags=re.IGNORECASE)
    if where:
        start = where.start()
        return f"{sparql[:start]}{extra}\n{sparql[start:]}"
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
