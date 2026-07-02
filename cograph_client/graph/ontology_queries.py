"""SPARQL query builders for ontology management."""

OMNIX_ONTO = "https://cograph.tech/onto"
RDFS = "http://www.w3.org/2000/01/rdf-schema"
RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns"
XSD = "http://www.w3.org/2001/XMLSchema"
# OGC GeoSPARQL — the standard vocabulary for geometry literals. A `geo` attribute
# carries the range ``geo:wktLiteral`` and stores its value as WKT ("POINT(lon lat)")
# so coordinates are a first-class, datatype-tagged literal the spatio-temporal index
# can read directly (rather than guessing from attribute names at read time).
GEOSPARQL = "http://www.opengis.net/ont/geosparql"


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


def upsert_type(graph_uri: str, name: str, description: str = "", parent_type: str | None = None) -> str:
    """Atomically UPSERT a type declaration — idempotent under agent retries.

    Unlike :func:`insert_type` (blind ``INSERT DATA``), this REPLACES the
    single-valued ``rdfs:comment`` and ``rdfs:subClassOf`` instead of appending,
    so re-asserting a *changed* description or parent does not leave a second
    stale triple behind.

    Predicate handling:
      - ``rdf:type rdfs:Class`` and ``rdfs:label`` are plain idempotent
        ``INSERT DATA`` (re-asserting an identical triple is a no-op in RDF).
      - ``rdfs:comment`` and ``rdfs:subClassOf`` are SINGLE-VALUED and emitted as
        atomic ``DELETE/INSERT/WHERE`` operations: the old value is removed and
        the new one set in one update.

    Empty-description / None-parent semantics (authoritative upsert): if
    ``description`` is empty or ``parent_type`` is None we still DELETE any
    existing value (clearing it) but do NOT INSERT a replacement. The resulting
    graph state therefore reflects exactly the arguments passed — an upsert with
    no description never leaves a stale comment, and clearing a parent un-roots
    the type. The multi-operation update string separates each DELETE/INSERT/
    WHERE block with ``;``.
    """
    uri = type_uri(name)

    # Plain idempotent inserts: rdf:type rdfs:Class + rdfs:label.
    insert_block = (
        f"INSERT DATA {{\n"
        f"  GRAPH <{graph_uri}> {{\n"
        f'    <{uri}> <{RDF}#type> <{RDFS}#Class> .\n'
        f'    <{uri}> <{RDFS}#label> "{_esc(name)}" .\n'
        f"  }}\n"
        f"}}"
    )
    ops = [insert_block]

    # Single-valued rdfs:comment: delete old, insert new only if non-empty.
    if description:
        comment_insert = f"INSERT {{ GRAPH <{graph_uri}> {{ <{uri}> <{RDFS}#comment> \"{_esc(description)}\" }} }}\n"
    else:
        comment_insert = ""
    ops.append(
        f"DELETE {{ GRAPH <{graph_uri}> {{ <{uri}> <{RDFS}#comment> ?c }} }}\n"
        f"{comment_insert}"
        f"WHERE {{ GRAPH <{graph_uri}> {{ OPTIONAL {{ <{uri}> <{RDFS}#comment> ?c }} }} }}"
    )

    # Single-valued rdfs:subClassOf: delete old, insert new only if parent given.
    if parent_type:
        parent_insert = f"INSERT {{ GRAPH <{graph_uri}> {{ <{uri}> <{RDFS}#subClassOf> <{type_uri(parent_type)}> }} }}\n"
    else:
        parent_insert = ""
    ops.append(
        f"DELETE {{ GRAPH <{graph_uri}> {{ <{uri}> <{RDFS}#subClassOf> ?p }} }}\n"
        f"{parent_insert}"
        f"WHERE {{ GRAPH <{graph_uri}> {{ OPTIONAL {{ <{uri}> <{RDFS}#subClassOf> ?p }} }} }}"
    )

    return " ;\n".join(ops)


def upsert_type_comment(graph_uri: str, name: str, description: str = "") -> str:
    """Idempotently set ONLY a type's ``rdfs:comment`` (single-valued), leaving
    ``rdfs:subClassOf`` and every other triple untouched.

    Unlike :func:`upsert_type` — which also REPLACES ``rdfs:subClassOf`` and so
    CLEARS it when called with no ``parent_type`` — this never touches the
    hierarchy. Writing a subtype's description must not be able to wipe a
    ``subClassOf`` edge that a separate step (``insert_subtype`` /
    ``_synthesize_ancestors``) just created. Re-asserts ``rdf:type rdfs:Class`` +
    ``rdfs:label`` idempotently so the type exists; an empty ``description``
    clears any existing comment without inserting a replacement.
    """
    uri = type_uri(name)
    insert_block = (
        f"INSERT DATA {{\n"
        f"  GRAPH <{graph_uri}> {{\n"
        f'    <{uri}> <{RDF}#type> <{RDFS}#Class> .\n'
        f'    <{uri}> <{RDFS}#label> "{_esc(name)}" .\n'
        f"  }}\n"
        f"}}"
    )
    comment_insert = (
        f"INSERT {{ GRAPH <{graph_uri}> {{ <{uri}> <{RDFS}#comment> \"{_esc(description)}\" }} }}\n"
        if description
        else ""
    )
    comment_block = (
        f"DELETE {{ GRAPH <{graph_uri}> {{ <{uri}> <{RDFS}#comment> ?c }} }}\n"
        f"{comment_insert}"
        f"WHERE {{ GRAPH <{graph_uri}> {{ OPTIONAL {{ <{uri}> <{RDFS}#comment> ?c }} }} }}"
    )
    return f"{insert_block} ;\n{comment_block}"


def upsert_attribute(graph_uri: str, type_name: str, attr_name: str, description: str = "", datatype: str = "string") -> str:
    """Atomically UPSERT an attribute declaration — idempotent under agent retries.

    Unlike :func:`insert_attribute` (blind ``INSERT DATA``), this REPLACES the
    single-valued ``rdfs:range`` and ``rdfs:comment`` instead of appending. This
    matters because ``rdfs:range`` flips between an XSD primitive and a
    ``types/`` URI when an attribute is later seen carrying an entity value (i.e.
    becomes a relationship); a blind re-insert would leave the property with two
    conflicting ranges.

    Predicate handling:
      - ``rdf:type rdf:Property``, ``rdfs:label`` and ``rdfs:domain`` are plain
        idempotent ``INSERT DATA`` (re-asserting identical triples is a no-op).
      - ``rdfs:range`` and ``rdfs:comment`` are SINGLE-VALUED and emitted as
        atomic ``DELETE/INSERT/WHERE`` operations.

    ``rdfs:range`` is always set (``_datatype_to_xsd`` maps a primitive name to
    an XSD URI and any other name to that type's ``types/`` URI), so the range
    block always inserts a fresh value after deleting the old one. ``rdfs:comment``
    follows the same clear-on-empty rule as :func:`upsert_type`: an empty
    ``description`` deletes any existing comment without inserting a replacement.
    """
    t_uri = type_uri(type_name)
    a_uri = attr_uri(type_name, attr_name)
    xsd_type = _datatype_to_xsd(datatype)

    # Plain idempotent inserts: rdf:type rdf:Property + rdfs:label + rdfs:domain.
    insert_block = (
        f"INSERT DATA {{\n"
        f"  GRAPH <{graph_uri}> {{\n"
        f'    <{a_uri}> <{RDF}#type> <{RDF}#Property> .\n'
        f'    <{a_uri}> <{RDFS}#label> "{_esc(attr_name)}" .\n'
        f'    <{a_uri}> <{RDFS}#domain> <{t_uri}> .\n'
        f"  }}\n"
        f"}}"
    )
    ops = [insert_block]

    # Single-valued rdfs:range: always replaced (range is always known).
    ops.append(
        f"DELETE {{ GRAPH <{graph_uri}> {{ <{a_uri}> <{RDFS}#range> ?r }} }}\n"
        f"INSERT {{ GRAPH <{graph_uri}> {{ <{a_uri}> <{RDFS}#range> <{xsd_type}> }} }}\n"
        f"WHERE {{ GRAPH <{graph_uri}> {{ OPTIONAL {{ <{a_uri}> <{RDFS}#range> ?r }} }} }}"
    )

    # Single-valued rdfs:comment: delete old, insert new only if non-empty.
    if description:
        comment_insert = f"INSERT {{ GRAPH <{graph_uri}> {{ <{a_uri}> <{RDFS}#comment> \"{_esc(description)}\" }} }}\n"
    else:
        comment_insert = ""
    ops.append(
        f"DELETE {{ GRAPH <{graph_uri}> {{ <{a_uri}> <{RDFS}#comment> ?c }} }}\n"
        f"{comment_insert}"
        f"WHERE {{ GRAPH <{graph_uri}> {{ OPTIONAL {{ <{a_uri}> <{RDFS}#comment> ?c }} }} }}"
    )

    return " ;\n".join(ops)


def set_object_property_range(graph_uri: str, type_name: str, attr_name: str, target_type: str) -> str:
    """Re-point an existing property's ``rdfs:range`` at a type URI.

    Used to UPGRADE a predicate that was first registered as a primitive
    attribute (range ``xsd:string`` etc.) once it is later seen carrying an
    entity-valued object — i.e. it is really a relationship to ``target_type``.
    Without this the schema-only Explorer overview can't see the edge (it keys
    on ``rdfs:range`` being a ``types/`` URI), even though the instance triple
    exists. Deletes any existing range first so the property keeps exactly one.
    """
    a_uri = attr_uri(type_name, attr_name)
    rng = type_uri(target_type)
    return (
        f"DELETE {{ GRAPH <{graph_uri}> {{ <{a_uri}> <{RDFS}#range> ?old }} }}\n"
        f"INSERT {{ GRAPH <{graph_uri}> {{ <{a_uri}> <{RDFS}#range> <{rng}> }} }}\n"
        f"WHERE {{ GRAPH <{graph_uri}> {{ OPTIONAL {{ <{a_uri}> <{RDFS}#range> ?old }} }} }}"
    )


def retract_object_property(graph_uri: str, type_name: str, attr_name: str) -> str:
    """Retract a type-level object-property declaration to QUARANTINE (ADR 0004 §4).

    The inverse of :func:`set_object_property_range`: when reconciliation finds a
    non-core relationship whose live support has fallen below the drift floor, it
    must stop being a *declared* type-level edge. The schema-only Explorer
    overview keys an edge on the property having an ``rdfs:range`` that points at
    a ``types/`` URI and an ``rdfs:domain`` on the source type; deleting BOTH
    triples removes the edge from the overview while leaving the underlying
    instance triples untouched (row conservation, ADR 0003 §2 — only the schema
    *declaration* is withheld, the data still ingests).

    Quarantine-not-delete (ADR 0004 §2) is the *caller's* responsibility — it
    records support/source/timestamp in the quarantine store before issuing this
    retraction. This builder is the deterministic SPARQL half: it removes exactly
    the range and domain triples for ``attr_name`` and nothing else. The
    ``OPTIONAL`` wrappers make it a no-op-safe retraction (a property already
    missing its range or domain still retracts cleanly, no error).
    """
    a_uri = attr_uri(type_name, attr_name)
    return (
        f"DELETE {{ GRAPH <{graph_uri}> {{\n"
        f"  <{a_uri}> <{RDFS}#range> ?range .\n"
        f"  <{a_uri}> <{RDFS}#domain> ?domain .\n"
        f"}} }}\n"
        f"WHERE {{ GRAPH <{graph_uri}> {{\n"
        f"  OPTIONAL {{ <{a_uri}> <{RDFS}#range> ?range }}\n"
        f"  OPTIONAL {{ <{a_uri}> <{RDFS}#domain> ?domain }}\n"
        f"}} }}"
    )


#: Canonical value of the ``textKind`` marker for free-running prose
#: attributes (ONTA-177). Kept as a plain literal (not a boolean) so future
#: kinds ("code", "label", …) can share the same single-valued predicate.
TEXT_KIND_FREE_TEXT = "free_text"

#: Canonical value of the ``textKind`` marker for a DURABLE decided-NO
#: candidacy verdict (ONTA-173): the candidacy tier genuinely ADJUDICATED this
#: attribute (the LLM REASON layer declined a TEXT-shaped column, or the
#: reconciler's name-blind heuristic classified it NOT_CANDIDATE) and found it
#: not free text. Persisting the NO matters: absence from the marker map means
#: "never decided" — the reconciler would re-sample the attribute every run,
#: and the name-blind ≥120-char auto tier could later overrule the LLM's
#: explicit NO. In ``text_markers.get_free_text_map`` any kind other than
#: ``free_text`` reads back as ``is_free_text=False`` while remaining PRESENT
#: in the map (presence = decided), which is exactly the skip signal the
#: reconciler keys on. NOTE: ``semantic/reconciler.py`` predates this constant
#: and carries a same-valued local duplicate (``TEXT_KIND_NOT_TEXT`` at module
#: scope) — converge it onto this one in a follow-up.
TEXT_KIND_NOT_TEXT = "not_text"


def upsert_attribute_text_kind(
    graph_uri: str, type_name: str, attr_name: str, text_kind: str = TEXT_KIND_FREE_TEXT,
) -> str:
    """Idempotently set ONLY an attribute's ``<onto/textKind>`` marker
    (single-valued), leaving every other triple of the property untouched.

    ONTA-177: schema-time free-text candidacy (profiler ``ValueShape.TEXT``
    proposes, the LLM REASON layer adjudicates ambiguous cases) is persisted
    as ``<attr_uri> <onto/textKind> "free_text"`` so the semantic instance
    index (ONTA-173) and the query-side type filter (ONTA-176) can read the
    verdict without re-deciding it. Follows the same atomic
    ``DELETE/INSERT/WHERE`` pattern as :func:`upsert_type_comment` /
    :func:`upsert_attribute` for single-valued predicates: re-ingesting the
    same file replaces the marker instead of stacking duplicates, and a
    changed verdict (e.g. a future re-adjudication) never leaves two
    conflicting kinds behind. An empty ``text_kind`` clears any existing
    marker without inserting a replacement (the clear-on-empty rule the rest
    of this module's upserts use).
    """
    a_uri = attr_uri(type_name, attr_name)
    if text_kind:
        kind_insert = (
            f"INSERT {{ GRAPH <{graph_uri}> {{ <{a_uri}> <{OMNIX_ONTO}/textKind> \"{_esc(text_kind)}\" }} }}\n"
        )
    else:
        kind_insert = ""
    return (
        f"DELETE {{ GRAPH <{graph_uri}> {{ <{a_uri}> <{OMNIX_ONTO}/textKind> ?k }} }}\n"
        f"{kind_insert}"
        f"WHERE {{ GRAPH <{graph_uri}> {{ OPTIONAL {{ <{a_uri}> <{OMNIX_ONTO}/textKind> ?k }} }} }}"
    )


def text_kind_map_query(graph_uri: str) -> str:
    """Select every ``textKind`` marker in a graph: ``?attr ?kind`` rows.

    Feeds the per-tenant ``{attribute predicate URI -> is_free_text}`` cache in
    :mod:`cograph_client.graph.text_markers` (ONTA-177), which query-side
    consumers (semantic instance index routing, ONTA-176) read instead of
    hitting Neptune per request.
    """
    return (
        f"SELECT ?attr ?kind FROM <{graph_uri}>\n"
        f"WHERE {{\n"
        f"  ?attr <{OMNIX_ONTO}/textKind> ?kind .\n"
        f"}}"
    )


def mark_core_slot(graph_uri: str, type_name: str, slot_name: str) -> str:
    """Mark one of ``type_name``'s declared attributes/relationship slots as
    CONSTITUTIVE (a core slot, ADR 0003 §3 / Pass D).

    Emits ``<attr_uri> <onto/coreSlot> "true"^^xsd:boolean``. Core slots may
    have zero data in the ingested file — the marker is what lets enrichment
    later query "instances with empty core slots" as its work queue, and what
    the governance pipeline (COG-56) keys its review on.
    """
    a_uri = attr_uri(type_name, slot_name)
    return (
        f"INSERT DATA {{\n"
        f"  GRAPH <{graph_uri}> {{\n"
        f'    <{a_uri}> <{OMNIX_ONTO}/coreSlot> "true"^^<{XSD}#boolean> .\n'
        f"  }}\n"
        f"}}"
    )


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


def get_attribute_range_query(graph_uri: str, type_name: str, attr_name: str) -> str:
    """Fetch the single ``rdfs:range`` currently declared for one attribute.

    Returns ``?range`` (zero rows if the attribute / its range is undeclared).
    Used by enrichment to decide whether declaring an enriched attribute would
    DOWNGRADE an existing richer range (an XSD primitive like ``xsd:integer`` or
    a relationship ``types/<Target>`` URI) down to ``xsd:string`` — it must not.
    """
    a_uri = attr_uri(type_name, attr_name)
    return (
        f"SELECT ?range FROM <{graph_uri}>\n"
        f"WHERE {{\n"
        f"  <{a_uri}> <{RDFS}#range> ?range .\n"
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


PRIMITIVE_TYPES = {"string", "integer", "float", "boolean", "datetime", "uri", "geo"}


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


# The XSD ``rdfs:range`` URI an attribute carries when it is a plain string
# attribute — the weakest primitive range, the only one enrichment is allowed to
# overwrite (a string range is "untyped enough" that an inferred richer type is a
# strict improvement; anything else is a downgrade we must preserve).
XSD_STRING = f"{XSD}#string"

_DATATYPE_TO_XSD = {
    "string": f"{XSD}#string",
    "integer": f"{XSD}#integer",
    "float": f"{XSD}#float",
    "boolean": f"{XSD}#boolean",
    "datetime": f"{XSD}#dateTime",
    "uri": f"{RDFS}#Resource",
    # WGS84 point/geometry as WKT — read by the spatio-temporal index.
    "geo": f"{GEOSPARQL}#wktLiteral",
}


def _datatype_to_xsd(datatype: str) -> str:
    if datatype in _DATATYPE_TO_XSD:
        return _DATATYPE_TO_XSD[datatype]
    # Treat as a reference to another type
    return type_uri(datatype)


def xsd_to_datatype(range_uri: str) -> str:
    """Reverse of :func:`_datatype_to_xsd`: map a declared ``rdfs:range`` URI back
    to the ``datatype`` name :func:`upsert_attribute` accepts, so an existing
    range can be RE-asserted verbatim.

    A primitive XSD/Resource URI maps to its name (``…#integer`` -> ``integer``);
    a ``types/<X>`` relationship URI maps to the bare type name ``X`` (which
    ``_datatype_to_xsd`` round-trips back to the same ``types/<X>`` URI). Any
    other/unknown URI falls back to ``string`` so a malformed range can never
    crash a declaration."""
    for name, uri in _DATATYPE_TO_XSD.items():
        if range_uri == uri:
            return name
    if range_uri.startswith(_TYPES_URI):
        return range_uri[len(_TYPES_URI):].rstrip("/")
    return "string"
