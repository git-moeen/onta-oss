"""Execute a confirmed normalization rule (v1: list_explode).

:func:`apply_rule` rewrites a KG graph in place to split collapsed multi-value
cells into atomic ones. Two shapes:

* **relationship, target=entity** (the ``speaks`` case): an edge points at a
  COMPOSITE entity whose local-name/label packs several atomic values joined by
  a delimiter (``…/entities/Language/English__Russian__Ukrainian``). We split
  it, mint a CANONICAL atomic entity IRI per atomic value (slug-derived, so
  "Russian" from any composite maps to the SAME node — free dedup, no ER pass),
  re-point the edge at the atomic entities, drop the composite edge, and finally
  drop any composite entity left with no inbound edge for this predicate.

* **attribute, target=literal** (the skills/disciplines case): a literal packs
  several items with a delimiter. We split into atomic literals, write N triples,
  and remove the original packed literal.

Idempotent: re-running finds nothing to split (the values are already atomic) and
is a no-op. Returns ``{edges_rewritten, atomic_created, orphans_dropped}``.
"""

from __future__ import annotations

import re

import structlog

from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.ontology_queries import RDF, RDFS, type_uri
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import (
    batched_insert_triples,
    delete_triples,
    kg_graph_uri,
)

logger = structlog.stdlib.get_logger("cograph.normalization.execute")

RDF_TYPE = f"{RDF}#type"
RDFS_LABEL = f"{RDFS}#label"
ENTITY_URI_PREFIX = "https://cograph.tech/entities/"
ATTRS_INFIX = "/attrs/"
ONTO_PRED_PREFIX = "https://cograph.tech/onto/"
NAME_ATTR_SUFFIX = "/attrs/name"

# Slug-aware delimiters: the slug "__" is the de-slugified form of a source-list
# separator (", " etc.). We keep it last so we try the longer composite-name
# split form too. Each is a literal substring to split on.
_FALLBACK_DELIMITERS = [", ", "; ", " / ", " | ", " - ", "__"]


async def apply_rule(neptune: NeptuneClient, tenant_id: str, rule) -> dict:
    """Apply a confirmed list_explode rule. Returns a summary dict."""
    if rule.rule_type != "list_explode":
        raise ValueError(f"unsupported rule_type {rule.rule_type!r} (v1: list_explode)")

    kg_graph = kg_graph_uri(tenant_id, rule.kg_name)
    delimiters = _delimiters(rule)
    target = (rule.params or {}).get("target")
    pred_leaf = rule.predicate

    if rule.target_kind == "relationship" or target == "entity":
        if rule.target_kind == "attribute" and target == "entity":
            # attribute -> atomic ENTITIES is noted as a follow-up; v1 focuses on
            # the relationship+literal cases (see module docstring / PR notes).
            logger.info(
                "attribute_to_entity_unsupported",
                predicate=pred_leaf,
                note="promotion of attribute literals to atomic entities is a follow-up",
            )
            return {"edges_rewritten": 0, "atomic_created": 0, "orphans_dropped": 0}
        return await _explode_relationship(neptune, kg_graph, pred_leaf, delimiters)
    return await _explode_literal(neptune, kg_graph, pred_leaf, delimiters)


def _delimiters(rule) -> list[str]:
    delims = list((rule.params or {}).get("delimiters") or [])
    # Always include the slug "__" — composite entity names use it even when the
    # source literal used ", " (the slugifier maps both to "__").
    for d in _FALLBACK_DELIMITERS:
        if d not in delims:
            delims.append(d)
    # Longest-first so " / " is tried before "/" etc. — avoids splitting inside a
    # token that legitimately contains the shorter delimiter.
    return sorted(set(delims), key=len, reverse=True)


def _split(value: str, delimiters: list[str]) -> list[str]:
    """Split ``value`` on any of the delimiters into trimmed, de-duped atoms.

    Returns the original single-element list when no delimiter is present (i.e.
    the value is already atomic — the idempotency guarantee).
    """
    # Build one regex alternation of the (escaped) delimiters, longest first.
    pattern = "|".join(re.escape(d) for d in sorted(delimiters, key=len, reverse=True))
    if not pattern:
        return [value.strip()] if value.strip() else []
    parts = re.split(pattern, value)
    atoms: list[str] = []
    seen: set[str] = set()
    for p in parts:
        a = p.strip()
        if a and a not in seen:
            seen.add(a)
            atoms.append(a)
    return atoms


def _slug(value: str) -> str:
    """Canonical slug for an atomic value → its canonical atomic entity IRI tail.

    Same character class as the resolver's ``_safe_id`` so atomic IRIs line up
    with how the rest of the system mints entity IRIs. Deterministic, so the same
    atomic value always maps to the same node (free dedup).
    """
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", value.strip())
    return safe[:200] if safe else "unknown"


def _target_type_from_uri(composite_uri: str) -> str | None:
    """``…/entities/<TargetType>/<slug>`` → ``<TargetType>``."""
    if not composite_uri.startswith(ENTITY_URI_PREFIX):
        return None
    tail = composite_uri[len(ENTITY_URI_PREFIX):]
    head = tail.split("/", 1)[0]
    return head or None


async def _explode_relationship(
    neptune: NeptuneClient, kg_graph: str, pred_leaf: str, delimiters: list[str]
) -> dict:
    """Split composite relationship targets into canonical atomic entities."""
    onto_pred = ONTO_PRED_PREFIX + pred_leaf
    attr_pred_suffix = ATTRS_INFIX + pred_leaf  # any …/attrs/<leaf> form

    # 1) Find every (subject, predicate-as-used, composite) edge whose object is a
    #    composite entity. We match BOTH the onto/<leaf> predicate (the normal
    #    relationship form) and any types/<T>/attrs/<leaf> predicate (a predicate
    #    first seen as an attribute then carrying an entity object). The composite
    #    is identified by its name/label containing a delimiter.
    delim_filter = " || ".join(
        f'CONTAINS(?cname, "{_sparql_str(d)}")' for d in delimiters
    )
    q = (
        f"SELECT ?s ?p ?composite ?clabel FROM <{kg_graph}> WHERE {{\n"
        f"  ?s ?p ?composite .\n"
        f"  FILTER(?p = <{onto_pred}> || STRENDS(STR(?p), \"{_sparql_str(attr_pred_suffix)}\"))\n"
        f'  FILTER(STRSTARTS(STR(?composite), "{ENTITY_URI_PREFIX}"))\n'
        f"  OPTIONAL {{ ?composite <{RDFS_LABEL}> ?clabel }}\n"
        f'  BIND(COALESCE(?clabel, REPLACE(STR(?composite), "^.*/", "")) AS ?cname)\n'
        f"  FILTER({delim_filter})\n"
        f"}}"
    )
    _, rows = parse_sparql_results(await neptune.query(q))

    edges_to_delete: list[tuple[str, str, str]] = []
    edges_to_add: list[tuple[str, str, str]] = []
    atomic_triples: list[tuple[str, str, str]] = []
    atomic_seen: set[str] = set()
    composites_touched: set[str] = set()

    for r in rows:
        s = r.get("s", "")
        p = r.get("p", "")
        composite = r.get("composite", "")
        if not s or not p or not composite:
            continue
        target_type = _target_type_from_uri(composite)
        if not target_type:
            continue
        # Prefer the rdfs:label (the human value) for the split; fall back to the
        # URL-decoded local-name. The "__" slug split recovers atoms from names.
        clabel = r.get("clabel", "")
        source = clabel or _decode_local_name(composite)
        atoms = _split(source, delimiters)
        if len(atoms) <= 1:
            # Already atomic for this composite — nothing to do (idempotency).
            continue
        composites_touched.add(composite)
        # Re-point the edge to one CANONICAL atomic entity per atom; the canonical
        # IRI is slug-derived so the same atom (e.g. "Russian") from any composite
        # maps to the SAME node. Always re-point using the onto/<leaf> predicate
        # (the proper relationship form) regardless of the predicate as-used.
        for atom in atoms:
            atom_uri = f"{ENTITY_URI_PREFIX}{target_type}/{_slug(atom)}"
            edges_to_add.append((s, onto_pred, atom_uri))
            if atom_uri not in atomic_seen:
                atomic_seen.add(atom_uri)
                atomic_triples.append((atom_uri, RDF_TYPE, type_uri(target_type)))
                atomic_triples.append((atom_uri, RDFS_LABEL, atom))
                # Mirror ingest: also store the human value under attrs/name so the
                # Explorer Data table shows it (see explore.get_type_records).
                atomic_triples.append(
                    (atom_uri, type_uri(target_type) + "/attrs/name", atom)
                )
        edges_to_delete.append((s, p, composite))

    # 2) Apply: add atomic entity triples + new edges, then delete composite edges.
    if atomic_triples:
        for sparql in batched_insert_triples(kg_graph, atomic_triples):
            await neptune.update(sparql)
    if edges_to_add:
        for sparql in batched_insert_triples(kg_graph, edges_to_add):
            await neptune.update(sparql)
    if edges_to_delete:
        # delete_triples batches all in one DELETE DATA; chunk to stay safe.
        for i in range(0, len(edges_to_delete), 500):
            await neptune.update(delete_triples(kg_graph, edges_to_delete[i : i + 500]))

    # 3) Drop now-orphaned composite entities (no inbound edge for this predicate).
    orphans_dropped = await _drop_orphan_composites(
        neptune, kg_graph, onto_pred, attr_pred_suffix, composites_touched
    )

    summary = {
        "edges_rewritten": len(edges_to_delete),
        "atomic_created": len(atomic_seen),
        "orphans_dropped": orphans_dropped,
    }
    logger.info("explode_relationship_done", predicate=pred_leaf, **summary)
    return summary


async def _drop_orphan_composites(
    neptune: NeptuneClient,
    kg_graph: str,
    onto_pred: str,
    attr_pred_suffix: str,
    composites: set[str],
) -> int:
    """Delete composite entities with no remaining inbound edge for the predicate.

    Removes ALL triples of each orphan (its rdf:type, labels, attrs). Only the
    composites we actually re-pointed are candidates; each is re-checked live so
    a composite still referenced elsewhere is kept.
    """
    dropped = 0
    for composite in composites:
        ask = (
            f"ASK FROM <{kg_graph}> WHERE {{\n"
            f"  {{ ?s <{onto_pred}> <{composite}> }}\n"
            f"  UNION\n"
            f"  {{ ?s ?p2 <{composite}> . FILTER(STRENDS(STR(?p2), \"{_sparql_str(attr_pred_suffix)}\")) }}\n"
            f"}}"
        )
        try:
            still_referenced = await neptune.ask(ask)
        except Exception:
            logger.warning("orphan_check_failed", composite=composite, exc_info=True)
            continue
        if still_referenced:
            continue
        # No inbound predicate edge — delete the composite node entirely.
        await neptune.update(
            f"DELETE {{ GRAPH <{kg_graph}> {{ <{composite}> ?p ?o }} }}\n"
            f"WHERE {{ GRAPH <{kg_graph}> {{ <{composite}> ?p ?o }} }}"
        )
        dropped += 1
    return dropped


async def _explode_literal(
    neptune: NeptuneClient, kg_graph: str, pred_leaf: str, delimiters: list[str]
) -> dict:
    """Split packed attribute literals into N atomic literals."""
    onto_pred = ONTO_PRED_PREFIX + pred_leaf
    attr_pred_suffix = ATTRS_INFIX + pred_leaf

    delim_filter = " || ".join(
        f'CONTAINS(STR(?o), "{_sparql_str(d)}")' for d in delimiters
    )
    q = (
        f"SELECT ?s ?p ?o FROM <{kg_graph}> WHERE {{\n"
        f"  ?s ?p ?o .\n"
        f"  FILTER(?p = <{onto_pred}> || STRENDS(STR(?p), \"{_sparql_str(attr_pred_suffix)}\"))\n"
        f"  FILTER(isLiteral(?o))\n"
        f"  FILTER({delim_filter})\n"
        f"}}"
    )
    _, rows = parse_sparql_results(await neptune.query(q))

    to_delete: list[tuple[str, str, str]] = []
    to_add: list[tuple[str, str, str]] = []
    rewritten = 0
    atomic_count = 0
    for r in rows:
        s = r.get("s", "")
        p = r.get("p", "")
        o = r.get("o", "")
        if not s or not p:
            continue
        atoms = _split(o, delimiters)
        if len(atoms) <= 1:
            continue  # already atomic — idempotent no-op
        for atom in atoms:
            to_add.append((s, p, atom))
            atomic_count += 1
        to_delete.append((s, p, o))
        rewritten += 1

    if to_add:
        for sparql in batched_insert_triples(kg_graph, to_add):
            await neptune.update(sparql)
    if to_delete:
        for i in range(0, len(to_delete), 500):
            await neptune.update(delete_triples(kg_graph, to_delete[i : i + 500]))

    summary = {
        "edges_rewritten": rewritten,
        "atomic_created": atomic_count,
        "orphans_dropped": 0,
    }
    logger.info("explode_literal_done", predicate=pred_leaf, **summary)
    return summary


def _decode_local_name(uri: str) -> str:
    """The local-name of an entity URI, percent-decoded (best-effort)."""
    from urllib.parse import unquote

    tail = uri.rstrip("/").split("/")[-1]
    return unquote(tail)


def _sparql_str(s: str) -> str:
    """Escape a Python string for embedding inside a SPARQL double-quoted literal.

    Used for CONTAINS/STRENDS argument literals — the only place we splice a
    delimiter/suffix into a query. Escapes backslash, quote, and newline.
    """
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
