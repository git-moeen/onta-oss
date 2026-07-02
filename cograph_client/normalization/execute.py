"""Execute a confirmed normalization rule.

:func:`apply_rule` rewrites a KG graph in place. Two rule types ship today.

``list_explode`` splits collapsed multi-value cells into atomic ones. Two shapes:

* **relationship, target=entity** (the ``speaks`` case): an edge points at a
  COMPOSITE entity whose local-name/label packs several atomic values joined by
  a delimiter (``…/entities/Language/English__Russian__Ukrainian``). We split
  it, mint a CANONICAL atomic entity IRI per atomic value (slug-derived, so
  "Russian" from any composite maps to the SAME node — free dedup, no ER pass),
  re-point the edge at the atomic entities, drop the composite edge, and finally
  run a single graph-state-keyed orphan sweep that deletes EVERY composite
  entity of the target type left with no inbound edge for this predicate. The
  sweep is keyed on graph state (not the edges this pass touched), so it is
  complete (catches composites an inline per-edge drop would miss) and
  re-runnable (a later apply still sweeps leftovers from a buggy earlier run).
  The sweep's target type is resolved from the ONTOLOGY — the predicate's
  declared ``rdfs:range`` (``speaks → Language``), a bounded single-subject
  lookup — so a pure re-run with ``edges_rewritten == 0`` still resolves the type
  and sweeps lingering orphans to zero, with no unbounded full-graph scan
  (COG-118).

* **attribute, target=literal** (the skills/disciplines case): a literal packs
  several items with a delimiter. We split into atomic literals, write N triples,
  and remove the original packed literal.

``strip_emoji`` removes emoji / pictographic junk characters from text literals
(the ``skills = "🎨 design"`` case): for each matching ``attrs/<pred>`` (or
``onto/<pred>``) literal we strip emoji codepoints + variation selectors + ZWJ +
skin-tone modifiers, collapse the leftover whitespace, and rewrite ONLY the
literals that actually changed. A value with no emoji is untouched (idempotent
re-run is a no-op). A literal that becomes empty after stripping (a pure-emoji
value) is dropped. It operates per-literal, so it works whether ``skills`` is
still one packed literal or already exploded into atomic literals.

Idempotent: re-running finds nothing to change (values are already atomic /
emoji-free) and is a no-op. ``list_explode`` returns
``{edges_rewritten, atomic_created, orphans_dropped}``; ``strip_emoji`` returns
``{literals_cleaned, triples_rewritten}``.
"""

from __future__ import annotations

import re

import structlog

from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.kg_writer import delete_facts, insert_facts, refresh_after_write
from cograph_client.graph.ontology_queries import RDF, RDFS, attr_uri, type_uri
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import (
    kg_graph_uri,
    tenant_graph_uri,
)

logger = structlog.stdlib.get_logger("cograph.normalization.execute")

RDF_TYPE = f"{RDF}#type"
RDFS_LABEL = f"{RDFS}#label"
RDFS_RANGE = f"{RDFS}#range"
ENTITY_URI_PREFIX = "https://cograph.tech/entities/"
ATTRS_INFIX = "/attrs/"
ONTO_PRED_PREFIX = "https://cograph.tech/onto/"
NAME_ATTR_SUFFIX = "/attrs/name"

# Slug-aware delimiters: the slug "__" is the de-slugified form of a source-list
# separator (", " etc.). We keep it last so we try the longer composite-name
# split form too. Each is a literal substring to split on.
_FALLBACK_DELIMITERS = [", ", "; ", " / ", " | ", " - ", "__"]

# Emoji / pictographic / junk codepoints to strip from text literals
# (strip_emoji). Scoped to the symbol/pictograph blocks so ordinary letters
# (incl. accented), digits, and real-skill-name punctuation (& + - / # . etc.)
# are left ALONE — e.g. "c++", "C#", "Node.js", "café", "R&D" survive intact.
#   U+200D            zero-width joiner (binds emoji sequences)
#   U+FE0E/U+FE0F     variation selectors (text/emoji presentation)
#   U+1F3FB–U+1F3FF   skin-tone modifiers
#   U+1F1E6–U+1F1FF   regional-indicator letters (flags)
#   U+2600–U+27BF     Misc Symbols + Dingbats
#   U+2B00–U+2BFF     Misc Symbols & Arrows (incl. ⭐ stars, ✅-adjacent)
#   U+1F000–U+1FAFF   the emoji/pictograph planes (Emoticons, Misc Symbols &
#                     Pictographs, Transport & Map, Supplemental, Symbols &
#                     Pictographs Extended-A, etc.)
#   U+2190–U+21FF     Arrows (decorative junk that shows up in scraped text)
#   U+2300–U+23FF     Misc Technical (⌚⏰ etc.)
#   U+2B50 etc. fall inside the ranges above.
_EMOJI_PATTERN = re.compile(
    "["
    "\U0000200d"
    "\U0000fe0e\U0000fe0f"
    "\U0001f3fb-\U0001f3ff"
    "\U0001f1e6-\U0001f1ff"
    "\U00002190-\U000021ff"
    "\U00002300-\U000023ff"
    "\U00002600-\U000027bf"
    "\U00002b00-\U00002bff"
    "\U0001f000-\U0001faff"
    "]+"
)
# Collapse the whitespace left behind once emoji are removed.
_WS_PATTERN = re.compile(r"\s+")


async def apply_rule(neptune: NeptuneClient, tenant_id: str, rule) -> dict:
    """Apply a confirmed rule (``list_explode`` or ``strip_emoji``). Returns a summary.

    On any apply that actually mutates the graph, fire the same fire-and-forget
    type-stats recompute enrichment uses (``schedule_recompute``) so the
    Explorer's precomputed counts don't go stale (COG-118). A pure no-op
    (idempotent re-run that changed nothing) skips it.
    """
    if rule.rule_type not in ("list_explode", "strip_emoji"):
        raise ValueError(
            f"unsupported rule_type {rule.rule_type!r} "
            f"(supported: list_explode, strip_emoji)"
        )

    kg_graph = kg_graph_uri(tenant_id, rule.kg_name)
    onto_graph = tenant_graph_uri(tenant_id)

    summary, deleted_subjects = await _dispatch(neptune, kg_graph, onto_graph, rule)

    if _summary_mutated(summary):
        # Shared post-write housekeeping (graph/kg_writer.py) — same path every
        # KG writer uses. affected_types=() because normalization changes instance
        # data + counts but NEVER the type SCHEMA (no new types/attributes), so no
        # re-embed is needed; the shared refresh still invalidates the NL-planning
        # cache and recomputes type-stats. deleted_subjects carries any orphan
        # composites the sweep removed (ADR 0007) so the SAME refresh evicts them
        # from the derived secondary indexes — no ghost rows left behind.
        await refresh_after_write(
            neptune,
            tenant_id=tenant_id,
            kg_name=rule.kg_name,
            affected_types=(),
            deleted_subjects=deleted_subjects,
        )
    return summary


async def _dispatch(
    neptune: NeptuneClient, kg_graph: str, onto_graph: str, rule
) -> tuple[dict, list[str]]:
    """Route to the rule-type handler; return ``(summary, deleted_subjects)``.

    ``deleted_subjects`` are the whole-entity URIs the handler removed (the orphan
    sweep's swept composites) so ``apply_rule``'s single refresh can evict them
    from derived indexes. Attribute/edge-level deletes (a subject survives, only
    some triples go) are NOT subjects here.
    """
    if rule.rule_type == "strip_emoji":
        return await _strip_emoji(neptune, kg_graph, rule)

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
            return {"edges_rewritten": 0, "atomic_created": 0, "orphans_dropped": 0}, []
        return await _explode_relationship(
            neptune, kg_graph, onto_graph, rule.type_name, pred_leaf, delimiters
        )
    return await _explode_literal(neptune, kg_graph, pred_leaf, delimiters)


def _summary_mutated(summary: dict) -> bool:
    """True iff this apply actually changed the graph (so a recompute is worth it).

    Covers both summary shapes: list_explode's counters and strip_emoji's. A
    purely idempotent re-run reports all-zero and we skip the recompute.
    """
    return any(
        int(summary.get(k, 0))
        for k in (
            "edges_rewritten",
            "atomic_created",
            "orphans_dropped",
            "triples_rewritten",
            "literals_cleaned",
        )
    )


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


def _atom_uri(target_type: str, atom: str) -> str:
    """Canonical atomic entity IRI for ``atom`` of ``target_type``.

    Single source of truth for how atomic IRIs are minted: ``…/entities/
    <TargetType>/<slug>``. Used both to RE-POINT an edge at the clean atomic node
    and to decide idempotency — the skip check compares an atom's canonical IRI
    to the composite's own IRI, so they MUST be minted the same way for the
    equality to be exact (COG-118).
    """
    return f"{ENTITY_URI_PREFIX}{target_type}/{_slug(atom)}"


def _target_type_from_uri(composite_uri: str) -> str | None:
    """``…/entities/<TargetType>/<slug>`` → ``<TargetType>``."""
    if not composite_uri.startswith(ENTITY_URI_PREFIX):
        return None
    tail = composite_uri[len(ENTITY_URI_PREFIX):]
    head = tail.split("/", 1)[0]
    return head or None


async def _explode_relationship(
    neptune: NeptuneClient,
    kg_graph: str,
    onto_graph: str,
    domain_type: str,
    pred_leaf: str,
    delimiters: list[str],
) -> tuple[dict, list[str]]:
    """Split composite relationship targets into canonical atomic entities.

    Returns ``(summary, orphan_uris)`` — the composite subjects the final sweep
    removed, so the caller's single refresh can evict them from derived indexes.
    """
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
        if not atoms:
            # Nothing to split (empty/whitespace-only source) — nothing to do.
            continue
        # Skip ONLY when the target is already a clean atomic node: a single atom
        # whose CANONICAL IRI is the composite's own IRI. That is the genuine
        # idempotency case (re-running on `…/Language/English` is a no-op). A
        # single atom whose canonical IRI DIFFERS from the composite's IRI means
        # the target carries a junk delimiter (leading/trailing/doubled, e.g.
        # `…/Industry/__Agriculture` → atom "Agriculture" → `…/Industry/Agriculture`)
        # and MUST be re-pointed to the clean node — same as the multi-atom path —
        # so the malformed node becomes a sweepable orphan (COG-118). The equality
        # uses the SAME minting helper as the re-point below, so the check is exact.
        if len(atoms) == 1 and _atom_uri(target_type, atoms[0]) == composite:
            continue
        composites_touched.add(composite)
        # Re-point the edge to one CANONICAL atomic entity per atom; the canonical
        # IRI is slug-derived so the same atom (e.g. "Russian") from any composite
        # maps to the SAME node. Always re-point using the onto/<leaf> predicate
        # (the proper relationship form) regardless of the predicate as-used.
        for atom in atoms:
            atom_uri = _atom_uri(target_type, atom)
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
        await insert_facts(neptune, kg_graph, atomic_triples)
    if edges_to_add:
        await insert_facts(neptune, kg_graph, edges_to_add)
    if edges_to_delete:
        # Concrete-triple removal via the shared primitive (ADR 0007); delete_facts
        # batches internally (no oversized statement). These are edge drops — the
        # subject survives — so they are NOT deleted_subjects.
        await delete_facts(
            neptune,
            kg_graph,
            triples=edges_to_delete,
            reason="normalization:list_explode composite-edge drop",
        )

    # 3) Final orphan sweep. After ALL edges for this predicate are re-pointed,
    #    delete EVERY composite node of the relationship's target type(s) that has
    #    no inbound onto/<pred> (or attrs/<pred>) edge left — keyed on graph state,
    #    not on the composites we happened to touch this pass. That makes it both
    #    complete (one DELETE/WHERE per type catches the ones a per-edge drop
    #    misses) and re-runnable (a second apply still sweeps leftover orphans
    #    from a buggy earlier run, even when nothing was rewritten this pass).
    #    The target type comes from the ONTOLOGY (the predicate's rdfs:range), a
    #    cheap single-subject lookup that works on a pure re-run regardless of
    #    whether any edge was rewritten this pass (COG-118).
    target_types = await _composite_target_types(
        neptune, onto_graph, domain_type, pred_leaf, composites_touched
    )
    orphan_uris = await _sweep_orphan_composites(
        neptune, kg_graph, onto_pred, attr_pred_suffix, target_types, delimiters
    )

    summary = {
        "edges_rewritten": len(edges_to_delete),
        "atomic_created": len(atomic_seen),
        "orphans_dropped": len(orphan_uris),
    }
    logger.info("explode_relationship_done", predicate=pred_leaf, **summary)
    return summary, orphan_uris


async def _composite_target_types(
    neptune: NeptuneClient,
    onto_graph: str,
    domain_type: str,
    pred_leaf: str,
    composites: set[str],
) -> set[str]:
    """The relationship's target type(s), for scoping the final orphan sweep.

    PRIMARY path (COG-118): resolve the type from the ONTOLOGY — the predicate's
    declared ``rdfs:range``. The relationship property is
    ``<types/<domain_type>/attrs/<pred_leaf>>`` and its range is the target type's
    ``types/<TargetType>`` URI (the same range the Explorer summary / type-edges
    read). This is a bounded single-subject lookup in the tenant ontology graph —
    cheap, reliable, and INDEPENDENT of whether any edge was rewritten this pass,
    so a pure re-run (``edges_rewritten == 0``) still resolves the type and sweeps
    lingering orphans to zero. It replaces the old unbounded full-graph
    ``SELECT DISTINCT ?t`` scan that timed out on live data and silently skipped
    the sweep (logged ``composite_target_type_query_failed``).

    FALLBACK: if the ontology declares no ``types/`` range for the predicate
    (un-upgraded attribute, or range missing), derive the type(s) from the
    composites we re-pointed this pass — their IRI carries ``…/entities/
    <TargetType>/…``. This keeps the first-pass split path working even before the
    range is upgraded. Scoping to a real target type means the sweep never touches
    unrelated types.
    """
    onto_types = await _range_target_types(neptune, onto_graph, domain_type, pred_leaf)
    if onto_types:
        return onto_types

    # No usable ontology range — derive from this pass's re-pointed composites.
    types: set[str] = set()
    for composite in composites:
        t = _target_type_from_uri(composite)
        if t:
            types.add(t)
    if not types:
        # Nothing rewritten this pass AND no ontology range: we cannot scope a
        # sweep. Surface it (not a silent skip) so a missing range is visible.
        logger.warning(
            "sweep_target_type_unresolved",
            domain_type=domain_type,
            predicate=pred_leaf,
            note="no ontology rdfs:range and no composites re-pointed this pass",
        )
    return types


async def _range_target_types(
    neptune: NeptuneClient, onto_graph: str, domain_type: str, pred_leaf: str
) -> set[str]:
    """Read the predicate's ``rdfs:range`` from the ontology → its target type(s).

    Bounded single-subject query: the relationship property's URI is fully known
    (``attr_uri(domain_type, pred_leaf)``), so this never scans the KG data graph.
    Returns the set of target type NAMES whose range URI is a ``types/`` URI
    (a relationship range); XSD/primitive ranges are ignored (not entity targets).
    A query error is logged and treated as "no range" so the caller falls back
    rather than crashing.
    """
    prop_uri = attr_uri(domain_type, pred_leaf)
    q = (
        f"SELECT ?range FROM <{onto_graph}> WHERE {{\n"
        f"  <{prop_uri}> <{RDFS_RANGE}> ?range .\n"
        f"}}"
    )
    try:
        _, rows = parse_sparql_results(await neptune.query(q))
    except Exception:
        logger.warning(
            "sweep_range_lookup_failed",
            domain_type=domain_type,
            predicate=pred_leaf,
            exc_info=True,
        )
        return set()
    types: set[str] = set()
    for r in rows:
        t = _target_type_from_type_uri(r.get("range", ""))
        if t:
            types.add(t)
    return types


async def _sweep_orphan_composites(
    neptune: NeptuneClient,
    kg_graph: str,
    onto_pred: str,
    attr_pred_suffix: str,
    target_types: set[str],
    delimiters: list[str],
) -> list[str]:
    """Final, graph-state-keyed sweep of orphaned composite nodes.

    For each target type, delete ALL triples of every entity that is (a) of that
    type, (b) composite-named (local-name or rdfs:label contains a rule
    delimiter), and (c) has ZERO inbound ``onto/<pred>`` (or ``…/attrs/<pred>``)
    edges. One SELECT per type resolves the orphan set (complete — catches every
    orphan a per-edge drop would miss; re-runnable — a later apply still sweeps
    leftovers), then the removal routes through the shared ``delete_facts``
    primitive (ADR 0007) so a swept subject is evicted from the derived secondary
    indexes too — no ghost rows keyed to a deleted subject. Atomic nodes (no
    delimiter) and still-referenced composites are left untouched.

    Returns the URIs of the orphan composite subjects removed (the summary count
    is ``len(...)``, and the caller feeds them to ``refresh_after_write`` as
    ``deleted_subjects``).
    """
    if not target_types:
        return []

    delim_filter = " || ".join(
        f'CONTAINS(?cname, "{_sparql_str(d)}")' for d in delimiters
    )
    dropped: list[str] = []
    for target_type in sorted(target_types):
        t_uri = type_uri(target_type)
        # An orphaned composite ?c of this type. SELECT the subjects, then remove
        # them by URI via delete_facts (subject-scoped) — so the set that is
        # evicted from derived indexes is exactly the set removed from Neptune.
        orphan_where = (
            f"  ?c <{RDF_TYPE}> <{t_uri}> .\n"
            f'  FILTER(STRSTARTS(STR(?c), "{ENTITY_URI_PREFIX}"))\n'
            f"  OPTIONAL {{ ?c <{RDFS_LABEL}> ?clabel }}\n"
            f'  BIND(COALESCE(?clabel, REPLACE(STR(?c), "^.*/", "")) AS ?cname)\n'
            f"  FILTER({delim_filter})\n"
            f"  FILTER NOT EXISTS {{ ?s <{onto_pred}> ?c }}\n"
            f"  FILTER NOT EXISTS {{ ?s2 ?p2 ?c . "
            f"FILTER(STRENDS(STR(?p2), \"{_sparql_str(attr_pred_suffix)}\")) }}\n"
        )
        select_q = (
            f"SELECT DISTINCT ?c FROM <{kg_graph}> WHERE {{\n"
            f"{orphan_where}"
            f"}}"
        )
        try:
            _, rows = parse_sparql_results(await neptune.query(select_q))
            orphan_uris = [r["c"] for r in rows if r.get("c")]
        except Exception:
            logger.warning(
                "orphan_select_failed", target_type=target_type, exc_info=True
            )
            continue
        if not orphan_uris:
            continue
        try:
            await delete_facts(
                neptune,
                kg_graph,
                subjects=orphan_uris,
                touched_types=[target_type],
                reason="normalization:list_explode orphan-composite sweep",
            )
        except Exception:
            logger.warning(
                "orphan_sweep_failed", target_type=target_type, exc_info=True
            )
            continue
        dropped.extend(orphan_uris)
    return dropped


def _target_type_from_type_uri(t_uri: str) -> str | None:
    """``https://cograph.tech/types/<TargetType>`` → ``<TargetType>``."""
    prefix = type_uri("")
    if not t_uri.startswith(prefix):
        return None
    tail = t_uri[len(prefix):].strip("/")
    return tail or None


async def _explode_literal(
    neptune: NeptuneClient, kg_graph: str, pred_leaf: str, delimiters: list[str]
) -> tuple[dict, list[str]]:
    """Split packed attribute literals into N atomic literals.

    Returns ``(summary, [])`` — literal splits replace an attribute value on a
    surviving subject, so nothing here is a deleted subject.
    """
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
        await insert_facts(neptune, kg_graph, to_add)
    if to_delete:
        await delete_facts(
            neptune,
            kg_graph,
            triples=to_delete,
            reason="normalization:list_explode packed-literal replace",
        )

    summary = {
        "edges_rewritten": rewritten,
        "atomic_created": atomic_count,
        "orphans_dropped": 0,
    }
    logger.info("explode_literal_done", predicate=pred_leaf, **summary)
    return summary, []


def _strip_emoji_value(value: str) -> str:
    """Remove emoji / pictographic junk from one text value, collapse whitespace.

    Pure + deterministic. ``"🎨 design"`` → ``"design"``; ``"design 🚀"`` →
    ``"design"``; ``"ai 🚀 growth"`` → ``"ai growth"``; a pure-emoji value → ``""``
    (the caller drops empties). A value with no emoji is returned UNCHANGED after
    a no-op whitespace collapse, so re-running is idempotent and ordinary names
    (``"c++"``, ``"café"``, ``"R&D"``) are never touched.
    """
    stripped = _EMOJI_PATTERN.sub(" ", value)
    return _WS_PATTERN.sub(" ", stripped).strip()


async def _strip_emoji(neptune: NeptuneClient, kg_graph: str, rule) -> tuple[dict, list[str]]:
    """Strip emoji/junk from this predicate's literals; rewrite only what changed.

    Selects every ``attrs/<leaf>`` (or ``onto/<leaf>``) literal for the
    predicate, cleans each value, and — for the literals that actually changed —
    deletes the old triple and (unless the cleaned value is empty) inserts the
    cleaned one. Unchanged literals (already emoji-free) and non-literal objects
    are left alone, so the pass is idempotent. ``targets`` in params is reserved
    for future relationship-label cleaning; v1 cleans attribute literals.
    """
    pred_leaf = rule.predicate
    onto_pred = ONTO_PRED_PREFIX + pred_leaf
    attr_pred_suffix = ATTRS_INFIX + pred_leaf

    # Pull every literal for the predicate (both predicate forms). No CONTAINS
    # pre-filter — emoji are spread across many codepoints, so we clean in Python
    # and only rewrite the rows that change (the SELECT is bounded by predicate).
    q = (
        f"SELECT ?s ?p ?o FROM <{kg_graph}> WHERE {{\n"
        f"  ?s ?p ?o .\n"
        f"  FILTER(?p = <{onto_pred}> || STRENDS(STR(?p), \"{_sparql_str(attr_pred_suffix)}\"))\n"
        f"  FILTER(isLiteral(?o))\n"
        f"}}"
    )
    _, rows = parse_sparql_results(await neptune.query(q))

    to_delete: list[tuple[str, str, str]] = []
    to_add: list[tuple[str, str, str]] = []
    literals_cleaned = 0
    for r in rows:
        s = r.get("s", "")
        p = r.get("p", "")
        o = r.get("o", "")
        if not s or not p:
            continue
        cleaned = _strip_emoji_value(o)
        if cleaned == o:
            continue  # no emoji / already clean — idempotent no-op
        literals_cleaned += 1
        to_delete.append((s, p, o))
        if cleaned:
            to_add.append((s, p, cleaned))
        # else: cleaned is empty (pure-emoji value) — drop the triple entirely.

    if to_add:
        await insert_facts(neptune, kg_graph, to_add)
    if to_delete:
        await delete_facts(
            neptune,
            kg_graph,
            triples=to_delete,
            reason="normalization:strip_emoji literal cleanup",
        )

    summary = {
        "literals_cleaned": literals_cleaned,
        "triples_rewritten": len(to_delete),
    }
    logger.info("strip_emoji_done", predicate=pred_leaf, **summary)
    return summary, []


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
