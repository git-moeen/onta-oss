"""Async executor for enrichment jobs.

Reads entities from Neptune, runs them through the source funnel
(lite tier = wikidata, with cache), and either stages results for
review or applies them directly based on conflict_policy.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime, timezone
from typing import Optional

import structlog

from cograph_client.enrichment.cache import EnrichmentCache
from cograph_client.enrichment.canonicalize import apply_canonicalizer
from cograph_client.enrichment.job_store import JobStore
from cograph_client.enrichment.models import (
    ConflictPolicy,
    ConflictReview,
    EnrichJob,
    EnrichScope,
    JobStatus,
    RowResult,
    Verdict,
)
from cograph_client.enrichment.sources.base import (
    SourceAdapter,
    get_adapter,
    register_adapter,
)
from cograph_client.enrichment.strategy import (
    AttributeStrategy,
    TypeStrategy,
    load_strategy,
)
from cograph_client.enrichment.tiers import get_chain
from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.ontology_queries import upsert_attribute
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import (
    insert_triples,
    kg_graph_uri,
    tenant_graph_uri,
)

logger = structlog.stdlib.get_logger("cograph.enrichment")


TYPE_URI_PREFIX = "https://cograph.tech/types/"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDF_PROPERTY = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
RDFS_DOMAIN = "http://www.w3.org/2000/01/rdf-schema#domain"
# Relationship instance triples use the `…/onto/<predName>` namespace (minted in
# nlp/pipeline.py + resolver/schema_resolver.py); literal-attribute instance
# triples use the `…/types/<Type>/attrs/<name>` (attr_uri) namespace. A scope
# predicate's ontology declaration doesn't tell us which the data uses, so a
# resolved local-name maps to BOTH candidate instance IRIs. NOTE: the name the
# Explorer DISPLAYS comes from the entity's `rdfs:label` (set at ingest), which
# may differ from — or exist WITHOUT — an `…/attrs/name` literal, so a scope on a
# name/title VALUE must also match `rdfs:label` (see `_scope_block`).
ONTO_PRED_PREFIX = "https://cograph.tech/onto/"
NAME_FALLBACK_ATTRS = ["name", "title", "headline"]
WORKER_POOL_SIZE = 8
PROGRESS_FLUSH_EVERY = 10

# rdfs:comment stamped on an enrichment-declared attribute so the ontology /schema
# view + Explorer can distinguish a schema slot that arrived via enrichment from
# one declared by ingest or the ontology endpoint. Enriched values are strings
# (the wikidata/adapter funnel yields literal text), so the declared range is
# always xsd:string — matching what the instance triples actually carry.
ENRICH_ATTR_DESCRIPTION = "Added by enrichment job"
ENRICH_ATTR_DATATYPE = "string"

# Hard ceiling on a single adapter lookup (COG-112). A misbehaving adapter — a
# stalled TCP/TLS connect, a server that dribbles keepalive bytes (which resets
# httpx's per-byte read timeout forever), or any adapter whose own client lacks a
# timeout — must NEVER hang the whole job. Without this wrapper a single
# `await adapter.lookup(...)` that never returns AND never raises strands the job
# in `running` with zero logs and no failure (the exact production symptom: logs
# stop right after the scoped SELECT, no outbound adapter HTTP, no
# enrichment_job_failed). This bound makes such a stall surface as a visible,
# retryable `enrichment_adapter_timeout` log instead (verdicts=[] → the chain
# moves on and the job completes). Generous enough to cover a slow
# multi-step adapter (e.g. wikidata's search→claims→label round-trips) while
# still failing fast relative to "forever". Overridable via the
# COGRAPH_ADAPTER_LOOKUP_TIMEOUT_S env var.
ADAPTER_LOOKUP_TIMEOUT_S = float(os.environ.get("COGRAPH_ADAPTER_LOOKUP_TIMEOUT_S", "30"))


def _type_uri(type_name: str) -> str:
    return f"{TYPE_URI_PREFIX}{type_name}"


def _attr_uri(type_name: str, attr: str) -> str:
    return f"{TYPE_URI_PREFIX}{type_name}/attrs/{attr}"


def _esc_lit(value: str) -> str:
    """Escape a string for use inside a SPARQL double-quoted literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# A well-formed http(s) IRI with none of the characters that could break out of
# a SPARQL ``<…>`` term (``<``, ``>``, ``"``, ``{``, ``}``, whitespace). The
# Pydantic validators on the request models reject bad input at the API
# boundary; this is the executor-level backstop so a malformed URI can never be
# spliced into a VALUES block (defense in depth — SPARQL injection fix #1).
_IRI_RE = re.compile(r'^https?://[^\s<>"{}]+$')


def _validate_entity_uris(entity_uris: list[str]) -> list[str]:
    """Return ``entity_uris`` unchanged, or raise ``ValueError`` if any entry is
    not a safe http(s) IRI (no ``<>"{}`` or whitespace)."""
    for u in entity_uris:
        if not isinstance(u, str) or not _IRI_RE.match(u):
            raise ValueError(f"invalid entity URI for scoped enrichment: {u!r}")
    return entity_uris


def _local_name(uri_or_value: str) -> str:
    """Last path / fragment segment of a URI; the value itself if not a URI."""
    s = uri_or_value.rstrip("/")
    if "#" in s:
        s = s.split("#")[-1]
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    return s


def _safe_iri(uri: str) -> bool:
    """A concrete predicate/label IRI is safe to interpolate into ``<…>`` only if
    it carries none of the chars that could break out of the term. The resolved
    IRIs are built from ``attr_uri``/``onto/`` + an ontology-known leaf so they
    are well-formed, but this is the executor-level backstop (defense in depth)."""
    return isinstance(uri, str) and bool(_IRI_RE.match(uri))


def _pred_path(iris: list[str]) -> str:
    """A SPARQL property-path term that matches any of ``iris`` with the
    predicate BOUND (so Neptune uses the POS index).

    A single IRI → ``<iri>``; multiple IRIs → an alternation ``(<a>|<b>|…)``.
    Bound-predicate paths are predicate-indexed, unlike a variable predicate +
    ``FILTER(?p IN (…))`` which on Neptune does NOT use the predicate index and
    degrades to a scan (the second COG-112 perf bug)."""
    if len(iris) == 1:
        return f"<{iris[0]}>"
    return "(" + "|".join(f"<{u}>" for u in iris) + ")"


def _scope_block(type_name: str, scope: EnrichScope, pred_iris: list[str]) -> str:
    """INLINE SPARQL graph patterns restricting ``?e`` (already typed) to a
    value-scope — emitted directly into the WHERE next to ``?e a <Type>`` so the
    planner can drive from the selective bound-predicate triple, NOT wrapped in a
    ``FILTER EXISTS``. The patterns are a top-level UNION of (a) the predicate-bound
    arms (attribute literal / relationship target) and (b) a direct ``rdfs:label``
    match, so a name/title scope matches the value the user SEES even when it is
    carried only as ``rdfs:label`` and not as an ``…/attrs/<attr>`` literal.

    ``scope.predicate`` is an attribute OR relationship **local-name** (e.g.
    ``hasLevel``, ``property_type``). It has already been resolved (case-
    insensitively) against the type's ontology-declared predicates to a list of
    **concrete instance predicate IRIs** (``pred_iris``) by
    :func:`_resolve_scope_predicate_iris`. Those IRIs are matched as a
    **bound-predicate property-path alternation** — ``?e (<a>|<b>) ?sv`` — so
    Neptune uses its POS (predicate-object) index instead of scanning.

    COG-112 three-layer perf history:
      1. The original form ``?e ?p ?sv . FILTER(LCASE(REPLACE(STR(?p), …)))``
         was an O(entities × predicates) local-name scan that timed out.
      2. The first fix resolved the predicate to concrete IRIs but still matched
         with a VARIABLE predicate + ``FILTER(?p IN (<a>,<b>))``. On Neptune a
         variable predicate + FILTER does NOT use the predicate index, so it
         still scanned. The second fix bound the predicate via a property path.
      3. The bound property path lived inside ``FILTER EXISTS { … }`` wrapped
         around ``?e a <Type>``, so Neptune evaluated the EXISTS **once per type
         instance** (O(all instances), 13.5k Mentors) — still a timeout. This
         fix inlines the scope as join patterns so the optimizer starts from the
         selective bound-predicate triple (~9k haslevel edges via the POS index),
         joins/filters down to the <10 matches, and only then (in the caller)
         hydrates attributes. The caller wraps this in a ``SELECT DISTINCT ?e``
         sub-select so the multi-arm UNION can never multiply ``?e`` rows.

    Injection safety is preserved: ``scope.predicate`` was validated as a safe
    local-name AND resolved to an ontology-known IRI, so each interpolated
    ``pred_iri`` is a concrete, well-formed IRI (re-checked by :func:`_safe_iri`),
    never raw user text. ``scope.value`` still appears only as a lower-cased,
    ``_esc_lit``-escaped string *literal*.

    Case-insensitive value matching (#2) is kept on both sides via ``LCASE``.
    Attribute vs relationship is discriminated by the OBJECT (no extra round-trip):

      - **literal attribute** — the object is a literal; match its string value
        case-insensitively: ``FILTER(isLiteral(?sv) && LCASE(STR(?sv)) = "<v>")``.
      - **relationship to a node** — the object is an IRI; ``?sv`` is already
        bounded by the predicate triple to the handful of related TARGET nodes,
        so match ANY literal property on it case-insensitively
        (``?sv ?slp ?stl . FILTER(isLiteral(?stl) && LCASE(STR(?stl)) = "<v>")``),
        OR the target IRI's local-name as a fallback. We deliberately do NOT pin
        the label predicate to ``…/types/<sourceType>/attrs/*`` here: the target
        node's display name lives under the TARGET type's namespace (e.g. a
        ``Level`` node's "Manager" is ``…/types/Level/attrs/name``), so binding
        the source type's attr predicate would match ZERO targets (COG-112 bug).
        ``?slp`` is a variable (not interpolated) and the value stays
        ``_esc_lit``-escaped, so this is injection-safe; ``?sv`` is bounded so the
        open ``?slp`` carries no perf cost.
      - **literal attribute carried only as rdfs:label** — the entity's displayed
        name comes from ``rdfs:label`` (set at ingest), while attribute values
        live under ``…/attrs/<attr>``. A name/title scope on the value the user
        SEES (the label) must therefore also match the entity's ``rdfs:label``,
        because the entity may carry its name ONLY as ``rdfs:label`` and NOT as an
        ``attrs/name`` literal — in which case the predicate triple above binds
        nothing and the literal arm matches ZERO (COG-112 literal-attribute bug).
        This is an INDEPENDENT branch (it must not require ``?e {pred_path} ?sv``
        to bind first), so the whole block is a top-level UNION: the
        predicate-bound arms on one side, the direct ``rdfs:label`` match on the
        other. ``rdfs:label`` is a single bound predicate (POS-indexed), so the
        extra branch stays cheap; the caller's ``SELECT DISTINCT ?e`` collapses an
        entity matched by more than one branch to a single row.

    If ``pred_iris`` is empty (predicate not declared on the type) the block
    emits ``FILTER(false)`` so the scope matches NOTHING fast — never the old
    unbounded scan (COG-112 fix #3).
    """
    safe_iris = [u for u in pred_iris if _safe_iri(u)]
    if not safe_iris:
        # Predicate not resolvable on this type → match nothing, fast & honest.
        return "  FILTER(false)"

    v_lower = _esc_lit(scope.value.lower())
    pred_path = _pred_path(safe_iris)

    # COG-112 literal-attribute fix. The REAL data (resolver/schema_resolver.py)
    # stores a literal attribute's value under the `…/types/<Type>/attrs/<attr>`
    # (attr_uri) namespace, while `rdfs:label` carries the opaque ENTITY-ID slug
    # (set at ingest) — NOT the human name. So PR #37's rdfs:label arm could never
    # match a literal scope like name="Jane Doe" (label = the slug), and the
    # value lives only on the attrs/<attr> literal.
    #
    # The pre-existing predicate-bound arm DOES list the attr IRI inside the
    # property-path ALTERNATION `(<…/attrs/name>|<…/onto/name>)`. In a standards
    # engine that matches; on Neptune, mixing the literal-attribute namespace
    # with the (zero-triple) relationship `…/onto/<attr>` alternative inside a
    # path that then feeds `FILTER(isLiteral(?sv))` is exactly the shape that
    # silently bound nothing in production. So we add a DEDICATED, single-bound-
    # predicate literal arm per attr_uri IRI (POS-indexed, no alternation, no
    # downstream isLiteral on a path-bound object) so the literal-attribute value
    # match is unambiguous and reliable. The original alternation arms and PR
    # #37's rdfs:label arm are kept (belt-and-suspenders); relationship behavior
    # is byte-for-byte unchanged. The caller's SELECT DISTINCT ?e collapses an
    # entity matched by more than one arm to a single row.
    attr_value_arms = "".join(
        # Dedicated literal arm: ?e <…/attrs/<attr>> ?av (single bound predicate,
        # POS-indexed) → match its literal value case-insensitively. The IRI is a
        # concrete, _safe_iri-checked, ontology-derived term; the value stays
        # _esc_lit-escaped + lower-cased → injection-safe.
        f" UNION {{\n"
        f"    ?e <{iri}> ?av .\n"
        f"    FILTER(isLiteral(?av) && LCASE(STR(?av)) = \"{v_lower}\")\n"
        f"  }}"
        for iri in safe_iris
        if "/attrs/" in iri  # only the literal-attribute namespace, not …/onto/
    )

    return (
        # Top-level UNION: predicate-bound arms (attribute literal / relationship)
        # on one side, a direct rdfs:label match + dedicated attrs/<attr> literal
        # arm(s) on the other. The latter branches are INDEPENDENT of the
        # property-path triple so they match even when the alternation arm doesn't.
        f"  {{\n"
        # Match the predicate by a BOUND property path (POS-indexed, no scan).
        # Inlined directly into the WHERE — the planner drives from this selective
        # triple instead of evaluating an EXISTS once per type instance.
        f"    ?e {pred_path} ?sv .\n"
        f"    {{\n"
        # Literal-attribute arm.
        f"      FILTER(isLiteral(?sv) && LCASE(STR(?sv)) = \"{v_lower}\")\n"
        f"    }} UNION {{\n"
        # Relationship arm: ?sv is the target node, already bounded to the handful
        # of related nodes by the predicate triple above. Match ANY literal value
        # on it case-insensitively (its display name lives on the TARGET type's
        # namespace, e.g. …/types/Level/attrs/name — NOT the scope source type's,
        # so we must not pin the predicate to <sourceType>/attrs/* here). ?slp is a
        # variable (no interpolation); the value stays escaped → injection-safe.
        f"      ?sv ?slp ?stl .\n"
        f"      FILTER(isLiteral(?stl) && LCASE(STR(?stl)) = \"{v_lower}\")\n"
        f"    }} UNION {{\n"
        # … or the target IRI's local-name as a fallback.
        f"      FILTER(isIRI(?sv) && LCASE(REPLACE(STR(?sv), \"^.*[/#]\", \"\")) = \"{v_lower}\")\n"
        f"    }}\n"
        f"  }} UNION {{\n"
        # Displayed-name arm: PR #37's rdfs:label match. Kept as belt-and-
        # suspenders — for the real ADP-style data rdfs:label is the entity-id
        # slug, so this rarely matches a name VALUE, but it is harmless and covers
        # any KG that did set rdfs:label to the human name. ?lbl is a fresh var;
        # the value stays _esc_lit-escaped → injection-safe.
        f"    ?e <{RDFS_LABEL}> ?lbl .\n"
        f"    FILTER(isLiteral(?lbl) && LCASE(STR(?lbl)) = \"{v_lower}\")\n"
        f"  }}"
        # Dedicated attrs/<attr> literal arm(s) — the actual COG-112 fix.
        f"{attr_value_arms}"
    )


def _scope_subselect(
    type_name: str,
    scope: EnrichScope,
    pred_iris: list[str],
    limit: Optional[int] = None,
) -> str:
    """A bounded ``SELECT DISTINCT ?e`` sub-select that reduces ``?e`` to the
    scoped (and, if ``limit`` given, capped) subset of ``type_name`` instances
    BEFORE any attribute hydration.

    The inner WHERE inlines :func:`_scope_block`'s join patterns next to
    ``?e a <Type>`` so the planner drives from the selective bound-predicate
    triple (POS-indexed) → joins/filters down to the matches, never evaluating
    an EXISTS once per type instance (the third COG-112 perf layer).

    ``DISTINCT`` collapses the multi-arm scope UNION so an entity matching more
    than one arm yields a single ``?e`` row — the caller can then attach the
    GROUP_CONCAT attribute OPTIONALs without row multiplication. The ``LIMIT``
    is applied INSIDE the sub-select so the cap bounds the selected entities (and
    the subsequent attribute hydration), matching the no-scope ``LIMIT`` semantics.
    """
    type_uri = _type_uri(type_name)
    limit_clause = f"\n    LIMIT {int(limit)}" if limit else ""
    scope_patterns = _scope_block(type_name, scope, pred_iris)
    return (
        f"  {{ SELECT DISTINCT ?e WHERE {{\n"
        f"    ?e a <{type_uri}> .\n"
        f"  {scope_patterns}\n"
        f"  }}{limit_clause} }}\n"
    )


def _resolve_scope_predicate_query(graph_uri: str, type_name: str) -> str:
    """SELECT every predicate the ontology declares on ``type_name`` (leaf + URI).

    Same shape the Explorer summary / records paths use (``?attr a rdf:Property ;
    rdfs:domain <type> ; rdfs:label ?label``). Tiny — bounded by the type's
    attribute count, not its instance count — so it is cheap to run at create
    time. Returns the ontology attribute URI and its label so the caller can
    derive both candidate instance predicate IRIs (attr_uri vs onto/<leaf>)."""
    t_uri = _type_uri(type_name)
    return (
        f"SELECT ?attr ?label FROM <{graph_uri}> WHERE {{\n"
        f"  ?attr <{RDF_TYPE}> <{RDF_PROPERTY}> .\n"
        f"  ?attr <{RDFS_DOMAIN}> <{t_uri}> .\n"
        f"  ?attr <{RDFS_LABEL}> ?label .\n"
        f"}}"
    )


def _instance_pred_iris_for_leaf(type_name: str, leaf: str) -> list[str]:
    """The concrete instance predicate IRIs a declared predicate ``leaf`` can use.

    A literal attribute is stored under ``…/types/<Type>/attrs/<leaf>``
    (``attr_uri``); a relationship is stored under ``…/onto/<leaf>``
    (``ONTO_PRED_PREFIX``). The ontology declaration alone doesn't pin which, so
    we match BOTH — both are concrete, ontology-derived IRIs, so this stays a
    bounded, predicate-indexed match (no scan)."""
    return [_attr_uri(type_name, leaf), f"{ONTO_PRED_PREFIX}{leaf}"]


def _resolve_pred_iris_from_bindings(
    type_name: str, predicate: str, bindings: list[dict]
) -> list[str]:
    """Case-insensitively resolve ``predicate`` against the type's declared
    predicates (from :func:`_resolve_scope_predicate_query` bindings) to the
    concrete instance predicate IRIs to match. Returns ``[]`` when the predicate
    is not declared as an ATTRIBUTE on the type.

    This is the ONTOLOGY arm of resolution: it gives case-normalisation (a
    request ``hasLevel`` → the stored ``haslevel`` leaf) and covers literal
    attributes. RELATIONSHIPS are NOT declared this way (they live under
    ``…/onto/<pred>``), so they return ``[]`` HERE — the caller
    (:meth:`EnrichmentExecutor._resolve_scope_predicate_iris`) UNIONS this with a
    direct build from the input predicate so relationships still resolve."""
    want = predicate.strip().lower()
    iris: list[str] = []
    seen: set[str] = set()
    for row in bindings:
        attr_uri_val = row.get("attr", "")
        label = row.get("label", "")
        # The leaf is the ontology attr-URI's last segment; the label is the
        # human name. Match against both (case-insensitive) so a request can use
        # either the stored leaf or the declared label.
        leaf = _local_name(attr_uri_val) if attr_uri_val else ""
        candidates_lower = {c.lower() for c in (leaf, label) if c}
        if want in candidates_lower and leaf:
            for iri in _instance_pred_iris_for_leaf(type_name, leaf):
                if iri not in seen:
                    seen.add(iri)
                    iris.append(iri)
    return iris


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _build_select_query(
    graph_uri: str,
    type_name: str,
    attributes: list[str],
    limit: Optional[int],
    scope: Optional[EnrichScope] = None,
    entity_uris: Optional[list[str]] = None,
    scope_pred_iris: Optional[list[str]] = None,
) -> str:
    """Entity-selection SELECT for an enrich job.

    Whole-type by default. Scoped subsets (COG-112):

      - ``entity_uris`` (lower-level primitive, wins over ``scope``): restricts
        to exactly those URIs via a ``VALUES ?e {…}`` block (still constrained
        to ``?e a <Type>`` so a stray URI of another type is ignored).
      - ``scope``: restricts to entities of the type whose attribute/relationship
        matches, via a bounded ``SELECT DISTINCT ?e`` sub-select (see
        :func:`_scope_subselect`) that inlines the scope join patterns (see
        :func:`_scope_block`) next to ``?e a <Type>`` — so the planner drives
        from the selective bound-predicate triple instead of evaluating an
        ``EXISTS`` once per type instance (COG-112 perf fix #4). The LIMIT is
        applied INSIDE the sub-select so it caps the SELECTED entities before the
        attribute OPTIONALs hydrate them. ``scope_pred_iris`` are the concrete
        instance predicate IRI(s) the scope predicate resolved to (see
        :func:`_resolve_scope_predicate_iris`, which UNIONS the ontology-declared
        resolution with a direct build so relationships under ``…/onto/<pred>``
        resolve too). A valid predicate therefore always yields candidates; an
        empty list only arises for an empty/invalid predicate, in which case the
        scope matches nothing (fast, ``FILTER(false)``, no per-entity scan).
    """
    type_uri = _type_uri(type_name)
    attr_uris = [_attr_uri(type_name, a) for a in attributes]
    fallback_uris = [_attr_uri(type_name, a) for a in NAME_FALLBACK_ATTRS]

    in_list = ", ".join(f"<{u}>" for u in attr_uris) if attr_uris else "<urn:none>"
    fallback_in = ", ".join(f"<{u}>" for u in fallback_uris)

    # Subset constraint. entity_uris (explicit primitive) wins over scope.
    #
    # For a `scope`, the typed+scoped+capped `?e` set is produced by a bounded
    # DISTINCT sub-select (the LIMIT lives INSIDE it so it caps the SELECTED
    # entities, then attribute OPTIONALs hydrate that bounded set — COG-112 fix
    # #4). For the no-scope / entity_uris paths the outer query keeps the bare
    # `?e a <Type>` + a trailing top-level LIMIT exactly as before.
    if entity_uris:
        values = " ".join(f"<{u}>" for u in _validate_entity_uris(entity_uris))
        type_clause = f"  ?e a <{type_uri}> .\n"
        subset_clause = f"  VALUES ?e {{ {values} }}\n"
        limit_clause = f"\nLIMIT {int(limit)}" if limit else ""
    elif scope is not None:
        # The sub-select emits `?e a <Type>` + the inline scope patterns and caps
        # to LIMIT internally; the outer query must not re-type or re-LIMIT.
        type_clause = ""
        subset_clause = _scope_subselect(
            type_name, scope, scope_pred_iris or [], limit
        )
        limit_clause = ""
    else:
        type_clause = f"  ?e a <{type_uri}> .\n"
        subset_clause = ""
        limit_clause = f"\nLIMIT {int(limit)}" if limit else ""

    # GROUP_CONCAT predicate::value for all matching attribute triples.
    # Also pull a label / name fallback for entity_label.
    return (
        f"SELECT ?e ?label ?nameAttr\n"
        f'  (GROUP_CONCAT(DISTINCT CONCAT(STR(?p), "::", STR(?o)); separator="||") AS ?vals)\n'
        f"FROM <{graph_uri}> WHERE {{\n"
        f"{type_clause}"
        f"{subset_clause}"
        f"  OPTIONAL {{ ?e <{RDFS_LABEL}> ?label }}\n"
        f"  OPTIONAL {{ ?e ?fp ?nameAttr . FILTER(?fp IN ({fallback_in})) }}\n"
        f"  OPTIONAL {{ ?e ?p ?o . FILTER(?p IN ({in_list})) }}\n"
        f"}} GROUP BY ?e ?label ?nameAttr"
        f"{limit_clause}"
    )


def _parse_vals(vals_field: str) -> dict[str, str]:
    """Parse ?vals (predicate::value pairs joined by '||') into a dict.

    If the same predicate appears multiple times, the first one wins.
    """
    out: dict[str, str] = {}
    if not vals_field:
        return out
    for chunk in vals_field.split("||"):
        if "::" not in chunk:
            continue
        p, _, v = chunk.partition("::")
        if p and p not in out:
            out[p] = v
    return out


def _values_match(existing: str, candidate: str) -> bool:
    """Loose match: case-insensitive substring or exact equality."""
    if not existing or not candidate:
        return False
    a = existing.strip().lower()
    b = candidate.strip().lower()
    if a == b:
        return True
    return a in b or b in a


def _values_match_with_strategy(
    existing: str, candidate: str, attr_strategy: AttributeStrategy | None
) -> bool:
    """Apply canonicalizer + aliases to the existing value before matching."""
    if attr_strategy is None:
        return _values_match(existing, candidate)
    transformed = existing
    if attr_strategy.canonicalizer:
        transformed = apply_canonicalizer(attr_strategy.canonicalizer, transformed)
    # Alias dictionary: literal lookup AND match against the transformed form.
    if attr_strategy.aliases:
        if existing in attr_strategy.aliases:
            transformed = attr_strategy.aliases[existing]
        elif transformed in attr_strategy.aliases:
            transformed = attr_strategy.aliases[transformed]
    return _values_match(transformed, candidate)


class EnrichmentExecutor:
    def __init__(
        self,
        neptune_client: NeptuneClient,
        job_store: JobStore,
        cache: EnrichmentCache,
        wikidata_adapter: SourceAdapter,
    ) -> None:
        self._neptune = neptune_client
        self._jobs = job_store
        self._cache = cache
        self._wikidata = wikidata_adapter
        # Register the wikidata adapter into the global adapter registry so
        # chain-based lookups can resolve it by name. Idempotent.
        try:
            register_adapter(wikidata_adapter)
        except Exception:  # noqa: BLE001
            pass

    async def _resolve_scope_predicate_iris(
        self, tenant_id: str, type_name: str, scope: EnrichScope
    ) -> list[str]:
        """Resolve ``scope.predicate`` (a local-name) to the concrete instance
        predicate IRI(s) to match.

        The candidate IRIs are the **union** of two sources:

          1. **Ontology-declared resolution** — match ``scope.predicate``
             case-insensitively against the type's ``rdf:Property``/
             ``rdfs:domain``/``rdfs:label`` declarations (see
             :func:`_resolve_pred_iris_from_bindings`). This gives
             case-normalisation: a request ``hasLevel`` resolves to the stored
             ``haslevel`` leaf. Attributes are always declared this way.
          2. **Direct build from the (validated) input predicate** —
             :func:`_instance_pred_iris_for_leaf` → ``…/types/<Type>/attrs/<pred>``
             and ``…/onto/<pred>``. **Relationships (object properties) like
             ``haslevel`` are stored under ``…/onto/<pred>`` and are NOT declared
             as an attribute** (``rdf:Property ; rdfs:domain <Type> ;
             rdfs:label``), so the ontology arm alone returns ``[]`` for them
             (COG-112 root cause). The direct build always yields
             ``…/onto/<pred>`` (and the attr IRI), so a relationship scope
             matches the instance edges. The UI dropdown sends the exact stored
             local-name, so casing is correct for the direct build.

        The predicate has already been validated as a safe local-name by the
        Pydantic model validator, so building IRIs from it is injection-safe;
        each candidate is still re-checked with :func:`_safe_iri`. The direct
        build LOWER-CASES the predicate so it agrees with the ontology arm's
        normalised leaf and the live instance data (predicates are minted
        lower-cased — e.g. ``…/onto/haslevel``, ``…/attrs/name``): a mixed-case
        request like ``hasLevel`` never leaks verbatim into the candidate IRIs.
        A syntactically valid predicate therefore ALWAYS yields candidates —
        only an empty/invalid predicate (or one whose direct-build IRIs all fail
        ``_safe_iri``) yields ``[]``.

        On a Neptune error during the ontology query we skip the ontology arm
        but still return the direct build (so create stays fast, never 500s, and
        a relationship scope still resolves even if the ontology read fails).
        """
        onto_graph = tenant_graph_uri(tenant_id)
        query = _resolve_scope_predicate_query(onto_graph, type_name)
        ontology_iris: list[str] = []
        try:
            raw = await self._neptune.query(query)
            _, bindings = parse_sparql_results(raw)
            ontology_iris = _resolve_pred_iris_from_bindings(
                type_name, scope.predicate, bindings
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "scope_predicate_resolve_failed",
                tenant_id=tenant_id,
                type_name=type_name,
                predicate=scope.predicate,
                error=str(exc),
            )
        # Union the ontology-declared resolution (case-normalised attributes)
        # with the direct build (covers relationships under …/onto/<pred>).
        # Lower-case the predicate for the direct build so it agrees with the
        # ontology leaf and the lower-cased instance predicates (no mixed-case
        # leak). Dedup, preserve order, keep each IRI passing _safe_iri.
        direct_iris = _instance_pred_iris_for_leaf(
            type_name, scope.predicate.strip().lower()
        )
        iris: list[str] = []
        seen: set[str] = set()
        for iri in [*ontology_iris, *direct_iris]:
            if iri not in seen and _safe_iri(iri):
                seen.add(iri)
                iris.append(iri)
        return iris

    async def count_entities(
        self,
        tenant_id: str,
        kg_name: str,
        type_name: str,
        scope: Optional[EnrichScope] = None,
        entity_uris: Optional[list[str]] = None,
    ) -> int:
        """Count entities the job will enrich.

        Whole-type by default; honors the same subset semantics as the
        entity-selection query (COG-112). ``entity_uris`` wins over ``scope``
        (matching :func:`_build_select_query`).

        NOTE (COG-112 non-blocking): create-job no longer calls this in the
        request path — the executor's background SELECT resolves the matched
        subset and sets ``progress.total``. This method remains as a standalone
        utility (e.g. a future "preview count" endpoint) and is exercised by the
        tests; it shares the same index-efficient SPARQL as the SELECT.

        For a ``scope`` the predicate is resolved to a concrete instance IRI
        first (cheap, ontology-bounded), and the COUNT runs over the SAME bounded
        ``SELECT DISTINCT ?e`` sub-select the SELECT uses (see
        :func:`_scope_subselect`): the inline scope patterns let the planner drive
        from the selective BOUND-predicate property path (POS-indexed) instead of
        a variable predicate + ``FILTER`` (which Neptune does NOT predicate-index),
        a per-triple scan, OR an ``EXISTS`` evaluated once per type instance — the
        three-layer COG-112 perf fix. (No LIMIT: the COUNT reflects the full
        scoped subset, not the capped SELECT.)
        """
        graph_uri = kg_graph_uri(tenant_id, kg_name)
        if entity_uris:
            values = " ".join(f"<{u}>" for u in _validate_entity_uris(entity_uris))
            type_clause = f"  ?e a <{_type_uri(type_name)}> .\n"
            subset_clause = f"  VALUES ?e {{ {values} }}\n"
        elif scope is not None:
            pred_iris = await self._resolve_scope_predicate_iris(
                tenant_id, type_name, scope
            )
            # A valid predicate always resolves to candidate IRIs now (the direct
            # build covers relationships under …/onto/<pred>); only an
            # empty/invalid predicate yields []. In that degenerate case the
            # scope matches nothing → short-circuit to 0 without issuing the
            # COUNT (honest matched-0, fast).
            if not pred_iris:
                return 0
            # Count over the SAME bounded DISTINCT sub-select the SELECT uses: the
            # inline scope patterns let the planner drive from the selective
            # bound-predicate triple instead of evaluating an EXISTS once per
            # type instance (COG-112 fix #4). No LIMIT — the COUNT reflects the
            # full scoped subset, not the capped SELECT.
            type_clause = ""
            subset_clause = _scope_subselect(type_name, scope, pred_iris)
        else:
            type_clause = f"  ?e a <{_type_uri(type_name)}> .\n"
            subset_clause = ""
        query = (
            f"SELECT (COUNT(DISTINCT ?e) AS ?n) FROM <{graph_uri}> WHERE {{\n"
            f"{type_clause}"
            f"{subset_clause}"
            f"}}"
        )
        raw = await self._neptune.query(query)
        _, bindings = parse_sparql_results(raw)
        if not bindings:
            return 0
        try:
            n = int(bindings[0].get("n", "0"))
        except (TypeError, ValueError):
            return 0
        return n

    async def run(self, job: EnrichJob, tenant_id: str) -> None:
        try:
            job.status = JobStatus.running
            job.started_at = _now()
            await self._jobs.update(job)

            # Load ontology-driven strategy. Always returns a TypeStrategy.
            strategy = await load_strategy(self._neptune, tenant_id, job.type_name)
            # Cache-key version for this strategy. A change here auto-invalidates
            # the cache (different key -> clean miss). TODO(ADR-0005 §2): the ADR
            # wants a real strategy_version field on TypeStrategy/AttributeStrategy;
            # derive a stable string until that lands.
            strategy_version = str(getattr(strategy, "version", "v1"))
            # Track which adapter names were missing so we warn once per job.
            missing_adapter_names: set[str] = set()

            graph_uri = kg_graph_uri(tenant_id, job.kg_name)
            # Resolve a scope predicate to concrete instance IRI(s) so the
            # entity-selection SELECT matches a predicate-indexed term, not an
            # unbounded per-entity scan (COG-112 perf fix). entity_uris path
            # never needs this. A valid predicate always resolves to candidate
            # IRIs (the direct build covers relationships under …/onto/<pred>);
            # only an empty/invalid predicate yields [] → the scope block matches
            # nothing (consistent with count_entities → matched 0).
            scope_pred_iris: Optional[list[str]] = None
            if job.scope is not None and not job.entity_uris:
                scope_pred_iris = await self._resolve_scope_predicate_iris(
                    tenant_id, job.type_name, job.scope
                )
            sel = _build_select_query(
                graph_uri,
                job.type_name,
                job.attributes,
                job.limit,
                scope=job.scope,
                entity_uris=job.entity_uris,
                scope_pred_iris=scope_pred_iris,
            )
            raw = await self._neptune.query(sel)
            _, bindings = parse_sparql_results(raw)

            entities: list[dict] = []
            for row in bindings:
                e_uri = row.get("e", "")
                if not e_uri:
                    continue
                label = row.get("label") or row.get("nameAttr") or _slug_from_uri(e_uri)
                vals = _parse_vals(row.get("vals", ""))
                entities.append({"uri": e_uri, "label": label, "vals": vals})

            job.progress.total = len(entities) * len(job.attributes)
            await self._jobs.update(job)

            sem = asyncio.Semaphore(WORKER_POOL_SIZE)
            counter = {"n": 0}
            counter_lock = asyncio.Lock()

            async def process_entity(ent: dict) -> list[RowResult]:
                results: list[RowResult] = []
                async with sem:
                    # Cooperative cancellation: check ONCE per entity, not once per
                    # attribute. The per-attribute check meant `len(attributes)`
                    # job-store reads per entity; under the durable PostgresJobStore
                    # each is a pooled-connection round-trip, so a wide job fanned
                    # `WORKER_POOL_SIZE`-way could exhaust the (max_size=10) asyncpg
                    # pool and stall the next write indefinitely (COG-112). One read
                    # per entity keeps cancellation responsive without that pressure.
                    latest = await self._jobs.get(job.id)
                    if latest and latest.status == JobStatus.cancelled:
                        return results

                    for attribute in job.attributes:
                        existing = ent["vals"].get(_attr_uri(job.type_name, attribute))
                        attr_strategy = strategy.attributes.get(attribute)

                        # Strategy merge: request value wins; ontology fills gaps.
                        # confidence_min: if ontology specifies one and the
                        # request is at the default (0.85), take the ontology
                        # value. Pragmatic heuristic since EnrichRequest has no
                        # "unset" sentinel.
                        effective_confidence = job.confidence_min
                        if attr_strategy and attr_strategy.confidence_min is not None:
                            if abs(job.confidence_min - 0.85) < 1e-9:
                                effective_confidence = attr_strategy.confidence_min

                        # Adapter chain: per-attribute sources (if any) override
                        # the tier chain.
                        if attr_strategy and attr_strategy.sources:
                            chain = list(attr_strategy.sources)
                        else:
                            chain = get_chain(job.tier)

                        verdicts = await self._lookup_chain(
                            ent["label"],
                            attribute,
                            chain,
                            job,
                            missing_adapter_names,
                            effective_confidence,
                            strategy_version,
                        )
                        best = self._pick_best(verdicts, effective_confidence)

                        action: str
                        if best is None:
                            action = "no_match"
                        elif existing is None or existing == "":
                            action = "filled"
                        elif _values_match_with_strategy(
                            existing, best.value, attr_strategy
                        ):
                            action = "verified"
                        else:
                            action = "conflict"

                        results.append(
                            RowResult(
                                entity_uri=ent["uri"],
                                attribute=attribute,
                                existing_value=existing,
                                verdict=best,
                                action=action,  # type: ignore[arg-type]
                            )
                        )

                        async with counter_lock:
                            counter["n"] += 1
                            if action == "filled":
                                job.progress.filled += 1
                            elif action == "verified":
                                job.progress.verified += 1
                            elif action == "conflict":
                                job.progress.conflicts += 1
                            elif action == "skipped":
                                job.progress.skipped += 1
                            job.progress.processed = counter["n"]
                            if counter["n"] % PROGRESS_FLUSH_EVERY == 0:
                                await self._jobs.update(job)
                return results

            tasks = [asyncio.create_task(process_entity(e)) for e in entities]
            all_rows: list[RowResult] = []
            for t in tasks:
                rows = await t
                all_rows.extend(rows)

            # Re-check cancellation after work loop.
            latest = await self._jobs.get(job.id)
            if latest and latest.status == JobStatus.cancelled:
                job.status = JobStatus.cancelled
                job.completed_at = _now()
                await self._jobs.update(job)
                return

            # Keep conflicts AND fills/verifications in results so the cited
            # verdict (value + source_url + provenance) is retrievable via the
            # job API, not just conflicts. Skips/no-matches carry no verdict.
            job.results = [r for r in all_rows if r.action in ("conflict", "filled", "verified")]

            # Apply phase
            policy = job.conflict_policy
            if policy == ConflictPolicy.stage:
                job.status = JobStatus.review
                job.completed_at = _now()
                await self._jobs.update(job)
                return

            triples = self._select_triples_for_policy(all_rows, job.type_name, policy)
            if triples:
                # Declare schema, THEN write data. Enrichment must EXTEND THE
                # ONTOLOGY (COG-112): before writing instance values, upsert the
                # ontology declaration for every attribute that actually got a
                # value (primary + its provenance companions) into the tenant
                # (ontology) graph, so an enriched attribute is first-class schema
                # — visible in the /schema view, the Explorer column schema, and
                # the Enrich dialog's predicate dropdown, not just as orphan data.
                # One idempotent upsert per attribute (not per row). This runs in
                # the apply branch only (skip/verify/overwrite); `stage` returned
                # above before any write, so review-mode declares nothing.
                applied_attrs = self._applied_attribute_names(all_rows, policy)
                await self._declare_attributes(tenant_id, job.type_name, applied_attrs)
                await self._neptune.update(insert_triples(graph_uri, triples))
                # New attribute values were written → the Explorer's precomputed
                # type-stats (coverage %, counts) are now stale. Recompute them
                # in the background so the panels refresh without waiting for the
                # summary-cache TTL. Only fire when something was actually applied.
                self._schedule_stats_recompute(tenant_id, job.kg_name)
            job.status = JobStatus.applied
            job.completed_at = _now()
            await self._jobs.update(job)

        except Exception as exc:  # noqa: BLE001
            logger.exception("enrichment_job_failed", job_id=job.id, error=str(exc))
            job.status = JobStatus.failed
            job.error = str(exc)
            job.completed_at = _now()
            try:
                await self._jobs.update(job)
            except Exception:  # noqa: BLE001
                pass

    async def _lookup(
        self,
        entity_label: str,
        attribute: str,
        job: EnrichJob,
        cache_hit_inc: bool,
        strategy_version: str = "v1",
    ) -> list[Verdict]:
        source = self._wikidata.name
        cached = await self._cache.get(
            entity_label, attribute, source, job.type_name, strategy_version
        )
        if cached is not None:
            if cache_hit_inc:
                job.progress.cache_hits += 1
            return cached
        verdicts = await self._wikidata.lookup(entity_label, attribute, {})
        await self._cache.put(
            entity_label, attribute, source, verdicts, job.type_name, strategy_version
        )
        return verdicts

    async def _lookup_chain(
        self,
        entity_label: str,
        attribute: str,
        chain: list[str],
        job: EnrichJob,
        missing: set[str],
        confidence_min: float,
        strategy_version: str = "v1",
    ) -> list[Verdict]:
        """Walk an adapter chain, returning verdicts from the first adapter
        that yields one with confidence >= confidence_min.

        - "cache" entries in the chain are skipped (cache is a layer wrapped
          around each adapter call, not an adapter itself).
        - Unregistered adapter names are skipped with a one-shot warning per
          job, never fail the job.
        """
        cache_hit_counted = False
        for name in chain:
            if name == "cache":
                # Cache is a layer, not an adapter.
                continue
            adapter = get_adapter(name)
            if adapter is None:
                if name not in missing:
                    missing.add(name)
                    logger.warning(
                        "enrichment_adapter_missing",
                        adapter=name,
                        job_id=job.id,
                        tier=job.tier.value if hasattr(job.tier, "value") else str(job.tier),
                    )
                continue
            cached = await self._cache.get(
                entity_label, attribute, adapter.name, job.type_name, strategy_version
            )
            if cached is not None:
                if not cache_hit_counted:
                    job.progress.cache_hits += 1
                    cache_hit_counted = True
                verdicts = cached
            else:
                try:
                    # Bound every adapter call so one stalled lookup (e.g. a
                    # hung network call whose own client lacks a total-operation
                    # timeout) can never strand the whole job (COG-112).
                    verdicts = await asyncio.wait_for(
                        adapter.lookup(entity_label, attribute, {}),
                        timeout=ADAPTER_LOOKUP_TIMEOUT_S,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "enrichment_adapter_timeout",
                        adapter=name,
                        job_id=job.id,
                        timeout_s=ADAPTER_LOOKUP_TIMEOUT_S,
                        entity=entity_label,
                        attribute=attribute,
                    )
                    verdicts = []
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "enrichment_adapter_error",
                        adapter=name,
                        job_id=job.id,
                        error=str(exc),
                    )
                    verdicts = []
                await self._cache.put(
                    entity_label,
                    attribute,
                    adapter.name,
                    verdicts,
                    job.type_name,
                    strategy_version,
                )
            # Stop at first sufficient-confidence verdict.
            if any(v.confidence >= confidence_min for v in verdicts):
                return verdicts
        # No adapter yielded a sufficiently-confident verdict; return last (may
        # be empty). For simplicity return [] so caller treats as no_match.
        return []

    def _pick_best(
        self, verdicts: list[Verdict], confidence_min: float
    ) -> Optional[Verdict]:
        eligible = [v for v in verdicts if v.confidence >= confidence_min]
        if not eligible:
            return None
        return max(eligible, key=lambda v: v.confidence)

    @staticmethod
    def _provenance_triples(
        entity_uri: str, type_name: str, attribute: str, verdict
    ) -> list[tuple[str, str, str]]:
        """Persist where + when an enriched value came from, as queryable
        attributes (`<attr>_source_url`, `<attr>_provenance`) on the entity — so
        the citation is visible through /ask and the Explorer, not just in the
        adapter. Audit-friendly: every enriched fact carries its source."""
        out: list[tuple[str, str, str]] = []
        if getattr(verdict, "source_url", None):
            out.append((entity_uri, _attr_uri(type_name, f"{attribute}_source_url"), verdict.source_url))
        prov = (verdict.source or "")
        if getattr(verdict, "reasoning", None):
            prov = f"{prov} ({verdict.reasoning})" if prov else verdict.reasoning
        if prov:
            out.append((entity_uri, _attr_uri(type_name, f"{attribute}_provenance"), prov))
        return out

    @staticmethod
    def _row_is_applied(r: RowResult, policy: ConflictPolicy) -> bool:
        """Whether a row's verdict actually contributes instance triples under
        ``policy``. Single source of truth shared by :meth:`_select_triples_for_policy`
        (which data to write) and :meth:`_applied_attribute_names` (which schema to
        declare) so the two can never drift."""
        if r.verdict is None:
            return False
        if policy == ConflictPolicy.overwrite:
            return r.action in ("filled", "conflict", "verified")
        if policy in (ConflictPolicy.verify, ConflictPolicy.skip):
            return r.action == "filled"
        return False

    def _select_triples_for_policy(
        self, rows: list[RowResult], type_name: str, policy: ConflictPolicy
    ) -> list[tuple[str, str, str]]:
        triples: list[tuple[str, str, str]] = []
        for r in rows:
            if not self._row_is_applied(r, policy):
                continue
            p = _attr_uri(type_name, r.attribute)
            triples.append((r.entity_uri, p, r.verdict.value))
            triples.extend(self._provenance_triples(r.entity_uri, type_name, r.attribute, r.verdict))
        return triples

    def _applied_attribute_names(
        self, rows: list[RowResult], policy: ConflictPolicy
    ) -> list[str]:
        """The attribute names (primary + their provenance companions) that
        ACTUALLY received a written value under ``policy`` — the set whose ontology
        declarations the apply step upserts so an enriched attribute becomes
        first-class schema (visible in the /schema view, the Explorer column
        schema, and the Enrich dialog's predicate dropdown). Attributes that found
        nothing are excluded so enrichment never pollutes the ontology with empty
        slots. Order-preserving + deduped so the caller issues one declaration per
        attribute, not one per row.

        The provenance companions (``<attr>_source_url``, ``<attr>_provenance``)
        are declared only when :meth:`_provenance_triples` actually emitted them
        for some applied row — matching the citation columns that were really
        written (a verdict with no ``source_url`` writes no ``_source_url`` triple,
        so we don't declare a phantom column)."""
        names: list[str] = []
        seen: set[str] = set()

        def _add(name: str) -> None:
            if name not in seen:
                seen.add(name)
                names.append(name)

        for r in rows:
            if not self._row_is_applied(r, policy):
                continue
            _add(r.attribute)
            # Mirror the companion provenance triples this row actually wrote, so
            # the declared schema matches the data exactly.
            for _s, prov_pred, _o in self._provenance_triples(
                r.entity_uri, "", r.attribute, r.verdict
            ):
                _add(_local_name(prov_pred))
        return names

    async def _declare_attributes(
        self, tenant_id: str, type_name: str, attr_names: list[str]
    ) -> None:
        """Upsert each enrichment-applied attribute's ontology declaration into the
        TENANT (ontology) graph so it becomes first-class schema. Reuses the same
        idempotent :func:`upsert_attribute` the ontology endpoint uses
        (``rdf:Property ; rdfs:label ; rdfs:domain <Type> ; rdfs:range xsd:string``),
        one update per attribute. Called BEFORE the instance ``insert_triples``
        write (declare schema, then write data) and inside the job's try/except so
        a declaration failure fails the job, consistent with existing behavior."""
        onto_graph = tenant_graph_uri(tenant_id)
        for name in attr_names:
            await self._neptune.update(
                upsert_attribute(
                    onto_graph,
                    type_name,
                    name,
                    description=ENRICH_ATTR_DESCRIPTION,
                    datatype=ENRICH_ATTR_DATATYPE,
                )
            )

    async def apply_decisions(
        self, job_id: str, decisions: list[ConflictReview]
    ) -> int:
        job = await self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        graph_uri = kg_graph_uri(job.tenant_id, job.kg_name)
        triples: list[tuple[str, str, str]] = []
        applied = 0  # number of accepted facts (provenance triples don't count)
        applied_attrs: list[str] = []
        seen_attrs: set[str] = set()
        for d in decisions:
            if d.decision != "accept":
                continue
            p = _attr_uri(job.type_name, d.attribute)
            triples.append((d.entity_uri, p, d.proposed.value))
            prov = self._provenance_triples(d.entity_uri, job.type_name, d.attribute, d.proposed)
            triples.extend(prov)
            # Track the attribute names (primary + provenance companions actually
            # written) so we can declare them in the ontology, mirroring run().
            for name in [d.attribute, *(_local_name(pp) for _s, pp, _o in prov)]:
                if name not in seen_attrs:
                    seen_attrs.add(name)
                    applied_attrs.append(name)
            applied += 1
        if triples:
            # Declare schema, THEN write data — accepted review decisions extend
            # the ontology too (COG-112), so the enriched attribute is first-class
            # schema, mirroring the auto-apply path in run().
            await self._declare_attributes(job.tenant_id, job.type_name, applied_attrs)
            await self._neptune.update(insert_triples(graph_uri, triples))
            # Accepted facts were written → refresh the Explorer's precomputed
            # type-stats in the background (mirrors the auto-apply path in run()).
            self._schedule_stats_recompute(job.tenant_id, job.kg_name)
        job.status = JobStatus.applied
        job.completed_at = _now()
        await self._jobs.update(job)
        return applied

    def _schedule_stats_recompute(self, tenant_id: str, kg_name: str) -> None:
        """Fire-and-forget a type-stats recompute after an enrichment write.

        Lazy-imported from the explore route to avoid a module-load import cycle
        (the API route modules cross-reference one another; ingest.py uses the
        same lazy pattern). ``schedule_recompute`` is best-effort — it swallows
        Neptune errors internally — so this never affects the job's outcome.
        """
        from cograph_client.api.routes.explore import schedule_recompute

        schedule_recompute(self._neptune, tenant_id, kg_name)


def _slug_from_uri(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1]
