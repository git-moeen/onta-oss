"""Async executor for enrichment jobs.

Reads entities from Neptune, runs them through the source funnel
(lite tier = wikidata, with cache), and either stages results for
review or applies them directly based on conflict_policy.
"""

from __future__ import annotations

import asyncio
import hashlib
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
    JobErrorItem,
    JobStatus,
    ProviderLog,
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
from cograph_client.enrichment.extraction import coerce_url_attribute_value
from cograph_client.enrichment.tiers import get_chain
from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.kg_writer import insert_facts, refresh_after_write
from cograph_client.graph.ontology_queries import (
    PRIMITIVE_TYPES,
    XSD_STRING,
    get_attribute_range_query,
    upsert_attribute,
    xsd_to_datatype,
)
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import (
    kg_graph_uri,
    tenant_graph_uri,
)
from cograph_client.resolver.models import ValidatedTriple
from cograph_client.resolver.validator import validate_triple

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
# one declared by ingest or the ontology endpoint.
ENRICH_ATTR_DESCRIPTION = "Added by enrichment job"
# Default declared range when a brand-new enriched attribute carries no values we
# can type (empty / non-numeric). The actual range is INFERRED per-attribute from
# the applied values (_infer_datatype_from_values) and, for an attribute already
# declared with a richer range by ingestion, the existing range is PRESERVED
# rather than downgraded (see _declare_attributes) — so this is only the floor.
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

# Cap stored per-provider error/summary messages so a chatty adapter exception
# can't bloat the job payload (it is serialized whole into the job store).
_MAX_ERROR_MSG = 300


class _ProviderTally:
    """Accumulates per-provider outcomes across a single enrichment run so the
    job can carry a ``provider_logs`` (what each provider we used did) and an
    ``error_summary`` (the potential errors, aggregated) for the run-detail view.

    Concurrency: the executor's worker pool runs cooperatively under one event
    loop and every ``record*`` mutation is synchronous (no ``await`` between read
    and write), so the plain counters here are race-free — the same property the
    existing ``job.progress`` increments rely on. No lock needed.
    """

    def __init__(self) -> None:
        self._by_provider: dict[str, ProviderLog] = {}
        # (provider, kind) -> [count, first_sample_message]
        self._errors: dict[tuple[str, str], list] = {}

    def _log(self, provider: str) -> ProviderLog:
        pl = self._by_provider.get(provider)
        if pl is None:
            pl = ProviderLog(provider=provider)
            self._by_provider[provider] = pl
        return pl

    def _bump_error(self, provider: str, kind: str, message: str) -> None:
        key = (provider, kind)
        rec = self._errors.get(key)
        if rec is None:
            self._errors[key] = [1, (message or "")[:_MAX_ERROR_MSG]]
        else:
            rec[0] += 1  # keep the first representative message

    def record_missing(self, provider: str) -> None:
        """A chain named a provider that isn't registered here (call once per
        provider per job — the caller already gates on a 'missing' set)."""
        self._log(provider).status = "skipped"
        self._bump_error(
            provider,
            "missing",
            f"provider '{provider}' is not registered on this deployment",
        )

    def record_attempt(
        self,
        provider: str,
        *,
        cache_hit: bool,
        outcome: str,  # "match" | "no_match" | "timeout" | "error"
        error_msg: Optional[str] = None,
    ) -> None:
        pl = self._log(provider)
        if cache_hit:
            pl.cache_hits += 1
        else:
            pl.attempts += 1
        if outcome == "match":
            pl.matches += 1
        elif outcome == "no_match":
            pl.no_match += 1
        elif outcome == "timeout":
            pl.timeouts += 1
            pl.last_error = (error_msg or "lookup timed out")[:_MAX_ERROR_MSG]
            self._bump_error(provider, "timeout", error_msg or "lookup timed out")
        elif outcome == "error":
            pl.errors += 1
            if error_msg:
                pl.last_error = error_msg[:_MAX_ERROR_MSG]
            self._bump_error(provider, "error", error_msg or "lookup failed")

    def to_logs(self) -> list[ProviderLog]:
        out: list[ProviderLog] = []
        for pl in self._by_provider.values():
            if pl.status != "skipped":
                if pl.matches > 0:
                    pl.status = "ok"
                elif pl.errors > 0 or pl.timeouts > 0:
                    pl.status = "error"
                else:
                    pl.status = "no_match"
            out.append(pl)
        return out

    def to_error_summary(self) -> list[JobErrorItem]:
        items = [
            JobErrorItem(provider=prov, kind=kind, message=msg, count=count)  # type: ignore[arg-type]
            for (prov, kind), (count, msg) in self._errors.items()
        ]
        items.sort(key=lambda e: e.count, reverse=True)
        return items


def _type_uri(type_name: str) -> str:
    return f"{TYPE_URI_PREFIX}{type_name}"


def _attr_uri(type_name: str, attr: str) -> str:
    return f"{TYPE_URI_PREFIX}{type_name}/attrs/{attr}"


def _esc_lit(value: str) -> str:
    """Escape a string for use inside a SPARQL double-quoted literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _strategy_version_with_instructions(
    strategy_version: str, instructions: Optional[str]
) -> str:
    """Fold optional ``instructions`` into the cache ``strategy_version`` string.

    Custom instructions can change what an agentic adapter returns, so two
    different instruction sets must NOT collide on a cached verdict. Rather than
    widen the cache key tuple (and every call site), we append a short stable
    hash of the instructions to ``strategy_version`` — a different instructions
    string yields a different key (clean miss), the same string reuses the
    cached verdict, and the no-instructions path is BYTE-FOR-BYTE the old
    ``strategy_version`` (so existing caches/keys are unchanged)."""
    if not instructions:
        return strategy_version
    digest = hashlib.sha256(instructions.encode("utf-8")).hexdigest()[:12]
    return f"{strategy_version}+instr:{digest}"


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


def _is_int(v: str) -> bool:
    """True if ``v`` parses as a plain int (optional leading sign only).

    Mirrors agent/capabilities/web_ingest_cap.py's helper — kept as a small local
    copy rather than imported so the enrichment layer takes no dependency on the
    agent layer.

    This MUST agree with the write-side validator (``resolver.validator``): its
    ``validate_value`` accepts integers as ``^-?\\d+$`` and ``coerce_value`` does
    ``int(float(v))`` — neither strips thousands separators. So we reject ``,`` and
    ``_`` groupings here too. If the inference layer declared ``xsd:integer`` for a
    comma-grouped value the validator would then REJECT (drop) it at write time, so
    a column like ``"1,234"`` must declare ``string`` and keep the value as a
    visible string literal rather than vanish."""
    if not isinstance(v, str) or "_" in v or "," in v:
        return False
    try:
        int(v)
        return True
    except (ValueError, AttributeError):
        return False


def _is_float(v: str) -> bool:
    """True if ``v`` parses as a finite float (optional leading sign only).

    Like :func:`_is_int`, this MUST agree with the write-side validator, which does
    not strip thousands separators — so we reject ``,`` and ``_`` groupings (else a
    comma value would be declared numeric and then dropped at write). Python's
    ``float()`` also parses the special tokens ``inf``/``-inf``/``infinity``/``nan``,
    none of which are real numeric data, so we reject those too. Ordinary decimals
    and scientific notation of real numbers (``8.5``, ``1e10``) still parse True."""
    if not isinstance(v, str) or "_" in v or "," in v:
        return False
    # Reject the non-finite special tokens float() accepts (inf/-inf/infinity/nan).
    cleaned = v.strip().lstrip("+-").lower()
    if cleaned in ("inf", "infinity", "nan"):
        return False
    try:
        f = float(v)
    except (ValueError, AttributeError):
        return False
    # Belt-and-suspenders: any non-finite result (should already be caught above)
    # is not a real float value.
    import math

    return math.isfinite(f)


ENTITY_URI_PREFIX = "https://cograph.tech/entities/"


def _is_iso_datetime(v: str) -> bool:
    """True if ``v`` parses as an ISO-8601 date or datetime via
    :meth:`datetime.fromisoformat`.

    Accepts plain dates (``2026-06-28``), datetimes (``2026-06-28T21:24:50``),
    and timezone-aware forms (``…+00:00`` and a trailing ``Z``, which Python's
    pre-3.11 ``fromisoformat`` rejects, so we normalise ``Z`` to ``+00:00``
    first). A bare integer like ``2026`` is deliberately NOT a date here — the
    caller only reaches this helper for values that already failed int/float and
    contain a date separator, so an all-integer column can never be misread as a
    date."""
    if not isinstance(v, str):
        return False
    s = v.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        datetime.fromisoformat(s)
        return True
    except ValueError:
        return False


def _entity_iri_type(value: str) -> str | None:
    """Parse the ``<TypeName>`` out of a canonical entity IRI of the form
    ``https://cograph.tech/entities/<TypeName>/<id>``, else None.

    Returns the bare type name (e.g. ``Manufacturer``) so the caller can decide
    whether a column of entity IRIs is a relationship to a single target type.
    Returns None for anything that is not such an IRI — a literal, a different
    URI namespace, or a malformed entities IRI missing the ``<id>`` segment."""
    if not isinstance(value, str) or not value.startswith(ENTITY_URI_PREFIX):
        return None
    rest = value[len(ENTITY_URI_PREFIX):]
    parts = rest.split("/", 1)
    # Need a non-empty <TypeName> AND a non-empty <id> segment.
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0]


def _infer_datatype_from_values(values: list[str]) -> str:
    """Cheap datatype guess from the actual enriched values for one attribute.

    Precedence (first match wins), each requiring ALL non-empty values to agree:
      1. ``integer`` — every value parses as an int.
      2. ``float`` — every value parses as a float.
      3. ``datetime`` — every value is an ISO-8601 date/datetime (checked only
         for values that failed int/float AND carry a date separator ``-``/``T``/
         ``:``, so an all-integer column like ``2026`` is never misread as a
         date). ``datetime`` is the name ``_datatype_to_xsd`` maps to
         ``xsd:dateTime``.
      4. a bare ``<TypeName>`` (a RELATIONSHIP range) — every value is a
         canonical entity IRI (``…/entities/<TypeName>/<id>``) AND they all share
         the SAME ``<TypeName>``. ``_datatype_to_xsd`` maps that bare name to the
         ``types/<TypeName>`` URI (an object-property range). Mixed IRI types →
         no single range, so we fall through to string (don't guess).
      5. ``string`` — the safe floor (also for empty / all-blank).

    Date and relationship detection (E2) are now attempted because they DO
    round-trip reliably: an ISO date and a canonical entity IRI are both exact,
    machine-minted forms, unlike free-text. Mirrors web_ingest_cap._infer_datatype
    for the primitive cases."""
    vals = [str(v).strip() for v in values if v not in (None, "")]
    vals = [v for v in vals if v]
    if not vals:
        return "string"
    if all(_is_int(v) for v in vals):
        return "integer"
    if all(_is_float(v) for v in vals):
        return "float"
    # Date only for values that look temporal (carry a date separator) and are
    # not numeric — guards an all-integer column from a date false-positive.
    if all(any(c in v for c in "-T:") and _is_iso_datetime(v) for v in vals):
        return "datetime"
    # Relationship: all values are entity IRIs sharing one target type.
    iri_types = [_entity_iri_type(v) for v in vals]
    if all(t is not None for t in iri_types) and len(set(iri_types)) == 1:
        return iri_types[0]  # bare <TypeName> → types/<TypeName> range
    return "string"


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
        # Per-provider activity + error accumulator for this run; stamped onto the
        # job at every terminal path so the run-detail view shows which providers
        # we used and a summary of the errors hit. Defined before the try so the
        # failure path can still surface whatever was recorded before the crash.
        tally = _ProviderTally()
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
            # Fold optional custom instructions into the cache version so two
            # different instruction sets never collide on a cached verdict (an
            # agentic adapter can read job.instructions and return a different
            # value). No instructions → unchanged version (clean reuse of the
            # existing cache keys). See _strategy_version_with_instructions.
            strategy_version = _strategy_version_with_instructions(
                strategy_version, job.instructions
            )
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
                    for attribute in job.attributes:
                        # Cooperative cancellation
                        latest = await self._jobs.get(job.id)
                        if latest and latest.status == JobStatus.cancelled:
                            return results

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

                        # Adapter chain precedence (most specific wins):
                        #   1. per-attribute ontology strategy sources, then
                        #   2. the request-level job.sources override, then
                        #   3. the tier default chain.
                        if attr_strategy and attr_strategy.sources:
                            chain = list(attr_strategy.sources)
                        elif job.sources:
                            # Request-level provider override. Keep only names
                            # that resolve to a registered adapter; if the
                            # override names ONLY unavailable providers (e.g. a
                            # premium adapter not registered on this deployment),
                            # fall back to the tier default chain rather than
                            # enriching nothing — matching the UI's "falls back
                            # to Auto if unavailable" promise. A partially-valid
                            # override uses just its available names.
                            available = [
                                s for s in job.sources if get_adapter(s) is not None
                            ]
                            chain = available if available else get_chain(job.tier)
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
                            tally=tally,
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
                            elif action == "no_match":
                                job.progress.no_match += 1
                            job.progress.processed = counter["n"]
                            if counter["n"] % PROGRESS_FLUSH_EVERY == 0:
                                await self._jobs.update(job)
                return results

            tasks = [asyncio.create_task(process_entity(e)) for e in entities]
            all_rows: list[RowResult] = []
            for t in tasks:
                rows = await t
                all_rows.extend(rows)

            # Stamp the per-provider activity log + aggregated error summary onto
            # the job now, so every terminal path below (cancelled, review,
            # applied) persists "which providers we used + the errors we hit".
            job.provider_logs = tally.to_logs()
            job.error_summary = tally.to_error_summary()

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

            # One structured summary on the common terminal path (covers BOTH the
            # review and applied states below). Makes the miss count visible from
            # logs so a run that simply found nothing is distinguishable from a
            # broken pipeline. NOT emitted on the cancelled/failed early-returns.
            logger.info(
                "enrichment_job_summary",
                job_id=job.id,
                type_name=job.type_name,
                tier=job.tier.value if hasattr(job.tier, "value") else str(job.tier),
                total=job.progress.total,
                filled=job.progress.filled,
                verified=job.progress.verified,
                conflicts=job.progress.conflicts,
                no_match=job.progress.no_match,
                sources_tried=sorted(
                    {
                        r.verdict.source
                        for r in all_rows
                        if r.verdict and getattr(r.verdict, "source", None)
                    }
                ),
            )

            # Apply phase
            policy = job.conflict_policy
            # `stage` semantics (ONTA-159): a conflict-free fill (the target field
            # was empty) has nothing to reconcile, so it is applied immediately —
            # exactly like `skip`. Only genuine value-vs-value CONFLICTS are held
            # for human review. Previously `stage` also held fills, but the review
            # surface (`GET /jobs/{id}/conflicts`) lists ONLY conflict rows, so
            # conflict-free staged fills were stranded: staged yet invisible and
            # un-approvable — a job sat "In review" with zero reviewable items.
            # So under `stage` we WRITE like `skip` (fills only) and land in
            # `review` only when there is at least one real conflict to resolve.
            has_conflicts = any(r.action == "conflict" for r in job.results)
            write_policy = (
                ConflictPolicy.skip if policy == ConflictPolicy.stage else policy
            )

            # `applied_attr_values` is the source of truth for "was anything
            # applied?" — the attributes (primary + provenance companions) that
            # actually received a written value under `write_policy`, mapped to
            # their values. Empty ⇒ nothing to declare or write.
            applied_attr_values = self._applied_attribute_values(all_rows, write_policy)
            if applied_attr_values:
                # Declare schema, THEN write data. Enrichment must EXTEND THE
                # ONTOLOGY (COG-112): before writing instance values, upsert the
                # ontology declaration for every attribute that actually got a
                # value (primary + its provenance companions) into the tenant
                # (ontology) graph, so an enriched attribute is first-class schema
                # — visible in the /schema view, the Explorer column schema, and
                # the Enrich dialog's predicate dropdown, not just as orphan data.
                # One idempotent upsert per attribute (not per row), each declared
                # with a range inferred from its actual applied values and never
                # downgrading an existing richer range. Runs for every write
                # policy AND for `stage`'s conflict-free fills (which now write
                # via `write_policy=skip`); only true conflicts held for review
                # declare nothing until accepted.
                #
                # `_declare_attributes` RETURNS the {attr -> resolved_datatype} map
                # it just declared, so we type each INSTANCE value with the SAME
                # datatype the attribute is DECLARED with (P1 fix): the stored
                # literal (`"92"^^xsd:integer`) now matches the declared range,
                # instead of a bare `xsd:string` literal the typed NL filters miss.
                resolved_datatypes = await self._declare_attributes(
                    tenant_id, job.type_name, applied_attr_values
                )
                # Build the instance triples USING that resolved-datatype map:
                # primitives route through validate_triple (typed literal, or a
                # skip on a non-conforming value); relationships write the entity
                # IRI directly; provenance companions stay plain string literals.
                triples = self._select_triples_for_policy(
                    all_rows, job.type_name, write_policy, resolved_datatypes
                )
                # Single shared write path — identical to CSV/JSON ingestion
                # (graph/kg_writer.py): batched insert, then post-write
                # housekeeping (invalidate the NL-planning cache, re-embed the
                # enriched type so semantic retrieval doesn't serve a stale schema
                # embedding, and recompute the Explorer's type-stats). Only fires
                # when something was actually applied.
                await insert_facts(self._neptune, graph_uri, triples)
                await refresh_after_write(
                    self._neptune,
                    tenant_id=tenant_id,
                    kg_name=job.kg_name,
                    affected_types={job.type_name},
                )
            # `stage` with at least one real conflict stays in `review` — those
            # conflicts are now the ONLY thing the review queue holds (the fills
            # were just applied above). Everything else — a `stage` run with no
            # conflicts, or any write policy — is `applied`.
            if policy == ConflictPolicy.stage and has_conflicts:
                job.status = JobStatus.review
            else:
                job.status = JobStatus.applied
            job.completed_at = _now()
            await self._jobs.update(job)

        except Exception as exc:  # noqa: BLE001
            logger.exception("enrichment_job_failed", job_id=job.id, error=str(exc))
            job.status = JobStatus.failed
            job.error = str(exc)
            job.completed_at = _now()
            # Surface whatever providers ran (and any per-provider errors) before
            # the fatal crash, plus the crash itself as a job-level error entry.
            job.provider_logs = tally.to_logs()
            job.error_summary = tally.to_error_summary() + [
                JobErrorItem(kind="job", message=str(exc)[:_MAX_ERROR_MSG])
            ]
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
        # Thread optional custom instructions into the lookup context (empty
        # when none), mirroring _lookup_chain. Wikidata ignores it harmlessly.
        ctx = {"instructions": job.instructions} if job.instructions else {}
        # URL-targeted enrichment: hand any user-supplied pages to the adapter so
        # a URL-aware premium adapter (e.g. Firecrawl) reads values FROM them.
        # Wikidata ignores it harmlessly. Empty by default → unchanged call shape.
        if job.source_urls:
            ctx["target_urls"] = list(job.source_urls)
        verdicts = await self._wikidata.lookup(entity_label, attribute, ctx)
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
        tally: Optional["_ProviderTally"] = None,
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
                    if tally is not None:
                        tally.record_missing(name)
                continue
            # Per-attempt outcome for the provider log: "match" | "no_match" |
            # "timeout" | "error", with the cache flag tracked separately.
            from_cache = False
            err_outcome: Optional[str] = None
            err_msg: Optional[str] = None
            cached = await self._cache.get(
                entity_label, attribute, adapter.name, job.type_name, strategy_version
            )
            if cached is not None:
                if not cache_hit_counted:
                    job.progress.cache_hits += 1
                    cache_hit_counted = True
                verdicts = cached
                from_cache = True
            else:
                # Optional custom instructions ride in the adapter lookup
                # context dict. Adapters that don't use it (wikidata) ignore it
                # harmlessly; agentic/premium adapters can read it. Empty when no
                # instructions so the call shape is unchanged in the common case.
                ctx = {"instructions": job.instructions} if job.instructions else {}
                # URL-targeted enrichment: hand any user-supplied pages to the
                # adapter via ``target_urls`` so a URL-aware premium adapter
                # (e.g. Firecrawl) reads values FROM them. Free adapters ignore
                # it harmlessly; empty by default → unchanged call shape.
                if job.source_urls:
                    ctx["target_urls"] = list(job.source_urls)
                try:
                    # Bound every adapter call so one stalled lookup (e.g. a
                    # hung network call whose own client lacks a total-operation
                    # timeout) can never strand the whole job (COG-112).
                    verdicts = await asyncio.wait_for(
                        adapter.lookup(entity_label, attribute, ctx),
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
                    err_outcome = "timeout"
                    err_msg = f"timed out after {ADAPTER_LOOKUP_TIMEOUT_S:.0f}s"
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "enrichment_adapter_error",
                        adapter=name,
                        job_id=job.id,
                        error=str(exc),
                    )
                    verdicts = []
                    err_outcome = "error"
                    err_msg = str(exc)
                await self._cache.put(
                    entity_label,
                    attribute,
                    adapter.name,
                    verdicts,
                    job.type_name,
                    strategy_version,
                )
            # URL-valued attributes (website, *_url, datatype uri): the answer is
            # a URL, and a single-pass extractor run over page text otherwise
            # lifts page chrome ("Skip to content", "Platform") or the entity
            # name as the value. Coerce to a URL — keeping an already-URL value
            # (e.g. Wikidata's official site) and only falling back to the
            # resolved source_url citation when the value isn't a URL (ONTA-157).
            # Applied here, the one shared post-adapter seam, so it covers every
            # provider (and re-coerces stale cached verdicts on read).
            verdicts = [coerce_url_attribute_value(attribute, v) for v in verdicts]
            sufficient = any(v.confidence >= confidence_min for v in verdicts)
            if tally is not None:
                outcome = (
                    err_outcome
                    if err_outcome is not None
                    else ("match" if sufficient else "no_match")
                )
                tally.record_attempt(
                    adapter.name,
                    cache_hit=from_cache,
                    outcome=outcome,
                    error_msg=err_msg,
                )
            # Stop at first sufficient-confidence verdict.
            if sufficient:
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
        self,
        rows: list[RowResult],
        type_name: str,
        policy: ConflictPolicy,
        resolved_datatypes: dict[str, str],
    ) -> list[tuple[str, str, str]]:
        """Build the instance triples to write for ``policy``.

        ``resolved_datatypes`` is the ``{attribute -> datatype}`` map
        :meth:`_declare_attributes` just declared, so each primary value is TYPED
        with the SAME datatype its attribute is DECLARED with (P1 fix). The primary
        value goes through :meth:`_instance_triples_for_value` (relationship → IRI;
        primitive → ``validate_triple`` typed literal, skipped if non-conforming);
        the provenance companions (``*_source_url`` / ``*_provenance``) stay plain
        string literals exactly as before."""
        triples: list[tuple[str, str, str]] = []
        for r in rows:
            if not self._row_is_applied(r, policy):
                continue
            # Default to ``string`` if a datatype somehow wasn't resolved for this
            # attribute (defensive — _declare_attributes covers every applied attr).
            datatype = resolved_datatypes.get(r.attribute, "string")
            triples.extend(
                self._instance_triples_for_value(
                    r.entity_uri, type_name, r.attribute, r.verdict.value, datatype
                )
            )
            # Provenance companions are user-facing citations (URLs / free text) —
            # always plain string literals, never typed.
            triples.extend(self._provenance_triples(r.entity_uri, type_name, r.attribute, r.verdict))
        return triples

    def _applied_attribute_values(
        self, rows: list[RowResult], policy: ConflictPolicy
    ) -> dict[str, list[str]]:
        """The attribute names (primary + their provenance companions) that
        ACTUALLY received a written value under ``policy``, mapped to the list of
        string VALUES applied for each — the set whose ontology declarations the
        apply step upserts so an enriched attribute becomes first-class schema
        (visible in the /schema view, the Explorer column schema, and the Enrich
        dialog's predicate dropdown). Attributes that found nothing are excluded
        so enrichment never pollutes the ontology with empty slots. Insertion-
        ordered + value-accumulating so the caller issues one declaration per
        attribute (not one per row) AND can infer that attribute's datatype from
        the actual values written.

        The provenance companions (``<attr>_source_url``, ``<attr>_provenance``)
        are declared only when :meth:`_provenance_triples` actually emitted them
        for some applied row — matching the citation columns that were really
        written (a verdict with no ``source_url`` writes no ``_source_url`` triple,
        so we don't declare a phantom column). Their values are strings (URLs /
        free text) so they naturally infer ``string``."""
        out: dict[str, list[str]] = {}

        for r in rows:
            if not self._row_is_applied(r, policy):
                continue
            out.setdefault(r.attribute, []).append(r.verdict.value)
            # Mirror the companion provenance triples this row actually wrote, so
            # the declared schema (and its inferred datatype) matches the data.
            for _s, prov_pred, prov_val in self._provenance_triples(
                r.entity_uri, "", r.attribute, r.verdict
            ):
                out.setdefault(_local_name(prov_pred), []).append(prov_val)
        return out

    async def _resolve_declared_datatype(
        self, onto_graph: str, type_name: str, attr_name: str, values: list[str]
    ) -> str:
        """Resolve the ``datatype`` to declare for one enriched attribute, never
        DOWNGRADING an existing richer range.

        Two inputs combine:
          1. The datatype INFERRED from the actual applied ``values`` (integer /
             float / string) — so a numeric enriched attribute is typed, not
             stamped ``xsd:string`` blindly.
          2. The attribute's range as ALREADY declared in the ontology. If that
             existing range is anything other than ``xsd:string`` — a richer XSD
             primitive (integer/float/dateTime) OR a relationship ``types/<X>``
             URI declared by ingestion — it is PRESERVED verbatim; enrichment must
             not clobber an ingest-inferred integer or a relationship edge down to
             a string.

        Net rule: ``existing_range if (existing_range and existing_range !=
        xsd:string) else inferred``. With no existing range, or an existing
        ``xsd:string``, the inferred datatype wins (so a brand-new attribute is
        typed correctly, and a previously-untyped string slot can be upgraded)."""
        inferred = _infer_datatype_from_values(values)
        raw = await self._neptune.query(
            get_attribute_range_query(onto_graph, type_name, attr_name)
        )
        _, bindings = parse_sparql_results(raw)
        existing = bindings[0].get("range") if bindings else None
        if existing and existing != XSD_STRING:
            # Re-assert the existing richer range verbatim (round-trips back to the
            # same URI through upsert_attribute -> _datatype_to_xsd).
            return xsd_to_datatype(existing)
        return inferred

    async def _declare_attributes(
        self, tenant_id: str, type_name: str, attr_values: dict[str, list[str]]
    ) -> dict[str, str]:
        """Upsert each enrichment-applied attribute's ontology declaration into the
        TENANT (ontology) graph so it becomes first-class schema. Reuses the same
        idempotent :func:`upsert_attribute` the ontology endpoint uses
        (``rdf:Property ; rdfs:label ; rdfs:domain <Type> ; rdfs:range <…>``), one
        update per attribute. The declared ``rdfs:range`` is resolved per attribute
        by :meth:`_resolve_declared_datatype`: inferred from the actual applied
        values, but never downgrading an existing richer range. Called BEFORE the
        instance ``insert_triples`` write (declare schema, then write data) and
        inside the job's try/except so a declaration failure fails the job,
        consistent with existing behavior.

        ``attr_values`` maps each applied attribute name (primary + provenance
        companions) to the string values written for it.

        Returns the ``{attribute_name -> resolved_datatype}`` map so the caller can
        type each INSTANCE value with the SAME datatype the attribute is DECLARED
        with (P1 data-correctness fix): a numeric value must be stored as a typed
        literal (``"92"^^xsd:integer``) matching the declared integer range, not as
        a bare ``xsd:string`` literal the typed NL filters then miss. Computing the
        datatype ONCE here and reusing it for both the declaration and the value
        typing is what keeps the declared range and the stored literal in lock-step.
        The provenance companions resolve to ``string`` (URLs / free text) and are
        intentionally never typed as anything richer."""
        onto_graph = tenant_graph_uri(tenant_id)
        resolved: dict[str, str] = {}
        for name, values in attr_values.items():
            datatype = await self._resolve_declared_datatype(
                onto_graph, type_name, name, values
            )
            resolved[name] = datatype
            await self._neptune.update(
                upsert_attribute(
                    onto_graph,
                    type_name,
                    name,
                    description=ENRICH_ATTR_DESCRIPTION,
                    datatype=datatype,
                )
            )
        return resolved

    @staticmethod
    def _instance_triples_for_value(
        entity_uri: str,
        type_name: str,
        attribute: str,
        value: str,
        datatype: str,
    ) -> list[tuple[str, str, str]]:
        """Build the instance triple(s) for ONE applied attribute value, typed with
        the SAME resolved ``datatype`` the attribute is DECLARED with (P1 fix).

        Two branches mirror ingestion's value-typing path
        (resolver/schema_resolver.py around 1393–1410):

        - **relationship** — ``datatype`` is NOT a primitive (it is an entity-type
          name, e.g. ``Manufacturer``), so the value is an entity IRI. Write it
          directly as the object IRI; ``_escape_value`` wraps an ``http(s)`` object
          as ``<…>``. We do NOT run ``validate_triple`` for relationships (the IRI
          is already the right shape and isn't an XSD-typed literal).
        - **primitive** (string/integer/float/datetime/boolean/uri) — route the
          value through the SAME ``validate_triple`` ingestion uses so the stored
          literal is properly TYPED (``"92^^…#integer"`` → a typed literal via
          ``_escape_value``). A ``ValidatedTriple`` is written; a ``RejectedValue``
          (value can't conform/coerce to the declared range) yields NO triple — we
          skip it rather than pin a mismatched literal that the typed NL filters
          would then miss (validate_triple already logs the rejection).

        Returns ``[]`` when a primitive value is rejected; otherwise the single
        instance triple."""
        attr_uri_str = _attr_uri(type_name, attribute)
        # Relationship: a non-primitive datatype is an entity-type name → the value
        # is the target entity IRI. Write the edge directly (no validate_triple).
        if datatype not in PRIMITIVE_TYPES:
            return [(entity_uri, attr_uri_str, value)]
        # Primitive: type the literal exactly as ingestion does. validate_triple
        # returns a ValidatedTriple (typed object) on conform/coerce, else a
        # RejectedValue (skip — never write a literal that mismatches the range).
        validated = validate_triple(
            entity_uri,
            attr_uri_str,
            value,
            datatype,
            entity_id=entity_uri,
            attribute_name=attribute,
        )
        if isinstance(validated, ValidatedTriple):
            return [(validated.subject, validated.predicate, validated.object)]
        return []

    async def apply_decisions(
        self, job_id: str, decisions: list[ConflictReview]
    ) -> int:
        job = await self._jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        graph_uri = kg_graph_uri(job.tenant_id, job.kg_name)
        applied = 0  # number of accepted facts (provenance triples don't count)
        # Insertion-ordered map of applied attribute name -> the string values
        # written for it, so declarations infer the right range (and never
        # downgrade an existing one), mirroring run()'s _applied_attribute_values.
        applied_attr_values: dict[str, list[str]] = {}
        # The accepted decisions whose primary value we'll later type + write.
        accepted: list[ConflictReview] = []
        for d in decisions:
            if d.decision != "accept":
                continue
            accepted.append(d)
            prov = self._provenance_triples(d.entity_uri, job.type_name, d.attribute, d.proposed)
            # Track the attribute names + values (primary + provenance companions
            # actually written) so we declare them in the ontology, mirroring run().
            applied_attr_values.setdefault(d.attribute, []).append(d.proposed.value)
            for _s, prov_pred, prov_val in prov:
                applied_attr_values.setdefault(_local_name(prov_pred), []).append(prov_val)
            applied += 1
        if applied_attr_values:
            # Declare schema, THEN write data — accepted review decisions extend
            # the ontology too (COG-112), so the enriched attribute is first-class
            # schema, mirroring the auto-apply path in run(). The returned
            # {attr -> resolved_datatype} map types each INSTANCE value with the
            # SAME datatype the attribute is DECLARED with (P1 fix): the stored
            # literal matches the declared range instead of a bare xsd:string.
            resolved_datatypes = await self._declare_attributes(
                job.tenant_id, job.type_name, applied_attr_values
            )
            # Build the instance triples USING that map: primitives route through
            # validate_triple (typed literal, or a skip on a non-conforming value);
            # relationships write the entity IRI directly; provenance companions
            # stay plain string literals.
            triples: list[tuple[str, str, str]] = []
            for d in accepted:
                datatype = resolved_datatypes.get(d.attribute, "string")
                triples.extend(
                    self._instance_triples_for_value(
                        d.entity_uri, job.type_name, d.attribute, d.proposed.value, datatype
                    )
                )
                triples.extend(
                    self._provenance_triples(
                        d.entity_uri, job.type_name, d.attribute, d.proposed
                    )
                )
            # Same shared write path as run() / ingestion (graph/kg_writer.py):
            # batched insert + post-write housekeeping (cache-invalidate,
            # re-embed the type, recompute stats).
            await insert_facts(self._neptune, graph_uri, triples)
            await refresh_after_write(
                self._neptune,
                tenant_id=job.tenant_id,
                kg_name=job.kg_name,
                affected_types={job.type_name},
            )
        job.status = JobStatus.applied
        job.completed_at = _now()
        await self._jobs.update(job)
        return applied


def _slug_from_uri(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1]
