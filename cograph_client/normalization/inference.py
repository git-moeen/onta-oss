"""Inference agent — propose normalization rules for a type's predicates.

:func:`suggest_rules` enumerates a type's attributes and relationships from the
ontology, draws a few INDEPENDENT samples of distinct values per predicate from
the KG graph, and asks an LLM (once per predicate) whether that predicate needs
normalization and of which kind(s). It detects two problems today:

* ``list_explode`` — multi-valued source cells that were collapsed into one
  composite value (a delimited literal, or a composite entity whose
  local-name/label packs several atomic values joined by the slug ``__``).
* ``strip_emoji`` — text values carrying emoji / pictographic junk characters
  (``"🎨 design; 🚀 growth"``) that should be removed.

A single predicate can warrant BOTH (``skills`` often needs ``list_explode``
AND ``strip_emoji``), so the LLM returns a LIST of recommended rules per
predicate and :func:`suggest_rules` emits one :class:`NormalizationRule` per
recommendation (deduped by ``(predicate, rule_type)``; the deterministic id
folds in ``rule_type`` so the two never collide).

Nothing is persisted here — the route persists the returned ``suggested`` rules
so a human can confirm them. Rules come back ranked by confidence (desc).
"""

from __future__ import annotations

import json
import os

import structlog

from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.ontology_queries import RDF, RDFS, type_uri
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import kg_graph_uri
from cograph_client.normalization.rules import NormalizationRule, make_rule_id
from cograph_client.resolver.llm_router import PRIMARY_MODEL, openrouter_chat

logger = structlog.stdlib.get_logger("cograph.normalization.inference")

ENTITY_URI_PREFIX = "https://cograph.tech/entities/"
ATTRS_INFIX = "/attrs/"
ONTO_PRED_PREFIX = "https://cograph.tech/onto/"

# How many independent draws to pool, and how many distinct values per draw.
_NUM_SAMPLES = 3
_VALUES_PER_SAMPLE = 20

SUGGEST_SYSTEM = """\
You are a data-normalization analyst for a knowledge graph. You are shown the \
distinct VALUES a single predicate takes across many entities of one type, and \
you must decide whether those values need NORMALIZATION before they are useful.

You can recommend ZERO, ONE, or MULTIPLE of these normalization types for the \
SAME predicate (a value can have more than one problem at once):

1) list_explode — a multi-valued source cell that was collapsed into ONE \
composite value instead of split into N atomic values. Tell-tale signs:
   - a literal packing several items with a delimiter: "English, Russian, \
     Ukrainian", "Python; SQL; Go", "Sales / Marketing", "a|b|c";
   - an entity whose name/local-name packs several items with the slugified \
     delimiter "__" (a list separator turned into "__" at ingest), e.g. \
     "English__Russian__Ukrainian", "Sales__Marketing".
   params: {"delimiters": ["<each delimiter you observed>"], \
"target": "entity"|"literal"} — use "entity" when the values are entity \
names/local-names (the predicate is a relationship to other entities), \
"literal" when they are plain attribute literals.

2) strip_emoji — text values carry emoji, pictographs, or other non-text JUNK \
characters that should be removed, leaving the real text: "🎨 design", \
"ai 🚀", "growth ✨", "🔥🔥 sales". Recommend this whenever you see emoji / \
pictographic / symbol junk mixed into otherwise-text values. Do NOT recommend \
it for ordinary punctuation that belongs to real values (e.g. "c++", "C#", \
"Node.js", "R&D", accented letters like "café"). \
   params: {"targets": ["attribute"]}

A predicate can need BOTH at once — e.g. skills = "🎨 design; ai; 🚀 growth" \
needs list_explode (split on "; ") AND strip_emoji (remove 🎨 and 🚀). In that \
case return BOTH rules.

CRITICAL — do NOT false-split single multi-word values. Many legitimate single \
values contain spaces or punctuation and must be left intact: "Bahasa Indonesia", \
"Mandarin Chinese", "Standard Arabic", "Hong Kong", "New York", "Saint Kitts and \
Nevis", "Trinidad and Tobago". A space is NOT a delimiter. Only treat a value as \
a packed list when a clear list-delimiter (comma, semicolon, pipe, slash, or the \
slug "__") separates items that are each individually plausible standalone values.

If the values are already atomic AND emoji-free, return an empty "rules" list. \
If you see a normalization problem that is NEITHER list_explode NOR strip_emoji \
(casing, trimming, units, value mapping), do NOT invent a rule for it — leave \
"rules" empty and explain the observation in a rule's rationale only if you are \
also returning a supported rule.

Respond with STRICT JSON only, no markdown:
{
  "rules": [
    {
      "rule_type": "list_explode"|"strip_emoji",
      "params": { ...rule-type-specific params (see above)... },
      "confidence": 0.0,
      "rationale": "one or two sentences"
    }
  ]
}
Return an empty list ({"rules": []}) when no normalization is needed. Set each \
confidence in [0,1] reflecting how sure you are that problem is present."""

SUGGEST_USER_TEMPLATE = """\
Type: {type_name}
Predicate: {predicate}   (kind: {target_kind})

Distinct sample values for this predicate (pooled from several independent draws):
{values}

Which normalization(s) does this predicate need (list_explode, strip_emoji, both, \
or none)? Respond with strict JSON ({{"rules": [...]}})."""


async def suggest_rules(
    neptune: NeptuneClient,
    tenant_id: str,
    kg_name: str,
    type_name: str,
) -> list[NormalizationRule]:
    """Infer normalization rules for every predicate of ``type_name`` in ``kg_name``.

    Returns ``suggested`` rules ranked by confidence (desc). Does NOT persist.
    """
    predicates = await _list_predicates(neptune, tenant_id, type_name)
    if not predicates:
        logger.info("no_predicates", tenant=tenant_id, kg=kg_name, type=type_name)
        return []

    api_key = _openrouter_key()
    kg_graph = kg_graph_uri(tenant_id, kg_name)
    t_uri = type_uri(type_name)

    rules: list[NormalizationRule] = []
    for pred_uri, target_kind in predicates:
        samples = await _sample_values(
            neptune, kg_graph, t_uri, pred_uri, target_kind
        )
        if not samples:
            continue
        recommendations = await _ask_llm(
            api_key, type_name, pred_uri, target_kind, samples
        )
        if not recommendations:
            continue
        pred_leaf = _predicate_leaf(pred_uri)
        # Dedupe by rule_type within a predicate (one rule per (predicate,
        # rule_type)); first recommendation of a type wins.
        seen_types: set[str] = set()
        for rec in recommendations:
            rule = _rule_from_recommendation(
                rec, kg_name, type_name, pred_leaf, target_kind, samples
            )
            if rule is None or rule.rule_type in seen_types:
                continue
            seen_types.add(rule.rule_type)
            rules.append(rule)

    # Rank ALL emitted rules (across predicates and rule types) by confidence.
    rules.sort(key=lambda r: r.confidence, reverse=True)
    return rules


async def suggest_rules_for_predicates(
    neptune: NeptuneClient,
    tenant_id: str,
    kg_name: str,
    type_name: str,
    predicate_leaves: list[str],
) -> list[NormalizationRule]:
    """Infer rules for ONLY the named predicate(s) of ``type_name``.

    A TARGETED variant of :func:`suggest_rules`: the unified agent must NOT scan
    every predicate of a type (that is the slow whole-type path that timed out
    before — COG-118). The agent names the predicate(s) the instruction refers to
    (e.g. "speaks") and we sample + ask the LLM for just those, reusing the exact
    same per-predicate machinery (``_sample_values`` / ``_ask_llm`` /
    ``_rule_from_recommendation``) so there is no logic duplication.

    Matching is by leaf name, case-insensitively, against the type's declared
    predicates. An unknown leaf is simply skipped (no rule), never an unbounded
    scan. Returns ``suggested`` rules ranked by confidence (desc); persists
    nothing (the caller decides).
    """
    wanted = {p.strip().lower() for p in predicate_leaves if p and p.strip()}
    if not wanted:
        return []
    declared = await _list_predicates(neptune, tenant_id, type_name)
    if not declared:
        return []
    api_key = _openrouter_key()
    kg_graph = kg_graph_uri(tenant_id, kg_name)
    t_uri = type_uri(type_name)

    rules: list[NormalizationRule] = []
    for pred_uri, target_kind in declared:
        pred_leaf = _predicate_leaf(pred_uri)
        if pred_leaf.lower() not in wanted:
            continue
        samples = await _sample_values(
            neptune, kg_graph, t_uri, pred_uri, target_kind
        )
        if not samples:
            continue
        recommendations = await _ask_llm(
            api_key, type_name, pred_uri, target_kind, samples
        )
        if not recommendations:
            continue
        seen_types: set[str] = set()
        for rec in recommendations:
            rule = _rule_from_recommendation(
                rec, kg_name, type_name, pred_leaf, target_kind, samples
            )
            if rule is None or rule.rule_type in seen_types:
                continue
            seen_types.add(rule.rule_type)
            rules.append(rule)

    rules.sort(key=lambda r: r.confidence, reverse=True)
    return rules


async def sample_predicate_values(
    neptune: NeptuneClient,
    tenant_id: str,
    kg_name: str,
    type_name: str,
    predicate_leaf: str,
) -> tuple[list[str], str]:
    """Sample current distinct values for ONE predicate (for cheap composite
    detection + dry-run previews). Returns ``(samples, target_kind)``.

    Reuses :func:`_list_predicates` + :func:`_sample_values` so the sampling
    query shape stays identical to inference. ``target_kind`` is "attribute" if
    the predicate is unknown/declared as an attribute, "relationship" otherwise.
    A bounded, predicate-targeted read — never a whole-type scan.
    """
    leaf = predicate_leaf.strip().lower()
    if not leaf:
        return [], "attribute"
    declared = await _list_predicates(neptune, tenant_id, type_name)
    kg_graph = kg_graph_uri(tenant_id, kg_name)
    t_uri = type_uri(type_name)
    for pred_uri, target_kind in declared:
        if _predicate_leaf(pred_uri).lower() == leaf:
            samples = await _sample_values(
                neptune, kg_graph, t_uri, pred_uri, target_kind
            )
            return samples, target_kind
    return [], "attribute"


_SUPPORTED_RULE_TYPES = {"list_explode", "strip_emoji"}
_DEFAULT_DELIMITERS = [", ", "; ", " / ", " | ", "__"]


def _rule_from_recommendation(
    rec: dict,
    kg_name: str,
    type_name: str,
    pred_leaf: str,
    target_kind: str,
    samples: list[str],
) -> NormalizationRule | None:
    """Build one :class:`NormalizationRule` from one LLM recommendation, or None.

    Unsupported / malformed rule types are logged and skipped so a stray entry
    can't poison the whole predicate. The id folds in ``rule_type`` so
    list_explode and strip_emoji on the SAME predicate get distinct ids.
    """
    if not isinstance(rec, dict):
        return None
    rule_type = rec.get("rule_type")
    if rule_type not in _SUPPORTED_RULE_TYPES:
        logger.info(
            "unsupported_rule_type_skipped",
            predicate=pred_leaf,
            rule_type=rule_type,
            rationale=rec.get("rationale", ""),
        )
        return None
    params = dict(rec.get("params") or {})
    if rule_type == "list_explode":
        # Default the target from the predicate kind if the LLM omitted it.
        params.setdefault(
            "target", "entity" if target_kind == "relationship" else "literal"
        )
        if not params.get("delimiters"):
            params["delimiters"] = _DEFAULT_DELIMITERS
    elif rule_type == "strip_emoji":
        # Cleaning attribute literals is the default (the skills case).
        if not params.get("targets"):
            params["targets"] = ["attribute"]
    return NormalizationRule(
        id=make_rule_id(kg_name, type_name, pred_leaf, rule_type),
        kg_name=kg_name,
        type_name=type_name,
        predicate=pred_leaf,
        target_kind=target_kind,
        rule_type=rule_type,
        params=params,
        confidence=float(rec.get("confidence", 0.0) or 0.0),
        rationale=rec.get("rationale", ""),
        sample_values=samples[:25],
        status="suggested",
    )


def _openrouter_key() -> str:
    from cograph_client.config import settings

    return settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")


def _predicate_leaf(pred_uri: str) -> str:
    """Recover the predicate leaf name from either an attr URI or an onto URI."""
    if ATTRS_INFIX in pred_uri:
        return pred_uri.split(ATTRS_INFIX, 1)[1]
    if pred_uri.startswith(ONTO_PRED_PREFIX):
        return pred_uri[len(ONTO_PRED_PREFIX):]
    return pred_uri.rstrip("/").split("/")[-1]


async def _list_predicates(
    neptune: NeptuneClient, tenant_id: str, type_name: str
) -> list[tuple[str, str]]:
    """List a type's declared predicates from the ontology.

    Returns ``[(predicate_uri, target_kind)]``. ``target_kind`` is
    ``"relationship"`` when the declared ``rdfs:range`` is a ``types/`` URI,
    else ``"attribute"``. Uses the same attr-definition query shape the Explorer
    uses (``rdfs:domain`` = the type, with an optional ``rdfs:range``).
    """
    from cograph_client.graph.queries import tenant_graph_uri

    onto_graph = tenant_graph_uri(tenant_id)
    t_uri = type_uri(type_name)
    q = (
        f"SELECT ?attr ?range FROM <{onto_graph}> WHERE {{\n"
        f"  ?attr <{RDF}#type> <{RDF}#Property> .\n"
        f"  ?attr <{RDFS}#domain> <{t_uri}> .\n"
        f"  OPTIONAL {{ ?attr <{RDFS}#range> ?range }}\n"
        f"}}"
    )
    _, rows = parse_sparql_results(await neptune.query(q))
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for r in rows:
        attr = r.get("attr", "")
        if not attr or attr in seen:
            continue
        seen.add(attr)
        rng = r.get("range", "")
        kind = "relationship" if rng.startswith("https://cograph.tech/types/") else "attribute"
        out.append((attr, kind))
    return out


async def _sample_values(
    neptune: NeptuneClient,
    kg_graph: str,
    t_uri: str,
    pred_uri: str,
    target_kind: str,
) -> list[str]:
    """Pool 2-3 INDEPENDENT draws of distinct values for one predicate.

    Each draw varies ORDER BY + OFFSET so the pooled set is representative rather
    than the same head every time. For a relationship we sample the target
    entity's human value (its ``rdfs:label``, falling back to the entity-URI
    leaf); for an attribute we sample the literal object. The predicate is the
    ONTOLOGY attr URI (``…/attrs/<leaf>``); instance triples use either that attr
    URI (attributes) or the ``…/onto/<leaf>`` predicate (relationships), so the
    instance pattern matches on the LEAF via either form.
    """
    pred_leaf = _predicate_leaf(pred_uri)
    onto_pred = ONTO_PRED_PREFIX + pred_leaf
    rdfs_label = f"{RDFS}#label"

    pooled: list[str] = []
    seen: set[str] = set()
    orderings = ["ASC(?v)", "DESC(?v)", "ASC(?o)"]
    for i in range(_NUM_SAMPLES):
        offset = i * _VALUES_PER_SAMPLE
        order = orderings[i % len(orderings)]
        if target_kind == "relationship":
            # ?o is the target entity; ?v is its human label (or URI leaf).
            value_bind = (
                f"  OPTIONAL {{ ?o <{rdfs_label}> ?lbl }}\n"
                f'  BIND(COALESCE(?lbl, REPLACE(STR(?o), "^.*/", "")) AS ?v)\n'
            )
            obj_pattern = (
                f"  {{ ?e <{pred_uri}> ?o }} UNION {{ ?e <{onto_pred}> ?o }}\n"
            )
        else:
            value_bind = "  BIND(STR(?o) AS ?v)\n"
            obj_pattern = (
                f"  {{ ?e <{pred_uri}> ?o }} UNION {{ ?e <{onto_pred}> ?o }}\n"
            )
        q = (
            f"SELECT DISTINCT ?v FROM <{kg_graph}> WHERE {{\n"
            f"  ?e <{RDF}#type> <{t_uri}> .\n"
            f"{obj_pattern}"
            f"{value_bind}"
            f"}} ORDER BY {order} LIMIT {_VALUES_PER_SAMPLE} OFFSET {offset}"
        )
        try:
            _, rows = parse_sparql_results(await neptune.query(q))
        except Exception:
            logger.warning("sample_query_failed", predicate=pred_uri, exc_info=True)
            continue
        for r in rows:
            v = (r.get("v") or "").strip()
            if v and v not in seen:
                seen.add(v)
                pooled.append(v)
    return pooled


async def _ask_llm(
    api_key: str,
    type_name: str,
    pred_uri: str,
    target_kind: str,
    samples: list[str],
) -> list[dict]:
    """One LLM call per predicate. Returns a list of recommendation dicts.

    Empty list = no normalization needed (or an error / no key) — the caller
    treats every non-recommendation the same way.
    """
    if not api_key:
        logger.warning("no_openrouter_key_for_inference")
        return []
    values_block = "\n".join(f"- {v}" for v in samples)
    user = SUGGEST_USER_TEMPLATE.format(
        type_name=type_name,
        predicate=_predicate_leaf(pred_uri),
        target_kind=target_kind,
        values=values_block,
    )
    try:
        text = await openrouter_chat(
            api_key,
            SUGGEST_SYSTEM,
            user,
            model=PRIMARY_MODEL,
            temperature=0,
            max_tokens=600,
            timeout=60,
        )
    except Exception:
        logger.warning("inference_llm_failed", predicate=pred_uri, exc_info=True)
        return []
    return _parse_recommendations(text)


def _parse_recommendations(text: str) -> list[dict]:
    """Parse the LLM reply into a list of recommendation dicts.

    Accepts the current ``{"rules": [...]}`` shape; also tolerates a bare list
    ``[...]`` and the legacy single-verdict shape
    (``{"needs_normalization": ..., "rule_type": ...}``) so a stale response or
    a partial upgrade degrades gracefully instead of dropping the predicate.
    """
    data = _parse_json_object(text)
    if data is None:
        return []
    # New shape: {"rules": [ {...}, ... ]}.
    if isinstance(data, dict) and isinstance(data.get("rules"), list):
        return [r for r in data["rules"] if isinstance(r, dict)]
    # Legacy single-verdict shape — promote it to a one-element list if it
    # actually asked for a rule.
    if isinstance(data, dict) and "needs_normalization" in data:
        if data.get("needs_normalization") and data.get("rule_type"):
            return [
                {
                    "rule_type": data.get("rule_type"),
                    "params": data.get("params") or {},
                    "confidence": data.get("confidence", 0.0),
                    "rationale": data.get("rationale", ""),
                }
            ]
        return []
    return []


def _parse_json_object(text: str):
    """Best-effort JSON parse, tolerant of code fences and a prose wrapper."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = [l for l in stripped.split("\n") if not l.strip().startswith("```")]
        stripped = "\n".join(lines)
    # Grab the outermost {...} (object) — the response is always an object today.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        stripped = stripped[start : end + 1]
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        logger.warning("verdict_parse_failed", raw=text[:300])
        return None
