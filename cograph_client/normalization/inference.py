"""Inference agent — propose normalization rules for a type's predicates.

:func:`suggest_rules` enumerates a type's attributes and relationships from the
ontology, draws a few INDEPENDENT samples of distinct values per predicate from
the KG graph, and asks an LLM (once per predicate) whether that predicate needs
normalization and of which kind. v1 only needs to detect ``list_explode`` —
multi-valued source cells that were collapsed into one composite value (a
delimited literal, or a composite entity whose local-name/label packs several
atomic values joined by the slugified delimiter ``__``).

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

v1 detects exactly ONE problem: list_explode — a multi-valued source cell that \
was collapsed into ONE composite value instead of split into N atomic values. \
Tell-tale signs:
- a literal packing several items with a delimiter: "English, Russian, Ukrainian", \
  "Python; SQL; Go", "Sales / Marketing", "a|b|c";
- an entity whose name/local-name packs several items with the slugified \
  delimiter "__" (a list separator that was turned into "__" at ingest), e.g. \
  "English__Russian__Ukrainian", "Sales__Marketing".

CRITICAL — do NOT false-split single multi-word values. Many legitimate single \
values contain spaces or punctuation and must be left intact: "Bahasa Indonesia", \
"Mandarin Chinese", "Standard Arabic", "Hong Kong", "New York", "Saint Kitts and \
Nevis", "Trinidad and Tobago". A space is NOT a delimiter. Only treat a value as \
a packed list when a clear list-delimiter (comma, semicolon, pipe, slash, or the \
slug "__") separates items that are each individually plausible standalone values.

If the values are already atomic (each value is a single item), return \
needs_normalization=false. If you see a different normalization problem that is \
NOT list_explode (casing, trimming, units, value mapping), return \
needs_normalization=false with rule_type=null and explain it in the rationale — \
v1 will not act on it but the note is recorded.

Respond with STRICT JSON only, no markdown:
{
  "needs_normalization": true|false,
  "rule_type": "list_explode"|null,
  "params": {"delimiters": ["<each delimiter you observed>"], "target": "entity"|"literal"},
  "confidence": 0.0,
  "rationale": "one or two sentences"
}
For target: use "entity" when the values are entity names/local-names (the \
predicate is a relationship to other entities), "literal" when they are plain \
attribute literals. Set confidence in [0,1] reflecting how sure you are the \
values are a packed delimited list."""

SUGGEST_USER_TEMPLATE = """\
Type: {type_name}
Predicate: {predicate}   (kind: {target_kind})

Distinct sample values for this predicate (pooled from several independent draws):
{values}

Does this predicate need list_explode normalization? Respond with strict JSON."""


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
        verdict = await _ask_llm(api_key, type_name, pred_uri, target_kind, samples)
        if not verdict or not verdict.get("needs_normalization"):
            continue
        rule_type = verdict.get("rule_type")
        if rule_type != "list_explode":
            # v1 only acts on list_explode; a non-null other type isn't emitted.
            logger.info(
                "non_list_explode_skipped",
                predicate=pred_uri,
                rule_type=rule_type,
                rationale=verdict.get("rationale", ""),
            )
            continue
        pred_leaf = _predicate_leaf(pred_uri)
        params = verdict.get("params") or {}
        # Default the target from the predicate kind if the LLM omitted it.
        params.setdefault(
            "target", "entity" if target_kind == "relationship" else "literal"
        )
        if not params.get("delimiters"):
            params["delimiters"] = _DEFAULT_DELIMITERS
        rules.append(
            NormalizationRule(
                id=make_rule_id(kg_name, type_name, pred_leaf),
                kg_name=kg_name,
                type_name=type_name,
                predicate=pred_leaf,
                target_kind=target_kind,
                rule_type="list_explode",
                params=params,
                confidence=float(verdict.get("confidence", 0.0) or 0.0),
                rationale=verdict.get("rationale", ""),
                sample_values=samples[:25],
                status="suggested",
            )
        )

    rules.sort(key=lambda r: r.confidence, reverse=True)
    return rules


_DEFAULT_DELIMITERS = [", ", "; ", " / ", " | ", "__"]


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
) -> dict | None:
    """One LLM call per predicate. Returns parsed verdict dict, or None on error."""
    if not api_key:
        logger.warning("no_openrouter_key_for_inference")
        return None
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
        return None
    return _parse_verdict(text)


def _parse_verdict(text: str) -> dict | None:
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        lines = [l for l in stripped.split("\n") if not l.strip().startswith("```")]
        stripped = "\n".join(lines)
    # Be tolerant of a leading/trailing prose wrapper: grab the outermost {...}.
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        stripped = stripped[start : end + 1]
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        logger.warning("verdict_parse_failed", raw=text[:300])
        return None
    return data if isinstance(data, dict) else None
