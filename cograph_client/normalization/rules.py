"""Normalization rules — model + tenant-ontology-graph store.

A :class:`NormalizationRule` is an INFERRED, human-confirmed instruction for
fixing a systematic data-shape problem on one predicate of one type in one KG.
Two ``rule_type``\\ s ship today:

* ``"list_explode"`` — multi-valued source cells that got collapsed into one
  composite value instead of split into N atomic ones (e.g.
  ``speaks -> Language/English__Russian__Ukrainian`` becoming three edges to the
  atomic ``English`` / ``Russian`` / ``Ukrainian`` entities).
* ``"strip_emoji"`` — text literals carrying emoji / pictographic junk
  characters (e.g. ``skills = "🎨 design"`` → ``"design"``); the junk is removed
  and the leftover whitespace collapsed, in place.

Rules live as ordinary triples in the **tenant ontology graph**
(:func:`tenant_graph_uri`) — one ``…/entities/NormalizationRule/<id>`` resource
with an ``rdf:type`` plus one predicate per field. ``params`` / ``sample_values``
are JSON-encoded into a single literal so the schema can grow new
``rule_type``\\ s (``trim``, ``case``, ``value_map``, ``unit_canonical``, …)
without any store change. All writes go through the shared escaping helpers, so
the store is SPARQL-injection-safe like the rest of the codebase.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.kg_writer import delete_facts, insert_facts
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import (
    _escape_literal,
    tenant_graph_uri,
)

# Namespaces. The rule resource is an entity (so it shows up under the same
# `…/entities/<Type>/<id>` shape every other resource uses); its fields hang off
# a dedicated `…/onto/norm/<field>` predicate namespace so they never collide
# with real ontology predicates.
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RULE_TYPE_URI = "https://cograph.tech/types/NormalizationRule"
RULE_ENTITY_PREFIX = "https://cograph.tech/entities/NormalizationRule/"
NORM_NS = "https://cograph.tech/onto/norm/"

# One predicate per scalar field. params / sample_values are JSON blobs.
P_KG = NORM_NS + "kgName"
P_TYPE = NORM_NS + "typeName"
P_PREDICATE = NORM_NS + "predicate"
P_TARGET_KIND = NORM_NS + "targetKind"
P_RULE_TYPE = NORM_NS + "ruleType"
P_PARAMS = NORM_NS + "params"
P_CONFIDENCE = NORM_NS + "confidence"
P_RATIONALE = NORM_NS + "rationale"
P_SAMPLE_VALUES = NORM_NS + "sampleValues"
P_STATUS = NORM_NS + "status"
P_CREATED_AT = NORM_NS + "createdAt"
P_APPLIED_AT = NORM_NS + "appliedAt"

RuleStatus = Literal["suggested", "confirmed", "rejected", "applied"]
TargetKind = Literal["attribute", "relationship"]


class NormalizationRule(BaseModel):
    """One inferred normalization for a (kg, type, predicate).

    ``rule_type`` is an open string (not an Enum) on purpose: ``params`` carries
    every rule-type-specific knob, so new rule types can be added with zero
    store-schema change. Two rule types ship today:

    * ``"list_explode"`` — params ``{"delimiters": [", ", ";", " / "],
      "target": "entity"|"literal"}``.
    * ``"strip_emoji"`` — params ``{"targets": ["attribute"]}`` (which kinds of
      object to clean; attribute literals by default). Removes emoji /
      pictographic junk from text values.

    A single predicate can warrant BOTH (e.g. ``skills`` may need
    ``list_explode`` AND ``strip_emoji``); :func:`make_rule_id` folds
    ``rule_type`` into the id so they don't collide in the store.
    """

    id: str
    kg_name: str
    type_name: str
    predicate: str
    target_kind: TargetKind
    rule_type: str = "list_explode"
    params: dict = Field(default_factory=dict)
    confidence: float = 0.0
    rationale: str = ""
    sample_values: list[str] = Field(default_factory=list)
    status: RuleStatus = "suggested"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    applied_at: Optional[str] = None

    @property
    def uri(self) -> str:
        return RULE_ENTITY_PREFIX + self.id


def make_rule_id(
    kg_name: str, type_name: str, predicate: str, rule_type: str = "list_explode"
) -> str:
    """Deterministic id for a (kg, type, predicate, rule_type) so re-suggesting is idempotent.

    The id incorporates ``rule_type`` so two *different* normalizations on the
    SAME predicate (e.g. ``list_explode`` AND ``strip_emoji`` on ``skills``) get
    DISTINCT ids and don't clobber each other in the store. ``list_explode``
    keeps its historical id shape (``<kg>__<type>__<pred>``) so previously-stored
    rules and the existing callers stay byte-compatible; every other rule_type
    gets a ``__<rule_type>`` suffix.

    Sanitized to URI-safe chars so it slots straight into the entity IRI without
    any further escaping (it is only ever spliced into ``<…>`` term positions).
    """
    import re

    raw = f"{kg_name}__{type_name}__{predicate}"
    if rule_type and rule_type != "list_explode":
        raw = f"{raw}__{rule_type}"
    return re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:200] or "rule"


class NormalizationRuleStore:
    """Persist + read :class:`NormalizationRule`\\ s in the tenant ontology graph.

    All methods are async (one Neptune round-trip each). :meth:`save` is
    idempotent — it DELETEs any prior triples for the rule's id, then INSERTs the
    current field set, so re-saving an updated rule never leaves stale field
    triples behind (matching the upsert discipline elsewhere in the codebase).
    """

    def __init__(self, neptune: NeptuneClient):
        self._neptune = neptune

    async def save(self, tenant_id: str, rule: NormalizationRule) -> None:
        graph = tenant_graph_uri(tenant_id)
        # Clear-then-write through the shared write path (ADR 0007): the rule is a
        # metadata subject in the tenant ontology graph, so delete_facts drops its
        # prior field triples (subject-scoped) and insert_facts writes the current
        # set. No refresh_after_write — a rule row is config metadata (never
        # instance data / geometry / schema), so there is no derived-index,
        # ontology-cache, or type-stats fan-out to run for it.
        await delete_facts(
            self._neptune, graph, subjects=[rule.uri], reason="normalization-rule upsert"
        )
        await insert_facts(self._neptune, graph, self._rule_to_triples(rule))

    async def get(self, tenant_id: str, rule_id: str) -> Optional[NormalizationRule]:
        graph = tenant_graph_uri(tenant_id)
        uri = RULE_ENTITY_PREFIX + rule_id
        q = (
            f"SELECT ?p ?o FROM <{graph}> WHERE {{\n"
            f"  <{uri}> ?p ?o .\n"
            f"}}"
        )
        _, rows = parse_sparql_results(await self._neptune.query(q))
        if not rows:
            return None
        fields = {r["p"]: r["o"] for r in rows if "p" in r and "o" in r}
        return self._rule_from_fields(rule_id, fields)

    async def list(
        self,
        tenant_id: str,
        kg: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list[NormalizationRule]:
        """List rules, optionally filtered by KG name and/or status.

        Filters are applied as escaped string-literal equality in SPARQL (never
        spliced into an IRI), so they are injection-safe.
        """
        graph = tenant_graph_uri(tenant_id)
        filters = ""
        if kg is not None:
            filters += f'  ?s <{P_KG}> "{_escape_literal(kg)}" .\n'
        if status is not None:
            filters += f'  ?s <{P_STATUS}> "{_escape_literal(status)}" .\n'
        q = (
            f"SELECT ?s ?p ?o FROM <{graph}> WHERE {{\n"
            f"  ?s <{RDF_TYPE}> <{RULE_TYPE_URI}> .\n"
            f"{filters}"
            f"  ?s ?p ?o .\n"
            f"}}"
        )
        _, rows = parse_sparql_results(await self._neptune.query(q))
        by_subject: dict[str, dict[str, str]] = {}
        for r in rows:
            s, p, o = r.get("s"), r.get("p"), r.get("o")
            if not s or not p:
                continue
            by_subject.setdefault(s, {})[p] = o
        out: list[NormalizationRule] = []
        for s, fields in by_subject.items():
            rule_id = s[len(RULE_ENTITY_PREFIX):] if s.startswith(RULE_ENTITY_PREFIX) else s
            rule = self._rule_from_fields(rule_id, fields)
            if rule is not None:
                out.append(rule)
        # Stable, useful default order: highest confidence first.
        out.sort(key=lambda r: r.confidence, reverse=True)
        return out

    async def update_status(
        self,
        tenant_id: str,
        rule_id: str,
        status: RuleStatus,
        applied_at: Optional[str] = None,
    ) -> Optional[NormalizationRule]:
        """Set a rule's status (and optionally applied_at). Idempotent.

        Returns the updated rule, or None if it doesn't exist.
        """
        rule = await self.get(tenant_id, rule_id)
        if rule is None:
            return None
        rule.status = status
        if applied_at is not None:
            rule.applied_at = applied_at
        await self.save(tenant_id, rule)
        return rule

    # --- serialization -------------------------------------------------------

    @staticmethod
    def _rule_to_triples(rule: NormalizationRule) -> list[tuple[str, str, str]]:
        uri = rule.uri
        triples: list[tuple[str, str, str]] = [
            (uri, RDF_TYPE, RULE_TYPE_URI),
            (uri, P_KG, rule.kg_name),
            (uri, P_TYPE, rule.type_name),
            (uri, P_PREDICATE, rule.predicate),
            (uri, P_TARGET_KIND, rule.target_kind),
            (uri, P_RULE_TYPE, rule.rule_type),
            (uri, P_PARAMS, json.dumps(rule.params)),
            # Confidence as a typed literal so a downstream query can sort
            # numerically. Full XSD URI (not the `xsd:` prefix) — `_escape_value`
            # emits `"<v>"^^<xsd:decimal>` verbatim, and a bare prefix would be a
            # relative IRI Neptune rejects.
            (uri, P_CONFIDENCE, f"{rule.confidence}^^http://www.w3.org/2001/XMLSchema#decimal"),
            (uri, P_RATIONALE, rule.rationale),
            (uri, P_SAMPLE_VALUES, json.dumps(rule.sample_values)),
            (uri, P_STATUS, rule.status),
            (uri, P_CREATED_AT, rule.created_at),
        ]
        if rule.applied_at:
            triples.append((uri, P_APPLIED_AT, rule.applied_at))
        return triples

    @staticmethod
    def _rule_from_fields(
        rule_id: str, fields: dict[str, str]
    ) -> Optional[NormalizationRule]:
        if fields.get(RDF_TYPE) != RULE_TYPE_URI:
            return None

        def _json(raw: str, default):
            if not raw:
                return default
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return default

        try:
            confidence = float(fields.get(P_CONFIDENCE, "0") or "0")
        except (TypeError, ValueError):
            confidence = 0.0

        return NormalizationRule(
            id=rule_id,
            kg_name=fields.get(P_KG, ""),
            type_name=fields.get(P_TYPE, ""),
            predicate=fields.get(P_PREDICATE, ""),
            target_kind=fields.get(P_TARGET_KIND, "attribute"),  # type: ignore[arg-type]
            rule_type=fields.get(P_RULE_TYPE, "list_explode"),
            params=_json(fields.get(P_PARAMS, ""), {}),
            confidence=confidence,
            rationale=fields.get(P_RATIONALE, ""),
            sample_values=_json(fields.get(P_SAMPLE_VALUES, ""), []),
            status=fields.get(P_STATUS, "suggested"),  # type: ignore[arg-type]
            created_at=fields.get(P_CREATED_AT, ""),
            applied_at=fields.get(P_APPLIED_AT) or None,
        )
