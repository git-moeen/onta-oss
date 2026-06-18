"""Ontology evolution resolver (COG-84).

Turns a fuzzy natural-language *ask* — e.g. "track which company a person works
for" — plus a graph scope into a structured PLAN of ontology changes: which to
auto-APPLY (high confidence, no new type) versus return as PROPOSALS (a new type
must be created, or the match is ambiguous).

The service is **plan-only**: it never writes to Neptune. A later REST layer
(COG-81) consumes the :class:`ResolutionResult` and applies it.

Pipeline (see ``resolve``):
  1. INTENT PARSE — one LLM call decomposes the ask into change-intents
     ``{subject_phrase, kind, name_phrase, target_phrase?, datatype_hint?}``.
  2. SUBJECT TYPE — embedding retrieve narrows candidates, then ``TypeMatcher``
     judges SAME / SUBTYPE / DIFFERENT against the existing ontology.
  3. NAME — ``resolve_attribute`` (attributes) or ``normalize_predicate``
     (relationships) against that subject type's existing schema → reuse/extend.
  4. TARGET TYPE — a relationship's target type is resolved like the subject; a
     new target type is a creation.
  5. CONFIDENCE GATE — APPLY when the subject is an existing SAME match AND the
     attr/rel is a clean reuse/extend; otherwise PROPOSE.

All reused primitives are OSS (``TypeMatcher``, ``OntologyEmbeddingService``,
``resolve_attribute``, ``normalize_predicate``). No proprietary imports.
"""

from __future__ import annotations

import json
import os

import anthropic
import structlog

from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.ontology_queries import get_full_ontology_query
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.models.ontology import ResolutionResult, ResolvedChange
from cograph_client.resolver.attribute_resolver import AttributeSchema, resolve_attribute
from cograph_client.resolver.llm_router import PRIMARY_MODEL, openrouter_chat
from cograph_client.resolver.models import (
    AttrAction,
    ExtractedAttribute,
    MatchVerdict,
    TypeMatch,
)
from cograph_client.resolver.predicate_normalizer import normalize_predicate
from cograph_client.resolver.type_matcher import TypeMatcher

logger = structlog.stdlib.get_logger("cograph.resolver.ontology")

TYPE_URI_PREFIX = "https://cograph.tech/types/"

# Primitive datatypes a relationship is NOT: anything else as a datatype_hint
# (or a populated target_phrase) marks the intent as a relationship to a type.
PRIMITIVE_DATATYPES = {"string", "integer", "float", "boolean", "datetime", "uri"}

# Confidence band below which an otherwise-clean SAME match is still routed to
# PROPOSE (the embedding/TypeMatcher signal is mid-band/ambiguous). TypeMatcher
# already auto-news anything below ~0.40, so this is the "matched but unsure"
# floor for auto-apply.
APPLY_CONFIDENCE_FLOOR = 0.70


# Intent-parse LLM contract. ONE call per ask; mirrors the CSVResolver
# provider/model config (OMNIX_EXTRACT_MODEL via OpenRouter, Anthropic offline
# fallback). Output is a flat list of change-intents.
INTENT_SYSTEM_PROMPT = """\
You decompose a natural-language request to evolve a knowledge-graph ontology
into a flat list of concrete CHANGE INTENTS. Each intent adds ONE attribute or
ONE relationship to ONE subject type.

For each intent return:
- "subject_phrase": the entity/type the change is ABOUT (e.g. "person", "company").
- "kind": "attribute" for an intrinsic literal of the subject (a date, number,
  price, measurement, flag, code, or free text), "relationship" when the value IS
  another entity/type the subject links to.
- "name_phrase": a short name for the property or the relationship verb
  (e.g. "birth date", "works for", "headquartered in").
- "target_phrase": for a relationship ONLY — the type the relationship points at
  (e.g. "company", "city"). Omit/empty for attributes.
- "datatype_hint": for an attribute ONLY — one of string, integer, float,
  boolean, datetime, uri. Omit for relationships.

Rules — PREFER ENTITIES OVER LABELS (reification):
- If the value NAMES a real-world thing that has its own identity — a place
  (neighborhood, city, region, zip code), an organization, a category, a person —
  make it a "relationship" and set target_phrase to that thing. This holds EVEN
  when the ask says "name of …": the name lives ON that entity, so link to the
  entity instead of copying its label onto the subject. (e.g. "the neighborhood
  name each property is in" → relationship, target_phrase "neighborhood" — NOT a
  "neighborhood_name" string.)
- Especially prefer a relationship when an existing type below already represents
  the thing (or something it clearly connects to) — reuse the ontology backbone.
- Use "attribute" (with datatype_hint) ONLY for intrinsic literals of the subject
  itself: dates, counts, prices, measurements, flags, codes, free text, and the
  subject's OWN proper name.
- One ask may imply several intents — return all of them.

Respond with valid JSON only, no markdown:
{"intents": [{"subject_phrase": "...", "kind": "attribute"|"relationship",
"name_phrase": "...", "target_phrase": "...", "datatype_hint": "..."}]}"""

INTENT_USER_TEMPLATE = """\
Existing ontology types (reuse these names when the subject/target matches one):
{existing_types}

Ask: "{ask}"

Decompose the ask into change intents and return the JSON now."""


# When the resolver introduces a NEW entity type, this second (optional) LLM call
# wires it into the EXISTING ontology backbone — the structural relationships the
# new thing should have to types that already exist (a new Neighborhood belongs to
# an existing City and ZipCode). Keeps expansion clean: new entities JOIN the
# graph (and reach the rest of it transitively) instead of dangling.
BACKBONE_SYSTEM_PROMPT = """\
A knowledge-graph ontology is gaining one or more NEW entity types. Connect each
new type into the EXISTING ontology so the graph stays well-linked.

For each new type, propose the obvious STRUCTURAL real-world relationships it
should have TO TYPES THAT ALREADY EXIST — especially containment / hierarchy (a
neighborhood is in a city and belongs to a zip code; a department is in a
company). This is what lets other entities reach the new one transitively (a
listing already links to a zip code, so linking neighborhood→zip code ties every
listing to its neighborhood).

Hard rules:
- target_type MUST be one of the EXISTING types listed, verbatim. Never target
  another new type and never invent a type.
- Prefer 1–3 high-confidence links per new type. Omit a new type entirely if
  nothing obvious connects it.
- "predicate" is a short relationship verb (e.g. "in_city", "in_zip_code",
  "belongs_to").

Respond with valid JSON only, no markdown:
{"links": [{"subject_type": "<new type>", "predicate": "<verb>",
"target_type": "<existing type>"}]}"""

BACKBONE_USER_TEMPLATE = """\
New entity types being added:
{new_types}

Existing types already in the ontology (the ONLY valid link targets):
{existing_types}

Propose how each new type connects into the existing types and return the JSON now."""


class OntologyResolver:
    """Resolves a fuzzy ask into a structured ontology-change plan.

    Construct with the OpenRouter key (for the intent-parse call + TypeMatcher
    cascade), the shared :class:`TypeMatcher`, and optionally the
    :class:`~cograph_client.nlp.ontology_embeddings.OntologyEmbeddingService`
    singleton (``get_embedding_service()``) and an Anthropic client for the
    offline intent-parse fallback. The resolver fetches the current ontology
    inventory itself (``resolve``'s ``neptune`` arg) OR accepts an already-built
    snapshot (``resolve_with_inventory``) so it stays unit-testable without
    Neptune.
    """

    # Mirror CSVResolver's provider/model config so the single intent-parse call
    # uses the same routing knobs as the rest of the resolver pipeline.
    EXTRACT_MODEL = os.environ.get("OMNIX_EXTRACT_MODEL", PRIMARY_MODEL)
    EXTRACT_PROVIDER = os.environ.get("OMNIX_EXTRACT_PROVIDER", "openrouter")
    INFER_MODEL = os.environ.get("OMNIX_INFER_MODEL", "claude-opus-4-8")

    def __init__(
        self,
        openrouter_key: str,
        type_matcher: TypeMatcher,
        embedding_service: object | None = None,
        anthropic_client: anthropic.AsyncAnthropic | None = None,
    ):
        self._openrouter_key = openrouter_key
        self._type_matcher = type_matcher
        self._embedding_service = embedding_service
        self._anthropic = anthropic_client

    # ── Public API ────────────────────────────────────────────────────────

    async def resolve(self, ask: str, graph_uri: str, neptune: NeptuneClient) -> ResolutionResult:
        """Resolve ``ask`` against the ontology of ``graph_uri``, fetching the
        current type/attribute inventory from ``neptune`` first.

        Returns a plan only — no triples are written.
        """
        inventory = await self._fetch_inventory(graph_uri, neptune)
        return await self.resolve_with_inventory(ask, graph_uri, inventory)

    async def resolve_with_inventory(
        self,
        ask: str,
        graph_uri: str,
        inventory: dict[str, "TypeInventory"],
    ) -> ResolutionResult:
        """Resolve ``ask`` against an already-fetched ontology ``inventory``
        (type name → :class:`TypeInventory`). Lets callers (and tests) supply a
        snapshot instead of a live Neptune client.
        """
        existing_types = {name: inv.description for name, inv in inventory.items()}

        intents = await self._parse_intents(ask, existing_types)

        applied: list[ResolvedChange] = []
        proposals: list[ResolvedChange] = []

        for intent in intents:
            change = await self._resolve_intent(intent, graph_uri, inventory, existing_types)
            if change is None:
                continue
            if change.action == "reuse" or change.action == "extend":
                applied.append(change)
            else:  # "create"
                proposals.append(change)

        # Backbone scaffold: any NEW entity type this ask introduces gets wired
        # into the existing ontology (e.g. a new Neighborhood → existing ZipCode /
        # City), so the graph stays connected and other entities reach it
        # transitively. Best-effort; emits 'create' proposals.
        proposals.extend(await self._scaffold_backbone(applied + proposals, inventory))

        summary = _summarize(ask, applied, proposals)
        logger.info(
            "ontology_resolved",
            ask=ask,
            applied=len(applied),
            proposals=len(proposals),
        )
        return ResolutionResult(applied=applied, proposals=proposals, summary=summary)

    # ── Step 1: intent parse (ONE LLM call) ───────────────────────────────

    async def _parse_intents(self, ask: str, existing_types: dict[str, str]) -> list[dict]:
        types_text = (
            "\n".join(
                f'- "{name}": {desc}' if desc else f'- "{name}"'
                for name, desc in existing_types.items()
            )
            or "(none)"
        )
        user = INTENT_USER_TEMPLATE.format(existing_types=types_text, ask=ask)

        try:
            raw = await self._call_intent_llm(user)
        except Exception as exc:  # provider outage / quota — fail soft, empty plan
            logger.warning("intent_parse_failed", ask=ask, error=str(exc))
            return []

        intents = raw.get("intents") if isinstance(raw, dict) else None
        if not isinstance(intents, list):
            logger.warning("intent_parse_bad_shape", ask=ask)
            return []
        return [i for i in intents if isinstance(i, dict) and i.get("subject_phrase")]

    async def _call_intent_llm(self, user_content: str) -> dict:
        """The single intent-parse LLM call. Tests monkeypatch this."""
        return await self._chat_json(INTENT_SYSTEM_PROMPT, user_content)

    async def _call_backbone_llm(self, user_content: str) -> dict:
        """The optional backbone-scaffold call (new types → existing types).
        Tests monkeypatch this."""
        return await self._chat_json(BACKBONE_SYSTEM_PROMPT, user_content)

    async def _chat_json(self, system: str, user_content: str) -> dict:
        """One JSON LLM call. OpenRouter primary→fallback when a key is configured
        (same as CSVResolver); Anthropic offline fallback otherwise."""
        if self.EXTRACT_PROVIDER == "openrouter" and self._openrouter_key:
            text = await openrouter_chat(
                self._openrouter_key,
                system,
                user_content,
                model=self.EXTRACT_MODEL,
                temperature=0.0,
                max_tokens=1024,
            )
            return json.loads(_strip_code_fences(text))
        if self._anthropic is None:
            raise RuntimeError("no LLM provider configured")
        msg = await self._anthropic.messages.create(
            model=self.INFER_MODEL,
            max_tokens=1024,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        return json.loads(_strip_code_fences(msg.content[0].text))

    # ── Backbone scaffold: wire new types into the existing ontology ──────

    async def _scaffold_backbone(
        self,
        changes: list[ResolvedChange],
        inventory: dict[str, "TypeInventory"],
    ) -> list[ResolvedChange]:
        """Wire each NEW entity type introduced by ``changes`` into the existing
        ontology backbone via one extra LLM call (new types → existing types).
        Returns extra 'create' relationship changes (so they surface as
        proposals). Fails soft — any error yields no backbone links."""
        existing = {name: inv.description for name, inv in inventory.items()}
        existing_lower = {n.lower(): n for n in existing}

        # New types = any type named by a 'create' change that isn't already in
        # the ontology (a new relationship target or a new subject).
        new_types: list[str] = []
        seen: set[str] = set()
        for c in changes:
            if c.action != "create":
                continue
            cands = [c.subject_type]
            if c.kind == "relationship":
                cands.append(c.datatype_or_target)
            for t in cands:
                t = (t or "").strip()
                if t and t.lower() not in existing_lower and t.lower() not in seen:
                    seen.add(t.lower())
                    new_types.append(t)
        if not new_types or not existing:
            return []

        new_text = "\n".join(f'- "{t}"' for t in new_types)
        existing_text = "\n".join(f'- "{n}"' for n in existing)
        user = BACKBONE_USER_TEMPLATE.format(new_types=new_text, existing_types=existing_text)
        try:
            raw = await self._call_backbone_llm(user)
        except Exception as exc:  # noqa: BLE001 — best-effort; never break the plan
            logger.warning("backbone_scaffold_failed", error=str(exc))
            return []

        links = raw.get("links") if isinstance(raw, dict) else None
        if not isinstance(links, list):
            return []

        new_lower = {t.lower() for t in new_types}
        out: list[ResolvedChange] = []
        emitted: set[tuple[str, str]] = set()
        for link in links:
            if not isinstance(link, dict):
                continue
            subj = str(link.get("subject_type", "")).strip()
            target_raw = str(link.get("target_type", "")).strip()
            verb = str(link.get("predicate", "")).strip()
            if not (subj and target_raw and verb):
                continue
            if subj.lower() not in new_lower:
                continue  # subject must be one of the new types
            target = existing_lower.get(target_raw.lower())
            if target is None:
                continue  # target must already exist — drop hallucinations
            predicate = normalize_predicate(verb, set())
            key = (subj.lower(), predicate)
            if key in emitted:
                continue
            emitted.add(key)
            out.append(
                ResolvedChange(
                    kind="relationship",
                    subject_type=subj,
                    name=predicate,
                    datatype_or_target=target,
                    action="create",
                    confidence=0.85,
                    reason=f"wire new type '{subj}' into the ontology backbone: {subj} → {target}",
                )
            )
        if out:
            logger.info("backbone_scaffold", new_types=new_types, links=len(out))
        return out

    # ── Steps 2–5: resolve one intent into a ResolvedChange ───────────────

    async def _resolve_intent(
        self,
        intent: dict,
        graph_uri: str,
        inventory: dict[str, "TypeInventory"],
        existing_types: dict[str, str],
    ) -> ResolvedChange | None:
        subject_phrase = str(intent.get("subject_phrase", "")).strip()
        if not subject_phrase:
            return None

        name_phrase = str(intent.get("name_phrase", "")).strip()
        target_phrase = str(intent.get("target_phrase", "") or "").strip()
        datatype_hint = str(intent.get("datatype_hint", "") or "").strip().lower()

        # Determine kind. Trust an explicit "relationship"; otherwise a populated
        # target_phrase or a non-primitive datatype_hint also implies a
        # relationship (datatype = another entity).
        kind = str(intent.get("kind", "")).strip().lower()
        if kind not in ("attribute", "relationship"):
            kind = "relationship" if target_phrase else "attribute"
        if kind == "attribute" and (
            target_phrase or (datatype_hint and datatype_hint not in PRIMITIVE_DATATYPES)
        ):
            kind = "relationship"
            if not target_phrase and datatype_hint:
                target_phrase = datatype_hint

        # Step 2: resolve the subject type.
        subject_match = await self._match_type(subject_phrase, graph_uri, existing_types)
        subject_new = subject_match.is_new
        subject_type = subject_match.resolved

        # The subject type's existing schema only matters when reusing an
        # existing type; a brand-new type has no schema to match against.
        subject_inv = inventory.get(subject_type)

        if kind == "relationship":
            return await self._resolve_relationship(
                subject_type, subject_match, subject_inv, name_phrase,
                target_phrase, graph_uri, existing_types,
            )
        return self._resolve_attribute_intent(
            subject_type, subject_match, subject_inv, name_phrase, datatype_hint,
        )

    def _resolve_attribute_intent(
        self,
        subject_type: str,
        subject_match: TypeMatch,
        subject_inv: "TypeInventory | None",
        name_phrase: str,
        datatype_hint: str,
    ) -> ResolvedChange:
        datatype = datatype_hint if datatype_hint in PRIMITIVE_DATATYPES else "string"
        existing_attrs = subject_inv.attribute_schemas() if subject_inv else {}

        resolved = resolve_attribute(
            ExtractedAttribute(name=name_phrase or "value", value="", datatype=datatype),
            existing_attrs,
        )
        # COERCE collapses to a reuse for planning purposes (the attribute name
        # already exists; we keep its declared datatype).
        attr_reused = resolved.action in (AttrAction.REUSE, AttrAction.COERCE)

        action, confidence, reason = self._gate(
            subject_match=subject_match,
            name_reused=attr_reused,
            target_new=False,
            what=f"attribute '{resolved.name}'",
        )
        return ResolvedChange(
            kind="attribute",
            subject_type=subject_type,
            name=resolved.name,
            datatype_or_target=resolved.datatype,
            action=action,
            confidence=confidence,
            reason=reason,
        )

    async def _resolve_relationship(
        self,
        subject_type: str,
        subject_match: TypeMatch,
        subject_inv: "TypeInventory | None",
        name_phrase: str,
        target_phrase: str,
        graph_uri: str,
        existing_types: dict[str, str],
    ) -> ResolvedChange:
        # Step 3: normalize the predicate against the subject's existing rels.
        existing_predicates = subject_inv.relationship_predicates() if subject_inv else set()
        predicate = normalize_predicate(name_phrase or "related_to", existing_predicates)
        rel_reused = predicate in existing_predicates

        # Step 4: resolve the relationship target type (a new target = creation).
        target_new = True
        target_type = target_phrase or "Entity"
        if target_phrase:
            target_match = await self._match_type(target_phrase, graph_uri, existing_types)
            target_type = target_match.resolved
            target_new = target_match.is_new

        action, confidence, reason = self._gate(
            subject_match=subject_match,
            name_reused=rel_reused,
            target_new=target_new,
            what=f"relationship '{predicate}' → {target_type}",
        )
        return ResolvedChange(
            kind="relationship",
            subject_type=subject_type,
            name=predicate,
            datatype_or_target=target_type,
            action=action,
            confidence=confidence,
            reason=reason,
        )

    # ── Step 5: confidence gate ───────────────────────────────────────────

    def _gate(
        self,
        subject_match: TypeMatch,
        name_reused: bool,
        target_new: bool,
        what: str,
    ) -> tuple[str, float, str]:
        """Decide action + confidence + reason for one resolved change.

        APPLY (reuse/extend) requires: the subject is an EXISTING SAME match,
        confident enough (>= APPLY_CONFIDENCE_FLOOR), and no new type is needed
        (subject not new, target not new). Anything else PROPOSES a "create".
        """
        subject_same = subject_match.verdict == MatchVerdict.SAME and not subject_match.is_new

        if subject_match.is_new:
            return (
                "create",
                subject_match.confidence,
                f"new subject type '{subject_match.resolved}' must be created for {what}",
            )
        if target_new:
            return (
                "create",
                subject_match.confidence,
                f"new target type must be created for {what}",
            )
        if not subject_same:
            return (
                "create",
                subject_match.confidence,
                f"subject type match is {subject_match.verdict.value} (not a clean SAME) for {what}",
            )
        if subject_match.confidence < APPLY_CONFIDENCE_FLOOR:
            return (
                "create",
                subject_match.confidence,
                f"subject match confidence {subject_match.confidence:.2f} below apply floor for {what}",
            )

        # Clean existing subject + confident: reuse if the name already exists,
        # otherwise extend the existing type with the new property.
        if name_reused:
            return (
                "reuse",
                subject_match.confidence,
                f"{what} already exists on '{subject_match.resolved}' — nothing to add",
            )
        return (
            "extend",
            subject_match.confidence,
            f"extend existing type '{subject_match.resolved}' with {what}",
        )

    # ── Type matching helper ──────────────────────────────────────────────

    async def _match_type(
        self,
        phrase: str,
        graph_uri: str,
        existing_types: dict[str, str],
    ) -> TypeMatch:
        """Resolve a subject/target phrase to a type via the shared TypeMatcher
        cascade (exact name → cache → embeddings → LLM). The embedding retrieve
        narrows candidates inside the matcher; an empty ontology yields a new
        type deterministically."""
        return await self._type_matcher.match(
            proposed_type=phrase,
            proposed_description="",
            existing_types=existing_types,
        )

    # ── Ontology inventory ────────────────────────────────────────────────

    async def _fetch_inventory(
        self, graph_uri: str, neptune: NeptuneClient
    ) -> dict[str, "TypeInventory"]:
        """Read the current type/attribute inventory of ``graph_uri`` from
        Neptune (one ``get_full_ontology_query``). Relationship vs primitive
        attributes are distinguished by whether the range URI points at a
        ``types/`` URI (same rule the NL pipeline + embeddings use)."""
        raw = await neptune.query(get_full_ontology_query(graph_uri))
        _, bindings = parse_sparql_results(raw)
        return build_inventory(bindings)


# ---------------------------------------------------------------------------
# Ontology snapshot
# ---------------------------------------------------------------------------


class TypeInventory:
    """A snapshot of one existing type: its primitive attributes (name →
    datatype) and its relationship predicates (predicate → target type).

    Constructed from ``get_full_ontology_query`` bindings (``build_inventory``),
    or directly by tests. ``attribute_schemas`` / ``relationship_predicates``
    adapt it to the shapes ``resolve_attribute`` / ``normalize_predicate``
    expect.
    """

    __slots__ = ("name", "description", "attributes", "relationships")

    def __init__(
        self,
        name: str,
        description: str = "",
        attributes: dict[str, str] | None = None,
        relationships: dict[str, str] | None = None,
    ):
        self.name = name
        self.description = description
        #: attribute name → primitive datatype
        self.attributes: dict[str, str] = attributes or {}
        #: predicate name → target type name
        self.relationships: dict[str, str] = relationships or {}

    def attribute_schemas(self) -> dict[str, AttributeSchema]:
        """Adapt to the ``dict[str, AttributeSchema]`` ``resolve_attribute`` wants."""
        return {
            name: AttributeSchema(name=name, datatype=dtype)
            for name, dtype in self.attributes.items()
        }

    def relationship_predicates(self) -> set[str]:
        """Adapt to the ``set[str]`` ``normalize_predicate`` wants."""
        return set(self.relationships.keys())


def build_inventory(bindings: list[dict]) -> dict[str, TypeInventory]:
    """Build the type inventory from ``get_full_ontology_query`` bindings.

    Each binding may carry a type (``typeLabel``) and optionally one of its
    attributes (``attrLabel`` + ``range``). A range that points at a ``types/``
    URI is a relationship (range = the target type); otherwise it's a primitive
    attribute whose datatype is the XSD local name.
    """
    inv: dict[str, TypeInventory] = {}
    for row in bindings:
        type_label = row.get("typeLabel", "")
        if not type_label:
            continue
        ti = inv.get(type_label)
        if ti is None:
            ti = TypeInventory(name=type_label)
            inv[type_label] = ti

        attr_label = row.get("attrLabel", "")
        if not attr_label:
            continue
        range_str = row.get("range", "")
        if range_str.startswith(TYPE_URI_PREFIX):
            target = range_str[len(TYPE_URI_PREFIX):]
            ti.relationships[attr_label] = target
        else:
            dtype = range_str.split("#")[-1] if "#" in range_str else "string"
            ti.attributes[attr_label] = dtype
    return inv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [ln for ln in stripped.split("\n") if not ln.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    return stripped


def _summarize(
    ask: str,
    applied: list[ResolvedChange],
    proposals: list[ResolvedChange],
) -> str:
    parts = [f'Ask "{ask}": ', f"{len(applied)} change(s) ready to apply"]
    if proposals:
        parts.append(f", {len(proposals)} proposed (new type / ambiguous)")
    parts.append(".")
    return "".join(parts)
