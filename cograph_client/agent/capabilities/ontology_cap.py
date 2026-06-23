"""Ontology capability — inspect the schema, or declare/extend it through the agent.

Reuses the EXISTING ontology engine end-to-end (no reimplementation):

* **inspect / describe** (read-only) — lists the active type's declared
  attributes + relationships (and, when no type is in scope, the tenant's types)
  by reusing :func:`cograph_client.normalization.inference.list_type_schema`
  (the same bounded single round-trip read the enrich/normalize planners ground
  their extraction in) and :func:`...ontology_queries.list_types_query`. A
  read-only request mutates nothing, so it is surfaced as an ``answer`` step the
  planner returns directly (like :class:`QueryCapability`), not a confirm plan.

* **declare / extend** (mutating) — proposes ONE :class:`PlanStep` to add an
  attribute (or a new type) to the ontology. ``execute`` commits it through the
  SAME idempotent atomic upsert builders the ontology endpoint + the enrichment
  "declare-then-write" path use (:func:`...ontology_queries.upsert_attribute`,
  :func:`...ontology_queries.insert_type`), so a retry is safe and the schema
  the agent writes is byte-identical to what the REST/MCP surface writes.

The agent never calls the ``/ontology/*`` HTTP routes — it drives the same query
builders + Neptune client directly. Ontology edits are FREE (no paid calls), so
every declare step carries ``cost = {"paid_calls": 0, "estimated_usd": 0.0}``.
"""

from __future__ import annotations

import json

import structlog

from cograph_client.agent.registry import AgentContext, PlanStep
from cograph_client.graph.ontology_queries import (
    PRIMITIVE_TYPES,
    insert_type,
    list_types_query,
    upsert_attribute,
)
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import tenant_graph_uri
from cograph_client.normalization.inference import list_type_schema
from cograph_client.resolver.llm_router import PRIMARY_MODEL, openrouter_chat

logger = structlog.stdlib.get_logger("cograph.agent.ontology")

# Ontology edits never call a paid external source — the cost is always free.
# Key names match the web plan-step cost contract EXACTLY (``step.cost.paid_calls``
# / ``step.cost.estimated_usd`` — web/app/components/explore/useAgentChat.ts
# AgentStepCost + AgentChat.tsx PlanStepRow). Do NOT rename without updating both.
_FREE_COST = {"paid_calls": 0, "estimated_usd": 0.0}


class OntologyCapability:
    name = "ontology"

    def describe(self) -> str:
        return (
            "Inspect or change the SCHEMA (ontology) of a type: describe its "
            "declared attributes/relationships, or declare a new attribute or "
            "type. Use for 'what attributes does X have', 'show the schema', "
            "'add a <field> attribute to <type>', 'create a <type> type'."
        )

    # --- inspect (read-only, surfaced as an answer) -------------------------- #

    async def describe_ontology(self, ctx: AgentContext) -> dict:
        """Render the ontology for the current scope as an answer payload.

        With a ``type_name`` in scope: the type's declared attributes +
        relationships (with target types). Without one: the tenant's type list.
        Read-only — a single bounded ontology query, never an instance scan.
        """
        if ctx.type_name:
            schema = await list_type_schema(ctx.neptune, ctx.tenant_id, ctx.type_name)
            attributes = schema.get("attributes", [])
            relationships = schema.get("relationships", [])
            answer = _format_type_schema(ctx.type_name, attributes, relationships)
            return {
                "answer": answer,
                "ontology": {
                    "type": ctx.type_name,
                    "attributes": attributes,
                    "relationships": relationships,
                },
                "narrative": answer,
                "rows": [],
            }
        types = await self._list_types(ctx)
        answer = _format_type_list(types)
        return {
            "answer": answer,
            "ontology": {"types": types},
            "narrative": answer,
            "rows": [],
        }

    async def _list_types(self, ctx: AgentContext) -> list[dict]:
        """The tenant's declared types (name + description), reusing the same
        ontology query the ``/ontology/types`` route uses."""
        onto_graph = tenant_graph_uri(ctx.tenant_id)
        _, rows = parse_sparql_results(
            await ctx.neptune.query(list_types_query(onto_graph))
        )
        seen: set[str] = set()
        types: list[dict] = []
        for r in rows:
            label = r.get("label", "")
            if not label or label in seen:
                continue
            seen.add(label)
            types.append({"name": label, "description": r.get("comment", "")})
        return types

    # --- plan: inspect → answer step; declare → confirm step ---------------- #

    async def plan(self, ctx: AgentContext, instruction: str) -> list[PlanStep]:
        """Classify the ontology request into inspect | declare and build a step.

        * inspect → a single no-write ``answer`` step carrying the rendered
          schema (the planner surfaces an ``action == "answer"`` step directly
          as ``{kind:"answer", ...}``, like a question).
        * declare → a single ``declare_attribute`` / ``declare_type`` PlanStep
          (a proposal the user confirms; ``execute`` runs the atomic upsert).
        """
        directive = await self._extract_directive(ctx, instruction)
        op = directive.get("op")

        if op == "inspect":
            payload = await self.describe_ontology(ctx)
            return [
                PlanStep(
                    capability=self.name,
                    action="answer",
                    params={"answer_payload": payload},
                    rationale="Read-only ontology inspection; no schema changes.",
                    confidence=1.0,
                    preview={"summary": "Describes the ontology; writes nothing."},
                    cost=dict(_FREE_COST),
                )
            ]

        if op == "declare_type":
            type_name = directive.get("type_name") or ""
            if not type_name:
                return []
            description = directive.get("description", "") or ""
            parent = directive.get("parent_type") or None
            return [
                PlanStep(
                    capability=self.name,
                    action="declare_type",
                    params={
                        "type_name": type_name,
                        "description": description,
                        "parent_type": parent,
                    },
                    rationale=(
                        f"Declare a new ontology type '{type_name}'"
                        + (f" under {parent}" if parent else "")
                        + "."
                    ),
                    confidence=float(directive.get("confidence", 0.8) or 0.8),
                    preview={
                        "summary": (
                            f"Add a new type '{type_name}' to the ontology"
                            + (f" as a subtype of {parent}" if parent else "")
                            + "."
                        ),
                        "type": type_name,
                        "parent_type": parent,
                    },
                    cost=dict(_FREE_COST),
                )
            ]

        if op == "declare_attribute":
            type_name = directive.get("type_name") or ctx.type_name or ""
            attribute = directive.get("attribute") or ""
            if not type_name or not attribute:
                return []
            datatype = directive.get("datatype") or "string"
            description = directive.get("description", "") or ""
            return [
                PlanStep(
                    capability=self.name,
                    action="declare_attribute",
                    params={
                        "type_name": type_name,
                        "attribute": attribute,
                        "datatype": datatype,
                        "description": description,
                    },
                    rationale=(
                        f"Declare attribute '{attribute}' ({datatype}) on "
                        f"{type_name}."
                    ),
                    confidence=float(directive.get("confidence", 0.8) or 0.8),
                    preview={
                        "summary": (
                            f"Add a '{attribute}' attribute (type {datatype}) to "
                            f"the {type_name} schema. Existing records are "
                            f"unaffected until values are filled in."
                        ),
                        "type": type_name,
                        "attribute": attribute,
                        "datatype": datatype,
                    },
                    cost=dict(_FREE_COST),
                )
            ]

        # Unrecognized / underspecified → no step (planner clarifies).
        return []

    async def execute(self, ctx: AgentContext, step: PlanStep) -> dict:
        """Run one ontology step.

        * ``answer`` → return the precomputed inspect payload (no write).
        * ``declare_type`` / ``declare_attribute`` → commit the atomic upsert
          through the SAME builders the ontology endpoint / enrichment declare
          path use, against the TENANT (ontology) graph.
        """
        p = step.params
        if step.action == "answer":
            payload = p.get("answer_payload")
            if payload is None:  # re-run safety: recompute if not carried
                payload = await self.describe_ontology(ctx)
            return {"kind": "answer", **payload}

        onto_graph = tenant_graph_uri(ctx.tenant_id)

        if step.action == "declare_type":
            type_name = p["type_name"]
            await ctx.neptune.update(
                insert_type(
                    onto_graph,
                    type_name,
                    description=p.get("description", "") or "",
                    parent_type=p.get("parent_type") or None,
                )
            )
            return {
                "kind": "ack",
                "capability": self.name,
                "action": step.action,
                "type_name": type_name,
                "message": f"Declared type '{type_name}' in the ontology.",
            }

        if step.action == "declare_attribute":
            type_name = p["type_name"]
            attribute = p["attribute"]
            await ctx.neptune.update(
                upsert_attribute(
                    onto_graph,
                    type_name,
                    attribute,
                    description=p.get("description", "") or "",
                    datatype=p.get("datatype", "string") or "string",
                )
            )
            return {
                "kind": "ack",
                "capability": self.name,
                "action": step.action,
                "type_name": type_name,
                "attribute": attribute,
                "message": (
                    f"Declared attribute '{attribute}' on {type_name} in the "
                    "ontology."
                ),
            }

        raise ValueError(f"unknown ontology action: {step.action!r}")

    # --- LLM directive extraction, grounded in the type's real schema -------- #

    async def _extract_directive(self, ctx: AgentContext, instruction: str) -> dict:
        """LLM-extract {op, type_name?, attribute?, datatype?, ...}.

        Grounded in the active type's real schema so a declare maps to a clean
        predicate name and an inspect is recognized. Degrades to a deterministic
        keyword heuristic when there is no key or the LLM errors, so the agent
        never 500s on extraction.
        """
        schema = {"attributes": [], "relationships": []}
        if ctx.type_name:
            try:
                schema = await list_type_schema(
                    ctx.neptune, ctx.tenant_id, ctx.type_name
                )
            except Exception:  # noqa: BLE001 — schema is only for grounding
                logger.warning("agent_ontology_schema_failed", exc_info=True)

        parsed: dict | None = None
        if ctx.openrouter_key:
            attr_names = [a for a in schema.get("attributes", []) if a]
            rels_block = ", ".join(
                f"{r['name']} (-> {r.get('target_type') or '?'})"
                for r in schema.get("relationships", [])
                if r.get("name")
            ) or "(none)"
            user = _EXTRACT_USER_TEMPLATE.format(
                type_name=ctx.type_name or "(none selected)",
                attributes=", ".join(attr_names) or "(none)",
                relationships=rels_block,
                instruction=instruction,
            )
            try:
                text = await openrouter_chat(
                    ctx.openrouter_key,
                    _EXTRACT_SYSTEM,
                    user,
                    model=PRIMARY_MODEL,
                    temperature=0,
                    max_tokens=300,
                    timeout=30,
                )
                parsed = _parse_json_object(text)
            except Exception:
                logger.warning("agent_ontology_extract_failed", exc_info=True)
                parsed = None
        if not parsed:
            parsed = _heuristic_directive(instruction)
        return _validate_directive(parsed, ctx.type_name)


# --------------------------------------------------------------------------- #
# Rendering helpers (answer text for inspect)
# --------------------------------------------------------------------------- #
def _format_type_schema(
    type_name: str, attributes: list[str], relationships: list[dict]
) -> str:
    lines = [f"Ontology for {type_name}:"]
    if attributes:
        lines.append("Attributes: " + ", ".join(attributes))
    else:
        lines.append("Attributes: (none declared)")
    if relationships:
        rels = ", ".join(
            f"{r['name']} → {r.get('target_type') or '?'}"
            for r in relationships
            if r.get("name")
        )
        lines.append("Relationships: " + rels)
    else:
        lines.append("Relationships: (none declared)")
    return "\n".join(lines)


def _format_type_list(types: list[dict]) -> str:
    if not types:
        return "No types are declared in this tenant's ontology yet."
    names = ", ".join(t["name"] for t in types)
    return f"Types in the ontology ({len(types)}): {names}"


# --------------------------------------------------------------------------- #
# Directive extraction + validation
# --------------------------------------------------------------------------- #
_EXTRACT_SYSTEM = """\
You translate a user's ontology (schema) instruction into ONE operation, \
GROUNDED in the active type's real schema. You are given the active type and its \
actual ATTRIBUTE and RELATIONSHIP names. Decide which operation the user wants:

- "inspect": READ-ONLY — the user wants to SEE the schema ("what attributes does \
this type have", "describe the ontology", "show the schema", "list the types", \
"what relationships are there").
- "declare_attribute": ADD a new attribute to a type ("add a founded_year \
attribute", "give Company a website field", "track an email for mentors").
- "declare_type": ADD a new type to the ontology ("create a Product type", "add \
a new type called Venue under Place").

Return STRICT JSON only (no markdown):
{
  "op": "inspect" | "declare_attribute" | "declare_type",
  "type_name": "<type name>" | null,
  "attribute": "<new attribute leaf name, for declare_attribute>" | null,
  "datatype": "string" | "integer" | "float" | "boolean" | "datetime" | "uri" | \
"<another type name for a relationship>",
  "parent_type": "<parent type name, for declare_type>" | null,
  "description": "<short description>" | null,
  "confidence": 0.0
}

RULES:
- For "declare_attribute": "attribute" is a clean lowercase singular noun (e.g. \
"website", "founded_year") — NEVER a modifier word ("the", "a", "new"). If the \
user does not name a type and one is already active, leave "type_name" null (the \
active type is used). Pick "datatype" from the listed primitives; use another \
TYPE name only when the user clearly wants a relationship to that type.
- For "declare_type": "type_name" is the new type's name (PascalCase if the user \
uses it). "parent_type" only if the user says "under X" / "a kind of X".
- For "inspect": set type_name to the type the user names, else null.
- Set "confidence" in [0,1]."""

_EXTRACT_USER_TEMPLATE = """\
Active type: {type_name}
Attributes: {attributes}
Relationships: {relationships}

Instruction: {instruction}

Which ontology operation does this ask for? Respond with strict JSON."""

_VALID_OPS = {"inspect", "declare_attribute", "declare_type"}


def _validate_directive(parsed: dict, active_type: str | None) -> dict:
    """Sanitize an extracted directive: clamp the op, clean names/datatype."""
    if not isinstance(parsed, dict):
        return {"op": None}
    op = parsed.get("op")
    if op not in _VALID_OPS:
        return {"op": None}

    out: dict = {"op": op, "confidence": parsed.get("confidence", 0.8)}

    type_name = _clean_name(parsed.get("type_name"))
    description = parsed.get("description") or ""
    out["description"] = description if isinstance(description, str) else ""

    if op == "inspect":
        out["type_name"] = type_name or active_type
        return out

    if op == "declare_type":
        out["type_name"] = type_name
        out["parent_type"] = _clean_name(parsed.get("parent_type")) or None
        return out

    # declare_attribute
    out["type_name"] = type_name or active_type
    out["attribute"] = _clean_attr(parsed.get("attribute"))
    out["datatype"] = _clean_datatype(parsed.get("datatype"))
    return out


# Stray modifier / filler words an extractor must never emit as a name.
_STOPWORDS = {
    "the", "a", "an", "new", "this", "that", "their", "its", "his", "her",
    "of", "for", "to", "with", "attribute", "field", "type", "property",
}


def _clean_name(value) -> str:
    """Reduce an extracted type/parent name to a usable identifier, or ""."""
    if not isinstance(value, str):
        return ""
    v = value.strip()
    return v if v and v.lower() not in _STOPWORDS else ""


def _clean_attr(value) -> str:
    """Reduce an extracted attribute phrase to a clean leaf noun, or "".

    Drops leading modifier words ("a new website" -> "website") and slugs spaces
    to underscores so the result is a usable attribute leaf name.
    """
    if not isinstance(value, str):
        return ""
    words = [w for w in value.strip().split() if w]
    while words and words[0].lower() in _STOPWORDS:
        words.pop(0)
    kept: list[str] = []
    for w in words:
        if w.lower() in _STOPWORDS:
            break
        kept.append(w)
    cleaned = "_".join(kept).strip("_-")
    return cleaned if cleaned and cleaned.lower() not in _STOPWORDS else ""


def _clean_datatype(value) -> str:
    """Keep a primitive datatype, else treat a bare name as a relationship target
    type, else default to ``string`` (the same default the ontology endpoint uses)."""
    if not isinstance(value, str) or not value.strip():
        return "string"
    v = value.strip()
    if v in PRIMITIVE_TYPES:
        return v
    # A non-primitive, non-stopword name is a target type (relationship range).
    cleaned = _clean_name(v)
    return cleaned or "string"


# --------------------------------------------------------------------------- #
# Deterministic fallback (no LLM key / LLM error)
# --------------------------------------------------------------------------- #
_INSPECT_HINTS = (
    "what attribute", "which attribute", "what relationship", "describe",
    "show the schema", "show schema", "the schema", "list the type",
    "list types", "what type", "which type", "inspect", "what fields",
    "what columns", "what does", "view the ontology", "show the ontology",
)


def _heuristic_directive(instruction: str) -> dict:
    """Best-effort op detection from keywords when the LLM is unavailable.

    Deliberately conservative: it confidently recognizes an INSPECT request (the
    most common ontology ask and the one that's always safe — read-only). For a
    declare it can spot the op and a quoted/trailing attribute or type name, but
    when in doubt it returns ``{"op": None}`` so the planner clarifies rather
    than mis-declaring schema from a vague instruction.
    """
    text = instruction.lower()
    if any(h in text for h in _INSPECT_HINTS):
        return {"op": "inspect", "confidence": 0.6}

    if ("add" in text or "declare" in text or "create" in text or "new" in text):
        if "type" in text:
            # "create a Product type" / "add a new type Venue"
            name = _last_capitalized(instruction)
            if name:
                return {"op": "declare_type", "type_name": name, "confidence": 0.5}
        # "add a website attribute" / "add a founded_year field"
        attr = _attr_after_add(instruction)
        if attr:
            return {
                "op": "declare_attribute",
                "attribute": attr,
                "confidence": 0.5,
            }
    return {"op": None}


def _last_capitalized(instruction: str) -> str:
    """The last Capitalized token (a likely type name) in the instruction."""
    caps = [w for w in instruction.split() if w[:1].isupper() and w.isalpha()]
    return caps[-1] if caps else ""


_ADD_ATTR_STOPWORDS = {
    "a", "an", "the", "new", "add", "declare", "create", "attribute", "field",
    "property", "column", "to", "for", "on", "of",
}


def _attr_after_add(instruction: str) -> str:
    """Pull the attribute noun out of 'add a <noun> attribute/field' phrasing."""
    words = instruction.replace("'", " ").replace('"', " ").split()
    candidates = [
        w.strip(".,") for w in words
        if w.strip(".,").lower() not in _ADD_ATTR_STOPWORDS and w.strip(".,")
    ]
    # Take the first plausible noun-like token.
    for c in candidates:
        leaf = _clean_attr(c)
        if leaf:
            return leaf
    return ""


def _parse_json_object(text: str) -> dict | None:
    """Best-effort parse of an LLM JSON object reply (tolerant of code fences)."""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = "\n".join(
            l for l in stripped.split("\n") if not l.strip().startswith("```")
        )
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        stripped = stripped[start : end + 1]
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None
