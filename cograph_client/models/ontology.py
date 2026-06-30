from typing import Literal

from pydantic import BaseModel, Field


class AttributeDefinition(BaseModel):
    name: str = Field(min_length=1)
    description: str = ""
    datatype: str = Field(default="string", description="string, integer, float, boolean, datetime, uri, geo (WKT point / 'lat,lon'), or a type name for relationships")


class TypeCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    parent_type: str | None = Field(default=None, description="Parent type name for subtype relationship")
    attributes: list[AttributeDefinition] = Field(default_factory=list)


class TypeResponse(BaseModel):
    name: str
    description: str = ""
    parent_type: str | None = None
    attributes: list[AttributeDefinition] = Field(default_factory=list)
    subtypes: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)


class AttributeAdd(BaseModel):
    attributes: list[AttributeDefinition] = Field(min_length=1)


class SubtypeAdd(BaseModel):
    subtype: str = Field(min_length=1, description="Name of the child type")


# ---------------------------------------------------------------------------
# Ontology evolution resolver (COG-84)
#
# The OntologyResolver (cograph_client/resolver/ontology_resolver.py) turns a
# fuzzy natural-language "ask" into a structured PLAN of ontology changes. These
# models are the plan it returns over REST: each ResolvedChange is one
# attribute/relationship the ask implies, classified as a clear REUSE/EXTEND on
# an existing type (auto-APPLY) or a creation / ambiguous match (PROPOSE). The
# resolver NEVER writes to Neptune — a later /resolve→/apply REST pair (COG-81)
# consumes this plan. JSON-serializable by construction (plain pydantic).
# ---------------------------------------------------------------------------


class ResolvedChange(BaseModel):
    """One concrete ontology change the ask implies, already resolved against
    the current ontology.

    ``action`` is the verb the apply layer (COG-81) executes:
      - ``reuse``  — the attribute/relationship already exists on ``subject_type``;
        nothing to write (the ask is already satisfied).
      - ``extend`` — ``subject_type`` exists, but this attribute/relationship is
        new on it: add the property (``insert_attribute`` / object-property).
      - ``create`` — a NEW type must be minted first (the subject type itself is
        new, or a relationship target type doesn't exist yet), then the
        property is added. These are the changes that land in ``proposals``.
    """

    kind: Literal["attribute", "relationship"]
    subject_type: str = Field(description="Resolved type the change attaches to (existing name, or a proposed new one)")
    name: str = Field(description="Resolved attribute name (attribute) or predicate (relationship), normalized")
    datatype_or_target: str = Field(
        description=(
            "For an attribute: the primitive datatype (string/integer/float/"
            "boolean/datetime/uri). For a relationship: the target type name "
            "(its range) the predicate points at."
        ),
    )
    action: Literal["reuse", "extend", "create"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", description="One-line human-readable rationale for the action/gate decision")


class ResolutionResult(BaseModel):
    """The full PLAN the OntologyResolver returns for one ask.

    ``applied`` are high-confidence changes the caller may auto-APPLY (existing
    subject type + a clear reuse/extend, no new type). ``proposals`` are changes
    that need confirmation — a new type must be created, or the match was
    mid-band/ambiguous. The split is advisory: the resolver writes nothing, and
    the apply layer (COG-81) decides what to commit.
    """

    applied: list[ResolvedChange] = Field(default_factory=list)
    proposals: list[ResolvedChange] = Field(default_factory=list)
    summary: str = ""
    dry_run: bool = Field(
        default=False,
        description="True when the caller requested plan-only mode: nothing was written and every change is surfaced under `proposals`.",
    )


class ResolveRequest(BaseModel):
    """Body for ``POST /graphs/{tenant}/ontology/resolve`` (COG-81).

    ``ask`` is the fuzzy natural-language evolution request (e.g. "track which
    company a person works for"). ``knowledge_graph`` is an optional scope hint
    carried for parity with the rest of the API; the resolver resolves against
    the tenant's ontology graph regardless.
    """

    ask: str = Field(min_length=1, description="Natural-language ontology-evolution request")
    knowledge_graph: str | None = Field(default=None, description="Optional KG scope hint")
    dry_run: bool = Field(
        default=False,
        description=(
            "Plan-only mode. When false (default, the MCP/agent path) the route "
            "auto-applies the resolver's high-confidence changes and returns the "
            "rest as proposals. When true (the interactive Explorer path) nothing "
            "is written: every change — what would have auto-applied plus the "
            "proposals — is returned under `proposals`, `applied` is empty."
        ),
    )
