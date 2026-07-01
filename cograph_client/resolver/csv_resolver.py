"""CSV schema inference, then deterministic mapping for all rows (no LLM per
row).

Inference (ADR 0003, default ``OMNIX_CSV_INFERENCE_V2=1``) is the
evidence-grounded pipeline: deterministic profile (Pass A) → REASON LLM call
(Pass B) → adversarial REFUTE LLM call (Pass C) → conceptual COMPLETE LLM
call (Pass D — dependent-entity promotions + constitutive core slots, max 3
per type) → conversion to :class:`CSVSchemaMapping`. Set
``OMNIX_CSV_INFERENCE_V2=0`` to fall back to the legacy single-LLM-call path
(verbatim, including its post-hoc keyword patches)."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

import anthropic
import httpx
import structlog
from pydantic import ValidationError

from cograph_client.resolver.models import (
    ColumnMapping,
    ColumnProfile,
    ColumnRole,
    CoreSlot,
    CoreSlotTests,
    CSVSchemaMapping,
    DatasetConstant,
    EntityRelationSpec,
    EntitySpec,
    ExtractedAttribute,
    ExtractedEntity,
    ExtractedRelationship,
    InferenceAudit,
    OntologyExtensions,
    RejectedSlot,
    SchemaViolation,
    TableProfile,
    TypeExtension,
    ValueShape,
)
from cograph_client.resolver.llm_router import PRIMARY_MODEL, openrouter_chat
from cograph_client.resolver.profiler import profile_table

logger = structlog.stdlib.get_logger("cograph.resolver.csv")


@dataclass
class AppliedMapping:
    """Result of ``CSVResolver.apply_mapping``: the extracted entities and
    relationships plus row-conservation accounting (ADR 0003 §2 — input rows
    are never silently dropped).

    Iterates as the legacy ``(entities, relationships)`` pair, so existing
    two-value unpacking call sites keep working unchanged; new callers read
    ``rows_in`` / ``rows_dropped`` / ``drops_by_entity`` off the object.
    """

    entities: list[ExtractedEntity]
    relationships: list[ExtractedRelationship]
    #: Number of input rows this call received.
    rows_in: int = 0
    #: Rows that produced no entity at all. Only possible when every owned
    #: value in the row is empty — a principled skip (nothing to assert),
    #: never a silent drop: it is always counted and logged.
    rows_dropped: int = 0
    #: Skipped entity-instances per mapping entity. Keyed by entity_type in
    #: single-entity mode, by EntitySpec.name in multi-entity mode (one row
    #: can mint some entities while skipping an all-empty one without the
    #: row itself counting as dropped).
    drops_by_entity: dict[str, int] = field(default_factory=dict)

    def __iter__(self):
        yield self.entities
        yield self.relationships

CSV_SCHEMA_SYSTEM = """\
You are a knowledge graph schema inference engine. Given CSV column names and
sample rows, decide how to turn the table into entities, attributes, and
relationships.

STEP 1 — How many real-world entities does ONE ROW describe?
Wide/denormalized exports usually bundle SEVERAL distinct entities per row — a
person, a transaction, a place, an organization, a product. Read the column-name
clusters AND the sample values: each distinct real-world "noun" that has its own
identity is a separate entity. This is the COMMON case for exports across every
domain (orders, claims, encounters, bookings, rosters, listings…). Default to
multi-entity unless the row genuinely describes ONE thing.

MULTI-ENTITY output (the usual case) — return:
- `entities`: one object per entity, each with `name` (a local handle),
  `type_name` (PascalCase singular), and an id — either `id_column` (a natural
  key column like order_id / patient_id / sku) OR `id_from` (the columns that
  together identify it, e.g. ["customer_email"] or ["first_name","last_name",
  "phone"]) when there is no single id column.
- every column tagged with `entity` = the entity `name` it belongs to.
- `relationships`: {{subject, predicate, object}} edges between entity `name`s;
  predicate is a snake_case verb (order `placed_by` customer, order `contains`
  product, encounter `treated_by` provider, claim `filed_against` policy).
- SAME TYPE TWICE: if two column-clusters are the same base type in different
  roles (buyer & seller, sender & receiver, patient & provider, applicant &
  co_applicant) make them TWO separate entities with distinct names and
  role-distinct relationships — NEVER merge them into one.

SINGLE-ENTITY output — ONLY when the row describes one thing (a product catalog,
a transactions ledger, a lab result, a sensor reading): OMIT `entities` and
`relationships`, return entity_type + columns with exactly one column = type_id.
Do not invent entities for a genuinely flat row.

Type naming & reuse: the user message lists the tenant's EXISTING ontology
types. Reuse one ONLY when your entity is genuinely the SAME real-world concept
(another order → Order; another guest → Person). If none genuinely matches,
propose a NEW accurate PascalCase type name — NEVER force-fit a different concept
onto an available type just because it exists (a hospital is a Facility, not a
Property; a drug is a Drug, not a Product; an airport is an Airport, not a City).

Column roles & datatypes (both modes):
- role = type_id (single-entity only) | attribute | relationship.
- IN-ROW entities are expressed via the `entities` array — NOT via relationship
  columns. Use a `relationship` column (with `target_type`) only for a shared
  out-of-row dimension that is NOT one of your in-row entities (e.g. a bare
  country or category name with no other columns describing it).
- Datatype from the VALUE, not its JSON type (values may arrive as numbers or
  strings): numbers → integer/float, dates → datetime, true/false → boolean,
  URLs → uri, else string.
- NEVER use a date, timestamp, or a non-unique label as an id. If no unique key
  column exists, use `id_from` (a composite of the columns that identify it).

Respond with valid JSON only. No markdown."""

CSV_SCHEMA_USER = """\
Column names: {columns}

Sample rows (first {n} of {total}):
{sample_rows}

Existing ontology types:
{existing_types}

Follow these two worked examples (different domains — generalize the pattern,
do not copy the type names).

EXAMPLE A — a WIDE multi-entity row. Columns: order_id, order_date,
customer_id, customer_email, sku, product_name, qty, ship_country
{{
  "entity_type": "Order",
  "columns": [
    {{"column_name": "order_id", "role": "attribute", "datatype": "string", "attribute_name": "order_id", "entity": "order"}},
    {{"column_name": "order_date", "role": "attribute", "datatype": "datetime", "attribute_name": "order_date", "entity": "order"}},
    {{"column_name": "qty", "role": "attribute", "datatype": "integer", "attribute_name": "qty", "entity": "order"}},
    {{"column_name": "customer_id", "role": "attribute", "datatype": "string", "attribute_name": "customer_id", "entity": "customer"}},
    {{"column_name": "customer_email", "role": "attribute", "datatype": "string", "attribute_name": "email", "entity": "customer"}},
    {{"column_name": "sku", "role": "attribute", "datatype": "string", "attribute_name": "sku", "entity": "product"}},
    {{"column_name": "product_name", "role": "attribute", "datatype": "string", "attribute_name": "name", "entity": "product"}},
    {{"column_name": "ship_country", "role": "relationship", "target_type": "Country", "datatype": "string", "attribute_name": "ship_country", "entity": "order"}}
  ],
  "entities": [
    {{"name": "order", "type_name": "Order", "id_column": "order_id"}},
    {{"name": "customer", "type_name": "Customer", "id_column": "customer_id"}},
    {{"name": "product", "type_name": "Product", "id_column": "sku"}}
  ],
  "relationships": [
    {{"subject": "order", "predicate": "placed_by", "object": "customer"}},
    {{"subject": "order", "predicate": "contains", "object": "product"}}
  ]
}}

EXAMPLE B — a FLAT single-entity row (omit entities/relationships). Columns:
isbn, title, author_name, price, published_date
{{
  "entity_type": "Book",
  "columns": [
    {{"column_name": "isbn", "role": "type_id", "datatype": "string", "attribute_name": "isbn"}},
    {{"column_name": "title", "role": "attribute", "datatype": "string", "attribute_name": "title"}},
    {{"column_name": "author_name", "role": "relationship", "target_type": "Author", "datatype": "string", "attribute_name": "author_name"}},
    {{"column_name": "price", "role": "attribute", "datatype": "float", "attribute_name": "price"}},
    {{"column_name": "published_date", "role": "attribute", "datatype": "datetime", "attribute_name": "published_date"}}
  ]
}}

Now return the JSON for the columns above — tag EVERY column. Use the
multi-entity shape (with `entities`) whenever the row bundles more than one
real-world entity."""


# --- ADR 0003 Pass B (REASON) -----------------------------------------------
# Every rule below is structural — statable without domain nouns (ADR 0003 §4
# litmus test). No keyword lists, no worked examples encoding a domain's
# answer. The post-hoc NAME_HINTS / FORCE_RELATIONSHIP patches of the legacy
# path are intentionally absent from this pipeline.

REASON_SYSTEM = """\
You convert a CSV table into a knowledge-graph schema (entities, attributes, relationships).
You are given a STATISTICAL PROFILE of every column computed over the FULL table. Reason from that
evidence, not from column names. Apply these domain-independent rules:

ENTITY DECOMPOSITION
- Columns that travel together (mutual functional dependency) and include an id-like member describe ONE
  entity; a code column paired with its label/title column are the SAME entity (code = its key, title = its label).
- A column with low card_ratio that repeats across rows (distinct far below row count) is a DIMENSION: a shared
  entity referenced by many rows. Model it as its own entity + an edge, NOT a literal attribute.
- A near-unique, free-text column is a literal attribute of the row's primary entity.

KEYS (row conservation is mandatory)
- An entity key must be a column that is BOTH ~100% complete AND unique.
- If none qualifies, use a composite of identifying columns or a synthetic id. NEVER key on an incomplete
  column: it silently drops every row missing that value.
- A key column must ALSO be emitted as a queryable attribute, not consumed as identity only.

EDGES
- An edge predicate names the RELATIONSHIP (a role/verb) between two entities, never the source column name.

NAMES / LABELS (a name is optional, not mandatory)
- Not every entity has a name. Map a column to a "name"/label attribute ONLY when it is a genuine
  human-identifying proper name of that entity. A reified/measurement or dependent entity (a score,
  rating, price, ranking, or an issued identifier) has NO proper name — do NOT designate a descriptive
  or composite column as its "name", and prefer a composite/synthetic key over keying it on such a
  label. It is identified by its value + the entities it links to.

TYPE REUSE
- The user message lists the tenant's EXISTING ontology types. Reuse one ONLY when your entity is genuinely
  the SAME real-world concept. If none genuinely matches, propose a NEW accurate PascalCase type name —
  NEVER force-fit a different concept onto an available type just because it exists.

Output strict JSON: {"entities":[{"name","type_name","key_strategy":"column|composite|synthetic","key_columns":[...],
"why","confidence"}], "columns":[{"column","role":"attribute|relationship|key","entity","predicate_or_attr","why","confidence"}],
"relationships":[{"subject","predicate","object","why"}]}.
A column with role "relationship" references a shared out-of-row entity that is NOT one of your in-row
entities: its "entity" is the in-row source, "predicate_or_attr" is the edge predicate, and it must also
carry "target_type" (PascalCase type the values name). Prefer promoting dimensions to in-row entities.
Where evidence is ambiguous, lower confidence and state what is unresolved instead of guessing. JSON only."""

REASON_USER = """\
COLUMN PROFILE (computed over {rows_profiled} of {total_rows} rows):
{profile}

SAMPLE ROWS ({n} highest-density rows — value context only; trust the profile for statistics):
{sample_rows}

EXISTING ONTOLOGY TYPES:
{existing_types}

Return the schema JSON now."""


# --- ADR 0003 Pass C (REFUTE) -----------------------------------------------

REFUTE_SYSTEM = """\
You are an adversarial schema reviewer. Given a profile and a proposed schema, TRY TO BREAK it
using only these structural failure templates (no domain knowledge):
1. KEY DROPS ROWS — any entity keyed on a column with completeness < 0.99.
2. DIMENSION AS LITERAL — any column with card_ratio < 0.5 that repeats, modeled as a literal attribute instead of an entity/edge.
3. COLUMN-NAMED EDGE — any relationship predicate equal to a source column name rather than a relation/verb.
4. KEYLESS ENTITY — any entity with no stable key strategy.
5. DUPLICATE/DEAD ATTR — near-duplicate attribute names, or attributes over an all-empty column.
6. LOST KEY — a key column not also emitted as an attribute.
7. SPARSE / MIS-DOMAINED EDGE — a relationship whose coverage on its declared source type is below the support floor (few of that type's rows populate it), OR that reuses a predicate which holds at high coverage on a sibling source type but is attached here at low coverage to a different source type. Either way the edge is not a type-level property of its declared domain.
List every violation as {template, location, evidence, severity}. Then output a CORRECTED schema JSON in the
same shape as the input. If nothing is wrong, return violations:[] and echo the schema. JSON only:
{"violations":[...], "corrected": {...}}"""

REFUTE_USER = """\
COLUMN PROFILE (computed over {rows_profiled} of {total_rows} rows):
{profile}

PROPOSED SCHEMA:
{schema}

Review against the failure templates and return the violations + corrected schema JSON now."""


# --- ADR 0003 Pass D (COMPLETE) ----------------------------------------------
# The validated completion prompt (COG-52). Concept knowledge enters ONLY via
# the three constitutive-slot tests (existence/identity/universality) — no
# domain keyword lists, no domain-noun examples beyond the validated artifact.
# The explicit two-step framing is load-bearing: with promotion phrased as a
# side-note, the model rejected dependent identifiers instead of promoting
# them and attached dataset constants to the wrong slot.

COMPLETE_SYSTEM = """\
You are an ontology completion reviewer for a knowledge graph.
Input: a schema inferred from ONE dataset (types, attributes, relationships) plus the dataset's
column profile. The schema only models what is IN the data. Your job is to make each type
CONCEPTUALLY COHERENT — and nothing more.

Work in TWO STEPS, in order.

STEP 1 — DEPENDENT-ENTITY PROMOTION. Scan every attribute in the schema and ask: does this
value exist only RELATIVE TO some party or context the data does not model? An identifier,
listing, offer, account-number, policy-number, registration etc. issued BY some external party
is not a property of the thing it points to — it is a DEPENDENT ENTITY whose identity includes
its issuer (the same target can carry different identifiers at different issuers; identical
identifier strings at different issuers are different things). Promote such attributes to their
own type. A promoted type's constitutive slots are typically: the issuing party (relationship),
the thing it identifies (relationship), and the identifier string itself (attribute).
The signature that demands promotion: "X is a <party>-specific identifier" — if you find
yourself writing that sentence, promote X; do not merely reject it.

STEP 2 — CORE SLOTS. For each type (including promoted ones), propose its CORE slots:
relationships/attributes that are CONSTITUTIVE of the concept. A slot is core ONLY if it
passes ALL THREE tests:
1. EXISTENCE — an instance of this concept cannot exist in the real world without a value for
   this slot (even when this dataset has no column for it).
2. IDENTITY — the slot is required to individuate instances (two instances differing only here
   are genuinely different things), OR the type is a dependent entity that exists only relative
   to the slot's target (e.g. an identifier issued BY some party exists only relative to the issuer).
3. UNIVERSALITY — holds for every instance of the concept in any dataset or domain.

HARD RULES:
- Max 3 core slots per type. If you list more, you are listing "commonly associated", not
  "constitutive" — cut.
- Every candidate you considered but did not mark core goes in `rejected` with the failed test
  named. Be aggressive about rejecting: category/classification, price, dates, descriptions,
  status etc. are almost never constitutive.
- If the dataset context implies a single constant value for a missing core slot (e.g. the whole
  file is one party's catalog/export), set `dataset_constant` with the implied value and your
  confidence — the pipeline can then materialize ONE instance instead of leaving the slot empty.
  Attach the constant ONLY to the slot whose ROLE matches the party's role in producing this
  dataset (a catalog's publisher is the issuer/offerer of its identifiers — it is NOT the maker
  of the products listed).
- A promoted/dependent or measurement type has NO name of its own: never add a "name"/label core
  slot for it. Its identity is its constitutive slots (the parties it depends on + the identifier or
  value it carries), not a human-readable label.

Output strict JSON:
{"types":[{"type","promoted_from_attribute": null|"<attr>","core_slots":[{"name","kind":"relationship|attribute",
"target_type":null|"<T>","why","tests":{"existence":true,"identity":true,"universality":true},
"dataset_constant":null|{"value","confidence"}}],"rejected":[{"name","failed_test","why"}]}]}
JSON only."""

COMPLETE_USER = """\
COLUMN PROFILE (computed over {rows_profiled} of {total_rows} rows):
{profile}

INFERRED SCHEMA (models only what is IN the data):
{schema}

Apply the two steps and return the completion JSON now."""


def _v2_enabled() -> bool:
    """ADR 0003 feature flag: ``OMNIX_CSV_INFERENCE_V2`` defaults ON; set it
    to 0 (or false/no/off) to run the legacy single-call inference verbatim."""
    return os.environ.get("OMNIX_CSV_INFERENCE_V2", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


# --- COG-58: wide-table schema inference ------------------------------------
# Above this column count the per-column REASON tagging is SPLIT: one global
# entity-decomposition pass (output bounded by entity count) followed by
# chunked column-assignment passes of at most this many columns each (output
# bounded by chunk size). This keeps any single LLM call's output within its
# token budget — the wide-table failure mode is a ~300-column REASON pass whose
# per-column JSON exceeds max_tokens, truncates, fails validation, retries, and
# eventually 422s or hits the 120s route timeout. Narrow tables (<= threshold)
# keep the single-call path unchanged.
MAX_INFERENCE_COLUMNS = int(os.environ.get("OMNIX_CSV_MAX_INFERENCE_COLUMNS", "40"))

# Output-token budget for v2 passes, scaled to the column count. The REFUTE and
# COMPLETE passes still echo the (whole) schema, so even with chunked REASON
# they need headroom proportional to the column count — bounded so we never
# request an absurd budget from the provider.
_V2_BASE_MAX_TOKENS = 4096
_V2_MAX_TOKENS_CAP = 32768
_V2_TOKENS_PER_COLUMN = 80

# Bound on concurrent column-assignment chunk calls in the wide path.
_WIDE_CHUNK_CONCURRENCY = 5


def _v2_max_tokens(n_columns: int) -> int:
    """Per-pass output budget scaled to column count (COG-58)."""
    return min(
        _V2_MAX_TOKENS_CAP,
        max(_V2_BASE_MAX_TOKENS, 1024 + n_columns * _V2_TOKENS_PER_COLUMN),
    )


def _chunked(seq: list, size: int) -> list[list]:
    """Split a list into consecutive chunks of at most ``size``."""
    return [seq[i : i + size] for i in range(0, len(seq), size)]


# --- COG-58 Pass B split: ENTITY (global) -----------------------------------
# Same entity-decomposition rules as REASON_SYSTEM, but the model returns ONLY
# the entity list + inter-entity edges — never a per-column array. Output is
# therefore bounded by the (small) entity count, not the column count, so this
# pass is safe at any table width.

ENTITY_SYSTEM = """\
You decompose a CSV table into the knowledge-graph ENTITIES it describes (not
the columns yet). You are given a STATISTICAL PROFILE of every column computed
over the full table. Reason from that evidence, not from column names.

ENTITY DECOMPOSITION
- Columns that travel together (mutual functional dependency) and include an
  id-like member describe ONE entity; a code column paired with its label/title
  column are the SAME entity (code = its key, title = its label).
- A column with low card_ratio that repeats across rows (distinct far below row
  count) is a DIMENSION: a shared entity referenced by many rows. Model it as
  its own entity, NOT a literal attribute of the row's primary entity.
- Wide/denormalized exports usually bundle SEVERAL distinct entities per row — a
  person, a transaction, a place, an organization, a product. Default to
  multi-entity unless the row genuinely describes ONE thing.
- SAME TYPE TWICE: two column-clusters of the same base type in different roles
  (buyer & seller, patient & provider, applicant & co_applicant) are TWO
  separate entities with distinct names — NEVER merge them.

KEYS (row conservation is mandatory)
- An entity key must be a column that is BOTH ~100% complete AND unique.
- If none qualifies, use a composite of identifying columns or a synthetic id.
  NEVER key on an incomplete column: it silently drops every row missing it.

TYPE REUSE
- The user message lists the tenant's EXISTING ontology types. Reuse one ONLY
  when your entity is genuinely the SAME real-world concept. Otherwise propose a
  NEW accurate PascalCase type name — never force-fit a different concept.

Output strict JSON ONLY (no columns array):
{"entities":[{"name","type_name","key_strategy":"column|composite|synthetic",
"key_columns":[...],"why","confidence"}],
"relationships":[{"subject","predicate","object","why"}]}
An edge predicate names the RELATIONSHIP (a role/verb) between two entity
names, never a source column name. JSON only."""

ENTITY_USER = """\
COLUMN PROFILE (computed over {rows_profiled} of {total_rows} rows):
{profile}

SAMPLE ROWS ({n} highest-density rows — value context only; trust the profile for statistics):
{sample_rows}

EXISTING ONTOLOGY TYPES:
{existing_types}

This table has {n_columns} columns — return ONLY the entity decomposition
(entities + inter-entity relationships). Column assignment happens separately.
Return the JSON now."""


# --- COG-58 Pass B split: COLUMN ASSIGNMENT (chunked) -----------------------
# Given the already-decided entities, assign a BATCH of columns to them. Output
# is bounded by the batch size, so arbitrarily wide tables are handled by
# running this pass once per chunk and merging the column arrays.

COLUMN_ASSIGN_SYSTEM = """\
You assign CSV columns to an ALREADY-DECIDED set of knowledge-graph entities.
The entities (with their types and keys) are fixed — do NOT invent new entities.
For EACH column in the batch, decide:
- role: "attribute" (a literal property), "relationship" (a reference to a
  shared OUT-OF-ROW entity that is NOT one of the decided entities — carry a
  "target_type"), or "key" (an identifying column of its owner entity).
- entity: the NAME of the decided entity this column belongs to.
- predicate_or_attr: a snake_case attribute name, or the edge predicate for a
  relationship (a role/verb, never the raw column name).
Datatypes are derived later from the profile — do not emit them.
Tag EVERY column in the batch exactly once. Output strict JSON ONLY:
{"columns":[{"column","role":"attribute|relationship|key","entity",
"predicate_or_attr","target_type":null|"<T>","why","confidence"}]}
JSON only."""

COLUMN_ASSIGN_USER = """\
DECIDED ENTITIES (fixed — assign each column to one of these by name):
{entities}

COLUMN PROFILE for THIS BATCH (computed over {rows_profiled} of {total_rows} rows):
{profile}

SAMPLE ROWS ({n} highest-density rows — value context only):
{sample_rows}

Assign every column in this batch to one of the decided entities and return the
columns JSON now."""


class CSVResolver:
    # Primary schema-inference model, routed through OpenRouter with the
    # configured fallback. Defaults to the shared primary.
    EXTRACT_MODEL = os.environ.get("OMNIX_EXTRACT_MODEL", PRIMARY_MODEL)
    EXTRACT_PROVIDER = os.environ.get("OMNIX_EXTRACT_PROVIDER", "openrouter")
    # Anthropic-SDK offline fallback (used only when no OpenRouter key is set) —
    # must be a NATIVE Anthropic model id. Env-overridable.
    INFER_MODEL = os.environ.get("OMNIX_INFER_MODEL", "claude-opus-4-8")

    def __init__(self, client: anthropic.AsyncAnthropic, openrouter_key: str = ""):
        self._client = client
        self._openrouter_key = openrouter_key or os.environ.get("OPENROUTER_API_KEY", "")

    async def infer_schema(
        self,
        headers: list[str],
        sample_rows: list[dict[str, str]],
        existing_types: dict[str, str],
        total_rows: int = 0,
    ) -> CSVSchemaMapping:
        """Infer column-to-ontology mapping from sample rows.

        Default (``OMNIX_CSV_INFERENCE_V2`` unset or truthy): the ADR 0003
        evidence-grounded pipeline — deterministic profile (Pass A), REASON
        LLM call (Pass B), adversarial REFUTE LLM call (Pass C), conceptual
        COMPLETE LLM call (Pass D), then conversion to the same
        :class:`CSVSchemaMapping` contract the legacy path returns (extended
        with optional ``key_strategy``/``why``/``confidence``/``violations``/
        ``inference_audit``/``ontology_extensions`` fields).

        ``OMNIX_CSV_INFERENCE_V2=0``: the legacy single-LLM-call path,
        verbatim — including its NAME_HINTS / FORCE_RELATIONSHIP post-hoc
        patches, which the v2 pipeline deliberately retires (ADR 0003 §4).

        Each LLM call keeps the existing retry contract: one retry at
        temperature 0.3 when the response fails validation, then propagate
        (the /ingest/csv/schema route converts that into its 422 guidance).
        """
        if _v2_enabled():
            return await self._infer_schema_v2(headers, sample_rows, existing_types, total_rows)
        return await self._infer_schema_legacy(headers, sample_rows, existing_types, total_rows)

    async def _infer_schema_legacy(
        self,
        headers: list[str],
        sample_rows: list[dict[str, str]],
        existing_types: dict[str, str],
        total_rows: int = 0,
    ) -> CSVSchemaMapping:
        """Legacy single-call inference (pre-ADR 0003), kept verbatim behind
        ``OMNIX_CSV_INFERENCE_V2=0``: one LLM call with one retry at higher
        temperature if the response fails validation."""
        types_str = "\n".join(f"- {name}" for name in existing_types) if existing_types else "(none)"

        # Prefer rows with the most non-empty fields. CSVs whose leading rows
        # are mostly-empty (e.g. `status=deleted` records with only slug+url)
        # otherwise feed the LLM a near-blank sample, which reliably produces
        # malformed JSON keys (observed: `column118 name`).
        ranked_samples = _rank_sample_rows(sample_rows)[:10]
        sample_str = "\n".join(
            json.dumps(row, default=str) for row in ranked_samples
        )

        user_content = CSV_SCHEMA_USER.format(
            columns=", ".join(headers),
            n=len(ranked_samples),
            total=total_rows or len(sample_rows),
            sample_rows=sample_str,
            existing_types=types_str,
        )

        try:
            data = await self._call_llm(user_content, temperature=0.0)
            mapping = self._build_mapping(data)
        except (ValidationError, KeyError, json.JSONDecodeError) as e:
            logger.warning("csv_schema_validation_retry", error=str(e))
            data = await self._call_llm(user_content, temperature=0.3)
            mapping = self._build_mapping(data)

        # In multi-entity mode, ids come from the EntitySpec specs (not a
        # type_id column), so the single-entity type_id enforcement below is
        # skipped. The geographic/entity promotion pass still runs (columns keep
        # their `entity` owner).
        multi = mapping.entities is not None

        # Validate: must have exactly one type_id (single-entity mode only)
        if not multi:
            id_cols = [c for c in mapping.columns if c.role == ColumnRole.TYPE_ID]
            if len(id_cols) != 1:
                logger.warning("csv_schema_no_id", id_cols=len(id_cols))
                # Fallback: use first column as ID
                if mapping.columns:
                    mapping.columns[0].role = ColumnRole.TYPE_ID

        # Post-processing: if the chosen type_id is numeric, prefer a string
        # column with a name-like label (institution, title, name, etc.)
        # Numeric IDs cause deduplication when values repeat.
        id_col = None if multi else next((c for c in mapping.columns if c.role == ColumnRole.TYPE_ID), None)
        if id_col and id_col.datatype in ("integer", "float"):
            NAME_HINTS = {"name", "title", "institution", "series_title", "label", "id"}
            for col in mapping.columns:
                col_key = (col.attribute_name or col.column_name).lower().replace(" ", "_")
                if col_key in NAME_HINTS and col.role != ColumnRole.TYPE_ID:
                    logger.info(
                        "csv_type_id_override",
                        old=id_col.column_name,
                        new=col.column_name,
                        reason="numeric ID replaced with name-like column",
                    )
                    id_col.role = col.role
                    col.role = ColumnRole.TYPE_ID
                    break

        # Post-processing: enforce entity-first for known geographic/entity columns
        # The LLM sometimes ignores the prompt and treats these as string attributes
        FORCE_RELATIONSHIP = {
            # Geographic
            "city": "City",
            "state": "State",
            "country": "Country",
            "region": "Region",
            "zipcode": "ZipCode",
            "zip_code": "ZipCode",
            "zip": "ZipCode",
            "postal_code": "PostalCode",
            "county": "County",
            "district": "District",
            "neighborhood": "Neighborhood",
            "area": "Area",
            # People
            "owner": "Person",
            "agent": "Person",
            "broker": "Person",
            "manager": "Person",
            "seller": "Person",
            "buyer": "Person",
            "author": "Person",
            "creator": "Person",
            # Organizations
            "company": "Company",
            "brokerage": "Company",
            "firm": "Company",
            "agency": "Company",
            "school": "School",
            "university": "University",
        }
        for col in mapping.columns:
            col_key = (col.attribute_name or col.column_name).lower().replace(" ", "_")
            if col.role == ColumnRole.ATTRIBUTE and col_key in FORCE_RELATIONSHIP:
                col.role = ColumnRole.RELATIONSHIP
                col.target_type = FORCE_RELATIONSHIP[col_key]
                col.datatype = "string"
                logger.info("csv_column_promoted", column=col.column_name, target_type=col.target_type)

        logger.info(
            "csv_schema_inferred",
            entity_type=mapping.entity_type,
            columns=len(mapping.columns),
            relationships=sum(1 for c in mapping.columns if c.role == ColumnRole.RELATIONSHIP),
        )
        return mapping

    # --- ADR 0003 v2 pipeline: profile → reason → refute → complete --------

    async def _infer_schema_v2(
        self,
        headers: list[str],
        sample_rows: list[dict[str, str]],
        existing_types: dict[str, str],
        total_rows: int = 0,
    ) -> CSVSchemaMapping:
        """Evidence-grounded inference (ADR 0003 Passes A–D).

        Pass A profiles the provided rows deterministically; Pass B (REASON)
        proposes a schema grounded in that profile with per-decision
        ``why``/``confidence``; Pass C (REFUTE) adversarially checks it
        against the six structural failure templates and corrects it; Pass D
        (COMPLETE) makes each type conceptually coherent — dependent-entity
        promotions plus constitutive core slots under the three hard tests,
        capped at 3 per type. The corrected schema is converted to the
        existing ``CSVSchemaMapping`` contract, with the completion output on
        ``ontology_extensions``. No post-hoc keyword patches run on this path.
        """
        profile = profile_table(headers, sample_rows, total_rows)
        profile_json = json.dumps(profile.to_prompt_dict())
        types_str = "\n".join(f"- {name}" for name in existing_types) if existing_types else "(none)"

        # Same density ranking as the legacy path: the sample exists for value
        # context only — statistics come from the profile.
        ranked_samples = _rank_sample_rows(sample_rows)[:6]
        sample_str = "\n".join(json.dumps(row, default=str) for row in ranked_samples)

        # COG-58: scale each pass's output budget to the column count so the
        # REFUTE/COMPLETE echoes (which still carry the whole schema) aren't
        # truncated on wide tables.
        max_tokens = _v2_max_tokens(len(headers))

        # Pass B — REASON. Narrow tables: one call. Wide tables (COG-58): a
        # global entity-decomposition pass plus chunked column-assignment passes
        # so no single call must emit a per-column tag for every column.
        if len(headers) > MAX_INFERENCE_COLUMNS:
            proposed = await self._reason_wide(
                headers, profile, profile_json, types_str, ranked_samples, sample_str,
            )
        else:
            reason_user = REASON_USER.format(
                rows_profiled=profile.rows_profiled,
                total_rows=profile.total_rows,
                profile=profile_json,
                n=len(ranked_samples),
                sample_rows=sample_str,
                existing_types=types_str,
            )
            # Retry once at temperature 0.3, then propagate.
            try:
                proposed = await self._call_llm_v2(
                    REASON_SYSTEM, reason_user, temperature=0.0, max_tokens=max_tokens,
                )
                _check_reason_shape(proposed)
            except (ValidationError, KeyError, json.JSONDecodeError) as e:
                logger.warning("csv_reason_validation_retry", error=str(e))
                proposed = await self._call_llm_v2(
                    REASON_SYSTEM, reason_user, temperature=0.3, max_tokens=max_tokens,
                )
                _check_reason_shape(proposed)

        refute_user = REFUTE_USER.format(
            rows_profiled=profile.rows_profiled,
            total_rows=profile.total_rows,
            profile=profile_json,
            schema=json.dumps(proposed),
        )

        # Pass C — REFUTE (same retry contract).
        try:
            refuted = await self._call_llm_v2(
                REFUTE_SYSTEM, refute_user, temperature=0.0, max_tokens=max_tokens,
            )
            violations, corrected = _check_refute_shape(refuted, proposed)
        except (ValidationError, KeyError, json.JSONDecodeError) as e:
            logger.warning("csv_refute_validation_retry", error=str(e))
            refuted = await self._call_llm_v2(
                REFUTE_SYSTEM, refute_user, temperature=0.3, max_tokens=max_tokens,
            )
            violations, corrected = _check_refute_shape(refuted, proposed)

        complete_user = COMPLETE_USER.format(
            rows_profiled=profile.rows_profiled,
            total_rows=profile.total_rows,
            profile=profile_json,
            schema=json.dumps(corrected),
        )

        # Pass D — COMPLETE (same retry contract; a >3-core-slot response
        # fails pydantic validation here, triggering the retry).
        try:
            completed = await self._call_llm_v2(
                COMPLETE_SYSTEM, complete_user, temperature=0.0, max_tokens=max_tokens,
            )
            extensions = _check_complete_shape(completed)
        except (ValidationError, KeyError, json.JSONDecodeError) as e:
            logger.warning("csv_complete_validation_retry", error=str(e))
            completed = await self._call_llm_v2(
                COMPLETE_SYSTEM, complete_user, temperature=0.3, max_tokens=max_tokens,
            )
            extensions = _check_complete_shape(completed)

        mapping = self._convert_v2(corrected, violations, profile, extensions)
        logger.info(
            "csv_schema_inferred_v2",
            entities=[e.type_name for e in (mapping.entities or [])],
            columns=len(mapping.columns),
            violations=[v.template for v in mapping.violations],
            promotions=[
                t.type_name for t in extensions.types if t.promoted_from_attribute
            ],
            held_for_review=[
                t.type_name for t in extensions.types
                if t.held_for_review or any(s.held_for_review for s in t.core_slots)
            ],
        )
        return mapping

    async def _reason_wide(
        self,
        headers: list[str],
        profile: TableProfile,
        profile_json: str,
        types_str: str,
        ranked_samples: list[dict],
        sample_str: str,
    ) -> dict:
        """COG-58 wide-table REASON: split the single per-column pass into a
        global entity-decomposition call plus chunked column-assignment calls,
        then merge into the standard ``{entities, columns, relationships}``
        shape that :func:`_check_reason_shape` validates and :meth:`_convert_v2`
        consumes.

        No single call's output scales with the total column count: the entity
        pass emits only the (small) entity list, and each column pass emits tags
        for at most :data:`MAX_INFERENCE_COLUMNS` columns. Coverage is
        guaranteed deterministically — any column the model drops is backfilled
        as an attribute of the first entity, so every header is tagged exactly
        once (row conservation, ADR 0003 §2). The chunk calls run concurrently
        under a small semaphore so a very wide table doesn't fan out unbounded.
        """
        import asyncio

        # --- Pass B1: global entity decomposition (output bounded by entity count).
        entity_user = ENTITY_USER.format(
            rows_profiled=profile.rows_profiled,
            total_rows=profile.total_rows,
            profile=profile_json,
            n=len(ranked_samples),
            sample_rows=sample_str,
            existing_types=types_str,
            n_columns=len(headers),
        )

        def _check_entities(data: dict) -> None:
            if not isinstance(data, dict):
                raise KeyError("entity output must be a JSON object")
            ents = data.get("entities")
            if not isinstance(ents, list) or not ents:
                raise KeyError("entities")
            for e in ents:
                if not isinstance(e, dict) or not e.get("type_name"):
                    raise KeyError("entities[].type_name")

        try:
            decomposition = await self._call_llm_v2(
                ENTITY_SYSTEM, entity_user, temperature=0.0, max_tokens=_V2_BASE_MAX_TOKENS,
            )
            _check_entities(decomposition)
        except (ValidationError, KeyError, json.JSONDecodeError) as e:
            logger.warning("csv_entity_validation_retry", error=str(e))
            decomposition = await self._call_llm_v2(
                ENTITY_SYSTEM, entity_user, temperature=0.3, max_tokens=_V2_BASE_MAX_TOKENS,
            )
            _check_entities(decomposition)

        entities = decomposition["entities"]
        relationships = decomposition.get("relationships") or []
        # Every entity needs a stable name handle for column assignment + convert.
        for e in entities:
            if not e.get("name"):
                e["name"] = _snake_case(e["type_name"])
        entity_names = [e["name"] for e in entities]
        valid_names = set(entity_names)
        default_owner = entity_names[0]

        entities_brief = "\n".join(
            f'- "{e["name"]}" (type {e["type_name"]}, key '
            f'{e.get("key_strategy") or "?"}: '
            f'{", ".join(e.get("key_columns") or []) or "none"})'
            for e in entities
        )

        profile_dict = profile.to_prompt_dict()
        all_col_profiles = profile_dict.get("columns", {})

        # --- Pass B2: chunked column assignment (output bounded by chunk size).
        chunks = _chunked(headers, MAX_INFERENCE_COLUMNS)
        sem = asyncio.Semaphore(_WIDE_CHUNK_CONCURRENCY)

        def _check_cols(data: dict) -> None:
            if not isinstance(data, dict):
                raise KeyError("column output must be a JSON object")
            cols = data.get("columns")
            if not isinstance(cols, list) or not cols:
                raise KeyError("columns")

        async def _assign_chunk(chunk: list[str]) -> list[dict]:
            chunk_profile = {
                "rows_profiled": profile_dict.get("rows_profiled"),
                "total_rows": profile_dict.get("total_rows"),
                "columns": {h: all_col_profiles.get(h, {}) for h in chunk},
            }
            chunk_samples = "\n".join(
                json.dumps({h: row.get(h) for h in chunk}, default=str)
                for row in ranked_samples
            )
            user = COLUMN_ASSIGN_USER.format(
                entities=entities_brief,
                rows_profiled=profile.rows_profiled,
                total_rows=profile.total_rows,
                profile=json.dumps(chunk_profile),
                n=len(ranked_samples),
                sample_rows=chunk_samples,
            )
            tokens = _v2_max_tokens(len(chunk))
            async with sem:
                try:
                    data = await self._call_llm_v2(
                        COLUMN_ASSIGN_SYSTEM, user, temperature=0.0, max_tokens=tokens,
                    )
                    _check_cols(data)
                except (ValidationError, KeyError, json.JSONDecodeError) as e:
                    logger.warning("csv_column_assign_retry", error=str(e), chunk=len(chunk))
                    data = await self._call_llm_v2(
                        COLUMN_ASSIGN_SYSTEM, user, temperature=0.3, max_tokens=tokens,
                    )
                    _check_cols(data)
            return [
                c for c in data.get("columns", [])
                if isinstance(c, dict) and c.get("column")
            ]

        chunk_results = await asyncio.gather(*[_assign_chunk(c) for c in chunks])

        # Merge + coverage repair. First tag per column wins; unknown owners are
        # reassigned to the default entity; any untagged header is backfilled as
        # an attribute so EVERY column is tagged exactly once (row conservation).
        header_set = set(headers)
        columns: list[dict] = []
        seen: set[str] = set()
        for chunk_cols in chunk_results:
            for col in chunk_cols:
                name = col["column"]
                if name in seen or name not in header_set:
                    continue
                if col.get("entity") not in valid_names:
                    col["entity"] = default_owner
                columns.append(col)
                seen.add(name)

        tagged = len(seen)
        for h in headers:
            if h not in seen:
                columns.append({
                    "column": h,
                    "role": "attribute",
                    "entity": default_owner,
                    "predicate_or_attr": _snake_case(h),
                    "why": "backfilled — model did not tag this column",
                    "confidence": 0.3,
                })
                seen.add(h)

        proposed = {
            "entities": entities,
            "columns": columns,
            "relationships": relationships,
        }
        _check_reason_shape(proposed)
        logger.info(
            "csv_reason_wide",
            columns=len(headers),
            chunks=len(chunks),
            entities=[e["type_name"] for e in entities],
            backfilled=len(headers) - tagged,
        )
        return proposed

    def _convert_v2(
        self,
        corrected: dict,
        violations: list[dict],
        profile: TableProfile,
        extensions: OntologyExtensions | None = None,
    ) -> CSVSchemaMapping:
        """Convert a (corrected) Pass B/C schema into the ``CSVSchemaMapping``
        contract consumed by ``apply_mapping`` and the web Explorer.

        - ``key_strategy: "column"`` → ``EntitySpec.id_column`` (first key
          column); ``"composite"`` → ``EntitySpec.id_from``; ``"synthetic"``
          → neither, so ``apply_mapping`` mints deterministic content-hash
          keys per row (COG-51).
        - ``role: "key"`` columns become regular ATTRIBUTE columns owned by
          their entity — identity is carried by the owning ``EntitySpec``,
          and COG-51 guarantees the key value is also emitted as a queryable
          attribute (refute template 6, LOST KEY).
        - ``role: "relationship"`` columns become out-of-row RELATIONSHIP
          columns (dimension target + edge), exactly like the legacy contract.
        - Datatypes are derived deterministically from the Pass A value-shape
          evidence (the v2 schema carries none — the profile already measured
          the values).
        """
        specs: list[EntitySpec] = []
        for ent in corrected.get("entities", []):
            type_name = ent["type_name"]
            name = ent.get("name") or _snake_case(type_name)
            strategy = ent.get("key_strategy")
            key_columns = [c for c in (ent.get("key_columns") or []) if c]
            id_column: str | None = None
            id_from: list[str] | None = None
            if strategy == "column" and key_columns:
                id_column = key_columns[0]
            elif strategy == "composite" and key_columns:
                id_from = key_columns
            elif strategy in ("column", "composite"):
                # Keyed strategy declared with no key columns — degrade to a
                # synthetic key (row conservation) rather than dropping rows.
                logger.warning(
                    "csv_v2_key_strategy_degraded", entity=name, declared=strategy,
                )
                strategy = "synthetic"
            elif strategy != "synthetic":
                # Unknown/missing strategy: infer it from the key columns.
                if len(key_columns) == 1:
                    strategy, id_column = "column", key_columns[0]
                elif key_columns:
                    strategy, id_from = "composite", key_columns
                else:
                    strategy = "synthetic"
            specs.append(EntitySpec(
                name=name,
                type_name=type_name,
                id_column=id_column,
                id_from=id_from,
                key_strategy=strategy,
                confidence=_as_confidence(ent.get("confidence")),
                why=ent.get("why"),
            ))

        spec_names = {s.name for s in specs}
        default_owner = specs[0].name if len(specs) == 1 else None

        columns: list[ColumnMapping] = []
        for col in corrected.get("columns", []):
            column_name = col["column"]
            role = str(col.get("role") or "attribute").strip().lower()
            owner = col.get("entity") or default_owner
            if owner not in spec_names:
                # Repair: a flat (single-entity) schema may omit owners.
                logger.warning(
                    "csv_v2_unowned_column", column=column_name, entity=owner,
                )
                owner = default_owner
            raw_attr = col.get("predicate_or_attr")
            attr_name = _snake_case(raw_attr) if raw_attr else _snake_case(column_name)
            shared = {
                "column_name": column_name,
                "attribute_name": attr_name,
                "entity": owner,
                "confidence": _as_confidence(col.get("confidence")),
                "why": col.get("why"),
            }
            if role == "relationship":
                target = col.get("target_type") or _pascal_case(attr_name)
                columns.append(ColumnMapping(
                    role=ColumnRole.RELATIONSHIP,
                    target_type=target,
                    datatype="string",
                    **shared,
                ))
            else:
                # "key" and "attribute" both land as attribute columns: the
                # identity half of a key column lives on its EntitySpec.
                columns.append(ColumnMapping(
                    role=ColumnRole.ATTRIBUTE,
                    datatype=_datatype_from_profile(profile.column(column_name)),
                    **shared,
                ))

        relationships: list[EntityRelationSpec] = []
        for rel in corrected.get("relationships") or []:
            if not isinstance(rel, dict):
                continue
            subject, predicate, obj = rel.get("subject"), rel.get("predicate"), rel.get("object")
            if not (subject and predicate and obj):
                continue
            if subject not in spec_names or obj not in spec_names:
                logger.warning(
                    "csv_v2_dangling_relationship", subject=subject, object=obj,
                )
                continue
            relationships.append(EntityRelationSpec(
                subject=subject, predicate=predicate, object=obj, why=rel.get("why"),
            ))

        return CSVSchemaMapping(
            # entity_type is ignored in multi-entity mode; keep it meaningful
            # for older readers that only look at the headline type.
            entity_type=specs[0].type_name,
            columns=columns,
            entities=specs,
            relationships=relationships or None,
            violations=[_as_violation(v) for v in violations],
            inference_audit=InferenceAudit(
                pipeline="reason_refute_v2",
                rows_profiled=profile.rows_profiled,
                total_rows=profile.total_rows,
                profile=profile.to_prompt_dict(),
            ),
            ontology_extensions=extensions,
        )

    async def _call_llm(self, user_content: str, temperature: float = 0.0) -> dict:
        if self.EXTRACT_PROVIDER == "openrouter" and self._openrouter_key:
            return await self._infer_via_openrouter(user_content, temperature)
        return await self._infer_via_anthropic(user_content, temperature)

    async def _call_llm_v2(
        self,
        system: str,
        user_content: str,
        temperature: float = 0.0,
        max_tokens: int = _V2_BASE_MAX_TOKENS,
    ) -> dict:
        """LLM seam for the v2 passes — like ``_call_llm`` but the system
        prompt varies per pass (REASON vs REFUTE). ``max_tokens`` is scaled to
        the column count by callers (COG-58) so a wide-table pass that must
        echo every column isn't truncated. Tests monkeypatch this."""
        if self.EXTRACT_PROVIDER == "openrouter" and self._openrouter_key:
            return await self._chat_openrouter(system, user_content, temperature, max_tokens=max_tokens)
        return await self._chat_anthropic(system, user_content, temperature, max_tokens=max_tokens)

    def _build_mapping(self, data: dict) -> CSVSchemaMapping:
        # Gemini Flash occasionally emits `datatype: null` for a column it
        # can't classify. Coerce to "string" so the pydantic model doesn't
        # reject the whole inference — callers can always retry the
        # downstream resolver pass if the string guess turns out wrong.
        for col in data.get("columns", []):
            if col.get("datatype") is None:
                col["datatype"] = "string"
            if col.get("role") is None:
                col["role"] = "attribute"
        # Multi-entity mode is opt-in: the model returns a non-empty `entities`
        # array only for wide CSVs that bundle several entities. Absent → legacy.
        entities = data.get("entities") or None
        relationships = data.get("relationships") or None
        # `entity_type` is required in single-entity mode — its absence signals a
        # malformed LLM response and (by raising KeyError) triggers the retry. In
        # multi-entity mode it's ignored, so a placeholder is fine.
        entity_type = data.get("entity_type")
        if entity_type is None:
            if entities is None:
                raise KeyError("entity_type")
            entity_type = "Entity"
        return CSVSchemaMapping(
            entity_type=entity_type,
            columns=[ColumnMapping(**col) for col in data["columns"]],
            entities=[EntitySpec(**e) for e in entities] if entities else None,
            relationships=(
                [EntityRelationSpec(**r) for r in relationships] if relationships else None
            ),
        )

    async def _infer_via_openrouter(self, user_content: str, temperature: float = 0.0) -> dict:
        return await self._chat_openrouter(CSV_SCHEMA_SYSTEM, user_content, temperature)

    async def _chat_openrouter(
        self,
        system: str,
        user_content: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
        text = await openrouter_chat(
            self._openrouter_key,
            system,
            user_content,
            model=self.EXTRACT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=60,
        )
        return json.loads(_strip_code_fences(text))

    async def _chat_anthropic(
        self,
        system: str,
        user_content: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict:
        """Anthropic fallback for the v2 passes: free-form JSON (the pass
        output shapes differ, so no fixed output_config schema here)."""
        msg = await self._client.messages.create(
            model=self.INFER_MODEL,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )
        return json.loads(_strip_code_fences(msg.content[0].text))

    async def _infer_via_anthropic(self, user_content: str, temperature: float = 0.0) -> dict:
        msg = await self._client.messages.create(
            model=self.INFER_MODEL,
            max_tokens=2048,
            temperature=temperature,
            system=CSV_SCHEMA_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "entity_type": {"type": "string"},
                            "columns": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "column_name": {"type": "string"},
                                        "role": {"type": "string", "enum": ["type_id", "attribute", "relationship"]},
                                        "target_type": {"type": ["string", "null"]},
                                        "datatype": {"type": "string"},
                                        "attribute_name": {"type": ["string", "null"]},
                                        "entity": {"type": ["string", "null"]},
                                    },
                                    "required": ["column_name", "role", "datatype"],
                                    "additionalProperties": False,
                                },
                            },
                            "entities": {
                                "type": ["array", "null"],
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "type_name": {"type": "string"},
                                        "id_column": {"type": ["string", "null"]},
                                        "id_from": {"type": ["array", "null"], "items": {"type": "string"}},
                                    },
                                    "required": ["name", "type_name"],
                                    "additionalProperties": False,
                                },
                            },
                            "relationships": {
                                "type": ["array", "null"],
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "subject": {"type": "string"},
                                        "predicate": {"type": "string"},
                                        "object": {"type": "string"},
                                    },
                                    "required": ["subject", "predicate", "object"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["entity_type", "columns"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        return json.loads(msg.content[0].text)

    @staticmethod
    def apply_mapping(
        mapping: CSVSchemaMapping,
        rows: list[dict[str, str]],
    ) -> AppliedMapping:
        """Deterministically convert all CSV rows to entities + relationships. No LLM.

        Returns an :class:`AppliedMapping`, which unpacks as the legacy
        ``(entities, relationships)`` tuple and additionally carries
        row-conservation accounting (ADR 0003 §2): input rows are never
        silently dropped — an empty natural key falls back to a deterministic
        content-hash synthetic key, and a row is skipped (and counted) only
        when every owned value is empty.

        ``mapping.ontology_extensions`` (ADR 0003 Pass D, COG-52) is consumed
        here too: a promoted attribute's values become instances of the
        promoted type with identifies-edges back to their owner entity, and a
        relationship core slot carrying a dataset constant materializes ONE
        instance of its target type plus per-instance edges. held_for_review
        items are NOT filtered — the confirm gate is client-side (whatever
        mapping the client posts to /ingest/csv/rows is applied).
        """
        # Multi-entity mode: one row expands into several fully-attributed,
        # linked entities. Legacy single-entity path below is untouched.
        if mapping.entities:
            return CSVResolver._apply_multi_entity(mapping, rows)

        id_col = next((c for c in mapping.columns if c.role == ColumnRole.TYPE_ID), None)
        if not id_col:
            # Degenerate mapping (no key column at all): nothing can be minted.
            # Account for every row so the mismatch is loud, not silent.
            drops = {mapping.entity_type: len(rows)} if rows else {}
            if rows:
                logger.warning(
                    "csv_rows_dropped",
                    rows_in=len(rows),
                    rows_dropped=len(rows),
                    drops_by_entity=drops,
                    reason="mapping has no type_id column",
                )
            return AppliedMapping(
                [], [], rows_in=len(rows), rows_dropped=len(rows), drops_by_entity=drops,
            )

        entities: list[ExtractedEntity] = []
        relationships: list[ExtractedRelationship] = []
        seen_rel_entities: dict[str, str] = {}  # safe_id → type for relationship targets
        rel_entity_names: dict[str, str] = {}  # safe_id → original value for name attr
        rows_dropped = 0
        drops_by_entity: dict[str, int] = {}
        # ADR 0003 Pass D: promoted-type instances + dataset-constant edges.
        applier = _ExtensionApplier(mapping)

        for row in rows:
            # Owned non-empty values, keyed by column name — feed both the
            # synthetic-key fallback and the nothing-to-assert skip. Emptiness
            # matches the attribute loop below (strip-if-str, falsy = empty).
            owned_values: dict[str, str] = {}
            for col in mapping.columns:
                raw = row.get(col.column_name, "")
                if isinstance(raw, str):
                    raw = raw.strip()
                if raw:
                    owned_values[col.column_name] = raw if isinstance(raw, str) else str(raw)

            entity_id = _cell(row, id_col.column_name)
            if entity_id:
                safe_id = _safe_id(entity_id)
            elif owned_values:
                # ADR 0003 §2: never silently drop a row. Empty natural key
                # with values to assert → deterministic content-hash key.
                safe_id = _synthetic_key(mapping.entity_type, owned_values)
            else:
                # Empty key AND every owned value empty: nothing to assert.
                # Principled skip — counted and logged, never silent.
                rows_dropped += 1
                drops_by_entity[mapping.entity_type] = (
                    drops_by_entity.get(mapping.entity_type, 0) + 1
                )
                continue

            attrs: list[ExtractedAttribute] = []
            entity_rels: list[ExtractedRelationship] = []

            for col in mapping.columns:
                raw_value = row.get(col.column_name, "")
                if isinstance(raw_value, str):
                    raw_value = raw_value.strip()
                if not raw_value:
                    continue

                attr_name = col.attribute_name or _snake_case(col.column_name)

                if col.role == ColumnRole.TYPE_ID:
                    # The key is URI + label material AND a regular attribute.
                    # Consuming it as URI-only made the key unqueryable
                    # (ADR 0003 §2 — "key consumed, not kept").
                    attrs.append(ExtractedAttribute(
                        name=attr_name,
                        value=raw_value if isinstance(raw_value, str) else str(raw_value),
                        datatype=col.datatype,
                    ))

                # Handle JSON arrays, pipe-delimited, and comma-delimited strings
                # by expanding into multiple values for relationships
                elif col.role == ColumnRole.RELATIONSHIP and col.target_type:
                    values: list[str] = []
                    if isinstance(raw_value, list):
                        values = [v.strip() for v in raw_value if isinstance(v, str) and v.strip()]
                    elif "|" in raw_value:
                        values = [v.strip() for v in raw_value.split("|") if v.strip()]
                    elif ", " in raw_value:
                        # Comma-delimited: split if parts are short (not addresses)
                        parts = [v.strip() for v in raw_value.split(", ") if v.strip()]
                        if all(len(p) < 30 for p in parts) and len(parts) >= 2:
                            values = parts
                        else:
                            values = [raw_value]
                    else:
                        values = [raw_value]

                    for value in values:
                        target_id = _safe_id(value)
                        entity_rels.append(ExtractedRelationship(
                            source_id=safe_id,
                            predicate=attr_name,
                            target_id=target_id,
                        ))
                        if target_id not in seen_rel_entities:
                            seen_rel_entities[target_id] = col.target_type
                            rel_entity_names[target_id] = value

                elif col.role == ColumnRole.ATTRIBUTE:
                    value = str(raw_value) if not isinstance(raw_value, str) else raw_value
                    # Split pipe-delimited attribute values into multiple triples.
                    # "PHASE1|PHASE2" becomes two separate attribute triples so that
                    # exact-match SPARQL filters work without CONTAINS.
                    if "|" in value and col.datatype == "string":
                        for v in value.split("|"):
                            v = v.strip()
                            if v:
                                attrs.append(ExtractedAttribute(
                                    name=attr_name,
                                    value=v,
                                    datatype=col.datatype,
                                ))
                    else:
                        attrs.append(ExtractedAttribute(
                            name=attr_name,
                            value=value,
                            datatype=col.datatype,
                        ))

            entities.append(ExtractedEntity(
                type_name=mapping.entity_type,
                id=safe_id,
                attributes=attrs,
            ))
            relationships.extend(entity_rels)

            if applier.active:
                # The single main entity is the only owner handle (None).
                applier.process_row(row, {None: safe_id})

        # Create stub entities for relationship targets (so they exist in the graph)
        for target_id, target_type in seen_rel_entities.items():
            entities.append(ExtractedEntity(
                type_name=target_type,
                id=target_id,
                attributes=[ExtractedAttribute(name="name", value=rel_entity_names.get(target_id, target_id.replace("_", " ")), datatype="string")],
            ))

        # ADR 0003 Pass D: merge materialized extension instances + edges.
        entities.extend(applier.entities)
        relationships.extend(applier.relationships)

        if rows_dropped:
            logger.warning(
                "csv_rows_dropped",
                rows_in=len(rows),
                rows_dropped=rows_dropped,
                drops_by_entity=drops_by_entity,
                reason="all owned values empty (nothing to assert)",
            )
        return AppliedMapping(
            entities,
            relationships,
            rows_in=len(rows),
            rows_dropped=rows_dropped,
            drops_by_entity=drops_by_entity,
        )

    @staticmethod
    def _entity_key(spec, row: dict) -> str | None:
        """Deterministic key for one in-row entity: its id_column value, or a
        composite of id_from columns. None when the key resolves empty."""
        if spec.id_column:
            v = (row.get(spec.id_column) or "").strip()
            return _safe_id(v) if v else None
        if spec.id_from:
            parts = [(row.get(c) or "").strip() for c in spec.id_from]
            if not any(parts):
                return None
            return _safe_id("|".join(parts))
        return None

    @staticmethod
    def _apply_multi_entity(
        mapping: CSVSchemaMapping,
        rows: list[dict[str, str]],
    ) -> AppliedMapping:
        """Multi-entity mode: one row → several fully-attributed, linked entities.

        Each `EntitySpec` is keyed by its id_column or an id_from composite.
        Columns route to their owner entity (`ColumnMapping.entity`). Inter-entity
        relationships reference the same deterministic ids the entities are minted
        from, so edges resolve to real URIs (not stubs). Entities dedup across
        rows by (type, id) with attribute union — collapsing repeated keys (e.g.
        many reservations → 5 Properties) into one entity. ER fires per
        ER-enabled type downstream (schema_resolver); nothing ER-specific here.

        Row conservation (ADR 0003 §2): an entity whose natural key resolves
        empty gets a deterministic content-hash synthetic key from its owned
        non-empty values; it is skipped (and counted in `drops_by_entity`)
        only when ALL of its owned values are empty. A row counts in
        `rows_dropped` only when it minted no entity at all.

        Ontology extensions (ADR 0003 Pass D): promotions and dataset
        constants are materialized per row against the entity keys minted
        above — see :class:`_ExtensionApplier`.
        """
        specs = {e.name: e for e in (mapping.entities or [])}

        # Route columns to their owner entity; drop (and log) unowned columns.
        cols_by_entity: dict[str, list[ColumnMapping]] = {name: [] for name in specs}
        for col in mapping.columns:
            if col.role == ColumnRole.TYPE_ID:
                continue  # in multi-entity mode, ids come from EntitySpec
            owner = col.entity
            if owner is None or owner not in specs:
                logger.warning(
                    "csv_multi_unowned_column", column=col.column_name, entity=owner,
                )
                continue
            cols_by_entity[owner].append(col)

        entities_by_key: dict[tuple[str, str], ExtractedEntity] = {}
        relationships: list[ExtractedRelationship] = []

        def add_entity(type_name: str, key: str, attrs: list[ExtractedAttribute]) -> None:
            ekey = (type_name, key)
            ent = entities_by_key.get(ekey)
            if ent is None:
                entities_by_key[ekey] = ExtractedEntity(
                    type_name=type_name, id=key, attributes=list(attrs),
                )
                return
            seen = {(a.name, a.value) for a in ent.attributes}
            for a in attrs:
                if (a.name, a.value) not in seen:
                    ent.attributes.append(a)
                    seen.add((a.name, a.value))

        rows_dropped = 0
        drops_by_entity: dict[str, int] = {}
        # ADR 0003 Pass D: promoted-type instances + dataset-constant edges.
        applier = _ExtensionApplier(mapping)

        for row in rows:
            row_ids: dict[str, str] = {}
            for name, spec in specs.items():
                # Owned non-empty values for this entity: its key column(s)
                # plus the columns routed to it — never another entity's
                # columns (unowned data must not leak into the key hash).
                owned_values: dict[str, str] = {}
                key_columns = ([spec.id_column] if spec.id_column else []) + list(spec.id_from or [])
                for column in key_columns:
                    v = _cell(row, column)
                    if v:
                        owned_values[column] = v
                for col in cols_by_entity[name]:
                    raw = row.get(col.column_name, "")
                    if isinstance(raw, str):
                        raw = raw.strip()
                    if raw:
                        owned_values[col.column_name] = raw if isinstance(raw, str) else str(raw)

                key = CSVResolver._entity_key(spec, row)
                if key is None:
                    if not owned_values:
                        # Empty key AND every owned value empty: nothing to
                        # assert. Principled skip — counted, never silent.
                        drops_by_entity[name] = drops_by_entity.get(name, 0) + 1
                        continue
                    # ADR 0003 §2: never silently drop an entity. Empty natural
                    # key with values to assert → deterministic content-hash key.
                    key = _synthetic_key(spec.type_name, owned_values)
                row_ids[name] = key
                attrs: list[ExtractedAttribute] = []
                # The key column's value is also a regular attribute, not just
                # URI + label material (ADR 0003 §2 — "key consumed, not
                # kept"). When the id_column is routed to this entity it flows
                # through the column loop below under its mapped name;
                # otherwise emit it here.
                if spec.id_column and not any(
                    c.column_name == spec.id_column for c in cols_by_entity[name]
                ):
                    key_value = _cell(row, spec.id_column)
                    if key_value:
                        key_col = next(
                            (c for c in mapping.columns if c.column_name == spec.id_column),
                            None,
                        )
                        attrs.append(ExtractedAttribute(
                            name=(
                                key_col.attribute_name
                                if key_col and key_col.attribute_name
                                else _snake_case(spec.id_column)
                            ),
                            value=key_value,
                            datatype=key_col.datatype if key_col else "string",
                        ))
                for col in cols_by_entity[name]:
                    raw = row.get(col.column_name, "")
                    if isinstance(raw, str):
                        raw = raw.strip()
                    if not raw:
                        continue
                    attr_name = col.attribute_name or _snake_case(col.column_name)
                    if col.role == ColumnRole.RELATIONSHIP and col.target_type:
                        # Out-of-row reference (e.g. country) → stub target + edge.
                        for value in _rel_values(raw):
                            tid = _safe_id(value)
                            relationships.append(ExtractedRelationship(
                                source_id=key, predicate=attr_name, target_id=tid,
                            ))
                            add_entity(col.target_type, tid, [ExtractedAttribute(
                                name="name", value=value, datatype="string",
                            )])
                    elif col.role == ColumnRole.ATTRIBUTE:
                        value = str(raw)
                        if "|" in value and col.datatype == "string":
                            for v in value.split("|"):
                                v = v.strip()
                                if v:
                                    attrs.append(ExtractedAttribute(
                                        name=attr_name, value=v, datatype=col.datatype,
                                    ))
                        else:
                            attrs.append(ExtractedAttribute(
                                name=attr_name, value=value, datatype=col.datatype,
                            ))
                add_entity(spec.type_name, key, attrs)

            if specs and not row_ids:
                # The whole row minted nothing — every entity was all-empty.
                rows_dropped += 1

            # Inter-entity edges — only when both endpoints exist this row.
            for rel in (mapping.relationships or []):
                s = row_ids.get(rel.subject)
                o = row_ids.get(rel.object)
                if s and o:
                    relationships.append(ExtractedRelationship(
                        source_id=s, predicate=rel.predicate, target_id=o,
                    ))

            if applier.active:
                applier.process_row(row, row_ids)

        if rows_dropped:
            logger.warning(
                "csv_rows_dropped",
                rows_in=len(rows),
                rows_dropped=rows_dropped,
                drops_by_entity=drops_by_entity,
                reason="all owned values empty (nothing to assert)",
            )
        elif drops_by_entity:
            logger.warning(
                "csv_entities_skipped",
                rows_in=len(rows),
                drops_by_entity=drops_by_entity,
                reason="all owned values empty (nothing to assert)",
            )
        return AppliedMapping(
            # ADR 0003 Pass D: materialized extension instances merge in last.
            list(entities_by_key.values()) + applier.entities,
            relationships + applier.relationships,
            rows_in=len(rows),
            rows_dropped=rows_dropped,
            drops_by_entity=drops_by_entity,
        )


# --- ADR 0003 Pass D: consuming ontology_extensions in apply_mapping --------


@dataclass
class _ConstantEdgePlan:
    """One dataset constant to materialize: ONE instance of ``target_type``
    labelled ``value``, plus a ``predicate`` edge from every source instance
    (ADR 0003 §3 — the slot is filled by the dataset's single implied party,
    not left empty)."""

    predicate: str
    target_type: str
    value: str


@dataclass
class _PromotionPlan:
    """How one dependent-entity promotion is applied per row: the source
    column's value mints an instance of ``type_name`` carrying the id string
    as ``id_attr``, linked to its owner entity via ``identifies_predicate``,
    plus any dataset-constant edges (e.g. the issuer)."""

    type_name: str
    source_column: str
    #: EntitySpec.name owning the source column in multi-entity mode;
    #: None = the single main entity (legacy single-entity mode).
    owner: str | None
    id_attr: str
    identifies_predicate: str
    constants: list[_ConstantEdgePlan]


@dataclass
class _TypeConstantPlan:
    """Dataset constants attached to a NON-promoted type already in the
    mapping: per-row edges from that type's instances to the one
    materialized constant instance."""

    owner: str | None  # EntitySpec.name (multi) / None (single main entity)
    constants: list[_ConstantEdgePlan]


def _find_source_column(mapping: CSVSchemaMapping, attr: str) -> ColumnMapping | None:
    """Locate the column a promoted attribute came from. The completion pass
    names the schema attribute (e.g. the ``predicate_or_attr``), which may
    differ from the raw header — match on attribute name first, then on the
    (normalized) column name."""
    want = _snake_case(attr)
    for col in mapping.columns:
        if col.attribute_name and _snake_case(col.attribute_name) == want:
            return col
    for col in mapping.columns:
        if _snake_case(col.column_name) == want:
            return col
    return None


def _build_extension_plans(
    mapping: CSVSchemaMapping,
) -> tuple[list[_PromotionPlan], list[_TypeConstantPlan]]:
    """Compile ``mapping.ontology_extensions`` into per-row application plans.

    ``held_for_review`` is deliberately NOT filtered here: the confirm gate
    is client-side (`/ingest/csv/schema` flags held items; whatever the
    client posts back to `/ingest/csv/rows` is applied as-is — COG-56 adds
    judge-panel gating). Extensions that cannot be grounded in the mapping
    (unknown source attribute / type) are skipped with a structured warning,
    never an error — they still pre-register in the ontology at ingest.
    """
    ext = mapping.ontology_extensions
    if ext is None or not ext.types:
        return [], []
    multi = bool(mapping.entities)
    specs_by_name = {s.name: s for s in (mapping.entities or [])}

    promotions: list[_PromotionPlan] = []
    type_constants: list[_TypeConstantPlan] = []
    for t in ext.types:
        constants = [
            _ConstantEdgePlan(
                predicate=_snake_case(s.name),
                target_type=s.target_type or _pascal_case(s.name),
                value=s.dataset_constant.value,
            )
            for s in t.core_slots
            if s.kind == "relationship" and s.dataset_constant and s.dataset_constant.value
        ]
        if t.promoted_from_attribute:
            col = _find_source_column(mapping, t.promoted_from_attribute)
            if col is None:
                logger.warning(
                    "csv_extension_source_column_missing",
                    type=t.type_name, attribute=t.promoted_from_attribute,
                )
                continue
            owner = col.entity if (multi and col.entity in specs_by_name) else None
            if multi and owner is None:
                logger.warning(
                    "csv_extension_unowned_source_column",
                    type=t.type_name, column=col.column_name,
                )
            owner_type = (
                specs_by_name[owner].type_name if owner else mapping.entity_type
            )
            id_attr = next(
                (_snake_case(s.name) for s in t.core_slots if s.kind == "attribute"),
                _snake_case(t.promoted_from_attribute),
            )
            identifies = next(
                (
                    _snake_case(s.name) for s in t.core_slots
                    if s.kind == "relationship"
                    and s.target_type == owner_type
                    and not s.dataset_constant
                ),
                "identifies",
            )
            promotions.append(_PromotionPlan(
                type_name=t.type_name,
                source_column=col.column_name,
                owner=owner,
                id_attr=id_attr,
                identifies_predicate=identifies,
                constants=constants,
            ))
        elif constants:
            if multi:
                owners: list[str | None] = [
                    s.name for s in (mapping.entities or [])
                    if s.type_name == t.type_name
                ]
            else:
                owners = [None] if t.type_name == mapping.entity_type else []
            if not owners:
                # A type with no instance source (e.g. a zero-instance issuer
                # type the completion invented): nothing to materialize here —
                # it still exists in the ontology via ingest pre-registration.
                continue
            for owner in owners:
                type_constants.append(_TypeConstantPlan(owner=owner, constants=constants))
    return promotions, type_constants


class _ExtensionApplier:
    """Materializes ontology extensions while ``apply_mapping`` walks the
    rows. Both mapping paths feed it one call per row with the row's resolved
    owner keys; it accumulates its own (deduplicated) entities and edges,
    merged into the result afterwards.

    Determinism mirrors the rest of ``apply_mapping``: instance ids derive
    from cell values only, edges dedup on (source, predicate, target), and a
    dataset constant becomes exactly ONE instance no matter how many rows
    reference it.
    """

    def __init__(self, mapping: CSVSchemaMapping):
        self._promotions, self._type_constants = _build_extension_plans(mapping)
        self._entities: dict[tuple[str, str], ExtractedEntity] = {}
        self._relationships: list[ExtractedRelationship] = []
        self._seen_edges: set[tuple[str, str, str]] = set()

    @property
    def active(self) -> bool:
        return bool(self._promotions or self._type_constants)

    @property
    def entities(self) -> list[ExtractedEntity]:
        return list(self._entities.values())

    @property
    def relationships(self) -> list[ExtractedRelationship]:
        return list(self._relationships)

    def process_row(self, row: dict, owner_keys: dict[str | None, str]) -> None:
        """Apply every plan to one row. ``owner_keys`` maps an owner handle
        (EntitySpec.name, or None for the single main entity) to the entity
        key minted for this row — absent when the owner was skipped, in which
        case the promoted instance is still minted but carries no
        identifies edge (nothing to point at)."""
        for plan in self._promotions:
            value = _cell(row, plan.source_column)
            if not value:
                continue
            pid = _safe_id(value)
            self._ensure_entity(
                plan.type_name, pid,
                ExtractedAttribute(name=plan.id_attr, value=value, datatype="string"),
            )
            owner_key = owner_keys.get(plan.owner)
            if owner_key:
                self._edge(pid, plan.identifies_predicate, owner_key)
            self._constant_edges(pid, plan.constants)
        for type_plan in self._type_constants:
            owner_key = owner_keys.get(type_plan.owner)
            if owner_key:
                self._constant_edges(owner_key, type_plan.constants)

    def _constant_edges(self, source_id: str, constants: list[_ConstantEdgePlan]) -> None:
        for c in constants:
            cid = _safe_id(c.value)
            self._ensure_entity(
                c.target_type, cid,
                ExtractedAttribute(name="name", value=c.value, datatype="string"),
            )
            self._edge(source_id, c.predicate, cid)

    def _ensure_entity(self, type_name: str, key: str, attr: ExtractedAttribute) -> None:
        ent = self._entities.get((type_name, key))
        if ent is None:
            self._entities[(type_name, key)] = ExtractedEntity(
                type_name=type_name, id=key, attributes=[attr],
            )
        elif not any(a.name == attr.name and a.value == attr.value for a in ent.attributes):
            ent.attributes.append(attr)

    def _edge(self, source_id: str, predicate: str, target_id: str) -> None:
        edge = (source_id, predicate, target_id)
        if edge in self._seen_edges:
            return
        self._seen_edges.add(edge)
        self._relationships.append(ExtractedRelationship(
            source_id=source_id, predicate=predicate, target_id=target_id,
        ))


def _rank_sample_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort by descending non-empty field count; stable on ties (preserves
    original order). Used so the LLM gets the most informative rows when the
    head of the CSV is sparse (deleted/empty records). Does not mutate input."""
    def score(row: dict) -> int:
        return sum(
            1 for v in row.values()
            if v is not None and (not isinstance(v, str) or v.strip() != "")
        )
    indexed = list(enumerate(rows))
    indexed.sort(key=lambda t: (-score(t[1]), t[0]))
    return [r for _, r in indexed]


# --- ADR 0003 v2 helpers -----------------------------------------------------


def _strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in markdown fences despite 'JSON only'."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [l for l in stripped.split("\n") if not l.strip().startswith("```")]
        stripped = "\n".join(lines)
    return stripped


def _check_reason_shape(data: dict) -> None:
    """Reject degenerate Pass B (or corrected Pass C) output by raising
    KeyError, which feeds the existing retry-then-422 contract. Only the
    load-bearing structure is enforced; optional fields (why, confidence,
    relationships) may be absent."""
    if not isinstance(data, dict):
        raise KeyError("schema must be a JSON object")
    entities = data.get("entities")
    if not isinstance(entities, list) or not entities:
        raise KeyError("entities")
    for ent in entities:
        if not isinstance(ent, dict) or not ent.get("type_name"):
            raise KeyError("entities[].type_name")
    columns = data.get("columns")
    if not isinstance(columns, list) or not columns:
        raise KeyError("columns")
    for col in columns:
        if not isinstance(col, dict) or not col.get("column"):
            raise KeyError("columns[].column")


def _check_refute_shape(data: dict, proposed: dict) -> tuple[list[dict], dict]:
    """Validate Pass C output; returns ``(violations, corrected)``.

    The reviewer must echo the schema when nothing is wrong — but a model
    that returns ``violations: []`` without the echo is repaired (the
    proposed schema stands). Violations without a corrected schema are
    degenerate output → KeyError → retry."""
    if not isinstance(data, dict):
        raise KeyError("refute output must be a JSON object")
    violations = data.get("violations")
    if not isinstance(violations, list):
        raise KeyError("violations")
    violations = [v for v in violations if isinstance(v, dict)]
    corrected = data.get("corrected")
    if not isinstance(corrected, dict) or not corrected:
        if violations:
            raise KeyError("corrected")
        corrected = proposed
    _check_reason_shape(corrected)
    return violations, corrected


#: Below this confidence a completion item is held for client-side review
#: (COG-52 wiring note 4); ALL promotions are held regardless of confidence.
COMPLETION_REVIEW_THRESHOLD = 0.7


def _check_complete_shape(data: dict) -> OntologyExtensions:
    """Validate Pass D (COMPLETE) output and convert it to
    :class:`OntologyExtensions`, computing ``held_for_review`` flags.

    Degenerate output raises ``KeyError`` (feeding the retry-then-422
    contract); a type with more than 3 core slots raises pydantic
    ``ValidationError`` from the ``max_length=3`` cap — the boundedness rule
    is enforced structurally, not just prompted. An empty ``types`` list is
    accepted (nothing to extend), but the key itself must be present.

    Held-for-review marking (client-side confirm gate — see the model
    docstrings): every promotion is held; a type or slot with confidence
    below :data:`COMPLETION_REVIEW_THRESHOLD` is held; a dataset constant
    without a usable confidence is held (the prompt mandates one).
    """
    if not isinstance(data, dict):
        raise KeyError("completion output must be a JSON object")
    raw_types = data.get("types")
    if not isinstance(raw_types, list):
        raise KeyError("types")

    parsed: list[TypeExtension] = []
    for t in raw_types:
        if not isinstance(t, dict):
            raise KeyError("types[] entries must be objects")
        type_name = t.get("type") or t.get("type_name")
        if not type_name:
            raise KeyError("types[].type")

        core_slots: list[CoreSlot] = []
        for s in t.get("core_slots") or []:
            if not isinstance(s, dict) or not s.get("name"):
                raise KeyError("core_slots[].name")
            constant: DatasetConstant | None = None
            dc = s.get("dataset_constant")
            if isinstance(dc, dict) and dc.get("value") is not None:
                constant = DatasetConstant(
                    value=str(dc["value"]),
                    confidence=_as_confidence(dc.get("confidence")),
                )
            tests = s.get("tests")
            slot_confidence = _as_confidence(s.get("confidence"))
            held = (
                (slot_confidence is not None and slot_confidence < COMPLETION_REVIEW_THRESHOLD)
                or (constant is not None and (
                    constant.confidence is None
                    or constant.confidence < COMPLETION_REVIEW_THRESHOLD
                ))
            )
            kind = str(s.get("kind") or "attribute").strip().lower()
            core_slots.append(CoreSlot(
                name=str(s["name"]),
                kind="relationship" if kind == "relationship" else "attribute",
                target_type=s.get("target_type") or None,
                why=s.get("why"),
                tests=CoreSlotTests(
                    existence=bool(tests.get("existence")),
                    identity=bool(tests.get("identity")),
                    universality=bool(tests.get("universality")),
                ) if isinstance(tests, dict) else None,
                dataset_constant=constant,
                confidence=slot_confidence,
                held_for_review=held,
            ))

        rejected = [
            RejectedSlot(
                name=str(r.get("name")),
                failed_test=str(r.get("failed_test") or ""),
                why=r.get("why"),
            )
            for r in (t.get("rejected") or [])
            if isinstance(r, dict) and r.get("name")
        ]

        promoted = t.get("promoted_from_attribute") or None
        type_confidence = _as_confidence(t.get("confidence"))
        parsed.append(TypeExtension(
            type_name=str(type_name),
            promoted_from_attribute=promoted,
            core_slots=core_slots,  # >3 → ValidationError (boundedness cap)
            rejected=rejected,
            confidence=type_confidence,
            held_for_review=bool(promoted) or (
                type_confidence is not None
                and type_confidence < COMPLETION_REVIEW_THRESHOLD
            ),
        ))
    return OntologyExtensions(types=parsed)


def _as_violation(v: dict) -> SchemaViolation:
    """Lenient parse of one refute violation entry."""
    return SchemaViolation(
        template=str(v.get("template") or ""),
        location=str(v.get("location") or ""),
        evidence=str(v.get("evidence") or ""),
        severity=str(v.get("severity") or "warning"),
    )


def _as_confidence(value) -> float | None:
    """Coerce a model-emitted confidence to a clamped float (None if junk)."""
    try:
        if value is None:
            return None
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return None


def _is_int(v: str) -> bool:
    try:
        int(v)
    except ValueError:
        return False
    return True


def _datatype_from_profile(col: ColumnProfile | None) -> str:
    """Deterministic datatype from Pass A value-shape evidence. The v2 schema
    carries no datatype field — the profile already measured the values, so
    nothing is gained by asking the LLM to re-guess. Purely structural checks
    (no column-name inspection)."""
    if col is None:
        return "string"
    if col.value_shape == ValueShape.DATE:
        return "datetime"
    if col.value_shape == ValueShape.NUMBER:
        return "integer" if col.examples and all(_is_int(e) for e in col.examples) else "float"
    lowered = [e.lower() for e in col.examples]
    if lowered and all(e in ("true", "false") for e in lowered):
        return "boolean"
    if lowered and all(e.startswith(("http://", "https://")) for e in lowered):
        return "uri"
    return "string"


def _pascal_case(name: str) -> str:
    """Mechanical PascalCase of a snake_case handle (fallback target type for
    a relationship column whose target_type the model omitted)."""
    return "".join(p.capitalize() for p in _snake_case(name).split("_")) or "Entity"


def _rel_values(raw_value) -> list[str]:
    """Split a relationship cell into one or more target labels (JSON array,
    pipe-delimited, or short comma-delimited). Mirrors the legacy single-entity
    splitting so multi-entity and legacy paths behave identically."""
    if isinstance(raw_value, list):
        return [v.strip() for v in raw_value if isinstance(v, str) and v.strip()]
    raw_value = str(raw_value)
    if "|" in raw_value:
        return [v.strip() for v in raw_value.split("|") if v.strip()]
    if ", " in raw_value:
        parts = [v.strip() for v in raw_value.split(", ") if v.strip()]
        if all(len(p) < 30 for p in parts) and len(parts) >= 2:
            return parts
    return [raw_value.strip()] if raw_value.strip() else []


def _safe_id(raw: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw.strip())
    return safe[:200] if safe else "unknown"


def _cell(row: dict, column: str) -> str:
    """One cell as a stripped string ('' when missing/empty). Non-string
    values (typed JSON cells) are stringified deterministically."""
    raw = row.get(column, "")
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raw = str(raw)
    return raw.strip()


def _synthetic_key(type_name: str, owned_values: dict[str, str]) -> str:
    """Deterministic content-hash key for a row whose natural key resolves
    empty (ADR 0003 §2 — row conservation). Depends only on the entity type
    and the row's owned non-empty column values — never on batch position,
    row index, or anything random — so identical rows collapse into one
    entity (true duplicates) and batched / re-run ingest stays idempotent.
    """
    material = type_name + "|" + "|".join(
        sorted(f"{col}={val}" for col, val in owned_values.items())
    )
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]
    return _safe_id(digest)


def _snake_case(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "unnamed"
