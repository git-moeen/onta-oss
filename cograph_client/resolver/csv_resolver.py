"""CSV schema inference — one LLM call to infer column mapping, then
deterministic mapping for all rows. No LLM per row."""

from __future__ import annotations

import json
import os
import re

import anthropic
import httpx
import structlog
from pydantic import ValidationError

from cograph_client.resolver.models import (
    ColumnMapping,
    ColumnRole,
    CSVSchemaMapping,
    ExtractedAttribute,
    ExtractedEntity,
    ExtractedRelationship,
)

logger = structlog.stdlib.get_logger("cograph.resolver.csv")

CSV_SCHEMA_SYSTEM = """\
You are a knowledge graph schema inference engine.
Given CSV column names and sample rows, determine:
1. What entity type this CSV represents (PascalCase singular noun)
2. Which column is the entity identifier (type_id) — pick the most natural unique key
3. Which columns are literal attributes (and their datatypes)
4. Which columns reference other entities (relationships)

Entity-first principle — PREFER RELATIONSHIPS OVER ATTRIBUTES:
When in doubt whether a column should be an attribute or a relationship, \
ALWAYS make it a relationship. Entities can have properties added later; \
string literals are dead ends in a knowledge graph.

These should ALWAYS be relationships, never string attributes:
- Geographic columns: city, state, country, region, zipcode, zip_code, \
  postal_code, neighborhood, county, district, area
- People columns: owner, agent, broker, manager, seller, buyer, author, creator
- Organization columns: company, brokerage, firm, agency, school, university
- Any column whose values are names of real-world things that could have \
  their own properties

Only use attribute for truly atomic values that will never need their own \
properties: prices, counts, measurements, dates, booleans, IDs, URLs.

Rules:
- Exactly one column must be type_id (the primary identifier / natural key)
- Columns with numeric values → integer or float (as attributes)
- Columns with dates → datetime (as attributes)
- Columns with true/false → boolean (as attributes)
- URL columns → uri (as attributes)

Respond with valid JSON only. No markdown."""

CSV_SCHEMA_USER = """\
Column names: {columns}

Sample rows (first {n} of {total}):
{sample_rows}

Existing ontology types:
{existing_types}

Return JSON:
{{
  "entity_type": "TypeName",
  "columns": [
    {{
      "column_name": "exact_column_name",
      "role": "type_id" | "attribute" | "relationship",
      "target_type": "TargetTypeName or null",
      "datatype": "string|integer|float|boolean|datetime|uri",
      "attribute_name": "snake_case_name"
    }}
  ]
}}"""


class CSVResolver:
    EXTRACT_MODEL = os.environ.get("OMNIX_EXTRACT_MODEL", "deepseek/deepseek-v3.2")
    EXTRACT_PROVIDER = os.environ.get("OMNIX_EXTRACT_PROVIDER", "openrouter")

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
        """Infer column-to-ontology mapping from sample rows. Single LLM call,
        with one retry at higher temperature if the response fails validation."""
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

        # Validate: must have exactly one type_id
        id_cols = [c for c in mapping.columns if c.role == ColumnRole.TYPE_ID]
        if len(id_cols) != 1:
            logger.warning("csv_schema_no_id", id_cols=len(id_cols))
            # Fallback: use first column as ID
            if mapping.columns:
                mapping.columns[0].role = ColumnRole.TYPE_ID

        # Post-processing: if the chosen type_id is numeric, prefer a string
        # column with a name-like label (institution, title, name, etc.)
        # Numeric IDs cause deduplication when values repeat.
        id_col = next((c for c in mapping.columns if c.role == ColumnRole.TYPE_ID), None)
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

    async def _call_llm(self, user_content: str, temperature: float = 0.0) -> dict:
        if self.EXTRACT_PROVIDER == "openrouter" and self._openrouter_key:
            return await self._infer_via_openrouter(user_content, temperature)
        return await self._infer_via_anthropic(user_content, temperature)

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
        return CSVSchemaMapping(
            entity_type=data["entity_type"],
            columns=[ColumnMapping(**col) for col in data["columns"]],
        )

    async def _infer_via_openrouter(self, user_content: str, temperature: float = 0.0) -> dict:
        async with httpx.AsyncClient(timeout=60) as client:
            res = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.EXTRACT_MODEL,
                    "messages": [
                        {"role": "system", "content": CSV_SCHEMA_SYSTEM},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": 2048,
                    "temperature": temperature,
                },
            )
            res.raise_for_status()
            text = res.json()["choices"][0]["message"]["content"]
            stripped = text.strip()
            if stripped.startswith("```"):
                lines = [l for l in stripped.split("\n") if not l.strip().startswith("```")]
                stripped = "\n".join(lines)
            return json.loads(stripped)

    async def _infer_via_anthropic(self, user_content: str, temperature: float = 0.0) -> dict:
        msg = await self._client.messages.create(
            model="claude-sonnet-4-6",
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
                                    },
                                    "required": ["column_name", "role", "datatype"],
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
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelationship]]:
        """Deterministically convert all CSV rows to entities + relationships. No LLM."""
        id_col = next((c for c in mapping.columns if c.role == ColumnRole.TYPE_ID), None)
        if not id_col:
            return [], []

        entities: list[ExtractedEntity] = []
        relationships: list[ExtractedRelationship] = []
        seen_rel_entities: dict[str, str] = {}  # safe_id → type for relationship targets
        rel_entity_names: dict[str, str] = {}  # safe_id → original value for name attr

        for row in rows:
            entity_id = row.get(id_col.column_name, "").strip()
            if not entity_id:
                continue

            safe_id = _safe_id(entity_id)
            attrs: list[ExtractedAttribute] = []
            entity_rels: list[ExtractedRelationship] = []

            for col in mapping.columns:
                if col.role == ColumnRole.TYPE_ID:
                    continue
                raw_value = row.get(col.column_name, "")
                if isinstance(raw_value, str):
                    raw_value = raw_value.strip()
                if not raw_value:
                    continue

                attr_name = col.attribute_name or _snake_case(col.column_name)

                # Handle JSON arrays, pipe-delimited, and comma-delimited strings
                # by expanding into multiple values for relationships
                if col.role == ColumnRole.RELATIONSHIP and col.target_type:
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

        # Create stub entities for relationship targets (so they exist in the graph)
        for target_id, target_type in seen_rel_entities.items():
            entities.append(ExtractedEntity(
                type_name=target_type,
                id=target_id,
                attributes=[ExtractedAttribute(name="name", value=rel_entity_names.get(target_id, target_id.replace("_", " ")), datatype="string")],
            ))

        return entities, relationships


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


def _safe_id(raw: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", raw.strip())
    return safe[:200] if safe else "unknown"


def _snake_case(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s or "unnamed"
