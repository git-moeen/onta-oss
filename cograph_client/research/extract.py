"""The extractor — schema-valid row extraction (ADR 0006 §Extract).

Fetching produces :class:`~cograph_client.research.types.FetchedPage`\\ s of raw
page text; this stage turns them into structured
:class:`~cograph_client.research.types.ResearchRow`\\ s that match the planner's
:class:`~cograph_client.research.types.TargetSchema`. Because the schema was
fixed up front, extraction is a bounded, per-page strict-JSON lift rather than
open-ended reading: one :func:`~cograph_client.resolver.llm_router.openrouter_chat`
call per page pulls ``{"rows": [{"values": {<field>: ...}, "confidence": ...}]}``
into the fixed set of columns.

Each row carries its source page as a citation — the cite-or-abstain substrate
the verifier gates on. Extraction uses a FAST model (the per-page task is
mechanical, not reasoning-heavy) and mirrors the OSS strict-JSON pattern of the
enrichment extractor: ``json_object`` response format, ``temperature=0``, parsed
defensively, and — critically — it NEVER raises. A page that errors is skipped
(logged), not fatal; no key / no pages yields ``[]``.

The whole pass is budget-bounded (stop when the run can no longer afford an LLM
call) and output-capped at ``max_rows``, with a light case-insensitive dedupe
that merges the citations of duplicate rows.

Boundary: OSS file. Imports only stdlib / ``cograph_client.*`` / ``httpx``.
No ``from cograph.*`` and no proprietary identifiers.
"""

from __future__ import annotations

import os
from typing import Optional

import structlog

from cograph_client.research.types import (
    Budget,
    FetchedPage,
    ResearchRow,
    TargetSchema,
)
from cograph_client.resolver.llm_router import openrouter_chat

__all__ = ["extract_rows"]

logger = structlog.stdlib.get_logger("cograph.research.extract")

# Fast, cheap default: the per-page strict-JSON lift is mechanical, not
# reasoning-heavy, so it does not need the frontier model. Env-overridable.
EXTRACT_MODEL_DEFAULT = "google/gemini-2.5-flash"

# Cap on how much page text we send to the model per call — bounds cost/latency
# and avoids token-limit truncation. A leaderboard/table answer lives well
# within this; the static fetcher already caps pages near this range.
_MAX_TEXT_CHARS = 16000

_EXTRACT_SYSTEM = (
    "You are a precise information-extraction engine. You read web page text "
    "and extract records into a FIXED schema. You return ONLY a single JSON "
    "object and nothing else. You never invent values not supported by the "
    "text: a field you cannot find is null, not a guess."
)


def _extract_model() -> str:
    return os.environ.get("OMNIX_RESEARCH_EXTRACT_MODEL", EXTRACT_MODEL_DEFAULT)


def _build_prompt(text: str, schema: TargetSchema, question: str) -> str:
    field_lines = []
    for f in schema.fields:
        if not f.name:
            continue
        desc = f" — {f.description}" if f.description else ""
        req = " (required)" if f.required else ""
        field_lines.append(f"  - {f.name} ({f.type}){req}{desc}")
    fields_block = "\n".join(field_lines) or "  - answer (string)"
    names = schema.field_names() or ["answer"]
    keys = ", ".join(f'"{n}": <string|null>' for n in names)
    q = f"\nThe records answer this QUESTION: {question}\n" if question else ""
    return (
        f"Extract every record about `{schema.entity}` from the TEXT below into "
        "this fixed schema:\n"
        f"{fields_block}\n"
        f"{q}"
        "Return ONLY a JSON object shaped EXACTLY like:\n"
        '{"rows": [{"values": {' + keys + '}, "confidence": <0..1>}]}\n'
        "Rules:\n"
        "- One object in `rows` per record found in the text.\n"
        "- `values` has EXACTLY the schema field names as keys; a value the "
        "text does not state is null (never a guess, never page chrome).\n"
        "- `confidence` is your 0..1 certainty the record is correct.\n"
        "- If the text contains no matching records, return "
        '{"rows": []}.\n\n'
        f"TEXT:\n{text[:_MAX_TEXT_CHARS]}"
    )


def _coerce_values(raw: object, field_names: list[str]) -> dict[str, str]:
    """Keep only string-coercible, non-empty values keyed by a schema field.

    Non-string scalars are coerced to ``str``; ``None`` / empty / non-scalar
    values are dropped. Keys outside the schema are ignored so a chatty model
    can't widen the row shape.
    """
    if not isinstance(raw, dict):
        return {}
    allowed = {n.lower(): n for n in field_names}
    out: dict[str, str] = {}
    for k, v in raw.items():
        canonical = allowed.get(str(k).strip().lower())
        if canonical is None:
            continue
        if v is None:
            continue
        if isinstance(v, bool):
            sval = "true" if v else "false"
        elif isinstance(v, (str, int, float)):
            sval = str(v).strip()
        else:
            # list / dict / other — not a scalar cell value; skip it.
            continue
        if sval:
            out[canonical] = sval
    return out


def _dedupe_key(row: ResearchRow, schema: TargetSchema) -> str:
    """A case-insensitive identity for a row over ALL its schema fields.

    Keying on the FULL value tuple (not just the required fields) — matching
    ``harness._merge_rows`` — so two genuinely distinct records that happen to
    share a required value (e.g. two models both named "GPT" with different
    scores) are NOT collapsed into one, silently dropping the second."""
    names = schema.field_names() or sorted(row.values.keys())
    return "\x1f".join(
        str(row.values.get(n, "")).strip().lower() for n in names
    )


async def extract_rows(
    pages: list[FetchedPage],
    schema: TargetSchema,
    *,
    question: str = "",
    openrouter_key: str = "",
    model: Optional[str] = None,
    max_rows: int = 200,
    budget: "Optional[Budget]" = None,
) -> list[ResearchRow]:
    """Extract schema-valid rows from fetched pages via per-page LLM calls.

    For each page with content, one
    :func:`~cograph_client.resolver.llm_router.openrouter_chat` call (strict
    ``json_object``, ``temperature=0``, a fast model by default) lifts records
    into the fixed :class:`~cograph_client.research.types.TargetSchema`. Each
    resulting :class:`~cograph_client.research.types.ResearchRow` cites its
    source page.

    Rows are concatenated across pages, lightly de-duplicated (case-insensitive
    over all schema fields, merging citations), and capped at ``max_rows``.

    NEVER raises. A page that errors is skipped (logged); no ``openrouter_key``
    or no pages returns ``[]``. The pass is budget-bounded: it stops before a
    page whose call the ``budget`` can no longer afford, and meters each call via
    ``note_llm(1)``.

    Args:
        pages: Fetched pages to extract from (only those with content are used).
        schema: The target schema the planner fixed.
        question: Optional question, added to the prompt for disambiguation.
        openrouter_key: OpenRouter API key. Empty → ``[]``.
        model: Optional model override; defaults to a fast model
            (``OMNIX_RESEARCH_EXTRACT_MODEL`` or ``google/gemini-2.5-flash``).
        max_rows: Hard cap on the total rows returned.
        budget: Optional per-run :class:`~cograph_client.research.types.Budget`.

    Returns:
        The extracted, de-duplicated, capped list of
        :class:`~cograph_client.research.types.ResearchRow`.
    """
    if not openrouter_key or not pages:
        return []
    if max_rows <= 0:
        return []

    # Lazy import to keep the module import graph light and avoid pulling the
    # extraction module's transitive deps unless extraction actually runs.
    from cograph_client.enrichment.extraction import _try_parse_json

    field_names = schema.field_names() or ["answer"]
    use_model = model or _extract_model()

    kept: list[ResearchRow] = []
    index: dict[str, ResearchRow] = {}

    for page in pages:
        if len(kept) >= max_rows:
            break
        if not page.has_content():
            continue

        # Budget gate: stop the whole pass once we can't afford another call.
        if budget is not None and not budget.can_call_llm():
            break
        if budget is not None:
            budget.note_llm(1)

        try:
            content = await openrouter_chat(
                openrouter_key,
                _EXTRACT_SYSTEM,
                _build_prompt(page.text, schema, question),
                model=use_model,
                temperature=0,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            logger.warning(
                "research.extract.page_failed", url=page.url, error=str(exc)[:200]
            )
            continue

        if not content:
            continue

        obj = _try_parse_json(content)
        if obj is None:
            logger.warning("research.extract.bad_json", url=page.url)
            continue

        raw_rows = obj.get("rows")
        if not isinstance(raw_rows, list):
            continue

        for raw_row in raw_rows:
            if len(kept) >= max_rows:
                break
            if not isinstance(raw_row, dict):
                continue
            values = _coerce_values(raw_row.get("values"), field_names)
            if not values:
                # A row with no non-empty values is not a record — drop it.
                continue
            try:
                conf = float(raw_row.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            conf = max(0.0, min(1.0, conf))

            row = ResearchRow(
                values=values, citations=[page.url], confidence=conf
            )

            # Light dedupe: merge citations of an identical earlier row rather
            # than emit a duplicate record.
            key = _dedupe_key(row, schema)
            prior = index.get(key)
            if prior is not None:
                if page.url and page.url not in prior.citations:
                    prior.citations.append(page.url)
                prior.confidence = max(prior.confidence, conf)
                continue

            index[key] = row
            kept.append(row)

    return kept[:max_rows]
