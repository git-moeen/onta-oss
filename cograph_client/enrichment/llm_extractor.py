"""LLM-backed single-pass value extractor for enrichment (ONTA-160).

The enrichment pipeline turns raw web-search text into a structured
:class:`~cograph_client.enrichment.models.Verdict` via
:func:`cograph_client.enrichment.extraction.extract_value`, which delegates the
actual *value lifting* to an injectable extractor. Historically every
production adapter fell through to the deterministic offline
``default_extractor`` whose last-resort heuristic returns "the first non-empty
line of the page" — which is page chrome ("Menu", "Skip to content") or the
entity's own name. The result was junk fills like ``founded_year = "Menu"`` or
``manufacturer = "ElevenLabs"``.

This module provides a real LLM-backed extractor that reads the page text and a
strict-JSON prompt, returning ``{"value": <str|null>, "confidence": <0..1>}``.
It mirrors the OSS OpenRouter call pattern already used by the resolver
(:func:`cograph_client.resolver.llm_router.openrouter_chat`), so the LLM
extractor stays OSS: the resolver already makes OSS LLM calls, and a value
extractor is the same shape of operation.

``get_default_extractor`` is the factory production code wires in: it returns
the async LLM extractor when an OpenRouter key is configured, and otherwise an
async wrapper around the deterministic offline ``default_extractor`` so tests
and local/offline runs stay deterministic with zero network.

Boundary: OSS file. Imports only stdlib / ``cograph_client.*`` / ``httpx``.
No ``from cograph.*`` and no proprietary identifiers.
"""

from __future__ import annotations

import os
from typing import Optional

from cograph_client.enrichment.extraction import (
    ExtractorFn,
    default_extractor,
    _try_parse_json,
)
from cograph_client.resolver.llm_router import openrouter_chat

# The model used for value extraction. Defaults to a fast, cheap model; the
# strict-JSON single-value task does not need a frontier model. Env-overridable.
EXTRACT_MODEL_DEFAULT = "google/gemini-2.5-flash"

# Cap on how much page text we send to the model. Enrichment snippets/pages can
# be large; the answer to a single attribute lives near the top, and keeping the
# prompt bounded controls cost/latency and avoids token-limit truncation.
_MAX_TEXT_CHARS = 12000

_EXTRACT_SYSTEM = (
    "You are a precise information-extraction engine. You read web page text "
    "and return ONLY a single JSON object. You never guess, never invent, and "
    "never use page navigation chrome or the entity's own name as the value."
)


def _extract_model() -> str:
    return os.environ.get("OMNIX_ENRICH_EXTRACT_MODEL", EXTRACT_MODEL_DEFAULT)


def _openrouter_key() -> str:
    from cograph_client.config import settings

    return settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")


def _build_prompt(raw_text: str, attribute: str, entity_label: str) -> str:
    text = (raw_text or "")[:_MAX_TEXT_CHARS]
    return (
        f"Extract the value of attribute `{attribute}` for entity "
        f"`{entity_label}` from the text below. Return ONLY a JSON object "
        '`{"value": <string|null>, "confidence": <0..1>}`. Return null if the '
        "text does not state it — DO NOT GUESS or use the entity name / page "
        "navigation as the value. For a typed attribute return the bare value "
        "(a year as 4 digits, a number as digits, no surrounding prose).\n\n"
        f"TEXT:\n{text}"
    )


async def llm_extract(
    raw_text: str, attribute: str, entity_label: str
) -> Optional[dict]:
    """Extract one attribute value from raw text via an OpenRouter LLM call.

    Returns a strict-JSON-shaped dict ``{"value": ..., "confidence": ...}`` on
    success, or ``None`` when the text states no value, the key is missing, or
    ANY error/timeout occurs. This function NEVER raises — the calling adapter
    treats ``None`` as "no value" and writes no fill.
    """
    if not raw_text or not raw_text.strip():
        return None

    api_key = _openrouter_key()
    if not api_key:
        return None

    try:
        content = await openrouter_chat(
            api_key,
            _EXTRACT_SYSTEM,
            _build_prompt(raw_text, attribute, entity_label),
            model=_extract_model(),
            temperature=0,
            max_tokens=256,
            response_format={"type": "json_object"},
            timeout=30,
        )
    except Exception:
        # Network error, HTTP error after fallback chain, timeout, malformed
        # response shape — all collapse to "no value". A junk fill is worse
        # than an empty field, so a failed extraction must yield nothing.
        return None

    # OpenRouter can return ``content: null`` (an empty / refused / tool-only
    # completion), which surfaces here as Python ``None`` — guard before parsing
    # so the "NEVER raises" contract holds (``_try_parse_json(None)`` would
    # ``AttributeError`` on ``None.strip()``).
    if not content:
        return None

    obj = _try_parse_json(content)
    if obj is None or "value" not in obj:
        return None
    return obj


async def _offline_extractor(
    raw_text: str, attribute: str, entity_label: str
) -> Optional[dict]:
    """Async wrapper around the deterministic offline ``default_extractor``.

    Used when no OpenRouter key is configured so tests / local / offline runs
    keep working deterministically with zero network. The underlying
    ``default_extractor`` is pure-CPU, so no thread offload is needed.
    """
    return default_extractor(raw_text, attribute, entity_label)


def get_default_extractor() -> ExtractorFn:
    """Return the production async extractor.

    The LLM extractor when an OpenRouter key is configured (so real adapters
    stop writing page chrome); otherwise the deterministic offline extractor so
    keyless environments (CI, local) remain reproducible without network.
    """
    if _openrouter_key():
        return llm_extract
    return _offline_extractor
