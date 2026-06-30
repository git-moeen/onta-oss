"""Tests for the LLM-backed value extractor (ONTA-160).

The LLM call is mocked at the ``openrouter_chat`` boundary — no live network.
These tests pin the contract the enrichment adapters rely on:

* a value present in the page text is extracted;
* an absent value yields ``None`` → no fill (no page-chrome junk);
* any error/timeout collapses to ``None`` (never raises);
* the factory wires the LLM extractor only when a key is present, else the
  deterministic offline extractor.
"""

from __future__ import annotations

import pytest

import cograph_client.enrichment.llm_extractor as llm_extractor
from cograph_client.enrichment.llm_extractor import (
    get_default_extractor,
    llm_extract,
)


@pytest.mark.asyncio
async def test_llm_extract_returns_value_present_in_text(monkeypatch):
    monkeypatch.setattr(llm_extractor, "_openrouter_key", lambda: "sk-test")

    async def fake_chat(*args, **kwargs):
        return '{"value": "2010", "confidence": 0.92}'

    monkeypatch.setattr(llm_extractor, "openrouter_chat", fake_chat)

    out = await llm_extract(
        "ElevenLabs was founded in 2010 by ...", "founded_year", "ElevenLabs"
    )
    assert out == {"value": "2010", "confidence": 0.92}


@pytest.mark.asyncio
async def test_llm_extract_absent_value_returns_none(monkeypatch):
    # Model honestly returns null because the text states no such attribute.
    monkeypatch.setattr(llm_extractor, "_openrouter_key", lambda: "sk-test")

    async def fake_chat(*args, **kwargs):
        return '{"value": null, "confidence": 0.0}'

    monkeypatch.setattr(llm_extractor, "openrouter_chat", fake_chat)

    out = await llm_extract("Menu Home About Contact", "founded_year", "ElevenLabs")
    # value present but null — the dict is returned so the calibration guard in
    # extract_value drops it. The contract here: "value" key present.
    assert out == {"value": None, "confidence": 0.0}


@pytest.mark.asyncio
async def test_llm_extract_missing_value_key_returns_none(monkeypatch):
    monkeypatch.setattr(llm_extractor, "_openrouter_key", lambda: "sk-test")

    async def fake_chat(*args, **kwargs):
        return '{"confidence": 0.5}'  # malformed: no "value"

    monkeypatch.setattr(llm_extractor, "openrouter_chat", fake_chat)

    out = await llm_extract("some text", "founded_year", "ElevenLabs")
    assert out is None


@pytest.mark.asyncio
async def test_llm_extract_error_collapses_to_none(monkeypatch):
    monkeypatch.setattr(llm_extractor, "_openrouter_key", lambda: "sk-test")

    async def boom(*args, **kwargs):
        raise RuntimeError("network down / timeout")

    monkeypatch.setattr(llm_extractor, "openrouter_chat", boom)

    # Never raises — a junk fill is worse than an empty field.
    out = await llm_extract("ElevenLabs founded 2010", "founded_year", "ElevenLabs")
    assert out is None


@pytest.mark.asyncio
async def test_llm_extract_null_content_returns_none(monkeypatch):
    # OpenRouter can return content: null (empty/refused/tool-only completion),
    # surfacing here as Python None. Must collapse to None, never raise
    # (None.strip() in the parser would AttributeError).
    monkeypatch.setattr(llm_extractor, "_openrouter_key", lambda: "sk-test")

    async def fake_chat(*args, **kwargs):
        return None

    monkeypatch.setattr(llm_extractor, "openrouter_chat", fake_chat)

    out = await llm_extract("ElevenLabs founded 2010", "founded_year", "ElevenLabs")
    assert out is None


@pytest.mark.asyncio
async def test_llm_extract_empty_content_returns_none(monkeypatch):
    monkeypatch.setattr(llm_extractor, "_openrouter_key", lambda: "sk-test")

    async def fake_chat(*args, **kwargs):
        return ""

    monkeypatch.setattr(llm_extractor, "openrouter_chat", fake_chat)

    out = await llm_extract("ElevenLabs founded 2010", "founded_year", "ElevenLabs")
    assert out is None


@pytest.mark.asyncio
async def test_llm_extract_no_key_returns_none(monkeypatch):
    monkeypatch.setattr(llm_extractor, "_openrouter_key", lambda: "")

    async def should_not_call(*args, **kwargs):
        raise AssertionError("openrouter_chat must not be called without a key")

    monkeypatch.setattr(llm_extractor, "openrouter_chat", should_not_call)

    out = await llm_extract("text", "attr", "Entity")
    assert out is None


@pytest.mark.asyncio
async def test_llm_extract_blank_text_returns_none(monkeypatch):
    monkeypatch.setattr(llm_extractor, "_openrouter_key", lambda: "sk-test")
    assert await llm_extract("", "attr", "Entity") is None
    assert await llm_extract("   \n ", "attr", "Entity") is None


def test_factory_returns_llm_extractor_when_key_present(monkeypatch):
    monkeypatch.setattr(llm_extractor, "_openrouter_key", lambda: "sk-test")
    assert get_default_extractor() is llm_extract


def test_factory_returns_offline_extractor_without_key(monkeypatch):
    monkeypatch.setattr(llm_extractor, "_openrouter_key", lambda: "")
    fn = get_default_extractor()
    assert fn is not llm_extract


@pytest.mark.asyncio
async def test_offline_factory_extractor_is_deterministic(monkeypatch):
    monkeypatch.setattr(llm_extractor, "_openrouter_key", lambda: "")
    fn = get_default_extractor()
    out = await fn('{"value": "Germany", "confidence": 0.8}', "country", "Bosch")
    assert out == {"value": "Germany", "confidence": 0.8}
