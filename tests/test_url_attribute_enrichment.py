"""ONTA-157 — URL-valued attribute enrichment must yield a URL, not page chrome.

Enriching a ``website`` / ``*_url`` attribute over a web snippet, the single-pass
extractor lifts the first plausible text from the page body — which for a real
site is nav chrome ("Skip to content", "Platform") or the entity name. The actual
website is the resolved citation (``source_url``). These tests lock the coercion:

* a URL-valued attribute whose extracted ``value`` is junk → value becomes the
  ``source_url`` (the Exa failure mode the user hit),
* a URL-valued attribute whose ``value`` is ALREADY a URL → kept (the Wikidata
  official-website case, where source_url is just the provenance page),
* a non-URL attribute → never touched,
* attribute-name + datatype detection and URL normalization.
"""

from __future__ import annotations

import pytest

from cograph_client.enrichment.extraction import (
    coerce_url_attribute_value,
    is_url_attribute,
    is_website_attribute,
    looks_like_url,
    normalize_url,
)
from cograph_client.enrichment.models import Verdict


def _v(value: str, source_url: str | None = None) -> Verdict:
    return Verdict(value=value, confidence=0.62, source="exa", source_url=source_url)


# --- is_url_attribute -------------------------------------------------------

@pytest.mark.parametrize(
    "attr",
    ["website", "Website", "url", "homepage", "home_page", "company_url",
     "profile_url", "site_uri", "vendor_website", "WEB SITE"],
)
def test_is_url_attribute_positive(attr):
    assert is_url_attribute(attr)


@pytest.mark.parametrize(
    "attr", ["name", "founded_year", "manufacturer", "score", "address", "country"],
)
def test_is_url_attribute_negative(attr):
    assert not is_url_attribute(attr)


def test_is_url_attribute_by_datatype():
    # A non-url-ish name still counts when the declared datatype is a URI.
    assert is_url_attribute("reference", datatype="uri")
    assert is_url_attribute("reference", datatype="URL")
    assert not is_url_attribute("reference", datatype="string")


# --- looks_like_url ---------------------------------------------------------

@pytest.mark.parametrize(
    "text", ["https://elevenlabs.io/", "http://x.ai", "elevenlabs.io",
             "inworld.ai/platform", "sub.domain.co.uk/path?q=1"],
)
def test_looks_like_url_positive(text):
    assert looks_like_url(text)


@pytest.mark.parametrize(
    "text", ["Skip to content", "Platform", "xAI", "# Canopy Labs (Inc)",
             "MiniMax", "", None, "e.g", "U.S.A", "not a url"],
)
def test_looks_like_url_negative(text):
    assert not looks_like_url(text)


def test_normalize_url_adds_scheme_and_trims_slash():
    assert normalize_url("elevenlabs.io") == "https://elevenlabs.io"
    assert normalize_url("https://x.ai/") == "https://x.ai"
    assert normalize_url("https://inworld.ai/platform") == "https://inworld.ai/platform"


# --- coerce_url_attribute_value (the core fix) ------------------------------

def test_junk_value_falls_back_to_source_url():
    """The Exa failure: value is page chrome, source_url is the real site."""
    v = coerce_url_attribute_value("website", _v("Skip to content", "https://elevenlabs.io/"))
    assert v.value == "https://elevenlabs.io"


def test_entity_name_value_falls_back_to_source_url():
    v = coerce_url_attribute_value("website", _v("xAI", "https://x.ai/"))
    assert v.value == "https://x.ai"


def test_url_shaped_value_is_kept_not_overwritten_by_source_url():
    """Wikidata case: value IS the official site; source_url is the Wikidata
    provenance page. Must NOT clobber the good value with the provenance URL."""
    v = coerce_url_attribute_value(
        "website",
        _v("https://elevenlabs.io", "https://www.wikidata.org/entity/Q123"),
    )
    assert v.value == "https://elevenlabs.io"


def test_bare_domain_value_is_normalized_in_place():
    v = coerce_url_attribute_value("website", _v("elevenlabs.io", "https://wikidata.org/Q1"))
    assert v.value == "https://elevenlabs.io"  # normalized, not replaced by source_url


def test_non_url_attribute_is_untouched():
    """A normal attribute must never be coerced, even if a source_url exists."""
    v = coerce_url_attribute_value("founded_year", _v("2014", "https://elevenlabs.io/"))
    assert v.value == "2014"


def test_junk_value_no_source_url_left_as_is():
    """Nothing better to use → leave the (junk) value rather than fabricate."""
    v = coerce_url_attribute_value("website", _v("Platform", None))
    assert v.value == "Platform"


def test_returns_same_object_when_nothing_to_change():
    original = _v("https://x.ai", "https://x.ai")
    out = coerce_url_attribute_value("website", original)
    assert out is original  # no needless copy when already normalized


# --- review hardening (PR #82 P2s) -----------------------------------------

@pytest.mark.parametrize("text", ["file.tar.gz", "report.docx", "archive.zip",
                                  "logo.png", "data.json", "build.css"])
def test_file_extension_tokens_are_not_urls(text):
    """A dotted filename must NOT be treated as a URL (else junk gets kept and
    https-prefixed instead of recovering the real citation)."""
    assert not looks_like_url(text)


def test_filename_value_recovers_source_url_not_kept_as_url():
    """Regression: 'file.tar.gz' lifted from page chrome must fall back to the
    real site, not become 'https://file.tar.gz'."""
    v = coerce_url_attribute_value("website", _v("file.tar.gz", "https://real.io/"))
    assert v.value == "https://real.io"


def test_idn_unicode_domain_is_a_url_and_preserved():
    """Raw-unicode (IDN) website values must be recognized and kept, not dropped
    and replaced by the provenance page."""
    assert looks_like_url("münchen.de")
    v = coerce_url_attribute_value("website", _v("münchen.de", "https://provenance.org/page"))
    assert v.value == "https://münchen.de"


def test_normalize_url_preserves_trailing_slash_inside_query():
    """A trailing slash inside a ?query/#fragment is significant — never strip
    it; only a plain path-terminating slash is trimmed."""
    assert normalize_url("https://x.ai/search?q=a/") == "https://x.ai/search?q=a/"
    assert normalize_url("https://x.ai/p#frag/") == "https://x.ai/p#frag/"
    assert normalize_url("https://x.ai/path/") == "https://x.ai/path"  # plain path slash trimmed


def test_non_website_url_attribute_does_not_inherit_provenance():
    """image_url / redirect_uri must NOT fall back to source_url (the citation
    page is not the image/redirect). Junk value is left as-is instead."""
    for attr in ("image_url", "redirect_uri", "logo_uri"):
        v = coerce_url_attribute_value(attr, _v("Logo", "https://page.com/about"))
        assert v.value == "Logo", attr


def test_non_website_url_attribute_still_normalizes_a_real_url_value():
    v = coerce_url_attribute_value("image_url", _v("cdn.example.com/a.png", None))
    assert v.value == "https://cdn.example.com/a.png"


@pytest.mark.parametrize("attr", ["website", "homepage", "url", "site", "company_website"])
def test_is_website_attribute_positive(attr):
    assert is_website_attribute(attr)


@pytest.mark.parametrize("attr", ["image_url", "redirect_uri", "logo_uri",
                                  "source_url", "founded_year", "name"])
def test_is_website_attribute_negative(attr):
    assert not is_website_attribute(attr)
