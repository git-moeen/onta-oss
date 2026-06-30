"""Unit tests for the shared URL-extraction helper (cograph_client.web_sources.url_extract).

This pure helper turns free text into the list of explicit links the agent
capabilities and planner act on, so it is the single place URL recognition is
defined. ReDoS-safe by construction (one character class, single ``+``).
"""

from cograph_client.web_sources.url_extract import extract_urls


def test_extracts_http_and_https_in_order_deduped():
    text = "see https://a.com/x and http://b.io/y plus https://a.com/x again"
    assert extract_urls(text) == ["https://a.com/x", "http://b.io/y"]


def test_strips_trailing_sentence_punctuation():
    assert extract_urls("go to https://a.com/p.") == ["https://a.com/p"]
    assert extract_urls("link: https://a.com/p, then more") == ["https://a.com/p"]
    assert extract_urls("really? https://a.com/p!") == ["https://a.com/p"]


def test_excludes_wrapping_brackets_and_quotes():
    assert extract_urls("(https://a.com/p)") == ["https://a.com/p"]
    assert extract_urls('"https://a.com/p"') == ["https://a.com/p"]
    assert extract_urls("[https://a.com/p]") == ["https://a.com/p"]


def test_preserves_query_and_fragment():
    assert extract_urls("https://a.com/p?q=1&x=2#frag") == ["https://a.com/p?q=1&x=2#frag"]


def test_ignores_non_http_schemes():
    # No http(s):// → not a link we will hand to the scraper.
    assert extract_urls("ftp://a.com file:///etc/passwd javascript:alert(1)") == []


def test_empty_none_and_no_links():
    assert extract_urls("") == []
    assert extract_urls(None) == []
    assert extract_urls("no links here at all") == []


def test_multiple_distinct_urls_order_preserved():
    text = "first https://one.com then https://two.com finally https://three.com"
    assert extract_urls(text) == [
        "https://one.com",
        "https://two.com",
        "https://three.com",
    ]


# ---------------------------------------------------------------------------
# Bare (scheme-less) domain detection — ONTA-152.
#
# A bare domain pasted without http(s):// must be recognised AND normalised to
# ``https://…`` so downstream fetch seams get a fetchable URL, while sentence
# tokens, code/file references, and version strings stay out of the result.
# ---------------------------------------------------------------------------


def test_bare_domain_promoted_and_normalized():
    # The ONTA-152 reproducer: a bare host pasted into an /agent message.
    assert extract_urls(
        "find a list of voice models from this page humannessindex.vapi.ai"
    ) == ["https://humannessindex.vapi.ai"]


def test_bare_subdomain_multi_label():
    assert extract_urls("check api.example.co.uk for the spec") == [
        "https://api.example.co.uk"
    ]


def test_www_prefixed_host_promoted():
    assert extract_urls("go to www.example.com please") == ["https://www.example.com"]


def test_bare_domain_with_path_promoted():
    assert extract_urls("the page example.com/path/to/thing has it") == [
        "https://example.com/path/to/thing"
    ]


def test_bare_domain_trailing_sentence_punctuation_stripped():
    assert extract_urls("see example.com/page.") == ["https://example.com/page"]
    assert extract_urls("look at humannessindex.vapi.ai, then stop") == [
        "https://humannessindex.vapi.ai"
    ]


def test_mixed_scheme_and_bare_in_one_message():
    # Scheme'd URL kept verbatim; bare domain normalised; order preserved.
    text = "compare https://one.com/a with humannessindex.vapi.ai today"
    assert extract_urls(text) == [
        "https://one.com/a",
        "https://humannessindex.vapi.ai",
    ]


def test_bare_domain_deduped_against_itself():
    text = "example.com/x and again example.com/x"
    assert extract_urls(text) == ["https://example.com/x"]


# --- False-positive guards: these must NOT be promoted to URLs. ------------


def test_guard_abbreviations_not_promoted():
    # Final label is not a known TLD (g/e/c) → never a URL.
    assert extract_urls("e.g. this, i.e. that, etc. and so on") == []
    assert extract_urls("python vs. rust performance") == []


def test_guard_source_file_references_not_promoted():
    assert extract_urls("edit config.py and schema_resolver.py") == []
    assert extract_urls("see models.ts and app.tsx") == []
    assert extract_urls("the package.json and tailwind.css files") == []


def test_guard_version_strings_not_promoted():
    assert extract_urls("upgrade to v2.5 from 1.0.3") == []


def test_guard_decimals_and_ellipses_not_promoted():
    assert extract_urls("pi is about 3.14 here") == []
    assert extract_urls("wait for it... then go") == []
