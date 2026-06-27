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
