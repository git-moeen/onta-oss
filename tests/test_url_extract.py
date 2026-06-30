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


# ---------------------------------------------------------------------------
# Adversarial-review regressions (ONTA-152 follow-up).
# ---------------------------------------------------------------------------


# P1-1: a bare multi-part public suffix is a registry, not a registrable host.
def test_bare_public_suffix_registry_rejected():
    assert extract_urls("see api domain co.uk here") == []
    assert extract_urls("registered under com.au somewhere") == []


# P1-1: but a real host *under* such a suffix still promotes.
def test_registrable_host_under_public_suffix_promoted():
    assert extract_urls("check api.example.co.uk for the spec") == [
        "https://api.example.co.uk"
    ]
    assert extract_urls("the site example.co.uk is up") == ["https://example.co.uk"]


# P1-1 decision: a bare `word.tld` with a real public TLD IS promoted. We can't
# distinguish a real domain from an unlucky abbreviation without DNS, and the
# URL-targeting feature prefers treating it as a link. Documented in the module.
def test_bare_word_tld_is_promoted_by_design():
    assert extract_urls("He said etc.com is funny") == ["https://etc.com"]
    assert extract_urls("End. Next.com starts") == ["https://next.com"]


# P1-1: the headline legit cases must NOT regress.
def test_legit_bare_domains_still_promote():
    assert extract_urls("example.com") == ["https://example.com"]
    assert extract_urls("data.gov has datasets") == ["https://data.gov"]
    assert extract_urls("from this page humannessindex.vapi.ai") == [
        "https://humannessindex.vapi.ai"
    ]
    assert extract_urls("open example.com/path now") == ["https://example.com/path"]


# P1-2: a bare host that also appears scheme-qualified must emit ONCE (by host).
def test_same_host_scheme_and_bare_deduped():
    assert extract_urls("see http://example.com and bare example.com later") == [
        "http://example.com"
    ]
    assert extract_urls("compare https://one.com/a with one.com today") == [
        "https://one.com/a"
    ]


# P1-2 boundary: distinct *paths* on the same host are different pages — keep both.
def test_same_host_distinct_paths_both_kept():
    assert extract_urls(
        "enrich from https://acme.example/a and https://acme.example/b"
    ) == ["https://acme.example/a", "https://acme.example/b"]


# P2-1: the host is lower-cased on normalisation; the path stays case-sensitive.
def test_bare_host_lowercased_path_preserved():
    assert extract_urls("see HUMANNESSINDEX.VAPI.AI") == [
        "https://humannessindex.vapi.ai"
    ]
    assert extract_urls("open Example.COM/Path/MixedCase") == [
        "https://example.com/Path/MixedCase"
    ]


# P2-2: ccTLD trade-off — sh/rs/go are deny-listed as source-file extensions,
# so genuine ccTLD domains under them are intentionally dropped (documented).
def test_cctld_extension_conflict_dropped_by_design():
    assert extract_urls("clone rust-lang.rs for the source") == []
    assert extract_urls("run deploy.sh and main.go") == []
