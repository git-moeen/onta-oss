"""URL extraction helper shared by the agent capabilities and planner.

The "URL-targeted" web-extraction feature lets a user hand us one or more
explicit links — in the chat message ("enrich these from https://… and
https://…") or as structured context from the Explorer — and have a premium
scraper/agent (e.g. Firecrawl) parse those pages to ingest new entities or
enrich existing ones.

This module is the single place that turns free text into a clean list of URLs,
so the discovery capability, the enrichment capability, and the planner all
recognise links the same way. It is pure, dependency-free, and vendor-neutral
(no scraper is named here) — the actual fetching lives behind the premium
``WebSourceProvider`` / ``SourceAdapter`` seams.

It recognises both fully-qualified ``http(s)://`` links and *bare* domains
pasted without a scheme (``humannessindex.vapi.ai``, ``example.com/path``,
``www.foo.org``). Bare domains are promoted **conservatively** — only when the
final label is a recognised public TLD, it is not a known source-file
extension, and the token carries a ``www.`` prefix, a path component, or at
least two dot-separated labels. This keeps sentence tokens (``e.g.``), code
references (``config.py``), and version strings (``v2.5``) out of the result.
Any promoted bare domain is normalised to ``https://<host>…`` so downstream
fetch seams always receive a fetchable URL.
"""

from __future__ import annotations

import re

# Match http(s) URLs. We stop at whitespace and at the common trailing
# punctuation/brackets that surround a link in prose, then strip a trailing run
# of sentence punctuation so "see https://example.com/x." yields ".../x".
_URL_RE = re.compile(r"https?://[^\s<>\"'`\]\)\}]+", re.IGNORECASE)

# Candidate scheme-less tokens: a dotted host (one or more ``label.`` runs) with
# an optional path/query/fragment tail. The leading negative look-behind keeps
# us from re-matching the host portion of an already-scheme'd URL (``//``) or an
# email (``@``), and from starting mid-label. Validation of the host (TLD,
# label count, file-ext deny) happens in ``_promote_bare_host`` — this regex is
# deliberately permissive (one alternation, bounded ``+``: ReDoS-safe).
_BARE_RE = re.compile(
    r"(?<![\w@./])"  # not glued to a word char, @, dot, or slash
    r"((?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,})"  # dotted host
    r"(/[^\s<>\"'`\]\)\}]*)?",  # optional path tail
)
_TRAILING_PUNCT = ".,;:!?"

# Curated common public TLDs. A bare token is only promoted when its final label
# is one of these — this is intentionally a small allow-list rather than a heavy
# public-suffix dependency, so it errs on the side of *not* firing.
_KNOWN_TLDS = frozenset(
    {
        "com", "org", "net", "io", "ai", "co", "dev", "app", "tech", "gov",
        "edu", "info", "biz", "me", "tv", "us", "uk", "ca", "de", "fr", "eu",
        "au", "jp", "cn", "in", "nl", "se", "no", "es", "it", "ch",
        "xyz", "cloud", "site", "online", "store", "news", "academy",
    }
)

# Source-file / asset extensions that look like a bare TLD but are filenames.
# A token whose final label is one of these is never promoted, even if it has
# multiple labels (e.g. ``schema_resolver.py``, ``models.ts``).
_FILE_EXTS = frozenset(
    {
        "py", "ts", "tsx", "js", "jsx", "json", "md", "csv", "txt", "yaml",
        "yml", "toml", "png", "jpg", "jpeg", "svg", "sh", "go", "rs", "java",
        "rb", "css", "html", "xml", "ini", "cfg", "lock", "sql",
    }
)


def _promote_bare_host(host: str, path: str) -> str | None:
    """Return ``https://<host><path>`` if ``host`` is a real bare domain, else None.

    Conservative gate: the final label must be a recognised public TLD, must not
    be a known source-file extension, and the token must look domain-like — a
    ``www.`` prefix, a path component, or ≥2 labels (so ``vs.`` / ``config.py``
    style tokens never promote).
    """
    labels = host.split(".")
    tld = labels[-1].lower()
    if tld in _FILE_EXTS or tld not in _KNOWN_TLDS:
        return None
    has_www = labels[0].lower() == "www"
    has_path = bool(path)
    # ≥2 labels means a host like "a.b" (registrable domain) or deeper; combined
    # with the TLD allow-list this is the "looks like a domain" signal. The regex
    # already requires at least one dot, so 2 is the practical minimum.
    if not (has_www or has_path or len(labels) >= 2):
        return None
    return f"https://{host}{path}"


def extract_urls(text: str | None) -> list[str]:
    """Return the URLs found in ``text``, de-duplicated and in order.

    Recognises both ``http(s)://`` links and bare scheme-less domains. Already
    scheme'd URLs are returned verbatim (trailing sentence punctuation stripped);
    bare domains are validated conservatively and normalised to ``https://…``.
    Order of first appearance is preserved and exact duplicates are dropped.
    Returns ``[]`` for empty/``None`` input.
    """
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()

    # Pass 1: explicit http(s) URLs. Record the spans so pass 2 can skip the host
    # portions of these (the negative look-behind handles most cases; the span
    # check is a belt-and-braces guard against re-emitting a scheme'd host).
    scheme_spans: list[tuple[int, int]] = []
    for m in _URL_RE.finditer(text):
        scheme_spans.append(m.span())
        url = m.group(0).rstrip(_TRAILING_PUNCT)
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)

    # Pass 2: bare scheme-less domains.
    for m in _BARE_RE.finditer(text):
        start, _end = m.span()
        # Skip anything overlapping an already-matched scheme'd URL.
        if any(s <= start < e for s, e in scheme_spans):
            continue
        host = m.group(1)
        path = (m.group(2) or "").rstrip(_TRAILING_PUNCT)
        url = _promote_bare_host(host, path)
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)

    return out


__all__ = ["extract_urls"]
