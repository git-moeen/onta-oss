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
final label is a recognised public TLD AND it is not a known source-file
extension AND the host is not *exactly* a multi-part public suffix (``co.uk``
on its own is a registry, not a registrable host). This keeps file references
(``config.py``), version strings (``v2.5``), and ellipses out of the result.
Any promoted bare domain is normalised to ``https://<host>…`` with the **host
lower-cased** (path left untouched) so downstream fetch seams always receive a
fetchable, canonical URL — and so a bare host de-dupes against the same host
written with a scheme.

Note on scope: a bare two-label token like ``word.tld`` (``etc.com``) IS
promoted whenever ``tld`` is a recognised public TLD — we can't distinguish a
real domain from an unlucky abbreviation without resolving DNS, and erring
toward "treat it as a link" matches user intent for the URL-targeting feature.
The guards above are what keep the common false positives (file refs, versions,
public-suffix registries) out.
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
#
# DELIBERATE ccTLD CONFLICT: ``sh`` (Saint Helena), ``rs`` (Serbia), and
# ``go`` are *also* real ccTLDs, so this deny-list silently drops genuine
# domains like ``rust-lang.rs`` or ``example.sh``. We accept that trade-off
# on purpose: in agent prose, ``foo.sh`` / ``foo.rs`` / ``foo.go`` is far more
# often a shell/Rust/Go source file than a Serbian/Saint-Helena website, and a
# false *miss* (degrades to open-web search) is cheaper than a false *fetch* of
# a code-reference token. Do NOT "fix" this by removing sh/rs/go from the
# deny-list without a stronger domain signal (e.g. an explicit scheme, which
# already bypasses this path).
_FILE_EXTS = frozenset(
    {
        "py", "ts", "tsx", "js", "jsx", "json", "md", "csv", "txt", "yaml",
        "yml", "toml", "png", "jpg", "jpeg", "svg", "sh", "go", "rs", "java",
        "rb", "css", "html", "xml", "ini", "cfg", "lock", "sql",
    }
)

# Multi-part public suffixes (registries under which people register a name, not
# registrable hosts themselves). A host that is *exactly* one of these — e.g.
# ``co.uk`` with no label in front — is a registry, not a site, and must NOT be
# promoted. ``api.example.co.uk`` and ``example.co.uk`` are fine: they carry a
# registrable label before the suffix. Small curated set (we avoid a heavy
# public-suffix dependency, matching the curated-TLD approach above).
_PUBLIC_SUFFIXES = frozenset(
    {
        "co.uk", "org.uk", "gov.uk", "ac.uk", "me.uk", "ltd.uk", "plc.uk",
        "com.au", "net.au", "org.au", "gov.au", "edu.au",
        "co.jp", "or.jp", "ne.jp", "ac.jp", "go.jp",
        "co.nz", "org.nz", "govt.nz",
        "co.za", "org.za",
        "com.br", "com.cn", "com.mx", "co.in", "co.kr",
    }
)


def _promote_bare_host(host: str, path: str) -> str | None:
    """Return ``https://<host><path>`` if ``host`` is a real bare domain, else None.

    Conservative gate, in order:
      * the final label must be a recognised public TLD;
      * the final label must NOT be a known source-file extension;
      * the host must not be *exactly* a multi-part public suffix (``co.uk``) —
        a registry needs a registrable label in front to be a real site.

    A bare two-label ``word.tld`` clears these and IS promoted (we can't tell a
    real domain from an abbreviation without DNS; the URL-targeting feature
    prefers treating it as a link). The host is lower-cased on normalisation so
    ``HUMANNESSINDEX.VAPI.AI`` → ``https://humannessindex.vapi.ai`` and bare
    hosts de-dupe against the same host written with a scheme; the path is left
    case-sensitive.
    """
    host_lc = host.lower()
    labels = host_lc.split(".")
    tld = labels[-1]
    if tld in _FILE_EXTS or tld not in _KNOWN_TLDS:
        return None
    # Reject a bare registry suffix (``co.uk``) with no registrable label.
    last_two = ".".join(labels[-2:])
    if host_lc == last_two and last_two in _PUBLIC_SUFFIXES:
        return None
    return f"https://{host_lc}{path}"


def _split_host(url: str) -> tuple[str, str]:
    """Return ``(host_lc, path)`` for ``url`` (scheme + any userinfo stripped)."""
    rest = url.split("://", 1)[-1].split("@", 1)[-1]
    parts = re.split(r"([/?#])", rest, maxsplit=1)
    host = parts[0].lower()
    path = "".join(parts[1:]) if len(parts) > 1 else ""
    return host, path


def extract_urls(text: str | None) -> list[str]:
    """Return the URLs found in ``text``, de-duplicated and in order.

    Recognises both ``http(s)://`` links and bare scheme-less domains. Already
    scheme'd URLs are returned verbatim (trailing sentence punctuation stripped);
    bare domains are validated conservatively and normalised to ``https://…``
    with the host lower-cased.

    De-duplication, all keyed on the lower-cased host (scheme-insensitive):
      * two URLs with the **same host and same path** collapse to the first seen
        (so ``http://example.com`` then bare ``example.com`` → one entry);
      * a **bare host with no path** is suppressed if that host already appeared
        on any earlier URL (so ``https://one.com/a`` + bare ``one.com`` → just
        the first) — a path-less bare mention adds nothing over a concrete page;
      * but distinct **paths on the same host** are kept (``/a`` and ``/b`` are
        different pages).

    Order of first appearance is preserved. Returns ``[]`` for empty/``None``.
    """
    if not text:
        return []
    out: list[str] = []
    seen_full: set[tuple[str, str]] = set()  # (host, path) already emitted
    seen_hosts: set[str] = set()  # hosts that appeared on any emitted URL

    def _emit(url: str, *, bare: bool) -> None:
        host, path = _split_host(url)
        # A path-less bare host adds nothing once the host is already present.
        if bare and not path and host in seen_hosts:
            return
        if (host, path) in seen_full:
            return
        seen_full.add((host, path))
        seen_hosts.add(host)
        out.append(url)

    # Pass 1: explicit http(s) URLs. Record the spans so pass 2 can skip the host
    # portions of these (the negative look-behind handles most cases; the span
    # check is a belt-and-braces guard against re-emitting a scheme'd host).
    scheme_spans: list[tuple[int, int]] = []
    for m in _URL_RE.finditer(text):
        scheme_spans.append(m.span())
        url = m.group(0).rstrip(_TRAILING_PUNCT)
        if url:
            _emit(url, bare=False)

    # Pass 2: bare scheme-less domains.
    for m in _BARE_RE.finditer(text):
        start, _end = m.span()
        # Skip anything overlapping an already-matched scheme'd URL.
        if any(s <= start < e for s, e in scheme_spans):
            continue
        host = m.group(1)
        path = (m.group(2) or "").rstrip(_TRAILING_PUNCT)
        url = _promote_bare_host(host, path)
        if url:
            _emit(url, bare=True)

    return out


__all__ = ["extract_urls"]
