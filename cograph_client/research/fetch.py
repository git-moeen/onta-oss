"""The fetch ladder — a cheap→expensive tower of page fetchers (ADR 0006 §Fetch).

A single fetch method is never enough: static HTTP misses JS-rendered content
("11 of 21 rows"), and a managed browser is too slow/costly to use for
everything. So fetching is a LADDER of :class:`PageFetcher`\\ s ordered by
``tier`` (0 = cheapest). The harness tries the cheapest rung first and escalates
only when the result looks incomplete.

OSS ships exactly one rung: :class:`StaticHttpFetcher` (tier 0) — a plain
``httpx`` GET with a stdlib HTML→text reduction, no paid vendor, byte-capped and
SSRF-guarded. Premium rungs (a Browserbase/Firecrawl JS-render fetcher at a
higher tier, a structured-API fetcher) register through :func:`register_page_fetcher`
and are dormant without their keys — the same plugin pattern as the enrichment
adapters. Cost is read generically via :func:`fetcher_cost` (defensive
``getattr``), so the harness never hardcodes a paid vendor's name.

Boundary: OSS. Imports only stdlib / ``cograph_client.*`` / ``httpx``.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from html.parser import HTMLParser
from typing import Optional, Protocol, runtime_checkable
from urllib.parse import urlparse

import httpx
import structlog

from cograph_client.research.types import FetchedPage

logger = structlog.stdlib.get_logger("cograph.research.fetch")

# Cap on bytes read off the wire and chars kept from a page — bounds memory,
# cost, and prompt size. A leaderboard/table answer lives well within this.
_MAX_BYTES = 2_000_000
_MAX_CHARS = 40_000
_DEFAULT_TIMEOUT = 20.0
# Redirects are followed MANUALLY (see StaticHttpFetcher.fetch) so each hop is
# re-validated against the SSRF guard — httpx's own follow_redirects would chase
# a 302 → http://127.0.0.1 past the guard. Bounded to stop redirect loops.
_MAX_REDIRECTS = 5
_UA = "Mozilla/5.0 (compatible; OntaResearchBot/1.0; +https://onta.sh/bot)"


@runtime_checkable
class PageFetcher(Protocol):
    """One rung of the fetch ladder.

    * ``name`` — stable id (``static`` / ``render`` / …).
    * ``tier`` — position on the ladder; lower is cheaper and tried first.
    * ``is_paid`` / ``cost_per_call`` — OPTIONAL cost signal, read via
      :func:`fetcher_cost` (defaulted free). Declared for typing/docs.
    * ``fetch(url, want)`` — retrieve one page. ``want`` is an optional
      free-text hint of what to pull (lets a render/extract fetcher target the
      right region). Must NEVER raise: a failed fetch returns
      ``FetchedPage(ok=False, error=...)``.
    """

    name: str
    tier: int
    is_paid: bool
    cost_per_call: float

    async def fetch(self, url: str, *, want: str = "") -> FetchedPage: ...


# Module-level registry — same shape as register_adapter / register_web_source.
_fetchers: dict[str, PageFetcher] = {}


def register_page_fetcher(fetcher: PageFetcher) -> None:
    """Register (or replace) a fetcher by name. Idempotent — last write wins."""
    _fetchers[fetcher.name] = fetcher


def get_page_fetchers() -> list[PageFetcher]:
    """All registered fetchers, ordered cheapest-tier-first (stable by name on
    ties). This IS the ladder the harness walks."""
    return sorted(
        _fetchers.values(),
        key=lambda f: (int(getattr(f, "tier", 0)), str(getattr(f, "name", ""))),
    )


def reset_page_fetchers() -> None:
    """Clear the registry. For tests."""
    _fetchers.clear()


def register_default_fetchers() -> None:
    """Register the OSS default ladder (the static fetcher at tier 0).

    Called at app boot alongside the other default registrations so a plain OSS
    deployment can fetch pages out of the box. Idempotent. A premium plugin adds
    higher rungs (JS render) on top without disturbing this one.
    """
    register_page_fetcher(StaticHttpFetcher())


def default_ladder() -> list[PageFetcher]:
    """The fetch ladder to use: the registered fetchers, or a lone
    :class:`StaticHttpFetcher` when nothing is registered (so the harness works in
    a bare unit test that never boots the app)."""
    fetchers = get_page_fetchers()
    return fetchers or [StaticHttpFetcher()]


def fetcher_cost(fetcher: PageFetcher) -> tuple[bool, float]:
    """Read a fetcher's cost signal generically (mirrors ``provider_cost``).

    Returns ``(is_paid, cost_per_call)``. Defensive ``getattr`` with free
    defaults; never raises on a malformed ``cost_per_call``.
    """
    try:
        cost = float(getattr(fetcher, "cost_per_call", 0.0) or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    if cost < 0.0:
        cost = 0.0
    is_paid = bool(getattr(fetcher, "is_paid", False)) or cost > 0.0
    return is_paid, cost


# --- SSRF guard --------------------------------------------------------------- #
# The harness fetches URLs chosen by an LLM / discovery provider, so a static
# fetcher is an SSRF surface. Refuse non-http(s) schemes and hosts that resolve
# to loopback / link-local / private / cloud-metadata ranges. Conservative by IP
# literal + obvious names; a deployment behind a locked-down egress proxy is the
# real defense, this is defense-in-depth.
_BLOCKED_HOST_RE = re.compile(
    r"^(localhost|.*\.local|.*\.internal|metadata\.google\.internal)$",
    re.IGNORECASE,
)


def _host_to_ip(host: str) -> Optional[ipaddress._BaseAddress]:
    """Parse a host into an IP across ENCODINGS, or None for a real hostname.

    An SSRF guard that only recognizes the canonical ``127.0.0.1`` literal is
    trivially bypassed: ``2130706433`` (decimal), ``0x7f000001`` (hex),
    ``0177.0.0.1`` (octal) and ``127.1`` (short) all resolve to loopback at
    connect time. We normalize every numeric IPv4 encoding here so the block
    decision sees the real address. Pure parsing, NO DNS — a genuine hostname
    returns None (its resolution is the egress proxy's job), which keeps this
    deterministic offline (tests/CI never hit the network)."""
    h = (host or "").rstrip(".")
    if not h:
        return None
    # 1. A plain literal (IPv4 or IPv6, incl. the [::1] form urlparse strips).
    try:
        return ipaddress.ip_address(h)
    except ValueError:
        pass
    # 2. A bare 32-bit decimal integer host, e.g. "2130706433".
    if h.isdigit():
        try:
            return ipaddress.ip_address(int(h))
        except ValueError:
            return None
    # 3. Hex / octal / short dotted IPv4 forms — let the C numeric parser
    #    (inet_aton, NO DNS) canonicalize them; a hostname raises OSError here.
    try:
        return ipaddress.IPv4Address(socket.inet_aton(h))
    except (OSError, ValueError):
        return None


def _is_blocked_host(host: str) -> bool:
    if not host:
        return True
    if _BLOCKED_HOST_RE.match(host.rstrip(".")):
        return True
    ip = _host_to_ip(host)
    if ip is None:
        return False  # a real hostname; DNS resolution is the egress proxy's job
    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def is_fetchable_url(url: str) -> bool:
    """True when ``url`` is an http(s) URL to a non-internal host."""
    try:
        parsed = urlparse((url or "").strip())
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    return not _is_blocked_host(parsed.hostname)


# --- HTML → text -------------------------------------------------------------- #
class _TextExtractor(HTMLParser):
    """Minimal readability: drop script/style/nav chrome, keep visible text and
    the ``<title>``. Not a full readability port — enough to feed an extractor;
    the premium render tier returns clean markdown for the hard pages."""

    # NB: do NOT skip <head> wholesale — <title> lives there. Its noisy children
    # (script/style) are skipped individually below.
    _SKIP = {"script", "style", "noscript", "template", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self.title = ""

    def handle_starttag(self, tag: str, attrs) -> None:  # noqa: ANN001
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        text = data.strip()
        if text:
            self._parts.append(text)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "\n".join(self._parts)).strip()


def html_to_text(html: str) -> tuple[str, str]:
    """Reduce an HTML document to ``(title, text)``. Never raises."""
    try:
        parser = _TextExtractor()
        parser.feed(html or "")
        parser.close()
        return parser.title.strip(), parser.text()
    except Exception:  # pragma: no cover - parser is lenient, guard anyway
        # Last resort: strip tags with a regex so we still return *something*.
        stripped = re.sub(r"<[^>]+>", " ", html or "")
        return "", re.sub(r"\s+", " ", stripped).strip()


class StaticHttpFetcher:
    """OSS default fetcher (tier 0): a plain ``httpx`` GET + stdlib HTML→text.

    Cheapest rung of the ladder. Reads at most ``_MAX_BYTES`` off the wire,
    follows redirects MANUALLY (re-validating each hop against the SSRF guard),
    refuses internal hosts, and reduces HTML to plain text (JSON/plain bodies
    pass through). Never raises — a failure returns ``FetchedPage(ok=False,
    error=...)`` so the harness can escalate or move on.
    """

    name = "static"
    tier = 0
    is_paid = False
    cost_per_call = 0.0

    def __init__(self, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self._timeout = timeout

    async def fetch(self, url: str, *, want: str = "") -> FetchedPage:
        if not is_fetchable_url(url):
            return FetchedPage(
                url=url, tier=self.name, ok=False, error="blocked or non-http(s) URL"
            )
        content_type = ""
        body = ""
        truncated = False
        current = url
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,  # followed manually below, re-validated per hop
                headers={"User-Agent": _UA, "Accept": "text/html,application/json,*/*"},
            ) as client:
                for _hop in range(_MAX_REDIRECTS + 1):
                    async with client.stream("GET", current) as resp:
                        # Manual redirect handling: re-run the SSRF guard on the
                        # target so a public URL can't 302 us onto an internal one.
                        if resp.is_redirect:
                            location = resp.headers.get("location", "")
                            nxt = (
                                str(httpx.URL(current).join(location))
                                if location
                                else ""
                            )
                            if not nxt or not is_fetchable_url(nxt):
                                return FetchedPage(
                                    url=url,
                                    tier=self.name,
                                    ok=False,
                                    error="redirect to blocked or missing location",
                                )
                            current = nxt
                            continue
                        if resp.status_code >= 400:
                            return FetchedPage(
                                url=url,
                                tier=self.name,
                                ok=False,
                                error=f"HTTP {resp.status_code}",
                            )
                        content_type = resp.headers.get("content-type", "").lower()
                        chunks: list[bytes] = []
                        total = 0
                        async for chunk in resp.aiter_bytes():
                            chunks.append(chunk)
                            total += len(chunk)
                            if total >= _MAX_BYTES:
                                truncated = True
                                break
                        body = b"".join(chunks).decode("utf-8", errors="replace")
                        break
                else:
                    return FetchedPage(
                        url=url, tier=self.name, ok=False, error="too many redirects"
                    )
        except Exception as exc:  # network error, timeout, bad TLS, …
            return FetchedPage(url=url, tier=self.name, ok=False, error=str(exc)[:200])

        if "html" in content_type or (not content_type and "<html" in body[:2000].lower()):
            title, text = html_to_text(body)
        else:
            title, text = "", body  # JSON / plain text / CSV pass through

        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS]
            truncated = True

        return FetchedPage(
            url=url, text=text, title=title, tier=self.name, ok=True, truncated=truncated
        )


__all__ = [
    "FetchedPage",
    "PageFetcher",
    "StaticHttpFetcher",
    "default_ladder",
    "fetcher_cost",
    "get_page_fetchers",
    "html_to_text",
    "is_fetchable_url",
    "register_default_fetchers",
    "register_page_fetcher",
    "reset_page_fetchers",
]
