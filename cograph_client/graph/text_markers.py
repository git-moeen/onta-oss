"""Free-text attribute markers: candidacy classifier + per-tenant marker cache.

ONTA-177 (automated free-text candidacy for the semantic instance index,
ONTA-173). Two concerns live here, both deliberately in the ``graph/`` layer so
every consumer — the SchemaResolver seam, the CSV REASON conversion, the
``kg_writer`` refresh hook, and the future reconciler-side default heuristic
(ONTA-181) — can import them without layering violations:

1. :func:`classify_text_candidacy` — the NAME-BLIND candidacy classifier.
   The profiler's ``ValueShape.TEXT`` *proposes* candidates from value
   statistics alone (ADR 0003 litmus: no name inspection below the LLM layer);
   this function is the shared "propose" step, reusing the profiler's own
   ``_value_shape`` so the TEXT thresholds have exactly one source of truth.
   Unambiguously long prose is marked directly; borderline text shapes are
   returned as AMBIGUOUS for the LLM REASON layer — the only layer where the
   attribute NAME may be consulted — to adjudicate.

2. The per-tenant ``{attribute predicate URI -> is_free_text}`` map, read from
   the ``<attr> <onto/textKind> ?kind`` markers the schema pass writes.
   Query-side consumers (routing a query's type filter to the semantic index,
   ONTA-176) need this on every request, so it is TTL-cached (~60s — the
   multi-task safety valve: another worker/process writing markers is picked
   up within one TTL even if no local invalidation fired) and explicitly
   invalidated at the marker WRITE sites (the schema pass's candidacy seams
   and the reconciler's default heuristic) right after they upsert markers —
   NOT on every converged write, which would defeat the TTL on the hot path.

   Map semantics: ``True`` exactly when the kind is ``"free_text"``; any other
   kind (e.g. the durable decided-no ``"not_text"``) reads back as ``False``
   **while still being PRESENT in the map** — presence means "candidacy was
   decided", which is what lets the reconciler skip re-sampling and prevents
   the name-blind auto tier from overruling an LLM's explicit NO.
"""

from __future__ import annotations

import os
import time
from collections import Counter
from enum import Enum
from typing import Any, Sequence

import structlog

from cograph_client.graph.ontology_queries import (
    TEXT_KIND_FREE_TEXT,
    text_kind_map_query,
)
from cograph_client.graph.queries import tenant_graph_uri

logger = structlog.stdlib.get_logger("cograph.graph.text_markers")

#: Average value length (chars) at or above which a TEXT-shaped attribute is
#: UNAMBIGUOUSLY free-running prose and is marked without LLM adjudication.
#: Below it (but still TEXT-shaped: multi-word, avg > 25 chars) the values
#: could equally be postal addresses, organization names, or composite titles —
#: that is the AMBIGUOUS band the REASON pass adjudicates using the attribute
#: name (ONTA-177). Structural threshold, statable without domain nouns
#: (ADR 0003 litmus).
AUTO_FREE_TEXT_MIN_AVG_LEN = float(
    os.environ.get("COGRAPH_FREE_TEXT_AUTO_MIN_AVG_LEN", "120")
)


class TextCandidacy(str, Enum):
    """Verdict of the name-blind candidacy classifier for one attribute."""

    #: Not text-shaped (codes, labels, numbers, dates, empty) — never a
    #: semantic-index candidate; the LLM is not consulted.
    NOT_CANDIDATE = "not_candidate"
    #: Text-shaped but borderline — the REASON pass adjudicates by NAME.
    AMBIGUOUS = "ambiguous"
    #: Unambiguously long free-running prose — marked directly, no LLM.
    FREE_TEXT = "free_text"


def classify_text_candidacy(values: Sequence[Any]) -> TextCandidacy:
    """Classify an attribute's semantic-index candidacy from its VALUES only.

    Name-blind by construction (ADR 0003 litmus — the profiler layer never
    inspects attribute names; only the LLM layer may). Reuses the profiler's
    ``_value_shape`` so the TEXT shape thresholds (space fraction, average
    length) can never drift from Pass A's; only ``ValueShape.TEXT`` columns are
    candidates at all — the profiler *proposes*, everything else adjudicates
    within that proposal (ONTA-177).

    Cells may be anything JSON delivers (numbers, None, …); non-strings are
    stringified and empties dropped, mirroring the profiler's normalization.
    """
    # Lazy import, mirroring kg_writer's precedent for reaching into a higher
    # layer from graph/: resolver.profiler is stdlib+models-only (no cycle
    # today), but keeping the import inside the function guarantees this
    # module stays importable by kg_writer even if that ever changes.
    from cograph_client.resolver.models import ValueShape
    from cograph_client.resolver.profiler import _value_shape

    cleaned: list[str] = []
    for v in values:
        if v is None:
            continue
        s = v.strip() if isinstance(v, str) else str(v).strip()
        if s:
            cleaned.append(s)
    if not cleaned:
        return TextCandidacy.NOT_CANDIDATE
    if _value_shape(Counter(cleaned), len(cleaned)) is not ValueShape.TEXT:
        return TextCandidacy.NOT_CANDIDATE
    avg_len = sum(len(s) for s in cleaned) / len(cleaned)
    if avg_len >= AUTO_FREE_TEXT_MIN_AVG_LEN:
        return TextCandidacy.FREE_TEXT
    return TextCandidacy.AMBIGUOUS


# --- Per-tenant {predicate URI -> is_free_text} cache ------------------------

#: tenant_id -> (monotonic fetch time, marker map). Module-level like the
#: NL-planning ontology cache; TTL keeps concurrent workers/tasks eventually
#: consistent even when a write happened in a process that couldn't invalidate
#: this one (multi-task safety).
_cache: dict[str, tuple[float, dict[str, bool]]] = {}


def _ttl_s() -> float:
    """TTL in seconds (env-overridable; read per call so tests can tune it)."""
    return float(os.environ.get("COGRAPH_TEXT_MARKER_TTL_S", "60"))


async def get_free_text_map(neptune, tenant_id: str) -> dict[str, bool]:
    """Return ``{attribute predicate URI -> is_free_text}`` for one tenant.

    Reads every ``<attr> <onto/textKind> ?kind`` marker in the tenant's base
    (ontology) graph; ``True`` exactly when the kind is ``"free_text"``. A
    predicate absent from the map carries NO verdict (candidacy was never
    decided — e.g. the attribute predates ONTA-177 or arrived via a writer the
    schema pass never saw; ONTA-181's reconciler-side heuristic covers those).

    TTL-cached (~60s) per tenant. Best-effort on the fetch: a Neptune failure
    logs a warning and returns ``{}`` WITHOUT caching it, so the next call
    retries instead of pinning an empty map for a full TTL.
    """
    now = time.monotonic()
    hit = _cache.get(tenant_id)
    if hit is not None and now - hit[0] < _ttl_s():
        return hit[1]
    try:
        from cograph_client.graph.parser import parse_sparql_results

        raw = await neptune.query(text_kind_map_query(tenant_graph_uri(tenant_id)))
        _, bindings = parse_sparql_results(raw)
        marker_map = {
            row["attr"]: row.get("kind") == TEXT_KIND_FREE_TEXT
            for row in bindings
            if row.get("attr")
        }
    except Exception:  # noqa: BLE001 — a marker-map hiccup must never fail the caller
        logger.warning("text_marker_map_fetch_failed", tenant_id=tenant_id, exc_info=True)
        return {}
    _cache[tenant_id] = (now, marker_map)
    return marker_map


def invalidate(tenant_id: str) -> None:
    """Drop one tenant's cached marker map (called by the marker WRITE sites —
    the schema pass's candidacy seams and the reconciler's default heuristic —
    so a just-written marker is visible before the TTL expires)."""
    _cache.pop(tenant_id, None)


def invalidate_for_graph(graph_uri: str) -> None:
    """Drop the cached marker map for the tenant owning ``graph_uri``.

    Convenience for marker write sites that hold a graph URI rather than a
    bare tenant id (the SchemaResolver seams receive the tenant ONTOLOGY graph
    ``https://cograph.tech/graphs/{tenant}``). Deriving the tenant here keeps
    the URI-shape knowledge next to the cache instead of in every caller. An
    unrecognized shape over-invalidates (drops everything) — safe, since the
    only cost is one refetch per tenant on the next read.
    """
    prefix = tenant_graph_uri("")  # "https://cograph.tech/graphs/"
    if isinstance(graph_uri, str) and graph_uri.startswith(prefix):
        tenant_id = graph_uri[len(prefix):].split("/", 1)[0]
        if tenant_id:
            invalidate(tenant_id)
            return
    invalidate_all()


def invalidate_all() -> None:
    """Drop every tenant's cached marker map."""
    _cache.clear()


def reset_for_tests() -> None:
    """Test hook: restore pristine module state between tests."""
    _cache.clear()
