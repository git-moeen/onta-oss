"""Read-side routing for the spatio-temporal index (ONTA-157 Phase 2).

The write side (Phase 1) keeps the index populated; this module is the read side:
turn a natural-language geo/temporal question into a direct index lookup that
answers WITHOUT a Neptune round-trip. It is **purely the parsing/shaping layer** —
no LLM client, no Neptune, no index handle live here, so it is fully unit-testable.
The orchestration (LLM intent call, anchor resolution, index query) lives in
``NLQueryPipeline`` and is gated behind ``COGRAPH_SPATIAL_ROUTING_ENABLED`` (default
off) so the default query path stays byte-identical for evals.

Flow the pipeline drives:

1. ``looks_spatial(question)`` — a cheap regex pre-gate so a non-spatial question
   never pays for the intent LLM call even when routing is enabled.
2. LLM returns JSON matching :data:`SPATIAL_INTENT_SCHEMA`; ``parse_spatial_intent``
   turns it into an :class:`STQueryIntent` (or ``None`` → fall through to SPARQL).
3. Pipeline resolves the anchor to coords (explicit, or a KG entity's geometry) and
   queries the index; :func:`format_spatial_answer` renders the hits.

Scope of this cut (per the approved plan): ``radius`` and ``bbox`` lookups with an
optional denormalized ``target_type`` filter and an optional temporal predicate.
Hybrid "geo AND <other SPARQL filter>" (VALUES pre-filter) and free-text geocoding
of a non-KG anchor are deliberately out of scope and fall through to the normal
SPARQL path.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel

from cograph_client.spatiotemporal.protocol import STQueryResult

# Cheap pre-gate: proximity/geo phrasing. If a question matches none of these we
# skip the intent LLM call entirely (so routing-enabled non-spatial queries stay
# fast). Intentionally permissive — a false positive just costs one classify call
# that returns is_spatial=false; a false negative silently forgoes the fast path.
_SPATIAL_HINT_RE = re.compile(
    r"\b("
    r"near(?:by|est)?|next to|close(?:st)? to|closest|around|surrounding|"
    r"within|radius|proximity|vicinity|walking distance|"
    r"\d+\s*(?:km|mi|meters?|metres?|kilometers?|kilometres?|miles?)\b|"
    r"bounding box|bbox"
    r")\b",
    re.IGNORECASE,
)


def looks_spatial(question: str) -> bool:
    """True if the question plausibly carries geo/proximity intent (cheap gate)."""
    return bool(question) and bool(_SPATIAL_HINT_RE.search(question))


class SpatialAnchor(BaseModel):
    """The point a radius query is measured from: explicit coords, or a phrase
    naming a KG entity to resolve to its stored geometry."""

    lon: Optional[float] = None
    lat: Optional[float] = None
    entity_description: Optional[str] = None

    def has_coords(self) -> bool:
        return self.lon is not None and self.lat is not None


class STQueryIntent(BaseModel):
    """A parsed spatial query the index can serve directly.

    ``kind="radius"`` needs an ``anchor`` (coords or description) + ``radius_m``.
    ``kind="bbox"`` needs ``bbox`` = (min_lon, min_lat, max_lon, max_lat). Both may
    carry an optional ``target_type`` (post-filter on the denormalized type) and a
    temporal predicate (``as_of`` instant, or ``time_from``/``time_to`` window).
    """

    kind: str  # "radius" | "bbox"
    anchor: Optional[SpatialAnchor] = None
    radius_m: Optional[float] = None
    bbox: Optional[tuple[float, float, float, float]] = None
    target_type: Optional[str] = None
    as_of: Optional[str] = None
    time_from: Optional[str] = None
    time_to: Optional[str] = None


# JSON schema for the intent-detection LLM call. Strict-mode friendly: every field
# is required and nullable via a ["type","null"] union, additionalProperties off —
# matches how _generate_sparql declares its schema across providers.
SPATIAL_INTENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "is_spatial": {"type": "boolean"},
        "kind": {"type": ["string", "null"], "enum": ["radius", "bbox", None]},
        "anchor_lon": {"type": ["number", "null"]},
        "anchor_lat": {"type": ["number", "null"]},
        "anchor_description": {"type": ["string", "null"]},
        "radius_m": {"type": ["number", "null"]},
        "bbox": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "minItems": 4,
            "maxItems": 4,
        },
        "target_type": {"type": ["string", "null"]},
        "as_of": {"type": ["string", "null"]},
        "time_from": {"type": ["string", "null"]},
        "time_to": {"type": ["string", "null"]},
    },
    "required": [
        "is_spatial",
        "kind",
        "anchor_lon",
        "anchor_lat",
        "anchor_description",
        "radius_m",
        "bbox",
        "target_type",
        "as_of",
        "time_from",
        "time_to",
    ],
    "additionalProperties": False,
}

SPATIAL_INTENT_SYSTEM = (
    "You classify a user's question about a knowledge graph as a SPATIAL lookup or "
    "not, and extract its parameters. Answer ONLY with the JSON schema.\n\n"
    "Set is_spatial=true ONLY when the question asks for entities by geographic "
    "proximity or area:\n"
    "- radius: 'within 5 km of X', 'near X', 'closest to X' → kind='radius', set "
    "radius_m in METERS (convert km/miles; if no distance is given pick a sensible "
    "default like 5000), and set the anchor.\n"
    "- bbox: an explicit lon/lat box → kind='bbox', set bbox=[min_lon,min_lat,"
    "max_lon,max_lat].\n"
    "Anchor: if the question gives explicit coordinates use anchor_lon/anchor_lat; "
    "otherwise put the phrase naming the reference place in anchor_description "
    "(e.g. 'the Eiffel Tower', 'downtown').\n"
    "target_type: if the question asks for a specific entity type ('restaurants "
    "near X') put that type name, else null.\n"
    "Temporal: if the question constrains time, set as_of (a single instant) OR "
    "time_from/time_to (a range), ISO-8601; else null.\n"
    "If the question is NOT a proximity/area lookup (e.g. a count, a join, an "
    "attribute filter with no geography), set is_spatial=false and everything else "
    "null."
)


def parse_spatial_intent(raw: dict[str, Any]) -> Optional[STQueryIntent]:
    """Turn the intent LLM's JSON into an :class:`STQueryIntent`, or ``None``.

    ``None`` means "not a servable spatial lookup" — the caller falls through to the
    normal SPARQL path. Returns ``None`` when is_spatial is false, the kind is
    unknown, or the required parameters for that kind are missing/degenerate, so a
    half-formed intent never reaches the index.
    """
    if not isinstance(raw, dict) or not raw.get("is_spatial"):
        return None
    kind = raw.get("kind")
    if kind == "radius":
        radius_m = raw.get("radius_m")
        if not isinstance(radius_m, (int, float)) or radius_m <= 0:
            return None
        anchor = SpatialAnchor(
            lon=_num(raw.get("anchor_lon")),
            lat=_num(raw.get("anchor_lat")),
            entity_description=_str(raw.get("anchor_description")),
        )
        if not anchor.has_coords() and not anchor.entity_description:
            return None  # nothing to measure from
        return STQueryIntent(
            kind="radius",
            anchor=anchor,
            radius_m=float(radius_m),
            target_type=_str(raw.get("target_type")),
            as_of=_str(raw.get("as_of")),
            time_from=_str(raw.get("time_from")),
            time_to=_str(raw.get("time_to")),
        )
    if kind == "bbox":
        bbox = raw.get("bbox")
        if not _valid_bbox(bbox):
            return None
        return STQueryIntent(
            kind="bbox",
            bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
            target_type=_str(raw.get("target_type")),
            as_of=_str(raw.get("as_of")),
            time_from=_str(raw.get("time_from")),
            time_to=_str(raw.get("time_to")),
        )
    return None


def filter_by_type(
    hits: list[STQueryResult], target_type: Optional[str]
) -> list[STQueryResult]:
    """Keep only hits whose denormalized ``type`` matches ``target_type`` (case-
    insensitive). No target → unchanged. A hit lacking a ``type`` attr is dropped
    only when a target is requested (we can't confirm it matches)."""
    if not target_type:
        return hits
    want = target_type.strip().lower()
    return [h for h in hits if str(h.attrs.get("type", "")).strip().lower() == want]


def format_spatial_answer(
    hits: list[STQueryResult], intent: STQueryIntent
) -> str:
    """Render index hits into a concise human answer.

    Uses the denormalized ``label`` (falling back to the entity URI) and ``type``.
    Order is the index's; caps the listed names so a huge result set stays readable
    while still reporting the true total.
    """
    if not hits:
        where = _where_phrase(intent)
        return f"No entities found {where}."
    names = []
    for h in hits:
        label = h.attrs.get("label") or h.entity_uri
        typ = h.attrs.get("type")
        names.append(f"{label} ({typ})" if typ else str(label))
    shown = names[:20]
    more = len(names) - len(shown)
    head = f"Found {len(hits)} {_subject(intent)} {_where_phrase(intent)}: "
    tail = ", ".join(shown) + (f", and {more} more" if more > 0 else "")
    return head + tail


# --------------------------------------------------------------------------- util
def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) else None


def _str(v: Any) -> Optional[str]:
    return v if isinstance(v, str) and v.strip() else None


def _valid_bbox(v: Any) -> bool:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return False
    if not all(isinstance(x, (int, float)) for x in v):
        return False
    min_lon, min_lat, max_lon, max_lat = v
    return min_lon <= max_lon and min_lat <= max_lat


def _subject(intent: STQueryIntent) -> str:
    return (intent.target_type + "(s)") if intent.target_type else "entit(ies)"


def _where_phrase(intent: STQueryIntent) -> str:
    if intent.kind == "radius" and intent.radius_m:
        anchor = ""
        if intent.anchor and intent.anchor.entity_description:
            anchor = f" of {intent.anchor.entity_description}"
        elif intent.anchor and intent.anchor.has_coords():
            anchor = f" of ({intent.anchor.lon}, {intent.anchor.lat})"
        return f"within {_fmt_dist(intent.radius_m)}{anchor}"
    if intent.kind == "bbox":
        return "in the requested area"
    return "nearby"


def _fmt_dist(m: float) -> str:
    return f"{m / 1000:g} km" if m >= 1000 else f"{m:g} m"
