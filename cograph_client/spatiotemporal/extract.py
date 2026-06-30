"""Datatype-driven extraction of spatio-temporal facts from instance triples.

The write path hands us the exact RDF triples it just inserted; this module turns
them into :class:`SpatioTemporalFact` rows for the secondary index. It is **purely
datatype-driven** — an entity is indexed because it carries a geometry *literal*
(``geo:wktLiteral``), NOT because of what its type is named. A Person with a
birthplace point, an Event with a venue point, and a City with a centroid are all
indexed the same way: we attach the point to whatever entity carries it and never
reclassify the entity as a "location".

Scope of Phase 1:

* **Geometry** comes from a ``geo:wktLiteral`` object (``"POINT(lon lat)"``), the
  first-class form the shared write-path validator now produces for ``geo``
  attributes. An entity with **no** geometry literal yields **no** fact — this
  index requires a point (temporal-only entities are out of scope; see the
  protocol module docstring).
* **Validity time** is attached only when the entity carries an *unambiguous*
  validity interval — an explicit ``valid_from``/``valid_to`` (or ``effective_*``)
  attribute, or a recognized ``start``/``end`` *pair*. A lone date (``founded``,
  ``release_date``, …) is deliberately NOT treated as ``valid_time``: which of an
  entity's several dates bounds its *location* validity is a semantic call we do
  not guess. Such facts are stored with an open (``None``/``None``) validity.

Object encoding: the write path emits a typed literal as ``"<lexical>^^<type-uri>"``
(e.g. ``"POINT(2.29 48.85)^^http://www.opengis.net/ont/geosparql#wktLiteral"``),
exactly what :func:`cograph_client.graph.queries._escape_value` consumes. We split
on the final ``^^`` and only treat the tail as a datatype when it is an ``http`` URI.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import structlog

from cograph_client.spatiotemporal.protocol import SpatioTemporalFact

logger = structlog.stdlib.get_logger("cograph.spatiotemporal.extract")

Triple = tuple[str, str, str]

# Canonical datatype URIs (mirrors resolver.validator.GEO_WKT and the XSD map in
# graph.ontology_queries — duplicated here so this leaf module stays importable on
# its own, without reaching up into resolver/graph).
GEO_WKT = "http://www.opengis.net/ont/geosparql#wktLiteral"
_XSD = "http://www.w3.org/2001/XMLSchema"
_XSD_DATETIME = f"{_XSD}#dateTime"
_XSD_DATE = f"{_XSD}#date"

# Standard RDF predicates used only to denormalize small display fields onto the
# fact (so the hot read path needs no Neptune round-trip). Never used to decide
# whether to index — that is the geometry literal's job.
_RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
_RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
_LABEL_LOCALS = {"label", "name", "title"}

# Validity-interval recognition (local predicate names, snake_cased). We attach
# validity ONLY for an explicit validity bound or a complete start+end pair —
# never for a lone generic date. Kept deliberately small and explicit.
_VALID_FROM_LOCALS = {"valid_from", "effective_from", "valid_start"}
_VALID_TO_LOCALS = {"valid_to", "effective_to", "valid_end"}
_START_LOCALS = {"start_date", "start", "start_time", "begin", "starts_at"}
_END_LOCALS = {"end_date", "end", "end_time", "until", "ends_at"}

_POINT_RE = re.compile(
    r"POINT\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)", re.IGNORECASE
)


def _local_name(uri: str, *, lower: bool = True) -> str:
    """Last path/fragment segment of a URI (``…/types/City/founded`` → ``founded``).

    Lower-cased by default for case-insensitive predicate-name matching; pass
    ``lower=False`` to preserve case for a display value (e.g. a PascalCase type
    name). Returns the input unchanged when it is not a URI.
    """
    if not isinstance(uri, str):
        return ""
    tail = uri.rsplit("#", 1)[-1]
    tail = tail.rsplit("/", 1)[-1]
    return tail.lower() if lower else tail


def _split_typed(obj: str) -> tuple[str, Optional[str]]:
    """Split ``"<lexical>^^<type-uri>"`` into ``(lexical, type_uri)``.

    Only treats the tail as a datatype when it is an ``http`` URI, so a plain
    string literal that happens to contain ``^^`` is left intact (type ``None``).
    """
    if isinstance(obj, str) and "^^" in obj:
        lexical, type_uri = obj.rsplit("^^", 1)
        if type_uri.startswith("http"):
            return lexical, type_uri
    return obj, None


def _parse_point(lexical: str) -> Optional[tuple[float, float]]:
    """Parse ``"POINT(lon lat)"`` → ``(lon, lat)`` within WGS84 range, else None."""
    m = _POINT_RE.search(lexical)
    if not m:
        return None
    try:
        lon, lat = float(m.group(1)), float(m.group(2))
    except ValueError:
        return None
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return None
    return lon, lat


def _parse_dt(lexical: str) -> Optional[datetime]:
    """Parse an ISO-8601 date/datetime (trailing ``Z`` tolerated), else None.

    Always returns a **timezone-aware** datetime: a naive KG value (no offset —
    common, since ingestion normalizes to a bare ISO string) is assumed UTC. This
    keeps the index comparable: PostGIS ``tstzrange`` is tz-based and the
    in-memory backend would raise on a naive-vs-aware comparison if we let naive
    values through.
    """
    if not isinstance(lexical, str):
        return None
    s = lexical.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class _EntityAccumulator:
    """Per-subject scratch state collected in a single pass over the triples."""

    __slots__ = ("point", "label", "type_name", "from_dt", "to_dt", "start_dt", "end_dt")

    def __init__(self) -> None:
        self.point: Optional[tuple[float, float]] = None
        self.label: Optional[str] = None
        self.type_name: Optional[str] = None
        self.from_dt: Optional[datetime] = None
        self.to_dt: Optional[datetime] = None
        self.start_dt: Optional[datetime] = None
        self.end_dt: Optional[datetime] = None

    def validity(self) -> tuple[Optional[datetime], Optional[datetime]]:
        """Resolve (valid_from, valid_to): explicit validity bounds win; otherwise
        a complete start+end pair; otherwise open/open.

        An INVERTED range (from > to — a data-entry slip, or an ``end``-named
        predicate that doesn't actually bound validity) is discarded to an open
        range rather than passed through: PostGIS ``tstzrange(lo, hi)`` raises when
        ``lo > hi``, and because the upsert is batched that error would abort the
        whole write batch's index rows. Better to index the entity with open
        validity than to silently drop a batch of geometry facts.
        """
        if self.from_dt is not None or self.to_dt is not None:
            lo, hi = self.from_dt, self.to_dt
        elif self.start_dt is not None and self.end_dt is not None:
            lo, hi = self.start_dt, self.end_dt
        else:
            return None, None
        if lo is not None and hi is not None and lo > hi:
            return None, None
        return lo, hi


def extract_spatiotemporal_facts(
    triples: list[Triple],
    *,
    tenant_id: str,
    kg_name: str,
) -> list[SpatioTemporalFact]:
    """Build :class:`SpatioTemporalFact` rows for every entity that carries a
    geometry literal among ``triples``.

    Pure and side-effect free. Order-preserving (first geometry per entity wins).
    Entities without a geometry literal are skipped — this index requires a point.
    """
    acc: dict[str, _EntityAccumulator] = {}
    order: list[str] = []

    for s, p, o in triples:
        if not isinstance(s, str) or not isinstance(p, str):
            continue
        ent = acc.get(s)
        if ent is None:
            ent = acc[s] = _EntityAccumulator()
            order.append(s)

        # rdf:type → denormalized type display name (no effect on indexing).
        if p == _RDF_TYPE:
            if ent.type_name is None:
                ent.type_name = _local_name(o, lower=False)
            continue

        lexical, type_uri = _split_typed(o)

        # Geometry — the only signal that makes an entity indexable.
        if type_uri == GEO_WKT and ent.point is None:
            pt = _parse_point(lexical)
            if pt is not None:
                ent.point = pt
            continue

        # Temporal attributes (typed date/dateTime) routed by predicate local name.
        if type_uri in (_XSD_DATETIME, _XSD_DATE):
            local = _local_name(p)
            dt = _parse_dt(lexical)
            if dt is not None:
                if local in _VALID_FROM_LOCALS and ent.from_dt is None:
                    ent.from_dt = dt
                elif local in _VALID_TO_LOCALS and ent.to_dt is None:
                    ent.to_dt = dt
                elif local in _START_LOCALS and ent.start_dt is None:
                    ent.start_dt = dt
                elif local in _END_LOCALS and ent.end_dt is None:
                    ent.end_dt = dt
            continue

        # Label / name for denormalized display.
        if ent.label is None and (p == _RDFS_LABEL or _local_name(p) in _LABEL_LOCALS):
            if lexical and type_uri is None:
                ent.label = lexical

    facts: list[SpatioTemporalFact] = []
    for uri in order:
        ent = acc[uri]
        if ent.point is None:
            continue  # no geometry → not indexable
        lon, lat = ent.point
        valid_from, valid_to = ent.validity()
        attrs: dict[str, str] = {}
        if ent.label:
            attrs["label"] = ent.label
        if ent.type_name:
            attrs["type"] = ent.type_name
        facts.append(
            SpatioTemporalFact(
                entity_uri=uri,
                tenant_id=tenant_id,
                kg_name=kg_name,
                lon=lon,
                lat=lat,
                valid_from=valid_from,
                valid_to=valid_to,
                attrs=attrs,
            )
        )
    return facts
