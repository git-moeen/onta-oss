"""Schema-on-write validation — every value checked before Neptune insertion.

Conforms → insert. Coercible → coerce + insert. Invalid → reject + log.
"""

from __future__ import annotations

import re
from datetime import datetime

import structlog

from cograph_client.resolver.models import RejectedValue, ValidatedTriple, ValidationOutcome

logger = structlog.stdlib.get_logger("cograph.resolver.validator")


def coerce_value(value: str, target_datatype: str) -> str | None:
    """Try to coerce a value to the target datatype.

    Returns the coerced string representation, or None if not possible.
    """
    try:
        match target_datatype:
            case "string":
                return str(value)
            case "integer":
                return str(int(float(value)))
            case "float":
                return str(float(value))
            case "boolean":
                lower = value.lower().strip()
                if lower in ("true", "1", "yes", "on"):
                    return "true"
                if lower in ("false", "0", "no", "off"):
                    return "false"
                return None
            case "datetime":
                return _parse_datetime(value)
            case "geo":
                # "lat,lon" (Wikidata / combined-column form) or a WKT POINT both
                # canonicalize to "POINT(lon lat)"; anything out of WGS84 range or
                # unparseable → reject (None).
                return _to_wkt_point(value)
            case "uri":
                if value.startswith("http://") or value.startswith("https://"):
                    return value
                return None
            case _:
                return str(value)
    except (ValueError, TypeError):
        return None


def _parse_datetime(value: str) -> str | None:
    """Try common datetime formats, return ISO-8601 or None."""
    formats = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(value.strip(), fmt)
            return dt.isoformat()
        except ValueError:
            continue

    # Try ISO parse as last resort
    try:
        dt = datetime.fromisoformat(value.strip())
        return dt.isoformat()
    except ValueError:
        return None


def validate_value(value: str, datatype: str) -> bool:
    """Check if a value conforms to the expected datatype without coercion."""
    match datatype:
        case "string":
            return True
        case "integer":
            return bool(re.match(r"^-?\d+$", value.strip()))
        case "float":
            return bool(re.match(r"^-?\d+(\.\d+)?$", value.strip()))
        case "boolean":
            return value.lower().strip() in ("true", "false")
        case "datetime":
            return _parse_datetime(value) is not None
        case "geo":
            # Conforms only when already a valid WGS84 WKT POINT; the "lat,lon"
            # form is handled by coercion (so it is normalized, not stored verbatim).
            return _to_wkt_point(value) is not None and bool(
                _WKT_POINT_RE.match(value.strip())
            )
        case "uri":
            return value.startswith("http://") or value.startswith("https://")
        case _:
            return True


XSD = "http://www.w3.org/2001/XMLSchema"
# OGC GeoSPARQL WKT literal range for `geo` attributes (mirrors
# graph.ontology_queries._DATATYPE_TO_XSD["geo"]).
GEO_WKT = "http://www.opengis.net/ont/geosparql#wktLiteral"

_DATATYPE_TO_XSD = {
    "integer": f"{XSD}#integer",
    "float": f"{XSD}#float",
    "boolean": f"{XSD}#boolean",
    "datetime": f"{XSD}#dateTime",
    "geo": GEO_WKT,
}

# A WKT point "POINT(lon lat)" (WKT order is lon-then-lat) and a plain "lat,lon"
# pair (the Wikidata globecoordinate / combined-column form, which is lat-then-lon).
_WKT_POINT_RE = re.compile(
    r"^\s*POINT\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)\s*$",
    re.IGNORECASE,
)
_LATLON_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$"
)


def _to_wkt_point(value: str) -> str | None:
    """Canonicalize a coordinate to ``POINT(lon lat)`` WKT, or ``None``.

    Accepts either a WKT ``POINT(lon lat)`` or a ``"lat,lon"`` pair (the form the
    Wikidata adapter and combined coordinate columns produce). The original
    numeric lexical forms are preserved (no float re-formatting → no precision
    loss); only WGS84 range is enforced (lon ∈ [-180,180], lat ∈ [-90,90]).
    """
    if not isinstance(value, str):
        return None
    m = _WKT_POINT_RE.match(value)
    if m:
        lon_s, lat_s = m.group(1), m.group(2)
    else:
        m = _LATLON_RE.match(value)
        if not m:
            return None
        lat_s, lon_s = m.group(1), m.group(2)  # comma form is "lat,lon"
    try:
        lon, lat = float(lon_s), float(lat_s)
    except ValueError:
        return None
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return None
    return f"POINT({lon_s} {lat_s})"


def _typed_value(value: str, datatype: str) -> str:
    """Append XSD type annotation for non-string datatypes.

    Returns "500000^^http://www.w3.org/2001/XMLSchema#integer" for integers, etc.
    Plain strings return as-is (no annotation needed).
    Datetime values are normalized to full ISO-8601 with time component so that
    Neptune xsd:dateTime comparisons work correctly. ``geo`` values are normalized
    to canonical ``POINT(lon lat)`` WKT so the spatio-temporal index can parse them.
    """
    xsd = _DATATYPE_TO_XSD.get(datatype)
    if xsd:
        if datatype == "datetime":
            # Normalize to full ISO-8601 so Neptune dateTime comparisons work
            normalized = _parse_datetime(value)
            if normalized:
                value = normalized
        elif datatype == "geo":
            value = _to_wkt_point(value) or value
        return f"{value}^^{xsd}"
    return value


def validate_triple(
    subject: str,
    predicate: str,
    value: str,
    expected_datatype: str,
    entity_id: str = "",
    attribute_name: str = "",
) -> ValidatedTriple | RejectedValue:
    """Validate a single triple value against the expected datatype.

    Returns ValidatedTriple (OK or COERCED) or RejectedValue.
    Values are annotated with XSD types for non-string datatypes.
    """
    # Check if value conforms as-is
    if validate_value(value, expected_datatype):
        return ValidatedTriple(
            subject=subject,
            predicate=predicate,
            object=_typed_value(value, expected_datatype),
            outcome=ValidationOutcome.OK,
        )

    # Try coercion
    coerced = coerce_value(value, expected_datatype)
    if coerced is not None:
        logger.info(
            "value_coerced",
            entity=entity_id,
            attr=attribute_name,
            original=value,
            coerced=coerced,
            datatype=expected_datatype,
        )
        return ValidatedTriple(
            subject=subject,
            predicate=predicate,
            object=_typed_value(coerced, expected_datatype),
            outcome=ValidationOutcome.COERCED,
            original_value=value,
        )

    # Reject
    logger.warning(
        "value_rejected",
        entity=entity_id,
        attr=attribute_name,
        value=value,
        expected=expected_datatype,
    )
    return RejectedValue(
        entity_id=entity_id,
        attribute=attribute_name,
        value=value,
        expected_datatype=expected_datatype,
        reason=f"Cannot coerce '{value}' to {expected_datatype}",
    )
