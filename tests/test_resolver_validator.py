"""Tests for the schema-on-write validator."""

import pytest

from cograph_client.resolver.models import RejectedValue, ValidatedTriple, ValidationOutcome
from cograph_client.resolver.validator import coerce_value, validate_triple, validate_value


class TestCoerceValue:
    def test_string_passthrough(self):
        assert coerce_value("hello", "string") == "hello"

    def test_integer_from_string(self):
        assert coerce_value("42", "integer") == "42"

    def test_integer_from_float_string(self):
        assert coerce_value("42.7", "integer") == "42"

    def test_float_from_string(self):
        assert coerce_value("3.14", "float") == "3.14"

    def test_boolean_true_variants(self):
        for val in ["true", "1", "yes", "on", "True", "YES"]:
            assert coerce_value(val, "boolean") == "true"

    def test_boolean_false_variants(self):
        for val in ["false", "0", "no", "off", "False", "NO"]:
            assert coerce_value(val, "boolean") == "false"

    def test_boolean_invalid(self):
        assert coerce_value("maybe", "boolean") is None

    def test_datetime_iso(self):
        result = coerce_value("2026-04-04", "datetime")
        assert result is not None
        assert "2026-04-04" in result

    def test_datetime_us_format(self):
        result = coerce_value("04/04/2026", "datetime")
        assert result is not None

    def test_datetime_invalid(self):
        assert coerce_value("not-a-date", "datetime") is None

    def test_uri_valid(self):
        assert coerce_value("https://example.com", "uri") == "https://example.com"

    def test_uri_invalid(self):
        assert coerce_value("not-a-uri", "uri") is None

    def test_integer_non_numeric(self):
        assert coerce_value("abc", "integer") is None


class TestValidateValue:
    def test_string_always_valid(self):
        assert validate_value("anything", "string") is True

    def test_integer_valid(self):
        assert validate_value("42", "integer") is True
        assert validate_value("-7", "integer") is True

    def test_integer_invalid(self):
        assert validate_value("42.5", "integer") is False
        assert validate_value("abc", "integer") is False

    def test_float_valid(self):
        assert validate_value("3.14", "float") is True
        assert validate_value("42", "float") is True

    def test_boolean_valid(self):
        assert validate_value("true", "boolean") is True
        assert validate_value("false", "boolean") is True

    def test_boolean_invalid(self):
        assert validate_value("yes", "boolean") is False


class TestValidateTriple:
    def test_valid_triple(self):
        result = validate_triple(
            "s", "p", "42", "integer", entity_id="e1", attribute_name="count",
        )
        assert isinstance(result, ValidatedTriple)
        assert result.outcome == ValidationOutcome.OK

    def test_coerced_triple(self):
        result = validate_triple(
            "s", "p", "42.7", "integer", entity_id="e1", attribute_name="count",
        )
        assert isinstance(result, ValidatedTriple)
        assert result.outcome == ValidationOutcome.COERCED
        assert result.object == "42^^http://www.w3.org/2001/XMLSchema#integer"
        assert result.original_value == "42.7"

    def test_rejected_triple(self):
        result = validate_triple(
            "s", "p", "not-a-number", "integer", entity_id="e1", attribute_name="count",
        )
        assert isinstance(result, RejectedValue)
        assert result.expected_datatype == "integer"


GEO_WKT = "http://www.opengis.net/ont/geosparql#wktLiteral"


class TestGeoDatatype:
    """`geo` coerces a 'lat,lon' pair or a WKT POINT to a typed geo:wktLiteral."""

    def test_validate_value_wkt_conforms(self):
        # An already-canonical WKT POINT conforms (no coercion).
        assert validate_value("POINT(2.29 48.85)", "geo") is True

    def test_validate_value_latlon_not_conforming(self):
        # "lat,lon" is coercible, not conforming — so it gets normalized, not stored verbatim.
        assert validate_value("48.85,2.29", "geo") is False

    def test_validate_value_out_of_range(self):
        # lat 1920 is out of WGS84 range → not a coordinate.
        assert validate_value("1920,1080", "geo") is False

    def test_coerce_latlon_to_wkt(self):
        # Comma form is lat,lon; WKT is lon-then-lat.
        assert coerce_value("48.85,2.29", "geo") == "POINT(2.29 48.85)"

    def test_coerce_wkt_passthrough(self):
        assert coerce_value("POINT(2.29 48.85)", "geo") == "POINT(2.29 48.85)"

    def test_coerce_rejects_non_coord(self):
        assert coerce_value("Paris", "geo") is None
        assert coerce_value("1920,1080", "geo") is None  # out of range

    def test_triple_latlon_coerced_to_typed_wkt(self):
        result = validate_triple("s", "p", "48.85,2.29", "geo")
        assert isinstance(result, ValidatedTriple)
        assert result.outcome == ValidationOutcome.COERCED
        assert result.object == f"POINT(2.29 48.85)^^{GEO_WKT}"

    def test_triple_wkt_ok_and_typed(self):
        result = validate_triple("s", "p", "POINT(2.29 48.85)", "geo")
        assert isinstance(result, ValidatedTriple)
        assert result.outcome == ValidationOutcome.OK
        assert result.object == f"POINT(2.29 48.85)^^{GEO_WKT}"

    def test_triple_non_coord_rejected(self):
        result = validate_triple("s", "p", "somewhere", "geo")
        assert isinstance(result, RejectedValue)

    def test_precision_preserved(self):
        # No float re-formatting → exact lexical precision is kept.
        assert coerce_value("-33.8688197,151.2092955", "geo") == (
            "POINT(151.2092955 -33.8688197)"
        )
