"""Tests for ontology relationship registration and type-reference handling.

Covers:
1. _datatype_to_xsd maps type names to type URIs (not just XSD primitives)
2. _xsd_to_datatype detects type URI ranges and returns the type name
3. Relationship triples are registered as object properties in the ontology during ingestion
4. Extraction prompt includes existing types for type placement (same_as / parent_type)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cograph_client.graph.ontology_queries import (
    PRIMITIVE_TYPES,
    _datatype_to_xsd,
    insert_attribute,
    type_uri,
)
from cograph_client.api.routes.ontology import _xsd_to_datatype, TYPE_URI_PREFIX


# ---------------------------------------------------------------------------
# 1. _datatype_to_xsd: type-reference support
# ---------------------------------------------------------------------------

class TestDatatypeToXsd:
    def test_primitive_string(self):
        assert "XMLSchema#string" in _datatype_to_xsd("string")

    def test_primitive_integer(self):
        assert "XMLSchema#integer" in _datatype_to_xsd("integer")

    def test_primitive_float(self):
        assert "XMLSchema#float" in _datatype_to_xsd("float")

    def test_primitive_boolean(self):
        assert "XMLSchema#boolean" in _datatype_to_xsd("boolean")

    def test_primitive_datetime(self):
        assert "XMLSchema#dateTime" in _datatype_to_xsd("datetime")

    def test_primitive_uri(self):
        assert "Resource" in _datatype_to_xsd("uri")

    def test_type_reference_person(self):
        """Non-primitive datatype should map to a type URI."""
        result = _datatype_to_xsd("Person")
        assert result == type_uri("Person")
        assert result == "https://cograph.tech/types/Person"

    def test_type_reference_place(self):
        result = _datatype_to_xsd("Place")
        assert result == "https://cograph.tech/types/Place"

    def test_type_reference_not_xsd(self):
        """Type references should NOT produce XSD URIs."""
        result = _datatype_to_xsd("Vehicle")
        assert "XMLSchema" not in result
        assert "cograph.tech/types/Vehicle" in result


# ---------------------------------------------------------------------------
# 2. _xsd_to_datatype: reverse mapping detects type URIs
# ---------------------------------------------------------------------------

class TestXsdToDatatype:
    def test_xsd_string(self):
        assert _xsd_to_datatype("http://www.w3.org/2001/XMLSchema#string") == "string"

    def test_xsd_integer(self):
        assert _xsd_to_datatype("http://www.w3.org/2001/XMLSchema#integer") == "integer"

    def test_xsd_datetime(self):
        assert _xsd_to_datatype("http://www.w3.org/2001/XMLSchema#dateTime") == "datetime"

    def test_empty_string(self):
        assert _xsd_to_datatype("") == "string"

    def test_type_uri_person(self):
        """Type URI ranges should return the type name, not 'string'."""
        result = _xsd_to_datatype("https://cograph.tech/types/Person")
        assert result == "Person"

    def test_type_uri_place(self):
        result = _xsd_to_datatype("https://cograph.tech/types/Place")
        assert result == "Place"

    def test_type_uri_vehicle(self):
        result = _xsd_to_datatype("https://cograph.tech/types/Vehicle")
        assert result == "Vehicle"

    def test_type_uri_not_primitive(self):
        """Type references should not be returned as primitives."""
        result = _xsd_to_datatype("https://cograph.tech/types/Residence")
        assert result not in PRIMITIVE_TYPES
        assert result == "Residence"


# ---------------------------------------------------------------------------
# 3. insert_attribute with type reference produces type URI in range
# ---------------------------------------------------------------------------

class TestInsertAttributeTypeRef:
    def test_attribute_with_primitive_range(self):
        sparql = insert_attribute("https://cograph.tech/graphs/test", "Property", "price", "", "integer")
        assert "XMLSchema#integer" in sparql

    def test_attribute_with_type_reference_range(self):
        """When datatype is a type name, range should be the type URI."""
        sparql = insert_attribute("https://cograph.tech/graphs/test", "Property", "location", "", "Place")
        assert "cograph.tech/types/Place" in sparql
        assert "XMLSchema" not in sparql

    def test_attribute_with_type_reference_person(self):
        sparql = insert_attribute("https://cograph.tech/graphs/test", "Transaction", "buyer", "", "Person")
        assert "cograph.tech/types/Person" in sparql


# ---------------------------------------------------------------------------
# 4. PRIMITIVE_TYPES set is correct
# ---------------------------------------------------------------------------

class TestPrimitiveTypes:
    def test_contains_all_primitives(self):
        # `geo` (geo:wktLiteral, WKT point) is a primitive attribute range — its
        # values are literals, not entity references.
        expected = {"string", "integer", "float", "boolean", "datetime", "uri", "geo"}
        assert PRIMITIVE_TYPES == expected

    def test_type_names_not_primitive(self):
        for name in ["Person", "Place", "Vehicle", "Property", "City"]:
            assert name not in PRIMITIVE_TYPES


# ---------------------------------------------------------------------------
# 5. Roundtrip: insert_attribute → _xsd_to_datatype preserves type reference
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_primitive_roundtrip(self):
        """Primitive datatypes survive insert → parse roundtrip."""
        for dt in ["string", "integer", "float", "boolean", "datetime"]:
            xsd = _datatype_to_xsd(dt)
            assert _xsd_to_datatype(xsd) == dt

    def test_type_reference_roundtrip(self):
        """Type-reference datatypes survive insert → parse roundtrip."""
        for type_name in ["Person", "Place", "Vehicle", "Residence"]:
            xsd = _datatype_to_xsd(type_name)
            assert _xsd_to_datatype(xsd) == type_name
