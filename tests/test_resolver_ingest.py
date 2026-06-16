"""Integration test for the ingest API endpoint."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_anthropic_response():
    """Mock Anthropic response with extracted entities."""
    def make_response(text: str):
        mock = AsyncMock()
        content_block = MagicMock()
        content_block.text = text
        mock.content = [content_block]
        return mock
    return make_response


def test_ingest_endpoint_exists(client, auth_headers):
    """Verify the endpoint is registered and requires auth."""
    response = client.post("/graphs/test-tenant/ingest")
    assert response.status_code != 404


def test_ingest_requires_auth(client):
    response = client.post(
        "/graphs/test-tenant/ingest",
        json={"content": "test"},
    )
    assert response.status_code == 401


def test_ingest_requires_content(client, auth_headers):
    response = client.post(
        "/graphs/test-tenant/ingest",
        json={},
        headers=auth_headers,
    )
    assert response.status_code == 422


@patch("cograph_client.api.routes.ingest.SchemaResolver")
def test_ingest_returns_result(mock_resolver_cls, client, auth_headers):
    """Test that ingest endpoint calls resolver and returns result."""
    from cograph_client.resolver.models import IngestResult
    mock_instance = AsyncMock()
    mock_instance.ingest.return_value = IngestResult(
        entities_extracted=2,
        entities_resolved=2,
        triples_inserted=10,
        types_created=["Property"],
        attributes_added=["Property.price"],
    )
    mock_resolver_cls.return_value = mock_instance

    response = client.post(
        "/graphs/test-tenant/ingest",
        json={"content": "A 3-bedroom house at 123 Main St for $500,000", "source": "test"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["entities_extracted"] == 2
    assert data["triples_inserted"] == 10
    assert "Property" in data["types_created"]


# --- COG-52 / ADR 0003 Pass D: /ingest/csv/rows pre-registration -------------
#
# The pre-registration loop must write promoted types and EVERY core slot to
# the tenant ontology — including slots with ZERO data in the file — and mark
# each with a coreSlot triple (the enrichment work-queue hook, ADR 0003 §3).
# SchemaResolver is mocked (no LLM, no row insertion under test); the
# generated SPARQL is asserted on the mocked Neptune client, in the same
# string-assertion style as tests/test_ontology_queries.py.


def _extension_mapping() -> dict:
    """A posted mapping carrying one promotion in the canonical
    dependent-identifier shape. issued_by/identifies/code have NO backing
    column — zero-data slots that must still be declared."""
    return {
        "entity_type": "Item",
        "columns": [
            {"column_name": "name", "role": "type_id", "datatype": "string",
             "attribute_name": "name"},
            {"column_name": "code", "role": "attribute", "datatype": "string",
             "attribute_name": "code"},
        ],
        "ontology_extensions": {
            "types": [
                {"type_name": "Code", "promoted_from_attribute": "code",
                 "held_for_review": True,  # applied anyway: confirm gate is client-side
                 "core_slots": [
                     {"name": "issued_by", "kind": "relationship", "target_type": "Issuer",
                      "why": "an identifier exists only relative to its issuer",
                      "dataset_constant": None},
                     {"name": "identifies", "kind": "relationship", "target_type": "Item"},
                     {"name": "code", "kind": "attribute"},
                 ],
                 "rejected": [{"name": "notes", "failed_test": "existence"}]},
            ],
        },
    }


@patch("cograph_client.api.routes.ingest.SchemaResolver")
def test_csv_rows_preregisters_promoted_types_and_core_slots(
    mock_resolver_cls, client, auth_headers, mock_neptune,
):
    from cograph_client.resolver.models import IngestResult

    mock_instance = AsyncMock()
    mock_instance._fetch_ontology.return_value = ({}, {})
    mock_instance._resolve_and_insert.return_value = IngestResult()
    mock_resolver_cls.return_value = mock_instance

    response = client.post(
        "/graphs/test-tenant/ingest/csv/rows",
        json={"mapping": _extension_mapping(),
              "rows": [{"name": "One", "code": "C-1"}]},
        headers=auth_headers,
    )
    assert response.status_code == 200

    updates = [call.args[0] for call in mock_neptune.update.call_args_list]

    # The promoted type is declared, with its promotion provenance comment.
    assert any("types/Code" in u and "Class" in u and "promoted from attribute 'code'" in u
               for u in updates)
    # The issuer type exists even though NO column (and no row) references
    # it — zero instances, declared enrichment target.
    assert any("types/Issuer" in u and "Class" in u for u in updates)
    # Zero-data relationship slots are declared with their target as range…
    assert any("types/Code/attrs/issued_by" in u and "types/Issuer" in u for u in updates)
    assert any("types/Code/attrs/identifies" in u and "types/Item" in u for u in updates)
    # …the id-string attribute slot too…
    assert any("types/Code/attrs/code" in u and "XMLSchema#string" in u for u in updates)
    # …and EVERY core slot carries the coreSlot marker triple.
    for slot in ("issued_by", "identifies", "code"):
        assert any(f"types/Code/attrs/{slot}" in u and "coreSlot" in u
                   and '"true"^^<http://www.w3.org/2001/XMLSchema#boolean>' in u
                   for u in updates), slot


@patch("cograph_client.api.routes.ingest.SchemaResolver")
def test_csv_rows_core_slot_preregistration_uses_attribute_schema(
    mock_resolver_cls, client, auth_headers, mock_neptune,
):
    """Regression: pre-registered core slots must land in existing_attrs as
    AttributeSchema, NOT bare marker strings. A str there crashes the insert
    pass with `'str' object has no attribute 'datatype'` the moment any ingested
    entity of the extension type has an attribute matching the slot name."""
    from cograph_client.resolver.attribute_resolver import AttributeSchema
    from cograph_client.resolver.models import IngestResult

    mock_instance = AsyncMock()
    mock_instance._fetch_ontology.return_value = ({}, {})
    mock_instance._resolve_and_insert.return_value = IngestResult()
    mock_resolver_cls.return_value = mock_instance

    response = client.post(
        "/graphs/test-tenant/ingest/csv/rows",
        json={"mapping": _extension_mapping(),
              "rows": [{"name": "One", "code": "C-1"}]},
        headers=auth_headers,
    )
    assert response.status_code == 200

    # existing_attrs is the 4th positional arg handed to _resolve_and_insert.
    existing_attrs = mock_instance._resolve_and_insert.call_args.args[3]
    code_slots = existing_attrs.get("Code", {})
    assert code_slots, "extension type 'Code' core slots were not pre-registered"
    for slot_name, schema in code_slots.items():
        assert isinstance(schema, AttributeSchema), (
            f"core slot {slot_name!r} stored as {type(schema).__name__}, "
            f"must be AttributeSchema (regression: bare 'core' string)"
        )
    # Relationship slots keep their target type as datatype; attribute slots are string.
    assert code_slots["issued_by"].datatype == "Issuer"
    assert code_slots["code"].datatype == "string"


@patch("cograph_client.api.routes.ingest.SchemaResolver")
def test_csv_rows_without_extensions_writes_no_core_slots(
    mock_resolver_cls, client, auth_headers, mock_neptune,
):
    from cograph_client.resolver.models import IngestResult

    mock_instance = AsyncMock()
    mock_instance._fetch_ontology.return_value = ({}, {})
    mock_instance._resolve_and_insert.return_value = IngestResult()
    mock_resolver_cls.return_value = mock_instance

    mapping = _extension_mapping()
    del mapping["ontology_extensions"]  # pre-COG-52 payload
    response = client.post(
        "/graphs/test-tenant/ingest/csv/rows",
        json={"mapping": mapping, "rows": [{"name": "One", "code": "C-1"}]},
        headers=auth_headers,
    )
    assert response.status_code == 200
    updates = [call.args[0] for call in mock_neptune.update.call_args_list]
    assert not any("coreSlot" in u for u in updates)
    assert not any("types/Issuer" in u for u in updates)
