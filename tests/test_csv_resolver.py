"""Tests for CSV schema inference and deterministic mapping."""

import pytest

from cograph_client.resolver.csv_resolver import (
    CSVResolver,
    _rank_sample_rows,
    _safe_id,
    _snake_case,
)
from cograph_client.resolver.models import ColumnMapping, ColumnRole, CSVSchemaMapping


class TestSafeId:
    def test_basic(self):
        assert _safe_id("hello world") == "hello_world"

    def test_special_chars(self):
        assert _safe_id("123 Main St, #4") == "123_Main_St___4"

    def test_truncation(self):
        long = "a" * 300
        assert len(_safe_id(long)) == 200

    def test_empty(self):
        assert _safe_id("") == "unknown"


class TestSnakeCase:
    def test_basic(self):
        assert _snake_case("Hello World") == "hello_world"

    def test_camel(self):
        assert _snake_case("listingPrice") == "listingprice"

    def test_special(self):
        assert _snake_case("Bed/Bath Count") == "bed_bath_count"


class TestApplyMapping:
    def _make_mapping(self):
        return CSVSchemaMapping(
            entity_type="Property",
            columns=[
                ColumnMapping(column_name="address", role=ColumnRole.TYPE_ID, datatype="string"),
                ColumnMapping(column_name="price", role=ColumnRole.ATTRIBUTE, datatype="integer", attribute_name="price"),
                ColumnMapping(column_name="bedrooms", role=ColumnRole.ATTRIBUTE, datatype="integer", attribute_name="bedrooms"),
                ColumnMapping(column_name="city", role=ColumnRole.RELATIONSHIP, target_type="City", datatype="string", attribute_name="city"),
            ],
        )

    def test_basic_mapping(self):
        mapping = self._make_mapping()
        rows = [
            {"address": "123 Main St", "price": "500000", "bedrooms": "3", "city": "Austin"},
            {"address": "456 Oak Ave", "price": "350000", "bedrooms": "2", "city": "Dallas"},
        ]
        entities, rels = CSVResolver.apply_mapping(mapping, rows)

        # 2 property entities + 2 city stub entities
        assert len(entities) == 4
        property_entities = [e for e in entities if e.type_name == "Property"]
        city_entities = [e for e in entities if e.type_name == "City"]
        assert len(property_entities) == 2
        assert len(city_entities) == 2

        # 2 relationships (property → city)
        assert len(rels) == 2
        assert all(r.predicate == "city" for r in rels)

    def test_attributes_mapped(self):
        mapping = self._make_mapping()
        rows = [{"address": "123 Main St", "price": "500000", "bedrooms": "3", "city": "Austin"}]
        entities, _ = CSVResolver.apply_mapping(mapping, rows)

        prop = next(e for e in entities if e.type_name == "Property")
        attr_names = {a.name for a in prop.attributes}
        assert "price" in attr_names
        assert "bedrooms" in attr_names

    def test_empty_rows(self):
        mapping = self._make_mapping()
        entities, rels = CSVResolver.apply_mapping(mapping, [])
        assert entities == []
        assert rels == []

    def test_skips_empty_id(self):
        mapping = self._make_mapping()
        rows = [{"address": "", "price": "100", "bedrooms": "1", "city": "Austin"}]
        entities, _ = CSVResolver.apply_mapping(mapping, rows)
        # No property entity created (empty ID), no relationship so no stub either
        property_entities = [e for e in entities if e.type_name == "Property"]
        assert len(property_entities) == 0

    def test_deduplicates_relationship_targets(self):
        mapping = self._make_mapping()
        rows = [
            {"address": "123 Main", "price": "500000", "bedrooms": "3", "city": "Austin"},
            {"address": "456 Oak", "price": "350000", "bedrooms": "2", "city": "Austin"},
        ]
        entities, rels = CSVResolver.apply_mapping(mapping, rows)

        city_entities = [e for e in entities if e.type_name == "City"]
        # Austin should only appear once as a stub entity
        assert len(city_entities) == 1


class TestRankSampleRows:
    def test_picks_dense_rows_first(self):
        sparse = [{"slug": f"s{i}", "url": f"u{i}", "name": "", "bio": "", "email": ""} for i in range(12)]
        dense = [
            {"slug": f"d{i}", "url": f"u{i}", "name": f"n{i}", "bio": f"b{i}", "email": f"e{i}@x"}
            for i in range(5)
        ]
        ranked = _rank_sample_rows(sparse + dense)
        # All 5 dense rows should land in the top 5
        assert all(r["name"] != "" for r in ranked[:5])

    def test_stable_on_ties(self):
        rows = [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}, {"a": "5", "b": "6"}]
        ranked = _rank_sample_rows(rows)
        # All scored equally — order preserved
        assert ranked == rows

    def test_does_not_mutate_input(self):
        rows = [{"a": ""}, {"a": "x"}]
        original = list(rows)
        _rank_sample_rows(rows)
        assert rows == original

    def test_treats_whitespace_as_empty(self):
        rows = [{"a": "   ", "b": ""}, {"a": "x", "b": "y"}]
        ranked = _rank_sample_rows(rows)
        assert ranked[0]["a"] == "x"

    def test_handles_none_values(self):
        rows = [{"a": None, "b": None}, {"a": "x", "b": "y"}]
        ranked = _rank_sample_rows(rows)
        assert ranked[0]["a"] == "x"


class TestInferSchemaRetry:
    @pytest.mark.asyncio
    async def test_retries_on_validation_error(self, monkeypatch):
        from unittest.mock import AsyncMock

        resolver = CSVResolver(client=None, openrouter_key="")
        valid_data = {
            "entity_type": "Mentor",
            "columns": [
                {"column_name": "slug", "role": "type_id", "datatype": "string"},
                {"column_name": "name", "role": "attribute", "datatype": "string", "attribute_name": "name"},
            ],
        }
        # First call: malformed key shape (raises KeyError in _build_mapping); second: valid
        bad_data = {"entity_type_oops": "Mentor", "columns": []}

        call_log: list[float] = []

        async def fake_call(user_content: str, temperature: float = 0.0):
            call_log.append(temperature)
            return bad_data if len(call_log) == 1 else valid_data

        monkeypatch.setattr(resolver, "_call_llm", fake_call)

        mapping = await resolver.infer_schema(
            headers=["slug", "name"],
            sample_rows=[{"slug": "s", "name": "n"}],
            existing_types={},
            total_rows=1,
        )
        assert mapping.entity_type == "Mentor"
        assert call_log == [0.0, 0.3]

    @pytest.mark.asyncio
    async def test_propagates_when_retry_also_fails(self, monkeypatch):
        resolver = CSVResolver(client=None, openrouter_key="")
        bad_data = {"entity_type_oops": "Mentor", "columns": []}

        async def fake_call(user_content: str, temperature: float = 0.0):
            return bad_data

        monkeypatch.setattr(resolver, "_call_llm", fake_call)

        with pytest.raises(KeyError):
            await resolver.infer_schema(
                headers=["slug"],
                sample_rows=[{"slug": "s"}],
                existing_types={},
                total_rows=1,
            )


class TestBatchedInsertTriples:
    def test_batching(self):
        from cograph_client.graph.queries import batched_insert_triples

        triples = [(f"s{i}", "p", "o") for i in range(1200)]
        batches = batched_insert_triples("https://g", triples, batch_size=500)
        assert len(batches) == 3  # 500 + 500 + 200
        assert "INSERT DATA" in batches[0]

    def test_empty(self):
        from cograph_client.graph.queries import batched_insert_triples
        assert batched_insert_triples("https://g", []) == []

    def test_small(self):
        from cograph_client.graph.queries import batched_insert_triples
        triples = [("s", "p", "o")]
        batches = batched_insert_triples("https://g", triples)
        assert len(batches) == 1
