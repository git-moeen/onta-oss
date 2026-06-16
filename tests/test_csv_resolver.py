"""Tests for CSV schema inference and deterministic mapping."""

import copy
import csv
import os
from pathlib import Path

import pytest

from cograph_client.resolver.csv_resolver import (
    COMPLETE_SYSTEM,
    REASON_SYSTEM,
    REFUTE_SYSTEM,
    CSVResolver,
    _check_complete_shape,
    _rank_sample_rows,
    _safe_id,
    _snake_case,
)
from cograph_client.resolver.models import (
    ColumnMapping,
    ColumnRole,
    CoreSlot,
    CSVSchemaMapping,
    DatasetConstant,
    EntityRelationSpec,
    EntitySpec,
    OntologyExtensions,
    TypeExtension,
)
from cograph_client.resolver.profiler import profile_table

# The dataset CSVs live in the proprietary parent repo and are gitignored —
# present on dev machines, absent in fresh OSS clones. Tests that need them
# skip with a clear reason when the file is missing.
DATASETS_ROOT = Path(
    os.environ.get("COGRAPH_DATASETS_ROOT") or Path(__file__).resolve().parents[2]
)


def _load_dataset(relpath: str) -> tuple[list[str], list[dict]]:
    path = DATASETS_ROOT / relpath
    if not path.exists():
        pytest.skip(
            f"dataset CSV not present: {path} "
            "(gitignored, lives in the parent repo — set COGRAPH_DATASETS_ROOT)"
        )
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = list(reader)
    return headers, rows


# Three PMS-style rows: each packs a guest (Person), a reservation, and a
# property. Two reservations are for the same guest (John Smith) at the same
# property; the third is a different guest at a different property.
_PMS_ROWS = [
    {"reservation_id": "R1", "property_id": "HTL-NYC-01", "property_name": "Grand NYC",
     "check_in_date": "2026-06-01", "total_charges_usd": "1200", "status": "CHECKED_OUT",
     "guest_first_name": "John", "guest_last_name": "Smith",
     "guest_email": "john.smith@gmail.com", "guest_phone": "+1 212 555 0001"},
    {"reservation_id": "R2", "property_id": "HTL-NYC-01", "property_name": "Grand NYC",
     "check_in_date": "2026-07-01", "total_charges_usd": "800", "status": "BOOKED",
     "guest_first_name": "John", "guest_last_name": "Smith",
     "guest_email": "john.smith@gmail.com", "guest_phone": "+1 212 555 0001"},
    {"reservation_id": "R3", "property_id": "HTL-LON-01", "property_name": "London Park",
     "check_in_date": "2026-06-15", "total_charges_usd": "950", "status": "CHECKED_OUT",
     "guest_first_name": "Sara", "guest_last_name": "Khan",
     "guest_email": "sara.khan@gmail.com", "guest_phone": "+44 20 555 0002"},
]


def _pms_multi_mapping() -> CSVSchemaMapping:
    def col(name, entity, attr=None, dt="string", role=ColumnRole.ATTRIBUTE, target=None):
        return ColumnMapping(column_name=name, role=role, datatype=dt,
                             attribute_name=attr or name, target_type=target, entity=entity)
    return CSVSchemaMapping(
        entity_type="",
        entities=[
            EntitySpec(name="guest", type_name="Person", id_from=["guest_email"]),
            EntitySpec(name="reservation", type_name="Reservation", id_column="reservation_id"),
            EntitySpec(name="property", type_name="Property", id_column="property_id"),
        ],
        relationships=[
            EntityRelationSpec(subject="reservation", predicate="made_by", object="guest"),
            EntityRelationSpec(subject="reservation", predicate="at_property", object="property"),
        ],
        columns=[
            col("guest_first_name", "guest", "first_name"),
            col("guest_last_name", "guest", "last_name"),
            col("guest_email", "guest", "email"),
            col("guest_phone", "guest", "phone"),
            col("check_in_date", "reservation", "check_in_date", dt="date"),
            col("total_charges_usd", "reservation", "total_charges_usd", dt="float"),
            col("status", "reservation", "status"),
            col("property_name", "property", "name"),
        ],
    )


class TestMultiEntityMapping:
    def test_expands_one_row_into_three_types(self):
        entities, rels = CSVResolver.apply_mapping(_pms_multi_mapping(), _PMS_ROWS)
        by_type: dict[str, list] = {}
        for e in entities:
            by_type.setdefault(e.type_name, []).append(e)
        assert set(by_type) == {"Person", "Reservation", "Property"}
        # 3 reservations (one per row), 2 properties (deduped), 2 guests (deduped).
        assert len(by_type["Reservation"]) == 3
        assert len(by_type["Property"]) == 2
        assert len(by_type["Person"]) == 2

    def test_reservation_attributes_land_on_reservation(self):
        entities, _ = CSVResolver.apply_mapping(_pms_multi_mapping(), _PMS_ROWS)
        res = next(e for e in entities if e.type_name == "Reservation" and e.id == "R1")
        attr_names = {a.name for a in res.attributes}
        assert {"check_in_date", "total_charges_usd", "status"} <= attr_names
        # Guest fields must NOT be on the reservation.
        assert "email" not in attr_names and "first_name" not in attr_names

    def test_person_carries_er_signals(self):
        entities, _ = CSVResolver.apply_mapping(_pms_multi_mapping(), _PMS_ROWS)
        person = next(e for e in entities if e.type_name == "Person")
        attr_names = {a.name for a in person.attributes}
        assert {"first_name", "last_name", "email", "phone"} <= attr_names

    def test_inter_entity_edges_point_at_real_ids(self):
        entities, rels = CSVResolver.apply_mapping(_pms_multi_mapping(), _PMS_ROWS)
        res_ids = {e.id for e in entities if e.type_name == "Reservation"}
        prop_ids = {e.id for e in entities if e.type_name == "Property"}
        made_by = [r for r in rels if r.predicate == "made_by"]
        at_prop = [r for r in rels if r.predicate == "at_property"]
        assert len(made_by) == 3 and len(at_prop) == 3
        # Edge endpoints are the real entity ids (not stubs).
        assert all(r.source_id in res_ids for r in at_prop)
        assert all(r.target_id in prop_ids for r in at_prop)

    def test_property_dedup_merges_attrs_not_duplicates(self):
        entities, _ = CSVResolver.apply_mapping(_pms_multi_mapping(), _PMS_ROWS)
        nyc = [e for e in entities if e.type_name == "Property" and e.id == "HTL-NYC-01"]
        assert len(nyc) == 1
        assert any(a.value == "Grand NYC" for a in nyc[0].attributes)

    def test_skips_entity_with_missing_key(self):
        rows = _PMS_ROWS + [{"reservation_id": "R4", "property_id": "",
                             "guest_email": "x@y.com", "guest_first_name": "X"}]
        entities, rels = CSVResolver.apply_mapping(_pms_multi_mapping(), rows)
        # R4 has no property → Property not created, at_property edge skipped,
        # but the reservation + guest + made_by edge still exist.
        assert any(e.id == "R4" for e in entities if e.type_name == "Reservation")
        assert len([r for r in rels if r.predicate == "at_property"]) == 3

    def test_legacy_single_entity_unaffected(self):
        # entities=None → legacy path, byte-for-byte behavior.
        mapping = CSVSchemaMapping(
            entity_type="Listing",
            columns=[
                ColumnMapping(column_name="address", role=ColumnRole.TYPE_ID, datatype="string"),
                ColumnMapping(column_name="price", role=ColumnRole.ATTRIBUTE, datatype="integer"),
            ],
        )
        entities, _ = CSVResolver.apply_mapping(mapping, [{"address": "1 Main", "price": "500"}])
        assert len(entities) == 1 and entities[0].type_name == "Listing"


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

    def test_empty_id_gets_synthetic_key(self):
        # COG-51 / ADR 0003 §2 inverted this contract (formerly
        # test_skips_empty_id): an empty natural key with non-empty owned
        # values used to silently drop the row; it now mints a deterministic
        # content-hash synthetic key so the row is conserved.
        mapping = self._make_mapping()
        rows = [{"address": "", "price": "100", "bedrooms": "1", "city": "Austin"}]
        applied = CSVResolver.apply_mapping(mapping, rows)
        property_entities = [e for e in applied.entities if e.type_name == "Property"]
        assert len(property_entities) == 1
        assert applied.rows_dropped == 0
        # The synthetic id is a content hash, not derived from the empty key.
        assert property_entities[0].id != "unknown"

    def test_type_id_value_also_an_attribute(self):
        # ADR 0003 §2 "key consumed, not kept": the key column's value must
        # land as a regular attribute too, not just as URI/label material.
        mapping = self._make_mapping()
        rows = [{"address": "123 Main St", "price": "500000", "bedrooms": "3", "city": "Austin"}]
        entities, _ = CSVResolver.apply_mapping(mapping, rows)
        prop = next(e for e in entities if e.type_name == "Property")
        assert any(a.name == "address" and a.value == "123 Main St" for a in prop.attributes)

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


# --- COG-51 / ADR 0003 §2: row conservation -------------------------------
#
# Deterministic fixture mirroring the production catalog shape that exposed
# the silent-drop bug: 1000 rows, key column exactly 75% complete (production
# was 74.7%), one low-cardinality dimension column (300 distinct / 1000 rows
# = 0.3 card_ratio), one per-row-unique name column. No randomness anywhere —
# synthetic keys must reproduce across batches and re-runs.


def _catalog_rows(n: int = 1000) -> list[dict[str, str]]:
    return [
        {
            "item_key": f"KEY-{i:04d}" if i % 4 != 3 else "",
            "item_name": f"Item {i:04d}",
            "dimension_code": f"D{i % 300:03d}",
        }
        for i in range(n)
    ]


def _catalog_mapping() -> CSVSchemaMapping:
    return CSVSchemaMapping(
        entity_type="Item",
        columns=[
            ColumnMapping(column_name="item_key", role=ColumnRole.TYPE_ID,
                          datatype="string", attribute_name="item_key"),
            ColumnMapping(column_name="item_name", role=ColumnRole.ATTRIBUTE,
                          datatype="string", attribute_name="item_name"),
            ColumnMapping(column_name="dimension_code", role=ColumnRole.ATTRIBUTE,
                          datatype="string", attribute_name="dimension_code"),
        ],
    )


class TestRowConservationSingleEntity:
    """Input rows are never silently dropped (ADR 0003 §2, single-entity path)."""

    def test_catalog_shape_conserves_all_1000_rows(self):
        applied = CSVResolver.apply_mapping(_catalog_mapping(), _catalog_rows(1000))
        items = [e for e in applied.entities if e.type_name == "Item"]
        assert len(items) == 1000
        assert applied.rows_in == 1000
        assert applied.rows_dropped == 0
        assert applied.drops_by_entity == {}

    def test_key_attribute_present_exactly_when_source_value_exists(self):
        applied = CSVResolver.apply_mapping(_catalog_mapping(), _catalog_rows(1000))
        items = [e for e in applied.entities if e.type_name == "Item"]
        with_key = [e for e in items if any(a.name == "item_key" for a in e.attributes)]
        # Exactly the 750 rows whose key cell is non-empty carry the attribute.
        assert len(with_key) == 750
        keyed = next(e for e in items if e.id == "KEY-0000")
        assert any(a.name == "item_key" and a.value == "KEY-0000" for a in keyed.attributes)

    def test_two_batch_ingest_produces_identical_ids(self):
        rows = _catalog_rows(1000)
        single = CSVResolver.apply_mapping(_catalog_mapping(), rows)
        batch1 = CSVResolver.apply_mapping(_catalog_mapping(), rows[:500])
        batch2 = CSVResolver.apply_mapping(_catalog_mapping(), rows[500:])
        batched_ids = [e.id for e in batch1.entities] + [e.id for e in batch2.entities]
        assert [e.id for e in single.entities] == batched_ids
        # Re-running the same batch is idempotent.
        again = CSVResolver.apply_mapping(_catalog_mapping(), rows[:500])
        assert [e.id for e in again.entities] == [e.id for e in batch1.entities]

    def test_identical_keyless_rows_collapse_to_one_id(self):
        dup = {"item_key": "", "item_name": "Same", "dimension_code": "D001"}
        applied = CSVResolver.apply_mapping(_catalog_mapping(), [dict(dup), dict(dup)])
        # Content-hash determinism makes true duplicates share one id — the
        # graph collapses them on URI. Neither row is dropped.
        assert len({e.id for e in applied.entities}) == 1
        assert applied.rows_dropped == 0

    def test_all_empty_row_skipped_and_accounted(self):
        rows = [
            {"item_key": "KEY-1", "item_name": "One", "dimension_code": "D1"},
            {"item_key": "", "item_name": "", "dimension_code": ""},
            {"item_key": "  ", "item_name": " ", "dimension_code": ""},  # whitespace = empty
        ]
        applied = CSVResolver.apply_mapping(_catalog_mapping(), rows)
        assert len(applied.entities) == 1
        assert applied.rows_in == 3
        assert applied.rows_dropped == 2
        assert applied.drops_by_entity == {"Item": 2}

    def test_unmapped_columns_do_not_affect_synthetic_key(self):
        # Only owned (mapped) columns feed the content hash.
        base = {"item_key": "", "item_name": "Same", "dimension_code": "D1"}
        with_extra = dict(base, unmapped_noise="zzz")
        a = CSVResolver.apply_mapping(_catalog_mapping(), [base])
        b = CSVResolver.apply_mapping(_catalog_mapping(), [with_extra])
        assert a.entities[0].id == b.entities[0].id

    def test_legacy_tuple_unpacking_still_works(self):
        # AppliedMapping iterates as the legacy (entities, relationships) pair.
        entities, rels = CSVResolver.apply_mapping(_catalog_mapping(), _catalog_rows(8))
        assert isinstance(entities, list) and isinstance(rels, list)
        assert len(entities) == 8


class TestRowConservationMultiEntity:
    """ADR 0003 §2 invariants on the multi-entity path."""

    def test_synthetic_key_when_natural_key_empty_but_attrs_present(self):
        rows = _PMS_ROWS + [
            {"reservation_id": "", "property_id": "HTL-PAR-01", "property_name": "Paris Centre",
             "check_in_date": "2026-08-01", "total_charges_usd": "640", "status": "BOOKED",
             "guest_first_name": "Ana", "guest_last_name": "Lima",
             "guest_email": "ana.lima@gmail.com", "guest_phone": "+33 1 555 0003"},
        ]
        applied = CSVResolver.apply_mapping(_pms_multi_mapping(), rows)
        reservations = [e for e in applied.entities if e.type_name == "Reservation"]
        # The keyless reservation is conserved under a synthetic key.
        assert len(reservations) == 4
        assert applied.rows_dropped == 0
        synthetic = next(e for e in reservations if e.id not in {"R1", "R2", "R3"})
        # Edges reference the synthetic id (no orphaned endpoints).
        made_by = [r for r in applied.relationships if r.predicate == "made_by"]
        assert any(r.source_id == synthetic.id for r in made_by)
        # Deterministic: re-applying the same rows mints the same ids.
        again = CSVResolver.apply_mapping(_pms_multi_mapping(), rows)
        assert {e.id for e in again.entities} == {e.id for e in applied.entities}

    def test_key_column_value_emitted_as_attribute(self):
        applied = CSVResolver.apply_mapping(_pms_multi_mapping(), _PMS_ROWS)
        res = next(e for e in applied.entities if e.type_name == "Reservation" and e.id == "R1")
        assert any(a.name == "reservation_id" and a.value == "R1" for a in res.attributes)
        prop = next(e for e in applied.entities if e.type_name == "Property" and e.id == "HTL-NYC-01")
        assert any(a.name == "property_id" and a.value == "HTL-NYC-01" for a in prop.attributes)

    def test_id_column_also_mapped_as_column_not_duplicated(self):
        # When the id_column is routed to its entity as a regular column, the
        # value appears exactly once, under the mapped attribute name.
        mapping = _pms_multi_mapping()
        mapping.columns.append(ColumnMapping(
            column_name="reservation_id", role=ColumnRole.ATTRIBUTE,
            datatype="string", attribute_name="confirmation_code", entity="reservation",
        ))
        applied = CSVResolver.apply_mapping(mapping, _PMS_ROWS)
        res = next(e for e in applied.entities if e.type_name == "Reservation" and e.id == "R1")
        hits = [a for a in res.attributes if a.value == "R1"]
        assert [a.name for a in hits] == ["confirmation_code"]

    def test_all_empty_entity_counted_but_row_not_dropped(self):
        # R4 mints a reservation + guest but its property is all-empty: the
        # skipped property instance is accounted, the row is NOT dropped.
        rows = _PMS_ROWS + [{"reservation_id": "R4", "property_id": "",
                             "guest_email": "x@y.com", "guest_first_name": "X"}]
        applied = CSVResolver.apply_mapping(_pms_multi_mapping(), rows)
        assert applied.rows_dropped == 0
        assert applied.drops_by_entity == {"property": 1}

    def test_fully_empty_row_dropped_and_counted(self):
        empty = {k: "" for k in _PMS_ROWS[0]}
        applied = CSVResolver.apply_mapping(_pms_multi_mapping(), _PMS_ROWS + [empty])
        assert applied.rows_in == 4
        assert applied.rows_dropped == 1
        # Every entity spec was all-empty on that row.
        assert applied.drops_by_entity == {"guest": 1, "reservation": 1, "property": 1}


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
    """Retry contract of the LEGACY single-call path. These tests predate the
    ADR 0003 v2 pipeline (now the default), so they pin OMNIX_CSV_INFERENCE_V2=0
    to keep exercising the code path they were written for; the v2 retry
    contract is covered in TestInferSchemaV2Retry below."""

    @pytest.mark.asyncio
    async def test_retries_on_validation_error(self, monkeypatch):
        from unittest.mock import AsyncMock

        monkeypatch.setenv("OMNIX_CSV_INFERENCE_V2", "0")
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
        monkeypatch.setenv("OMNIX_CSV_INFERENCE_V2", "0")
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


# --- COG-53 / ADR 0003 Passes B+C: profile → reason → refute → convert ------
#
# The LLM seam (_call_llm_v2) is scripted per pass: reason/refute/complete
# outputs are consumed in order, so retries pull the next entry. No network
# anywhere. Tests that predate the COG-52 completion pass omit the complete
# queue and get the benign default ({"types": []} — nothing to extend).


def _mock_v2(monkeypatch, resolver, reason_outputs, refute_outputs,
             complete_outputs=None):
    """Script the v2 LLM seam. Returns the call log as (pass, temperature)."""
    calls: list[tuple[str, float]] = []
    queues = {
        "reason": list(reason_outputs),
        "refute": list(refute_outputs),
        # Default: a valid no-op completion, so pre-COG-52 tests keep running
        # the full pipeline without scripting Pass D themselves.
        "complete": list(complete_outputs if complete_outputs is not None
                         else [{"types": []}]),
    }

    async def fake(system, user_content, temperature=0.0):
        assert system in (REASON_SYSTEM, REFUTE_SYSTEM, COMPLETE_SYSTEM)
        which = {REASON_SYSTEM: "reason", REFUTE_SYSTEM: "refute",
                 COMPLETE_SYSTEM: "complete"}[system]
        calls.append((which, temperature))
        return queues[which].pop(0)

    monkeypatch.setattr(resolver, "_call_llm_v2", fake)
    return calls


async def _run_v2(monkeypatch, headers, rows, reason, refute, complete=None):
    resolver = CSVResolver(client=None, openrouter_key="")
    calls = _mock_v2(
        monkeypatch, resolver, [reason], [refute],
        [complete] if complete is not None else None,
    )
    mapping = await resolver.infer_schema(headers, rows, {}, total_rows=len(rows))
    return mapping, calls


def _echo_refute(schema: dict) -> dict:
    """A clean refute pass: nothing wrong, schema echoed (the prompt contract)."""
    return {"violations": [], "corrected": copy.deepcopy(schema)}


_SIMPLE_HEADERS = ["rec_id", "label"]


def _simple_rows(n: int = 10) -> list[dict[str, str]]:
    return [{"rec_id": f"R{i}", "label": f"Label {i}"} for i in range(n)]


_SIMPLE_REASON = {
    "entities": [
        {"name": "rec", "type_name": "Record", "key_strategy": "column",
         "key_columns": ["rec_id"], "why": "complete_unique_key", "confidence": 0.95},
    ],
    "columns": [
        {"column": "rec_id", "role": "key", "entity": "rec",
         "predicate_or_attr": "rec_id", "why": "key kept queryable", "confidence": 0.95},
        {"column": "label", "role": "attribute", "entity": "rec",
         "predicate_or_attr": "label", "why": "near-unique label", "confidence": 0.9},
    ],
    "relationships": [],
}


# Grainger-shaped scenario (the production failure ADR 0003 documents): the
# reason pass keys the item on the 75%-complete column; the refute pass fires
# KEY DROPS ROWS and corrects the key strategy to synthetic. Reuses
# _catalog_rows: item_key 75% complete, item_name unique, dimension_code
# 300-distinct dimension.
_GRAINGER_REASON = {
    "entities": [
        {"name": "item", "type_name": "Item", "key_strategy": "column",
         "key_columns": ["item_key"], "why": "id-shaped column", "confidence": 0.6},
        {"name": "dimension", "type_name": "Dimension", "key_strategy": "column",
         "key_columns": ["dimension_code"], "why": "low_cardinality_repeated", "confidence": 0.9},
    ],
    "columns": [
        {"column": "item_key", "role": "key", "entity": "item",
         "predicate_or_attr": "item_key", "why": "key", "confidence": 0.6},
        {"column": "item_name", "role": "attribute", "entity": "item",
         "predicate_or_attr": "item_name", "why": "near-unique label", "confidence": 0.9},
        {"column": "dimension_code", "role": "key", "entity": "dimension",
         "predicate_or_attr": "dimension_code", "why": "dimension key", "confidence": 0.9},
    ],
    "relationships": [
        {"subject": "item", "predicate": "categorized_as", "object": "dimension",
         "why": "row links each item to one shared dimension"},
    ],
}


def _grainger_refute() -> dict:
    corrected = copy.deepcopy(_GRAINGER_REASON)
    corrected["entities"][0]["key_strategy"] = "synthetic"
    corrected["entities"][0]["key_columns"] = []
    return {
        "violations": [
            {"template": "KEY DROPS ROWS", "location": "entities.item",
             "evidence": "item_key completeness 0.75 < 0.99", "severity": "error"},
        ],
        "corrected": corrected,
    }


class TestInferSchemaV2:
    """The v2 pipeline end-to-end with scripted LLM outputs."""

    @pytest.mark.asyncio
    async def test_refute_corrects_incomplete_key_grainger_shape(self, monkeypatch):
        rows = _catalog_rows(1000)
        mapping, calls = await _run_v2(
            monkeypatch, ["item_key", "item_name", "dimension_code"],
            rows, _GRAINGER_REASON, _grainger_refute(),
        )
        # Reason, refute, then complete (COG-52), each once, all at temp 0.
        assert calls == [("reason", 0.0), ("refute", 0.0), ("complete", 0.0)]
        item = next(e for e in mapping.entities if e.type_name == "Item")
        assert item.key_strategy == "synthetic"
        assert item.id_column is None and item.id_from is None
        assert [v.template for v in mapping.violations] == ["KEY DROPS ROWS"]
        assert mapping.violations[0].severity == "error"
        # Row conservation holds where the proposed schema would have dropped 250.
        applied = CSVResolver.apply_mapping(mapping, rows)
        assert applied.rows_in == 1000 and applied.rows_dropped == 0
        assert len([e for e in applied.entities if e.type_name == "Item"]) == 1000
        assert len([e for e in applied.entities if e.type_name == "Dimension"]) == 300
        assert len([r for r in applied.relationships if r.predicate == "categorized_as"]) == 1000

    @pytest.mark.asyncio
    async def test_role_key_column_is_identity_and_attribute(self, monkeypatch):
        rows = _simple_rows()
        mapping, _ = await _run_v2(
            monkeypatch, _SIMPLE_HEADERS, rows, _SIMPLE_REASON, _echo_refute(_SIMPLE_REASON),
        )
        rec = mapping.entities[0]
        # Identity half: the EntitySpec is keyed on the column…
        assert rec.key_strategy == "column" and rec.id_column == "rec_id"
        # …attribute half: the same column is a regular ATTRIBUTE mapping.
        key_col = next(c for c in mapping.columns if c.column_name == "rec_id")
        assert key_col.role == ColumnRole.ATTRIBUTE
        applied = CSVResolver.apply_mapping(mapping, rows)
        rec0 = next(e for e in applied.entities if e.id == "R0")
        assert any(a.name == "rec_id" and a.value == "R0" for a in rec0.attributes)

    @pytest.mark.asyncio
    async def test_decision_provenance_flows_to_mapping(self, monkeypatch):
        mapping, _ = await _run_v2(
            monkeypatch, _SIMPLE_HEADERS, _simple_rows(), _SIMPLE_REASON,
            _echo_refute(_SIMPLE_REASON),
        )
        assert mapping.entities[0].why == "complete_unique_key"
        assert mapping.entities[0].confidence == pytest.approx(0.95)
        label = next(c for c in mapping.columns if c.column_name == "label")
        assert label.why == "near-unique label"
        assert label.confidence == pytest.approx(0.9)
        audit = mapping.inference_audit
        assert audit is not None
        assert audit.pipeline == "reason_refute_v2"
        assert audit.rows_profiled == 10 and audit.total_rows == 10
        assert set(audit.profile["columns"]) == {"rec_id", "label"}

    @pytest.mark.asyncio
    async def test_datatypes_derived_from_profile_evidence(self, monkeypatch):
        headers = ["rec_id", "when", "amount", "score", "flag", "link"]
        rows = [
            {"rec_id": f"R{i}", "when": f"2026-01-{i + 1:02d}", "amount": str(i),
             "score": f"{i}.5", "flag": "true" if i % 2 else "false",
             "link": f"https://example.test/{i}"}
            for i in range(10)
        ]
        reason = {
            "entities": [{"name": "rec", "type_name": "Record",
                          "key_strategy": "column", "key_columns": ["rec_id"]}],
            "columns": [
                {"column": h, "role": "key" if h == "rec_id" else "attribute",
                 "entity": "rec", "predicate_or_attr": h}
                for h in headers
            ],
            "relationships": [],
        }
        mapping, _ = await _run_v2(monkeypatch, headers, rows, reason, _echo_refute(reason))
        dt = {c.column_name: c.datatype for c in mapping.columns}
        assert dt["when"] == "datetime"
        assert dt["amount"] == "integer"
        assert dt["score"] == "float"
        assert dt["flag"] == "boolean"
        assert dt["link"] == "uri"
        assert dt["rec_id"] == "string"

    @pytest.mark.asyncio
    async def test_v2_never_applies_keyword_patches(self, monkeypatch):
        # 'city' is in the legacy FORCE_RELATIONSHIP keyword map. The v2 path
        # must honor the (corrected) schema's attribute decision — the keyword
        # map is exactly the anti-pattern ADR 0003 §4 retires.
        headers = ["listing_id", "city"]
        rows = [{"listing_id": f"L{i}", "city": f"City {i}"} for i in range(10)]
        reason = {
            "entities": [{"name": "listing", "type_name": "Listing",
                          "key_strategy": "column", "key_columns": ["listing_id"]}],
            "columns": [
                {"column": "listing_id", "role": "key", "entity": "listing",
                 "predicate_or_attr": "listing_id"},
                {"column": "city", "role": "attribute", "entity": "listing",
                 "predicate_or_attr": "city"},
            ],
            "relationships": [],
        }
        mapping, _ = await _run_v2(monkeypatch, headers, rows, reason, _echo_refute(reason))
        city = next(c for c in mapping.columns if c.column_name == "city")
        assert city.role == ColumnRole.ATTRIBUTE
        assert city.target_type is None

    @pytest.mark.asyncio
    async def test_flag_off_runs_legacy_path_with_keyword_patch(self, monkeypatch):
        monkeypatch.setenv("OMNIX_CSV_INFERENCE_V2", "0")
        resolver = CSVResolver(client=None, openrouter_key="")

        async def fake_legacy(user_content, temperature=0.0):
            return {
                "entity_type": "Listing",
                "columns": [
                    {"column_name": "listing_id", "role": "type_id", "datatype": "string"},
                    {"column_name": "city", "role": "attribute", "datatype": "string",
                     "attribute_name": "city"},
                ],
            }

        async def v2_must_not_run(system, user_content, temperature=0.0):
            raise AssertionError("v2 seam must not be called when the flag is off")

        monkeypatch.setattr(resolver, "_call_llm", fake_legacy)
        monkeypatch.setattr(resolver, "_call_llm_v2", v2_must_not_run)
        mapping = await resolver.infer_schema(
            ["listing_id", "city"], [{"listing_id": "L1", "city": "Austin"}], {}, 1,
        )
        # Legacy path verbatim: the FORCE_RELATIONSHIP keyword patch still fires…
        city = next(c for c in mapping.columns if c.column_name == "city")
        assert city.role == ColumnRole.RELATIONSHIP
        assert city.target_type == "City"
        # …and no v2 provenance is emitted.
        assert mapping.violations == []
        assert mapping.inference_audit is None
        assert mapping.ontology_extensions is None

    @pytest.mark.asyncio
    async def test_relationship_role_column_becomes_dimension_edge(self, monkeypatch):
        headers = ["rec_id", "origin"]
        rows = [{"rec_id": f"R{i}", "origin": ["north", "south"][i % 2]} for i in range(10)]
        reason = {
            "entities": [{"name": "rec", "type_name": "Record",
                          "key_strategy": "column", "key_columns": ["rec_id"]}],
            "columns": [
                {"column": "rec_id", "role": "key", "entity": "rec",
                 "predicate_or_attr": "rec_id"},
                {"column": "origin", "role": "relationship", "entity": "rec",
                 "predicate_or_attr": "originates_from", "target_type": "Zone"},
            ],
            "relationships": [],
        }
        mapping, _ = await _run_v2(monkeypatch, headers, rows, reason, _echo_refute(reason))
        col = next(c for c in mapping.columns if c.column_name == "origin")
        assert col.role == ColumnRole.RELATIONSHIP
        assert col.target_type == "Zone"
        assert col.attribute_name == "originates_from"
        applied = CSVResolver.apply_mapping(mapping, rows)
        zones = [e for e in applied.entities if e.type_name == "Zone"]
        assert len(zones) == 2  # deduped dimension stubs
        assert len([r for r in applied.relationships if r.predicate == "originates_from"]) == 10

    @pytest.mark.asyncio
    async def test_relationship_without_target_type_falls_back_to_pascal(self, monkeypatch):
        headers = ["rec_id", "origin"]
        rows = [{"rec_id": f"R{i}", "origin": "north"} for i in range(5)]
        reason = {
            "entities": [{"name": "rec", "type_name": "Record",
                          "key_strategy": "column", "key_columns": ["rec_id"]}],
            "columns": [
                {"column": "rec_id", "role": "key", "entity": "rec",
                 "predicate_or_attr": "rec_id"},
                {"column": "origin", "role": "relationship", "entity": "rec",
                 "predicate_or_attr": "origin_zone"},
            ],
        }
        mapping, _ = await _run_v2(monkeypatch, headers, rows, reason, _echo_refute(reason))
        col = next(c for c in mapping.columns if c.column_name == "origin")
        assert col.target_type == "OriginZone"

    @pytest.mark.asyncio
    async def test_flat_schema_columns_default_to_single_entity(self, monkeypatch):
        # A flat (one-entity) schema may omit per-column owners — repair by
        # routing every column to the only entity.
        reason = copy.deepcopy(_SIMPLE_REASON)
        for col in reason["columns"]:
            col.pop("entity")
        mapping, _ = await _run_v2(
            monkeypatch, _SIMPLE_HEADERS, _simple_rows(), reason, _echo_refute(reason),
        )
        assert all(c.entity == "rec" for c in mapping.columns)

    @pytest.mark.asyncio
    async def test_dangling_relationship_dropped(self, monkeypatch):
        reason = copy.deepcopy(_SIMPLE_REASON)
        reason["relationships"] = [
            {"subject": "rec", "predicate": "linked_to", "object": "ghost"},
        ]
        mapping, _ = await _run_v2(
            monkeypatch, _SIMPLE_HEADERS, _simple_rows(), reason, _echo_refute(reason),
        )
        assert mapping.relationships is None

    def test_old_payload_without_new_fields_parses(self):
        # Backward compat: a mapping serialized before COG-53 (no key_strategy,
        # why, confidence, violations, inference_audit) must parse unchanged.
        payload = {
            "entity_type": "Item",
            "columns": [{"column_name": "a", "role": "attribute", "datatype": "string"}],
            "entities": [{"name": "item", "type_name": "Item", "id_column": "a"}],
            "relationships": [{"subject": "item", "predicate": "linked_to", "object": "item"}],
        }
        mapping = CSVSchemaMapping.model_validate(payload)
        assert mapping.violations == []
        assert mapping.inference_audit is None
        assert mapping.entities[0].key_strategy is None
        assert mapping.entities[0].why is None and mapping.entities[0].confidence is None
        assert mapping.columns[0].why is None and mapping.columns[0].confidence is None
        assert mapping.relationships[0].why is None
        # Pre-multi-entity payloads (no entities at all) parse too.
        flat = CSVSchemaMapping.model_validate({
            "entity_type": "Book",
            "columns": [{"column_name": "isbn", "role": "type_id", "datatype": "string"}],
        })
        assert flat.entities is None and flat.violations == []


class TestInferSchemaV2Retry:
    """Per-call retry-at-0.3 contract on the v2 path (mirrors the legacy
    contract the /ingest/csv/schema 422 guidance depends on)."""

    @pytest.mark.asyncio
    async def test_reason_retries_at_higher_temp(self, monkeypatch):
        resolver = CSVResolver(client=None, openrouter_key="")
        calls = _mock_v2(
            monkeypatch, resolver,
            [{"entities": [], "columns": []}, _SIMPLE_REASON],  # 1st degenerate
            [_echo_refute(_SIMPLE_REASON)],
        )
        mapping = await resolver.infer_schema(_SIMPLE_HEADERS, _simple_rows(), {}, 10)
        assert mapping.entities[0].type_name == "Record"
        assert calls == [("reason", 0.0), ("reason", 0.3), ("refute", 0.0), ("complete", 0.0)]

    @pytest.mark.asyncio
    async def test_refute_retries_at_higher_temp(self, monkeypatch):
        resolver = CSVResolver(client=None, openrouter_key="")
        # 1st refute is degenerate: violations found but no corrected schema.
        bad_refute = {"violations": [{"template": "KEYLESS ENTITY"}]}
        calls = _mock_v2(
            monkeypatch, resolver,
            [_SIMPLE_REASON],
            [bad_refute, _echo_refute(_SIMPLE_REASON)],
        )
        mapping = await resolver.infer_schema(_SIMPLE_HEADERS, _simple_rows(), {}, 10)
        assert mapping.entities[0].type_name == "Record"
        assert calls == [("reason", 0.0), ("refute", 0.0), ("refute", 0.3), ("complete", 0.0)]

    @pytest.mark.asyncio
    async def test_propagates_when_reason_retry_also_fails(self, monkeypatch):
        resolver = CSVResolver(client=None, openrouter_key="")
        bad = {"entities": [], "columns": []}
        _mock_v2(monkeypatch, resolver, [bad, copy.deepcopy(bad)], [])
        with pytest.raises(KeyError):
            await resolver.infer_schema(_SIMPLE_HEADERS, _simple_rows(), {}, 10)

    @pytest.mark.asyncio
    async def test_clean_refute_without_echo_keeps_proposed_schema(self, monkeypatch):
        # violations: [] with the echo omitted is repaired, not retried — the
        # proposed schema stands.
        resolver = CSVResolver(client=None, openrouter_key="")
        calls = _mock_v2(
            monkeypatch, resolver, [_SIMPLE_REASON], [{"violations": []}],
        )
        mapping = await resolver.infer_schema(_SIMPLE_HEADERS, _simple_rows(), {}, 10)
        assert mapping.entities[0].type_name == "Record"
        assert mapping.violations == []
        assert calls == [("reason", 0.0), ("refute", 0.0), ("complete", 0.0)]


class TestV2RefuteTemplates:
    """Conversion of corrected schemas for each of the refutation
    templates (ADR 0003 Pass C). The mocked refute output mirrors the repair
    each template demands; the converter must honor it. Template 1's
    synthetic-key repair is covered by the Grainger-shape test above — the
    composite repair is covered here."""

    def test_refute_prompt_lists_all_seven_templates(self):
        """The REFUTE system prompt enumerates the original six structural
        templates plus the ADR 0004 drift template (#7)."""
        for name in (
            "KEY DROPS ROWS",
            "DIMENSION AS LITERAL",
            "COLUMN-NAMED EDGE",
            "KEYLESS ENTITY",
            "DUPLICATE/DEAD ATTR",
            "LOST KEY",
            "SPARSE / MIS-DOMAINED EDGE",
        ):
            assert name in REFUTE_SYSTEM
        # The drift template is numbered 7 and stays domain-free (structural
        # wording: coverage / source type / predicate — no domain nouns).
        assert "7. SPARSE / MIS-DOMAINED EDGE" in REFUTE_SYSTEM

    @pytest.mark.asyncio
    async def test_t1_key_drops_rows_corrected_to_composite(self, monkeypatch):
        headers = ["first", "last", "badge"]
        rows = [{"first": f"F{i}", "last": f"L{i}",
                 "badge": f"B{i}" if i % 2 else ""} for i in range(20)]
        proposed = {
            "entities": [{"name": "member", "type_name": "Member",
                          "key_strategy": "column", "key_columns": ["badge"]}],
            "columns": [
                {"column": "first", "role": "attribute", "entity": "member",
                 "predicate_or_attr": "first"},
                {"column": "last", "role": "attribute", "entity": "member",
                 "predicate_or_attr": "last"},
                {"column": "badge", "role": "key", "entity": "member",
                 "predicate_or_attr": "badge"},
            ],
        }
        corrected = copy.deepcopy(proposed)
        corrected["entities"][0]["key_strategy"] = "composite"
        corrected["entities"][0]["key_columns"] = ["first", "last"]
        refute = {
            "violations": [{"template": "KEY DROPS ROWS", "location": "entities.member",
                            "evidence": "badge completeness 0.5 < 0.99", "severity": "error"}],
            "corrected": corrected,
        }
        mapping, _ = await _run_v2(monkeypatch, headers, rows, proposed, refute)
        member = mapping.entities[0]
        assert member.key_strategy == "composite"
        assert member.id_from == ["first", "last"] and member.id_column is None
        applied = CSVResolver.apply_mapping(mapping, rows)
        assert applied.rows_dropped == 0
        assert len(applied.entities) == 20

    @pytest.mark.asyncio
    async def test_t2_dimension_as_literal_promoted_to_entity_and_edge(self, monkeypatch):
        headers = ["rec_id", "group_code"]
        rows = [{"rec_id": f"R{i}", "group_code": f"G{i % 4}"} for i in range(20)]
        proposed = {
            "entities": [{"name": "rec", "type_name": "Record",
                          "key_strategy": "column", "key_columns": ["rec_id"]}],
            "columns": [
                {"column": "rec_id", "role": "key", "entity": "rec",
                 "predicate_or_attr": "rec_id"},
                {"column": "group_code", "role": "attribute", "entity": "rec",
                 "predicate_or_attr": "group_code"},
            ],
        }
        corrected = {
            "entities": [
                proposed["entities"][0],
                {"name": "group", "type_name": "Group",
                 "key_strategy": "column", "key_columns": ["group_code"]},
            ],
            "columns": [
                proposed["columns"][0],
                {"column": "group_code", "role": "key", "entity": "group",
                 "predicate_or_attr": "group_code"},
            ],
            "relationships": [
                {"subject": "rec", "predicate": "belongs_to", "object": "group",
                 "why": "shared dimension referenced by many rows"},
            ],
        }
        refute = {
            "violations": [{"template": "DIMENSION AS LITERAL", "location": "columns.group_code",
                            "evidence": "card_ratio 0.2, values repeat", "severity": "warning"}],
            "corrected": corrected,
        }
        mapping, _ = await _run_v2(monkeypatch, headers, rows, proposed, refute)
        assert [v.template for v in mapping.violations] == ["DIMENSION AS LITERAL"]
        assert {e.type_name for e in mapping.entities} == {"Record", "Group"}
        assert mapping.relationships[0].predicate == "belongs_to"
        assert mapping.relationships[0].why == "shared dimension referenced by many rows"
        applied = CSVResolver.apply_mapping(mapping, rows)
        assert len([e for e in applied.entities if e.type_name == "Group"]) == 4
        assert len([r for r in applied.relationships if r.predicate == "belongs_to"]) == 20

    @pytest.mark.asyncio
    async def test_t3_column_named_edge_renamed_to_relation(self, monkeypatch):
        headers = ["rec_id", "maker_name"]
        rows = [{"rec_id": f"R{i}", "maker_name": f"M{i % 3}"} for i in range(12)]
        proposed = {
            "entities": [
                {"name": "rec", "type_name": "Record",
                 "key_strategy": "column", "key_columns": ["rec_id"]},
                {"name": "maker", "type_name": "Maker",
                 "key_strategy": "column", "key_columns": ["maker_name"]},
            ],
            "columns": [
                {"column": "rec_id", "role": "key", "entity": "rec",
                 "predicate_or_attr": "rec_id"},
                {"column": "maker_name", "role": "key", "entity": "maker",
                 "predicate_or_attr": "name"},
            ],
            "relationships": [
                {"subject": "rec", "predicate": "maker_name", "object": "maker"},
            ],
        }
        corrected = copy.deepcopy(proposed)
        corrected["relationships"][0]["predicate"] = "made_by"
        refute = {
            "violations": [{"template": "COLUMN-NAMED EDGE", "location": "relationships[0]",
                            "evidence": "predicate equals source column 'maker_name'",
                            "severity": "warning"}],
            "corrected": corrected,
        }
        mapping, _ = await _run_v2(monkeypatch, headers, rows, proposed, refute)
        assert [v.template for v in mapping.violations] == ["COLUMN-NAMED EDGE"]
        predicates = {r.predicate for r in mapping.relationships}
        assert predicates == {"made_by"}
        assert "maker_name" not in predicates

    @pytest.mark.asyncio
    async def test_t4_keyless_entity_gets_synthetic_strategy(self, monkeypatch):
        headers = ["note_text", "initials"]
        rows = [{"note_text": f"observation number {i}", "initials": f"A{i % 5}"}
                for i in range(15)]
        proposed = {
            "entities": [{"name": "note", "type_name": "Note", "key_columns": []}],
            "columns": [
                {"column": "note_text", "role": "attribute", "entity": "note",
                 "predicate_or_attr": "note_text"},
                {"column": "initials", "role": "attribute", "entity": "note",
                 "predicate_or_attr": "initials"},
            ],
        }
        corrected = copy.deepcopy(proposed)
        corrected["entities"][0]["key_strategy"] = "synthetic"
        refute = {
            "violations": [{"template": "KEYLESS ENTITY", "location": "entities.note",
                            "evidence": "no complete+unique column; no key strategy declared",
                            "severity": "error"}],
            "corrected": corrected,
        }
        mapping, _ = await _run_v2(monkeypatch, headers, rows, proposed, refute)
        note = mapping.entities[0]
        assert note.key_strategy == "synthetic"
        assert note.id_column is None and note.id_from is None
        applied = CSVResolver.apply_mapping(mapping, rows)
        assert applied.rows_dropped == 0
        # Deterministic content-hash keys: one entity per distinct row.
        assert len({e.id for e in applied.entities}) == 15

    @pytest.mark.asyncio
    async def test_t5_dead_attribute_dropped_from_corrected(self, monkeypatch):
        headers = ["rec_id", "notes"]
        rows = [{"rec_id": f"R{i}", "notes": ""} for i in range(10)]
        proposed = {
            "entities": [{"name": "rec", "type_name": "Record",
                          "key_strategy": "column", "key_columns": ["rec_id"]}],
            "columns": [
                {"column": "rec_id", "role": "key", "entity": "rec",
                 "predicate_or_attr": "rec_id"},
                {"column": "notes", "role": "attribute", "entity": "rec",
                 "predicate_or_attr": "notes"},
            ],
        }
        corrected = copy.deepcopy(proposed)
        corrected["columns"] = [c for c in corrected["columns"] if c["column"] != "notes"]
        refute = {
            "violations": [{"template": "DUPLICATE/DEAD ATTR", "location": "columns.notes",
                            "evidence": "completeness 0.0 — all-empty column",
                            "severity": "warning"}],
            "corrected": corrected,
        }
        mapping, _ = await _run_v2(monkeypatch, headers, rows, proposed, refute)
        assert [v.template for v in mapping.violations] == ["DUPLICATE/DEAD ATTR"]
        assert all(c.column_name != "notes" for c in mapping.columns)

    @pytest.mark.asyncio
    async def test_t6_lost_key_reemitted_as_attribute(self, monkeypatch):
        headers = ["code", "title"]
        rows = [{"code": f"C{i}", "title": f"T{i}"} for i in range(10)]
        proposed = {
            "entities": [{"name": "item", "type_name": "Item",
                          "key_strategy": "column", "key_columns": ["code"]}],
            # The key column is consumed as identity only — not in columns.
            "columns": [
                {"column": "title", "role": "attribute", "entity": "item",
                 "predicate_or_attr": "title"},
            ],
        }
        corrected = copy.deepcopy(proposed)
        corrected["columns"].append(
            {"column": "code", "role": "key", "entity": "item", "predicate_or_attr": "code"},
        )
        refute = {
            "violations": [{"template": "LOST KEY", "location": "columns.code",
                            "evidence": "key column not also emitted as an attribute",
                            "severity": "warning"}],
            "corrected": corrected,
        }
        mapping, _ = await _run_v2(monkeypatch, headers, rows, proposed, refute)
        assert mapping.entities[0].id_column == "code"
        code_col = next(c for c in mapping.columns if c.column_name == "code")
        assert code_col.role == ColumnRole.ATTRIBUTE
        applied = CSVResolver.apply_mapping(mapping, rows)
        item0 = next(e for e in applied.entities if e.id == "C0")
        assert any(a.name == "code" and a.value == "C0" for a in item0.attributes)


# --- COG-52 / ADR 0003 Pass D: completion (promotions + core slots) ---------
#
# Deterministic Grainger-shaped fixture mirroring the production catalog the
# ADR documents: 1000 rows, sku 75% complete (production: 74.7%), mpn fully
# complete, manufacturer_name a 290-distinct dimension, specs all-empty. The
# scripted reason output carries the production failures; refute repairs them
# (synthetic key, dead column dropped); complete promotes the dependent
# identifiers and attaches the issuer dataset constant.

_GRAINGER_HEADERS = [
    "sku", "mpn", "manufacturer_name", "item_description",
    "specs", "category", "uom", "list_price",
]


def _grainger_rows(n: int = 1000) -> list[dict[str, str]]:
    return [
        {
            "sku": f"GGR-{i:07d}" if i % 4 != 3 else "",
            "mpn": f"MPN-{i:05d}",
            "manufacturer_name": f"Maker {i % 290:03d}",
            "item_description": f"Industrial item {i:04d}",
            "specs": "",
            "category": f"Cat-{i % 12:02d}",
            "uom": "EA",
            "list_price": f"{(i % 900) + 100}.50",
        }
        for i in range(n)
    ]


_GRAINGER_V2_REASON = {
    "entities": [
        {"name": "product", "type_name": "Product", "key_strategy": "column",
         "key_columns": ["sku"], "why": "id-shaped column", "confidence": 0.6},
        {"name": "manufacturer", "type_name": "Manufacturer", "key_strategy": "column",
         "key_columns": ["manufacturer_name"], "why": "low_cardinality_repeated dimension",
         "confidence": 0.9},
    ],
    "columns": [
        {"column": "sku", "role": "key", "entity": "product",
         "predicate_or_attr": "sku", "why": "key kept queryable", "confidence": 0.6},
        {"column": "mpn", "role": "attribute", "entity": "product",
         "predicate_or_attr": "mpn", "confidence": 0.8},
        {"column": "manufacturer_name", "role": "key", "entity": "manufacturer",
         "predicate_or_attr": "name", "confidence": 0.9},
        {"column": "item_description", "role": "attribute", "entity": "product",
         "predicate_or_attr": "item_description"},
        {"column": "specs", "role": "attribute", "entity": "product",
         "predicate_or_attr": "specs"},
        {"column": "category", "role": "attribute", "entity": "product",
         "predicate_or_attr": "category"},
        {"column": "uom", "role": "attribute", "entity": "product",
         "predicate_or_attr": "uom"},
        {"column": "list_price", "role": "attribute", "entity": "product",
         "predicate_or_attr": "list_price"},
    ],
    "relationships": [
        {"subject": "product", "predicate": "manufactured_by", "object": "manufacturer",
         "why": "edge names the relation, not the source column"},
    ],
}


def _grainger_v2_refute() -> dict:
    corrected = copy.deepcopy(_GRAINGER_V2_REASON)
    corrected["entities"][0]["key_strategy"] = "synthetic"
    corrected["entities"][0]["key_columns"] = []
    corrected["columns"] = [c for c in corrected["columns"] if c["column"] != "specs"]
    return {
        "violations": [
            {"template": "KEY DROPS ROWS", "location": "entities.product",
             "evidence": "sku completeness 0.75 < 0.99", "severity": "error"},
            {"template": "DUPLICATE/DEAD ATTR", "location": "columns.specs",
             "evidence": "completeness 0.0 — all-empty column", "severity": "warning"},
        ],
        "corrected": corrected,
    }


_ALL_TESTS_PASS = {"existence": True, "identity": True, "universality": True}

_GRAINGER_COMPLETE = {
    "types": [
        {"type": "SKU", "promoted_from_attribute": "sku",
         "core_slots": [
             {"name": "issued_by", "kind": "relationship", "target_type": "Supplier",
              "why": "a distributor-specific identifier exists only relative to its issuer",
              "tests": _ALL_TESTS_PASS,
              "dataset_constant": {"value": "Grainger", "confidence": 0.9}},
             {"name": "identifies", "kind": "relationship", "target_type": "Product",
              "why": "an identifier identifies exactly one thing",
              "tests": _ALL_TESTS_PASS, "dataset_constant": None},
             {"name": "sku", "kind": "attribute", "target_type": None,
              "why": "the identifier string itself",
              "tests": _ALL_TESTS_PASS, "dataset_constant": None},
         ],
         "rejected": []},
        {"type": "MPN", "promoted_from_attribute": "mpn",
         "core_slots": [
             {"name": "issued_by", "kind": "relationship", "target_type": "Manufacturer",
              "why": "a maker-specific part number exists only relative to its issuer",
              "tests": _ALL_TESTS_PASS, "dataset_constant": None},
             {"name": "identifies", "kind": "relationship", "target_type": "Product",
              "why": "an identifier identifies exactly one thing",
              "tests": _ALL_TESTS_PASS, "dataset_constant": None},
             {"name": "mpn", "kind": "attribute", "target_type": None,
              "why": "the identifier string itself",
              "tests": _ALL_TESTS_PASS, "dataset_constant": None},
         ],
         "rejected": []},
        {"type": "Product", "promoted_from_attribute": None,
         "core_slots": [],
         "rejected": [
             {"name": "category", "failed_test": "universality",
              "why": "classification, not constitutive"},
             {"name": "list_price", "failed_test": "existence",
              "why": "a product can exist unpriced"},
             {"name": "specs", "failed_test": "existence",
              "why": "all-empty ad-hoc column"},
         ]},
    ],
}


async def _run_grainger(monkeypatch):
    return await _run_v2(
        monkeypatch, _GRAINGER_HEADERS, _grainger_rows(1000),
        _GRAINGER_V2_REASON, _grainger_v2_refute(), _GRAINGER_COMPLETE,
    )


class TestCompletionPassGraingerShape:
    """The COG-52 acceptance scenario, end-to-end with scripted LLM outputs."""

    @pytest.mark.asyncio
    async def test_extensions_carry_promotions_core_slots_and_rejections(self, monkeypatch):
        mapping, calls = await _run_grainger(monkeypatch)
        assert calls == [("reason", 0.0), ("refute", 0.0), ("complete", 0.0)]

        ext = mapping.ontology_extensions
        assert ext is not None
        by_type = {t.type_name: t for t in ext.types}
        # Dependent-identifier promotions, each recording its source attribute.
        assert by_type["SKU"].promoted_from_attribute == "sku"
        assert by_type["MPN"].promoted_from_attribute == "mpn"
        # Canonical dependent-identifier shape: issuer + identifies + id-string.
        for name in ("SKU", "MPN"):
            slots = {s.name: s for s in by_type[name].core_slots}
            assert slots["issued_by"].kind == "relationship" and slots["issued_by"].target_type
            assert slots["identifies"].kind == "relationship"
            assert slots["identifies"].target_type == "Product"
            assert any(s.kind == "attribute" for s in by_type[name].core_slots)
            assert len(by_type[name].core_slots) <= 3
        # The all-empty ad-hoc column minted NO attribute…
        assert all(c.column_name != "specs" for c in mapping.columns)
        # …and the rejected audit list is non-empty and names it.
        assert by_type["Product"].rejected
        assert any(r.name == "specs" for r in by_type["Product"].rejected)
        assert all(r.failed_test for r in by_type["Product"].rejected)

    @pytest.mark.asyncio
    async def test_applied_instances_and_edges(self, monkeypatch):
        rows = _grainger_rows(1000)
        mapping, _ = await _run_grainger(monkeypatch)
        applied = CSVResolver.apply_mapping(mapping, rows)
        # Row conservation still holds under the extension machinery.
        assert applied.rows_in == 1000 and applied.rows_dropped == 0

        by_type: dict[str, list] = {}
        for e in applied.entities:
            by_type.setdefault(e.type_name, []).append(e)
        # 750 non-empty sku cells → 750 SKU instances; mpn is fully complete.
        assert len(by_type["SKU"]) == 750
        assert len(by_type["MPN"]) == 1000
        assert len(by_type["Product"]) == 1000
        assert len(by_type["Manufacturer"]) == 290

        # Promoted instances identify their owner products (real ids, not stubs).
        product_ids = {e.id for e in by_type["Product"]}
        identifies = [r for r in applied.relationships if r.predicate == "identifies"]
        assert len(identifies) == 750 + 1000
        assert all(r.target_id in product_ids for r in identifies)

        # Dataset constant: exactly ONE issuer instance + per-instance edges.
        assert len(by_type["Supplier"]) == 1
        supplier = by_type["Supplier"][0]
        assert any(a.name == "name" and a.value == "Grainger" for a in supplier.attributes)
        issued = [r for r in applied.relationships if r.predicate == "issued_by"]
        sku_ids = {e.id for e in by_type["SKU"]}
        assert len(issued) == 750
        assert all(r.target_id == supplier.id for r in issued)
        assert all(r.source_id in sku_ids for r in issued)

        # MPN's issuer slot carries no constant → declarative only (the
        # Manufacturer instances present are the in-row dimension entities).
        assert not any(r.source_id not in sku_ids for r in issued)

        # Promoted instances carry the id string as a queryable attribute.
        sku0 = next(e for e in by_type["SKU"] if e.id == "GGR-0000000")
        assert any(a.name == "sku" and a.value == "GGR-0000000" for a in sku0.attributes)
        # The dead column never reaches instance data.
        assert not any(a.name == "specs" for e in applied.entities for a in e.attributes)

    @pytest.mark.asyncio
    async def test_promotions_held_for_review(self, monkeypatch):
        mapping, _ = await _run_grainger(monkeypatch)
        by_type = {t.type_name: t for t in mapping.ontology_extensions.types}
        # ALL promotions are judge-panel material → held for client confirm.
        assert by_type["SKU"].held_for_review
        assert by_type["MPN"].held_for_review
        # An unpromoted type with no low-confidence signal is not held…
        assert not by_type["Product"].held_for_review
        # …and the 0.9-confidence dataset constant is not held either.
        slots = {s.name: s for s in by_type["SKU"].core_slots}
        assert not slots["issued_by"].held_for_review


# Hand-built single-entity mapping + extensions: exercises apply_mapping's
# extension consumption directly, no LLM mocking. Canonical dependent-
# identifier shape with abstract names.


def _promotion_extensions(constant_confidence: float | None = 0.9) -> OntologyExtensions:
    return OntologyExtensions(types=[TypeExtension(
        type_name="Code",
        promoted_from_attribute="code",
        core_slots=[
            CoreSlot(name="issued_by", kind="relationship", target_type="Issuer",
                     dataset_constant=DatasetConstant(
                         value="Acme Registry", confidence=constant_confidence)),
            CoreSlot(name="identifies", kind="relationship", target_type="Item"),
            CoreSlot(name="code", kind="attribute"),
        ],
        held_for_review=True,  # promotions are always held (client confirm gate)
    )])


def _code_mapping(extensions: OntologyExtensions | None) -> CSVSchemaMapping:
    return CSVSchemaMapping(
        entity_type="Item",
        columns=[
            ColumnMapping(column_name="name", role=ColumnRole.TYPE_ID,
                          datatype="string", attribute_name="name"),
            ColumnMapping(column_name="code", role=ColumnRole.ATTRIBUTE,
                          datatype="string", attribute_name="code"),
        ],
        ontology_extensions=extensions,
    )


class TestApplyMappingExtensions:
    """apply_mapping consumption of ontology_extensions (single-entity path;
    the multi-entity path is covered by the Grainger-shape e2e above)."""

    def test_promoted_values_become_instances_with_identifies_edges(self):
        rows = [
            {"name": "One", "code": "C-1"},
            {"name": "Two", "code": "C-2"},
            {"name": "Three", "code": ""},  # empty source cell mints nothing
        ]
        applied = CSVResolver.apply_mapping(_code_mapping(_promotion_extensions()), rows)
        codes = [e for e in applied.entities if e.type_name == "Code"]
        assert {e.id for e in codes} == {"C-1", "C-2"}
        for code in codes:
            assert any(a.name == "code" and a.value == code.id for a in code.attributes)
        identifies = [r for r in applied.relationships if r.predicate == "identifies"]
        assert {(r.source_id, r.target_id) for r in identifies} == {
            ("C-1", "One"), ("C-2", "Two"),
        }
        # The promoted attribute still lands on the owner too (additive — the
        # column mapping is untouched by completion).
        one = next(e for e in applied.entities if e.id == "One")
        assert any(a.name == "code" and a.value == "C-1" for a in one.attributes)

    def test_dataset_constant_materializes_exactly_one_instance(self):
        rows = [{"name": f"N{i}", "code": f"C-{i}"} for i in range(50)]
        applied = CSVResolver.apply_mapping(_code_mapping(_promotion_extensions()), rows)
        issuers = [e for e in applied.entities if e.type_name == "Issuer"]
        assert len(issuers) == 1
        assert issuers[0].id == _safe_id("Acme Registry")
        assert any(a.name == "name" and a.value == "Acme Registry"
                   for a in issuers[0].attributes)
        edges = [r for r in applied.relationships if r.predicate == "issued_by"]
        assert len(edges) == 50  # one per promoted instance
        assert all(r.target_id == issuers[0].id for r in edges)

    def test_repeated_values_dedupe_instances_and_edges(self):
        rows = [{"name": "A", "code": "C-1"}, {"name": "B", "code": "C-1"}]
        applied = CSVResolver.apply_mapping(_code_mapping(_promotion_extensions()), rows)
        codes = [e for e in applied.entities if e.type_name == "Code"]
        assert len(codes) == 1  # same identifier string at the same issuer
        identifies = [r for r in applied.relationships if r.predicate == "identifies"]
        assert len(identifies) == 2  # …but it identifies two distinct owners
        issued = [r for r in applied.relationships if r.predicate == "issued_by"]
        assert len(issued) == 1  # edge dedup: one instance, one issuer

    def test_held_extensions_still_applied(self):
        # The confirm gate is CLIENT-SIDE: a held promotion present in the
        # posted mapping is applied — the client removes what the user rejects.
        ext = _promotion_extensions(constant_confidence=0.5)  # held constant too
        assert ext.types[0].held_for_review
        applied = CSVResolver.apply_mapping(
            _code_mapping(ext), [{"name": "One", "code": "C-1"}],
        )
        assert any(e.type_name == "Code" for e in applied.entities)
        assert any(e.type_name == "Issuer" for e in applied.entities)

    def test_constant_on_unpromoted_type_edges_from_its_instances(self):
        ext = OntologyExtensions(types=[TypeExtension(
            type_name="Item",
            core_slots=[CoreSlot(
                name="recorded_in", kind="relationship", target_type="Ledger",
                dataset_constant=DatasetConstant(value="Main", confidence=0.95),
            )],
        )])
        rows = [{"name": "One", "code": "C-1"}, {"name": "Two", "code": "C-2"}]
        applied = CSVResolver.apply_mapping(_code_mapping(ext), rows)
        ledgers = [e for e in applied.entities if e.type_name == "Ledger"]
        assert len(ledgers) == 1
        edges = [r for r in applied.relationships if r.predicate == "recorded_in"]
        assert {r.source_id for r in edges} == {"One", "Two"}
        assert all(r.target_id == ledgers[0].id for r in edges)

    def test_promotion_with_unknown_source_attribute_skipped(self):
        ext = OntologyExtensions(types=[TypeExtension(
            type_name="Ghost", promoted_from_attribute="no_such_column",
        )])
        applied = CSVResolver.apply_mapping(
            _code_mapping(ext), [{"name": "One", "code": "C-1"}],
        )
        # Ungroundable promotion is skipped (it still pre-registers in the
        # ontology at ingest) — never an error, never bogus instances.
        assert not any(e.type_name == "Ghost" for e in applied.entities)
        assert len(applied.entities) == 1

    def test_mapping_without_extensions_unchanged(self):
        rows = [{"name": "One", "code": "C-1"}]
        applied = CSVResolver.apply_mapping(_code_mapping(None), rows)
        assert {e.type_name for e in applied.entities} == {"Item"}
        assert applied.relationships == []


class TestCompletionShapeValidation:
    """_check_complete_shape: held_for_review marking, boundedness, and the
    degenerate-output → KeyError retry contract."""

    def test_promotion_always_held(self):
        ext = _check_complete_shape({"types": [{
            "type": "T", "promoted_from_attribute": "t", "confidence": 0.95,
            "core_slots": [], "rejected": [],
        }]})
        assert ext.types[0].held_for_review

    def test_low_type_confidence_held(self):
        ext = _check_complete_shape({"types": [{"type": "T", "confidence": 0.5}]})
        assert ext.types[0].held_for_review

    def test_confident_unpromoted_type_not_held(self):
        ext = _check_complete_shape({"types": [{"type": "T", "confidence": 0.9}]})
        assert not ext.types[0].held_for_review

    def test_low_confidence_constant_holds_slot(self):
        ext = _check_complete_shape({"types": [{"type": "T", "core_slots": [
            {"name": "s", "kind": "relationship", "target_type": "U",
             "dataset_constant": {"value": "x", "confidence": 0.6}},
        ]}]})
        assert ext.types[0].core_slots[0].held_for_review

    def test_constant_without_confidence_holds_slot(self):
        # The prompt mandates a confidence on dataset constants — one without
        # is suspect, so it is held for the user to confirm.
        ext = _check_complete_shape({"types": [{"type": "T", "core_slots": [
            {"name": "s", "kind": "relationship", "target_type": "U",
             "dataset_constant": {"value": "x"}},
        ]}]})
        assert ext.types[0].core_slots[0].held_for_review

    def test_confident_constant_not_held(self):
        ext = _check_complete_shape({"types": [{"type": "T", "core_slots": [
            {"name": "s", "kind": "relationship", "target_type": "U",
             "dataset_constant": {"value": "x", "confidence": 0.9}},
        ]}]})
        assert not ext.types[0].core_slots[0].held_for_review

    def test_tests_verdicts_preserved(self):
        ext = _check_complete_shape({"types": [{"type": "T", "core_slots": [
            {"name": "s", "kind": "attribute",
             "tests": {"existence": True, "identity": False, "universality": True}},
        ]}]})
        tests = ext.types[0].core_slots[0].tests
        assert tests.existence and not tests.identity and tests.universality

    def test_empty_types_list_is_valid_noop(self):
        assert _check_complete_shape({"types": []}) == OntologyExtensions(types=[])

    def test_missing_types_key_raises(self):
        with pytest.raises(KeyError):
            _check_complete_shape({"extensions": []})

    def test_unnamed_type_raises(self):
        with pytest.raises(KeyError):
            _check_complete_shape({"types": [{"core_slots": []}]})

    def test_unnamed_core_slot_raises(self):
        with pytest.raises(KeyError):
            _check_complete_shape({"types": [{"type": "T", "core_slots": [{"kind": "attribute"}]}]})

    def test_more_than_three_core_slots_fails_validation(self):
        # ADR 0003 boundedness cap: >3 "constitutive" slots means the model
        # listed commonly-associated, not constitutive — structural reject.
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            _check_complete_shape({"types": [{"type": "T", "core_slots": [
                {"name": f"s{i}", "kind": "attribute"} for i in range(4)
            ]}]})

    def test_typeextension_model_enforces_cap_directly(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TypeExtension(
                type_name="T",
                core_slots=[CoreSlot(name=f"s{i}") for i in range(4)],
            )

    def test_mapping_json_without_ontology_extensions_parses(self):
        # Backward compat: payloads serialized before COG-52 parse unchanged.
        mapping = CSVSchemaMapping.model_validate({
            "entity_type": "Item",
            "columns": [{"column_name": "a", "role": "attribute", "datatype": "string"}],
        })
        assert mapping.ontology_extensions is None
        # And a round-trip with extensions survives serialization.
        ext_mapping = _code_mapping(_promotion_extensions())
        again = CSVSchemaMapping.model_validate(ext_mapping.model_dump())
        assert again.ontology_extensions == ext_mapping.ontology_extensions


class TestCompletionRetry:
    """Pass D keeps the per-call retry-at-0.3 contract of Passes B/C."""

    @pytest.mark.asyncio
    async def test_complete_retries_at_higher_temp(self, monkeypatch):
        resolver = CSVResolver(client=None, openrouter_key="")
        calls = _mock_v2(
            monkeypatch, resolver, [_SIMPLE_REASON], [_echo_refute(_SIMPLE_REASON)],
            [{"missing": "types"}, {"types": []}],  # 1st degenerate
        )
        mapping = await resolver.infer_schema(_SIMPLE_HEADERS, _simple_rows(), {}, 10)
        assert mapping.ontology_extensions == OntologyExtensions(types=[])
        assert calls == [
            ("reason", 0.0), ("refute", 0.0), ("complete", 0.0), ("complete", 0.3),
        ]

    @pytest.mark.asyncio
    async def test_propagates_when_complete_retry_also_fails(self, monkeypatch):
        resolver = CSVResolver(client=None, openrouter_key="")
        bad = {"missing": "types"}
        _mock_v2(
            monkeypatch, resolver, [_SIMPLE_REASON], [_echo_refute(_SIMPLE_REASON)],
            [bad, copy.deepcopy(bad)],
        )
        with pytest.raises(KeyError):
            await resolver.infer_schema(_SIMPLE_HEADERS, _simple_rows(), {}, 10)

    @pytest.mark.asyncio
    async def test_overlong_core_slots_retry_then_propagate(self, monkeypatch):
        # The boundedness cap feeds the same retry contract: a >3-slot
        # completion is invalid output, retried once, then propagated.
        from pydantic import ValidationError
        resolver = CSVResolver(client=None, openrouter_key="")
        fat = {"types": [{"type": "T", "core_slots": [
            {"name": f"s{i}", "kind": "attribute"} for i in range(4)
        ]}]}
        calls = _mock_v2(
            monkeypatch, resolver, [_SIMPLE_REASON], [_echo_refute(_SIMPLE_REASON)],
            [fat, copy.deepcopy(fat)],
        )
        with pytest.raises(ValidationError):
            await resolver.infer_schema(_SIMPLE_HEADERS, _simple_rows(), {}, 10)
        assert calls[-2:] == [("complete", 0.0), ("complete", 0.3)]


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="live reason+refute inference needs OPENROUTER_API_KEY (network)",
)
class TestLiveHotelPMSInference:
    """End-to-end Pass A→B→C against the real model (deepseek/deepseek-v3.2
    via OpenRouter) on the hotel PMS export — ADR 0003's generality check:
    zero domain hints anywhere in the prompts. Run on a dev machine with:

        COGRAPH_DATASETS_ROOT=<parent repo> OPENROUTER_API_KEY=sk-or-... \
            pytest tests/test_csv_resolver.py -m integration -v
    """

    @pytest.mark.asyncio
    async def test_pms_reservations_reason_refute(self):
        headers, rows = _load_dataset("demo_data/hotel_design_partner/pms_reservations.csv")
        resolver = CSVResolver(client=None, openrouter_key=os.environ["OPENROUTER_API_KEY"])
        # Pin the validated model/provider for live runs regardless of env.
        resolver.EXTRACT_MODEL = "deepseek/deepseek-v3.2"
        resolver.EXTRACT_PROVIDER = "openrouter"

        mapping = await resolver.infer_schema(headers, rows, {}, total_rows=len(rows))

        # 1. The row decomposes into the three core concepts (exact names may
        #    legitimately vary under a domain-free prompt; accept close synonyms).
        assert mapping.entities, "v2 must return in-row entity specs"
        type_names = {e.type_name for e in mapping.entities}
        assert any(t in type_names for t in ("Reservation", "Booking")), type_names
        assert any(t in type_names for t in ("Property", "Hotel")), type_names
        assert any(t in type_names for t in ("Guest", "Person", "Customer")), type_names

        # 2. No entity keyed on a <99%-complete column (KEY DROPS ROWS).
        profile = profile_table(headers, rows, len(rows))
        for spec in mapping.entities:
            key_cols = ([spec.id_column] if spec.id_column else []) + list(spec.id_from or [])
            for key_col in key_cols:
                col = profile.column(key_col)
                assert col is not None, f"{spec.name} keyed on unknown column {key_col!r}"
                assert col.completeness >= 0.99, (
                    f"{spec.name} keyed on {key_col} "
                    f"({col.completeness:.2f} complete — drops rows)"
                )

        # 3. Edge predicates name relations, never source columns — the
        #    structural form of "predicates are verbs" (COLUMN-NAMED EDGE).
        lower_headers = {h.lower() for h in headers}
        rels = list(mapping.relationships or [])
        assert rels, "a PMS row bundles several entities — edges expected"
        for rel in rels:
            assert rel.predicate.lower() not in lower_headers, (
                f"edge predicate {rel.predicate!r} is a source column name"
            )

        # 4. The audit trail is present for the Explorer.
        assert mapping.inference_audit is not None
        assert mapping.inference_audit.pipeline == "reason_refute_v2"


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="live completion inference needs OPENROUTER_API_KEY (network)",
)
class TestLiveGraingerCompletion:
    """End-to-end Pass A→B→C→D against the real model (deepseek/deepseek-v3.2
    via OpenRouter) on the Grainger-shaped catalog — the COG-52 acceptance
    scenario with zero domain hints in any prompt. Run on a dev machine with:

        COGRAPH_DATASETS_ROOT=<parent repo> OPENROUTER_API_KEY=sk-or-... \
            pytest tests/test_csv_resolver.py -m integration -v
    """

    @pytest.mark.asyncio
    async def test_grainger_catalog_reason_refute_complete(self):
        headers, rows = _load_dataset("benchmarks/datasets/grainger-shaped-catalog.csv")
        resolver = CSVResolver(client=None, openrouter_key=os.environ["OPENROUTER_API_KEY"])
        # Pin the validated model/provider for live runs regardless of env.
        resolver.EXTRACT_MODEL = "deepseek/deepseek-v3.2"
        resolver.EXTRACT_PROVIDER = "openrouter"

        mapping = await resolver.infer_schema(headers, rows, {}, total_rows=len(rows))

        ext = mapping.ontology_extensions
        assert ext is not None and ext.types, "Pass D must emit extensions"

        # 1. A dependent-identifier promotion exists, with an issuer-like and
        #    an identifies-like relationship slot (two distinct targets — the
        #    canonical (issued_by → Issuer, identifies → Target) shape).
        promoted = [t for t in ext.types if t.promoted_from_attribute]
        assert promoted, "expected at least one dependent-identifier promotion"
        assert any(
            len({s.target_type for s in t.core_slots
                 if s.kind == "relationship" and s.target_type}) >= 2
            for t in promoted
        ), [(t.type_name, [(s.name, s.kind, s.target_type) for s in t.core_slots])
            for t in promoted]
        # Every promotion is held for the client-side confirm gate.
        assert all(t.held_for_review for t in promoted)

        # 2. The all-empty column minted NO attribute.
        assert all(c.column_name != "specs" for c in mapping.columns), [
            c.column_name for c in mapping.columns
        ]

        # 3. Boundedness + audit: ≤3 core slots per type (the pydantic cap
        #    enforces this — assert it held), and the rejected list is
        #    non-empty somewhere (aggressive rejection is mandated).
        assert all(len(t.core_slots) <= 3 for t in ext.types)
        assert any(t.rejected for t in ext.types), "expected rejected candidates"
