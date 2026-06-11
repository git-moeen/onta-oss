"""Tests for the deterministic table profiler (ADR 0003 Pass A)."""

import csv
import json
import os
import random
import time
from pathlib import Path

import pytest

from cograph_client.resolver.models import TableProfile, ValueShape
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


def _mutual_pairs(profile: TableProfile) -> set[frozenset]:
    """fd_mutual as unordered pairs, so tests don't depend on header order."""
    return {frozenset(pair) for pair in profile.fd_mutual}


def _profile_columns(**columns: list) -> TableProfile:
    """Build a profile from per-column value lists (all the same length)."""
    headers = list(columns)
    n_rows = max((len(v) for v in columns.values()), default=0)
    rows = [{h: columns[h][i] for h in headers} for i in range(n_rows)]
    return profile_table(headers, rows)


# ---------------------------------------------------------------------------
# Value shapes — every branch
# ---------------------------------------------------------------------------


class TestValueShape:
    def _shape(self, values: list) -> ValueShape:
        return _profile_columns(col=values).column("col").value_shape

    def test_empty_column(self):
        assert self._shape(["", "  ", None, ""]) == ValueShape.EMPTY

    def test_iso_dates(self):
        assert self._shape(["2026-01-02", "2026/3/4", "2026-12-31", "1999-1-1", "2026-05-05"]) == ValueShape.DATE

    def test_day_first_dates(self):
        assert self._shape(["1/2/26", "12-31-2026", "3/4/99", "25/12/2026", "9/9/26"]) == ValueShape.DATE

    def test_timestamps_count_as_dates(self):
        # The date regex is a prefix match — ISO timestamps qualify.
        assert self._shape(["2026-01-02T10:00:00", "2026-01-03 23:59", "2026-01-04T00:00", "2026-01-05T01:02", "2026-01-06 12:00"]) == ValueShape.DATE

    def test_date_below_threshold_is_not_date(self):
        # 3/5 = 0.6 dates, not > 0.8 — and the rest aren't numeric/code-shaped
        # enough either, so it lands on label.
        values = ["2026-01-02", "2026-01-03", "2026-01-04", "not a date x", "also no y"]
        assert self._shape(values) == ValueShape.LABEL

    def test_numbers(self):
        assert self._shape(["1", "2.5", "-3", "1e5", "1404.17"]) == ValueShape.NUMBER

    def test_numbers_beat_code_regex(self):
        # Pure digits match the code/id token regex too; number wins (checked first).
        assert self._shape(["12345", "67890", "11111", "22222", "33333"]) == ValueShape.NUMBER

    def test_code_id(self):
        assert self._shape(["RES-1000001", "HTL-DXB-01", "SKU_9.2", "a-b_c.d", "X1"]) == ValueShape.CODE_ID

    def test_code_id_requires_short_average(self):
        # Token-shaped but avg length > 24 → not code/id; no spaces → label.
        long_tokens = ["a" * 40, "b" * 41, "c" * 42, "d" * 43, "e" * 44]
        assert self._shape(long_tokens) == ValueShape.LABEL

    def test_code_id_leading_punctuation_disqualifies(self):
        # The token regex requires an alphanumeric first character.
        assert self._shape(["-abc", "-def", "-ghi", "-jkl", "-mno"]) == ValueShape.LABEL

    def test_text(self):
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "An employee fell from a ladder while replacing a fixture.",
            "Multiple fractures were reported after the incident occurred.",
        ]
        assert self._shape(sentences) == ValueShape.TEXT

    def test_short_spaced_values_are_labels(self):
        # Has spaces but avg length <= 25 → label, not text.
        assert self._shape(["New York", "Grand NYC", "London Park", "San Juan", "Palm Dubai"]) == ValueShape.LABEL

    def test_long_unspaced_values_are_labels(self):
        # avg length > 25 but no spaces (and not tokens — has commas) → label.
        values = ["a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x", "z,y,x,w,v,u,t,s,r,q,p,o,n,m,l,k,j,i,h,g,f,e,d,c"]
        assert self._shape(values) == ValueShape.LABEL

    def test_shape_ignores_empty_cells(self):
        # Empties don't dilute the fractions: 5 dates + 5 empties is still a date column.
        values = ["2026-01-0%d" % i for i in range(1, 6)] + ["", None, "", "", ""]
        assert self._shape(values) == ValueShape.DATE


# ---------------------------------------------------------------------------
# Per-column metrics
# ---------------------------------------------------------------------------


class TestColumnMetrics:
    def test_completeness_distinct_uniqueness_card_ratio(self):
        p = _profile_columns(c=["a", "b", "a", "", None])
        col = p.column("c")
        assert col.completeness == pytest.approx(3 / 5)
        assert col.distinct == 2
        assert col.uniqueness == pytest.approx(2 / 3)
        assert col.card_ratio == pytest.approx(2 / 5)

    def test_whitespace_only_is_empty(self):
        col = _profile_columns(c=["  ", "\t", "x"]).column("c")
        assert col.completeness == pytest.approx(1 / 3)
        assert col.distinct == 1

    def test_values_are_stripped_before_comparison(self):
        col = _profile_columns(c=[" a", "a ", "a"]).column("c")
        assert col.distinct == 1

    def test_examples_are_top3_by_frequency(self):
        col = _profile_columns(c=["x", "y", "y", "z", "z", "z", "w"]).column("c")
        assert col.examples == ["z", "y", "x"]

    def test_examples_capped_at_three(self):
        col = _profile_columns(c=["a", "b", "c", "d", "e"]).column("c")
        assert len(col.examples) == 3

    def test_examples_exclude_empty(self):
        col = _profile_columns(c=["", "", "", "only"]).column("c")
        assert col.examples == ["only"]

    def test_missing_keys_treated_as_empty(self):
        # Rows are dicts from JSON — a column may be absent from some rows.
        p = profile_table(["a", "b"], [{"a": "1"}, {"a": "2", "b": "x"}])
        assert p.column("b").completeness == pytest.approx(0.5)

    def test_empty_rows_list(self):
        p = profile_table(["a"], [])
        col = p.column("a")
        assert p.rows_profiled == 0
        assert col.completeness == 0.0
        assert col.uniqueness == 0.0
        assert col.card_ratio == 0.0
        assert col.value_shape == ValueShape.EMPTY

    def test_rows_profiled_vs_total_rows(self):
        p = profile_table(["a"], [{"a": "1"}, {"a": "2"}], total_rows=5000)
        assert p.rows_profiled == 2
        assert p.total_rows == 5000

    def test_total_rows_defaults_to_rows_profiled(self):
        p = profile_table(["a"], [{"a": "1"}, {"a": "2"}])
        assert p.total_rows == 2


# ---------------------------------------------------------------------------
# Derived flags — boundaries
# ---------------------------------------------------------------------------


class TestDerivedFlags:
    def test_complete_unique_key(self):
        col = _profile_columns(c=[f"id{i}" for i in range(100)]).column("c")
        assert col.complete_unique_key
        assert not col.incomplete

    def test_one_empty_in_100_kills_key_flag(self):
        # completeness 0.99 is NOT > 0.99 — boundary is strict.
        values = [f"id{i}" for i in range(99)] + [""]
        assert not _profile_columns(c=values).column("c").complete_unique_key

    def test_one_duplicate_in_100_kills_key_flag(self):
        # uniqueness 99/100 = 0.99 is NOT > 0.99.
        values = [f"id{i}" for i in range(99)] + ["id0"]
        assert not _profile_columns(c=values).column("c").complete_unique_key

    def test_incomplete_below_098(self):
        values = [f"id{i}" for i in range(97)] + ["", "", ""]
        assert _profile_columns(c=values).column("c").incomplete

    def test_not_incomplete_at_exactly_098(self):
        # completeness 98/100 = 0.98 is NOT < 0.98 — boundary is strict.
        values = [f"id{i}" for i in range(98)] + ["", ""]
        assert not _profile_columns(c=values).column("c").incomplete

    def test_low_cardinality_repeated(self):
        col = _profile_columns(c=["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"]).column("c")
        assert col.low_cardinality_repeated

    def test_constant_column_is_not_low_cardinality(self):
        # distinct must be > 1.
        assert not _profile_columns(c=["k"] * 10).column("c").low_cardinality_repeated

    def test_card_ratio_at_exactly_half_not_flagged(self):
        # 5 distinct / 10 rows = 0.5, NOT < 0.5.
        values = ["a", "b", "c", "d", "e"] * 2
        assert not _profile_columns(c=values).column("c").low_cardinality_repeated

    def test_low_cardinality_requires_actual_repeats(self):
        # 4 distinct over 10 rows but every non-empty value occurs once —
        # sparse-unique, not dimension-shaped.
        values = ["a", "b", "c", "d", "", "", "", "", "", ""]
        assert not _profile_columns(c=values).column("c").low_cardinality_repeated

    def test_all_empty_column_flags(self):
        col = _profile_columns(c=["", "", ""]).column("c")
        assert col.value_shape == ValueShape.EMPTY
        assert col.incomplete
        assert not col.complete_unique_key
        assert not col.low_cardinality_repeated


# ---------------------------------------------------------------------------
# Non-string cells (rows arrive from JSON, not just CSV strings)
# ---------------------------------------------------------------------------


class TestNonStringCells:
    def test_ints_and_floats(self):
        col = _profile_columns(c=[1, 2.5, 3, 1, 2.5]).column("c")
        assert col.value_shape == ValueShape.NUMBER
        assert col.distinct == 3
        assert col.completeness == 1.0

    def test_none_is_empty(self):
        col = _profile_columns(c=[None, None, "x", None]).column("c")
        assert col.completeness == pytest.approx(0.25)

    def test_lists_are_normalized_deterministically(self):
        col = _profile_columns(c=[["a", "b"], ["a", "b"], ["c"]]).column("c")
        assert col.distinct == 2

    def test_empty_list_is_empty(self):
        col = _profile_columns(c=[[], ["x"], []]).column("c")
        assert col.completeness == pytest.approx(1 / 3)

    def test_int_zero_is_a_value_not_empty(self):
        # Falsy numbers are real values — only None/''/empty containers are empty.
        col = _profile_columns(c=[0, 0, 1]).column("c")
        assert col.completeness == 1.0

    def test_mixed_types_count_distinct_by_normalized_string(self):
        # int 1 and string "1" normalize to the same value.
        col = _profile_columns(c=[1, "1", 2]).column("c")
        assert col.distinct == 2

    def test_numbers_in_fd_columns(self):
        # FD detection works over numeric cells too.
        codes = [10, 20, 10, 20, 10, 20, 10, 20]
        titles = ["ten", "twenty", "ten", "twenty", "ten", "twenty", "ten", "twenty"]
        p = _profile_columns(code=codes, title=titles)
        assert frozenset({"code", "title"}) in _mutual_pairs(p)


# ---------------------------------------------------------------------------
# Functional dependencies
# ---------------------------------------------------------------------------


class TestFunctionalDependencies:
    def test_mutual_fd_code_title(self):
        codes = ["c1", "c2", "c3"] * 4
        titles = ["t1", "t2", "t3"] * 4
        p = _profile_columns(code=codes, title=titles)
        assert frozenset({"code", "title"}) in _mutual_pairs(p)
        assert p.fd_oneway == []

    def test_oneway_fd_hierarchy(self):
        # Each city has one country; countries span multiple cities.
        cities = ["nyc", "sf", "lon", "par"] * 3
        countries = ["us", "us", "uk", "fr"] * 3
        p = _profile_columns(city=cities, country=countries)
        assert ("city", "country") in p.fd_oneway
        assert _mutual_pairs(p) == set()

    def test_violated_fd_not_reported(self):
        a = ["x", "x", "y", "y", "z", "z"]
        b = ["1", "2", "3", "4", "5", "6"]  # x maps to both 1 and 2
        p = _profile_columns(a=a, b=b)
        assert p.fd_mutual == []
        assert ("a", "b") not in p.fd_oneway

    def test_support_below_5_not_reported(self):
        p = _profile_columns(code=["c1", "c2"] * 2, title=["t1", "t2"] * 2)
        assert p.fd_mutual == []
        assert p.fd_oneway == []

    def test_constant_determinant_not_reported(self):
        # A constant column "determines" everything trivially — noise.
        p = _profile_columns(const=["k"] * 8, dim=["a", "b"] * 4)
        assert p.fd_mutual == []
        assert p.fd_oneway == []

    def test_near_unique_determinant_filtered(self):
        # 10 distinct A over 10 rows: a unique column trivially maps each
        # value to one B — len(distinct A) < 0.95*support fails (10 < 9.5).
        a = [f"u{i}" for i in range(10)]
        b = ["x", "y"] * 5
        p = _profile_columns(a=a, b=b)
        assert ("a", "b") not in p.fd_oneway
        assert p.fd_mutual == []

    def test_two_unique_key_columns_do_not_pair(self):
        # Two parallel key columns map 1:1 but both are near-unique — the
        # filter must kill the pair in both directions.
        a = [f"id{i}" for i in range(20)]
        b = [f"conf{i}" for i in range(20)]
        p = _profile_columns(a=a, b=b)
        assert p.fd_mutual == []
        assert p.fd_oneway == []

    def test_repeated_determinant_passes_near_unique_filter(self):
        # 4 distinct over 12 supported rows: 4 < 0.95*12 — kept.
        a = ["p1", "p2", "p3", "p4"] * 3
        b = ["n1", "n2", "n3", "n4"] * 3
        p = _profile_columns(a=a, b=b)
        assert frozenset({"a", "b"}) in _mutual_pairs(p)

    def test_empty_cells_do_not_count_or_violate(self):
        # Rows where either side is empty are excluded from support and can't
        # violate the mapping (sparse code<->title pairs still pair up).
        codes = ["c1", "c2", "", "c1", "c2", "", "c1", "c2", "", "c1", "c2", ""]
        titles = ["t1", "t2", "", "t1", "t2", "", "t1", "t2", "", "t1", "t2", ""]
        p = _profile_columns(code=codes, title=titles)
        assert frozenset({"code", "title"}) in _mutual_pairs(p)

    def test_high_cardinality_columns_excluded(self):
        # distinct > 400 → not an FD candidate even with a perfect mapping.
        n = 450
        a = [f"code{i}" for i in range(n)] * 2
        b = [f"title{i}" for i in range(n)] * 2
        p = _profile_columns(a=a, b=b)
        assert p.fd_mutual == []
        assert p.fd_oneway == []

    def test_all_empty_column_excluded(self):
        p = _profile_columns(empty=[""] * 8, dim=["a", "b"] * 4)
        assert p.fd_mutual == []
        assert p.fd_oneway == []

    def test_pair_order_follows_headers(self):
        p = profile_table(
            ["code", "title"],
            [{"code": c, "title": t} for c, t in zip(["c1", "c2"] * 4, ["t1", "t2"] * 4)],
        )
        assert p.fd_mutual == [("code", "title")]

    def test_oneway_reports_determinant_first(self):
        # country comes first in the headers, but the FD runs city -> country.
        p = profile_table(
            ["country", "city"],
            [
                {"country": c, "city": ci}
                for c, ci in zip(["us", "us", "uk", "fr"] * 3, ["nyc", "sf", "lon", "par"] * 3)
            ],
        )
        assert ("city", "country") in p.fd_oneway


# ---------------------------------------------------------------------------
# to_prompt_dict
# ---------------------------------------------------------------------------


class TestToPromptDict:
    def _profile(self) -> TableProfile:
        # "plain" has 6 distinct over 10 rows (card_ratio 0.6): complete but
        # neither unique nor dimension-shaped — no flags at all.
        plain = ["p1", "p2", "p3", "p4", "p5", "p6", "p1", "p2", "p3", "p4"]
        rows = [
            {"id": f"R{i}", "dim": ["a", "b"][i % 2], "dim_name": ["alpha", "beta"][i % 2],
             "plain": plain[i], "note": ""}
            for i in range(10)
        ]
        return profile_table(["id", "dim", "dim_name", "plain", "note"], rows, total_rows=100)

    def test_structure_and_coverage(self):
        d = self._profile().to_prompt_dict()
        assert d["rows_profiled"] == 10
        assert d["total_rows"] == 100
        assert set(d["columns"]) == {"id", "dim", "dim_name", "plain", "note"}
        assert "dim <-> dim_name" in d["fd_mutual"]

    def test_flags_only_when_set(self):
        d = self._profile().to_prompt_dict()
        assert d["columns"]["id"]["flags"] == ["complete_unique_key"]
        assert d["columns"]["dim"]["flags"] == ["low_cardinality_repeated"]
        # 'note' is all-empty → incomplete flag present.
        assert "incomplete" in d["columns"]["note"]["flags"]
        # No spurious flags on an unremarkable column — key omitted entirely.
        assert "flags" not in d["columns"]["plain"]

    def test_floats_rounded(self):
        rows = [{"c": "a"}, {"c": "b"}, {"c": "a"}]
        d = profile_table(["c"], rows).to_prompt_dict()
        assert d["columns"]["c"]["complete"] == 1.0
        assert d["columns"]["c"]["unique"] == round(2 / 3, 3)

    def test_long_examples_truncated(self):
        long_value = "x" * 100
        d = profile_table(["c"], [{"c": long_value}] * 3).to_prompt_dict(max_example_len=40)
        example = d["columns"]["c"]["examples"][0]
        assert len(example) == 40
        assert example.endswith("…")

    def test_oneway_rendered_as_arrows(self):
        p = _profile_columns(
            city=["nyc", "sf", "lon", "par"] * 3,
            country=["us", "us", "uk", "fr"] * 3,
        )
        assert "city -> country" in p.to_prompt_dict()["fd_oneway"]

    def test_json_serializable(self):
        json.dumps(self._profile().to_prompt_dict())


# ---------------------------------------------------------------------------
# Performance budget
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_5k_rows_30_cols_under_one_second(self):
        rng = random.Random(42)
        headers = [f"col{i}" for i in range(30)]
        rows = []
        for r in range(5000):
            row = {}
            for i in range(10):  # unique-ish id columns
                row[f"col{i}"] = f"id{i}-{r}"
            for i in range(10, 20):  # low-cardinality numeric columns
                row[f"col{i}"] = str(rng.randint(0, 50))
            for i in range(20, 30):  # low-cardinality label columns
                row[f"col{i}"] = f"cat {rng.randint(0, 30)}"
            rows.append(row)

        start = time.perf_counter()
        p = profile_table(headers, rows, total_rows=5000)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, f"profiling took {elapsed:.2f}s (budget 1s)"
        assert p.rows_profiled == 5000
        assert len(p.columns) == 30


# ---------------------------------------------------------------------------
# Real datasets (skipped when the gitignored CSVs are absent)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def osha_profile() -> TableProfile:
    headers, rows = _load_dataset("benchmarks/datasets/osha-severe-injuries.csv")
    return profile_table(headers, rows, total_rows=len(rows))


@pytest.fixture(scope="module")
def cfpb_profile() -> TableProfile:
    headers, rows = _load_dataset("benchmarks/datasets/cfpb-debt-collection.csv")
    return profile_table(headers, rows, total_rows=len(rows))


@pytest.fixture(scope="module")
def hotel_profile() -> TableProfile:
    headers, rows = _load_dataset("demo_data/hotel_design_partner/pms_reservations.csv")
    return profile_table(headers, rows, total_rows=len(rows))


class TestOSHADataset:
    def test_code_title_mutual_fds(self, osha_profile):
        pairs = _mutual_pairs(osha_profile)
        assert frozenset({"Nature", "NatureTitle"}) in pairs
        assert frozenset({"Event", "EventTitle"}) in pairs
        assert frozenset({"Part of Body", "Part of Body Title"}) in pairs
        assert frozenset({"Secondary Source", "Secondary Source Title"}) in pairs

    def test_inspection_flagged_incomplete(self, osha_profile):
        col = osha_profile.column("Inspection")
        assert col is not None
        assert col.incomplete
        assert col.completeness == pytest.approx(0.394, abs=0.02)

    def test_secondary_source_flagged_incomplete(self, osha_profile):
        col = osha_profile.column("Secondary Source")
        assert col is not None
        assert col.incomplete
        assert col.completeness == pytest.approx(0.274, abs=0.02)

    def test_id_and_upa_are_complete_unique_keys(self, osha_profile):
        for name in ("ID", "UPA"):
            col = osha_profile.column(name)
            assert col is not None
            assert col.complete_unique_key, f"{name} should be a complete unique key"


class TestCFPBDataset:
    def test_only_complaint_id_is_complete_unique_key(self, cfpb_profile):
        keys = [c.name for c in cfpb_profile.columns if c.complete_unique_key]
        assert keys == ["Complaint ID"]

    def test_company_is_low_cardinality_repeated(self, cfpb_profile):
        col = cfpb_profile.column("Company")
        assert col is not None
        assert col.low_cardinality_repeated
        assert 1 < col.distinct
        assert col.card_ratio < 0.5


class TestHotelPMSDataset:
    def test_property_id_name_mutual_fd(self, hotel_profile):
        assert frozenset({"property_id", "property_name"}) in _mutual_pairs(hotel_profile)

    def test_loyalty_number_incomplete(self, hotel_profile):
        col = hotel_profile.column("loyalty_number")
        assert col is not None
        assert col.incomplete
        assert col.completeness == pytest.approx(0.36, abs=0.02)
