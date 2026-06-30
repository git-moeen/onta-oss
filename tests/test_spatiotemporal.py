"""Tests for the spatio-temporal entity index (COG-103).

Covers the in-memory default end-to-end, the factory/registration wiring, and the
PostGIS adapter's emitted SQL via a fake asyncpg pool (no real Postgres). One
``@pytest.mark.integration`` test exercises a live DB when ``OMNIX_DATABASE_URL`` is
set, and skips otherwise.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from cograph_client.spatiotemporal import (
    InMemorySpatioTemporalIndex,
    STQueryResult,
    SpatioTemporalFact,
    SpatioTemporalIndex,
    get_spatiotemporal_index,
    make_spatiotemporal_index,
    register_spatiotemporal_index,
    reset_spatiotemporal_index,
)
from cograph_client.spatiotemporal.postgis import PostGISSpatioTemporalIndex

# Real-world coordinates (lon, lat).
SF_FERRY = (-122.3933, 37.7956)  # SF Ferry Building
SF_CITY_HALL = (-122.4194, 37.7793)  # ~3 km from the Ferry Building
NYC_TIMES_SQ = (-73.9855, 40.7580)  # New York — ~4000 km away

TENANT = "demo-tenant"
OTHER_TENANT = "spider-bench"
KG = "kg1"
OTHER_KG = "kg2"


def _dt(y: int, m: int = 1, d: int = 1) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _fact(
    uri: str,
    lon: float,
    lat: float,
    *,
    tenant: str = TENANT,
    kg: str = KG,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    attrs: dict | None = None,
) -> SpatioTemporalFact:
    return SpatioTemporalFact(
        entity_uri=uri,
        tenant_id=tenant,
        kg_name=kg,
        lon=lon,
        lat=lat,
        valid_from=valid_from,
        valid_to=valid_to,
        attrs=attrs or {"label": uri},
    )


@pytest.fixture
def idx() -> InMemorySpatioTemporalIndex:
    return InMemorySpatioTemporalIndex()


@pytest.fixture(autouse=True)
def _reset_registry():
    reset_spatiotemporal_index()
    yield
    reset_spatiotemporal_index()


def _uris(results: list[STQueryResult]) -> set[str]:
    return {r.entity_uri for r in results}


# ---------------------------------------------------------------------------
# InMemory backend
# ---------------------------------------------------------------------------


async def test_protocol_conformance(idx):
    assert isinstance(idx, SpatioTemporalIndex)
    assert isinstance(PostGISSpatioTemporalIndex(dsn="postgres://x"), SpatioTemporalIndex)


async def test_upsert_and_query_radius(idx):
    await idx.upsert(_fact("e:ferry", *SF_FERRY))
    await idx.upsert_many(
        [_fact("e:cityhall", *SF_CITY_HALL), _fact("e:times", *NYC_TIMES_SQ)]
    )

    # 5 km around the Ferry Building → both SF points, not NYC.
    res = await idx.query_radius(TENANT, SF_FERRY[0], SF_FERRY[1], 5_000)
    assert _uris(res) == {"e:ferry", "e:cityhall"}

    # 500 m → only the Ferry Building itself.
    res = await idx.query_radius(TENANT, SF_FERRY[0], SF_FERRY[1], 500)
    assert _uris(res) == {"e:ferry"}


async def test_query_radius_returns_attrs(idx):
    await idx.upsert(_fact("e:ferry", *SF_FERRY, attrs={"label": "Ferry", "kind": "pier"}))
    res = await idx.query_radius(TENANT, SF_FERRY[0], SF_FERRY[1], 1_000)
    assert res[0].attrs == {"label": "Ferry", "kind": "pier"}


async def test_query_bbox(idx):
    await idx.upsert_many(
        [
            _fact("e:ferry", *SF_FERRY),
            _fact("e:cityhall", *SF_CITY_HALL),
            _fact("e:times", *NYC_TIMES_SQ),
        ]
    )
    # Box around the SF Bay area.
    res = await idx.query_bbox(TENANT, -122.6, 37.6, -122.3, 37.9)
    assert _uris(res) == {"e:ferry", "e:cityhall"}


async def test_query_polygon(idx):
    await idx.upsert_many(
        [_fact("e:ferry", *SF_FERRY), _fact("e:times", *NYC_TIMES_SQ)]
    )
    # A polygon box covering SF only.
    wkt = "POLYGON((-122.6 37.6, -122.3 37.6, -122.3 37.9, -122.6 37.9, -122.6 37.6))"
    res = await idx.query_polygon(TENANT, wkt)
    assert _uris(res) == {"e:ferry"}


async def test_query_polygon_bbox_fallback(idx):
    """An unparseable ring falls back to the BBOX of the coords (not all facts)."""
    await idx.upsert_many(
        [_fact("e:ferry", *SF_FERRY), _fact("e:times", *NYC_TIMES_SQ)]
    )
    # A malformed WKT whose outer ring won't parse (a non-numeric coordinate
    # token makes _parse_wkt_polygon return None), but whose extractable numeric
    # coords describe an SF-only box. The bbox fallback must include SF, exclude
    # NYC — NOT return everything (which the old `lambda: True` fallback did).
    malformed = "POLYGON((-122.6 37.6, -122.3 37.6, foo bar, -122.6 37.9))"
    res = await idx.query_polygon(TENANT, malformed)
    assert _uris(res) == {"e:ferry"}


async def test_query_polygon_no_coords_no_spatial_filter(idx):
    """Only when NO coords are extractable does the spatial filter drop entirely."""
    await idx.upsert_many(
        [_fact("e:ferry", *SF_FERRY), _fact("e:times", *NYC_TIMES_SQ)]
    )
    res = await idx.query_polygon(TENANT, "POLYGON((no numbers here))")
    assert _uris(res) == {"e:ferry", "e:times"}


async def test_time_window_overlap(idx):
    await idx.upsert_many(
        [
            _fact("e:old", *SF_FERRY, valid_from=_dt(2020), valid_to=_dt(2021)),
            _fact("e:now", *SF_FERRY, valid_from=_dt(2025), valid_to=_dt(2027)),
        ]
    )
    # Window in 2026 overlaps only the "now" fact.
    res = await idx.query_radius(
        TENANT, *SF_FERRY, 1_000, time_window=(_dt(2026), _dt(2026, 6))
    )
    assert _uris(res) == {"e:now"}

    # A wide window overlaps both.
    res = await idx.query_radius(
        TENANT, *SF_FERRY, 1_000, time_window=(_dt(2019), _dt(2030))
    )
    assert _uris(res) == {"e:old", "e:now"}


async def test_as_of_containment(idx):
    await idx.upsert_many(
        [
            _fact("e:old", *SF_FERRY, valid_from=_dt(2020), valid_to=_dt(2021)),
            _fact("e:now", *SF_FERRY, valid_from=_dt(2025), valid_to=_dt(2027)),
        ]
    )
    res = await idx.query_radius(TENANT, *SF_FERRY, 1_000, as_of=_dt(2026))
    assert _uris(res) == {"e:now"}

    # Half-open [from, to): the end instant is excluded.
    res = await idx.query_radius(TENANT, *SF_FERRY, 1_000, as_of=_dt(2021))
    assert _uris(res) == set()


async def test_as_of_wins_over_time_window(idx):
    # as_of contains the "now" fact; the window would also match "old" — but as_of
    # takes precedence per the Protocol contract, so only "now" comes back.
    await idx.upsert_many(
        [
            _fact("e:old", *SF_FERRY, valid_from=_dt(2020), valid_to=_dt(2021)),
            _fact("e:now", *SF_FERRY, valid_from=_dt(2025), valid_to=_dt(2027)),
        ]
    )
    res = await idx.query_radius(
        TENANT, *SF_FERRY, 1_000, time_window=(_dt(2019), _dt(2030)), as_of=_dt(2026)
    )
    assert _uris(res) == {"e:now"}


async def test_open_ended_validity(idx):
    await idx.upsert(_fact("e:openend", *SF_FERRY, valid_from=_dt(2020), valid_to=None))
    # Anything from 2020 on contains it.
    res = await idx.query_radius(TENANT, *SF_FERRY, 1_000, as_of=_dt(2099))
    assert _uris(res) == {"e:openend"}
    # Before 2020 it does not.
    res = await idx.query_radius(TENANT, *SF_FERRY, 1_000, as_of=_dt(2010))
    assert _uris(res) == set()


async def test_upsert_idempotent_on_entity_and_valid_time(idx):
    f1 = _fact("e:ferry", *SF_FERRY, valid_from=_dt(2025), valid_to=_dt(2026), attrs={"v": 1})
    f2 = _fact("e:ferry", *SF_FERRY, valid_from=_dt(2025), valid_to=_dt(2026), attrs={"v": 2})
    await idx.upsert(f1)
    await idx.upsert(f2)  # same (uri, valid_time) → replaces, no duplicate
    res = await idx.query_radius(TENANT, *SF_FERRY, 1_000)
    assert len(res) == 1
    assert res[0].attrs == {"v": 2}

    # A different valid_time for the same URI is a distinct row.
    f3 = _fact("e:ferry", *SF_FERRY, valid_from=_dt(2027), valid_to=_dt(2028), attrs={"v": 3})
    await idx.upsert(f3)
    res = await idx.query_radius(TENANT, *SF_FERRY, 1_000)
    assert len(res) == 2


async def test_delete(idx):
    await idx.upsert_many([_fact("e:ferry", *SF_FERRY), _fact("e:cityhall", *SF_CITY_HALL)])
    await idx.delete("e:ferry", TENANT)
    res = await idx.query_radius(TENANT, *SF_FERRY, 5_000)
    assert _uris(res) == {"e:cityhall"}


async def test_clear_per_tenant(idx):
    await idx.upsert(_fact("e:a", *SF_FERRY, tenant=TENANT))
    await idx.upsert(_fact("e:b", *SF_FERRY, tenant=OTHER_TENANT))
    await idx.clear(TENANT)
    assert _uris(await idx.query_radius(TENANT, *SF_FERRY, 5_000)) == set()
    assert _uris(await idx.query_radius(OTHER_TENANT, *SF_FERRY, 5_000)) == {"e:b"}


async def test_tenant_isolation(idx):
    await idx.upsert(_fact("e:a", *SF_FERRY, tenant=TENANT))
    await idx.upsert(_fact("e:b", *SF_FERRY, tenant=OTHER_TENANT))
    # Each query only sees its own tenant's facts.
    assert _uris(await idx.query_radius(TENANT, *SF_FERRY, 5_000)) == {"e:a"}
    assert _uris(await idx.query_bbox(OTHER_TENANT, -123, 37, -122, 38)) == {"e:b"}


# ---------------------------------------------------------------------------
# Per-KG dimension (memory)
# ---------------------------------------------------------------------------


async def test_kg_narrowing_on_query(idx):
    await idx.upsert(_fact("e:a", *SF_FERRY, kg=KG))
    await idx.upsert(_fact("e:b", *SF_CITY_HALL, kg=OTHER_KG))
    # No kg_name → both KGs in the tenant.
    assert _uris(await idx.query_radius(TENANT, *SF_FERRY, 5_000)) == {"e:a", "e:b"}
    # Narrowed to one KG.
    assert _uris(await idx.query_radius(TENANT, *SF_FERRY, 5_000, kg_name=KG)) == {"e:a"}
    assert (
        _uris(await idx.query_bbox(TENANT, -123, 37, -122, 38, kg_name=OTHER_KG))
        == {"e:b"}
    )


async def test_clear_one_kg_leaves_sibling(idx):
    """The crux of the kg_name dimension: dropping one KG must not wipe a sibling."""
    await idx.upsert(_fact("e:a", *SF_FERRY, kg=KG))
    await idx.upsert(_fact("e:b", *SF_CITY_HALL, kg=OTHER_KG))
    await idx.clear(TENANT, kg_name=KG)
    assert _uris(await idx.query_radius(TENANT, *SF_FERRY, 50_000)) == {"e:b"}
    # A tenant-wide clear (no kg_name) still removes everything.
    await idx.clear(TENANT)
    assert _uris(await idx.query_radius(TENANT, *SF_FERRY, 50_000)) == set()


async def test_same_uri_distinct_per_kg(idx):
    """Same entity URI in two KGs are distinct rows (kg_name is part of the key)."""
    await idx.upsert(_fact("e:shared", *SF_FERRY, kg=KG, attrs={"v": "a"}))
    await idx.upsert(_fact("e:shared", *SF_FERRY, kg=OTHER_KG, attrs={"v": "b"}))
    assert len(await idx.query_radius(TENANT, *SF_FERRY, 1_000)) == 2
    one = await idx.query_radius(TENANT, *SF_FERRY, 1_000, kg_name=OTHER_KG)
    assert one[0].attrs == {"v": "b"}


async def test_delete_scoped_to_kg(idx):
    await idx.upsert(_fact("e:shared", *SF_FERRY, kg=KG))
    await idx.upsert(_fact("e:shared", *SF_FERRY, kg=OTHER_KG))
    await idx.delete("e:shared", TENANT, kg_name=KG)
    remaining = await idx.query_radius(TENANT, *SF_FERRY, 1_000)
    assert len(remaining) == 1  # only the OTHER_KG row survives


# ---------------------------------------------------------------------------
# Factory + registration
# ---------------------------------------------------------------------------


def test_factory_returns_inmemory_when_no_dsn(monkeypatch):
    from cograph_client import config

    monkeypatch.setattr(config.settings, "database_url", "", raising=False)
    assert isinstance(make_spatiotemporal_index(), InMemorySpatioTemporalIndex)


def test_factory_returns_postgis_when_dsn_set(monkeypatch):
    from cograph_client import config

    monkeypatch.setattr(config.settings, "database_url", "postgres://u@h/db", raising=False)
    assert isinstance(make_spatiotemporal_index(), PostGISSpatioTemporalIndex)


def test_register_and_get_roundtrip(idx):
    register_spatiotemporal_index(idx)
    assert get_spatiotemporal_index() is idx
    register_spatiotemporal_index(None)
    # Falls back to a lazily-built (cached) default.
    default = get_spatiotemporal_index()
    assert isinstance(default, InMemorySpatioTemporalIndex)
    assert get_spatiotemporal_index() is default  # cached


# ---------------------------------------------------------------------------
# PostGIS adapter — fake asyncpg pool (no real DB)
# ---------------------------------------------------------------------------


class FakeConn:
    """Records every execute/fetch call; returns canned rows for fetch."""

    def __init__(self, recorder):
        self._rec = recorder
        self.rows: list[dict] = []

    async def execute(self, sql, *args):
        self._rec.append(("execute", sql, args))
        return "OK"

    async def fetch(self, sql, *args):
        self._rec.append(("fetch", sql, args))
        return self.rows

    def transaction(self):
        conn = self

        class _Tx:
            async def __aenter__(self_inner):
                return conn

            async def __aexit__(self_inner, *exc):
                return False

        return _Tx()


class FakeAcquire:
    def __init__(self, conn):
        self._conn = conn

    async def __aenter__(self):
        return self._conn

    async def __aexit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn

    def acquire(self):
        return FakeAcquire(self._conn)


@pytest.fixture
def pg(monkeypatch):
    """A PostGISSpatioTemporalIndex wired to a fake asyncpg pool.

    Yields (store, recorder, conn). ``recorder`` is the list of (op, sql, args).
    """
    recorder: list = []
    conn = FakeConn(recorder)
    pool = FakePool(conn)

    async def fake_create_pool(*a, **k):
        recorder.append(("create_pool", a, k))
        return pool

    import asyncpg

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    store = PostGISSpatioTemporalIndex(dsn="postgres://user@host/db")
    return store, recorder, conn


def _sqls(recorder, op=None) -> list[str]:
    return [sql for (o, sql, *_rest) in recorder if op is None or o == op]


async def test_postgis_ddl_and_extensions(pg):
    store, recorder, _conn = pg
    await store._ensure_pool()
    ddl = " ".join(_sqls(recorder, "execute"))
    assert "CREATE EXTENSION IF NOT EXISTS postgis" in ddl
    assert "CREATE EXTENSION IF NOT EXISTS btree_gist" in ddl
    assert "CREATE TABLE IF NOT EXISTS entity_spatiotemporal" in ddl
    assert "geometry(Geometry, 4326)" in ddl
    assert "valid_time tstzrange" in ddl
    assert "attrs      jsonb" in ddl
    # Composite GiST index over (tenant_id, kg_name, geom, valid_time).
    assert "USING GIST (tenant_id, kg_name, geom, valid_time)" in ddl
    assert "PRIMARY KEY (tenant_id, kg_name, entity_uri, valid_time)" in ddl
    assert "kg_name    text NOT NULL" in ddl


async def test_postgis_extension_failure_tolerated(monkeypatch):
    """A permission error on CREATE EXTENSION must not abort setup (COG-102 infra)."""
    recorder: list = []

    class PickyConn(FakeConn):
        async def execute(self, sql, *args):
            recorder.append(("execute", sql, args))
            if "CREATE EXTENSION" in sql:
                raise RuntimeError("permission denied to create extension")
            return "OK"

    conn = PickyConn(recorder)
    pool = FakePool(conn)

    async def fake_create_pool(*a, **k):
        return pool

    import asyncpg

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    store = PostGISSpatioTemporalIndex(dsn="postgres://u@h/db")
    await store._ensure_pool()  # must NOT raise
    ddl = " ".join(s for (_o, s, *_r) in recorder)
    assert "CREATE TABLE IF NOT EXISTS entity_spatiotemporal" in ddl


async def test_postgis_upsert_sql(pg):
    store, recorder, _conn = pg
    await store.upsert(
        _fact("e:ferry", *SF_FERRY, valid_from=_dt(2025), valid_to=_dt(2026), attrs={"label": "Ferry"})
    )
    # The last execute is the INSERT (DDL ran first).
    inserts = [
        (sql, args)
        for (op, sql, args) in recorder
        if op == "execute" and "INSERT INTO entity_spatiotemporal" in sql
    ]
    assert inserts, "no INSERT was emitted"
    sql, args = inserts[-1]
    assert "ST_SetSRID(ST_MakePoint($6, $7), 4326)" in sql
    assert "tstzrange($4::timestamptz, $5::timestamptz, '[)')" in sql
    assert "ON CONFLICT (tenant_id, kg_name, entity_uri, valid_time) DO UPDATE" in sql
    # args order: uri, tenant, attrs_json, from, to, lon, lat, kg_name
    assert args[0] == "e:ferry"
    assert args[1] == TENANT
    assert '"label": "Ferry"' in args[2]
    assert args[3] == _dt(2025) and args[4] == _dt(2026)
    assert args[5] == SF_FERRY[0] and args[6] == SF_FERRY[1]
    assert args[7] == KG


async def test_postgis_upsert_many_uses_transaction(pg):
    store, recorder, _conn = pg
    await store.upsert_many([_fact("e:a", *SF_FERRY), _fact("e:b", *SF_CITY_HALL)])
    inserts = [
        sql
        for (op, sql, _a) in recorder
        if op == "execute" and "INSERT INTO entity_spatiotemporal" in sql
    ]
    assert len(inserts) == 2


async def test_postgis_query_radius_sql(pg):
    store, recorder, conn = pg
    conn.rows = [{"entity_uri": "e:ferry", "attrs": '{"label": "Ferry"}'}]
    res = await store.query_radius(
        TENANT, SF_FERRY[0], SF_FERRY[1], 5_000, as_of=_dt(2026)
    )
    assert res == [STQueryResult(entity_uri="e:ferry", attrs={"label": "Ferry"})]
    sql, args = next((s, a) for (op, s, a) in recorder if op == "fetch")
    assert "ST_DWithin(geom::geography" in sql
    assert "ST_SetSRID(ST_MakePoint($2, $3), 4326)::geography" in sql
    assert "$4)" in sql  # radius
    # as_of containment predicate.
    assert "$5::timestamptz <@ valid_time" in sql
    assert args == (TENANT, SF_FERRY[0], SF_FERRY[1], 5_000.0, _dt(2026))


async def test_postgis_query_bbox_sql(pg):
    store, recorder, conn = pg
    conn.rows = []
    await store.query_bbox(
        TENANT, -122.6, 37.6, -122.3, 37.9, time_window=(_dt(2025), _dt(2026))
    )
    sql, args = next((s, a) for (op, s, a) in recorder if op == "fetch")
    assert "ST_MakeEnvelope($2, $3, $4, $5, 4326)" in sql
    # time_window overlap predicate at $6/$7, both bounds cast to timestamptz.
    assert "valid_time && tstzrange($6::timestamptz, $7::timestamptz, '[)')" in sql
    assert args == (TENANT, -122.6, 37.6, -122.3, 37.9, _dt(2025), _dt(2026))


async def test_postgis_query_polygon_sql(pg):
    store, recorder, conn = pg
    conn.rows = []
    wkt = "POLYGON((-122.6 37.6, -122.3 37.6, -122.3 37.9, -122.6 37.9, -122.6 37.6))"
    await store.query_polygon(TENANT, wkt)
    sql, args = next((s, a) for (op, s, a) in recorder if op == "fetch")
    assert "ST_Within(geom, ST_GeomFromText($2, 4326))" in sql
    # No temporal predicate when neither given.
    assert "tstzrange" not in sql and "<@" not in sql
    assert args == (TENANT, wkt)


async def test_postgis_null_bounds_are_typed(pg):
    """Open-ended validity + open-ended windows must bind NULLs as ``::timestamptz``.

    asyncpg can't infer the type of a bare ``None`` bound, so a real Postgres
    raises "could not determine data type of parameter" without the cast. Assert
    the emitted SQL casts both range bounds so the NULL is typed, not bare.
    """
    store, recorder, conn = pg

    # upsert with fully open-ended validity (valid_from=None, valid_to=None).
    await store.upsert(_fact("e:openend", *SF_FERRY, valid_from=None, valid_to=None))
    insert_sql, insert_args = next(
        (sql, args)
        for (op, sql, args) in recorder
        if op == "execute" and "INSERT INTO entity_spatiotemporal" in sql
    )
    assert "tstzrange($4::timestamptz, $5::timestamptz, '[)')" in insert_sql
    # The bound params are genuinely None (the cast is what makes them safe).
    assert insert_args[3] is None and insert_args[4] is None

    # overlap query with an open lower bound: time_window=(None, something).
    conn.rows = []
    await store.query_bbox(
        TENANT, -122.6, 37.6, -122.3, 37.9, time_window=(None, _dt(2026))
    )
    fetch_sql, fetch_args = next(
        (s, a) for (op, s, a) in recorder if op == "fetch"
    )
    assert (
        "valid_time && tstzrange($6::timestamptz, $7::timestamptz, '[)')"
        in fetch_sql
    )
    # Lower bound binds as a typed NULL; upper bound is the given datetime.
    assert fetch_args[5] is None and fetch_args[6] == _dt(2026)


async def test_postgis_delete_and_clear_sql(pg):
    store, recorder, _conn = pg
    await store.delete("e:ferry", TENANT)
    await store.clear(TENANT)
    deletes = _sqls(recorder, "execute")
    assert any(
        "DELETE FROM entity_spatiotemporal WHERE tenant_id = $1 AND entity_uri = $2" in s
        for s in deletes
    )
    assert any(
        "DELETE FROM entity_spatiotemporal WHERE tenant_id = $1" in s
        and "entity_uri" not in s
        for s in deletes
    )


async def test_postgis_kg_predicate_sql(pg):
    """Passing kg_name appends an ``AND kg_name = $n`` after the spatial params,
    with the temporal predicate renumbered to follow it."""
    store, recorder, conn = pg
    conn.rows = []
    await store.query_radius(
        TENANT, SF_FERRY[0], SF_FERRY[1], 5_000, kg_name=KG, as_of=_dt(2026)
    )
    sql, args = next((s, a) for (op, s, a) in recorder if op == "fetch")
    # tenant $1, lon/lat $2/$3, radius $4, kg_name $5, as_of $6.
    assert "AND kg_name = $5" in sql
    assert "$6::timestamptz <@ valid_time" in sql
    assert args == (TENANT, SF_FERRY[0], SF_FERRY[1], 5_000.0, KG, _dt(2026))


async def test_postgis_clear_kg_scoped_sql(pg):
    store, recorder, _conn = pg
    await store.clear(TENANT, kg_name=KG)
    deletes = _sqls(recorder, "execute")
    assert any(
        "DELETE FROM entity_spatiotemporal WHERE tenant_id = $1 AND kg_name = $2" in s
        for s in deletes
    )


async def test_postgis_parse_attrs_variants(pg):
    """asyncpg may return jsonb as str, bytes, or dict — all decode to a dict."""
    store, recorder, conn = pg
    conn.rows = [
        {"entity_uri": "e:str", "attrs": '{"a": 1}'},
        {"entity_uri": "e:bytes", "attrs": b'{"b": 2}'},
        {"entity_uri": "e:dict", "attrs": {"c": 3}},
        {"entity_uri": "e:none", "attrs": None},
    ]
    res = await store.query_bbox(TENANT, -1, -1, 1, 1)
    by_uri = {r.entity_uri: r.attrs for r in res}
    assert by_uri == {
        "e:str": {"a": 1},
        "e:bytes": {"b": 2},
        "e:dict": {"c": 3},
        "e:none": {},
    }


# ---------------------------------------------------------------------------
# Live DB integration test (skips without OMNIX_DATABASE_URL)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OMNIX_DATABASE_URL"),
    reason="OMNIX_DATABASE_URL not set; needs a live PostGIS database",
)
async def test_postgis_roundtrip_live():
    store = PostGISSpatioTemporalIndex()
    tenant = "test-st-integration"
    try:
        await store.clear(tenant)
        await store.upsert(
            _fact("e:ferry", *SF_FERRY, tenant=tenant, valid_from=_dt(2025), valid_to=_dt(2027))
        )
        await store.upsert(_fact("e:times", *NYC_TIMES_SQ, tenant=tenant))
        res = await store.query_radius(tenant, *SF_FERRY, 5_000)
        assert _uris(res) == {"e:ferry"}
        res = await store.query_radius(tenant, *SF_FERRY, 5_000, as_of=_dt(2026))
        assert _uris(res) == {"e:ferry"}
        res = await store.query_radius(tenant, *SF_FERRY, 5_000, as_of=_dt(2020))
        assert _uris(res) == set()
    finally:
        await store.clear(tenant)
