"""PostGIS-backed :class:`SpatioTemporalIndex` over a generic Postgres DSN (COG-103).

Durable, shared-across-tasks adapter using ``asyncpg``. Vendor-neutral by
construction: the *only* configuration is a plain DSN (``settings.database_url`` or an
explicit ``dsn=`` arg). No cloud-provider ARNs, account IDs, or hostnames live here —
the Aurora/Neon connection + infra that *provides* the DSN stay proprietary.

Schema::

    CREATE TABLE entity_spatiotemporal (
        entity_uri text NOT NULL,
        tenant_id  text NOT NULL,
        kg_name    text NOT NULL,
        geom       geometry(Geometry, 4326) NOT NULL,
        valid_time tstzrange NOT NULL,
        attrs      jsonb NOT NULL DEFAULT '{}'::jsonb,
        PRIMARY KEY (tenant_id, kg_name, entity_uri, valid_time)
    );
    CREATE INDEX ... USING GIST (tenant_id, kg_name, geom, valid_time);  -- btree_gist

The ``GIST(tenant_id, kg_name, geom, valid_time)`` composite index serves the hot
path — spatial predicate + temporal predicate scoped to one tenant (and optionally
one KG) — in one index scan (``btree_gist`` lets the scalar ``tenant_id`` /
``kg_name`` participate in a GiST index). Writes are idempotent upserts keyed on the
``(tenant_id, kg_name, entity_uri, valid_time)`` PK.

Pool + DDL are created lazily on first use, so importing this module / constructing
the store never touches the network. DDL is idempotent (``IF NOT EXISTS``).

Extensions: ``postgis`` + ``btree_gist`` must exist. We attempt
``CREATE EXTENSION IF NOT EXISTS`` but those may require superuser; per COG-102 the
infra layer bootstraps them, so we tolerate a permission failure and continue (the
table/index DDL then surfaces a clear error if the extension is genuinely absent).

**MobilityDB caveat:** this schema models *discrete* "located-during-range" facts —
a fixed geometry per ``valid_time``. For continuously moving objects / trajectories
use MobilityDB (``tgeompoint``) instead; see ``protocol.py`` module docstring.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence

import structlog

from cograph_client.config import settings
from cograph_client.spatiotemporal.protocol import (
    STQueryResult,
    SpatioTemporalFact,
    TimeWindow,
)

logger = structlog.stdlib.get_logger("cograph.spatiotemporal.postgis")


class PostGISSpatioTemporalIndex:
    """Durable :class:`SpatioTemporalIndex` backed by PostGIS via asyncpg."""

    _TABLE = "entity_spatiotemporal"
    _INDEX = "entity_spatiotemporal_gist"

    def __init__(self, dsn: Optional[str] = None) -> None:
        self._dsn = dsn if dsn is not None else settings.database_url
        self._pool: Any = None
        # DDL is applied lazily once; guard so concurrent first callers don't
        # each run it. Pool creation itself is delegated to the process-wide
        # shared pool (cograph_client.db.pool, ONTA-174) — one pool per DSN
        # across all durable stores, not one per store.
        import asyncio

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ setup
    async def _ensure_pool(self) -> Any:
        """Acquire the shared pool + apply table/index/extension DDL on first use."""
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is not None:
                return self._pool
            from cograph_client.db.pool import get_pg_pool

            pool = await get_pg_pool(self._dsn)
            async with pool.acquire() as conn:
                # Extensions may need superuser; infra (COG-102) bootstraps them.
                # Tolerate a permission error and let the DDL below surface any
                # genuine "extension missing" failure with a clear message.
                for ext in ("postgis", "btree_gist"):
                    try:
                        await conn.execute(
                            f"CREATE EXTENSION IF NOT EXISTS {ext}"
                        )
                    except Exception as exc:  # noqa: BLE001 - best-effort bootstrap
                        logger.warning(
                            "spatiotemporal_extension_bootstrap_skipped",
                            extension=ext,
                            error=str(exc),
                        )
                # Schema note (ONTA-157): the ``kg_name`` column + 4-col PK were
                # added on top of the original COG-103 3-col schema. This is a bare
                # ``CREATE TABLE IF NOT EXISTS`` with NO migration — load-bearing
                # assumption: the table has never been materialized in prod (the
                # index had zero callers before this change and the pool/table are
                # created lazily on first use, so nothing ever created it). If an
                # OLD-schema ``entity_spatiotemporal`` table somehow pre-exists, the
                # IF NOT EXISTS is a no-op and inserts will fail on the missing
                # ``kg_name`` / mismatched ON CONFLICT target — drop that empty
                # table so this DDL recreates it. (See the PR deploy note.)
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._TABLE} (
                        entity_uri text NOT NULL,
                        tenant_id  text NOT NULL,
                        kg_name    text NOT NULL,
                        geom       geometry(Geometry, 4326) NOT NULL,
                        valid_time tstzrange NOT NULL,
                        attrs      jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                        PRIMARY KEY (tenant_id, kg_name, entity_uri, valid_time)
                    )
                    """
                )
                # Composite GiST over (tenant_id, kg_name, geom, valid_time) —
                # requires btree_gist for the scalar tenant_id/kg_name to live in a
                # GiST index. Serves "this KG, near here, valid then" in one scan.
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._INDEX} "
                    f"ON {self._TABLE} USING GIST "
                    f"(tenant_id, kg_name, geom, valid_time)"
                )
            self._pool = pool
            return self._pool

    # ----------------------------------------------------------------- writes
    @staticmethod
    def _valid_time_sql() -> str:
        """SQL fragment building a tstzrange from $from/$to params (NULL = open).

        Both bounds are cast to ``timestamptz`` so an open-ended (``None``) bound
        binds as a *typed* NULL — without the cast asyncpg sends an untyped NULL
        and Postgres raises "could not determine data type of parameter".
        """
        return "tstzrange($4::timestamptz, $5::timestamptz, '[)')"

    async def upsert(self, fact: SpatioTemporalFact) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await self._upsert_conn(conn, fact)

    async def upsert_many(self, facts: Sequence[SpatioTemporalFact]) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for fact in facts:
                    await self._upsert_conn(conn, fact)

    async def _upsert_conn(self, conn: Any, fact: SpatioTemporalFact) -> None:
        """Idempotent upsert on the (tenant_id, kg_name, entity_uri, valid_time) PK.

        Geometry built via ``ST_SetSRID(ST_MakePoint($lon, $lat), 4326)``; ``attrs``
        re-denormalized on conflict so a replay refreshes display fields.
        """
        await conn.execute(
            f"""
            INSERT INTO {self._TABLE}
                (entity_uri, tenant_id, kg_name, geom, valid_time, attrs)
            VALUES (
                $1, $2, $8,
                ST_SetSRID(ST_MakePoint($6, $7), 4326),
                {self._valid_time_sql()},
                $3::jsonb
            )
            ON CONFLICT (tenant_id, kg_name, entity_uri, valid_time) DO UPDATE SET
                geom  = EXCLUDED.geom,
                attrs = EXCLUDED.attrs
            """,
            fact.entity_uri,
            fact.tenant_id,
            _attrs_json(fact.attrs),
            fact.valid_from,
            fact.valid_to,
            fact.lon,
            fact.lat,
            fact.kg_name,
        )

    # ---------------------------------------------------------------- queries
    @staticmethod
    def _kg_predicate(
        kg_name: Optional[str], start_param: int
    ) -> tuple[str, list[Any]]:
        """Optional per-KG narrowing (``AND kg_name = $n``).

        ``None`` → no predicate (search every KG in the tenant). Returns
        ("", []) so callers can concatenate unconditionally.
        """
        if kg_name is not None:
            return f" AND kg_name = ${start_param}", [kg_name]
        return "", []

    @staticmethod
    def _temporal_predicate(
        time_window: Optional[TimeWindow],
        as_of: Optional[datetime],
        start_param: int,
    ) -> tuple[str, list[Any]]:
        """Build the temporal SQL predicate + its bind params.

        ``as_of`` (containment, ``$n <@ valid_time``) takes precedence over
        ``time_window`` (overlap, ``valid_time && tstzrange($a, $b)``); neither →
        no temporal filter. Returns ("", []) when there is no predicate.
        """
        if as_of is not None:
            return f" AND ${start_param}::timestamptz <@ valid_time", [as_of]
        if time_window is not None:
            w_lo, w_hi = time_window
            # Cast both bounds to timestamptz so an open-ended (None) window bound
            # binds as a typed NULL — mirrors _valid_time_sql / the as_of cast and
            # avoids "could not determine data type of parameter" on real Postgres.
            return (
                f" AND valid_time && tstzrange("
                f"${start_param}::timestamptz, ${start_param + 1}::timestamptz, '[)')",
                [w_lo, w_hi],
            )
        return "", []

    async def _run_query(
        self, spatial_sql: str, params: list[Any]
    ) -> list[STQueryResult]:
        pool = await self._ensure_pool()
        sql = (
            f"SELECT entity_uri, attrs FROM {self._TABLE} "
            f"WHERE {spatial_sql}"
        )
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [
            STQueryResult(entity_uri=r["entity_uri"], attrs=_parse_attrs(r["attrs"]))
            for r in rows
        ]

    async def query_radius(
        self,
        tenant_id: str,
        lon: float,
        lat: float,
        radius_m: float,
        *,
        kg_name: Optional[str] = None,
        time_window: Optional[TimeWindow] = None,
        as_of: Optional[datetime] = None,
    ) -> list[STQueryResult]:
        # $1 tenant, $2 lon, $3 lat, $4 radius; kg + temporal params follow.
        spatial = (
            "tenant_id = $1 AND ST_DWithin("
            "geom::geography, "
            "ST_SetSRID(ST_MakePoint($2, $3), 4326)::geography, "
            "$4)"
        )
        params: list[Any] = [tenant_id, lon, lat, radius_m]
        kpred, kparams = self._kg_predicate(kg_name, len(params) + 1)
        params += kparams
        tpred, tparams = self._temporal_predicate(time_window, as_of, len(params) + 1)
        return await self._run_query(spatial + kpred + tpred, params + tparams)

    async def query_bbox(
        self,
        tenant_id: str,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        *,
        kg_name: Optional[str] = None,
        time_window: Optional[TimeWindow] = None,
        as_of: Optional[datetime] = None,
    ) -> list[STQueryResult]:
        spatial = (
            "tenant_id = $1 AND ST_Within("
            "geom, ST_MakeEnvelope($2, $3, $4, $5, 4326))"
        )
        params: list[Any] = [tenant_id, min_lon, min_lat, max_lon, max_lat]
        kpred, kparams = self._kg_predicate(kg_name, len(params) + 1)
        params += kparams
        tpred, tparams = self._temporal_predicate(time_window, as_of, len(params) + 1)
        return await self._run_query(spatial + kpred + tpred, params + tparams)

    async def query_polygon(
        self,
        tenant_id: str,
        wkt_polygon: str,
        *,
        kg_name: Optional[str] = None,
        time_window: Optional[TimeWindow] = None,
        as_of: Optional[datetime] = None,
    ) -> list[STQueryResult]:
        spatial = (
            "tenant_id = $1 AND ST_Within("
            "geom, ST_GeomFromText($2, 4326))"
        )
        params: list[Any] = [tenant_id, wkt_polygon]
        kpred, kparams = self._kg_predicate(kg_name, len(params) + 1)
        params += kparams
        tpred, tparams = self._temporal_predicate(time_window, as_of, len(params) + 1)
        return await self._run_query(spatial + kpred + tpred, params + tparams)

    # ----------------------------------------------------------------- delete
    async def delete(
        self, entity_uri: str, tenant_id: str, *, kg_name: Optional[str] = None
    ) -> None:
        pool = await self._ensure_pool()
        sql = f"DELETE FROM {self._TABLE} WHERE tenant_id = $1 AND entity_uri = $2"
        params: list[Any] = [tenant_id, entity_uri]
        if kg_name is not None:
            params.append(kg_name)
            sql += f" AND kg_name = ${len(params)}"
        async with pool.acquire() as conn:
            await conn.execute(sql, *params)

    async def rekey(
        self,
        old_uri: str,
        new_uri: str,
        tenant_id: str,
        *,
        kg_name: Optional[str] = None,
    ) -> None:
        pool = await self._ensure_pool()
        # Move old_uri's rows to new_uri, then drop the old rows — both in ONE
        # transaction. ON CONFLICT DO NOTHING makes new_uri (the ER-merge winner)
        # keep precedence when it already has a row at the same
        # (tenant, kg, valid_time): the loser's row is dropped rather than
        # clobbering the winner's geometry/attrs.
        kg_filter = ""
        move_params: list[Any] = [tenant_id, new_uri, old_uri]
        del_params: list[Any] = [tenant_id, old_uri]
        if kg_name is not None:
            move_params.append(kg_name)
            del_params.append(kg_name)
            kg_filter_move = f" AND kg_name = ${len(move_params)}"
            kg_filter_del = f" AND kg_name = ${len(del_params)}"
        else:
            kg_filter_move = ""
            kg_filter_del = ""
        move_sql = (
            f"INSERT INTO {self._TABLE} "
            f"(entity_uri, tenant_id, kg_name, geom, valid_time, attrs)\n"
            f"SELECT $2, tenant_id, kg_name, geom, valid_time, attrs "
            f"FROM {self._TABLE}\n"
            f"WHERE tenant_id = $1 AND entity_uri = $3{kg_filter_move}\n"
            f"ON CONFLICT (tenant_id, kg_name, entity_uri, valid_time) DO NOTHING"
        )
        del_sql = (
            f"DELETE FROM {self._TABLE} "
            f"WHERE tenant_id = $1 AND entity_uri = $2{kg_filter_del}"
        )
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(move_sql, *move_params)
                await conn.execute(del_sql, *del_params)

    async def clear(self, tenant_id: str, *, kg_name: Optional[str] = None) -> None:
        pool = await self._ensure_pool()
        sql = f"DELETE FROM {self._TABLE} WHERE tenant_id = $1"
        params: list[Any] = [tenant_id]
        if kg_name is not None:
            params.append(kg_name)
            sql += f" AND kg_name = ${len(params)}"
        async with pool.acquire() as conn:
            await conn.execute(sql, *params)


def _attrs_json(attrs: dict[str, Any]) -> str:
    """Serialize attrs to a JSON string for the ``$::jsonb`` bind."""
    import json

    return json.dumps(attrs or {})


def _parse_attrs(value: Any) -> dict[str, Any]:
    """asyncpg may hand jsonb back as str/bytes (no codec) or a decoded dict."""
    import json

    if value is None:
        return {}
    if isinstance(value, (bytes, bytearray)):
        value = value.decode()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return {}
    if isinstance(value, dict):
        return value
    return {}
