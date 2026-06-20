"""PostGIS-backed :class:`SpatioTemporalIndex` over a generic Postgres DSN (COG-103).

Durable, shared-across-tasks adapter using ``asyncpg``. Vendor-neutral by
construction: the *only* configuration is a plain DSN (``settings.database_url`` or an
explicit ``dsn=`` arg). No cloud-provider ARNs, account IDs, or hostnames live here —
the Aurora/Neon connection + infra that *provides* the DSN stay proprietary.

Schema::

    CREATE TABLE entity_spatiotemporal (
        entity_uri text NOT NULL,
        tenant_id  text NOT NULL,
        geom       geometry(Geometry, 4326) NOT NULL,
        valid_time tstzrange NOT NULL,
        attrs      jsonb NOT NULL DEFAULT '{}'::jsonb,
        PRIMARY KEY (tenant_id, entity_uri, valid_time)
    );
    CREATE INDEX ... USING GIST (tenant_id, geom, valid_time);  -- needs btree_gist

The ``GIST(tenant_id, geom, valid_time)`` composite index serves the hot path —
spatial predicate + temporal predicate scoped to one tenant — in one index scan
(``btree_gist`` lets the scalar ``tenant_id`` participate in a GiST index). Writes are
idempotent upserts keyed on the ``(tenant_id, entity_uri, valid_time)`` PK.

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
        # asyncpg.Pool is created lazily; guard creation so concurrent callers
        # don't each build a pool.
        import asyncio

        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ setup
    async def _ensure_pool(self) -> Any:
        """Lazily create the asyncpg pool + table/index/extensions on first use."""
        if self._pool is not None:
            return self._pool
        async with self._lock:
            if self._pool is not None:
                return self._pool
            import asyncpg  # imported lazily so the dependency stays optional

            pool = await asyncpg.create_pool(dsn=self._dsn)
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
                await conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._TABLE} (
                        entity_uri text NOT NULL,
                        tenant_id  text NOT NULL,
                        geom       geometry(Geometry, 4326) NOT NULL,
                        valid_time tstzrange NOT NULL,
                        attrs      jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                        PRIMARY KEY (tenant_id, entity_uri, valid_time)
                    )
                    """
                )
                # Composite GiST over (tenant_id, geom, valid_time) — requires
                # btree_gist for the scalar tenant_id to live in a GiST index.
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS {self._INDEX} "
                    f"ON {self._TABLE} USING GIST (tenant_id, geom, valid_time)"
                )
            self._pool = pool
            return self._pool

    # ----------------------------------------------------------------- writes
    @staticmethod
    def _valid_time_sql() -> str:
        """SQL fragment building a tstzrange from $from/$to params (NULL = open)."""
        return "tstzrange($4, $5, '[)')"

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
        """Idempotent upsert on the (tenant_id, entity_uri, valid_time) PK.

        Geometry built via ``ST_SetSRID(ST_MakePoint($lon, $lat), 4326)``; ``attrs``
        re-denormalized on conflict so a replay refreshes display fields.
        """
        await conn.execute(
            f"""
            INSERT INTO {self._TABLE}
                (entity_uri, tenant_id, geom, valid_time, attrs)
            VALUES (
                $1, $2,
                ST_SetSRID(ST_MakePoint($6, $7), 4326),
                {self._valid_time_sql()},
                $3::jsonb
            )
            ON CONFLICT (tenant_id, entity_uri, valid_time) DO UPDATE SET
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
        )

    # ---------------------------------------------------------------- queries
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
            return (
                f" AND valid_time && tstzrange(${start_param}, ${start_param + 1}, '[)')",
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
        time_window: Optional[TimeWindow] = None,
        as_of: Optional[datetime] = None,
    ) -> list[STQueryResult]:
        # $1 tenant, $2 lon, $3 lat, $4 radius; temporal params start at $5.
        spatial = (
            "tenant_id = $1 AND ST_DWithin("
            "geom::geography, "
            "ST_SetSRID(ST_MakePoint($2, $3), 4326)::geography, "
            "$4)"
        )
        params: list[Any] = [tenant_id, lon, lat, radius_m]
        tpred, tparams = self._temporal_predicate(time_window, as_of, len(params) + 1)
        return await self._run_query(spatial + tpred, params + tparams)

    async def query_bbox(
        self,
        tenant_id: str,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
        *,
        time_window: Optional[TimeWindow] = None,
        as_of: Optional[datetime] = None,
    ) -> list[STQueryResult]:
        spatial = (
            "tenant_id = $1 AND ST_Within("
            "geom, ST_MakeEnvelope($2, $3, $4, $5, 4326))"
        )
        params: list[Any] = [tenant_id, min_lon, min_lat, max_lon, max_lat]
        tpred, tparams = self._temporal_predicate(time_window, as_of, len(params) + 1)
        return await self._run_query(spatial + tpred, params + tparams)

    async def query_polygon(
        self,
        tenant_id: str,
        wkt_polygon: str,
        *,
        time_window: Optional[TimeWindow] = None,
        as_of: Optional[datetime] = None,
    ) -> list[STQueryResult]:
        spatial = (
            "tenant_id = $1 AND ST_Within("
            "geom, ST_GeomFromText($2, 4326))"
        )
        params: list[Any] = [tenant_id, wkt_polygon]
        tpred, tparams = self._temporal_predicate(time_window, as_of, len(params) + 1)
        return await self._run_query(spatial + tpred, params + tparams)

    # ----------------------------------------------------------------- delete
    async def delete(self, entity_uri: str, tenant_id: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._TABLE} "
                f"WHERE tenant_id = $1 AND entity_uri = $2",
                tenant_id,
                entity_uri,
            )

    async def clear(self, tenant_id: str) -> None:
        pool = await self._ensure_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"DELETE FROM {self._TABLE} WHERE tenant_id = $1", tenant_id
            )


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
