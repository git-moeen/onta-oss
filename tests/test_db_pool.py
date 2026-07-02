"""Shared asyncpg pool (cograph_client/db/pool.py, ONTA-174).

All tests run against a fake ``asyncpg.create_pool`` — no real Postgres. What
must hold:

* one pool per DSN, however many stores/callers ask for it (the whole point);
* distinct DSNs get distinct pools;
* per-connection init hooks are applied to new connections, and registering a
  hook AFTER a pool exists expires its connections on the next ``get_pg_pool``
  call — properly awaited, because asyncpg's ``expire_connections`` is a
  coroutine whose body never runs unawaited (the ONTA-176 review finding: the
  old fire-and-forget call was a silent no-op on real pools);
* ``reset_pg_pools`` forgets pools + hooks (test isolation contract that
  ``tests/test_spatiotemporal.py``'s ``pg`` fixture relies on).
"""

import asyncio

import pytest

from cograph_client.db.pool import (
    close_pg_pools,
    get_pg_pool,
    register_pool_init,
    reset_pg_pools,
)


class FakePool:
    def __init__(self, dsn, init):
        self.dsn = dsn
        self.init = init
        self.expired = 0
        self.closed = False

    async def expire_connections(self):
        # Coroutine on purpose — mirrors real asyncpg, so a fire-and-forget
        # (unawaited) call in pool.py would leave ``expired`` at 0 and fail
        # the late-hook test below.
        self.expired += 1

    async def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """Fake asyncpg + a clean module state around every test."""
    created: list[FakePool] = []

    async def fake_create_pool(*, dsn, init=None):
        p = FakePool(dsn, init)
        created.append(p)
        return p

    import asyncpg

    monkeypatch.setattr(asyncpg, "create_pool", fake_create_pool)
    reset_pg_pools()
    yield created
    reset_pg_pools()


async def test_one_pool_per_dsn(fresh):
    a = await get_pg_pool("postgres://u@h/db")
    b = await get_pg_pool("postgres://u@h/db")
    assert a is b
    assert len(fresh) == 1


async def test_distinct_dsns_get_distinct_pools(fresh):
    a = await get_pg_pool("postgres://u@h/one")
    b = await get_pg_pool("postgres://u@h/two")
    assert a is not b
    assert len(fresh) == 2


async def test_concurrent_first_callers_build_one_pool(fresh):
    pools = await asyncio.gather(*(get_pg_pool("postgres://u@h/db") for _ in range(8)))
    assert len({id(p) for p in pools}) == 1
    assert len(fresh) == 1


async def test_init_hooks_run_on_new_connections(fresh):
    seen = []

    async def hook(conn):
        seen.append(conn)

    register_pool_init(hook)
    pool = await get_pg_pool("postgres://u@h/db")
    # The pool was created with this module's dispatcher as its init callback;
    # simulate asyncpg establishing a new connection.
    await pool.init("fake-conn")
    assert seen == ["fake-conn"]


async def test_late_hook_expires_existing_pools(fresh):
    pool = await get_pg_pool("postgres://u@h/db")
    assert pool.expired == 0

    async def hook(conn):  # pragma: no cover - never invoked here
        pass

    register_pool_init(hook)
    # Registration itself is sync and defers the (async) expiry to the next
    # get_pg_pool call — the only way consumers obtain a pool.
    assert pool.expired == 0
    again = await get_pg_pool("postgres://u@h/db")
    assert again is pool
    assert pool.expired == 1  # existing connections recycled → hook applies
    # Idempotent: no further hook registrations → no further expiry churn.
    await get_pg_pool("postgres://u@h/db")
    assert pool.expired == 1


async def test_close_pg_pools_closes_and_forgets(fresh):
    pool = await get_pg_pool("postgres://u@h/db")
    await close_pg_pools()
    assert pool.closed
    # Next acquire builds a fresh pool rather than returning the closed one.
    again = await get_pg_pool("postgres://u@h/db")
    assert again is not pool
