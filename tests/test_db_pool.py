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


class SuspendingFakePool(FakePool):
    """A pool whose ``expire_connections`` SUSPENDS mid-body on an event.

    Real asyncpg's expire body never suspends today; this models a future
    asyncpg (or any awaitable-returning fake) whose expiry yields, to pin the
    hardening in ``_expire_if_hooks_changed``: the hook count is captured
    BEFORE the await, so a hook registered while the expiry is suspended is
    not silently marked synchronized.
    """

    def __init__(self, dsn, init):
        super().__init__(dsn, init)
        self.entered = asyncio.Event()  # set when expire_connections starts
        self.gate = asyncio.Event()     # test releases the suspended expiry

    async def expire_connections(self):
        self.entered.set()
        await self.gate.wait()
        self.expired += 1


async def test_hook_registered_mid_expire_still_triggers_later_expiry(monkeypatch, fresh):
    """The hook count is snapshotted BEFORE the awaited expiry (FIX: ONTA-173).

    If the count were recorded as ``len(_init_hooks)`` AFTER the await, a hook
    registered while ``expire_connections`` was suspended would be marked
    synchronized without any expiry having run for it — its codec would never
    reach already-pooled connections. Recording the pre-await snapshot keeps
    the next ``get_pg_pool`` call seeing a mismatch and expiring again.
    """
    import asyncpg

    async def create_suspending(*, dsn, init=None):
        return SuspendingFakePool(dsn, init)

    monkeypatch.setattr(asyncpg, "create_pool", create_suspending)
    pool = await get_pg_pool("postgres://u@h/db")

    async def hook_a(conn):  # pragma: no cover - never invoked here
        pass

    async def hook_b(conn):  # pragma: no cover - never invoked here
        pass

    register_pool_init(hook_a)  # pool synchronized at 0 hooks → mismatch
    task = asyncio.create_task(get_pg_pool("postgres://u@h/db"))
    await pool.entered.wait()   # the expiry for hook_a is suspended mid-body
    register_pool_init(hook_b)  # arrives DURING the in-flight expiry
    pool.gate.set()
    assert (await task) is pool
    assert pool.expired == 1
    # hook_b must still force one more expiry: the completed expire recorded
    # the PRE-await snapshot (1 hook), not the post-await len (2).
    await get_pg_pool("postgres://u@h/db")
    assert pool.expired == 2
    # ...and then it settles (no further churn once synchronized at 2).
    await get_pg_pool("postgres://u@h/db")
    assert pool.expired == 2


async def test_close_pg_pools_closes_and_forgets(fresh):
    pool = await get_pg_pool("postgres://u@h/db")
    await close_pg_pools()
    assert pool.closed
    # Next acquire builds a fresh pool rather than returning the closed one.
    again = await get_pg_pool("postgres://u@h/db")
    assert again is not pool
