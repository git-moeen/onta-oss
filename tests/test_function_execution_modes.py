"""Per-function execution mode tests (ADR 0002 §6, COG-41).

Covers cograph_client/resolver/functions.py:

  - defaults by cost class: sparql -> query_time, lambda -> cached; explicit
    fn['mode'] always wins; unknown kind is a definition error
  - query_time: computed on every run, $entity substituted, nothing stored
  - cached: first run computes + stores the DERIVED-tagged triple pair
    (value + computed-at), reuses within TTL, recomputes after expiry,
    no-TTL values stay fresh until invalidated
  - invalidate: deletes exactly the derived triples whose invalidate_on
    intersects the changed attrs
  - lambda seam: NotImplementedError without a runner; a supplied
    lambda_runner (the premium plug point) flows through the cached path
  - backward-compat regression: 'functions' rides the COG-39 strategy
    resolver as a bundle entry without disturbing 'er' resolution, and the
    new module coexists with the legacy cograph_client.functions package

All mocked — no live Neptune, no LLM, no network. The executor's clock is
injected; no env is touched.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest

from cograph_client.graph.client import NeptuneClient
from cograph_client.resolver.functions import (
    DERIVED_NS,
    FunctionExecutor,
    computed_at_predicate,
    default_mode,
    delete_derived_update,
    derived_predicate,
    derived_value_query,
    invalidate,
    store_derived_update,
)

GRAPH = "https://cograph.tech/graphs/test-tenant/kg/test"
ENTITY = "https://cograph.tech/entities/Guest/g1"
T0 = datetime(2026, 6, 9, 12, 0, 0, tzinfo=timezone.utc)

AGE_FN = {
    "name": "age",
    "kind": "sparql",
    "query": "SELECT ?age WHERE { $entity <https://cograph.tech/attrs/Guest/age> ?age }",
}


class Clock:
    """Injectable, advanceable now() for the executor."""

    def __init__(self, t: datetime):
        self.t = t

    def __call__(self) -> datetime:
        return self.t


def _result(variables: list[str], rows: list[dict[str, str]]) -> dict:
    return {
        "head": {"vars": variables},
        "results": {"bindings": [
            {k: {"type": "literal", "value": v} for k, v in row.items()} for row in rows
        ]},
    }


def _stored(value: str, computed_at: str) -> dict:
    return _result(
        ["value", "computed_at"], [{"value": value, "computed_at": computed_at}],
    )


@pytest.fixture
def mock_neptune():
    client = AsyncMock(spec=NeptuneClient)
    client.health.return_value = True
    client.query.return_value = {"head": {"vars": []}, "results": {"bindings": []}}
    client.update.return_value = None
    return client


# ---------------------------------------------------------------------------
# Defaults by cost class
# ---------------------------------------------------------------------------


class TestDefaultMode:
    def test_sparql_defaults_to_query_time(self):
        assert default_mode("sparql") == "query_time"

    def test_lambda_defaults_to_cached(self):
        assert default_mode("lambda") == "cached"

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="unknown function kind"):
            default_mode("cron")

    @pytest.mark.asyncio
    async def test_explicit_mode_beats_default(self, mock_neptune):
        """A sparql fn declared mode='cached' goes through the stored path —
        the cost-class default is only a fallback."""
        fn = {**AGE_FN, "mode": "cached"}
        mock_neptune.query.side_effect = [
            _result([], []),               # no stored value
            _result(["age"], [{"age": "39"}]),
        ]
        value = await FunctionExecutor(now=Clock(T0)).run(mock_neptune, GRAPH, ENTITY, fn)
        assert value == "39"
        # One ATOMIC update (COG-46): stale-pair delete + fresh insert together.
        assert mock_neptune.update.call_count == 1


# ---------------------------------------------------------------------------
# query_time — computed on read, always fresh, never stored
# ---------------------------------------------------------------------------


class TestQueryTime:
    @pytest.mark.asyncio
    async def test_executes_per_call_and_substitutes_entity(self, mock_neptune):
        mock_neptune.query.return_value = _result(["age"], [{"age": "39"}])
        executor = FunctionExecutor(now=Clock(T0))

        assert await executor.run(mock_neptune, GRAPH, ENTITY, AGE_FN) == "39"
        assert await executor.run(mock_neptune, GRAPH, ENTITY, AGE_FN) == "39"

        assert mock_neptune.query.call_count == 2  # one execution per run
        sparql = mock_neptune.query.call_args.args[0]
        assert f"<{ENTITY}>" in sparql      # $entity -> angle-bracketed IRI
        assert "$entity" not in sparql
        mock_neptune.update.assert_not_called()  # nothing is ever stored

    @pytest.mark.asyncio
    async def test_no_rows_returns_none(self, mock_neptune):
        assert await FunctionExecutor().run(mock_neptune, GRAPH, ENTITY, AGE_FN) is None

    @pytest.mark.asyncio
    async def test_sparql_without_query_raises(self, mock_neptune):
        with pytest.raises(ValueError, match="requires 'query'"):
            await FunctionExecutor().run(
                mock_neptune, GRAPH, ENTITY, {"name": "age", "kind": "sparql"},
            )


# ---------------------------------------------------------------------------
# cached — stored as tagged derived triples, TTL-governed
# ---------------------------------------------------------------------------


class TestCached:
    @pytest.mark.asyncio
    async def test_first_run_computes_and_stores_tagged_triples(self, mock_neptune):
        fn = {**AGE_FN, "mode": "cached", "ttl_seconds": 3600}
        mock_neptune.query.side_effect = [
            _result([], []),               # no stored value yet
            _result(["age"], [{"age": "39"}]),
        ]
        value = await FunctionExecutor(now=Clock(T0)).run(mock_neptune, GRAPH, ENTITY, fn)
        assert value == "39"

        updates = [c.args[0] for c in mock_neptune.update.call_args_list]
        assert len(updates) == 1  # one atomic delete+insert update (COG-46)
        insert = updates[-1]
        # Value lands under the distinct derived namespace (ADR 0001 rule 6) …
        assert f"<{derived_predicate('age')}>" in insert
        assert derived_predicate("age").startswith(DERIVED_NS)
        # … with a computed-at timestamp triple from the injected clock.
        assert f"<{computed_at_predicate('age')}>" in insert
        assert T0.isoformat() in insert
        assert f"GRAPH <{GRAPH}>" in insert

    @pytest.mark.asyncio
    async def test_reuses_stored_value_within_ttl(self, mock_neptune):
        fn = {**AGE_FN, "mode": "cached", "ttl_seconds": 3600}
        mock_neptune.query.return_value = _stored("39", T0.isoformat())
        clock = Clock(T0 + timedelta(seconds=600))  # well inside the TTL

        value = await FunctionExecutor(now=clock).run(mock_neptune, GRAPH, ENTITY, fn)
        assert value == "39"
        assert mock_neptune.query.call_count == 1   # stored-value read only
        mock_neptune.update.assert_not_called()     # no recompute, no rewrite

    @pytest.mark.asyncio
    async def test_recomputes_after_ttl_expiry(self, mock_neptune):
        fn = {**AGE_FN, "mode": "cached", "ttl_seconds": 3600}
        mock_neptune.query.side_effect = [
            _stored("39", T0.isoformat()),          # stale
            _result(["age"], [{"age": "40"}]),      # recompute
        ]
        clock = Clock(T0 + timedelta(seconds=7200))

        value = await FunctionExecutor(now=clock).run(mock_neptune, GRAPH, ENTITY, fn)
        assert value == "40"
        # Exactly ONE update call for the recompute (COG-46): the stale-pair
        # delete and the fresh insert ride a single atomic update request.
        assert mock_neptune.update.call_count == 1
        update = mock_neptune.update.call_args.args[0]
        assert update.startswith("DELETE WHERE")            # stale pair replaced …
        assert "INSERT DATA" in update                      # … in the SAME request
        assert clock.t.isoformat() in update                # fresh computed-at

    def test_store_update_is_atomic_delete_then_insert(self):
        """The replace update is one string: stale-pair DELETE WHEREs first,
        the fresh-pair INSERT DATA last, ';'-joined — a concurrent reader
        never sees the no-value gap of the old two-call pattern."""
        update = store_derived_update(GRAPH, ENTITY, "age", "40", T0.isoformat())
        assert update.count(";") >= 2  # two DELETE WHEREs + the INSERT joined
        delete_part, insert_part = update.rsplit(";", 1)
        assert delete_part.startswith("DELETE WHERE")
        assert f"<{derived_predicate('age')}>" in delete_part
        assert f"<{computed_at_predicate('age')}>" in delete_part
        assert insert_part.strip().startswith("INSERT DATA")
        assert f"<{derived_predicate('age')}>" in insert_part
        assert T0.isoformat() in insert_part
        assert f"GRAPH <{GRAPH}>" in insert_part

    @pytest.mark.asyncio
    async def test_no_ttl_means_fresh_until_invalidated(self, mock_neptune):
        fn = {**AGE_FN, "mode": "cached", "invalidate_on": ["birthdate"]}
        ancient = (T0 - timedelta(days=365)).isoformat()
        mock_neptune.query.return_value = _stored("39", ancient)

        value = await FunctionExecutor(now=Clock(T0)).run(mock_neptune, GRAPH, ENTITY, fn)
        assert value == "39"
        mock_neptune.update.assert_not_called()

    @pytest.mark.asyncio
    async def test_unparseable_timestamp_treated_as_stale(self, mock_neptune):
        fn = {**AGE_FN, "mode": "cached", "ttl_seconds": 3600}
        mock_neptune.query.side_effect = [
            _stored("39", "not-a-timestamp"),
            _result(["age"], [{"age": "40"}]),
        ]
        value = await FunctionExecutor(now=Clock(T0)).run(mock_neptune, GRAPH, ENTITY, fn)
        assert value == "40"


# ---------------------------------------------------------------------------
# invalidate — input-attr triggers delete the right derived triples
# ---------------------------------------------------------------------------


class TestInvalidate:
    @pytest.mark.asyncio
    async def test_deletes_only_intersecting_functions(self, mock_neptune):
        fns = [
            {"name": "age", "kind": "sparql", "mode": "cached",
             "invalidate_on": ["birthdate"]},
            {"name": "ltv", "kind": "lambda", "invalidate_on": ["bookings", "spend"]},
            {"name": "score", "kind": "lambda", "invalidate_on": ["reviews"]},
        ]
        invalidated = await invalidate(
            mock_neptune, GRAPH, ENTITY, ["birthdate", "spend"], fns,
        )
        assert invalidated == ["age", "ltv"]

        updates = [c.args[0] for c in mock_neptune.update.call_args_list]
        assert len(updates) == 2
        assert f"<{derived_predicate('age')}>" in updates[0]
        assert f"<{computed_at_predicate('age')}>" in updates[0]
        assert f"<{derived_predicate('ltv')}>" in updates[1]
        # 'score' (no intersection) is untouched.
        assert not any(derived_predicate("score") in u for u in updates)

    @pytest.mark.asyncio
    async def test_no_intersection_is_a_noop(self, mock_neptune):
        fns = [{"name": "age", "kind": "sparql", "invalidate_on": ["birthdate"]}]
        assert await invalidate(mock_neptune, GRAPH, ENTITY, ["email"], fns) == []
        mock_neptune.update.assert_not_called()

    def test_delete_update_targets_both_triples(self):
        update = delete_derived_update(GRAPH, ENTITY, "age")
        assert f"<{derived_predicate('age')}>" in update
        assert f"<{computed_at_predicate('age')}>" in update
        assert f"GRAPH <{GRAPH}>" in update


# ---------------------------------------------------------------------------
# lambda seam — OSS ships the plug point only
# ---------------------------------------------------------------------------


class TestLambdaSeam:
    @pytest.mark.asyncio
    async def test_raises_without_runner(self, mock_neptune):
        fn = {"name": "ltv", "kind": "lambda"}
        with pytest.raises(NotImplementedError, match="no OSS executor"):
            await FunctionExecutor(now=Clock(T0)).run(mock_neptune, GRAPH, ENTITY, fn)

    @pytest.mark.asyncio
    async def test_supplied_runner_flows_through_cached_path(self, mock_neptune):
        """A lambda_runner (premium plug point) computes the value; the cached
        default then stores it as derived triples like any other function."""
        runner = AsyncMock(return_value="9100")
        fn = {"name": "ltv", "kind": "lambda", "ttl_seconds": 86400}

        executor = FunctionExecutor(now=Clock(T0), lambda_runner=runner)
        assert await executor.run(mock_neptune, GRAPH, ENTITY, fn) == "9100"

        runner.assert_awaited_once_with(ENTITY, fn)
        insert = mock_neptune.update.call_args.args[0]
        assert f"<{derived_predicate('ltv')}>" in insert


# ---------------------------------------------------------------------------
# Backward compat — functions are a bundle entry, nothing else moved
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_functions_entry_rides_strategy_resolver(self):
        """'functions' resolves through COG-39's chain walk like any bundle
        entry — HotelGuest inherits Guest's function defs."""
        from cograph_client.resolver.strategy import resolve_entry

        registry = {"Guest": {"functions": [AGE_FN]}}
        parent_of = {"HotelGuest": "Guest", "Guest": "Person"}
        assert resolve_entry("HotelGuest", "functions", parent_of, [registry]) == [AGE_FN]

    def test_er_resolution_unchanged(self):
        """Regression: the ER chain walk over the real defaults still returns
        the identical config objects — the functions machinery is additive."""
        from cograph_client.resolver.er.types import (
            DEFAULT_GUEST_CONFIG,
            config_for_with_hierarchy,
        )

        cfg = config_for_with_hierarchy("HotelGuest", {"HotelGuest": "Guest"})
        assert cfg is DEFAULT_GUEST_CONFIG

    def test_new_module_coexists_with_legacy_functions_package(self):
        """cograph_client.resolver.functions must not shadow or break the
        pre-existing cograph_client.functions package."""
        import cograph_client.functions.registry as legacy_registry
        import cograph_client.resolver.functions as new_module

        assert hasattr(legacy_registry, "get_functions_for_entity")
        assert hasattr(new_module, "FunctionExecutor")

    def test_derived_query_reads_only_the_derived_namespace(self):
        """Derived reads never touch asserted-attribute predicates."""
        query = derived_value_query(GRAPH, ENTITY, "age")
        assert DERIVED_NS in query
        assert "attrs" not in query
