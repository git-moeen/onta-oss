"""Per-function execution mode (ADR 0002 §6).

Deterministic computed attributes are declared as the 'functions' entry of a
type's strategy bundle (resolver/strategy.py). Each function def is a plain
dict — same convention as bundle entries::

    {
        "name": "age",                      # derived attribute name
        "kind": "sparql" | "lambda",        # cost class
        "mode": "query_time" | "cached",    # optional; default by cost class
        "query": "...$entity...",           # kind='sparql': SPARQL template
        "ttl_seconds": 86400,               # cached: freshness window
        "invalidate_on": ["birthdate"],     # cached: recompute triggers
    }

Execution modes (ADR 0002 §6 — freshness is the default, caching is the
declared exception):

- ``query_time`` — computed on read, always fresh. Default for kind='sparql'
  (cheap local computation).
- ``cached`` — computed once and stored, recomputed when the TTL lapses or an
  ``invalidate_on`` attribute changes. Default for kind='lambda' (external,
  expensive, or metered).

Cached outputs are DERIVED data per ADR 0001 rule 6: stored under a distinct
predicate namespace (``https://cograph.tech/derived/<name>``) with a
per-function computed-at timestamp triple, so they are regenerable, trivially
identifiable, and never confused with asserted facts.

kind='lambda' is a plugin seam only (register_adapter style): the executor
takes an optional ``lambda_runner`` callable; without one, lambda functions
raise NotImplementedError — the premium tree supplies the runner.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from string import Template

from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import insert_triples

DERIVED_NS = "https://cograph.tech/derived/"

_XSD = "http://www.w3.org/2001/XMLSchema"

# Premium plug point: async (entity_uri, fn) -> computed scalar.
LambdaRunner = Callable[[str, dict], Awaitable[str]]


def default_mode(kind: str) -> str:
    """Default execution mode by cost class (explicit fn['mode'] always wins).

    'sparql' (cheap, local) -> 'query_time'; 'lambda' (external/metered) ->
    'cached'. Unknown kinds are a definition error, not a runtime fallback.
    """
    if kind == "sparql":
        return "query_time"
    if kind == "lambda":
        return "cached"
    raise ValueError(f"unknown function kind: {kind!r} (expected 'sparql' or 'lambda')")


def derived_predicate(name: str) -> str:
    """Predicate a cached function value is stored under (ADR 0001 rule 6 tag)."""
    return f"{DERIVED_NS}{name}"


def computed_at_predicate(name: str) -> str:
    """Per-function computed-at timestamp predicate (pairs with derived_predicate)."""
    return f"{DERIVED_NS}{name}/computedAt"


def derived_value_query(graph_uri: str, entity_uri: str, name: str) -> str:
    """SELECT the stored derived value + its computed-at timestamp, if any."""
    return (
        f"SELECT ?value ?computed_at FROM <{graph_uri}>\n"
        f"WHERE {{\n"
        f"  <{entity_uri}> <{derived_predicate(name)}> ?value .\n"
        f"  OPTIONAL {{ <{entity_uri}> <{computed_at_predicate(name)}> ?computed_at }}\n"
        f"}}\nLIMIT 1"
    )


def delete_derived_update(graph_uri: str, entity_uri: str, name: str) -> str:
    """DELETE one function's stored derived value + timestamp for an entity."""
    return (
        f"DELETE WHERE {{ GRAPH <{graph_uri}> "
        f"{{ <{entity_uri}> <{derived_predicate(name)}> ?v }} }};\n"
        f"DELETE WHERE {{ GRAPH <{graph_uri}> "
        f"{{ <{entity_uri}> <{computed_at_predicate(name)}> ?t }} }}"
    )


def store_derived_update(
    graph_uri: str, entity_uri: str, name: str, value: str, computed_at: str,
) -> str:
    """One ATOMIC SPARQL update replacing a stored derived value (COG-46).

    The stale-pair DELETE WHERE statements and the fresh-pair INSERT DATA are
    joined with ';' into a single update request, so the store processes them
    as one transaction — a concurrent reader can never observe the gap where
    the old value is gone and the new one is not yet written (the race the
    previous two-call delete-then-insert pattern had).
    """
    insert = insert_triples(graph_uri, [
        (entity_uri, derived_predicate(name), value),
        (entity_uri, computed_at_predicate(name), f"{computed_at}^^{_XSD}#dateTime"),
    ])
    return f"{delete_derived_update(graph_uri, entity_uri, name)};\n{insert}"


def _parse_timestamp(value: str) -> datetime | None:
    """Parse a stored computed-at value; None (=> stale) if unparseable."""
    try:
        ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


class FunctionExecutor:
    """Executes one function def against one entity, honoring its mode.

    ``now`` is injectable for tests (defaults to datetime.now(timezone.utc));
    ``lambda_runner`` is the premium seam for kind='lambda' — see module
    docstring.
    """

    def __init__(
        self,
        now: Callable[[], datetime] | None = None,
        lambda_runner: LambdaRunner | None = None,
    ):
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._lambda_runner = lambda_runner

    async def run(self, neptune, graph_uri: str, entity_uri: str, fn: dict) -> str | None:
        """Run a function for an entity, returning the scalar result.

        query_time: compute and return — nothing is stored. cached: return the
        stored derived value if fresh (within ttl_seconds of its computed-at;
        no ttl_seconds => fresh until invalidated), else recompute, store the
        tagged derived + computed-at triples, and return the new value.
        Returns None when a sparql computation yields no rows.
        """
        mode = fn.get("mode") or default_mode(fn["kind"])
        if mode == "query_time":
            return await self._compute(neptune, graph_uri, entity_uri, fn)

        stored = await self._fetch_stored(neptune, graph_uri, entity_uri, fn["name"])
        if stored is not None:
            value, computed_at = stored
            if self._is_fresh(computed_at, fn.get("ttl_seconds")):
                return value

        value = await self._compute(neptune, graph_uri, entity_uri, fn)
        if value is not None:
            await self._store(neptune, graph_uri, entity_uri, fn["name"], value)
        return value

    async def _compute(self, neptune, graph_uri: str, entity_uri: str, fn: dict) -> str | None:
        if fn["kind"] == "sparql":
            query = fn.get("query")
            if not query:
                raise ValueError(f"function {fn.get('name')!r}: kind='sparql' requires 'query'")
            sparql = Template(query).safe_substitute(entity=f"<{entity_uri}>")
            raw = await neptune.query(sparql)
            variables, bindings = parse_sparql_results(raw)
            if not bindings:
                return None
            row = bindings[0]
            return next((row[v] for v in variables if v in row), None)
        if fn["kind"] == "lambda":
            if self._lambda_runner is None:
                raise NotImplementedError(
                    f"function {fn.get('name')!r}: kind='lambda' has no OSS executor — "
                    f"pass a lambda_runner to FunctionExecutor (premium supplies one "
                    f"via the plugin protocol)"
                )
            return await self._lambda_runner(entity_uri, fn)
        raise ValueError(f"unknown function kind: {fn['kind']!r}")

    async def _fetch_stored(
        self, neptune, graph_uri: str, entity_uri: str, name: str,
    ) -> tuple[str, str] | None:
        raw = await neptune.query(derived_value_query(graph_uri, entity_uri, name))
        _, bindings = parse_sparql_results(raw)
        if not bindings or "value" not in bindings[0]:
            return None
        return bindings[0]["value"], bindings[0].get("computed_at", "")

    def _is_fresh(self, computed_at: str, ttl_seconds: int | None) -> bool:
        if ttl_seconds is None:
            return True  # no TTL: fresh until invalidate() removes it
        ts = _parse_timestamp(computed_at)
        if ts is None:
            return False  # missing/unparseable timestamp: treat as stale
        return (self._now() - ts).total_seconds() < ttl_seconds

    async def _store(self, neptune, graph_uri: str, entity_uri: str, name: str, value: str) -> None:
        # Replace, don't accumulate — atomically: the stale-pair DELETE and
        # the fresh-pair INSERT travel in ONE update request (COG-46), so a
        # concurrent reader never sees a missing derived value mid-replace.
        await neptune.update(
            store_derived_update(graph_uri, entity_uri, name, value, self._now().isoformat())
        )


async def invalidate(
    neptune, graph_uri: str, entity_uri: str, changed_attrs: list[str], fns: list[dict],
) -> list[str]:
    """Delete stored derived values whose invalidate_on intersects changed_attrs.

    The "recompute when input attrs change" trigger from ADR 0002 §6: callers
    on the write path pass the attribute names they just changed; any cached
    function declaring one of them in ``invalidate_on`` has its derived +
    computed-at triples deleted (next run() recomputes). Returns the names of
    the invalidated functions.
    """
    changed = set(changed_attrs)
    invalidated: list[str] = []
    for fn in fns:
        if changed & set(fn.get("invalidate_on", ())):
            await neptune.update(delete_derived_update(graph_uri, entity_uri, fn["name"]))
            invalidated.append(fn["name"])
    return invalidated
