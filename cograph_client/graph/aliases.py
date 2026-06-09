"""Attribute alias mechanism — alias-first amendments, lazy backfill (ADR 0002 §7).

Renaming an attribute or moving it up the hierarchy (``phone_num → phone``)
touches instance data. Instead of an eager rewrite, the ontology records an
alias triple::

    <old-attr-IRI> <https://cograph.tech/onto/aliasOf> <new-attr-IRI>

and the query path resolves through aliases immediately — nothing breaks on
day one. Old instance triples are rewritten lazily (backfill_aliases), after
which the alias is retired (retire_alias). Aliases are a migration vehicle,
not a permanent translation layer.

Chains are allowed (a renamed attribute can itself be renamed): fetch_alias_map
flattens ``a → b → c`` to ``a → c`` so every rewrite is one hop. Cyclic alias
data (``a → b → a``) is nonsensical — entries whose chain hits a cycle are
dropped with a warning rather than rewritten unpredictably.
"""

from __future__ import annotations

import math

import structlog

from cograph_client.graph.ontology_queries import OMNIX_ONTO
from cograph_client.graph.parser import parse_sparql_results

logger = structlog.stdlib.get_logger("cograph.graph.aliases")

ALIAS_OF = f"{OMNIX_ONTO}/aliasOf"


async def register_alias(neptune, graph_uri: str, old_attr_uri: str, new_attr_uri: str) -> None:
    """Record `old_attr_uri aliasOf new_attr_uri` in the (tenant) ontology graph."""
    if old_attr_uri == new_attr_uri:
        raise ValueError(f"alias must point to a different attribute, got {old_attr_uri} -> itself")
    await neptune.update(
        f"INSERT DATA {{\n"
        f"  GRAPH <{graph_uri}> {{\n"
        f"    <{old_attr_uri}> <{ALIAS_OF}> <{new_attr_uri}> .\n"
        f"  }}\n"
        f"}}"
    )


async def retire_alias(neptune, graph_uri: str, old_attr_uri: str) -> None:
    """Remove the alias triple for `old_attr_uri` — call after backfill completes."""
    await neptune.update(
        f"DELETE WHERE {{ GRAPH <{graph_uri}> {{ <{old_attr_uri}> <{ALIAS_OF}> ?new }} }}"
    )


def alias_map_query(graph_uri: str) -> str:
    """SELECT every alias edge so a caller can build the old->new map."""
    return (
        f"SELECT ?old ?new FROM <{graph_uri}>\n"
        f"WHERE {{\n"
        f"  ?old <{ALIAS_OF}> ?new .\n"
        f"}}"
    )


async def fetch_alias_map(neptune, graph_uri: str) -> dict[str, str]:
    """Fetch the alias map for an ontology graph, chains flattened.

    ``a → b`` and ``b → c`` resolve to ``{a: c, b: c}`` so every rewrite is a
    single hop. Entries whose chain hits a cycle (including self-aliases) are
    dropped — see module docstring.
    """
    raw = await neptune.query(alias_map_query(graph_uri))
    _, bindings = parse_sparql_results(raw)
    edges = {row["old"]: row["new"] for row in bindings if row.get("old") and row.get("new")}

    resolved: dict[str, str] = {}
    for old in edges:
        target = edges[old]
        seen = {old}
        while target in edges:
            if target in seen:
                logger.warning("alias_cycle_dropped", graph_uri=graph_uri, attr_uri=old)
                target = ""
                break
            seen.add(target)
            target = edges[target]
        if target and target != old:
            resolved[old] = target
    return resolved


def rewrite_query_attrs(sparql: str, alias_map: dict[str, str]) -> str:
    """Rewrite aliased (old) attribute IRIs to their new IRIs in generated SPARQL.

    String-level and conservative, same style as rewrite_type_predicate_to_closure:
    only full ``<IRI>`` tokens are matched (the angle brackets are part of the
    match, so `<.../attrs/phone>` never fires inside `<.../attrs/phone_num>`).
    Single-pass with one alternation so a replacement is never itself re-matched.
    Empty map => the query is returned untouched.
    """
    if not alias_map:
        return sparql
    import re

    pattern = "|".join(re.escape(f"<{old}>") for old in alias_map)
    return re.sub(pattern, lambda m: f"<{alias_map[m.group(0)[1:-1]]}>", sparql)


def _count_attr_query(graph_uri: str, attr_uri: str) -> str:
    return f"SELECT (COUNT(*) AS ?n) FROM <{graph_uri}> WHERE {{ ?s <{attr_uri}> ?o . }}"


def _backfill_batch_update(graph_uri: str, old_attr_uri: str, new_attr_uri: str, limit: int) -> str:
    """One batch of the lazy rewrite: DELETE/INSERT WHERE over a LIMITed subselect."""
    return (
        f"DELETE {{ GRAPH <{graph_uri}> {{ ?s <{old_attr_uri}> ?o }} }}\n"
        f"INSERT {{ GRAPH <{graph_uri}> {{ ?s <{new_attr_uri}> ?o }} }}\n"
        f"WHERE {{\n"
        f"  {{ SELECT ?s ?o WHERE {{ GRAPH <{graph_uri}> {{ ?s <{old_attr_uri}> ?o }} }} LIMIT {limit} }}\n"
        f"}}"
    )


async def backfill_aliases(
    neptune, data_graph_uri: str, alias_map: dict[str, str], batch_size: int = 1000
) -> int:
    """Lazily rewrite old-predicate instance triples to the new predicate.

    For each alias, counts the remaining old-predicate triples in the DATA
    graph, then issues batched DELETE/INSERT WHERE updates (batch_size triples
    per update). Returns the total number of triples rewritten. After a clean
    backfill the caller retires the alias via retire_alias.
    """
    total = 0
    for old_uri, new_uri in alias_map.items():
        raw = await neptune.query(_count_attr_query(data_graph_uri, old_uri))
        _, bindings = parse_sparql_results(raw)
        try:
            count = int(bindings[0].get("n", "0")) if bindings else 0
        except ValueError:
            count = 0
        if count <= 0:
            continue
        for _ in range(math.ceil(count / batch_size)):
            await neptune.update(_backfill_batch_update(data_graph_uri, old_uri, new_uri, batch_size))
        total += count
    return total
