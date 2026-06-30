"""KG registration is part of the shared write path (ONTA-153).

The bug: the ``<kg_uri> <onto/kg_name> "name"`` record that ``list_kgs`` reads to
populate the Explorer dropdown was written in exactly ONE place — ``create_kg``,
the Explorer's "New KG" button. Any non-UI writer (agent web-discovery, CLI, MCP)
that ingested into a brand-new ``kg_name`` wrote the instance data + ontology but
the KG never appeared in the dropdown (``list_kgs`` returned ``[]``).

Fix: ``refresh_after_write`` — the shared post-write housekeeping every writer
already calls — now idempotently registers the KG via ``ensure_kg_registered``.

These tests pin:
- a fresh write issues the guarded registration INSERT, a second one never
  duplicates it (``refresh_after_write`` + ``ensure_kg_registered``);
- SPARQL-literal injection is neutralized (``"`` / ``\\`` / newline produce a
  single well-formed escaped statement) for BOTH ``ensure_kg_registered`` and
  ``create_kg``, and a URI-breaking name is rejected outright;
- ``create_kg``'s re-POST response reports the EXISTING KG, not an empty/zero lie;
- a round-trip: register via ``ensure_kg_registered`` against an in-memory triple
  store, then ``list_kgs``'s metadata query returns the KG — locking the
  predicate/URI contract.
"""

import asyncio
from unittest.mock import AsyncMock

import cograph_client.api.routes.explore as explore_mod
import cograph_client.nlp.pipeline as pipeline_mod
from cograph_client.graph.kg_writer import (
    _KG_NAME_PRED,
    _kg_meta_uri,
    ensure_kg_registered,
    refresh_after_write,
)

TENANT = "test-tenant"


def _binding(**vals):
    return {k: {"value": v} for k, v in vals.items()}


def _is_registration_stmt(sparql: str, tenant_id: str, kg_name: str) -> bool:
    """True if ``sparql`` is the guarded KG-registration INSERT for this KG."""
    kg_uri = _kg_meta_uri(tenant_id, kg_name)
    return (
        "INSERT" in sparql
        and "NOT EXISTS" in sparql
        and _KG_NAME_PRED in sparql
        and kg_uri in sparql
    )


def _single_well_formed(sparql: str) -> bool:
    """Approximate 'no literal breakout': after removing escaped quotes (``\\"``),
    the remaining real delimiter quotes must be balanced (even count) and the
    statement must still be a single INSERT."""
    unescaped_quotes = sparql.replace('\\"', "").count('"')
    return unescaped_quotes % 2 == 0 and sparql.count("INSERT") == 1


# --- ensure_kg_registered: shape, idempotency, escaping, URI guard ------------


def test_ensure_kg_registered_issues_guarded_insert():
    """The helper sends a single NOT-EXISTS-guarded INSERT carrying the kg_name
    record (so it can't duplicate or clobber an existing registration). It does
    NOT write a stale kg_triple_count 0 (P2)."""

    async def run():
        neptune = AsyncMock()
        await ensure_kg_registered(neptune, TENANT, "fresh-kg")

        assert neptune.update.await_count == 1
        sparql = neptune.update.await_args.args[0]
        assert _is_registration_stmt(sparql, TENANT, "fresh-kg")
        assert "INSERT DATA" not in sparql
        assert "FILTER NOT EXISTS" in sparql
        # P2: no stale-on-arrival count is written; list_kgs computes it lazily.
        assert "kg_triple_count" not in sparql

    asyncio.run(run())


def test_ensure_kg_registered_is_idempotent_in_shape():
    """Calling twice yields a guarded (NOT EXISTS) statement BOTH times — the
    guard, not call-count bookkeeping, is what makes it non-duplicating."""

    async def run():
        neptune = AsyncMock()
        await ensure_kg_registered(neptune, TENANT, "k")
        await ensure_kg_registered(neptune, TENANT, "k")

        assert neptune.update.await_count == 2
        for call in neptune.update.await_args_list:
            assert _is_registration_stmt(call.args[0], TENANT, "k")

    asyncio.run(run())


def test_ensure_kg_registered_rejects_uri_breaking_name():
    """P0 (URI side): a name with characters that can't be created via the UI
    (``"`` / ``\\`` / newline / ``>`` / whitespace) would corrupt the registration
    URI when interpolated raw, so the helper validates against the UI pattern and
    skips rather than emitting a malformed/injected statement."""

    async def run():
        neptune = AsyncMock()
        for bad in ['evil" .', "back\\slash", "line\nbreak", "has>angle", "with space"]:
            neptune.update.reset_mock()
            await ensure_kg_registered(neptune, TENANT, bad)
            neptune.update.assert_not_awaited()

    asyncio.run(run())


def test_ensure_kg_registered_escaped_statement_is_well_formed():
    """A valid name produces a single well-formed statement with a properly
    escaped literal (P0, literal side) — no odd/unbalanced quotes."""

    async def run():
        neptune = AsyncMock()
        await ensure_kg_registered(neptune, TENANT, "valid-Name_1")
        sparql = neptune.update.await_args.args[0]
        assert _single_well_formed(sparql)

    asyncio.run(run())


def test_ensure_kg_registered_best_effort_on_failure():
    """A registration failure must never propagate out of the write path."""

    async def run():
        neptune = AsyncMock()
        neptune.update.side_effect = RuntimeError("neptune down")
        await ensure_kg_registered(neptune, TENANT, "k")  # must not raise

    asyncio.run(run())


def test_ensure_kg_registered_noop_without_name():
    async def run():
        neptune = AsyncMock()
        await ensure_kg_registered(neptune, TENANT, "")
        neptune.update.assert_not_awaited()

    asyncio.run(run())


# --- refresh_after_write hook: registers fresh KG, never duplicates -----------


def test_refresh_after_write_registers_fresh_kg(monkeypatch):
    """A writer producing facts into a fresh kg_name → refresh_after_write must
    issue the registration INSERT (the part that was missing for non-UI writers).
    A SECOND refresh must NOT duplicate it: every emission carries the NOT-EXISTS
    guard, so the second one is a no-op against an already-registered KG."""

    async def run():
        monkeypatch.setattr(
            pipeline_mod.NLQueryPipeline, "invalidate_cache", lambda graph: None
        )
        monkeypatch.setattr(pipeline_mod, "get_embedding_service", lambda: None)
        monkeypatch.setattr(
            explore_mod, "schedule_recompute",
            lambda neptune, tenant_id, kg_name: None,
        )

        neptune = AsyncMock()
        await refresh_after_write(neptune, tenant_id=TENANT, kg_name="brand-new")

        reg = [
            c.args[0] for c in neptune.update.await_args_list
            if _is_registration_stmt(c.args[0], TENANT, "brand-new")
        ]
        assert len(reg) == 1, "fresh write must register the KG exactly once"
        assert "FILTER NOT EXISTS" in reg[0]

        await refresh_after_write(neptune, tenant_id=TENANT, kg_name="brand-new")
        reg2 = [
            c.args[0] for c in neptune.update.await_args_list
            if _is_registration_stmt(c.args[0], TENANT, "brand-new")
        ]
        assert len(reg2) == 2
        assert all("FILTER NOT EXISTS" in s for s in reg2)

    asyncio.run(run())


def test_refresh_after_write_skips_registration_without_kg(monkeypatch):
    """A tenant-graph-only write (no kg_name) registers nothing."""

    async def run():
        monkeypatch.setattr(
            pipeline_mod.NLQueryPipeline, "invalidate_cache", lambda graph: None
        )
        monkeypatch.setattr(pipeline_mod, "get_embedding_service", lambda: None)
        monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

        neptune = AsyncMock()
        await refresh_after_write(neptune, tenant_id=TENANT, kg_name=None)
        neptune.update.assert_not_awaited()

    asyncio.run(run())


# --- create_kg route: escaping (P0) + truthful re-create response (P1) --------


def _create_route(*, existing_desc=None, existing_count=None):
    """A mock_neptune.query side_effect for create_kg's read-back. Returns the
    'existing' registration row (or empty to simulate read-back failure)."""

    def route(sparql, *a, **k):
        if "kg_name" in sparql and "SELECT" in sparql:
            if existing_desc is None and existing_count is None:
                return {"head": {"vars": []}, "results": {"bindings": []}}
            row = {}
            if existing_desc is not None:
                row["desc"] = existing_desc
            if existing_count is not None:
                row["count"] = existing_count
            return {
                "head": {"vars": list(row)},
                "results": {"bindings": [_binding(**row)]},
            }
        return {"head": {"vars": []}, "results": {"bindings": []}}

    return route


def test_create_kg_escapes_description_no_breakout(client, mock_neptune, auth_headers):
    """A description containing a quote/backslash must NOT break out of the SPARQL
    literal: the generated UPDATE has balanced (escaped) quotes (P0)."""
    sent = []
    mock_neptune.update.side_effect = lambda sparql, *a, **k: sent.append(sparql)
    mock_neptune.query.side_effect = _create_route(existing_desc="x", existing_count="0")

    resp = client.post(
        f"/graphs/{TENANT}/kgs",
        json={"name": "k", "description": 'evil" } } ; DROP ALL ; #'},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    assert sent, "create_kg must issue an UPDATE"
    update_sparql = sent[0]
    # The injected quote is escaped, so the statement stays single + well-formed.
    assert '\\"' in update_sparql
    assert _single_well_formed(update_sparql)


def test_create_kg_recreate_returns_existing_not_lie(client, mock_neptune, auth_headers):
    """Re-POSTing an existing KG no-ops the guarded INSERT; the response must
    report the EXISTING description + triple count, not description="" count=0 (P1)."""
    mock_neptune.update.return_value = None
    mock_neptune.query.side_effect = _create_route(
        existing_desc="the real description", existing_count="50000"
    )

    resp = client.post(
        f"/graphs/{TENANT}/kgs",
        json={"name": "k", "description": ""},  # caller sends empty on re-create
        headers=auth_headers,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["description"] == "the real description"
    assert body["triple_count"] == 50000


def test_create_kg_new_path_contract_unchanged(client, mock_neptune, auth_headers):
    """The create-new path still returns the values just written (description as
    given, count 0) — read back from the registration."""
    mock_neptune.update.return_value = None
    mock_neptune.query.side_effect = _create_route(
        existing_desc="brand new kg", existing_count="0"
    )

    resp = client.post(
        f"/graphs/{TENANT}/kgs",
        json={"name": "k", "description": "brand new kg"},
        headers=auth_headers,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "k"
    assert body["description"] == "brand new kg"
    assert body["triple_count"] == 0


# --- Round-trip: register, then list_kgs's metadata query returns the KG -------


class _InMemoryTripleStore:
    """A tiny SPARQL-ish store: applies the guarded registration INSERT and
    answers the metadata-list SELECT list_kgs issues. Just enough to lock the
    predicate + URI contract that ensure_kg_registered and list_kgs share — a
    coordinated drift in _kg_meta_uri / _KG_NAME_PRED would break this round-trip
    even though both sides change together."""

    def __init__(self):
        self.triples: dict[tuple[str, str], str] = {}  # (subject, predicate) -> object

    async def update(self, sparql, *a, **k):
        import re

        m = re.search(r'<([^>]+)>\s*<([^>]+)>\s*"([^"]*)"', sparql)
        if not m:
            return
        subj, pred, obj = m.group(1), m.group(2), m.group(3)
        if "NOT EXISTS" in sparql and (subj, pred) in self.triples:
            return  # honor the guard
        self.triples[(subj, pred)] = obj

    async def query(self, sparql, *a, **k):
        bindings = [
            _binding(name=obj)
            for (subj, pred), obj in self.triples.items()
            if pred == _KG_NAME_PRED
        ]
        return {"head": {"vars": ["name"]}, "results": {"bindings": bindings}}


def test_register_then_list_roundtrip():
    """ensure_kg_registered writes a record that list_kgs' own metadata query can
    read back — pinning the shared predicate/URI contract end-to-end."""

    async def run():
        store = _InMemoryTripleStore()
        await ensure_kg_registered(store, TENANT, "my-kg")

        from cograph_client.api.routes.knowledge_graphs import OMNIX_ONTO
        from cograph_client.graph.parser import parse_sparql_results
        from cograph_client.graph.queries import tenant_graph_uri

        base = tenant_graph_uri(TENANT)
        sparql = (
            f"SELECT ?name FROM <{base}> WHERE {{"
            f"  ?kg <{OMNIX_ONTO}/kg_name> ?name ."
            f"}}"
        )
        _, rows = parse_sparql_results(await store.query(sparql))
        names = [r.get("name") for r in rows]
        assert "my-kg" in names, "registered KG must be visible to list_kgs"

    asyncio.run(run())
