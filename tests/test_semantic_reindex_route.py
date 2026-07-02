"""Route-level tests for ONTA-181's HTTP surface.

* ``POST /graphs/{tenant}/kgs/{kg}/search/reindex`` — the on-demand reconcile
  trigger (202 + schedule row; NOT an inline long-running request; 503 when the
  semantic index is disabled; same ``get_tenant`` auth as every KG route).
* ``DELETE /graphs/{tenant}/kgs/{kg}`` — clears ONLY that KG's semantic rows
  (the kg_name isolation contract) and drops its reconcile schedule row.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

import cograph_client.graph.text_markers as tm
import cograph_client.semantic.reconciler as rec
from cograph_client.scheduling.store import get_schedule_store, reset_schedule_store
from cograph_client.semantic.extract import content_hash
from cograph_client.semantic.memory import InMemorySemanticIndex
from cograph_client.semantic.protocol import SemanticChunk
from cograph_client.semantic.registry import (
    register_semantic_index,
    reset_semantic_index,
)

TENANT = "test-tenant"  # conftest's static-key tenant


@pytest.fixture(autouse=True)
def _clean_state():
    reset_semantic_index()
    tm.reset_for_tests()
    rec.reset_for_tests()
    reset_schedule_store()
    yield
    reset_semantic_index()
    tm.reset_for_tests()
    rec.reset_for_tests()
    reset_schedule_store()


def _chunk(kg_name: str, n: int) -> SemanticChunk:
    text = f"chunk {n} of {kg_name}"
    return SemanticChunk(
        tenant_id=TENANT,
        kg_name=kg_name,
        entity_uri=f"https://cograph.tech/entities/Doc/{kg_name}-{n}",
        attr="description",
        chunk_ix=0,
        chunk_text=text,
        content_hash=content_hash(text),
    )


# --- reindex ------------------------------------------------------------------


def test_reindex_returns_202_and_seeds_due_now_schedule(
    monkeypatch, client, auth_headers
):
    monkeypatch.setenv("COGRAPH_SEMANTIC_INDEX_ENABLED", "true")
    register_semantic_index(InMemorySemanticIndex())

    resp = client.post(
        f"/graphs/{TENANT}/kgs/kg1/search/reindex", headers=auth_headers
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["kg_name"] == "kg1"
    assert body["schedule_id"] == rec.reconcile_schedule_id(TENANT, "kg1")
    # No runner in the test app (lifespan not entered) → in-process fallback.
    assert body["mode"] == "background-task"

    async def check():
        schedule = await get_schedule_store().get(body["schedule_id"])
        assert schedule is not None
        assert schedule.action == "semantic-reconcile"
        assert schedule.next_run <= datetime.now(timezone.utc)

    asyncio.run(check())


def test_reindex_reports_scheduled_mode_when_runner_present(
    monkeypatch, app, client, auth_headers
):
    monkeypatch.setenv("COGRAPH_SEMANTIC_INDEX_ENABLED", "true")
    app.state.schedule_runner = object()  # a live runner claims the row instead

    resp = client.post(
        f"/graphs/{TENANT}/kgs/kg1/search/reindex", headers=auth_headers
    )
    assert resp.status_code == 202
    assert resp.json()["mode"] == "scheduled"


def test_reindex_503_when_semantic_index_disabled(monkeypatch, client, auth_headers):
    monkeypatch.delenv("COGRAPH_SEMANTIC_INDEX_ENABLED", raising=False)
    resp = client.post(
        f"/graphs/{TENANT}/kgs/kg1/search/reindex", headers=auth_headers
    )
    assert resp.status_code == 503
    assert "COGRAPH_SEMANTIC_INDEX_ENABLED" in resp.json()["detail"]


def test_reindex_requires_auth(monkeypatch, client):
    monkeypatch.setenv("COGRAPH_SEMANTIC_INDEX_ENABLED", "true")
    resp = client.post(f"/graphs/{TENANT}/kgs/kg1/search/reindex")
    assert resp.status_code in (401, 403)


def test_reindex_scopes_to_the_keys_tenant(monkeypatch, client, auth_headers):
    """The same get_tenant dependency as every KG route: a single-tenant static
    key routes to ITS tenant regardless of the path (documented legacy
    behavior), so the reconcile work is scheduled under the KEY's tenant —
    never under the foreign tenant named in the path."""
    monkeypatch.setenv("COGRAPH_SEMANTIC_INDEX_ENABLED", "true")
    resp = client.post(
        "/graphs/other-tenant/kgs/kg1/search/reindex", headers=auth_headers
    )
    assert resp.status_code == 202
    # Scheduled under test-tenant (the key's tenant), not other-tenant.
    assert resp.json()["schedule_id"] == rec.reconcile_schedule_id(TENANT, "kg1")

    async def check():
        store = get_schedule_store()
        assert (
            await store.get(rec.reconcile_schedule_id("other-tenant", "kg1")) is None
        )

    asyncio.run(check())


# --- KG delete isolation ---------------------------------------------------------


def test_kg_delete_clears_only_that_kgs_semantic_rows(client, auth_headers):
    """Deleting KG A clears A's chunks and reconcile schedule; KG B's rows and
    schedule are untouched — the whole reason the index carries kg_name."""
    index = InMemorySemanticIndex()
    register_semantic_index(index)

    async def seed():
        await index.upsert_chunks([_chunk("kga", 1), _chunk("kga", 2), _chunk("kgb", 1)])
        store = get_schedule_store()
        await rec.ensure_reconcile_schedule(store, TENANT, "kga")
        await rec.ensure_reconcile_schedule(store, TENANT, "kgb")

    asyncio.run(seed())

    resp = client.delete(f"/graphs/{TENANT}/kgs/kga", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == {"deleted": "kga"}

    async def check():
        rows = await index.fetch_pending(limit=100)
        assert {r.kg_name for r in rows} == {"kgb"}
        store = get_schedule_store()
        assert await store.get(rec.reconcile_schedule_id(TENANT, "kga")) is None
        assert await store.get(rec.reconcile_schedule_id(TENANT, "kgb")) is not None

    asyncio.run(check())
