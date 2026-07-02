"""Drift guard: every method of getting data into a KG MUST funnel through the
one shared write path (graph/kg_writer.py).

The bug this prevents: ingestion and enrichment had forked into two write tails,
so an enriched attribute served stale embeddings / a stale NL-planning cache and
could be written un-batched, while an ingested one was fine. These tests fail the
moment a writer (ingestion via CSV/JSON, enrichment via search, or a NEW method
added later) hand-rolls its own insert or post-write housekeeping instead of
calling :func:`insert_facts` / :func:`refresh_after_write`.

Two layers:
- **Behavioral** — drive a real write entrypoint and assert it invoked the shared
  housekeeping (the part that drifted).
- **Structural** — source-level tripwire over the known writer modules, so a new
  ingest method that bypasses the shared path is caught even if no behavioral
  test happens to cover it.
"""

import inspect
import re
from unittest.mock import AsyncMock, patch

import cograph_client.api.routes.ingest as ingest_route_mod
import cograph_client.enrichment.executor as executor_mod
import cograph_client.resolver.schema_resolver as schema_resolver_mod


def _calls(src: str, name: str) -> bool:
    """True if ``src`` contains a CALL to ``name`` (``name(``) at a word boundary,
    so ``insert_triples`` does not match inside ``batched_insert_triples``."""
    return re.search(rf"(?<![\w]){re.escape(name)}\(", src) is not None


# --- Behavioral: the JSON/free-text ingest method delegates housekeeping -------


@patch("cograph_client.api.routes.ingest.refresh_after_write", new_callable=AsyncMock)
@patch("cograph_client.api.routes.ingest.SchemaResolver")
def test_ingest_route_delegates_housekeeping_to_shared_writer(
    mock_resolver_cls, mock_refresh, client, auth_headers
):
    """POST /ingest must run its post-write refresh through the shared
    refresh_after_write — not a re-inlined embed/cache-invalidate."""
    from cograph_client.resolver.models import IngestResult

    inst = AsyncMock()
    inst.ingest.return_value = IngestResult(
        entities_extracted=1,
        entities_resolved=1,
        triples_inserted=3,
        types_created=["Property"],
        attributes_added=["Property.price"],
    )
    mock_resolver_cls.return_value = inst

    resp = client.post(
        "/graphs/test-tenant/ingest",
        json={"content": "a house at 1 Main St for $500,000", "source": "t", "kg_name": "k"},
        headers=auth_headers,
    )

    assert resp.status_code == 200
    assert mock_refresh.await_count == 1
    kwargs = mock_refresh.await_args.kwargs
    assert kwargs["tenant_id"] == "test-tenant"
    assert kwargs["kg_name"] == "k"
    # types_created + the type of every attributes_added entry.
    assert kwargs["affected_types"] == {"Property"}


# --- Structural tripwire: no writer forks the shared path ----------------------


def test_enrichment_writer_uses_shared_path_not_bespoke_insert():
    """The enrichment executor writes via insert_facts + refresh_after_write and
    must NOT reintroduce a bare ``insert_triples`` instance write."""
    src = inspect.getsource(executor_mod)
    assert _calls(src, "insert_facts"), "enrichment must write via kg_writer.insert_facts"
    assert _calls(src, "refresh_after_write"), "enrichment must refresh via kg_writer.refresh_after_write"
    assert not _calls(src, "insert_triples"), (
        "enrichment reintroduced a bespoke insert_triples write — route it "
        "through graph/kg_writer.insert_facts (write-path convergence rule)."
    )


def test_ingest_writer_uses_shared_insert_and_refresh():
    """The ingest resolver writes through insert_facts, and the ingest routes
    delegate housekeeping to refresh_after_write rather than re-inlining the
    embed / cache-invalidate steps."""
    resolver_src = inspect.getsource(schema_resolver_mod)
    assert _calls(resolver_src, "insert_facts"), (
        "ingest resolver must write instance facts via kg_writer.insert_facts"
    )

    route_src = inspect.getsource(ingest_route_mod)
    assert _calls(route_src, "refresh_after_write"), (
        "ingest routes must run post-write housekeeping via kg_writer.refresh_after_write"
    )
    # Housekeeping must be DELEGATED — re-inlining these is exactly the drift.
    assert not _calls(route_src, "embed_types"), (
        "ingest route re-inlined embed_types — delegate to refresh_after_write"
    )
    assert "invalidate_cache" not in route_src, (
        "ingest route re-inlined the ontology-cache invalidation — delegate to "
        "refresh_after_write"
    )


def test_normalization_writer_uses_shared_path():
    """normalization/execute.py mutates instance data; its inserts must go
    through insert_facts and its post-write housekeeping through
    refresh_after_write — no bespoke batched insert or stats-recompute."""
    import cograph_client.normalization.execute as norm_mod

    src = inspect.getsource(norm_mod)
    assert _calls(src, "insert_facts"), "normalization must insert via kg_writer.insert_facts"
    assert _calls(src, "refresh_after_write"), (
        "normalization must run housekeeping via kg_writer.refresh_after_write"
    )
    assert not _calls(src, "batched_insert_triples"), (
        "normalization reintroduced a bespoke batched_insert_triples — route inserts "
        "through kg_writer.insert_facts (write-path convergence rule)."
    )
    assert "_schedule_stats_recompute" not in src, (
        "normalization reintroduced a bespoke stats-recompute — use "
        "kg_writer.refresh_after_write."
    )


def test_dedupe_writers_use_shared_refresh():
    """The dedupe / entity-resolution writers (which mutate counts, not schema)
    must run their post-write refresh through the shared refresh_after_write, not
    a bare schedule_recompute."""
    import cograph_client.agent.capabilities.dedup_cap as dedup_mod
    import cograph_client.api.routes.actions as actions_mod

    for mod, name in [(dedup_mod, "dedup_cap"), (actions_mod, "actions")]:
        src = inspect.getsource(mod)
        assert _calls(src, "refresh_after_write"), (
            f"{name} dedupe must refresh via kg_writer.refresh_after_write"
        )
        assert not _calls(src, "schedule_recompute"), (
            f"{name} reintroduced a bare schedule_recompute after a graph mutation — "
            "use kg_writer.refresh_after_write."
        )


def test_web_ingest_calls_refresh_after_write():
    """Web-discovery ingest (agent/capabilities/web_ingest_cap.py) CREATES new
    types/attributes/entities via the ingest engine, so its background job must run
    the same post-write housekeeping as every other writer — otherwise the
    ontology expansion stays invisible to NL planning + Explorer. The refresh must
    go through the shared refresh_after_write, not a re-inlined embed/cache step."""
    import cograph_client.agent.capabilities.web_ingest_cap as web_ingest_mod

    src = inspect.getsource(web_ingest_mod)
    assert _calls(src, "refresh_after_write"), (
        "web-discovery ingest must run post-write housekeeping via "
        "kg_writer.refresh_after_write"
    )
    assert not _calls(src, "embed_types"), (
        "web ingest re-inlined embed_types — delegate to refresh_after_write"
    )
    assert "invalidate_cache" not in src, (
        "web ingest re-inlined the ontology-cache invalidation — delegate to "
        "refresh_after_write"
    )


def test_shared_writer_is_the_single_housekeeping_owner():
    """Sanity: the shared writer itself is the one place embed/cache-invalidate/
    recompute live, so delegating to it actually centralizes the behavior."""
    import cograph_client.graph.kg_writer as kg_writer_mod

    src = inspect.getsource(kg_writer_mod)
    assert _calls(src, "batched_insert_triples"), "insert_facts must batch"
    assert "embed_types" in src
    assert "invalidate_cache" in src
    assert "schedule_recompute" in src


# --- Structural tripwire: semantic-index writers (ONTA-181) --------------------
#
# Background workers escaped the original tripwire (it only covered request-path
# writers). The semantic instance index has exactly TWO writers — the kg_writer
# hook (freshness) and the reconciler (correctness) — and both must stay on the
# shared seams: extraction via extract_semantic_chunks (one chunking/hashing
# contract), writes via the SemanticIndex protocol (upsert_chunks / delete /
# fill_embeddings / mark_embed_failed), and embeddings via the ONE shared embed
# client (nlp/embed_client.embed_texts). Kept as a separate, clearly-named test
# so a parallel extension of this file (route scanning) never collides with it.


def test_semantic_index_writers_use_shared_seams():
    """ONTA-181 drift guard for the semantic-index write hook + reconciler."""
    import cograph_client.graph.kg_writer as kg_writer_mod
    import cograph_client.semantic.reconciler as reconciler_mod

    # The write hook (kg_writer._index_semantic) chunks via the shared extractor
    # and writes via the protocol — no bespoke chunker/hasher, no direct rows.
    kg_src = inspect.getsource(kg_writer_mod)
    assert _calls(kg_src, "extract_semantic_chunks"), (
        "the kg_writer semantic hook must chunk via semantic.extract."
        "extract_semantic_chunks (one chunking/hashing contract)"
    )
    assert _calls(kg_src, "upsert_chunks"), (
        "the kg_writer semantic hook must write via SemanticIndex.upsert_chunks"
    )

    # The reconciler: same extractor + protocol writes + the ONE embed client.
    rec_src = inspect.getsource(reconciler_mod)
    assert _calls(rec_src, "extract_semantic_chunks"), (
        "the reconciler must re-extract via extract_semantic_chunks — a private "
        "chunker would fork the content_hash contract"
    )
    assert _calls(rec_src, "upsert_chunks"), (
        "the reconciler must upsert via SemanticIndex.upsert_chunks"
    )
    assert _calls(rec_src, "embed_texts"), (
        "the embed-fill sweep must embed via nlp.embed_client.embed_texts — the "
        "single embed-batch implementation (ONTA-174)"
    )
    assert _calls(rec_src, "fill_embeddings") and _calls(rec_src, "mark_embed_failed"), (
        "the sweep must persist fills/failures via the protocol's "
        "fill_embeddings / mark_embed_failed (content-hash guarded)"
    )
    assert "httpx" not in rec_src, (
        "the reconciler re-inlined an HTTP embedding call — use "
        "nlp.embed_client.embed_texts (shared embed client rule)."
    )
    # The reconciler is NOT an instance-data writer: it must never write
    # Neptune instance triples (its only Neptune writes are textKind ontology
    # markers via upsert_attribute_text_kind).
    assert not _calls(rec_src, "insert_facts")
    assert not _calls(rec_src, "insert_triples")
    assert not _calls(rec_src, "batched_insert_triples"), (
        "the reconciler grew a bespoke Neptune instance write — instance data "
        "goes through graph/kg_writer.insert_facts (write-path convergence rule)."
    )
