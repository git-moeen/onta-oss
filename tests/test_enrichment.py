"""Tests for the auto-enrichment feature (lite tier)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from cograph_client.enrichment.cache import EnrichmentCache
from cograph_client.enrichment.executor import (
    EnrichmentExecutor,
    _ProviderTally,
    _build_select_query,
    _entity_iri_type,
    _infer_datatype_from_values,
    _is_float,
    _is_int,
    _parse_vals,
    _resolve_pred_iris_from_bindings,
    _scope_block,
    _scope_subselect,
    _values_match,
)
from cograph_client.enrichment.job_store import InMemoryJobStore
from cograph_client.enrichment.models import (
    ConflictPolicy,
    ConflictReview,
    EnrichJob,
    EnrichmentTier,
    EnrichScope,
    JobStatus,
    Verdict,
)
from cograph_client.enrichment.sources.wikidata import (
    WikidataAdapter,
    _clean_label_candidates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(
    *,
    type_name: str = "Product",
    attributes: list[str] | None = None,
    policy: ConflictPolicy = ConflictPolicy.stage,
    confidence_min: float = 0.85,
    scope: EnrichScope | None = None,
    entity_uris: list[str] | None = None,
) -> EnrichJob:
    return EnrichJob(
        id="job-1",
        tenant_id="test-tenant",
        kg_name="kg",
        type_name=type_name,
        attributes=attributes or ["manufacturer"],
        tier=EnrichmentTier.lite,
        status=JobStatus.queued,
        created_at=datetime.now(timezone.utc),
        conflict_policy=policy,
        confidence_min=confidence_min,
        scope=scope,
        entity_uris=entity_uris,
    )


def _entities_query_response(rows: list[dict]) -> dict:
    bindings = []
    for r in rows:
        b: dict = {"e": {"type": "uri", "value": r["uri"]}}
        if r.get("label") is not None:
            b["label"] = {"type": "literal", "value": r["label"]}
        if r.get("vals") is not None:
            b["vals"] = {"type": "literal", "value": r["vals"]}
        bindings.append(b)
    return {"head": {"vars": ["e", "label", "nameAttr", "vals"]}, "results": {"bindings": bindings}}


def _count_response(n: int) -> dict:
    return {
        "head": {"vars": ["n"]},
        "results": {"bindings": [{"n": {"type": "literal", "value": str(n)}}]},
    }


def _range_response(range_uri: str | None = None) -> dict:
    """SPARQL result for get_attribute_range_query: zero rows when ``range_uri`` is
    None (attribute has no existing range → enrichment uses its inferred range), or
    one ``?range`` binding when an existing range should be preserved."""
    bindings = (
        [{"range": {"type": "uri", "value": range_uri}}] if range_uri else []
    )
    return {"head": {"vars": ["range"]}, "results": {"bindings": bindings}}


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------


def test_job_store_crud():
    async def run():
        store = InMemoryJobStore()
        job = _make_job()
        await store.create(job)

        got = await store.get("job-1")
        assert got is not None
        assert got.id == "job-1"

        # Update
        got.status = JobStatus.running
        await store.update(got)
        again = await store.get("job-1")
        assert again.status == JobStatus.running

        summaries = await store.list_for_tenant("test-tenant")
        assert len(summaries) == 1
        assert summaries[0].id == "job-1"

        await store.delete("job-1")
        assert await store.get("job-1") is None

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def test_cache_get_put():
    async def run():
        cache = EnrichmentCache()
        # Miss
        assert await cache.get("Bosch", "manufacturer", "wikidata") is None

        v = Verdict(value="Bosch GmbH", confidence=0.95, source="wikidata")
        await cache.put("Bosch", "manufacturer", "wikidata", [v])

        # Case-insensitive on entity_label
        hit = await cache.get("bosch", "manufacturer", "wikidata")
        assert hit is not None and len(hit) == 1
        assert hit[0].value == "Bosch GmbH"

        # Different attribute → still miss
        assert await cache.get("Bosch", "country", "wikidata") is None

    asyncio.run(run())


def test_cache_key_normalizes_label_and_versions(monkeypatch):
    """ADR-0005 §2 cache keying:

    (a) "City", "city", and "  City  " produce the SAME key (normalized label).
    (b) Changing strategy_version produces a DIFFERENT key (clean miss).
    """
    from cograph_client.enrichment import cache as cache_mod

    # (a) Normalized-label equivalence at the key level.
    k1 = cache_mod._key("Place", "City", "name", "v1", "wikidata")
    k2 = cache_mod._key("Place", "city", "name", "v1", "wikidata")
    k3 = cache_mod._key("Place", "  City  ", "name", "v1", "wikidata")
    assert k1 == k2 == k3
    assert cache_mod._normalize_label("  City  ") == "city"
    # Internal whitespace runs collapse to a single space.
    assert cache_mod._normalize_label("New   York") == "new york"

    async def run():
        cache = EnrichmentCache()
        v = Verdict(value="Springfield", confidence=0.95, source="wikidata")

        # Put under one strategy_version, then read back with label variants.
        await cache.put(
            "City", "name", "wikidata", [v],
            entity_type="Place", strategy_version="v1",
        )
        for variant in ("City", "city", "  City  "):
            hit = await cache.get(
                variant, "name", "wikidata",
                entity_type="Place", strategy_version="v1",
            )
            assert hit is not None and hit[0].value == "Springfield"

        # (b) A different strategy_version is a cache miss (auto-invalidation).
        miss = await cache.get(
            "City", "name", "wikidata",
            entity_type="Place", strategy_version="v2",
        )
        assert miss is None

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Wikidata adapter
# ---------------------------------------------------------------------------


def _mk_response(payload: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload
    return resp


def test_wikidata_adapter_unknown_attribute_returns_empty():
    async def run():
        adapter = WikidataAdapter()
        result = await adapter.lookup("Bosch", "not_a_known_attr", {})
        assert result == []

    asyncio.run(run())


def test_wikidata_client_has_granular_per_phase_timeout():
    """COG-112: the lazily-built httpx client must use an explicit per-phase
    ``httpx.Timeout`` (connect/read/write/pool all bounded), not a bare float.
    A bare total timeout does not bound a dribbling connection — that is what
    let the production lookup hang forever."""

    async def run():
        adapter = WikidataAdapter()
        client = await adapter._get_client()
        try:
            t = client.timeout
            assert isinstance(t, httpx.Timeout)
            # Every phase is bounded (no None == "no timeout").
            assert t.connect is not None and t.connect > 0
            assert t.read is not None and t.read > 0
            assert t.write is not None and t.write > 0
            assert t.pool is not None and t.pool > 0
        finally:
            await adapter.aclose()

    asyncio.run(run())


def test_wikidata_adapter_resolves_entity_id_claim():
    async def run():
        adapter = WikidataAdapter()
        # Inject a fake httpx client.
        client = AsyncMock()
        # Sequence: search → entities (claims) → entities (label for target)
        client.get.side_effect = [
            _mk_response({"search": [{"id": "Q176"}]}),
            _mk_response(
                {
                    "entities": {
                        "Q176": {
                            "claims": {
                                "P17": [
                                    {
                                        "mainsnak": {
                                            "datavalue": {
                                                "type": "wikibase-entityid",
                                                "value": {"id": "Q183"},
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            ),
            _mk_response(
                {
                    "entities": {
                        "Q183": {"labels": {"en": {"value": "Germany"}}}
                    }
                }
            ),
        ]
        adapter._client = client
        verdicts = await adapter.lookup("Bosch", "country", {})
        assert len(verdicts) == 1
        assert verdicts[0].value == "Germany"
        assert verdicts[0].source == "wikidata"
        assert verdicts[0].source_url == "https://www.wikidata.org/wiki/Q176"
        assert verdicts[0].confidence == 0.95

    asyncio.run(run())


def test_wikidata_adapter_handles_429_gracefully():
    async def run():
        adapter = WikidataAdapter()
        client = AsyncMock()
        client.get.side_effect = [_mk_response({}, status=429)]
        adapter._client = client
        verdicts = await adapter.lookup("Bosch", "country", {})
        assert verdicts == []

    asyncio.run(run())


def test_wikidata_adapter_no_search_results():
    async def run():
        adapter = WikidataAdapter()
        client = AsyncMock()
        # All 4 fallback candidates return no hits — capped at 4 search calls.
        client.get.side_effect = [_mk_response({"search": []})] * 4
        adapter._client = client
        verdicts = await adapter.lookup("ZZZNOPE", "country", {})
        assert verdicts == []

    asyncio.run(run())


def test_wikidata_label_strips_trailing_sku():
    """First search (full label) misses; SKU-stripped candidate hits.

    Confidence is reduced by 0.05 because we used the first fallback step.
    """
    async def run():
        adapter = WikidataAdapter()
        client = AsyncMock()
        # 1) original "Apple MacBook Pro M3" → empty
        # 2) "Apple MacBook Pro" → hit Q312 (Apple Inc.)
        # 3) entity claims for manufacturer (P176) → string value
        client.get.side_effect = [
            _mk_response({"search": []}),
            _mk_response({"search": [{"id": "Q312"}]}),
            _mk_response(
                {
                    "entities": {
                        "Q312": {
                            "claims": {
                                "P176": [
                                    {
                                        "mainsnak": {
                                            "datavalue": {
                                                "type": "string",
                                                "value": "Apple Inc.",
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            ),
        ]
        adapter._client = client
        verdicts = await adapter.lookup(
            "Apple MacBook Pro M3", "manufacturer", {}
        )
        assert len(verdicts) == 1
        assert verdicts[0].value == "Apple Inc."
        # Direct hit would be 0.95; one fallback step → 0.90.
        assert verdicts[0].confidence == pytest.approx(0.90)

    asyncio.run(run())


def test_wikidata_label_falls_back_to_first_two_tokens():
    """Original + SKU-strip both miss; first-2-tokens candidate hits.

    Confidence reduced by 0.10 (two fallback steps).
    """
    async def run():
        adapter = WikidataAdapter()
        client = AsyncMock()
        # Candidates for "Bosch fuel injector 0261545109":
        #   ["...", "Bosch fuel injector", "Bosch fuel", "Bosch"]
        # 1) original → empty
        # 2) "Bosch fuel injector" → empty
        # 3) "Bosch fuel" → hit Q234021
        # 4) entity claims for country (P17) → entity-id
        # 5) label for Q183 → "Germany"
        client.get.side_effect = [
            _mk_response({"search": []}),
            _mk_response({"search": []}),
            _mk_response({"search": [{"id": "Q234021"}]}),
            _mk_response(
                {
                    "entities": {
                        "Q234021": {
                            "claims": {
                                "P17": [
                                    {
                                        "mainsnak": {
                                            "datavalue": {
                                                "type": "wikibase-entityid",
                                                "value": {"id": "Q183"},
                                            }
                                        }
                                    }
                                ]
                            }
                        }
                    }
                }
            ),
            _mk_response(
                {
                    "entities": {
                        "Q183": {"labels": {"en": {"value": "Germany"}}}
                    }
                }
            ),
        ]
        adapter._client = client
        verdicts = await adapter.lookup(
            "Bosch fuel injector 0261545109", "country", {}
        )
        assert len(verdicts) == 1
        assert verdicts[0].value == "Germany"
        # Two fallback steps → 0.95 - 0.10 = 0.85.
        assert verdicts[0].confidence == pytest.approx(0.85)

    asyncio.run(run())


def test_wikidata_label_cleaning_unit():
    """Pure tokenizer/cleaner behavior."""
    assert _clean_label_candidates("Apple MacBook Pro M3") == [
        "Apple MacBook Pro M3",
        "Apple MacBook Pro",
        "Apple MacBook",
        "Apple",
    ]
    assert _clean_label_candidates("Bosch fuel injector 0261545109") == [
        "Bosch fuel injector 0261545109",
        "Bosch fuel injector",
        "Bosch fuel",
        "Bosch",
    ]
    # Sony case: trailing-only stripping leaves "headphones" in place;
    # SKU "WH-1000XM5" sits in the middle and is not stripped. Length is 3
    # so Candidate B (first 2 tokens) fires from the original list.
    assert _clean_label_candidates("Sony WH-1000XM5 headphones") == [
        "Sony WH-1000XM5 headphones",
        "Sony WH-1000XM5",
        "Sony",
    ]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_parse_vals():
    assert _parse_vals("") == {}
    out = _parse_vals("p1::v1||p2::v2||p1::dup")
    assert out == {"p1": "v1", "p2": "v2"}


def test_values_match():
    assert _values_match("Bosch", "Bosch GmbH")
    assert _values_match("Germany", "germany")
    assert not _values_match("Bosch", "Siemens")
    assert not _values_match("", "Bosch")


def test_build_select_query_includes_limit_and_attrs():
    q = _build_select_query("https://g/x", "Product", ["manufacturer", "country"], 50)
    assert "<https://cograph.tech/types/Product>" in q
    assert "<https://cograph.tech/types/Product/attrs/manufacturer>" in q
    assert "<https://cograph.tech/types/Product/attrs/country>" in q
    assert "LIMIT 50" in q


# ---------------------------------------------------------------------------
# COG-112 scoped enrichment: SPARQL generation
# ---------------------------------------------------------------------------


def test_build_select_query_no_scope_is_unchanged():
    """Neither scope nor entity_uris → no subset constraint; whole-type query."""
    q = _build_select_query("https://g/x", "Mentor", ["bio"], None)
    assert "?e a <https://cograph.tech/types/Mentor> ." in q
    # No subset machinery leaks in.
    assert "FILTER EXISTS" not in q
    assert "VALUES ?e" not in q


def test_build_select_query_scope_matches_bound_predicate_path():
    """A scope predicate is matched by a BOUND-predicate property-path alternation
    INLINED in the WHERE (the COG-112 fix #4) — never wrapped in ``FILTER EXISTS``
    (which Neptune evaluates once per type instance), never a variable predicate +
    ``FILTER(?p IN (...))`` (which Neptune does NOT predicate-index) and never an
    unbounded ``?e ?p ?sv`` scan — and the value is matched case-insensitively
    against a literal OR any literal property of the (already-bounded) target node
    (NOT pinned to the source type's attr namespace — COG-112 target-label fix).
    The scoped subset is reduced first by a ``SELECT DISTINCT ?e`` sub-select,
    then attributes are hydrated."""
    scope = EnrichScope(predicate="haslevel", value="Manager")
    # The resolved instance IRI(s) for the predicate (attr_uri + onto/<leaf>).
    pred_iris = [
        "https://cograph.tech/types/Mentor/attrs/haslevel",
        "https://cograph.tech/onto/haslevel",
    ]
    q = _build_select_query(
        "https://g/x", "Mentor", ["bio"], None, scope=scope, scope_pred_iris=pred_iris
    )

    # Typed to Mentor, but inside a bounded DISTINCT sub-select that the planner
    # can reduce to the scoped subset BEFORE the attribute OPTIONALs run. The
    # scope must NOT be wrapped in a FILTER EXISTS (the third COG-112 perf bug).
    assert "FILTER EXISTS" not in q
    assert "SELECT DISTINCT ?e WHERE {" in q
    assert "?e a <https://cograph.tech/types/Mentor> ." in q
    # The predicate is matched by a BOUND property-path alternation INLINED next
    # to ?e a <Type> — Neptune uses the POS index. The scope must NOT use a
    # variable predicate (?e ?p ?sv + FILTER(?p IN ...)). (The attribute-value
    # OPTIONAL legitimately uses a bounded FILTER(?p IN ...) for which attrs to
    # GROUP_CONCAT — that is not the scope predicate, so we assert specifically on
    # the scope's ?sv object var.)
    assert (
        "?e (<https://cograph.tech/types/Mentor/attrs/haslevel>|"
        "<https://cograph.tech/onto/haslevel>) ?sv ." in q
    )
    assert "?e ?p ?sv" not in q
    assert 'REPLACE(STR(?p)' not in q
    # Literal-attribute arm: case-insensitive literal match (value lower-cased).
    assert 'isLiteral(?sv) && LCASE(STR(?sv)) = "manager"' in q
    # Relationship arm: ?sv (the target node) is already bounded by the predicate
    # triple, so match ANY literal property on it (?sv ?slp ?stl) — NOT pinned to
    # the SOURCE type's attr namespace. The target's display name lives under the
    # TARGET type's namespace (e.g. …/types/Level/attrs/name), so binding the
    # source type's attr predicate would match ZERO targets (the COG-112 bug).
    assert "?sv ?slp ?stl ." in q
    assert 'isLiteral(?stl) && LCASE(STR(?stl)) = "manager"' in q
    # The relationship arm must NOT bind the TARGET-label predicate to ANY fixed
    # IRI(s) — the bug was pinning it to the SOURCE type's attr namespace
    # (…/types/Mentor/attrs/*) / rdfs:label, which matched the TARGET node under
    # the wrong namespace → zero matches. The target-label predicate is now the
    # free variable ?slp, never a `?sv <…> ?stl` bound predicate. (The OUTER
    # query still references …/attrs/name etc. for ENTITY-label hydration — that
    # is the ?fp OPTIONAL, unrelated to the scope arm — so we assert on the
    # `?sv <pred>` shape rather than substring-absence in the whole query.)
    assert "?sv <http://www.w3.org/2000/01/rdf-schema#label>" not in q
    assert "?sv <https://cograph.tech/types/Mentor/attrs/name>" not in q
    assert "?sv (<" not in q  # no property-path alternation pinned on the target
    # IRI local-name fallback for the relationship target.
    assert 'isIRI(?sv)' in q


def test_build_select_query_scope_unresolved_predicate_matches_nothing():
    """An unresolved scope predicate (no concrete IRIs) emits a fast
    ``FILTER(false)`` rather than the old unbounded per-entity predicate scan
    (COG-112 fix #3)."""
    scope = EnrichScope(predicate="haslevel", value="Manager")
    q = _build_select_query(
        "https://g/x", "Mentor", ["bio"], None, scope=scope, scope_pred_iris=[]
    )
    assert "FILTER(false)" in q
    # No predicate scan, no concrete-IRI EXISTS machinery.
    assert "FILTER EXISTS" not in q
    assert 'REPLACE(STR(?p)' not in q


def test_build_select_query_scope_escapes_value():
    """Quotes/backslashes in the scope value are escaped into the SPARQL literal."""
    scope = EnrichScope(predicate="title", value='Sr "Eng"')
    pred_iris = ["https://cograph.tech/types/Mentor/attrs/title"]
    q = _build_select_query(
        "https://g/x", "Mentor", ["bio"], None, scope=scope, scope_pred_iris=pred_iris
    )
    # The injected value is lower-cased AND quote-escaped.
    assert 'sr \\"eng\\"' in q


def test_build_select_query_scope_single_iri_no_alternation():
    """A single resolved IRI is emitted as a bare bound predicate (no parens),
    not an alternation — still POS-indexed, never a variable predicate."""
    scope = EnrichScope(predicate="title", value="Director")
    pred_iris = ["https://cograph.tech/types/Mentor/attrs/title"]
    q = _build_select_query(
        "https://g/x", "Mentor", ["bio"], None, scope=scope, scope_pred_iris=pred_iris
    )
    assert "?e <https://cograph.tech/types/Mentor/attrs/title> ?sv ." in q
    assert "?e ?p ?sv" not in q


def test_build_select_query_literal_attribute_scope_matches_rdfs_label():
    """COG-112: a literal-attribute (name) scope's SELECT must ALSO carry an
    ``rdfs:label`` arm so an entity whose displayed name lives only on
    ``rdfs:label`` (not as an ``attrs/name`` literal) is still selected — without
    regressing the bound-predicate attrs/<attr> literal arm."""
    scope = EnrichScope(predicate="name", value="Jane Doe")
    pred_iris = [
        "https://cograph.tech/types/Mentor/attrs/name",
        "https://cograph.tech/onto/name",
    ]
    q = _build_select_query(
        "https://g/x", "Mentor", ["bio"], None, scope=scope, scope_pred_iris=pred_iris
    )
    # attrs/<attr> literal arm (bound predicate, case-insensitive).
    assert (
        "?e (<https://cograph.tech/types/Mentor/attrs/name>|"
        "<https://cograph.tech/onto/name>) ?sv ." in q
    )
    assert 'isLiteral(?sv) && LCASE(STR(?sv)) = "jane doe"' in q
    # Displayed-name arm: the entity's rdfs:label matched directly.
    assert "?e <http://www.w3.org/2000/01/rdf-schema#label> ?lbl ." in q
    assert 'isLiteral(?lbl) && LCASE(STR(?lbl)) = "jane doe"' in q
    # No unbounded scan or EXISTS machinery snuck in.
    assert "FILTER EXISTS" not in q
    assert "?e ?p ?sv" not in q


def test_build_select_query_entity_uris_uses_values_block():
    """entity_uris → a VALUES ?e block; still constrained to the type."""
    uris = [
        "https://cograph.tech/entities/Mentor/m1",
        "https://cograph.tech/entities/Mentor/m2",
    ]
    q = _build_select_query("https://g/x", "Mentor", ["bio"], None, entity_uris=uris)
    assert "?e a <https://cograph.tech/types/Mentor> ." in q
    assert "VALUES ?e {" in q
    assert "<https://cograph.tech/entities/Mentor/m1>" in q
    assert "<https://cograph.tech/entities/Mentor/m2>" in q
    # No scope EXISTS machinery when using the explicit-URI primitive.
    assert "FILTER EXISTS" not in q


def test_build_select_query_entity_uris_wins_over_scope():
    """If both are passed, entity_uris is used (the documented precedence)."""
    uris = ["https://cograph.tech/entities/Mentor/m1"]
    scope = EnrichScope(predicate="haslevel", value="Manager")
    q = _build_select_query(
        "https://g/x", "Mentor", ["bio"], None, scope=scope, entity_uris=uris
    )
    assert "VALUES ?e {" in q
    assert "<https://cograph.tech/entities/Mentor/m1>" in q
    # The scope constraint must NOT appear when entity_uris wins.
    assert "FILTER EXISTS" not in q
    assert "<https://cograph.tech/onto/haslevel>" not in q


def test_scope_block_is_pure_helper():
    """_scope_block builds INLINE join patterns (no FILTER EXISTS wrapper)
    independent of the SELECT wrapper, using the concrete predicate IRI(s) it is
    handed. The first pattern is the bound-predicate triple so the planner can
    drive from it (COG-112 fix #4)."""
    block = _scope_block(
        "Mentor",
        EnrichScope(predicate="haslevel", value="Manager"),
        ["https://cograph.tech/onto/haslevel"],
    )
    # No EXISTS wrapper — the patterns are inlined directly into the WHERE.
    assert "FILTER EXISTS" not in block
    # The patterns are a top-level UNION (predicate-bound arms | rdfs:label arm).
    assert block.lstrip().startswith("{")
    # The FIRST triple is the selective bound-predicate triple (the predicate-bound
    # branch leads, so the planner still drives from it — POS-indexed).
    first_branch = block.split("} UNION {")[0]
    assert first_branch.lstrip(" {\n").startswith(
        "?e <https://cograph.tech/onto/haslevel> ?sv ."
    )
    # Predicate matched by a BOUND property path (single IRI → bare term) — no
    # variable predicate, no scan.
    assert "?e <https://cograph.tech/onto/haslevel> ?sv ." in block
    assert "FILTER(?p IN (" not in block
    assert "?e ?p ?sv" not in block
    assert "REPLACE(STR(?p)" not in block


def test_scope_block_multiple_iris_emit_alternation():
    """Multiple resolved IRIs are matched as a property-path alternation
    ``(<a>|<b>)`` with the predicate BOUND — POS-indexed, never a scan."""
    block = _scope_block(
        "Mentor",
        EnrichScope(predicate="haslevel", value="Manager"),
        [
            "https://cograph.tech/types/Mentor/attrs/haslevel",
            "https://cograph.tech/onto/haslevel",
        ],
    )
    assert (
        "?e (<https://cograph.tech/types/Mentor/attrs/haslevel>|"
        "<https://cograph.tech/onto/haslevel>) ?sv ." in block
    )
    assert "FILTER(?p IN (" not in block
    assert "?e ?p ?sv" not in block


def test_scope_block_empty_pred_iris_matches_nothing():
    """No concrete IRIs → FILTER(false) (fast matched-0), not an unbounded scan."""
    block = _scope_block("Mentor", EnrichScope(predicate="haslevel", value="x"), [])
    assert block.strip() == "FILTER(false)"


def test_scope_block_literal_attribute_also_matches_rdfs_label():
    """COG-112 literal-attribute bug: the value the user SEES in the Explorer is the
    entity's ``rdfs:label`` (set at ingest), which may exist WITHOUT a matching
    ``…/attrs/<attr>`` literal. A name/title scope must therefore ALSO match
    ``rdfs:label`` directly — as an INDEPENDENT branch (not gated on the predicate
    triple binding) — so the entity is selected even when it carries that name only
    as ``rdfs:label``. Previously the literal arm matched the ``attrs/name`` triple
    only, so such an entity matched 0."""
    block = _scope_block(
        "Mentor",
        EnrichScope(predicate="name", value="Irina Igoshkina"),
        [
            "https://cograph.tech/types/Mentor/attrs/name",
            "https://cograph.tech/onto/name",
        ],
    )
    # The attrs/<attr> literal arm is preserved (case-insensitive, lower-cased).
    assert 'isLiteral(?sv) && LCASE(STR(?sv)) = "irina igoshkina"' in block
    # The displayed-name arm matches the entity's rdfs:label directly — a SEPARATE
    # `?e <rdfs:label> ?lbl` triple, NOT a property on the predicate's object ?sv
    # (so it binds even when `?e <attrs/name> ?sv` matches nothing).
    assert "?e <http://www.w3.org/2000/01/rdf-schema#label> ?lbl ." in block
    assert 'isLiteral(?lbl) && LCASE(STR(?lbl)) = "irina igoshkina"' in block
    # The label arm is reached via a top-level UNION (its own branch), so it does
    # not require the bound-predicate triple to have produced a binding.
    assert "} UNION {" in block


def test_scope_subselect_dedups_and_caps():
    """The scoped subset is reduced by a bounded ``SELECT DISTINCT ?e`` sub-select
    that (a) types + scopes ?e with the inline patterns, (b) DISTINCT-dedups so a
    multi-arm UNION match can't multiply ?e rows, and (c) applies the LIMIT INSIDE
    the sub-select so it caps the SELECTED entities — never a FILTER EXISTS
    (COG-112 fix #4)."""
    scope = EnrichScope(predicate="haslevel", value="Manager")
    pred_iris = [
        "https://cograph.tech/types/Mentor/attrs/haslevel",
        "https://cograph.tech/onto/haslevel",
    ]
    sub = _scope_subselect("Mentor", scope, pred_iris, limit=50)
    # De-dup: a DISTINCT sub-select on ?e.
    assert "SELECT DISTINCT ?e WHERE {" in sub
    # Typed inside the sub-select.
    assert "?e a <https://cograph.tech/types/Mentor> ." in sub
    # Inline bound-predicate scope triple — no EXISTS wrapper.
    assert "FILTER EXISTS" not in sub
    assert (
        "?e (<https://cograph.tech/types/Mentor/attrs/haslevel>|"
        "<https://cograph.tech/onto/haslevel>) ?sv ." in sub
    )
    # LIMIT is INSIDE the sub-select (caps the selected entities).
    assert "LIMIT 50" in sub
    # Without a limit, no LIMIT is emitted (count path reuses this).
    sub_no_limit = _scope_subselect("Mentor", scope, pred_iris)
    assert "SELECT DISTINCT ?e WHERE {" in sub_no_limit
    assert "LIMIT" not in sub_no_limit


def test_build_select_query_scope_limit_caps_inside_subselect():
    """For a scoped SELECT the LIMIT lives INSIDE the DISTINCT sub-select (so it
    caps the SELECTED entities before the attribute OPTIONALs hydrate them), not
    as a top-level LIMIT on the GROUP BY (which would cap post-hydration rows)."""
    scope = EnrichScope(predicate="haslevel", value="Manager")
    pred_iris = ["https://cograph.tech/types/Mentor/attrs/haslevel"]
    q = _build_select_query(
        "https://g/x", "Mentor", ["bio"], 25, scope=scope, scope_pred_iris=pred_iris
    )
    # LIMIT appears within the sub-select, before the attribute OPTIONALs.
    sub_end = q.index("OPTIONAL")
    assert "LIMIT 25" in q[:sub_end]
    # The GROUP BY tail must NOT carry a second top-level LIMIT.
    assert q.rstrip().endswith("GROUP BY ?e ?label ?nameAttr")


def test_resolve_pred_iris_from_bindings_case_insensitive():
    """A request predicate resolves (case-insensitively) against the type's
    ontology-declared predicates to BOTH candidate instance IRIs; an unknown
    predicate resolves to []."""
    bindings = [
        {"attr": "https://cograph.tech/types/Mentor/attrs/haslevel", "label": "haslevel"},
        {"attr": "https://cograph.tech/types/Mentor/attrs/title", "label": "title"},
    ]
    # Mixed-case request matches the stored `haslevel` leaf/label.
    iris = _resolve_pred_iris_from_bindings("Mentor", "hasLevel", bindings)
    assert iris == [
        "https://cograph.tech/types/Mentor/attrs/haslevel",
        "https://cograph.tech/onto/haslevel",
    ]
    # Resolving by the declared label also works.
    assert _resolve_pred_iris_from_bindings("Mentor", "TITLE", bindings) == [
        "https://cograph.tech/types/Mentor/attrs/title",
        "https://cograph.tech/onto/title",
    ]
    # Unknown predicate → no IRIs (caller treats as matched 0, no scan).
    assert _resolve_pred_iris_from_bindings("Mentor", "nope", bindings) == []


# ---------------------------------------------------------------------------
# COG-112 review: SPARQL-injection hardening (validators + escaping)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_predicate",
    [
        "has>level",          # IRI-closing bracket
        "has level",          # whitespace
        'has"level',          # quote
        "has{level}",         # braces
        "",                   # empty
        "   ",                # whitespace-only
        "1level",             # must start with letter/underscore
        "ns:level",           # colon (would let it look like a prefixed name)
    ],
)
def test_enrich_scope_rejects_injecting_or_empty_predicate(bad_predicate):
    """An injecting / empty scope.predicate is rejected by the model validator
    (422 at the API boundary) and never reaches the SPARQL builder."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        EnrichScope(predicate=bad_predicate, value="Manager")


@pytest.mark.parametrize("bad_value", ["", "   "])
def test_enrich_scope_rejects_empty_value(bad_value):
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        EnrichScope(predicate="haslevel", value=bad_value)


def test_enrich_request_rejects_injecting_entity_uri():
    """A non-IRI / injecting entity_uris entry is rejected by the request model
    before it can be spliced into a VALUES block."""
    import pydantic

    from cograph_client.enrichment.models import EnrichRequest

    bad = [
        "https://cograph.tech/entities/Mentor/m1",  # valid
        "https://evil> } DROP",                      # injects out of <…>
    ]
    with pytest.raises(pydantic.ValidationError):
        EnrichRequest(
            type_name="Mentor",
            attributes=["bio"],
            kg_name="kg",
            entity_uris=bad,
        )
    # A clean list is accepted.
    ok = EnrichRequest(
        type_name="Mentor",
        attributes=["bio"],
        kg_name="kg",
        entity_uris=["https://cograph.tech/entities/Mentor/m1"],
    )
    assert ok.entity_uris == ["https://cograph.tech/entities/Mentor/m1"]


def test_build_select_query_rejects_injecting_entity_uri_at_executor():
    """Defense in depth: even if a bad URI reaches the builder (bypassing the
    request model), it raises rather than emitting an injectable VALUES term."""
    with pytest.raises(ValueError):
        _build_select_query(
            "https://g/x",
            "Mentor",
            ["bio"],
            None,
            entity_uris=["https://evil> } INSERT { ?s ?p ?o }"],
        )


# ---------------------------------------------------------------------------
# Executor end-to-end
# ---------------------------------------------------------------------------


class FakeWikidata:
    name = "wikidata"

    def __init__(self, mapping: dict[tuple[str, str], list[Verdict]]):
        self._mapping = mapping
        self.calls: list[tuple[str, str]] = []

    async def lookup(self, entity_label, attribute, context):
        self.calls.append((entity_label, attribute))
        return list(self._mapping.get((entity_label, attribute), []))


def test_executor_end_to_end_filled_verified_conflict():
    async def run():
        # Three entities: one missing manufacturer (filled), one with matching
        # value (verified), one with different value (conflict).
        mfr_pred = "https://cograph.tech/types/Product/attrs/manufacturer"
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
            {
                "uri": "https://cograph.tech/entities/Product/p2",
                "label": "Drill 18V",
                "vals": f"{mfr_pred}::Bosch",
            },
            {
                "uri": "https://cograph.tech/entities/Product/p3",
                "label": "Saw",
                "vals": f"{mfr_pred}::Acme Tools",
            },
        ]

        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")
                ],
                ("Drill 18V", "manufacturer"): [
                    Verdict(value="Bosch", confidence=0.95, source="wikidata")
                ],
                ("Saw", "manufacturer"): [
                    Verdict(value="Bosch", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["manufacturer"], policy=ConflictPolicy.stage)
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        assert final is not None
        assert final.status == JobStatus.review
        assert final.progress.total == 3
        assert final.progress.processed == 3
        assert final.progress.filled == 1
        assert final.progress.verified == 1
        assert final.progress.conflicts == 1
        # Fills, verifications, AND conflicts are retained in results so the
        # cited verdict (value + source_url + provenance) is retrievable, not
        # just conflicts. Skips/no-matches carry no verdict and are dropped.
        assert len(final.results) == 3
        assert {r.action for r in final.results} == {"filled", "verified", "conflict"}
        conflict = next(r for r in final.results if r.action == "conflict")
        assert conflict.existing_value == "Acme Tools"
        # NEW (ONTA-159): a conflict-free fill is APPLIED even under stage — only
        # the conflict is held for review. So a write DID happen and it carried
        # the p1 fill value; the conflicting p3 value was NOT written.
        neptune.update.assert_called()
        writes = " ".join(
            str(c.args[0]) if c.args else "" for c in neptune.update.call_args_list
        )
        assert "Robert Bosch GmbH" in writes  # the conflict-free fill landed
        # Prove the conflict was held by asserting its EXISTING value ("Acme
        # Tools") wasn't written. We can't assert on p3's PROPOSED verdict
        # ("Bosch") because it's a substring of the fill value above.
        assert "Acme Tools" not in writes  # the conflict was held, not overwritten

    asyncio.run(run())


def test_executor_overwrite_writes_triples():
    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        assert final.status == JobStatus.applied
        # Triple insert called.
        assert neptune.update.await_count >= 1

    asyncio.run(run())


def test_executor_cache_hit_increment():
    async def run():
        mfr_pred = "https://cograph.tech/types/Product/attrs/manufacturer"
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
            {"uri": "https://cograph.tech/entities/Product/p2", "label": "Bosch", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(policy=ConflictPolicy.stage)
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        # Second entity (same label) should hit cache.
        assert final.progress.cache_hits >= 1

    asyncio.run(run())


def test_executor_no_match_when_no_verdict():
    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Unknown", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata({})
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)
        job = _make_job()
        await store.create(job)
        await executor.run(job, "test-tenant")
        final = await store.get(job.id)
        assert final.progress.filled == 0
        assert final.progress.conflicts == 0
        assert final.progress.processed == 1
        # A stage-policy run that found NOTHING to stage must complete as
        # ``applied`` (a finished run that changed nothing), NOT ``review`` —
        # there is nothing to review, so "In review" with zero results would
        # strand the job and confuse the user.
        assert final.status == JobStatus.applied
        assert final.results == []

    asyncio.run(run())


def test_executor_apply_routes_through_shared_writer(monkeypatch):
    """The enrichment apply path MUST go through the shared writer's post-write
    housekeeping (graph/kg_writer.refresh_after_write) — the convergence
    guarantee with CSV/JSON ingestion. Regression guard: if someone reintroduces
    a bespoke write tail that skips re-embed / cache-invalidate, this fails."""
    import cograph_client.enrichment.executor as ex

    captured: dict = {}

    async def fake_refresh(neptune, *, tenant_id, kg_name, affected_types, recompute_stats=True):
        captured["called"] = True
        captured["tenant_id"] = tenant_id
        captured["kg_name"] = kg_name
        captured["affected_types"] = set(affected_types)

    monkeypatch.setattr(ex, "refresh_after_write", fake_refresh)

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        store = InMemoryJobStore()
        wikidata = FakeWikidata(
            {("Bosch", "manufacturer"): [Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")]}
        )
        executor = EnrichmentExecutor(neptune, store, EnrichmentCache(), wikidata)
        # overwrite policy + empty existing → action=filled → a write happens.
        job = _make_job(attributes=["manufacturer"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        assert captured.get("called") is True
        assert captured["kg_name"] == "kg"
        assert captured["affected_types"] == {"Product"}

    asyncio.run(run())


def test_executor_stage_with_no_results_completes_applied():
    """stage policy boundaries (ONTA-159): all no_match → ``applied`` (nothing to
    do); a conflict-free FILL → ``applied`` (auto-applied, nothing to reconcile);
    only a real value-vs-value CONFLICT → ``review``."""

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Unknown", "vals": ""},
        ]

        # No verdicts → no_match for every row → nothing staged.
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        store = InMemoryJobStore()
        empty_job = _make_job(policy=ConflictPolicy.stage)
        await store.create(empty_job)
        await EnrichmentExecutor(
            neptune, store, EnrichmentCache(), FakeWikidata({})
        ).run(empty_job, "test-tenant")
        done_empty = await store.get(empty_job.id)
        assert done_empty.status == JobStatus.applied
        assert done_empty.results == []

        # A confident verdict into an EMPTY field → conflict-free fill → APPLIED.
        neptune2 = AsyncMock()
        neptune2.query.return_value = _entities_query_response(rows)
        neptune2.update.return_value = None
        store2 = InMemoryJobStore()
        fill_job = _make_job(attributes=["manufacturer"], policy=ConflictPolicy.stage)
        await store2.create(fill_job)
        wikidata = FakeWikidata(
            {("Unknown", "manufacturer"): [Verdict(value="Acme", confidence=0.95, source="wikidata")]}
        )
        await EnrichmentExecutor(
            neptune2, store2, EnrichmentCache(), wikidata
        ).run(fill_job, "test-tenant")
        done_fill = await store2.get(fill_job.id)
        assert done_fill.status == JobStatus.applied  # ONTA-159: fills auto-apply
        assert done_fill.progress.filled == 1
        neptune2.update.assert_called()

        # A verdict that DIFFERS from an existing value → real conflict → REVIEW.
        mfr_pred = "https://cograph.tech/types/Product/attrs/manufacturer"
        rows3 = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Widget",
             "vals": f"{mfr_pred}::Acme Tools"},
        ]
        neptune3 = AsyncMock()
        neptune3.query.return_value = _entities_query_response(rows3)
        neptune3.update.return_value = None
        store3 = InMemoryJobStore()
        conflict_job = _make_job(attributes=["manufacturer"], policy=ConflictPolicy.stage)
        await store3.create(conflict_job)
        wikidata3 = FakeWikidata(
            {("Widget", "manufacturer"): [Verdict(value="Globex", confidence=0.95, source="wikidata")]}
        )
        await EnrichmentExecutor(
            neptune3, store3, EnrichmentCache(), wikidata3
        ).run(conflict_job, "test-tenant")
        done_conflict = await store3.get(conflict_job.id)
        assert done_conflict.status == JobStatus.review
        assert done_conflict.progress.conflicts == 1
        assert done_conflict.results[0].action == "conflict"

    asyncio.run(run())


def test_executor_no_match_is_counted_and_excluded_from_results():
    """A lookup that finds nothing is a FIRST-CLASS, COUNTED outcome — not a black
    hole. When the adapter chain returns NO verdict for EVERY (entity, attribute)
    pair, the finished job must report ``progress.no_match == n*a`` (every pair
    missed) with ``filled``/``verified``/``conflicts`` all 0, and the no_match rows
    (which carry no verdict) must NOT appear in ``job.results``."""

    async def run():
        # 2 entities × 2 attributes = 4 pairs, all missing.
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "ZZZNOPE", "vals": ""},
            {"uri": "https://cograph.tech/entities/Product/p2", "label": "QQQNADA", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        # Empty mapping → FakeWikidata returns [] for every (label, attribute).
        wikidata = FakeWikidata({})
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(
            attributes=["manufacturer", "country"],
            policy=ConflictPolicy.stage,
        )
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        assert final is not None
        # Every (entity, attribute) pair is a miss.
        assert final.progress.total == 4
        assert final.progress.processed == 4
        assert final.progress.no_match == 4  # 2 entities × 2 attributes
        # Nothing was filled / verified / conflicted.
        assert final.progress.filled == 0
        assert final.progress.verified == 0
        assert final.progress.conflicts == 0
        # no_match rows carry no verdict → they are dropped from results entirely.
        assert final.results == []
        assert all(r.action != "no_match" for r in final.results)
        # No SPARQL writes for a stage policy that found nothing.
        neptune.update.assert_not_called()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# COG-112: a hung adapter lookup must NOT strand the whole job (the production
# hang). A single ``await adapter.lookup(...)`` that never returns and never
# raises (a stalled network call) used to leave the job in ``running`` forever:
# logs stop right after the scoped SELECT, no outbound HTTP, no
# enrichment_job_failed, no completion. The executor now bounds every adapter
# call with ``asyncio.wait_for``, so a stall surfaces as a logged
# ``enrichment_adapter_timeout`` (verdicts=[] → the chain moves on) and the job
# completes. Each test wraps ``executor.run`` in its own ``asyncio.wait_for`` so
# that if the bound regresses the test FAILS (TimeoutError) instead of hanging
# CI forever.
# ---------------------------------------------------------------------------


class _HangingAdapter:
    """A SourceAdapter whose ``lookup`` never returns and never raises —
    mimics a stalled httpx network call (no connect/read timeout fires because
    the connection lingers). Named ``wikidata`` so the default ``lite`` chain
    (["wikidata"]) resolves it after the executor registers it."""

    name = "wikidata"

    def __init__(self) -> None:
        self.calls = 0

    async def lookup(self, entity_label, attribute, context):
        self.calls += 1
        await asyncio.Event().wait()  # block forever, never raise
        return []


def test_executor_hung_adapter_does_not_strand_job(monkeypatch):
    """Regression for COG-112: a forever-hanging adapter must time out per
    lookup and let the job finish, not leave it stuck in ``running``."""

    async def run():
        # Tiny per-adapter timeout so the test is fast. The executor reads this
        # env var at module import, so patch the module-level constant directly.
        import cograph_client.enrichment.executor as ex_mod

        monkeypatch.setattr(ex_mod, "ADAPTER_LOOKUP_TIMEOUT_S", 0.2)

        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Acme", "vals": ""},
            {"uri": "https://cograph.tech/entities/Product/p2", "label": "Globex", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        hang = _HangingAdapter()
        executor = EnrichmentExecutor(neptune, store, cache, hang)

        job = _make_job(policy=ConflictPolicy.skip)
        await store.create(job)

        # If the per-adapter timeout regresses, run() hangs → wait_for raises
        # TimeoutError → the test FAILS (loud) instead of hanging CI.
        await asyncio.wait_for(executor.run(job, "test-tenant"), timeout=10)

        final = await store.get(job.id)
        assert final is not None
        # The job MUST reach a terminal state, not be stuck in `running`.
        assert final.status == JobStatus.applied
        # The adapter was actually invoked (and timed out) for each entity.
        assert hang.calls == 2
        # Nothing usable came back, so no triples were written.
        neptune.update.assert_not_called()

    asyncio.run(run())


def test_executor_completes_with_fast_adapter_under_wait_for():
    """Control: with a fast adapter the same job completes well within the
    wait_for budget and writes triples — proving the timeout backstop does not
    interfere with the normal path."""

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(policy=ConflictPolicy.overwrite)
        await store.create(job)
        await asyncio.wait_for(executor.run(job, "test-tenant"), timeout=10)

        final = await store.get(job.id)
        assert final.status == JobStatus.applied
        assert final.progress.filled == 1
        assert neptune.update.await_count >= 1

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Optional enrichment knobs: instructions + sources override
# ---------------------------------------------------------------------------


class _NamedAdapter:
    """A SourceAdapter that yields a fixed verdict and records the context dict
    it was called with, so tests can assert instructions threading + which
    adapter a chain override actually invoked. Configurable ``name``."""

    def __init__(self, name: str, value: str = "FromAdapter") -> None:
        self.name = name
        self._value = value
        self.calls: list[tuple[str, str, dict]] = []

    async def lookup(self, entity_label, attribute, context):
        self.calls.append((entity_label, attribute, dict(context)))
        return [Verdict(value=self._value, confidence=0.95, source=self.name)]


def _single_product_neptune():
    rows = [
        {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
    ]
    neptune = AsyncMock()
    neptune.query.return_value = _entities_query_response(rows)
    neptune.update.return_value = None
    return neptune


def test_executor_sources_override_uses_named_chain():
    """When job.sources is set (and no per-attribute strategy sources), the
    executor walks THAT chain instead of the tier default — invoking the named,
    registered adapter."""

    async def run():
        from cograph_client.enrichment.sources.base import register_adapter

        neptune = _single_product_neptune()
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata({})  # tier default would call this; it must NOT
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        custom = _NamedAdapter("customsrc", value="Robert Bosch GmbH")
        register_adapter(custom)

        job = _make_job(policy=ConflictPolicy.stage)
        job.sources = ["customsrc"]
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        assert final.status == JobStatus.applied  # ONTA-159: conflict-free fill auto-applies
        assert final.progress.filled == 1
        # The override adapter was used; the tier-default wikidata was not.
        assert custom.calls and custom.calls[0][1] == "manufacturer"
        assert wikidata.calls == []

    asyncio.run(run())


def test_executor_sources_empty_falls_back_to_tier_chain():
    """An empty sources list is falsy → the executor keeps today's tier default
    (wikidata), so omitting/clearing the override changes nothing."""

    async def run():
        neptune = _single_product_neptune()
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(policy=ConflictPolicy.stage)
        job.sources = []  # explicitly empty → fall back
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        assert final.progress.filled == 1
        # Fell back to the tier default chain (wikidata).
        assert wikidata.calls == [("Bosch", "manufacturer")]

    asyncio.run(run())


def test_executor_sources_unknown_provider_falls_back_to_tier_chain():
    """A sources override naming ONLY unregistered (e.g. premium-only) providers
    falls back to the tier default chain rather than enriching nothing — matching
    the UI's "falls back to Auto if unavailable" promise. The job completes and
    the default (wikidata) adapter is still consulted."""

    async def run():
        neptune = _single_product_neptune()
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(policy=ConflictPolicy.stage)
        job.sources = ["exa"]  # not registered in OSS → no available override
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        # The all-unavailable override falls back to the tier chain, so wikidata
        # is consulted and the attribute is filled (not a silent empty job).
        assert final.status == JobStatus.applied  # ONTA-159: conflict-free fill auto-applies
        assert final.progress.filled == 1
        assert final.progress.processed == 1
        assert wikidata.calls != []

    asyncio.run(run())


def test_executor_instructions_flow_into_lookup_context():
    """job.instructions is threaded into the adapter lookup context dict; when
    absent the context stays empty (unchanged call shape)."""

    async def run():
        from cograph_client.enrichment.sources.base import register_adapter

        # With instructions.
        neptune = _single_product_neptune()
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        executor = EnrichmentExecutor(neptune, store, cache, FakeWikidata({}))
        adapter = _NamedAdapter("instr_src", value="Robert Bosch GmbH")
        register_adapter(adapter)

        job = _make_job(policy=ConflictPolicy.stage)
        job.sources = ["instr_src"]
        job.instructions = "Prefer the official legal entity name."
        await store.create(job)
        await executor.run(job, "test-tenant")

        assert adapter.calls
        assert adapter.calls[0][2] == {
            "instructions": "Prefer the official legal entity name."
        }

        # Without instructions → empty context.
        neptune2 = _single_product_neptune()
        store2 = InMemoryJobStore()
        executor2 = EnrichmentExecutor(
            neptune2, store2, EnrichmentCache(), FakeWikidata({})
        )
        adapter2 = _NamedAdapter("instr_src2", value="Robert Bosch GmbH")
        register_adapter(adapter2)
        job2 = _make_job(policy=ConflictPolicy.stage)
        job2.sources = ["instr_src2"]
        await store2.create(job2)
        await executor2.run(job2, "test-tenant")
        assert adapter2.calls and adapter2.calls[0][2] == {}

    asyncio.run(run())


def test_executor_instructions_vary_cache_key():
    """Two jobs with DIFFERENT instructions must not share a cached verdict —
    instructions are folded into the cache strategy_version, so the second job
    re-queries the adapter rather than reusing the first job's cached result."""

    async def run():
        from cograph_client.enrichment.sources.base import register_adapter

        store = InMemoryJobStore()
        cache = EnrichmentCache()  # shared across both jobs
        adapter = _NamedAdapter("cachesrc", value="Robert Bosch GmbH")
        register_adapter(adapter)

        async def run_job(instructions):
            neptune = _single_product_neptune()
            executor = EnrichmentExecutor(neptune, store, cache, FakeWikidata({}))
            job = _make_job(policy=ConflictPolicy.stage)
            job.id = f"job-{instructions or 'none'}"
            job.sources = ["cachesrc"]
            job.instructions = instructions
            await store.create(job)
            await executor.run(job, "test-tenant")

        await run_job("instruction A")
        await run_job("instruction B")
        # Different instructions → different cache key → adapter invoked twice.
        assert len(adapter.calls) == 2

        # Same instructions as the first run → cache HIT, no third invocation.
        await run_job("instruction A")
        assert len(adapter.calls) == 2

    asyncio.run(run())


def test_strategy_version_with_instructions_helper():
    """The cache-version helper is byte-for-byte identity without instructions,
    and stable + distinct per instruction string with them."""
    from cograph_client.enrichment.executor import (
        _strategy_version_with_instructions,
    )

    assert _strategy_version_with_instructions("v1", None) == "v1"
    assert _strategy_version_with_instructions("v1", "") == "v1"
    a = _strategy_version_with_instructions("v1", "do X")
    b = _strategy_version_with_instructions("v1", "do Y")
    assert a != b and a.startswith("v1+instr:")
    # Stable for the same input.
    assert a == _strategy_version_with_instructions("v1", "do X")


# ---------------------------------------------------------------------------
# COG-112 scoped enrichment: executor end-to-end
# ---------------------------------------------------------------------------


def _ontology_predicates_response(predicates: list[dict]) -> dict:
    """Build a SPARQL response for the scope-predicate-resolution SELECT
    (``?attr a rdf:Property ; rdfs:domain <type> ; rdfs:label ?label``).

    predicates: list of {"attr": <onto attr URI>, "label": <display label>}.
    """
    bindings = []
    for p in predicates:
        bindings.append(
            {
                "attr": {"type": "uri", "value": p["attr"]},
                "label": {"type": "literal", "value": p.get("label", "")},
            }
        )
    return {"head": {"vars": ["attr", "label"]}, "results": {"bindings": bindings}}


# Default ontology predicate declarations for the Mentor type used in the scope
# tests: a `haslevel` relationship and `title` literal attribute, plus the
# `name` display attribute. Keyed by ontology attr URI + its label.
_MENTOR_ONTO_PREDS = [
    {"attr": "https://cograph.tech/types/Mentor/attrs/haslevel", "label": "haslevel"},
    {"attr": "https://cograph.tech/types/Mentor/attrs/title", "label": "title"},
    {"attr": "https://cograph.tech/types/Mentor/attrs/name", "label": "name"},
]


def _capturing_neptune(scoped_rows: list[dict], onto_preds: list[dict] | None = None):
    """An AsyncMock Neptune whose entity-SELECT returns ``scoped_rows`` and that
    records the SELECT SPARQL it was asked.

    Two queries run over the tenant ontology graph (no ``/kg/`` segment): the
    scope-predicate resolution (``rdfs:domain`` + ``rdf:Property``) and the
    strategy load (``onto/matchKey`` etc.). The resolution returns
    ``onto_preds`` (default: the Mentor predicate set) so a scope predicate
    resolves to a concrete IRI; the strategy load returns empty (no strategy)."""
    neptune = AsyncMock()
    captured: dict[str, str] = {}
    preds = _MENTOR_ONTO_PREDS if onto_preds is None else onto_preds

    async def query(sparql, *args, **kwargs):
        # Tenant ontology graph (no /kg/ segment): either resolution or strategy.
        if "/graphs/test-tenant>" in sparql and "/kg/" not in sparql:
            # Scope-predicate resolution: domain + rdf:Property shape.
            if "#domain>" in sparql and "#Property>" in sparql:
                return _ontology_predicates_response(preds)
            # Strategy load: empty so no strategy is applied.
            return _strategy_query_response([])
        captured["select"] = sparql
        return _entities_query_response(scoped_rows)

    neptune.query.side_effect = query
    neptune.update.return_value = None
    return neptune, captured


def test_executor_scope_relationship_selects_only_scoped_entities():
    """Acceptance shape (COG-112): a scope on `haslevel`=Manager over Mentor must
    (a) put the scope constraint into the entity-selection SPARQL and (b) only
    enrich/count the entities the (mocked) Neptune returns for that scope — not
    the whole type. progress.total reflects the scoped matched count."""
    async def run():
        # Mocked Neptune returns ONLY the two Manager-level mentors for the
        # scoped SELECT (as a real Neptune would, given the constraint).
        scoped_rows = [
            {"uri": "https://cograph.tech/entities/Mentor/m1", "label": "Ada", "vals": ""},
            {"uri": "https://cograph.tech/entities/Mentor/m2", "label": "Grace", "vals": ""},
        ]
        neptune, captured = _capturing_neptune(scoped_rows)

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Ada", "bio"): [Verdict(value="Ada bio", confidence=0.95, source="wikidata")],
                ("Grace", "bio"): [Verdict(value="Grace bio", confidence=0.95, source="wikidata")],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(
            type_name="Mentor",
            attributes=["bio"],
            scope=EnrichScope(predicate="haslevel", value="Manager"),
        )
        await store.create(job)
        await executor.run(job, "test-tenant")

        # (a) The scope constraint is present in the entity-selection SPARQL,
        # inlined in a bounded DISTINCT sub-select (NOT a FILTER EXISTS — the
        # third COG-112 perf bug) and matched via a BOUND-predicate property path
        # — no variable predicate + FILTER, no unbounded ?e ?p ?sv scan.
        sel = captured["select"]
        assert "FILTER EXISTS" not in sel
        assert "SELECT DISTINCT ?e WHERE {" in sel
        assert (
            "?e (<https://cograph.tech/types/Mentor/attrs/haslevel>|"
            "<https://cograph.tech/onto/haslevel>) ?sv ." in sel
        )
        assert "?e ?p ?sv" not in sel
        assert "REPLACE(STR(?p)" not in sel
        # Relationship arm matches ANY literal property of the bounded target node
        # (?sv ?slp ?stl), NOT a label predicate pinned to the SOURCE type's attr
        # namespace — the target's name lives under the TARGET type's namespace
        # (e.g. …/types/Level/attrs/name), so pinning Mentor's would match 0.
        assert "?sv ?slp ?stl ." in sel
        assert 'isLiteral(?stl) && LCASE(STR(?stl)) = "manager"' in sel
        # The target-label predicate is the free variable ?slp, never bound to the
        # SOURCE type's attr namespace (the bug). (…/attrs/name still appears in
        # the OUTER ?fp entity-label OPTIONAL — unrelated — so assert on shape.)
        assert "?sv <https://cograph.tech/types/Mentor/attrs/name>" not in sel
        assert "?sv (<" not in sel

        # (b) Only the two scoped entities were processed (not all Mentors).
        final = await store.get(job.id)
        assert final is not None
        assert final.progress.total == 2  # 2 entities × 1 attribute
        assert final.progress.processed == 2
        assert final.progress.filled == 2
        # Each scoped entity was looked up exactly once for "bio".
        assert sorted(lbl for lbl, _ in wikidata.calls) == ["Ada", "Grace"]

    asyncio.run(run())


def test_executor_scope_predicate_casing_matches_via_lcase():
    """A mixed-case request predicate (`hasLevel`) selects entities stored under
    a `haslevel` local-name: the generated SELECT LCASEs the predicate
    local-name and compares to the lower-cased request value (#2)."""
    async def run():
        scoped_rows = [
            {"uri": "https://cograph.tech/entities/Mentor/m1", "label": "Ada", "vals": ""},
        ]
        neptune, captured = _capturing_neptune(scoped_rows)
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {("Ada", "bio"): [Verdict(value="Ada bio", confidence=0.95, source="wikidata")]}
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(
            type_name="Mentor",
            attributes=["bio"],
            # Request uses mixed case; storage local-name is `haslevel`.
            scope=EnrichScope(predicate="hasLevel", value="Manager"),
        )
        await store.create(job)
        await executor.run(job, "test-tenant")

        sel = captured["select"]
        # The mixed-case `hasLevel` request resolves (case-insensitively) to the
        # stored `haslevel` predicate's concrete IRI(s), matched as a bound
        # property path; the mixed-case form never leaks verbatim and there is no
        # variable-predicate scan.
        assert (
            "?e (<https://cograph.tech/types/Mentor/attrs/haslevel>|"
            "<https://cograph.tech/onto/haslevel>) ?sv ." in sel
        )
        assert "?e ?p ?sv" not in sel
        assert "hasLevel" not in sel
        assert "REPLACE(STR(?p)" not in sel

        final = await store.get(job.id)
        assert final.progress.total == 1
        assert final.progress.processed == 1

    asyncio.run(run())


def test_executor_scope_relationship_not_in_ontology_attrs_still_matches():
    """COG-112 root-cause fix: a RELATIONSHIP-style predicate (`haslevel`) that is
    NOT declared as an ontology ATTRIBUTE (`rdf:Property ; rdfs:domain <Type> ;
    rdfs:label`) — because relationships live under `…/onto/<pred>`, not the
    attr namespace — must STILL resolve to candidate instance IRIs via the direct
    build, so the SELECT carries the bound-predicate property path (NOT
    FILTER(false)). Previously the ontology-only resolver returned [] for
    relationships → FILTER(false) → matched 0 (the bug)."""
    async def run():
        scoped_rows = [
            {"uri": "https://cograph.tech/entities/Mentor/m1", "label": "Ada", "vals": ""},
        ]
        # Ontology declares only `title`/`name` as attributes — `haslevel`
        # (the relationship) is absent from the attribute bindings, exactly like
        # the live adp-mentors data.
        onto_preds = [
            {"attr": "https://cograph.tech/types/Mentor/attrs/title", "label": "title"},
            {"attr": "https://cograph.tech/types/Mentor/attrs/name", "label": "name"},
        ]
        neptune, captured = _capturing_neptune(scoped_rows, onto_preds=onto_preds)
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {("Ada", "bio"): [Verdict(value="Ada bio", confidence=0.95, source="wikidata")]}
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(
            type_name="Mentor",
            attributes=["bio"],
            scope=EnrichScope(predicate="haslevel", value="Manager"),
        )
        await store.create(job)
        await executor.run(job, "test-tenant")

        sel = captured["select"]
        # The direct build means `…/onto/haslevel` is ALWAYS a candidate, so the
        # bound-predicate property path is emitted (NOT FILTER(false)).
        assert "FILTER(false)" not in sel
        assert (
            "?e (<https://cograph.tech/types/Mentor/attrs/haslevel>|"
            "<https://cograph.tech/onto/haslevel>) ?sv ." in sel
        )
        # No unbounded scan, no FILTER EXISTS machinery.
        assert "FILTER EXISTS" not in sel
        assert "REPLACE(STR(?p)" not in sel
        # The scoped entity was actually processed (not matched-0).
        final = await store.get(job.id)
        assert final is not None
        assert final.progress.total == 1
        assert final.progress.processed == 1

    asyncio.run(run())


def test_resolve_scope_predicate_iris_unions_direct_build_for_relationships():
    """COG-112 root-cause unit test: `_resolve_scope_predicate_iris` returns the
    UNION of the ontology-declared resolution and the direct build. When the
    ontology query returns NO matching attribute (relationship case), the result
    still INCLUDES `…/onto/<pred>` (and the attr IRI) from the direct build."""
    async def run():
        neptune = AsyncMock()

        async def query(sparql, *args, **kwargs):
            # Ontology declares only `title` — `haslevel` is absent (relationship).
            if "#domain>" in sparql and "#Property>" in sparql:
                return _ontology_predicates_response(
                    [{"attr": "https://cograph.tech/types/Mentor/attrs/title", "label": "title"}]
                )
            return _strategy_query_response([])

        neptune.query.side_effect = query
        executor = EnrichmentExecutor(
            neptune, InMemoryJobStore(), EnrichmentCache(), FakeWikidata({})
        )

        iris = await executor._resolve_scope_predicate_iris(
            "test-tenant", "Mentor", EnrichScope(predicate="haslevel", value="Manager")
        )
        # The relationship is NOT in the ontology attribute bindings, but the
        # direct build always yields the onto/<pred> candidate (the fix).
        assert "https://cograph.tech/onto/haslevel" in iris
        assert "https://cograph.tech/types/Mentor/attrs/haslevel" in iris

        # An ATTRIBUTE declared in the ontology still resolves (and the union
        # dedups: the ontology arm and the direct build agree on the same IRIs).
        attr_iris = await executor._resolve_scope_predicate_iris(
            "test-tenant", "Mentor", EnrichScope(predicate="TITLE", value="Senior")
        )
        assert attr_iris == [
            "https://cograph.tech/types/Mentor/attrs/title",
            "https://cograph.tech/onto/title",
        ]

    asyncio.run(run())


def test_executor_scope_literal_attribute_constraint_in_sparql():
    """A scope on a literal attribute emits the literal-match arm in the SELECT."""
    async def run():
        scoped_rows = [
            {"uri": "https://cograph.tech/entities/Mentor/m1", "label": "Ada", "vals": ""},
        ]
        neptune, captured = _capturing_neptune(scoped_rows)
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {("Ada", "bio"): [Verdict(value="Ada bio", confidence=0.95, source="wikidata")]}
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(
            type_name="Mentor",
            attributes=["bio"],
            scope=EnrichScope(predicate="title", value="Director"),
        )
        await store.create(job)
        await executor.run(job, "test-tenant")

        sel = captured["select"]
        assert 'isLiteral(?sv) && LCASE(STR(?sv)) = "director"' in sel
        # Predicate matched by the concrete IRI(s) it resolved to, bound as a
        # property-path alternation — no variable predicate, no scan.
        assert (
            "?e (<https://cograph.tech/types/Mentor/attrs/title>|"
            "<https://cograph.tech/onto/title>) ?sv ." in sel
        )
        assert "?e ?p ?sv" not in sel
        assert "REPLACE(STR(?p)" not in sel

        final = await store.get(job.id)
        assert final.progress.total == 1
        assert final.progress.processed == 1

    asyncio.run(run())


def test_executor_scope_literal_name_value_matches_via_rdfs_label():
    """COG-112 literal-attribute bug end-to-end: a scope on the `name` attribute by
    the value the user SEES (the entity's displayed name, i.e. its `rdfs:label`)
    must reach the SELECT with an `rdfs:label` arm so the entity is selected even
    when it carries that name ONLY as `rdfs:label` and NOT as an `attrs/name`
    literal. Mocked Neptune returns the matching mentor (as a real Neptune would
    given the rdfs:label arm); the assertion is on the generated SPARQL + that the
    entity was processed (not matched-0)."""
    async def run():
        scoped_rows = [
            {
                "uri": "https://cograph.tech/entities/Mentor/m1",
                "label": "Irina Igoshkina",
                "vals": "",
            },
        ]
        neptune, captured = _capturing_neptune(scoped_rows)
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Irina Igoshkina", "bio"): [
                    Verdict(value="Irina bio", confidence=0.95, source="wikidata")
                ]
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(
            type_name="Mentor",
            attributes=["bio"],
            scope=EnrichScope(predicate="name", value="Irina Igoshkina"),
        )
        await store.create(job)
        await executor.run(job, "test-tenant")

        sel = captured["select"]
        # attrs/<attr> literal arm (value lower-cased, case-insensitive).
        assert 'isLiteral(?sv) && LCASE(STR(?sv)) = "irina igoshkina"' in sel
        # Displayed-name arm: the entity's rdfs:label matched directly — this is the
        # fix that lets a name carried ONLY on rdfs:label be selected.
        assert "?e <http://www.w3.org/2000/01/rdf-schema#label> ?lbl ." in sel
        assert 'isLiteral(?lbl) && LCASE(STR(?lbl)) = "irina igoshkina"' in sel
        assert "FILTER(false)" not in sel
        assert "FILTER EXISTS" not in sel

        # The matched mentor was actually processed (not matched-0 — the bug).
        final = await store.get(job.id)
        assert final is not None
        assert final.progress.total == 1
        assert final.progress.processed == 1
        assert [lbl for lbl, _ in wikidata.calls] == ["Irina Igoshkina"]

    asyncio.run(run())


def test_executor_scope_literal_attribute_value_emits_dedicated_attrs_arm():
    """COG-112 literal-attribute fix: a scope on the `name` attribute by its VALUE
    must emit a DEDICATED single-bound-predicate `attrs/<attr>` literal arm (the
    namespace the resolver actually writes literal attribute values under —
    `…/types/<Type>/attrs/<attr>`), so the match doesn't depend on the property-
    path alternation that mixes the literal namespace with the zero-triple
    `…/onto/<attr>` (the shape that silently bound nothing on Neptune)."""
    async def run():
        scoped_rows = [
            {"uri": "https://cograph.tech/entities/Mentor/m1", "label": "Jane Doe", "vals": ""},
        ]
        neptune, captured = _capturing_neptune(scoped_rows)
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {("Jane Doe", "bio"): [Verdict(value="bio", confidence=0.95, source="wikidata")]}
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(
            type_name="Mentor",
            attributes=["bio"],
            scope=EnrichScope(predicate="name", value="Jane Doe"),
        )
        await store.create(job)
        await executor.run(job, "test-tenant")

        sel = captured["select"]
        # Dedicated attrs/<attr> literal arm: a single BOUND predicate (no
        # alternation), the value lower-cased + escaped (case-insensitive).
        assert (
            "?e <https://cograph.tech/types/Mentor/attrs/name> ?av .\n"
            in sel
        )
        assert 'isLiteral(?av) && LCASE(STR(?av)) = "jane doe"' in sel
        # PR #37's rdfs:label arm is kept (belt-and-suspenders).
        assert "?e <http://www.w3.org/2000/01/rdf-schema#label> ?lbl ." in sel
        # The original alternation arm is preserved too.
        assert (
            "?e (<https://cograph.tech/types/Mentor/attrs/name>|"
            "<https://cograph.tech/onto/name>) ?sv ." in sel
        )
        assert "FILTER(false)" not in sel
        assert "FILTER EXISTS" not in sel

        final = await store.get(job.id)
        assert final is not None
        assert final.progress.total == 1
        assert final.progress.processed == 1

    asyncio.run(run())


def test_scope_block_literal_attribute_matches_attrs_name_value():
    """COG-112 literal-attribute fix, executed against a real SPARQL engine: a
    literal scope `name="Jane Doe"` selects the entity whose `attrs/name` literal
    is "Jane Doe" (case-insensitively), even though its `rdfs:label` is the opaque
    entity-id slug (how resolver/schema_resolver.py actually stores the data).

    Skipped when rdflib (not a project dependency) is unavailable; when present it
    proves the generated SELECT really matches, not just that the SPARQL is shaped
    right."""
    rdflib = pytest.importorskip("rdflib")
    from rdflib import Graph, Literal, RDF, URIRef
    from rdflib.namespace import XSD

    T = "https://cograph.tech/types/Mentor"
    g = Graph()
    # Entity whose name lives ONLY on attrs/name; rdfs:label is the id slug.
    e1 = URIRef("https://cograph.tech/entities/Mentor/4akvVWgTcS")
    g.add((e1, RDF.type, URIRef(T)))
    g.add((e1, URIRef("http://www.w3.org/2000/01/rdf-schema#label"), Literal("4akvVWgTcS")))
    g.add((e1, URIRef(f"{T}/attrs/name"), Literal("Jane Doe", datatype=XSD.string)))
    # A different mentor — must NOT match.
    e2 = URIRef("https://cograph.tech/entities/Mentor/xY9")
    g.add((e2, RDF.type, URIRef(T)))
    g.add((e2, URIRef(f"{T}/attrs/name"), Literal("John Smith")))

    pred_iris = [f"{T}/attrs/name", "https://cograph.tech/onto/name"]

    def _matches(value: str) -> list[str]:
        q = _build_select_query(
            "urn:g", "Mentor", ["bio"], 50,
            scope=EnrichScope(predicate="name", value=value),
            scope_pred_iris=pred_iris,
        ).replace("FROM <urn:g> ", "")  # rdflib: query the default graph
        return sorted(str(r[0]).rsplit("/", 1)[-1] for r in g.query(q))

    # Exact + case-insensitive match selects ONLY the Jane Doe entity by its
    # attrs/name literal — not the slug, not the other mentor.
    assert _matches("Jane Doe") == ["4akvVWgTcS"]
    assert _matches("jane doe") == ["4akvVWgTcS"]
    assert _matches("JANE DOE") == ["4akvVWgTcS"]
    assert _matches("John Smith") == ["xY9"]
    # A value matching nothing selects nothing (and injection-y input is escaped,
    # not executed — the query still parses and returns empty).
    assert _matches('no\"such') == []


def test_executor_entity_uris_subset_only_those_enriched():
    """entity_uris restricts the run to exactly those URIs via a VALUES block."""
    async def run():
        # Neptune returns only the requested subset for the VALUES query.
        subset_rows = [
            {"uri": "https://cograph.tech/entities/Mentor/m1", "label": "Ada", "vals": ""},
        ]
        neptune, captured = _capturing_neptune(subset_rows)
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {("Ada", "bio"): [Verdict(value="Ada bio", confidence=0.95, source="wikidata")]}
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(
            type_name="Mentor",
            attributes=["bio"],
            entity_uris=["https://cograph.tech/entities/Mentor/m1"],
        )
        await store.create(job)
        await executor.run(job, "test-tenant")

        sel = captured["select"]
        assert "VALUES ?e {" in sel
        assert "<https://cograph.tech/entities/Mentor/m1>" in sel
        assert "FILTER EXISTS" not in sel

        final = await store.get(job.id)
        assert final.progress.total == 1
        assert final.progress.processed == 1
        assert wikidata.calls == [("Ada", "bio")]

    asyncio.run(run())


def test_executor_no_scope_runs_whole_type():
    """No scope/entity_uris → the SELECT has no subset constraint (unchanged)."""
    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Mentor/m1", "label": "Ada", "vals": ""},
            {"uri": "https://cograph.tech/entities/Mentor/m2", "label": "Grace", "vals": ""},
        ]
        neptune, captured = _capturing_neptune(rows)
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata({})
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(type_name="Mentor", attributes=["bio"])
        await store.create(job)
        await executor.run(job, "test-tenant")

        sel = captured["select"]
        assert "FILTER EXISTS" not in sel
        assert "VALUES ?e" not in sel

        final = await store.get(job.id)
        assert final.progress.total == 2

    asyncio.run(run())


def test_count_entities_honors_scope_and_entity_uris():
    """count_entities applies the same subset constraints (matched count) and,
    for a scope, matches the resolved concrete predicate IRI(s) — not a scan."""
    async def run():
        neptune = AsyncMock()
        captured: dict[str, str] = {}

        async def query(sparql, *args, **kwargs):
            # Scope-predicate resolution (tenant ontology graph): return the
            # Mentor predicate declarations so `haslevel` resolves to an IRI.
            if "#domain>" in sparql and "#Property>" in sparql:
                return _ontology_predicates_response(_MENTOR_ONTO_PREDS)
            captured["q"] = sparql
            return _count_response(7)

        neptune.query.side_effect = query
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        executor = EnrichmentExecutor(neptune, store, cache, FakeWikidata({}))

        # Scope path.
        n = await executor.count_entities(
            "test-tenant", "kg", "Mentor",
            scope=EnrichScope(predicate="haslevel", value="Manager"),
        )
        assert n == 7
        # COUNT(DISTINCT ?e) over the SAME bounded DISTINCT sub-select the SELECT
        # uses — inline bound-predicate scope patterns, never a FILTER EXISTS.
        assert "FILTER EXISTS" not in captured["q"]
        assert "COUNT(DISTINCT ?e)" in captured["q"]
        assert "SELECT DISTINCT ?e WHERE {" in captured["q"]
        assert (
            "?e (<https://cograph.tech/types/Mentor/attrs/haslevel>|"
            "<https://cograph.tech/onto/haslevel>) ?sv ." in captured["q"]
        )
        assert "FILTER(?p IN (" not in captured["q"]
        assert "?e ?p ?sv" not in captured["q"]
        assert "REPLACE(STR(?p)" not in captured["q"]
        # The COUNT must NOT carry a LIMIT (it reflects the full scoped subset).
        assert "LIMIT" not in captured["q"]

        # entity_uris path (wins over scope).
        await executor.count_entities(
            "test-tenant", "kg", "Mentor",
            scope=EnrichScope(predicate="haslevel", value="Manager"),
            entity_uris=["https://cograph.tech/entities/Mentor/m1"],
        )
        assert "VALUES ?e {" in captured["q"]
        assert "FILTER EXISTS" not in captured["q"]

        # No-subset path: bare type count, no subset machinery.
        await executor.count_entities("test-tenant", "kg", "Mentor")
        assert "FILTER EXISTS" not in captured["q"]
        assert "VALUES ?e" not in captured["q"]

    asyncio.run(run())


def test_count_entities_relationship_not_in_ontology_attrs_still_counts():
    """COG-112 root-cause fix: a RELATIONSHIP predicate absent from the ontology
    attribute bindings (relationships live under `…/onto/<pred>`, not the attr
    namespace) still resolves to candidate IRIs via the direct build, so
    count_entities ISSUES the bounded COUNT (not a short-circuit to 0). Only a
    truly empty resolution would short-circuit; a valid predicate never does."""
    async def run():
        neptune = AsyncMock()
        count_calls = {"n": 0}

        async def query(sparql, *args, **kwargs):
            if "#domain>" in sparql and "#Property>" in sparql:
                # Type declares only `title` as an attribute — `haslevel`
                # (the relationship) is absent, like the live adp-mentors data.
                return _ontology_predicates_response(
                    [
                        {"attr": "https://cograph.tech/types/Mentor/attrs/title", "label": "title"},
                    ]
                )
            count_calls["n"] += 1
            # The COUNT must run over the bound-predicate property path that now
            # includes …/onto/haslevel.
            assert (
                "?e (<https://cograph.tech/types/Mentor/attrs/haslevel>|"
                "<https://cograph.tech/onto/haslevel>) ?sv ." in sparql
            )
            assert "FILTER(false)" not in sparql
            return _count_response(7)

        neptune.query.side_effect = query
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        executor = EnrichmentExecutor(neptune, store, cache, FakeWikidata({}))

        n = await executor.count_entities(
            "test-tenant", "kg", "Mentor",
            scope=EnrichScope(predicate="haslevel", value="Manager"),
        )
        assert n == 7
        # The COUNT query WAS issued (the relationship now resolves, no short-circuit).
        assert count_calls["n"] == 1

    asyncio.run(run())


def test_count_entities_scope_resolve_error_falls_back_to_direct_build():
    """A Neptune error during the ontology arm of scope-predicate resolution does
    NOT raise (create stays fast, never 500s) and does NOT collapse to matched 0:
    it skips the ontology arm and still uses the direct build (which always yields
    `…/onto/<pred>` and the attr IRI), so a relationship scope still resolves and
    the bounded COUNT is issued even when the ontology read fails (COG-112)."""
    async def run():
        neptune = AsyncMock()

        async def query(sparql, *args, **kwargs):
            if "#domain>" in sparql and "#Property>" in sparql:
                raise RuntimeError("neptune timeout")
            # The COUNT still runs over the direct-build bound-predicate path.
            assert (
                "?e (<https://cograph.tech/types/Mentor/attrs/haslevel>|"
                "<https://cograph.tech/onto/haslevel>) ?sv ." in sparql
            )
            assert "FILTER(false)" not in sparql
            return _count_response(5)

        neptune.query.side_effect = query
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        executor = EnrichmentExecutor(neptune, store, cache, FakeWikidata({}))

        n = await executor.count_entities(
            "test-tenant", "kg", "Mentor",
            scope=EnrichScope(predicate="haslevel", value="Manager"),
        )
        assert n == 5

    asyncio.run(run())


def test_apply_decisions_writes_accepted_only(monkeypatch):
    # apply_decisions now schedules a real stats recompute after a write; stub it
    # so this test stays focused on the write itself (and doesn't leave a
    # fire-and-forget recompute task draining against the AsyncMock).
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        neptune = AsyncMock()
        # Declaration reads the attribute's existing range first; none exists yet.
        neptune.query.return_value = _range_response()
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata({})
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)
        job = _make_job(policy=ConflictPolicy.stage)
        await store.create(job)

        decisions = [
            ConflictReview(
                entity_uri="https://cograph.tech/entities/Product/p1",
                attribute="manufacturer",
                existing_value="Acme",
                proposed=Verdict(value="Bosch", confidence=0.95, source="wikidata"),
                decision="accept",
            ),
            ConflictReview(
                entity_uri="https://cograph.tech/entities/Product/p2",
                attribute="manufacturer",
                existing_value="X",
                proposed=Verdict(value="Y", confidence=0.95, source="wikidata"),
                decision="reject",
            ),
        ]

        applied = await executor.apply_decisions(job.id, decisions)
        assert applied == 1
        neptune.update.assert_awaited()

    asyncio.run(run())


def test_executor_apply_schedules_stats_recompute(monkeypatch):
    """An auto-apply that writes triples must bust the Explorer summary cache by
    scheduling a stats recompute for the job's (tenant, kg)."""
    import cograph_client.api.routes.explore as explore_mod

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        explore_mod,
        "schedule_recompute",
        lambda client, tenant_id, kg_name: calls.append((tenant_id, kg_name)),
    )

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        assert final.status == JobStatus.applied

    asyncio.run(run())
    assert calls == [("test-tenant", "kg")]


def test_executor_no_apply_does_not_recompute(monkeypatch):
    """A stage job whose only result is a CONFLICT writes nothing (the conflict
    is held for review) → no recompute should be scheduled. (ONTA-159: a stage
    FILL now DOES write + recompute, so the no-write case must be a conflict.)"""
    import cograph_client.api.routes.explore as explore_mod

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        explore_mod,
        "schedule_recompute",
        lambda client, tenant_id, kg_name: calls.append((tenant_id, kg_name)),
    )

    async def run():
        mfr_pred = "https://cograph.tech/types/Product/attrs/manufacturer"
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch",
             "vals": f"{mfr_pred}::Acme Tools"},  # existing value → verdict will conflict
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        # stage + a sole conflict → held for review, nothing written.
        job = _make_job(attributes=["manufacturer"], policy=ConflictPolicy.stage)
        await store.create(job)
        await executor.run(job, "test-tenant")
        final = await store.get(job.id)
        assert final.status == JobStatus.review
        assert final.progress.conflicts == 1

    asyncio.run(run())
    assert calls == []


def test_apply_decisions_schedules_stats_recompute(monkeypatch):
    """A review-apply that accepts >=1 fact schedules a recompute for (tenant, kg)."""
    import cograph_client.api.routes.explore as explore_mod

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        explore_mod,
        "schedule_recompute",
        lambda client, tenant_id, kg_name: calls.append((tenant_id, kg_name)),
    )

    async def run():
        neptune = AsyncMock()
        # Declaration reads the attribute's existing range first; none exists yet.
        neptune.query.return_value = _range_response()
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata({})
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)
        job = _make_job(policy=ConflictPolicy.stage)
        await store.create(job)

        decisions = [
            ConflictReview(
                entity_uri="https://cograph.tech/entities/Product/p1",
                attribute="manufacturer",
                existing_value="Acme",
                proposed=Verdict(value="Bosch", confidence=0.95, source="wikidata"),
                decision="accept",
            ),
        ]
        applied = await executor.apply_decisions(job.id, decisions)
        assert applied == 1

    asyncio.run(run())
    # job's (tenant_id, kg_name) come from _make_job: "test-tenant" / "kg".
    assert calls == [("test-tenant", "kg")]


def test_apply_decisions_no_accept_does_not_recompute(monkeypatch):
    """All-reject review applies nothing → no recompute scheduled."""
    import cograph_client.api.routes.explore as explore_mod

    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        explore_mod,
        "schedule_recompute",
        lambda client, tenant_id, kg_name: calls.append((tenant_id, kg_name)),
    )

    async def run():
        neptune = AsyncMock()
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata({})
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)
        job = _make_job(policy=ConflictPolicy.stage)
        await store.create(job)

        decisions = [
            ConflictReview(
                entity_uri="https://cograph.tech/entities/Product/p2",
                attribute="manufacturer",
                existing_value="X",
                proposed=Verdict(value="Y", confidence=0.95, source="wikidata"),
                decision="reject",
            ),
        ]
        applied = await executor.apply_decisions(job.id, decisions)
        assert applied == 0

    asyncio.run(run())
    assert calls == []


# ---------------------------------------------------------------------------
# Enrichment extends the ontology (declare-then-write) — COG-112
# ---------------------------------------------------------------------------


def _declaration_updates(neptune) -> list[str]:
    """SPARQL update strings that look like an attribute ONTOLOGY declaration
    (carry ``rdf:Property`` + ``rdfs:domain``), as sent to the fake Neptune."""
    rdf_property = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
    rdfs_domain = "http://www.w3.org/2000/01/rdf-schema#domain"
    out: list[str] = []
    for call in neptune.update.await_args_list:
        sparql = call.args[0] if call.args else call.kwargs.get("sparql", "")
        if rdf_property in sparql and rdfs_domain in sparql:
            out.append(sparql)
    return out


def test_executor_apply_declares_attribute_in_ontology(monkeypatch):
    """An auto-apply that writes a value for an attribute must ALSO declare that
    attribute in the TENANT (ontology) graph — rdf:Property + rdfs:domain <Type> —
    plus its provenance companions, so the enriched attribute is first-class
    schema (COG-112)."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "company"): [
                    Verdict(
                        value="Robert Bosch GmbH",
                        confidence=0.95,
                        source="wikidata",
                        source_url="https://www.wikidata.org/wiki/Q234021",
                    )
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["company"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        assert final.status == JobStatus.applied

        decls = _declaration_updates(neptune)
        assert decls, "expected at least one ontology declaration update"

        tenant_graph = "https://cograph.tech/graphs/test-tenant"
        company_attr = "https://cograph.tech/types/Product/attrs/company"
        type_uri = "https://cograph.tech/types/Product"
        xsd_string = "http://www.w3.org/2001/XMLSchema#string"

        # The primary 'company' attribute is declared into the TENANT graph as an
        # rdf:Property with rdfs:domain <Product> and an xsd:string range.
        primary = [d for d in decls if company_attr in d]
        assert primary, "company attribute not declared"
        d = primary[0]
        assert tenant_graph in d
        assert type_uri in d
        assert xsd_string in d

        # Companion provenance attrs were written for this value (source_url +
        # provenance) → they must be declared too, into the same tenant graph.
        src_attr = "https://cograph.tech/types/Product/attrs/company_source_url"
        prov_attr = "https://cograph.tech/types/Product/attrs/company_provenance"
        assert any(src_attr in d for d in decls), "company_source_url not declared"
        assert any(prov_attr in d for d in decls), "company_provenance not declared"
        for d in decls:
            assert tenant_graph in d  # declarations go to the ontology graph only

    asyncio.run(run())


def test_executor_stage_mode_does_not_declare(monkeypatch):
    """A stage job whose only result is a CONFLICT writes nothing yet → it must
    NOT declare attributes in the ontology. (ONTA-159: a stage FILL now writes +
    declares, so the no-declare case must be a held conflict.)"""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        company_pred = "https://cograph.tech/types/Product/attrs/company"
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch",
             "vals": f"{company_pred}::Old Holdings"},  # existing → verdict conflicts
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "company"): [
                    Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["company"], policy=ConflictPolicy.stage)
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        assert final.status == JobStatus.review
        # The conflict is held, not written → nothing written, nothing declared.
        assert _declaration_updates(neptune) == []
        neptune.update.assert_not_called()

    asyncio.run(run())


def test_executor_no_match_does_not_declare(monkeypatch):
    """An attribute that found no value contributes no triples → it must NOT be
    declared (no ontology pollution with empty slots)."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Unknown", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata({})  # no verdicts → no_match
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["company"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        assert _declaration_updates(neptune) == []

    asyncio.run(run())


def test_apply_decisions_declares_accepted_attribute(monkeypatch):
    """Accepting a review decision also extends the ontology (declares the
    accepted attribute + its provenance companions in the tenant graph)."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        neptune = AsyncMock()
        neptune.update.return_value = None
        # Declaration reads the attribute's existing range first; none exists yet.
        neptune.query.return_value = _range_response()
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata({})
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)
        job = _make_job(attributes=["company"], policy=ConflictPolicy.stage)
        await store.create(job)

        decisions = [
            ConflictReview(
                entity_uri="https://cograph.tech/entities/Product/p1",
                attribute="company",
                existing_value="Acme",
                proposed=Verdict(
                    value="Robert Bosch GmbH",
                    confidence=0.95,
                    source="wikidata",
                    source_url="https://www.wikidata.org/wiki/Q234021",
                ),
                decision="accept",
            ),
        ]
        applied = await executor.apply_decisions(job.id, decisions)
        assert applied == 1

        decls = _declaration_updates(neptune)
        company_attr = "https://cograph.tech/types/Product/attrs/company"
        tenant_graph = "https://cograph.tech/graphs/test-tenant"
        assert any(company_attr in d for d in decls)
        assert all(tenant_graph in d for d in decls)

    asyncio.run(run())


def _query_router(entities: dict, *, existing_range: str | None = None):
    """An AsyncMock ``query`` side_effect that serves the entity-selection SELECT
    and the per-attribute range SELECT (``SELECT ?range``) from one fake Neptune.
    ``existing_range`` is the range the ontology already declares for the enriched
    attribute (None = undeclared)."""

    async def _route(sparql: str, *a, **k):
        if "SELECT ?range" in sparql:
            return _range_response(existing_range)
        return entities

    return _route


def test_executor_apply_infers_integer_range_for_numeric_values(monkeypatch):
    """A brand-new enriched attribute whose applied values are all numeric must be
    declared with an xsd:integer range — NOT blindly stamped xsd:string (the
    hardcoded-datatype bug)."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
            {"uri": "https://cograph.tech/entities/Product/p2", "label": "Makita", "vals": ""},
        ]
        neptune = AsyncMock()
        # No existing range for 'humanness_score' → inference decides the range.
        neptune.query.side_effect = _query_router(
            _entities_query_response(rows), existing_range=None
        )
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "humanness_score"): [
                    Verdict(value="92", confidence=0.95, source="wikidata")
                ],
                ("Makita", "humanness_score"): [
                    Verdict(value="87", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["humanness_score"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        score_attr = "https://cograph.tech/types/Product/attrs/humanness_score"
        xsd_integer = "http://www.w3.org/2001/XMLSchema#integer"
        xsd_string = "http://www.w3.org/2001/XMLSchema#string"
        primary = [d for d in _declaration_updates(neptune) if score_attr in d]
        assert primary, "humanness_score attribute not declared"
        d = primary[0]
        # The numeric attribute is typed as integer, NOT stamped string.
        assert xsd_integer in d
        assert xsd_string not in d

    asyncio.run(run())


def test_executor_apply_does_not_downgrade_existing_richer_range(monkeypatch):
    """If an attribute already carries a richer range (an ingest-inferred
    xsd:integer, or a relationship types/<Target> URI), applying an enrichment job
    on it must PRESERVE that range — never silently downgrade it to xsd:string."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def _assert_preserves(existing_range: str, value: str):
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.side_effect = _query_router(
            _entities_query_response(rows), existing_range=existing_range
        )
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {("Bosch", "rating"): [Verdict(value=value, confidence=0.95, source="wikidata")]}
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["rating"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        rating_attr = "https://cograph.tech/types/Product/attrs/rating"
        xsd_string = "http://www.w3.org/2001/XMLSchema#string"
        primary = [d for d in _declaration_updates(neptune) if rating_attr in d]
        assert primary, "rating attribute not declared"
        d = primary[0]
        # The pre-existing richer range is re-asserted verbatim, never downgraded.
        assert existing_range in d
        assert f"#range> <{xsd_string}>" not in d

    async def run():
        # (a) ingest-inferred integer range survives a string-valued enrichment.
        await _assert_preserves(
            "http://www.w3.org/2001/XMLSchema#integer", value="five stars"
        )
        # (b) a relationship range (types/<Target>) survives too — the edge stays.
        await _assert_preserves(
            "https://cograph.tech/types/Manufacturer", value="Robert Bosch GmbH"
        )

    asyncio.run(run())


# ---------------------------------------------------------------------------
# E2: date + entity-reference (relationship) datatype inference
# ---------------------------------------------------------------------------


def test_infer_datatype_dates_to_datetime():
    """ISO dates / datetimes (plain, Z-suffixed, +00:00) all infer ``datetime``."""
    assert _infer_datatype_from_values(["2026-06-28"]) == "datetime"
    assert (
        _infer_datatype_from_values(["2026-06-28", "2026-06-28T00:00:00Z"])
        == "datetime"
    )
    assert (
        _infer_datatype_from_values(["2026-06-28T21:24:50+00:00"]) == "datetime"
    )


def test_infer_datatype_integer_not_misread_as_date():
    """An all-integer column must stay ``integer`` — never a date false-positive,
    even though a bare year-like int contains no separator."""
    assert _infer_datatype_from_values(["2026", "1999", "42"]) == "integer"


def test_infer_datatype_entity_iris_same_type_is_relationship():
    """All values are entity IRIs sharing one ``<TypeName>`` → that bare type name
    (which maps to a ``types/<TypeName>`` relationship range)."""
    vals = [
        "https://cograph.tech/entities/Manufacturer/bosch",
        "https://cograph.tech/entities/Manufacturer/makita",
    ]
    assert _infer_datatype_from_values(vals) == "Manufacturer"


def test_infer_datatype_mixed_iri_types_falls_back_to_string():
    """Entity IRIs of DIFFERENT types have no single relationship range → string
    (don't guess)."""
    vals = [
        "https://cograph.tech/entities/Manufacturer/bosch",
        "https://cograph.tech/entities/Country/germany",
    ]
    assert _infer_datatype_from_values(vals) == "string"


def test_entity_iri_type_parses_and_rejects():
    """``_entity_iri_type`` extracts the type from a canonical entity IRI and
    returns None for non-matching values (literal, foreign URI, missing id)."""
    assert (
        _entity_iri_type("https://cograph.tech/entities/Manufacturer/bosch")
        == "Manufacturer"
    )
    assert _entity_iri_type("Robert Bosch GmbH") is None
    assert _entity_iri_type("https://cograph.tech/types/Manufacturer") is None
    # Missing <id> segment is not a complete entity IRI.
    assert _entity_iri_type("https://cograph.tech/entities/Manufacturer") is None
    assert _entity_iri_type("https://cograph.tech/entities/Manufacturer/") is None


def _instance_inserts(neptune) -> list[str]:
    """SPARQL update strings that look like an INSTANCE-data write (carry the
    entity-IRI prefix as a SUBJECT, i.e. the ``insert_facts`` triples) but are NOT
    ontology declarations (no ``rdf:Property`` + ``rdfs:domain``)."""
    rdf_property = "http://www.w3.org/1999/02/22-rdf-syntax-ns#Property"
    rdfs_domain = "http://www.w3.org/2000/01/rdf-schema#domain"
    out: list[str] = []
    for call in neptune.update.await_args_list:
        sparql = call.args[0] if call.args else call.kwargs.get("sparql", "")
        is_decl = rdf_property in sparql and rdfs_domain in sparql
        if "https://cograph.tech/entities/" in sparql and not is_decl:
            out.append(sparql)
    return out


def test_executor_apply_infers_datetime_range_for_date_values(monkeypatch):
    """A brand-new enriched attribute whose applied values are all ISO dates must
    be declared with an xsd:dateTime range — NOT stamped xsd:string."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
            {"uri": "https://cograph.tech/entities/Product/p2", "label": "Makita", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.side_effect = _query_router(
            _entities_query_response(rows), existing_range=None
        )
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "founded"): [
                    Verdict(value="2026-06-28", confidence=0.95, source="wikidata")
                ],
                ("Makita", "founded"): [
                    Verdict(
                        value="2026-06-28T00:00:00Z", confidence=0.95, source="wikidata"
                    )
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["founded"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        founded_attr = "https://cograph.tech/types/Product/attrs/founded"
        xsd_datetime = "http://www.w3.org/2001/XMLSchema#dateTime"
        xsd_string = "http://www.w3.org/2001/XMLSchema#string"
        primary = [d for d in _declaration_updates(neptune) if founded_attr in d]
        assert primary, "founded attribute not declared"
        d = primary[0]
        assert xsd_datetime in d
        assert f"#range> <{xsd_string}>" not in d

    asyncio.run(run())


def test_executor_apply_entity_iri_values_declare_relationship_and_write_iri(monkeypatch):
    """When all applied values are entity IRIs of one type ``Manufacturer``, the
    attribute is declared with a ``types/Manufacturer`` relationship range AND the
    instance triple's OBJECT is written as an IRI ``<…/entities/…>`` (the shared
    writer auto-detects ``https://`` objects), never a quoted literal."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Drill", "vals": ""},
            {"uri": "https://cograph.tech/entities/Product/p2", "label": "Saw", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.side_effect = _query_router(
            _entities_query_response(rows), existing_range=None
        )
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Drill", "manufacturer"): [
                    Verdict(
                        value="https://cograph.tech/entities/Manufacturer/bosch",
                        confidence=0.95,
                        source="wikidata",
                    )
                ],
                ("Saw", "manufacturer"): [
                    Verdict(
                        value="https://cograph.tech/entities/Manufacturer/makita",
                        confidence=0.95,
                        source="wikidata",
                    )
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["manufacturer"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        mfr_attr = "https://cograph.tech/types/Product/attrs/manufacturer"
        types_mfr = "https://cograph.tech/types/Manufacturer"
        # (a) declared as a relationship range types/Manufacturer, not xsd:string.
        primary = [d for d in _declaration_updates(neptune) if mfr_attr in d]
        assert primary, "manufacturer attribute not declared"
        d = primary[0]
        assert f"#range> <{types_mfr}>" in d
        assert "#range> <http://www.w3.org/2001/XMLSchema#string>" not in d

        # (b) the instance object is written as an IRI, not a quoted literal.
        inserts = _instance_inserts(neptune)
        joined = "\n".join(inserts)
        assert "<https://cograph.tech/entities/Manufacturer/bosch>" in joined
        assert '"https://cograph.tech/entities/Manufacturer/bosch"' not in joined

    asyncio.run(run())


def test_executor_apply_does_not_downgrade_datetime_or_relationship_range(monkeypatch):
    """No-downgrade holds for the E2 ranges too: an existing xsd:dateTime or a
    relationship types/<Target> range survives an enrichment whose own values
    would infer something weaker."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def _assert_preserves(existing_range: str, value: str):
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.side_effect = _query_router(
            _entities_query_response(rows), existing_range=existing_range
        )
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {("Bosch", "founded"): [Verdict(value=value, confidence=0.95, source="wikidata")]}
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["founded"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        founded_attr = "https://cograph.tech/types/Product/attrs/founded"
        xsd_string = "http://www.w3.org/2001/XMLSchema#string"
        primary = [d for d in _declaration_updates(neptune) if founded_attr in d]
        assert primary, "founded attribute not declared"
        d = primary[0]
        assert existing_range in d
        assert f"#range> <{xsd_string}>" not in d

    async def run():
        # An existing xsd:dateTime survives a free-text enrichment value.
        await _assert_preserves(
            "http://www.w3.org/2001/XMLSchema#dateTime", value="sometime in 2026"
        )
        # An existing relationship range survives a string-valued enrichment.
        await _assert_preserves(
            "https://cograph.tech/types/Manufacturer", value="Robert Bosch GmbH"
        )

    asyncio.run(run())


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singletons():
    from cograph_client.enrichment.cache import reset_enrichment_cache
    from cograph_client.enrichment.job_store import reset_job_store

    reset_job_store()
    reset_enrichment_cache()
    yield
    reset_job_store()
    reset_enrichment_cache()


def test_post_jobs_returns_job_id(client, auth_headers, mock_neptune):
    # The executor's background run loop may issue queries once spawned; we don't
    # care about its outcome here (create itself no longer counts entities).
    mock_neptune.query.return_value = _count_response(0)

    response = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Product",
            "attributes": ["manufacturer"],
            "kg_name": "kg",
            "tier": "lite",
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    # Non-blocking create (COG-112): matched count is resolved by the background
    # executor (job.progress.total), not at create time, so it is None here.
    assert data["matched_entities"] is None


def test_post_jobs_holds_strong_ref_to_background_task(
    client, auth_headers, mock_neptune, monkeypatch
):
    """COG-112 regression guard: the create path must keep a *strong* reference to
    the spawned executor task. A bare ``asyncio.create_task(...)`` is only
    weak-referenced by the loop and gets GC'd at the first await after the request
    returns — stranding the job right after it selects entities. We capture the
    coroutine handed to the executor and assert create routes it through the
    module-level ``_spawn`` helper (which registers it in ``_bg_tasks``), never as
    a bare task."""
    import cograph_client.api.routes.enrich as enrich_mod

    captured: list = []
    real_spawn = enrich_mod._spawn

    def _tracking_spawn(coro):
        captured.append(coro)
        real_spawn(coro)
        # Right after scheduling, the task must be held by the module set so it
        # cannot be garbage-collected mid-run.
        assert len(enrich_mod._bg_tasks) >= 1

    monkeypatch.setattr(enrich_mod, "_spawn", _tracking_spawn)
    mock_neptune.query.return_value = _count_response(0)

    response = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Product",
            "attributes": ["manufacturer"],
            "kg_name": "kg",
        },
    )
    assert response.status_code == 202
    # create scheduled exactly one background task via the strong-ref helper.
    assert len(captured) == 1


def test_post_jobs_with_scope_threads_scope_without_blocking(
    client, auth_headers, mock_neptune
):
    """A scoped create-job (COG-112): create is NON-BLOCKING — it does NOT call
    count_entities in the request path — so it can never time out on a slow
    scoped COUNT. The stored job persists the scope so the background executor
    resolves it and surfaces the matched count via progress.total."""

    async def _query(sparql, *args, **kwargs):
        # The executor's background run may issue queries once spawned; create
        # itself must not. Return a harmless shape for any background query.
        if "#domain>" in sparql and "#Property>" in sparql:
            return _ontology_predicates_response(_MENTOR_ONTO_PREDS)
        return _count_response(2)

    mock_neptune.query.side_effect = _query

    response = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Mentor",
            "attributes": ["bio"],
            "kg_name": "kg",
            "tier": "lite",
            "scope": {"predicate": "haslevel", "value": "Manager"},
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    # Matched count is resolved by the background executor, not at create time.
    assert data["matched_entities"] is None

    # The stored job retains the scope (full-job view) so the executor uses it.
    job = client.get(
        f"/graphs/test-tenant/enrich/jobs/{data['job_id']}", headers=auth_headers
    ).json()
    assert job["scope"] == {"predicate": "haslevel", "value": "Manager"}
    assert job["entity_uris"] is None


def test_post_jobs_does_not_block_on_count_entities(
    client, auth_headers, mock_neptune, monkeypatch
):
    """The create path must NOT await count_entities (COG-112 non-blocking
    guarantee): even if count_entities hangs/raises, create still returns a job
    id promptly. We monkeypatch the executor's count_entities to blow up if
    called and assert create succeeds without invoking it."""
    from cograph_client.enrichment import executor as executor_mod

    async def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("count_entities must not be called in create path")

    monkeypatch.setattr(
        executor_mod.EnrichmentExecutor, "count_entities", _boom
    )
    mock_neptune.query.return_value = _count_response(0)

    response = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Mentor",
            "attributes": ["bio"],
            "kg_name": "kg",
            "scope": {"predicate": "haslevel", "value": "Manager"},
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_post_jobs_with_entity_uris_subset(client, auth_headers, mock_neptune):
    """entity_uris on create-job persists the explicit subset; create is
    non-blocking so it does not count the subset up front (matched_entities is
    resolved later by the executor)."""
    mock_neptune.query.return_value = _count_response(1)
    uris = ["https://cograph.tech/entities/Mentor/m1"]
    response = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Mentor",
            "attributes": ["bio"],
            "kg_name": "kg",
            "entity_uris": uris,
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert data["matched_entities"] is None
    job = client.get(
        f"/graphs/test-tenant/enrich/jobs/{data['job_id']}", headers=auth_headers
    ).json()
    assert job["entity_uris"] == uris


def test_get_jobs_lists_jobs(client, auth_headers, mock_neptune):
    mock_neptune.query.return_value = _count_response(0)
    r = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Product",
            "attributes": ["manufacturer"],
            "kg_name": "kg",
        },
    )
    assert r.status_code == 202
    job_id = r.json()["job_id"]

    listing = client.get(
        "/graphs/test-tenant/enrich/jobs", headers=auth_headers
    )
    assert listing.status_code == 200
    rows = listing.json()
    ids = [j["id"] for j in rows]
    assert job_id in ids


def test_get_job_404(client, auth_headers, mock_neptune):
    response = client.get(
        "/graphs/test-tenant/enrich/jobs/does-not-exist", headers=auth_headers
    )
    assert response.status_code == 404


def test_conflicts_and_apply_flow(client, auth_headers, mock_neptune):
    """Seed a job directly, set a conflict result, then call /conflicts and /apply."""
    from cograph_client.enrichment.job_store import get_job_store
    from cograph_client.enrichment.models import RowResult

    job = _make_job(policy=ConflictPolicy.stage)
    job.tenant_id = "test-tenant"
    job.status = JobStatus.review
    verdict = Verdict(value="Bosch", confidence=0.95, source="wikidata")
    job.results = [
        RowResult(
            entity_uri="https://cograph.tech/entities/Product/p1",
            attribute="manufacturer",
            existing_value="Acme",
            verdict=verdict,
            action="conflict",
        )
    ]

    async def _seed():
        store = get_job_store()
        await store.create(job)

    asyncio.run(_seed())

    r = client.get(
        f"/graphs/test-tenant/enrich/jobs/{job.id}/conflicts", headers=auth_headers
    )
    assert r.status_code == 200
    conflicts = r.json()
    assert len(conflicts) == 1
    assert conflicts[0]["entity_uri"].endswith("/p1")

    apply_resp = client.post(
        f"/graphs/test-tenant/enrich/jobs/{job.id}/apply",
        headers=auth_headers,
        json={
            "decisions": [
                {
                    "entity_uri": "https://cograph.tech/entities/Product/p1",
                    "attribute": "manufacturer",
                    "existing_value": "Acme",
                    "proposed": verdict.model_dump(),
                    "decision": "accept",
                }
            ]
        },
    )
    assert apply_resp.status_code == 200
    assert apply_resp.json()["applied"] == 1
    assert mock_neptune.update.await_count >= 1


def test_cancel_job(client, auth_headers, mock_neptune):
    mock_neptune.query.return_value = _count_response(0)
    r = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Product",
            "attributes": ["manufacturer"],
            "kg_name": "kg",
        },
    )
    job_id = r.json()["job_id"]
    cancel = client.delete(
        f"/graphs/test-tenant/enrich/jobs/{job_id}", headers=auth_headers
    )
    assert cancel.status_code == 200
    assert cancel.json()["status"] == "cancelled"


# ---------------------------------------------------------------------------
# Tier registry
# ---------------------------------------------------------------------------


def test_adapter_cost_metadata_protocol():
    """COG-123: the OSS cost model reads is_paid/cost_per_call generically.

    - An adapter that declares neither (legacy / OSS Wikidata) is free.
    - is_paid OR a positive cost_per_call marks an adapter paid.
    - The cograph-shipped WikidataAdapter declares free explicitly.
    - A malformed cost_per_call coerces to 0.0 (never raises).
    """
    from cograph_client.enrichment.sources.base import adapter_cost
    from cograph_client.enrichment.sources.wikidata import WikidataAdapter

    class _Bare:  # declares nothing → free
        name = "bare"

    class _PaidFlagOnly:  # is_paid True, no cost → paid, $0
        name = "p1"
        is_paid = True
        cost_per_call = 0.0

    class _CostOnly:  # positive cost ⇒ paid even without is_paid
        name = "p2"
        cost_per_call = 0.02

    class _Malformed:
        name = "bad"
        is_paid = True
        cost_per_call = "not-a-number"

    assert adapter_cost(_Bare()) == (False, 0.0)
    assert adapter_cost(_PaidFlagOnly()) == (True, 0.0)
    assert adapter_cost(_CostOnly()) == (True, 0.02)
    # Malformed cost coerces to 0.0 but is_paid flag is honored.
    assert adapter_cost(_Malformed()) == (True, 0.0)
    # The OSS Wikidata adapter is free.
    assert adapter_cost(WikidataAdapter()) == (False, 0.0)


def test_register_tier_and_get_chain():
    from cograph_client.enrichment.tiers import (
        get_chain,
        register_tier,
        reset_tiers,
    )

    reset_tiers()
    try:
        assert get_chain(EnrichmentTier.lite) == ["wikidata"]
        register_tier(EnrichmentTier.base, ["wikidata", "web"])
        assert get_chain(EnrichmentTier.base) == ["wikidata", "web"]
        # Idempotent: last write wins.
        register_tier(EnrichmentTier.base, ["wikidata"])
        assert get_chain(EnrichmentTier.base) == ["wikidata"]
        # Returned list is a copy: mutating it does not affect the registry.
        chain = get_chain(EnrichmentTier.lite)
        chain.append("mutated")
        assert get_chain(EnrichmentTier.lite) == ["wikidata"]
    finally:
        reset_tiers()


def test_executor_skips_unregistered_adapter(caplog):
    """Chain with a missing adapter name should log a warning and not fail."""
    import logging

    from cograph_client.enrichment.tiers import (
        get_chain,
        register_tier,
        reset_tiers,
    )

    async def run():
        rows = [
            {
                "uri": "https://cograph.tech/entities/Product/p1",
                "label": "Bosch",
                "vals": "",
            },
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(
                        value="Robert Bosch GmbH",
                        confidence=0.95,
                        source="wikidata",
                    )
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        register_tier(EnrichmentTier.lite, ["wikidata", "nonexistent"])
        assert get_chain(EnrichmentTier.lite) == ["wikidata", "nonexistent"]

        job = _make_job(policy=ConflictPolicy.stage)
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        # Job did not fail.
        assert final is not None
        assert final.status != JobStatus.failed
        # Wikidata produced a verdict, so the job filled the empty slot.
        assert final.progress.filled == 1

    reset_tiers()
    caplog.set_level(logging.WARNING)
    try:
        asyncio.run(run())
    finally:
        reset_tiers()


# ---------------------------------------------------------------------------
# Strategy loader
# ---------------------------------------------------------------------------


def _strategy_query_response(rows: list[dict]) -> dict:
    """Build a SPARQL response for the strategy SELECT.

    rows: list of {"subj": uri, "p": uri, "o": value}
    """
    bindings = []
    for r in rows:
        b = {
            "subj": {"type": "uri", "value": r["subj"]},
            "p": {"type": "uri", "value": r["p"]},
            "o": {"type": "literal", "value": r["o"]},
        }
        bindings.append(b)
    return {
        "head": {"vars": ["subj", "p", "o"]},
        "results": {"bindings": bindings},
    }


def test_load_strategy_returns_empty_when_no_triples():
    from cograph_client.enrichment.strategy import load_strategy

    async def run():
        neptune = AsyncMock()
        neptune.query.return_value = _strategy_query_response([])
        s = await load_strategy(neptune, "test-tenant", "LineItem")
        assert s.type_name == "LineItem"
        assert s.match_key is None
        assert s.lookup_priority is None
        assert s.attributes == {}

    asyncio.run(run())


def test_load_strategy_parses_attribute_triples():
    from cograph_client.enrichment.strategy import load_strategy

    type_uri = "https://cograph.tech/types/LineItem"
    mpn_uri = "https://cograph.tech/types/LineItem/attrs/mpn"
    brand_uri = "https://cograph.tech/types/LineItem/attrs/brand"
    onto = "https://cograph.tech/onto"

    async def run():
        neptune = AsyncMock()
        neptune.query.return_value = _strategy_query_response(
            [
                {"subj": type_uri, "p": f"{onto}/matchKey", "o": "description"},
                {"subj": type_uri, "p": f"{onto}/lookupPriority", "o": "1"},
                {"subj": mpn_uri, "p": f"{onto}/enrichmentSource", "o": "wikidata"},
                {"subj": mpn_uri, "p": f"{onto}/enrichmentSource", "o": "web"},
                {"subj": mpn_uri, "p": f"{onto}/confidenceMin", "o": "0.9"},
                {"subj": mpn_uri, "p": f"{onto}/idPattern", "o": "^[A-Z0-9-]{6,20}$"},
                {"subj": mpn_uri, "p": f"{onto}/conflictPolicy", "o": "stage"},
                {"subj": brand_uri, "p": f"{onto}/canonicalizer", "o": "title-case"},
                {"subj": brand_uri, "p": f"{onto}/alias", "o": "KN→K&N"},
                {"subj": brand_uri, "p": f"{onto}/alias", "o": "Mfg→Manufacturing"},
                # Malformed alias should be silently skipped.
                {"subj": brand_uri, "p": f"{onto}/alias", "o": "bogus-no-arrow"},
            ]
        )
        s = await load_strategy(neptune, "test-tenant", "LineItem")
        assert s.match_key == "description"
        assert s.lookup_priority == 1
        assert "mpn" in s.attributes
        mpn = s.attributes["mpn"]
        assert mpn.sources == ["wikidata", "web"]
        assert mpn.confidence_min == 0.9
        assert mpn.id_pattern == "^[A-Z0-9-]{6,20}$"
        assert mpn.conflict_policy == "stage"
        brand = s.attributes["brand"]
        assert brand.canonicalizer == "title-case"
        assert brand.aliases == {"KN": "K&N", "Mfg": "Manufacturing"}

    asyncio.run(run())


def test_aliases_resolve_conflicts_to_verified():
    """Existing brand=KN, alias KN->K&N, verdict K&N -> verified, not conflict."""
    from cograph_client.enrichment.tiers import reset_tiers

    type_uri = "https://cograph.tech/types/Product"
    brand_uri = "https://cograph.tech/types/Product/attrs/brand"
    onto = "https://cograph.tech/onto"
    brand_pred = brand_uri  # the predicate stored on the entity row

    async def run():
        rows = [
            {
                "uri": "https://cograph.tech/entities/Product/p1",
                "label": "Filter",
                "vals": f"{brand_pred}::KN",
            },
        ]
        neptune = AsyncMock()

        async def query(sparql, *args, **kwargs):
            # First call inside run() is the strategy load (tenant graph URI),
            # subsequent calls are the entity SELECT.
            if "FROM <https://cograph.tech/graphs/test-tenant>" in sparql and "alias" in sparql:
                return _strategy_query_response(
                    [
                        {"subj": brand_uri, "p": f"{onto}/alias", "o": "KN→K&N"},
                    ]
                )
            return _entities_query_response(rows)

        neptune.query.side_effect = query
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Filter", "brand"): [
                    Verdict(value="K&N", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        reset_tiers()
        job = _make_job(
            type_name="Product",
            attributes=["brand"],
            policy=ConflictPolicy.stage,
        )
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        assert final is not None
        # ONTA-159: the alias resolved the would-be conflict to VERIFIED, so there
        # is no conflict left to review → the run completes APPLIED (nothing to do).
        assert final.status == JobStatus.applied
        assert final.progress.verified == 1, (
            f"expected verified, got progress={final.progress}"
        )
        assert final.progress.conflicts == 0

    reset_tiers()
    try:
        asyncio.run(run())
    finally:
        reset_tiers()


def test_canonicalize_title_case_handles_ampersand():
    from cograph_client.enrichment.canonicalize import apply_canonicalizer

    assert apply_canonicalizer("title-case", "k&n filters") == "K&N Filters"
    assert apply_canonicalizer("title-case", "AT&T") == "AT&T"
    assert apply_canonicalizer("title-case", "  bosch  gmbh  ").strip() == "Bosch Gmbh"
    # Unknown canonicalizer returns value unchanged.
    assert apply_canonicalizer("nope", "anything") == "anything"
    assert apply_canonicalizer(None, "x") == "x"
    assert apply_canonicalizer("trim", "  hi  ") == "hi"


def test_enrichment_plugin_loaded_at_startup(monkeypatch):
    """Plugin's register() runs during create_app()."""
    from cograph_client.api import app as app_module
    from cograph_client.config import settings

    monkeypatch.setattr(
        settings, "enrichment_plugin", "tests.fake_enrichment_plugin:register"
    )
    try:
        app_module.create_app()
        from tests import fake_enrichment_plugin

        assert fake_enrichment_plugin.LOADED is True
    finally:
        from tests import fake_enrichment_plugin

        fake_enrichment_plugin.LOADED = False


def test_enrichment_plugin_invalid_format_logged(monkeypatch):
    """Malformed plugin spec is logged but does not raise."""
    from cograph_client.api import app as app_module
    from cograph_client.config import settings

    monkeypatch.setattr(settings, "enrichment_plugin", "no_colon_here")
    # Must not raise.
    app_module.create_app()


# ---------------------------------------------------------------------------
# Verdict provenance contract (ADR-0005 §5)
# ---------------------------------------------------------------------------


def test_verdict_backcompat_and_provenance():
    # (1) Legacy construction still works; new fields default to None.
    legacy = Verdict(value="Bosch GmbH", confidence=0.95, source="wikidata")
    assert legacy.value == "Bosch GmbH"
    assert legacy.confidence == 0.95
    assert legacy.source == "wikidata"
    assert legacy.raw_confidence is None
    assert legacy.retrieved_at is None
    assert legacy.source_published_at is None
    assert legacy.grounding_score is None
    assert legacy.extraction_method is None
    assert legacy.calibration_method is None

    # (2) A fully-populated verdict round-trips through model_dump/model_validate.
    retrieved = datetime(2026, 6, 17, 12, 0, 0, tzinfo=timezone.utc)
    published = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    full = Verdict(
        value="Robert Bosch GmbH",
        confidence=0.91,
        source="exa",
        source_url="https://example.com/bosch",
        reasoning="matched on company registry",
        raw_confidence=0.42,
        retrieved_at=retrieved,
        source_published_at=published,
        grounding_score=0.88,
        extraction_method="llm-extract",
        calibration_method="isotonic",
    )
    dumped = full.model_dump()
    restored = Verdict.model_validate(dumped)
    assert restored == full
    assert restored.raw_confidence == 0.42
    assert restored.retrieved_at == retrieved
    assert restored.source_published_at == published
    assert restored.grounding_score == 0.88
    assert restored.extraction_method == "llm-extract"
    assert restored.calibration_method == "isotonic"


# ---------------------------------------------------------------------------
# COG-124: smart "auto" tier resolution + web confidence floor on create
# ---------------------------------------------------------------------------


class _FakePaidAdapter:
    """A registered PAID adapter so the ``core`` chain reads as paid (COG-123:
    paid is detected from declared metadata, never the adapter name)."""

    name = "fakepaid"
    is_paid = True
    cost_per_call = 0.01

    async def lookup(self, *args, **kwargs):  # pragma: no cover - never called
        return []


@pytest.fixture
def _paid_core_chain():
    """Register a paid adapter and point the ``core`` tier chain at it, so
    ``chain_has_paid(core)`` is True in this OSS test. Restores defaults after."""
    from cograph_client.enrichment.sources import base as base_mod
    from cograph_client.enrichment.tiers import register_tier, reset_tiers

    adapter = _FakePaidAdapter()
    base_mod.register_adapter(adapter)
    register_tier(EnrichmentTier.core, ["fakepaid"])
    try:
        yield
    finally:
        reset_tiers()
        base_mod._adapters.pop("fakepaid", None)


def test_resolve_auto_tier_no_key_falls_back_to_heuristic():
    """With no openrouter_key the resolver must NOT raise, must pick a concrete
    tier (never needs_clarification), and must set a routing_note."""
    from cograph_client.enrichment.tier_router import resolve_auto_tier

    async def run():
        # Open-web fact → core, leaning paid.
        web = await resolve_auto_tier(["company", "website"], "Person", None)
        assert web.resolved_tier == "core"
        assert web.needs_clarification is False
        assert web.routing_note  # non-empty explanation

        # Structured identifier → lite (free Wikidata).
        structured = await resolve_auto_tier(["iso_code"], "Country", None)
        assert structured.resolved_tier == "lite"
        assert structured.needs_clarification is False
        assert structured.routing_note

        # Unknown attribute with no clear structured signal → lean paid (core).
        unknown = await resolve_auto_tier(["vibe"], "Person", None)
        assert unknown.resolved_tier == "core"
        assert unknown.needs_clarification is False

        # Empty key string is treated as "no key" too — still concrete, no raise.
        empty = await resolve_auto_tier(["company"], "Person", "")
        assert empty.resolved_tier in ("lite", "core")
        assert empty.needs_clarification is False

    asyncio.run(run())


def test_create_job_auto_needs_clarification_creates_no_job(
    client, auth_headers, mock_neptune, monkeypatch
):
    """When the auto resolver returns needs_clarification, create returns status
    'needs_clarification' with candidates and creates NO job (the store stays
    empty / the executor is never spawned)."""
    import cograph_client.api.routes.enrich as enrich_mod
    from cograph_client.enrichment.tier_router import TierDecision

    async def _ambiguous(attributes, type_name, openrouter_key, timeout_s=8.0):
        return TierDecision(
            resolved_tier=None,
            needs_clarification=True,
            candidates=["lite", "core"],
            routing_note="ambiguous — pick a tier",
        )

    monkeypatch.setattr(enrich_mod, "resolve_auto_tier", _ambiguous)

    # Guard: the executor must never be spawned on the clarification branch.
    spawned: list = []
    monkeypatch.setattr(enrich_mod, "_spawn", lambda coro: spawned.append(coro))

    response = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Person",
            "attributes": ["company"],
            "kg_name": "kg",
            "tier": "auto",
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "needs_clarification"
    assert data["needs_clarification"] is True
    assert data["candidates"] == ["lite", "core"]
    assert data["job_id"] is None
    assert data["resolved_tier"] is None
    assert data["routing_note"] == "ambiguous — pick a tier"
    # NO job created, NO background work spawned.
    assert spawned == []
    listing = client.get(
        "/graphs/test-tenant/jobs?category=enrichment", headers=auth_headers
    ).json()
    assert listing == []


def test_create_job_auto_resolves_core_lowers_confidence_to_web_floor(
    client, auth_headers, mock_neptune, monkeypatch, _paid_core_chain
):
    """auto → core (paid) with the DEFAULT confidence must create a job at core
    AND lower confidence_min to the web floor (0.4) so web verdicts actually
    land instead of all being filtered out."""
    import cograph_client.api.routes.enrich as enrich_mod
    from cograph_client.enrichment.tier_router import TierDecision

    async def _to_core(attributes, type_name, openrouter_key, timeout_s=8.0):
        return TierDecision(
            resolved_tier="core",
            needs_clarification=False,
            routing_note="auto-routed to core",
        )

    monkeypatch.setattr(enrich_mod, "resolve_auto_tier", _to_core)
    mock_neptune.query.return_value = _count_response(0)

    response = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Person",
            "attributes": ["company"],
            "kg_name": "kg",
            "tier": "auto",
            # confidence_min omitted → default 0.85 sentinel → floor applies.
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert data["status"] == "queued"
    assert data["resolved_tier"] == "core"
    assert data["needs_clarification"] is False
    assert data["candidates"] is None
    assert data["routing_note"] == "auto-routed to core"

    job = client.get(
        f"/graphs/test-tenant/enrich/jobs/{data['job_id']}", headers=auth_headers
    ).json()
    assert job["tier"] == "core"
    # The web confidence floor was applied (0.85 default → 0.4).
    assert job["confidence_min"] == 0.4


def test_create_job_explicit_core_lowers_confidence_to_web_floor(
    client, auth_headers, mock_neptune, _paid_core_chain
):
    """An EXPLICIT (non-auto) core tier with the default confidence must ALSO get
    the web floor — previously only the agent path did this, so a direct core
    enrich filtered out every web verdict → 0 fills."""
    mock_neptune.query.return_value = _count_response(0)

    response = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Person",
            "attributes": ["company"],
            "kg_name": "kg",
            "tier": "core",
        },
    )
    assert response.status_code == 202
    data = response.json()
    # Uniform contract for explicit tiers: resolved_tier echoes the request tier.
    assert data["resolved_tier"] == "core"
    assert data["routing_note"] is None
    assert data["needs_clarification"] is False
    assert data["candidates"] is None

    job = client.get(
        f"/graphs/test-tenant/enrich/jobs/{data['job_id']}", headers=auth_headers
    ).json()
    assert job["tier"] == "core"
    assert job["confidence_min"] == 0.4


def test_create_job_explicit_core_respects_user_confidence(
    client, auth_headers, mock_neptune, _paid_core_chain
):
    """A user-supplied (non-default) confidence_min must be respected — the floor
    only overrides the UNSET 0.85 sentinel, never an explicit value."""
    mock_neptune.query.return_value = _count_response(0)

    response = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Person",
            "attributes": ["company"],
            "kg_name": "kg",
            "tier": "core",
            "confidence_min": 0.7,
        },
    )
    assert response.status_code == 202
    job = client.get(
        f"/graphs/test-tenant/enrich/jobs/{response.json()['job_id']}",
        headers=auth_headers,
    ).json()
    # User value preserved, NOT lowered to the floor.
    assert job["confidence_min"] == 0.7


def test_create_job_lite_does_not_lower_confidence(
    client, auth_headers, mock_neptune
):
    """A free tier (lite — no paid adapter) must NOT trigger the web floor; the
    default 0.85 confidence is preserved."""
    mock_neptune.query.return_value = _count_response(0)

    response = client.post(
        "/graphs/test-tenant/enrich/jobs",
        headers=auth_headers,
        json={
            "type_name": "Country",
            "attributes": ["iso_code"],
            "kg_name": "kg",
            "tier": "lite",
        },
    )
    assert response.status_code == 202
    data = response.json()
    assert data["resolved_tier"] == "lite"
    job = client.get(
        f"/graphs/test-tenant/enrich/jobs/{data['job_id']}", headers=auth_headers
    ).json()
    assert job["tier"] == "lite"
    assert job["confidence_min"] == 0.85


# ---------------------------------------------------------------------------
# P1: enriched INSTANCE values are stored as TYPED literals matching the
# declared range (not bare xsd:string). Bug: declared range said
# xsd:integer/dateTime but the stored literal was xsd:string, so typed NL
# filters silently returned empty. Fix routes each value through ingestion's
# validate_triple so the stored literal matches the declared datatype.
# ---------------------------------------------------------------------------


def test_executor_apply_writes_typed_integer_literal(monkeypatch):
    """A numeric enriched value is stored as a TYPED literal
    ``"92"^^<…#integer>`` matching the declared integer range — NOT a bare
    ``"92"`` xsd:string literal the typed NL filters would miss (the P1 bug)."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
            {"uri": "https://cograph.tech/entities/Product/p2", "label": "Makita", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.side_effect = _query_router(
            _entities_query_response(rows), existing_range=None
        )
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "humanness_score"): [
                    Verdict(value="92", confidence=0.95, source="wikidata")
                ],
                ("Makita", "humanness_score"): [
                    Verdict(value="87", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["humanness_score"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        joined = "\n".join(_instance_inserts(neptune))
        # The object is the TYPED literal, matching the declared xsd:integer range.
        assert '"92"^^<http://www.w3.org/2001/XMLSchema#integer>' in joined
        assert '"87"^^<http://www.w3.org/2001/XMLSchema#integer>' in joined
        # And NOT a bare (untyped) string literal — the regression we fixed.
        assert '"92" .' not in joined
        assert '"92"\n' not in joined

    asyncio.run(run())


def test_executor_apply_writes_comma_number_as_string_not_dropped(monkeypatch):
    """A comma-grouped number ("1,234") is declared ``string`` and survives as a
    plain string literal — NOT declared integer and then dropped by the validator
    (the comma data-loss regression the re-review flagged). Inference and the
    write-side validator agree: commas are not numeric."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
            {"uri": "https://cograph.tech/entities/Product/p2", "label": "Makita", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.side_effect = _query_router(
            _entities_query_response(rows), existing_range=None
        )
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "unit_sales"): [
                    Verdict(value="1,234", confidence=0.95, source="wikidata")
                ],
                ("Makita", "unit_sales"): [
                    Verdict(value="12,345", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["unit_sales"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        joined = "\n".join(_instance_inserts(neptune))
        # The comma value is written as a plain string literal — present, not dropped.
        assert '"1,234"' in joined
        assert '"12,345"' in joined
        # Declared as string, so NOT typed integer.
        decl = "\n".join(_declaration_updates(neptune))
        assert "XMLSchema#integer" not in decl

    asyncio.run(run())


def test_executor_apply_writes_typed_datetime_literal(monkeypatch):
    """A date enriched value is stored as a TYPED ``^^<…#dateTime>`` literal
    matching the declared dateTime range."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.side_effect = _query_router(
            _entities_query_response(rows), existing_range=None
        )
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "founded"): [
                    Verdict(value="2026-06-28", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["founded"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        joined = "\n".join(_instance_inserts(neptune))
        # The object carries the dateTime XSD type (value is normalized to full
        # ISO-8601 by validate_triple's _typed_value, so match the type suffix).
        assert "^^<http://www.w3.org/2001/XMLSchema#dateTime>" in joined
        # Not a bare untyped string form of the date.
        assert '"2026-06-28" .' not in joined

    asyncio.run(run())


def test_executor_apply_writes_entity_iri_object(monkeypatch):
    """An entity-IRI enriched value is written as an IRI object
    ``<https://cograph.tech/entities/…>`` (a relationship edge), never a quoted
    literal, and is declared with a relationship (types/<Target>) range."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Drill", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.side_effect = _query_router(
            _entities_query_response(rows), existing_range=None
        )
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Drill", "manufacturer"): [
                    Verdict(
                        value="https://cograph.tech/entities/Manufacturer/bosch",
                        confidence=0.95,
                        source="wikidata",
                    )
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["manufacturer"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        joined = "\n".join(_instance_inserts(neptune))
        # Object written as an IRI, not a quoted literal.
        assert "<https://cograph.tech/entities/Manufacturer/bosch>" in joined
        assert '"https://cograph.tech/entities/Manufacturer/bosch"' not in joined

        # Declared as a relationship range, not xsd:string.
        mfr_attr = "https://cograph.tech/types/Product/attrs/manufacturer"
        types_mfr = "https://cograph.tech/types/Manufacturer"
        primary = [d for d in _declaration_updates(neptune) if mfr_attr in d]
        assert primary, "manufacturer attribute not declared"
        assert f"#range> <{types_mfr}>" in primary[0]

    asyncio.run(run())


def test_executor_apply_skips_value_not_conforming_to_existing_range(monkeypatch):
    """An attribute already declared with an integer range: a non-conforming
    value ("five stars") is REJECTED (no instance triple written) while a
    conforming numeric value IS written as a typed literal. The P1 guarantee:
    we never PIN a mismatched literal under a declared richer range."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bad", "vals": ""},
            {"uri": "https://cograph.tech/entities/Product/p2", "label": "Good", "vals": ""},
        ]
        neptune = AsyncMock()
        # Existing range is xsd:integer for the 'rating' attribute.
        neptune.query.side_effect = _query_router(
            _entities_query_response(rows),
            existing_range="http://www.w3.org/2001/XMLSchema#integer",
        )
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                # Non-conforming → must be rejected (no triple).
                ("Bad", "rating"): [
                    Verdict(value="five stars", confidence=0.95, source="wikidata")
                ],
                # Conforming → must be written typed.
                ("Good", "rating"): [
                    Verdict(value="5", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["rating"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        joined = "\n".join(_instance_inserts(neptune))
        rating_attr = "https://cograph.tech/types/Product/attrs/rating"
        # The conforming value IS written, typed as integer.
        assert '"5"^^<http://www.w3.org/2001/XMLSchema#integer>' in joined
        # The non-conforming value is NOT written under the rating predicate, in
        # any form (neither as a literal nor coerced).
        assert "five stars" not in joined
        # Sanity: the rating predicate appears at most for the conforming row only.
        assert joined.count(f"<{rating_attr}>") == 1

    asyncio.run(run())


def test_executor_apply_provenance_stays_plain_string(monkeypatch):
    """The provenance companions (``*_source_url`` / ``*_provenance``) stay PLAIN
    string literals even when the primary value is typed — they are user-facing
    citations, never typed as anything richer."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.side_effect = _query_router(
            _entities_query_response(rows), existing_range=None
        )
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata(
            {
                ("Bosch", "humanness_score"): [
                    Verdict(
                        value="92",
                        confidence=0.95,
                        source="wikidata",
                        source_url="https://www.wikidata.org/wiki/Q234021",
                    )
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["humanness_score"], policy=ConflictPolicy.overwrite)
        await store.create(job)
        await executor.run(job, "test-tenant")

        joined = "\n".join(_instance_inserts(neptune))
        src_pred = (
            "https://cograph.tech/types/Product/attrs/humanness_score_source_url"
        )
        # The source_url object is a PLAIN quoted string literal (no ^^ type),
        # written as an http(s) value → _escape_value wraps it as an <IRI>… but the
        # provenance source_url IS a URL, so it lands as an IRI object, not a typed
        # literal. The provenance text companion, however, is a plain string.
        prov_pred = (
            "https://cograph.tech/types/Product/attrs/humanness_score_provenance"
        )
        # The provenance free-text companion is a plain string literal, no ^^ type.
        assert f"<{prov_pred}>" in joined
        # No XSD type annotation on the provenance object.
        for line in joined.splitlines():
            if prov_pred in line:
                assert "^^" not in line, f"provenance should be plain string: {line}"
        # And the primary value is still typed integer (the value WAS typed).
        assert '"92"^^<http://www.w3.org/2001/XMLSchema#integer>' in joined
        # The source_url predicate is present (declared + written).
        assert f"<{src_pred}>" in joined

    asyncio.run(run())


def test_apply_decisions_writes_typed_integer_literal(monkeypatch):
    """The review-apply path (apply_decisions) also types the accepted value:
    a numeric accepted value is stored as ``"92"^^<…#integer>``, matching the
    declared range — same P1 fix as the auto-apply run() path."""
    import cograph_client.api.routes.explore as explore_mod

    monkeypatch.setattr(explore_mod, "schedule_recompute", lambda *a, **k: None)

    async def run():
        neptune = AsyncMock()
        neptune.update.return_value = None
        # No existing range → inference types it integer from the value.
        neptune.query.return_value = _range_response()
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata({})
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)
        job = _make_job(attributes=["humanness_score"], policy=ConflictPolicy.stage)
        await store.create(job)

        decisions = [
            ConflictReview(
                entity_uri="https://cograph.tech/entities/Product/p1",
                attribute="humanness_score",
                existing_value="",
                proposed=Verdict(value="92", confidence=0.95, source="wikidata"),
                decision="accept",
            ),
        ]
        applied = await executor.apply_decisions(job.id, decisions)
        assert applied == 1

        joined = "\n".join(_instance_inserts(neptune))
        assert '"92"^^<http://www.w3.org/2001/XMLSchema#integer>' in joined
        assert '"92" .' not in joined

    asyncio.run(run())


# ---------------------------------------------------------------------------
# P1: numeric inference is tightened so a string column isn't mis-declared
# numeric. _is_int / _is_float reject underscores and non-finite tokens.
# ---------------------------------------------------------------------------


def test_is_int_rejects_underscores_and_commas_keeps_plain_ints():
    """``_is_int`` rejects underscore AND comma groupings, while plain ints still
    parse True. Commas are rejected so the inference layer agrees with the
    write-side validator (which does NOT strip commas): otherwise a column like
    ``"1,234"`` would be declared integer here and then DROPPED at write time."""
    assert _is_int("92") is True
    assert _is_int("-3") is True
    assert _is_int("1,000") is False  # comma rejected (matches the validator)
    assert _is_int("1_000") is False
    assert _is_int("five") is False


def test_is_float_rejects_non_finite_commas_and_underscores():
    """``_is_float`` rejects the non-finite special tokens float() accepts
    (inf/-inf/infinity/nan), comma groupings, and underscore groupings, while real
    decimals and real scientific notation still parse True. Commas are rejected to
    agree with the validator (which would drop a comma value declared float)."""
    assert _is_float("8.5") is True
    assert _is_float("1e10") is True  # real scientific notation
    assert _is_float("-2.0") is True
    assert _is_float("inf") is False
    assert _is_float("-inf") is False
    assert _is_float("infinity") is False
    assert _is_float("nan") is False
    assert _is_float("1_000.5") is False
    assert _is_float("1,000.5") is False  # comma rejected (matches the validator)


def test_infer_datatype_does_not_mis_declare_special_tokens():
    """A column of ``inf``/``nan``/underscore/comma strings is NOT mis-declared
    float/int — it falls through to ``string`` (the tightened helpers feed
    inference), so the values survive at write time as plain string literals
    instead of being declared numeric and then dropped by the validator."""
    assert _infer_datatype_from_values(["inf", "nan"]) == "string"
    assert _infer_datatype_from_values(["1_000"]) == "string"
    assert _infer_datatype_from_values(["1,234", "12,345"]) == "string"


# ---------------------------------------------------------------------------
# Provider logs + error summary (run-detail observability)
# ---------------------------------------------------------------------------


def test_provider_tally_rollup_status_and_errors():
    """The tally rolls each provider's per-attempt outcomes into a coarse status
    (ok / no_match / error / skipped) and aggregates failures into an ordered
    error summary with representative messages + counts."""
    tally = _ProviderTally()
    # wikidata: one live match, one cache no_match → produced a usable result → ok
    tally.record_attempt("wikidata", cache_hit=False, outcome="match")
    tally.record_attempt("wikidata", cache_hit=True, outcome="no_match")
    # exa: every attempt failed (a timeout + two errors) → error
    tally.record_attempt("exa", cache_hit=False, outcome="timeout", error_msg="timed out after 30s")
    tally.record_attempt("exa", cache_hit=False, outcome="error", error_msg="HTTP 503 a")
    tally.record_attempt("exa", cache_hit=False, outcome="error", error_msg="HTTP 503 b")
    # gemini: ran but found nothing → no_match
    tally.record_attempt("gemini", cache_hit=False, outcome="no_match")
    # perplexity: named in a chain but not registered here → skipped
    tally.record_missing("perplexity")

    logs = {p.provider: p for p in tally.to_logs()}
    assert logs["wikidata"].status == "ok"
    assert logs["wikidata"].matches == 1
    assert logs["wikidata"].attempts == 1  # cache hit not counted as an attempt
    assert logs["wikidata"].cache_hits == 1
    assert logs["exa"].status == "error"
    assert logs["exa"].errors == 2
    assert logs["exa"].timeouts == 1
    assert logs["exa"].last_error  # carries a representative message
    assert logs["gemini"].status == "no_match"
    assert logs["perplexity"].status == "skipped"

    errs = tally.to_error_summary()
    # exa errors are aggregated with count, ordered most-frequent first.
    by_kind = {(e.provider, e.kind): e for e in errs}
    assert by_kind[("exa", "error")].count == 2
    assert by_kind[("exa", "timeout")].count == 1
    assert by_kind[("perplexity", "missing")].count == 1
    assert errs[0].count >= errs[-1].count  # sorted by count desc


def test_executor_records_provider_logs_end_to_end():
    """A completed enrichment run carries a per-provider log: which provider was
    used, how many lookups matched vs found nothing — surfaced in run detail."""
    async def run():
        rows = [
            {"uri": "https://cograph.tech/entities/Product/p1", "label": "Bosch", "vals": ""},
            {"uri": "https://cograph.tech/entities/Product/p2", "label": "Unknown Co", "vals": ""},
        ]
        neptune = AsyncMock()
        neptune.query.return_value = _entities_query_response(rows)
        neptune.update.return_value = None

        store = InMemoryJobStore()
        cache = EnrichmentCache()
        # "Bosch" resolves; "Unknown Co" returns nothing (no_match).
        wikidata = FakeWikidata(
            {
                ("Bosch", "manufacturer"): [
                    Verdict(value="Robert Bosch GmbH", confidence=0.95, source="wikidata")
                ],
            }
        )
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["manufacturer"], policy=ConflictPolicy.stage)
        await store.create(job)
        await executor.run(job, "test-tenant")

        final = await store.get(job.id)
        assert final is not None
        assert len(final.provider_logs) == 1
        plog = final.provider_logs[0]
        assert plog.provider == "wikidata"
        assert plog.status == "ok"
        assert plog.attempts == 2
        assert plog.matches == 1
        assert plog.no_match == 1
        # A clean run records no errors.
        assert final.error_summary == []

    asyncio.run(run())


def test_lookup_chain_records_missing_provider():
    """A chain that names an unregistered provider records it as a 'skipped'
    provider log + a 'missing' error-summary entry, without failing the run."""
    async def run():
        neptune = AsyncMock()
        store = InMemoryJobStore()
        cache = EnrichmentCache()
        wikidata = FakeWikidata({})
        executor = EnrichmentExecutor(neptune, store, cache, wikidata)

        job = _make_job(attributes=["manufacturer"])
        tally = _ProviderTally()
        verdicts = await executor._lookup_chain(
            "Bosch",
            "manufacturer",
            ["ghost-provider"],  # not registered
            job,
            set(),
            0.85,
            tally=tally,
        )
        assert verdicts == []
        logs = tally.to_logs()
        assert len(logs) == 1
        assert logs[0].provider == "ghost-provider"
        assert logs[0].status == "skipped"
        errs = tally.to_error_summary()
        assert errs and errs[0].kind == "missing"

    asyncio.run(run())
