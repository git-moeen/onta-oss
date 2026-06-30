import json
import os
import re
import time

import anthropic
import httpx
import structlog

from cograph_client.graph.client import NeptuneClient
from cograph_client.graph.parser import parse_sparql_results
from cograph_client.graph.queries import parse_kg_graph_uri
from cograph_client.models.query import NLResult
from cograph_client.nlp.prompts import SPARQL_GENERATION_SYSTEM, build_generation_prompt
from cograph_client.nlp.validator import normalize_sparql, validate_sparql
from cograph_client.resolver.llm_router import model_chain
from cograph_client.spatiotemporal.routing import (
    SPATIAL_INTENT_SCHEMA,
    SPATIAL_INTENT_SYSTEM,
    filter_by_type,
    format_spatial_answer,
    looks_spatial,
    parse_spatial_intent,
)

logger = structlog.stdlib.get_logger("cograph.nlp.pipeline")

# In-memory ontology cache: {graph_uri: (summary_str, timestamp)}
_ontology_cache: dict[str, tuple[str, float]] = {}
ONTOLOGY_CACHE_TTL = 60  # seconds

# Cap on concurrent enum-discovery SPARQL queries (COG-58). Enum discovery
# fires one COUNT(DISTINCT) per attribute + per relationship; an unbounded
# asyncio.gather meant a wide table (hundreds of columns → hundreds of
# attributes) launched O(columns) simultaneous queries, throttling serverless
# Neptune (1–2.5 NCU). The semaphore keeps the round-trip count bounded
# regardless of column count, trading a little latency for stability.
MAX_ENUM_DISCOVERY_CONCURRENCY = int(
    os.environ.get("OMNIX_ENUM_DISCOVERY_CONCURRENCY", "8")
)

# Attribute-alias map cache (ADR 0002 §7): {graph_uri: (old->new map, timestamp)}
_alias_cache: dict[str, tuple[dict[str, str], float]] = {}

# Query generation provider config
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEFAULT_QUERY_MODEL = os.environ.get("OMNIX_QUERY_MODEL", "llama3.1-8b")
DEFAULT_QUERY_PROVIDER = os.environ.get("OMNIX_QUERY_PROVIDER", "cerebras")  # cerebras, openrouter, or anthropic

# Embedding service singleton
_embedding_service = None


def get_embedding_service():
    """Lazy-init singleton for the ontology embedding service."""
    global _embedding_service
    if _embedding_service is None:
        from cograph_client.config import settings
        if settings.openrouter_api_key:
            from cograph_client.nlp.ontology_embeddings import OntologyEmbeddingService
            _embedding_service = OntologyEmbeddingService(
                openrouter_api_key=settings.openrouter_api_key,
                s3_bucket=settings.embeddings_s3_bucket,
                s3_prefix=settings.embeddings_s3_prefix,
            )
    return _embedding_service


# Spatial fast-path helpers (ONTA-157 Phase 2). Module-level + pure so they're
# trivially testable; the orchestration that uses them lives on NLQueryPipeline.
_GEO_WKT_URI = "http://www.opengis.net/ont/geosparql#wktLiteral"
_POINT_RE = re.compile(
    r"POINT\s*\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)", re.IGNORECASE
)


def _parse_iso_dt(s):
    """ISO-8601 string → tz-aware (UTC-assumed) datetime, or None. Mirrors the
    extractor so a query bound and an indexed validity compare without raising."""
    if not s or not isinstance(s, str):
        return None
    from datetime import datetime, timezone

    t = s.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _parse_point_wkt(wkt: str):
    """``"POINT(lon lat)"`` → (lon, lat) in WGS84 range, else None."""
    if not isinstance(wkt, str):
        return None
    m = _POINT_RE.search(wkt)
    if not m:
        return None
    try:
        lon, lat = float(m.group(1)), float(m.group(2))
    except ValueError:
        return None
    if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
        return None
    return lon, lat


def _sanitize_sparql_literal(text: str) -> str:
    """Strip characters that could break out of a SPARQL string literal, and cap
    length — the anchor description comes from the LLM and is interpolated into a
    FILTER(CONTAINS(...)) literal."""
    return re.sub(r'["\\\n\r\t]', " ", text).strip().lower()[:80]


class NLQueryPipeline:
    def __init__(self, neptune: NeptuneClient, anthropic_key: str):
        self.neptune = neptune
        self.anthropic = anthropic.AsyncAnthropic(api_key=anthropic_key)
        from cograph_client.config import settings
        self._openrouter_key = settings.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._cerebras_key = os.environ.get("CEREBRAS_API_KEY", getattr(settings, "cerebras_api_key", ""))
        self._query_model = DEFAULT_QUERY_MODEL
        self._query_provider = DEFAULT_QUERY_PROVIDER
        # Attribute aliases (ADR 0002 §7): resolve renamed attribute IRIs in
        # generated SPARQL. Default OFF so the default Neptune call pattern
        # stays byte-identical (same gating pattern as COGRAPH_ER_ENABLED).
        self._aliases_enabled = os.environ.get("COGRAPH_ALIASES_ENABLED", "0") == "1"
        # Spatio-temporal read routing (ONTA-157 Phase 2): a geo/proximity question
        # is answered directly from the secondary index (no Neptune round-trip).
        # Default OFF — same gating discipline as the aliases flag — so the default
        # query path (and every eval) stays byte-identical until explicitly enabled.
        self._spatial_routing_enabled = (
            os.environ.get("COGRAPH_SPATIAL_ROUTING_ENABLED", "0") == "1"
        )

    async def ask(self, question: str, graph_uri: str, instance_graph: str | None = None, exclude_questions: list[str] | None = None, layer_graph_uris: list[str] | None = None) -> NLResult:
        """Answer a natural-language question over the graph.

        layer_graph_uris (ADR 0002 §1, COG-37, opt-in): a LayerStack's
        visible_graph_uris(). Generated queries are graph-scoped (FROM the
        data graph), so without this the subclass-closure path can't see
        subClassOf edges living in other layer graphs; when provided, each
        generated query gains FROM clauses for every visible layer. When
        None (the default), behavior is exactly as before.
        """
        timing: dict[str, float] = {}
        timing["model"] = f"{self._query_provider}:{self._query_model}"
        # Ontology is always fetched from the base tenant graph
        # Instance data may be in a different graph (KG-specific)
        data_graph = instance_graph or graph_uri

        t0 = time.time()
        # Try semantic retrieval first, fall back to full ontology
        ontology = None
        embedding_svc = get_embedding_service()
        if embedding_svc:
            try:
                from cograph_client.config import settings
                ontology = await embedding_svc.retrieve(graph_uri, question, top_k=settings.embeddings_top_k)
                if ontology:
                    timing["ontology_source"] = "semantic"
            except Exception:
                pass
        if ontology is None:
            ontology = await self._fetch_ontology(graph_uri, data_graph)
            timing["ontology_source"] = "full"
        timing["ontology_fetch_ms"] = round((time.time() - t0) * 1000, 1)

        # Spatio-temporal fast path (ONTA-157 Phase 2, gated). For a geo/proximity
        # question, answer directly from the secondary index — no SPARQL, no Neptune
        # round-trip. Returns None (and we fall through to the normal path unchanged)
        # whenever routing is off, the question isn't spatial, the KG can't be
        # scoped, the intent doesn't parse, or the anchor can't be resolved.
        if self._spatial_routing_enabled and looks_spatial(question):
            spatial = await self._try_spatial_fast_path(
                question, ontology, data_graph, timing, t0
            )
            if spatial is not None:
                return spatial

        # Attribute-alias map (ADR 0002 §7). Only fetched when the feature is
        # enabled; an empty map (no aliases registered) leaves every query
        # untouched, so zero aliases => zero behavior change.
        alias_map: dict[str, str] = {}
        if self._aliases_enabled:
            alias_map = await self._fetch_alias_map(graph_uri)

        # Retrieve few-shot examples from the example bank
        examples_text = ""
        try:
            from cograph_client.nlp.example_bank import get_example_bank, format_examples_for_prompt
            bank = get_example_bank()
            if bank and bank._examples:
                # Extract kg_name from data_graph URI for cross-dataset preference
                kg_name = data_graph.split("/kg/")[-1] if "/kg/" in data_graph else ""
                examples = await bank.retrieve(
                    question=question,
                    ontology_context=ontology,
                    exclude_questions=exclude_questions or [],
                    kg_name=kg_name,
                    top_k=3,
                )
                if examples:
                    examples_text = format_examples_for_prompt(examples)
                    timing["examples_retrieved"] = len(examples)
        except Exception:
            pass

        max_attempts = 3
        last_error = ""
        sparql = ""
        explanation = ""
        functions_needed: list[str] = []

        for attempt in range(max_attempts):
            t1 = time.time()
            if attempt == 0:
                llm_response = await self._generate_sparql(question, ontology, data_graph, examples_text=examples_text)
            else:
                llm_response = await self._generate_sparql(
                    question, ontology, data_graph,
                    error_feedback=f"The previous query failed with: {last_error}\nQuery was: {sparql}\nPlease fix the SPARQL syntax and try again.",
                )
            timing[f"sparql_gen_ms{f'_retry{attempt}' if attempt > 0 else ''}"] = round((time.time() - t1) * 1000, 1)

            sparql = normalize_sparql(llm_response.get("sparql", ""))
            # Fix bare attribute URIs using ontology context
            sparql = self._fix_attribute_uris(sparql, ontology)
            # Fix cross-type attribute misuse and rdf:type shorthand
            sparql = self._fix_common_sparql_issues(sparql, ontology, alias_map)
            if layer_graph_uris:
                # Layer-aware closure (COG-37): widen the graph scope so the
                # subClassOf* walk sees edges in every visible layer graph.
                from cograph_client.graph.ontology_queries import add_layer_from_clauses
                sparql = add_layer_from_clauses(sparql, layer_graph_uris)
            explanation = llm_response.get("explanation", "")
            functions_needed = llm_response.get("functions_needed", [])

            is_valid, error = validate_sparql(sparql)
            if not is_valid:
                last_error = error
                continue

            try:
                t2 = time.time()
                raw = await self.neptune.query(sparql)
                timing[f"neptune_exec_ms{f'_retry{attempt}' if attempt > 0 else ''}"] = round((time.time() - t2) * 1000, 1)
                _, bindings = parse_sparql_results(raw)
                answer = await self._format_answer(bindings, explanation)
                t_reph = time.time()
                narrative_answer = await self._rephrase_via_openrouter(question, bindings)
                timing["rephrase_ms"] = round((time.time() - t_reph) * 1000, 1)
                timing["total_ms"] = round((time.time() - t0) * 1000, 1)
                timing["attempts"] = attempt + 1
                return NLResult(
                    answer=answer,
                    sparql=sparql,
                    explanation=explanation,
                    ontology=ontology,
                    narrative_answer=narrative_answer,
                    functions_invoked=functions_needed,
                    timing=timing,
                )
            except Exception as e:
                last_error = str(e)
                continue

        timing["total_ms"] = round((time.time() - t0) * 1000, 1)
        timing["attempts"] = max_attempts
        return NLResult(
            answer=f"Could not answer after {max_attempts} attempts. Last error: {last_error}",
            sparql=sparql,
            explanation=explanation,
            ontology=ontology,
            timing=timing,
        )

    # ------------------------------------------------------------- spatial path
    async def _try_spatial_fast_path(
        self,
        question: str,
        ontology: str,
        data_graph: str,
        timing: dict,
        t0: float,
    ) -> NLResult | None:
        """Answer a geo/proximity question directly from the spatio-temporal index.

        Returns an ``NLResult`` on success, or ``None`` to fall through to the
        normal SPARQL path — when the graph isn't a per-KG instance graph, the LLM
        doesn't return a servable spatial intent, the anchor can't be resolved, or
        anything errors. Never raises into :meth:`ask` (best-effort fast path).
        """
        scope = parse_kg_graph_uri(data_graph)
        if scope is None:
            return None  # index rows are scoped per (tenant, kg); can't route otherwise
        tenant_id, kg_name = scope
        try:
            ts = time.time()
            raw = await self._detect_spatial_intent(question, ontology)
            intent = parse_spatial_intent(raw) if raw else None
            timing["spatial_intent_ms"] = round((time.time() - ts) * 1000, 1)
            if intent is None:
                return None

            from cograph_client.spatiotemporal.registry import get_spatiotemporal_index

            index = get_spatiotemporal_index()

            # Temporal predicate: a single instant (as_of) wins over a window.
            as_of = _parse_iso_dt(intent.as_of)
            window = None
            if as_of is None and (intent.time_from or intent.time_to):
                window = (_parse_iso_dt(intent.time_from), _parse_iso_dt(intent.time_to))

            tq = time.time()
            if intent.kind == "radius":
                coords = await self._resolve_anchor_coords(intent.anchor, data_graph)
                if coords is None:
                    return None  # "near X" but X didn't resolve → fall through
                lon, lat = coords
                hits = await index.query_radius(
                    tenant_id, lon, lat, intent.radius_m,
                    kg_name=kg_name, time_window=window, as_of=as_of,
                )
            else:  # bbox
                min_lon, min_lat, max_lon, max_lat = intent.bbox
                hits = await index.query_bbox(
                    tenant_id, min_lon, min_lat, max_lon, max_lat,
                    kg_name=kg_name, time_window=window, as_of=as_of,
                )
            timing["spatial_index_ms"] = round((time.time() - tq) * 1000, 1)

            hits = filter_by_type(hits, intent.target_type)
            answer = format_spatial_answer(hits, intent)
            timing["spatial_routed"] = "true"
            timing["total_ms"] = round((time.time() - t0) * 1000, 1)
            return NLResult(
                answer=answer,
                sparql="",
                explanation="Answered from the spatio-temporal index (no SPARQL).",
                ontology=ontology,
                narrative_answer=answer,
                functions_invoked=[],
                timing=timing,
            )
        except Exception:
            logger.warning("spatial_fast_path_failed", exc_info=True)
            return None

    async def _detect_spatial_intent(self, question: str, ontology: str) -> dict | None:
        """LLM classify: is this a servable spatial lookup, and with what params?
        Returns the raw JSON dict (caller parses) or None on error."""
        user = (
            f"Question: {question}\n\n"
            f"Knowledge-graph types/attributes (for the target type, if any):\n"
            f"{ontology[:2000]}"
        )
        try:
            return await self._structured_llm(
                SPATIAL_INTENT_SYSTEM, user, "spatial_intent", SPATIAL_INTENT_SCHEMA
            )
        except Exception:
            logger.warning("spatial_intent_detect_failed", exc_info=True)
            return None

    async def _resolve_anchor_coords(self, anchor, data_graph: str):
        """Resolve a radius anchor to ``(lon, lat)``: explicit coords, else a KG
        entity matched by ``entity_description`` (one Neptune lookup). None if
        unresolved — the caller then falls through to the SPARQL path."""
        if anchor is None:
            return None
        if anchor.has_coords():
            return (anchor.lon, anchor.lat)
        if not anchor.entity_description:
            return None
        return await self._resolve_anchor_via_neptune(
            anchor.entity_description, data_graph
        )

    async def _resolve_anchor_via_neptune(self, description: str, data_graph: str):
        """Find a KG entity whose label/text contains ``description`` AND that
        carries a ``geo:wktLiteral``; return that point's ``(lon, lat)`` or None.

        One scoped SELECT, LIMIT 1. The description is sanitized before it is
        interpolated into the FILTER literal."""
        desc = _sanitize_sparql_literal(description)
        for article in ("the ", "a ", "an "):
            if desc.startswith(article):
                desc = desc[len(article):]
        if not desc:
            return None
        q = (
            f"SELECT ?wkt FROM <{data_graph}> WHERE {{ "
            f"?e ?lp ?lbl . "
            f'FILTER(isLiteral(?lbl) && CONTAINS(LCASE(STR(?lbl)), "{desc}")) '
            f"?e ?gp ?wkt . "
            f"FILTER(datatype(?wkt) = <{_GEO_WKT_URI}>) "
            f"}} LIMIT 1"
        )
        try:
            raw = await self.neptune.query(q)
            _, rows = parse_sparql_results(raw)
        except Exception:
            logger.warning("anchor_resolve_failed", exc_info=True)
            return None
        if not rows:
            return None
        return _parse_point_wkt(rows[0].get("wkt", ""))

    async def _structured_llm(
        self, system: str, user: str, schema_name: str, schema: dict
    ) -> dict:
        """Provider-agnostic structured-JSON call for non-SPARQL classifiers (e.g.
        spatial-intent detection). Mirrors :meth:`_generate_sparql`'s provider
        selection but is a SEPARATE method on purpose — the SPARQL generators stay
        byte-identical so evals are unaffected."""
        if self._query_provider == "cerebras" and self._cerebras_key:
            endpoint = "https://api.cerebras.ai/v1/chat/completions"
            key, model = self._cerebras_key, self._query_model
        elif self._openrouter_key:
            endpoint = f"{OPENROUTER_BASE}/chat/completions"
            key, model = self._openrouter_key, self._query_model
        else:
            return await self._structured_via_anthropic(system, user, schema)
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                    },
                },
            )
            res.raise_for_status()
            text = res.json()["choices"][0]["message"]["content"].strip()
            if text.startswith("```"):
                text = "\n".join(
                    l for l in text.split("\n") if not l.strip().startswith("```")
                )
            return json.loads(text)

    async def _structured_via_anthropic(self, system: str, user: str, schema: dict) -> dict:
        message = await self.anthropic.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        return json.loads(message.content[0].text)

    async def select_entity_uris(
        self,
        description: str,
        type_name: str,
        graph_uri: str,
        instance_graph: str | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """Resolve an NL subset description to the IRIs of ``type_name`` entities.

        Turns a ranked/specific subset — e.g. "the 5 brokers with the most
        property listings" — into the concrete entity IRIs it names, so a caller
        (the agent's enrich planner) can enrich exactly those via ``entity_uris``
        instead of the whole type. Reuses the SAME NL→SPARQL generation +
        validation as :meth:`ask` (one query engine, no divergence); it only
        constrains the projection to the entity IRI (``?uri``) and extracts it.

        Returns a deduped, order-preserving list capped at ``limit``. Returns
        ``[]`` on any failure (unparseable/invalid SPARQL, Neptune error, or no
        IRI column) — never raises; the caller decides how to handle "couldn't
        resolve".
        """
        data_graph = instance_graph or graph_uri
        try:
            ontology = await self._fetch_ontology(graph_uri, data_graph)
        except Exception:
            logger.warning("select_entity_uris_ontology_failed", exc_info=True)
            return []
        cap = f" Return at most {int(limit)} rows." if limit else ""
        question = (
            f"Return ONLY the IRI of each {type_name} entity in this set: "
            f"{description}. The SELECT must project a single column named ?uri, "
            f"bound by `?uri a` the {type_name} class. Apply any ranking/ordering "
            f"and limit the set describes, but keep ?uri in the SELECT — do NOT "
            f"aggregate it away or replace it with a label.{cap}"
        )
        try:
            resp = await self._generate_sparql(question, ontology, data_graph)
            sparql = normalize_sparql(resp.get("sparql", ""))
            sparql = self._fix_attribute_uris(sparql, ontology)
            sparql = self._fix_common_sparql_issues(sparql, ontology)
            is_valid, error = validate_sparql(sparql)
            if not is_valid:
                logger.warning("select_entity_uris_invalid_sparql", error=error)
                return []
            raw = await self.neptune.query(sparql)
            _, bindings = parse_sparql_results(raw)
        except Exception:
            logger.warning("select_entity_uris_failed", exc_info=True)
            return []
        return self._entity_uris_from_bindings(bindings, limit)

    @staticmethod
    def _entity_uris_from_bindings(
        bindings: list[dict], limit: int | None = None
    ) -> list[str]:
        """Pull entity IRIs out of result bindings, order-preserving and deduped.

        Prefers the ``?uri`` column the resolver prompt asks for; if a row lacks
        it, falls back to the first http(s)-IRI value in that row. Caps at
        ``limit`` when given.
        """
        out: list[str] = []
        seen: set[str] = set()

        def _is_iri(v: object) -> bool:
            return isinstance(v, str) and v.startswith(("http://", "https://"))

        for row in bindings:
            val = row.get("uri")
            if not _is_iri(val):
                val = next((v for v in row.values() if _is_iri(v)), None)
            if val and val not in seen:
                seen.add(val)
                out.append(val)
                if limit and len(out) >= int(limit):
                    break
        return out

    async def _fetch_ontology(self, graph_uri: str, instance_graph: str | None = None) -> str:
        # Cache key includes instance graph so different KGs get filtered ontologies
        cache_key = f"{graph_uri}|{instance_graph or ''}"
        cached = _ontology_cache.get(cache_key)
        if cached and (time.time() - cached[1]) < ONTOLOGY_CACHE_TTL:
            return cached[0]

        from cograph_client.graph.ontology_queries import get_full_ontology_query, type_uri, attr_uri
        TYPE_URI_PREFIX = "https://cograph.tech/types/"
        try:
            # If querying a specific KG, find which types actually have instances
            active_types: set[str] | None = None
            if instance_graph and instance_graph != graph_uri:
                type_query = (
                    f"SELECT DISTINCT ?type FROM <{instance_graph}> "
                    f"WHERE {{ ?s <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> ?type }}"
                )
                type_raw = await self.neptune.query(type_query)
                _, type_bindings = parse_sparql_results(type_raw)
                active_types = set()
                for row in type_bindings:
                    t = row.get("type", "")
                    if t.startswith(TYPE_URI_PREFIX):
                        active_types.add(t[len(TYPE_URI_PREFIX):])

            raw = await self.neptune.query(get_full_ontology_query(graph_uri))
            _, bindings = parse_sparql_results(raw)

            types: dict[str, dict] = {}
            for row in bindings:
                tl = row.get("typeLabel", "")
                if not tl:
                    continue
                # Filter to only types with instances in the target KG
                if active_types is not None and tl not in active_types:
                    continue
                if tl not in types:
                    types[tl] = {"attributes": [], "relationships": [], "functions": set()}
                if row.get("attrLabel"):
                    attr_name = row["attrLabel"]
                    range_str = row.get("range", "")
                    if range_str.startswith(TYPE_URI_PREFIX):
                        target_type = range_str[len(TYPE_URI_PREFIX):]
                        # Relationship predicates use onto/ namespace in instance data
                        onto_uri = f"https://cograph.tech/onto/{attr_name}"
                        entry = f"{attr_name} → {target_type} — predicate URI: <{onto_uri}>"
                        if entry not in types[tl]["relationships"]:
                            types[tl]["relationships"].append(entry)
                    else:
                        dtype = range_str.split("#")[-1] if "#" in range_str else "string"
                        entry = f"{attr_name} ({dtype}) — URI: <{attr_uri(tl, attr_name)}>"
                        if entry not in types[tl]["attributes"]:
                            types[tl]["attributes"].append(entry)
                if row.get("funcName"):
                    types[tl]["functions"].add(row["funcName"])

            if not types:
                return "No ontology defined yet."

            # Discover enumerated values for low-cardinality string attributes.
            # Runs cardinality checks concurrently (asyncio.gather) instead of
            # serially, cutting ontology fetch from ~7s to ~500ms. Concurrency
            # is bounded by a semaphore (COG-58) so a wide table with hundreds
            # of attributes can't launch hundreds of simultaneous queries
            # against serverless Neptune — the count stays capped regardless of
            # column count.
            import asyncio
            MAX_ENUM_CARDINALITY = 25
            _enum_sem = asyncio.Semaphore(MAX_ENUM_DISCOVERY_CONCURRENCY)

            async def _gather_bounded(coros: list) -> list:
                """asyncio.gather, but each coroutine acquires the shared enum
                semaphore first so at most MAX_ENUM_DISCOVERY_CONCURRENCY run at
                once. Preserves return_exceptions semantics for callers."""
                async def _run(coro):
                    async with _enum_sem:
                        return await coro
                return await asyncio.gather(
                    *[_run(c) for c in coros], return_exceptions=True
                )
            enum_values: dict[str, dict[str, list[str]]] = {}
            enum_counts: dict[str, dict[str, int]] = {}
            empty_rels: set[tuple[str, str]] = set()
            if instance_graph:
                # Collect all attribute and relationship URIs for cardinality checks
                all_attrs: list[tuple[str, str, str]] = []  # (type_name, attr_name, uri)
                string_attrs: list[tuple[str, str, str]] = []  # string attrs only (for enum values)
                rel_uris: list[tuple[str, str, str]] = []  # (type_name, rel_name, onto_uri)
                for type_name, info in types.items():
                    for attr_entry in info["attributes"]:
                        a_name = attr_entry.split(" (")[0]
                        all_attrs.append((type_name, a_name, attr_uri(type_name, a_name)))
                        if "(string)" in attr_entry:
                            string_attrs.append((type_name, a_name, attr_uri(type_name, a_name)))
                    for rel_entry in info["relationships"]:
                        r_name = rel_entry.split(" →")[0].strip()
                        onto_uri = f"https://cograph.tech/onto/{r_name}"
                        rel_uris.append((type_name, r_name, onto_uri))

                # Define cardinality check function ONCE (used for both attrs and rels)
                async def _count_predicate(tn: str, an: str, uri: str) -> tuple[str, str, int]:
                    q = (
                        f"SELECT (COUNT(DISTINCT ?val) AS ?cnt) FROM <{instance_graph}> "
                        f"WHERE {{ ?s <{uri}> ?val }}"
                    )
                    raw = await self.neptune.query(q)
                    _, bindings = parse_sparql_results(raw)
                    cnt = int(bindings[0].get("cnt", 0)) if bindings else 0
                    return tn, an, cnt

                # Phase 1: Concurrent cardinality checks for ALL attributes
                if all_attrs:
                    try:
                        count_results = await _gather_bounded(
                            [_count_predicate(tn, an, uri) for tn, an, uri in all_attrs]
                        )

                        low_card_attrs: list[tuple[str, str, str]] = []
                        exceptions = sum(1 for r in count_results if isinstance(r, Exception))
                        if exceptions:
                            logger.warning("cardinality_check_exceptions", count=exceptions, total=len(count_results))
                        for result in count_results:
                            if isinstance(result, Exception):
                                continue
                            tn, an, cnt = result
                            enum_counts.setdefault(tn, {})[an] = cnt
                            if 0 < cnt <= MAX_ENUM_CARDINALITY:
                                low_card_attrs.append((tn, an, attr_uri(tn, an)))

                        # Phase 2: Concurrent value fetches for low-cardinality attrs
                        async def _fetch_vals(tn: str, an: str, uri: str) -> tuple[str, str, list[str]]:
                            q = (
                                f"SELECT DISTINCT ?val FROM <{instance_graph}> "
                                f"WHERE {{ ?s <{uri}> ?val }} LIMIT {MAX_ENUM_CARDINALITY}"
                            )
                            raw = await self.neptune.query(q)
                            _, bindings = parse_sparql_results(raw)
                            return tn, an, [r["val"] for r in bindings if r.get("val")]

                        if low_card_attrs:
                            val_results = await _gather_bounded(
                                [_fetch_vals(tn, an, uri) for tn, an, uri in low_card_attrs]
                            )
                            for result in val_results:
                                if isinstance(result, Exception):
                                    continue
                                tn, an, vals = result
                                if vals:
                                    enum_values.setdefault(tn, {})[an] = sorted(vals)
                    except Exception:
                        logger.warning("cardinality_attr_check_failed", exc_info=True)

                # Phase 3: Check relationship cardinality to hide empty ones
                empty_rels: set[tuple[str, str]] = set()  # (type_name, rel_name)
                if rel_uris:
                    try:
                        rel_counts = await _gather_bounded(
                            [_count_predicate(tn, rn, uri) for tn, rn, uri in rel_uris]
                        )
                        for result in rel_counts:
                            if isinstance(result, Exception):
                                continue
                            tn, rn, cnt = result
                            if cnt == 0:
                                empty_rels.add((tn, rn))
                    except Exception:
                        logger.warning("cardinality_rel_check_failed", exc_info=True)

            lines = []
            for type_name, info in types.items():
                lines.append(f"Type: {type_name} — URI: <{type_uri(type_name)}>")
                if info["attributes"]:
                    annotated = []
                    for attr_entry in sorted(info["attributes"]):
                        a_name = attr_entry.split(" (")[0]
                        if type_name in enum_values and a_name in enum_values[type_name]:
                            # Low-cardinality: show actual values
                            vals = enum_values[type_name][a_name]
                            val_str = ", ".join(f'"{v}"' for v in vals[:10])
                            if len(vals) > 10:
                                val_str += f", ... ({len(vals)} total)"
                            annotated.append(f"{attr_entry} [values: {val_str}]")
                        elif type_name in enum_counts and a_name in enum_counts[type_name]:
                            cnt = enum_counts[type_name][a_name]
                            if cnt == 0:
                                # Skip empty attributes — no data, would confuse the LLM
                                continue
                            elif cnt > MAX_ENUM_CARDINALITY:
                                # High-cardinality: just show the count
                                annotated.append(f"{attr_entry} [{cnt} unique values]")
                            else:
                                annotated.append(attr_entry)
                        else:
                            annotated.append(attr_entry)
                    lines.append(f"  Attributes: {', '.join(annotated)}")
                if info["relationships"]:
                    filtered_rels = [
                        r for r in info["relationships"]
                        if (type_name, r.split(" →")[0].strip()) not in empty_rels
                    ]
                    if filtered_rels:
                        lines.append(f"  Relationships: {', '.join(sorted(filtered_rels))}")
                if info["functions"]:
                    lines.append(f"  Functions: {', '.join(sorted(info['functions']))}")
            summary = "\n".join(lines)
            # Log types that made it into the summary
            types_in_summary = [l.split("—")[0].replace("Type:", "").strip() for l in lines if l.startswith("Type:")]
            logger.info("ontology_summary_built", types_shown=len(types_in_summary),
                        types_active=len(active_types) if active_types else "all",
                        types_with_attrs=len(types),
                        names=types_in_summary[:10])

            # Cache it
            _ontology_cache[cache_key] = (summary, time.time())
            return summary
        except Exception:
            logger.error("ontology_fetch_failed", exc_info=True)
            return "Could not fetch ontology. Graph may be empty."

    @staticmethod
    def _fix_attribute_uris(sparql: str, ontology_summary: str) -> str:
        """Fix incorrect URIs in generated SPARQL using the ontology as ground truth.

        This is the post-processing safety net (Fix B). It catches URI mistakes
        the LLM makes despite the prompt telling it to copy-paste exact URIs.

        Strategy:
        1. Extract ALL valid URIs from the ontology summary (attributes + relationships)
        2. Find ALL cograph.tech URIs in the SPARQL
        3. For each URI not in the valid set, fuzzy-match against valid URIs
        4. Replace with the best match if similarity is high enough

        Common mistakes this catches:
        - <https://cograph.tech/bedrooms> → <https://cograph.tech/types/Property/attrs/bedrooms>
        - <https://cograph.tech/onto/bedrooms> → <https://cograph.tech/types/Property/attrs/bedrooms>
        - <https://cograph.tech/types/Property/attrs/property_type> → .../attrs/home_type
        - <https://cograph.tech/Property> → <https://cograph.tech/types/Property>
        """
        import re
        from difflib import SequenceMatcher

        # Step 1: Build the set of ALL valid URIs from the ontology
        valid_uris: dict[str, str] = {}  # name → full URI

        # Attribute URIs: "attr_name (type) — URI: <https://cograph.tech/types/Type/attrs/attr_name>"
        for match in re.finditer(r"URI: <(https://cograph\.tech/types/(\w+)/attrs/(\w+))>", ontology_summary):
            full_uri = match.group(1)
            attr_name = match.group(3)
            valid_uris[attr_name] = full_uri
            # Also index by type/attr for disambiguation
            valid_uris[f"{match.group(2)}/{attr_name}"] = full_uri

        # Relationship URIs: "predicate URI: <https://cograph.tech/onto/pred_name>"
        for match in re.finditer(r"predicate URI: <(https://cograph\.tech/onto/(\w+))>", ontology_summary):
            full_uri = match.group(1)
            pred_name = match.group(2)
            valid_uris[pred_name] = full_uri

        # Type URIs: "Type: TypeName — URI: <https://cograph.tech/types/TypeName>"
        for match in re.finditer(r"URI: <(https://cograph\.tech/types/(\w+))>", ontology_summary):
            full_uri = match.group(1)
            type_name = match.group(2)
            if "/attrs/" not in full_uri:  # don't overwrite attr URIs
                valid_uris[type_name] = full_uri

        valid_uri_set = set(valid_uris.values())

        # Step 2: Find and fix all cograph.tech URIs in the SPARQL
        def _fix_uri(m: re.Match) -> str:
            uri = m.group(1)

            # Already valid? Keep it.
            if uri in valid_uri_set:
                return m.group(0)

            # Skip known system URIs
            if any(uri.startswith(f"https://cograph.tech/{p}") for p in ("graphs/", "entities/", "functions/", "kgs/")):
                return m.group(0)

            # Extract the "name" part from the URI for matching
            # e.g., "https://cograph.tech/bedrooms" → "bedrooms"
            # e.g., "https://cograph.tech/onto/listed_by" → "listed_by"
            # e.g., "https://cograph.tech/types/Property/attrs/property_type" → "property_type"
            parts = uri.replace("https://cograph.tech/", "").rstrip("/").split("/")
            name = parts[-1] if parts else ""

            if not name:
                return m.group(0)

            # Direct name match
            if name in valid_uris:
                return f"<{valid_uris[name]}>"

            # Fuzzy match against all valid URI names
            best_match = None
            best_ratio = 0.0
            for vname, vuri in valid_uris.items():
                # Compare the short name part only
                vshort = vname.split("/")[-1]
                ratio = SequenceMatcher(None, name, vshort).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = vuri

            if best_ratio >= 0.75 and best_match:
                return f"<{best_match}>"

            return m.group(0)

        return re.sub(r"<(https://cograph\.tech/[^>]+)>", _fix_uri, sparql)

    @staticmethod
    def _fix_common_sparql_issues(sparql: str, ontology_summary: str, alias_map: dict[str, str] | None = None) -> str:
        """Fix common SPARQL generation mistakes that the LLM makes.

        1. Replace `a` shorthand with full rdf:type URI
        2. Replace cross-type attribute URIs (e.g., Person/attrs/name used on a Movie)
           with rdfs:label
        3. Replace overview/description attributes used as display names with rdfs:label
        """
        import re

        RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
        RDFS_LABEL = "<http://www.w3.org/2000/01/rdf-schema#label>"

        # Fix 1: Replace `a` shorthand (only when used as predicate position)
        # Match "?var a <..." or "?var rdf:type <..."
        sparql = re.sub(
            r'(\?\w+)\s+a\s+(<https://cograph\.tech/)',
            rf'\1 {RDF_TYPE} \2',
            sparql,
        )
        sparql = re.sub(
            r'(\?\w+)\s+rdf:type\s+',
            rf'\1 {RDF_TYPE} ',
            sparql,
        )

        # Fix 2: Replace overview used ONLY when it's the sole "name" variable selected
        # and the entity type has no name attribute. This is conservative to avoid
        # breaking legitimate description/narrative queries.
        # Only replace Movie/attrs/overview when used in a "name-like" position
        overview_pattern = r'<https://cograph\.tech/types/Movie/attrs/overview>'
        if re.search(overview_pattern, sparql):
            # Check if the query is trying to get movie names (not filtering by overview content)
            # Heuristic: if overview appears in SELECT projection but not in FILTER
            select_part = sparql.split('WHERE')[0] if 'WHERE' in sparql else ''
            filter_uses_overview = 'overview' in sparql.split('FILTER')[1] if 'FILTER' in sparql else False
            if not filter_uses_overview:
                sparql = re.sub(overview_pattern, RDFS_LABEL[1:-1], sparql)

        # Fix 4: Rewrite type-assertion predicates to subclass-closure paths so a
        # query over a parent type returns subtype instances (ADR rule 2).
        # Deterministic, idempotent, no ontology lookup needed.
        from cograph_client.graph.ontology_queries import rewrite_type_predicate_to_closure
        sparql = rewrite_type_predicate_to_closure(sparql)

        # Fix 5: resolve attribute aliases (ADR 0002 §7) — a renamed attribute
        # keeps answering through its alias until backfill retires it. A None
        # or empty map (the default) leaves the query untouched.
        if alias_map:
            from cograph_client.graph.aliases import rewrite_query_attrs
            sparql = rewrite_query_attrs(sparql, alias_map)

        return sparql

    async def _fetch_alias_map(self, graph_uri: str) -> dict[str, str]:
        """Cached attribute-alias map for the tenant ontology graph (ADR 0002 §7).

        Failures degrade to an empty map — alias resolution never blocks /ask.
        """
        cached = _alias_cache.get(graph_uri)
        if cached and (time.time() - cached[1]) < ONTOLOGY_CACHE_TTL:
            return cached[0]
        from cograph_client.graph.aliases import fetch_alias_map
        try:
            alias_map = await fetch_alias_map(self.neptune, graph_uri)
        except Exception:
            alias_map = {}
        _alias_cache[graph_uri] = (alias_map, time.time())
        return alias_map

    @staticmethod
    def invalidate_cache(graph_uri: str) -> None:
        """Call after ingestion to clear the cached ontology for a graph."""
        _ontology_cache.pop(graph_uri, None)
        # Also clear any KG-specific cache entries
        keys_to_remove = [k for k in _ontology_cache if k.startswith(graph_uri)]
        for k in keys_to_remove:
            _ontology_cache.pop(k, None)
        # Alias map is keyed by the ontology graph URI alone
        _alias_cache.pop(graph_uri, None)
        # Invalidate embeddings
        svc = get_embedding_service()
        if svc:
            svc.invalidate(graph_uri)

    async def _rephrase_via_openrouter(self, question: str, bindings: list[dict], max_rows: int = 30) -> str:
        """Generate a 2-3 sentence narrative summary of SPARQL result bindings.

        Uses Llama 3.1 8B on Cerebras (via OpenRouter) for fast, cheap rephrase.
        Fails open: returns "" on any error so the main response is never broken.
        """
        if not self._openrouter_key:
            return ""

        try:
            # Build a compact tabular string from bindings
            if not bindings:
                table_str = "(no results)"
                truncation_note = ""
            else:
                rows = bindings[:max_rows]
                if rows:
                    cols = list(rows[0].keys())
                    lines = ["\t".join(cols)]
                    for row in rows:
                        lines.append("\t".join(str(row.get(c, "")) for c in cols))
                    table_str = "\n".join(lines)
                else:
                    table_str = "(no results)"
                truncation_note = (
                    f"\n(Showing {len(rows)} of {len(bindings)} total rows.)"
                    if len(bindings) > max_rows else ""
                )

            system_prompt = (
                "You are an analyst summarizing a database query result. Rules:\n"
                "- Lead with the specific count (e.g. 'Eleven founders match.').\n"
                "- If multiple rows share similar values, find the ONE row that stands out — "
                "different company, different prior company, or different category. "
                "Use that outlier as your hero example with its exact column values.\n"
                "- Keep to 2-3 sentences, max 80 words.\n"
                "- ONLY state facts visible in the rows. Never mix values from different rows.\n"
                "- Trust the row values as literal, authoritative facts. If a column has a value, "
                "that IS the answer for that column — never describe a present value as "
                "'unknown' or 'incomplete' just because it's a short code.\n"
                "- SEC filing type codes are canonical form names (e.g. D means Form D, "
                "10-K means annual report, 10-Q means quarterly, 8-K means material event, "
                "S-1 means IPO registration). State the code as-is — prefixing with 'Form' "
                "is fine; calling it unknown is not.\n"
                "- Do NOT use chatbot phrases like 'Sure!', 'Here you go', 'Great question'.\n"
                "- If the result is empty, say 'No matches found.' and stop.\n"
                "- Speak in plain English, not technical jargon."
            )

            user_prompt = (
                f"Question: {question}\n\n"
                f"Result ({len(bindings)} row{'s' if len(bindings) != 1 else ''}):\n"
                f"{table_str}{truncation_note}\n\n"
                "Summarize this result in 2-3 sentences."
            )

            t_rephrase = time.time()
            async with httpx.AsyncClient(timeout=10) as client:
                res = await client.post(
                    f"{OPENROUTER_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._openrouter_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "meta-llama/llama-3.1-8b-instruct",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_tokens": 300,
                        "temperature": 0.2,
                        "provider": {
                            "order": ["Cerebras", "Groq", "Nebius"],
                            "allow_fallbacks": True,
                        },
                    },
                )
                res.raise_for_status()
                data = res.json()
                narrative = data["choices"][0]["message"]["content"].strip()
            rephrase_ms = round((time.time() - t_rephrase) * 1000, 1)
            logger.info("narrative_rephrase_ok", rephrase_ms=rephrase_ms, rows=len(bindings))
            return narrative
        except Exception:
            logger.warning("narrative_rephrase_failed", exc_info=True)
            return ""

    async def _generate_sparql(self, question: str, ontology: str, graph_uri: str = "", error_feedback: str = "", examples_text: str = "") -> dict:
        prompt = build_generation_prompt(question, ontology, graph_uri, examples_text=examples_text)
        if error_feedback:
            prompt += f"\n\n{error_feedback}"

        if self._query_provider == "cerebras" and self._cerebras_key:
            return await self._generate_via_cerebras(prompt)
        if self._query_provider == "openrouter" and self._openrouter_key:
            return await self._generate_via_openrouter(prompt)
        if self._openrouter_key:
            return await self._generate_via_openrouter(prompt)
        return await self._generate_via_anthropic(prompt)

    async def _generate_via_cerebras(self, prompt: str) -> dict:
        """Generate SPARQL via Cerebras with structured output."""
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(
                "https://api.cerebras.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._cerebras_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._query_model,
                    "messages": [
                        {"role": "system", "content": SPARQL_GENERATION_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "max_completion_tokens": 512,
                    "temperature": 0,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "sparql_response",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "sparql": {"type": "string"},
                                    "explanation": {"type": "string"},
                                    "functions_needed": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["sparql", "explanation", "functions_needed"],
                                "additionalProperties": False,
                            },
                        },
                    },
                },
            )
            res.raise_for_status()
            data = res.json()
            return json.loads(data["choices"][0]["message"]["content"])

    async def _generate_via_openrouter(self, prompt: str) -> dict:
        """Generate SPARQL via OpenRouter (OpenAI-compatible API)."""
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._query_model,
                    "models": model_chain(self._query_model),
                    "messages": [
                        {"role": "system", "content": SPARQL_GENERATION_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1024,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "sparql_response",
                            "strict": True,
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "sparql": {"type": "string"},
                                    "explanation": {"type": "string"},
                                    "functions_needed": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": ["sparql", "explanation", "functions_needed"],
                                "additionalProperties": False,
                            },
                        },
                    },
                },
            )
            res.raise_for_status()
            data = res.json()
            text = data["choices"][0]["message"]["content"]
            # Strip code fences if present
            stripped = text.strip()
            if stripped.startswith("```"):
                lines = [l for l in stripped.split("\n") if not l.strip().startswith("```")]
                stripped = "\n".join(lines)
            return json.loads(stripped)

    async def _generate_via_anthropic(self, prompt: str) -> dict:
        """Fallback: generate SPARQL via Anthropic API."""
        message = await self.anthropic.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=SPARQL_GENERATION_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sparql": {"type": "string", "description": "The SPARQL SELECT query"},
                            "explanation": {"type": "string", "description": "Brief explanation of what the query does"},
                            "functions_needed": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of function names if computation is needed",
                            },
                        },
                        "required": ["sparql", "explanation", "functions_needed"],
                        "additionalProperties": False,
                    },
                },
            },
        )
        return json.loads(message.content[0].text)

    @staticmethod
    def _humanize_uri(uri: str) -> str:
        """Extract a human-readable name from an Omnix URI.

        Examples:
            https://cograph.tech/entities/Movie/12345 → 12345
            https://cograph.tech/types/Movie → Movie
            https://cograph.tech/entities/ConsumerComplaint/1431838 → 1431838
        """
        from urllib.parse import unquote
        path = unquote(uri.replace("https://cograph.tech/", ""))
        return path.split("/")[-1]

    async def _resolve_uri_labels(self, bindings: list[dict]) -> dict[str, str]:
        """Batch-resolve rdfs:label for all Omnix entity/type URIs in bindings.

        Returns a mapping from URI → human-readable label.
        Falls back to extracting the last URI path segment if no label is found.
        """
        # Collect all unique URIs that look like Omnix entities or types
        uris: set[str] = set()
        for row in bindings:
            for v in row.values():
                if isinstance(v, str) and (
                    v.startswith("https://cograph.tech/entities/")
                    or v.startswith("https://cograph.tech/types/")
                ):
                    uris.add(v)

        if not uris:
            return {}

        resolved: dict[str, str] = {}

        # Batch SPARQL query to fetch rdfs:label for all URIs at once
        values_clause = " ".join(f"<{u}>" for u in uris)
        label_query = (
            f"SELECT ?uri ?label WHERE {{ "
            f"VALUES ?uri {{ {values_clause} }} "
            f"?uri <http://www.w3.org/2000/01/rdf-schema#label> ?label . "
            f"}}"
        )
        try:
            raw = await self.neptune.query(label_query)
            _, label_bindings = parse_sparql_results(raw)
            for row in label_bindings:
                uri = row.get("uri", "")
                label = row.get("label", "")
                if uri and label:
                    resolved[uri] = label
        except Exception:
            logger.debug("uri_label_resolution_failed", uri_count=len(uris), exc_info=True)

        # Fall back to path extraction for any URIs that weren't resolved
        for uri in uris:
            if uri not in resolved:
                resolved[uri] = self._humanize_uri(uri)

        return resolved

    async def _format_answer(self, bindings: list[dict], explanation: str) -> str:
        if not bindings:
            return "No results found."

        # Resolve any entity/type URIs to human-readable labels
        uri_labels = await self._resolve_uri_labels(bindings)

        def _display(value: str) -> str:
            """Return the display form of a binding value, resolving URIs."""
            return uri_labels.get(value, value)

        if len(bindings) == 1 and len(bindings[0]) == 1:
            value = list(bindings[0].values())[0]
            return _display(str(value))
        lines = []
        for row in bindings[:20]:
            parts = [f"{k}: {_display(v)}" for k, v in row.items()]
            lines.append(", ".join(parts))
        result = "\n".join(lines)
        if len(bindings) > 20:
            result += f"\n... and {len(bindings) - 20} more results"
        return result
