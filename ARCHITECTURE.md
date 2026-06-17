# Omnix Architecture

Knowledge graph platform that ingests structured and unstructured data, resolves it
against a shared ontology, stores it in Neptune (RDF), and answers natural language
questions via LLM-generated SPARQL.

## Design Principles

1. **Ontology-first.** Every piece of data maps to typed entities with attributes and
   relationships. No free-floating key-value pairs. Types are PascalCase, attributes
   are snake_case, relationships connect entities to entities.

2. **Entity-first over attribute-first.** Real-world things (City, Organization, Person)
   become entities with their own URIs, not string attributes on other entities. A
   property listing's city is a relationship to a City entity, not a `city_name` string.

3. **Schema-on-write.** The ontology grows automatically during ingestion. New types,
   attributes, and relationships are created as data arrives. No upfront schema design
   required. The TypeMatcher ensures new data connects to existing types when possible.

4. **LLM for ambiguity, deterministic for everything else.** LLMs handle extraction,
   type matching, and query generation. But row mapping, validation, triple insertion,
   and URI construction are fully deterministic. One LLM call per schema inference, zero
   LLM calls per row.

5. **Multi-tenant, multi-KG.** Each tenant has a shared ontology graph and separate
   named graphs per knowledge graph. Types are reusable across KGs within a tenant.
   A City entity ingested from Zillow data is the same City entity used by clinical
   trials data.

6. **Eval-driven development.** Every change to the query pipeline or ingestion logic
   is validated against a 4-tier eval framework with LLM judges. Ground truth is
   computed from the full source dataset, not samples.

## System Architecture

```
                    CSV / JSON / Text
                           |
                    [LLM Extraction]
                           |
                    [Schema Resolver]
                     /      |       \
              TypeMatcher  AttrResolver  Validator
                     \      |       /
                    [Neptune SPARQL]
                     /            \
              Ontology Graph    KG Instance Graphs
                     \            /
                    [NL Query Pipeline]
                     /      |       \
              Embeddings  LLM Gen  SPARQL Exec
                           |
                      NL Answer
```

## Ingestion Pipeline

### CSV Ingestion (two-step, no LLM per row)

**Step 1 — Schema inference** (`omnix/resolver/csv_resolver.py:90-175`)

One LLM call. Input: column headers, 5 sample rows, existing ontology types.
Output: entity type name + per-column mapping (role, datatype, target type).

Column roles:
- `type_id` — unique identifier for each row's entity
- `attribute` — scalar value stored on the entity (string, integer, float, boolean, datetime)
- `relationship` — creates a link to another entity type

Post-processing enforces entity-first principles: geographic columns (city, state,
country) and people columns (broker, agent) are forced to relationships even if the
LLM mapped them as attributes.

**Wide tables (COG-58).** Above `OMNIX_CSV_MAX_INFERENCE_COLUMNS` columns (default 40)
the per-column REASON pass is split so no single LLM call must emit a tag for every
column within its output-token budget (the ~300-column failure mode: truncated JSON →
validation retry → 422/120s timeout). It becomes one global entity-decomposition call
(output bounded by entity count) followed by chunked column-assignment calls of at most
that many columns each (run concurrently under a small semaphore), merged back into the
same `{entities, columns, relationships}` shape. Coverage is guaranteed deterministically:
any column the model drops is backfilled as an attribute of the first entity, so every
header is tagged exactly once (row conservation, ADR 0003 §2). Each pass's output-token
budget also scales with the column count so the REFUTE/COMPLETE echoes aren't truncated.
Note: *type matching* (`type_matcher.py`) is **not** a wide-table bottleneck — it resolves
entity *type names*, not columns, and scales with the (small) distinct-type count via its
cache + exact-name short-circuit, so no per-column concurrency cap is needed there.

**Step 2 — Row mapping** (`omnix/resolver/csv_resolver.py:241-306`)

Fully deterministic. No LLM. Each row produces:
- One primary entity (the row itself)
- Stub entities for each relationship target (with a name attribute from the cell value)
- Attribute triples for scalar columns
- Relationship triples linking primary entity to targets

**Pipe-delimited value splitting:** Values containing `|` (e.g., `"Breast Cancer|Lung Cancer"`)
are split into separate triples. For relationships, each value becomes a separate entity
and relationship triple. For string attributes, each value becomes a separate attribute
triple. This enables exact-match SPARQL filters without CONTAINS.

**Step 3 — Resolution + insertion** (`omnix/resolver/schema_resolver.py:200+`)

For each batch of entities:
1. **Type matching** — is this type new, or does it match an existing type?
2. **Attribute resolution** — do these attributes already exist on the type?
3. **Batch dedup** — check which entity URIs already exist in Neptune (one SPARQL per 500 URIs)
4. **Validation** — check each triple value against the expected datatype, coerce if needed
5. **Batch insert** — write triples to Neptune (500 triples per INSERT DATA statement)

### Text/JSON Ingestion

Text is chunked (3000 chars, 200 overlap) via `omnix/resolver/chunker.py`. Each chunk
gets one LLM extraction call. Results are deduplicated by entity ID across chunks, then
follow the same resolution + insertion path.

### Enrichment (optional pre-processing)

`scripts/enrich_csv.py` — Two-phase LLM enrichment before ingestion:

**Phase 1 — Schema design** (1 LLM call): Analyze 10 sample rows and decide what
additional attributes to infer. The LLM proposes attribute names, types, vocabularies,
and derivation sources. No hardcoded fields — fully dataset-agnostic.

**Phase 2 — Row enrichment** (N concurrent LLM calls): Infer values for each row
using the Phase 1 schema. Runs at 20 concurrency via asyncio. Output is an enriched
CSV with new columns appended.

Example: For SF events data, the LLM independently proposed extracting neighborhood,
event_category, event_format, audience_tags, and time_of_day. For clinical trials
it would propose different attributes.

## Type Matching

`omnix/resolver/type_matcher.py` — 4-layer cascade, fast to slow:

| Layer | Speed | When Used | How |
|-------|-------|-----------|-----|
| Verdict cache | <1ms | Always checked first | JSON file lookup by (proposed, existing) pair |
| Embedding similarity | ~50ms | Cache miss | Cosine similarity between type name embeddings |
| LLM single judge | ~500ms | Ambiguous similarity | Claude claude-sonnet-4-6, structured output |
| 3-judge fan-out | ~1500ms | Low-confidence single judge | 3 independent calls, majority vote |

**Embedding thresholds:**
- >= 0.92 → SAME (skip LLM)
- < 0.55 → DIFFERENT (skip LLM)
- 0.55–0.92 → Pass top-3 candidates to LLM judge

**Verdicts:**
- `SAME` — proposed type matches existing, reuse it
- `SUBTYPE` — proposed type is a specialization (rdfs:subClassOf)
- `DIFFERENT` — genuinely new type, create it
- `FLAGGED` — 3-way judge split, needs human review

**Key rule:** subClassOf means "is a kind of" (type hierarchy), NOT geographic
containment. City is NOT a subtype of State.

## Ontology Model

### URI Patterns

| Thing | Pattern | Example |
|-------|---------|---------|
| Type | `https://omnix.dev/types/{TypeName}` | `https://omnix.dev/types/ClinicalTrial` |
| Attribute | `https://omnix.dev/types/{TypeName}/attrs/{attr}` | `https://omnix.dev/types/Event/attrs/name` |
| Relationship | `https://omnix.dev/onto/{predicate}` | `https://omnix.dev/onto/city` |
| Entity | `https://omnix.dev/entities/{TypeName}/{safe_id}` | `https://omnix.dev/entities/City/Austin` |
| Tenant graph | `https://omnix.dev/graphs/{tenant_id}` | `https://omnix.dev/graphs/demo-tenant` |
| KG graph | `https://omnix.dev/graphs/{tenant_id}/kg/{kg}` | `https://omnix.dev/graphs/demo-tenant/kg/zillow-austin` |

### Named Graph Structure

```
Tenant graph (ontology):
  - Type definitions (rdfs:Class, rdfs:label, rdfs:subClassOf)
  - Attribute definitions (rdf:Property, rdfs:domain, rdfs:range)
  - KG metadata (kg_name, kg_description)

KG instance graph (data):
  - Entity triples (rdf:type, attributes, relationships)
  - Provenance triples (ingested_at, source, batch_id)
```

Types and attributes live in the tenant graph and are shared across all KGs.
Instance data lives in KG-specific graphs. This means a City type defined
during Zillow ingestion is reusable when ingesting clinical trials.

### Datatype Handling

- Strings: plain literals (`"Austin"`)
- Numbers: `"500000"^^xsd:integer`, `"99.99"^^xsd:float`
- Booleans: `"true"^^xsd:boolean`
- Dates: `"2018-11-26T00:00:00"^^xsd:dateTime` (always normalized with time component)
- Relationships: object is an entity URI, not a literal

DateTime values are always normalized to full ISO-8601 with time component in
`validator.py:_typed_value`. This is required for Neptune xsd:dateTime comparisons
to work — storing `"2018-11-26"^^xsd:dateTime` (without time) causes silent comparison
failures.

## Query Pipeline

`omnix/nlp/pipeline.py` — NLQueryPipeline

### Flow

```
Question → Ontology Retrieval → Example Retrieval → SPARQL Generation → Validation → Execution → Formatting
```

**Step 1 — Ontology retrieval** (two modes):
- **Semantic** (preferred, ~300ms): Embed the question, cosine similarity against all
  type embeddings (top-15), expand 1 hop on relationship graph. Uses OpenRouter
  `text-embedding-3-small` (1536 dims).
- **Full** (fallback, 2-5s): Fetch all types from Neptune, discover enum values for
  low-cardinality string attributes via concurrent queries (asyncio.gather, not UNION).
  Cached for 60s.

**Step 2 — Example retrieval** (`omnix/nlp/example_bank.py`): Before generating
SPARQL, retrieve 3 similar working queries from the example bank. The bank stores
(question, SPARQL, embedding) pairs from previous eval runs. Retrieval algorithm:
embed the question, cosine similarity against bank, anti-cheat filter (exclude >0.95
similarity), cross-dataset preference, pattern diversity (pick examples with different
SPARQL patterns: count vs join vs filter). This is RAG for SPARQL — concrete examples
teach patterns better than abstract rules.

**Step 3 — SPARQL generation**: LLM generates a SELECT query using the ontology +
examples as context. Default model: `llama3.1-8b` via Cerebras. Configurable to
OpenRouter or Anthropic models. Structured JSON output with sparql, explanation,
functions_needed.

**Step 4 — Post-processing**: `_fix_attribute_uris` fuzzy-matches generated URIs
against the ontology to catch LLM URI mistakes (wrong namespace, missing /attrs/).

**Step 4.5 — SPARQL normalization** (`omnix/nlp/validator.py:normalize_sparql`):
Auto-fixes common LLM syntax mistakes before execution:
- Expands PREFIX declarations inline
- Moves FROM clauses to correct position
- Fixes bare aggregates (`SELECT COUNT(?x)` → `SELECT (COUNT(?x) AS ?count)`)
- Corrects `omnix.dev/` URI namespace mistakes

**Step 5 — Validation + execution**: Syntax check, then execute against Neptune.
On failure, retry with error feedback (up to 3 attempts).

**Step 6 — Formatting**: Parse SPARQL results, format for human readability
(attribute values, not entity URIs).

### SPARQL Generation Rules

Key rules in the system prompt (`omnix/nlp/prompts.py`):
- No PREFIX declarations, write full URIs
- Only use URIs that appear in the ontology
- Always include `FROM <graph_uri>` clause
- For relationship filtering, always traverse to entity name attribute with
  `FILTER(CONTAINS(LCASE(?name), "value"))`. Entity names may contain pipe-delimited
  multi-values. Use the exact phrasing from the user's question, never rephrase.
- Aggregates must be aliased: `SELECT (COUNT(?x) AS ?count)`, never bare `SELECT COUNT(?x)`
- For datetime comparisons, use full ISO-8601 format
- For enum attributes, use exact values as shown in ontology

### Enum Value Discovery

During full ontology fetch, cardinality is checked for ALL attributes and relationships
via concurrent SPARQL queries (one per predicate, via asyncio.gather). Concurrency is
**bounded by a semaphore** (`OMNIX_ENUM_DISCOVERY_CONCURRENCY`, default 8 — COG-58): a
wide table with hundreds of attributes would otherwise launch hundreds of simultaneous
queries and throttle serverless Neptune. The in-flight query count is now capped
regardless of column count, trading a little latency for stability.

- **Zero-cardinality predicates are hidden.** If an attribute or relationship has no data
  in Neptune, it is excluded from the ontology summary. This prevents the LLM from
  generating SPARQL against empty predicates (e.g., `attendees` with 0 data when
  `attendees_count` has 872 records, or `audience_type` with 0 data when `audience_tags`
  has data).
- **Low-cardinality string attributes** (<= 25 unique values) have their actual values
  included in the ontology context (e.g., `[values: "RECRUITING", "COMPLETED", "TERMINATED"]`).
- **High-cardinality attributes** show the unique count (e.g., `[824 unique values]`).

Neptune UNION queries scale linearly and are slower than parallel individual queries
for this use case.

## Embedding Service

`omnix/nlp/ontology_embeddings.py`

- **Model:** `openai/text-embedding-3-small` via OpenRouter
- **Dimensions:** 1536
- **Batch size:** 100 texts per API call
- **Storage:** In-memory dict keyed by graph URI, optional S3 persistence
- **Retrieval:** Top-K cosine similarity + 1-hop relationship expansion
- **Rebuild:** Triggered via `POST /graphs/{tenant}/embeddings/build` or automatically
  after ingestion for new/changed types

Each type is embedded as a text chunk containing the type name, its attributes
(with datatypes), and its relationship targets.

## Example Bank (RAG for SPARQL)

`omnix/nlp/example_bank.py` — Few-shot examples from eval history.

The example bank replaces static prompt rules with concrete working SPARQL examples.
Instead of teaching the LLM "use CONTAINS for entity names" (abstract), it shows
a real query that uses CONTAINS (concrete). The LLM adapts the pattern to the
current ontology.

- **Storage:** `eval_reports/example_bank.jsonl` with embedded question vectors
- **Model:** `openai/text-embedding-3-small` (1536 dims, same as ontology embeddings)
- **Cap:** 500 examples max (balanced across KGs)
- **Source:** `eval_reports/finetune_pairs.jsonl` — deduped correct (question, SPARQL) pairs

### Lifecycle: Auto-sync with Ontology

The example bank must stay in sync with the ontology. Stale examples (referencing
old predicate URIs or wrong datatypes) cause regressions because the LLM copies
broken SPARQL patterns.

**Auto-purge on KG delete** (`omnix/api/routes/knowledge_graphs.py`):
When `DELETE /kgs/{name}` is called, all examples for that KG are removed from
the bank. This prevents stale SPARQL patterns from poisoning few-shot retrieval
after reingest. The clear → reingest cycle starts with a clean slate.

**Auto-rebuild on eval completion** (`omnix/eval.py:run_full_eval`):
After each eval run saves correct pairs to `finetune_pairs.jsonl`, the example
bank is automatically rebuilt from ALL pairs with fresh embeddings. Every eval
round produces a better bank. The flow:

```
clear KG → bank purges stale examples for that KG
reingest KG → fresh ontology types, rdfs:label on entities
run eval → correct pairs saved → bank auto-rebuilt from all pairs
```

### Retrieval Algorithm

1. Embed the incoming question
2. Cosine similarity against all examples → top-10 candidates
3. **Anti-cheat:** Exclude any example with >0.90 similarity to excluded questions
4. **Cross-dataset preference:** Examples from different KGs score higher (0.9x penalty for same-KG)
5. **Same-dataset gate:** Same-KG examples must have similarity <0.75
6. **Pattern diversity:** Pick 3 examples with different pattern tags (count, join, filter, avg, etc.)

### Cross-domain Examples (by design)

The example bank intentionally uses cross-domain examples. When querying an IMDB
knowledge graph, the bank may return a working SPARQL example from the Coffee or
Events SF domain. This is not cheating. It is pattern transfer.

**What transfers:** SPARQL structural patterns (COUNT + JOIN, FILTER by relationship,
GROUP BY + HAVING, subqueries). The LLM sees "here's how to count entities filtered
by a named relationship" and adapts the pattern to the current ontology's types and
predicate URIs.

**What does NOT transfer:** Answer values, entity names, specific predicate URIs.
The LLM must still read the current ontology to generate correct URIs. A Coffee
example's `<https://omnix.dev/types/CoffeeLot/attrs/altitude>` is useless for
IMDB queries. Only the SPARQL structure carries over.

**Why this works:** A human developer does the same thing. You look at a working
query from a different project, understand the pattern, adapt it. Cross-domain
examples are a structural hint, not an answer key.

**Guard rails in place:**
- Same-KG examples with >0.75 similarity are blocked (prevents near-duplicate leaking)
- Same-KG examples get a 10% score penalty (prefers cross-domain)
- Pattern diversity ensures the LLM sees varied SPARQL structures, not 3 of the same type

### Anti-cheat for Evals

During eval, the system must not "cheat" by retrieving the exact question being
tested from the example bank. This would produce high scores without real
generalization.

**How it works:** The eval passes ALL eval question texts as `exclude_questions`
to the `/ask` endpoint (`omnix/models/query.py:NLQuery.exclude_questions`). The
pipeline passes these to `bank.retrieve()` (`omnix/nlp/pipeline.py`), which
excludes any bank example with >0.90 cosine similarity to any excluded question.

**Enforcement chain:**
```
eval.py: all_eval_questions = [q["question"] for q in questions]
  → body["exclude_questions"] = all_eval_questions
    → /ask endpoint: body.exclude_questions
      → pipeline.ask(exclude_questions=...)
        → bank.retrieve(exclude_questions=...)
          → cosine similarity > 0.90 → excluded
```

**Production queries do NOT exclude anything.** In production, using similar
examples is the correct behavior. It is RAG working as designed. The anti-cheat
constraint only applies during eval so scores reflect true generalization ability.

**Result:** Eval measures the floor (no example help for test questions). Production
gets the ceiling (examples boost accuracy for real user queries).

### Pattern Tags

Auto-detected from SPARQL text: count, avg, max, sum, filter, contains,
date_filter, join, multi_hop, group_by. Used for diversity selection so
the LLM sees varied patterns, not 3 of the same type.

## Failure Diagnosis

`omnix/eval_diagnosis.py` — Classifies failures by layer.

Three-stage triage per failed question:
- **Stage A — Graph Probe:** Query Neptune to check if expected data exists (~200ms)
- **Stage B — Pattern Match:** Rules-based classification (pipe chars, count gaps, case mismatch)
- **Stage C — LLM Fallback:** For ambiguous cases, LLM classifies with structured output

Output: `FailureDiagnosis(layer, sub_category, confidence, signature)` where layer
is one of ingestion/ontology/query. The signature enables pattern grouping across
questions that fail for the same root cause.

## Autonomous Eval Loop

`scripts/eval_loop_v2.py` — Multi-layer improvement loop.

Three-phase architecture, cheapest fixes first:
- **Phase 1: Query fixes** — prompt edits, ~0s per fix
- **Phase 2: Ontology fixes** — SPARQL UPDATE, ~1s per fix
- **Phase 3: Ingestion fixes** — delete + reingest, ~30min

Anti-overfitting safeguards:
- Monotonic pass set: questions passing 3+ consecutive times are "locked"
- Prompt length budget: 6000 chars max
- Cross-dataset validation: fix on one KG, smoke test on another
- Failure pattern tracking: prevents re-attempting same class of fix

## Eval Framework

`omnix/eval.py` — Iterative eval loop for measuring query accuracy.

### Question Generation (4 tiers)

| Tier | Complexity | Example |
|------|-----------|---------|
| 1 | Count/Lookup | "How many events are there?" |
| 2 | Filter | "How many events have more than 100 attendees?" |
| 3 | Join | "Which host has the most events?" |
| 4 | Multi-hop | "Average enrollment for Phase 2 trials in California?" |

Questions are generated by an LLM using full dataset statistics (not samples).
5 questions per tier, 20 total per eval run (configurable via `-n`).

### Ground Truth

**Q2Forge (primary, 100% reliable):** `scripts/q2forge.py` generates question-SPARQL
pairs, executes gold SPARQL against Neptune, and keeps only verified results. Ground
truth confidence is 100% because answers come from actual SPARQL execution.

**Legacy (pandas, ~70-80% reliable):** Computed from the full source CSV using pandas
expressions generated by an LLM judge. Kept for backward compatibility but Q2Forge
is preferred for all new eval work.

**Verified Ground Truth Location:** `eval_reports/verified_ground_truth/{kg}-{n}.json`

### Scoring

- Counts: +/- 2% tolerance
- Averages/sums: +/- 5% tolerance
- Verdicts: correct, partial, wrong, error
- Failure categories: bad_predicate_uri, missing_join, wrong_filter, wrong_aggregation,
  empty_result, uri_instead_of_value, other

### Models

- **Q2Forge question generation:** google/gemini-2.5-flash (OpenRouter)
- **Legacy question generation:** deepseek-v3.2 (OpenRouter)
- **Eval judge:** Fast programmatic judge (primary) or deepseek-v3.2 (OpenRouter)
- **Query model (SPARQL generation):** google/gemini-2.5-flash via OpenRouter (default)

### Eval Scripts

| Script | Purpose |
|--------|---------|
| `scripts/q2forge.py` | Generate execution-verified ground truth |
| `scripts/eval_baseline.py` | Run baseline eval with answer-equivalence |
| `scripts/cross_domain_eval.py` | Test cross-domain queries |
| `scripts/spider_bench.py` | Spider4SPARQL benchmark runner |
| `scripts/consolidate_finetune.py` | Consolidate fine-tuning data |
| `scripts/eval_loop_v2.py` | Multi-layer eval-fix loop |

### Current Scores (April 10, 2026)

| KG | Score | Questions |
|---|---|---|
| IMDB Movies | 100% | 20 |
| Video Games | 95% | 20 |
| CFPB Complaints | 95% | 20 |
| Exoplanets | 94% | 18 |
| Events SF | 90% | 20 |
| Universities | 90% | 20 |
| Olympics 2024 | 88% | 17 |
| Coffee Quality | 83% | 18 |
| **Overall** | **92.2%** | **153** |

## Infrastructure

Defined in `template.yaml` (AWS SAM).

### Components

| Component | Service | Config |
|-----------|---------|--------|
| Compute | ECS Fargate | 256 CPU / 512 MB (configurable) |
| Database | Neptune Serverless | 1–2.5 NCUs (configurable) |
| Load Balancer | ALB | Port 80, idle timeout 300s |
| Secrets | AWS Secrets Manager | Anthropic, OpenRouter, Cerebras keys |
| Functions | Lambda (x5) | Tier-2 compute functions |
| Network | VPC | 2 public + 2 private subnets |

### Deployment

Push to `main` or `feat/phase-1-core-platform` triggers `.github/workflows/deploy.yml`:
1. Build Docker image
2. Push to ECR (`omnix-demo-tenant`)
3. Force new ECS deployment (rolling update)

Deploy takes ~45s. Do NOT deploy during bulk ingestion — ECS rolling restart kills the
running container, causing 500 errors on in-flight requests.

### Neptune Operational Notes

- MinCapacity 1 NCU prevents full cold starts, but 500 errors still occur under
  sustained write load + ECS deploy simultaneously
- Writes should be batched (500 triples per INSERT DATA)
- Individual SPARQL queries are fast (~30ms). UNION queries scale linearly and should
  be avoided for batching — use asyncio.gather on individual queries instead.
- dateTime values must include time component for range comparisons to work

## Configuration

`omnix/config.py` — Pydantic Settings with `OMNIX_` env prefix.

| Setting | Default | Purpose |
|---------|---------|---------|
| `neptune_endpoint` | `http://localhost:8182` | Neptune SPARQL endpoint |
| `api_keys` | `{"dev-key-001": "demo-tenant"}` | API key → tenant mapping |
| `anthropic_api_key` | `""` | For type matching (claude-sonnet-4-6) |
| `openrouter_api_key` | `""` | For extraction, eval, embeddings |
| `cerebras_api_key` | `""` | For fast query generation |
| `function_arns` | `{}` | Lambda ARN mapping |
| `embeddings_s3_bucket` | `""` | S3 bucket for embedding persistence |
| `embeddings_s3_prefix` | `omnix/embeddings` | S3 key prefix |
| `embeddings_top_k` | `15` | Semantic retrieval result count |

Additional env vars (not in Settings):
- `OMNIX_QUERY_MODEL` — default `llama3.1-8b`
- `OMNIX_QUERY_PROVIDER` — default `cerebras`
- `OMNIX_EXTRACT_MODEL` — default `deepseek/deepseek-v3.2` (schema/entity extraction — the "propose" stage)
- `OMNIX_INFER_MODEL` — default `claude-sonnet-4-6` (Anthropic inference/extraction path for the v2 schema passes)
- `OMNIX_MATCH_MODEL` — default `claude-sonnet-4-6` (type matching: reuse-vs-expand verdict + ambiguous judge fan-out)
- `OMNIX_GOV_JUDGE_MODEL` — default `claude-sonnet-4-6` (OSS governance judge panel; premium `ShapeJudgePanel` uses `COGRAPH_GOV_JUDGE_MODEL`)
- `OMNIX_EVAL_MODEL` — default `deepseek/deepseek-v3.2`

## API Endpoints

All routes prefixed with `/graphs/{tenant}`.

| Method | Path | Rate Limit | Purpose |
|--------|------|------------|---------|
| GET | `/health` | — | Liveness check |
| POST | `/ingest` | 10/min | Ingest text/JSON |
| POST | `/ingest/csv/schema` | 10/min | Infer CSV schema |
| POST | `/ingest/csv/rows` | 30/min | Insert CSV rows |
| POST | `/ask` | 1000/min | Natural language query |
| POST | `/query` | 60/min | Raw SPARQL query |
| POST | `/update` | 30/min | Raw SPARQL update |
| GET | `/ontology/types` | — | List types |
| GET | `/ontology/types/{name}` | — | Type details |
| POST | `/ontology/types` | — | Create type |
| GET | `/kgs` | — | List knowledge graphs |
| POST | `/kgs` | — | Create KG |
| DELETE | `/kgs/{name}` | — | Delete KG + data |
| POST | `/embeddings/build` | 5/min | Rebuild all embeddings |

## Current State (April 2026)

### Knowledge Graphs

| KG | Entities | Triples | Domain |
|----|----------|---------|--------|
| clinical-trials | 5,000 | 218K | Medical |
| olympics-2024 | 6,012 | 59K | Sports |
| imdb-movies | 6,152 | 55K | Movies |
| events-sf | 3,199 | 42K | Events |
| exoplanets | 2,678 | 44K | Space |
| universities | 2,486 | 68K | Education |
| cfpb-complaints | 3,285 | 35K | Financial |
| video-games | 1,852 | 24K | Entertainment |
| zillow-austin | ~1,000 | 21K | Real Estate |
| coffee-quality | 1,616 | 21K | Food |

10 KGs, ~33,000 entities, ~587K total triples across 10 domains.

### Known Limitations

1. **Entity normalization** — "TX" and "Texas" create separate State entities. No alias
   resolution or abbreviation expansion during ingestion.

2. **Ground truth false negatives** — the pandas-based ground truth computation
   sometimes generates wrong expected values for enriched columns (computing ratios
   instead of counts, regex failures on parentheses in values).

3. **Eval question quality** — the eval question generator references exact attribute
   names and enum values rather than generating natural language questions a real user
   would ask.

4. **Deploy-time eval failures** — ECS rolling deploys take ~60s. Evals run during
   this window get 500/timeout errors. Wait for `gh run` completion before eval.

5. **Text ingestion** — de-scoped for early release. The extraction prompt exists but
   has no fact-vs-noise filtering. CSV ingestion is the supported path.
