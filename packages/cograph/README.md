# cograph

Node.js SDK and CLI for [Cograph](https://cograph.cloud) — turn raw data into a queryable knowledge graph.

## Quickstart

```bash
npx cograph
```

That's it. The first run opens your browser to sign in, saves a key to `~/.cograph/config.json`, and drops you into the interactive shell:

```text
  /ingest <file>      Ingest a CSV/JSON/text file
  /ask <question>     Ask in natural language
  /kg list|switch|create|delete <name>
  /types [query]      Types in the active KG, with entity counts
  /type <name>        Drill into one type — attributes & relationships
  /enrich <Type> <attrs...>   Plan + run an enrichment job (interactive)
  /enrich watch <job_id>      Live progress for a running job
  /enrich jobs                List recent enrichment jobs
  /enrich review <job_id>     Walk through conflicts and accept/reject
  /status             Graph stats
  /login              Re-authenticate
  /quit
```

Bare lines (no leading `/`) auto-route to `/ask`. Full walkthrough at [cograph.cloud/docs/quickstart](https://cograph.cloud/docs/quickstart).

## Self-hosted mode

Pointing the CLI at your own backend skips the browser sign-in:

```bash
cograph --local                         # defaults to http://localhost:8000
cograph --no-login                      # uses COGRAPH_API_URL env var
COGRAPH_API_URL=http://my-host:8000 cograph
```

When self-hosted, the prompt shows the host suffix: `cograph@localhost:8000 (kg) ▸`. Bare `cograph` still triggers the hosted-version login flow.

## Auto-enrichment

Fill and verify attributes on entities of a given type by looking them up in external sources, with a human review step before any write:

```text
> /enrich LineItem brand manufacturer
Plan: enrich LineItem.brand, .manufacturer · tier: lite · policy: stage
Job queued: enr_xxxxxxxx · 12,450 entities
[████████████████████] filled 6,200 · verified 1,400 · conflicts 320
Status: review · 320 conflicts pending. Run /enrich review enr_xxxxxxxx
```

Use `/enrich watch <job_id>` for live progress, `/enrich jobs` to list recent jobs, and `/enrich review <job_id>` to walk through conflicts and accept/reject each one. The `lite` tier uses Wikidata only (free, no API key).

## Install

```bash
npm install cograph        # or: npm install -g cograph
```

Requires Node 20+.

## Browsing what got ingested

After ingest, look around before asking questions:

```text
cograph (mentors) [37,715] ▸ /types
  Type           Entities
  Mentor              988
  Skill               412
  Industry             38

cograph (mentors) [37,715] ▸ /type Mentor
  Mentor  1,000 entities

  Attributes (6)
    .name           string      988  ( 99%)
    .level          string      714  ( 71%)
    ...

  Relationships (6)
    .title         → JobTitle    988  ( 99%) (+775 string)
    .skills        → Skill       987  ( 99%)
    ...
```

`/types <query>` filters by substring; `/type <name>` accepts case-insensitive prefix. Auto-attached system metadata (`rdfs:label`, `ingested_at`, `source`) is hidden by default — pass `--system` to see it. The `(+775 string)` annotation appears when the resolver produced both a literal value and a typed-entity link for the same column.

## SDK

```ts
import { Client, CographError } from "cograph";

const client = new Client({ apiKey: process.env.COGRAPH_API_KEY });

await client.ingest("sales.csv", { kg: "sales" });
const result = await client.ask("What's the average deal size by region?", { kg: "sales" });
console.log(result.answer);
```

### Constructor

```ts
new Client({
  apiKey?: string,    // env: COGRAPH_API_KEY
  baseUrl?: string,   // env: COGRAPH_API_URL (default: https://api.cograph.cloud)
  tenant?: string,    // env: COGRAPH_TENANT (default: demo-tenant)
})
```

### Methods

- `ingest(pathOrText, { kg?, contentType? })` — auto-detects CSV by extension and uses the two-step schema/rows flow; otherwise sends raw content.
- `ask(question, { kg? })` — returns `{ answer, sparql?, ... }`.
- `listKgs()`, `createKg(name, description?)`, `deleteKg(name)` — knowledge-graph CRUD.
- `ontologyTypes()` — list every type in the tenant ontology with attributes and parents.
- `ontologyResolve(ask, { knowledge_graph? })` — resolve a fuzzy natural-language ontology-evolution ask (no exact type/attribute/relationship names needed) against the current schema. Returns `{ applied, proposals, summary }`; high-confidence changes land automatically, ambiguous/new-type ones come back as `proposals`.
- `ontologyApply(proposal)` — confirm and commit a single `ResolvedChange` from `ontologyResolve`'s `proposals`. Pass the object through unchanged; returns `{ applied, operations, summary }`.
- `typeCounts(kg)` — `[{ name, entity_count }]` for the given KG, sorted desc. Powers `/types`.
- `typeUsage(kg, name, { includeSystem? })` — full breakdown for one type: attributes (with usage counts), relationships, and 3 sample entities. Powers `/type`. System predicates filtered by default.
- `exploreRecords(kg, type, { limit?, cursor? })` — one keyset-paginated page of entity instances (`{ columns, rows, total, next_cursor }`).
- `exploreTypeEdges(kg)` — undirected type→type edges for an overview graph (`[{ source, target, weight }]`).
- `normalizeSuggest(kg, type)`, `normalizeRules({ kg?, status? })`, `normalizeConfirmRule(id)`, `normalizeRejectRule(id)`, `normalizeApplyRule(id)` — inferred-normalization rule lifecycle.
- `ontologyRecommend(body?)` — recommend ontology relationships/changes for a KG.

All errors throw `CographError`.

### Raw / passthrough API (`client.raw.*`)

Every method above throws on a non-2xx status and some reshape the payload
(e.g. `listKgs()` unwraps `{ kgs: [] }`). When you instead want the backend
`Response` **verbatim** — to forward it 1:1 (e.g. from a web proxy route) or to
branch on `status` without a `try/catch` — use the `raw` namespace. Each raw
method maps to one canonical operation with the path encoded inside the SDK, so
callers pass **no path string**:

```ts
const client = new Client({ apiKey, tenant });

// Forward the backend response unchanged from a proxy:
const res = await client.raw.enrichJobs();          // GET …/enrich/jobs
return new Response(res.body, { status: res.status, headers: res.headers });

// A non-2xx is a Response, not a throw — and the body is never reshaped:
const r = await client.raw.enrichJob("missing");
if (r.status === 404) { /* … */ }
```

`client.raw` covers agent, ask, ingest (+ csv schema/rows), enrich jobs
(create/list/get/conflicts/apply/cancel), ontology (types/resolve/recommend/apply),
kgs (list/create/delete), explore (summary/records/type-edges/type-counts/search),
normalize (suggest/rules GET+POST/confirm/reject/apply) and tenants
(list/create/delete). Each returns `Promise<Response>` and only ever rejects on a
network error or timeout (i.e. when there is no HTTP response to return).

## One-shot CLI

For scripts and CI — every command is a single HTTP round-trip:

```bash
# List / create / delete knowledge graphs
npx cograph kg list
npx cograph kg create my-data --description "demo"
npx cograph kg delete my-data

# Ingest data
npx cograph ingest data.csv --kg my-data
npx cograph ingest --text "Alice works at Acme" --kg my-data

# Ask questions
npx cograph ask "How many companies?" --kg my-data
npx cograph ask "Top 5 deals" --kg my-data --debug

# Ontology + clear
npx cograph ontology types
npx cograph clear --kg my-data --yes
```

### Environment

- `COGRAPH_API_KEY` — required for headless / CI use; interactive `cograph login` writes one to `~/.cograph/config.json` automatically.
- `COGRAPH_API_URL` — default `https://api.cograph.cloud`.
- `COGRAPH_TENANT` — default `demo-tenant`. The login flow sets this to your user ID.

Legacy `OMNIX_*` vars are also accepted.

> PDF ingestion is not yet supported in the Node CLI. Use the Python CLI or POST raw bytes to the API.

## License

MIT
