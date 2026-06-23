import { existsSync, readFileSync, statSync } from "node:fs";
import { extname } from "node:path";
import { readConfig } from "./config.js";

export class CographError extends Error {
  status?: number;
  body?: string;

  constructor(message: string, opts?: { status?: number; body?: string }) {
    super(message);
    this.name = "CographError";
    this.status = opts?.status;
    this.body = opts?.body;
  }
}

export interface ClientOptions {
  apiKey?: string;
  baseUrl?: string;
  tenant?: string;
}

export interface IngestOptions {
  kg?: string;
  contentType?: "text" | "csv" | "json" | string;
  /** Rows per batch for CSV ingest. Default 200. Larger = fewer round-trips
   *  but higher per-request memory; 200 is a good balance for typical KGs. */
  batchSize?: number;
  /** Max number of batches in flight at once. Default 4. Higher saturates
   *  the backend faster but risks 429s on large ingests. */
  concurrency?: number;
  /** Called after each batch completes during CSV ingest, in batch order.
   *  Use for progress UI. Not invoked for text/json ingest. */
  onProgress?: (progress: IngestProgress) => void;
  /** CSV only. Called once after schema inference and BEFORE any rows are
   *  written, with the inferred mapping. Return the (possibly edited/approved)
   *  mapping to ingest, or `null` to cancel without writing anything. When
   *  omitted the inferred mapping is applied as-is (non-interactive). This is
   *  the same confirm/override gate the Explorer surfaces in its review step. */
  onSchemaInferred?: (
    mapping: Record<string, unknown>,
    info: { totalRows: number; rowsProfiled: number },
  ) => Promise<Record<string, unknown> | null>;
}

export interface IngestProgress {
  rowsProcessed: number;
  totalRows: number;
  entitiesResolved: number;
  triplesInserted: number;
}

/** Rows sent to schema inference. Profile fidelity = decision quality, so we
 *  send the whole file up to this cap, evenly strided across it (never the
 *  head — head-of-file bias is exactly what evidence-grounded inference fixes).
 *  Matches the Explorer's SCHEMA_SAMPLE_CAP. */
export const SCHEMA_SAMPLE_CAP = 5000;

function stridedSample<T>(rows: T[], cap: number = SCHEMA_SAMPLE_CAP): T[] {
  if (rows.length <= cap) return rows;
  const out: T[] = [];
  for (let i = 0; i < cap; i++) out.push(rows[Math.floor((i * rows.length) / cap)]!);
  return out;
}

export interface AskOptions {
  kg?: string;
  model?: string;
}

function envVar(name: string, fallback?: string): string | undefined {
  // Prefer COGRAPH_, fall back to OMNIX_ so old configs keep working.
  return (
    process.env[`COGRAPH_${name}`] ||
    process.env[`OMNIX_${name}`] ||
    fallback
  );
}

const EXT_FORMAT: Record<string, string> = {
  ".csv": "csv",
  ".json": "json",
  ".jsonl": "json",
  ".txt": "text",
};

/**
 * Parse a CSV string into an array of row objects.
 *
 * Minimal RFC-4180-ish parser: handles quoted fields with commas, escaped
 * quotes (`""`), CRLF/LF line endings. Does not handle BOM stripping or
 * encoding detection — we assume UTF-8 text in.
 */
export function parseCsv(content: string): Record<string, string>[] {
  const rows: string[][] = [];
  let cur: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < content.length; i++) {
    const ch = content[i];
    if (inQuotes) {
      if (ch === '"') {
        if (content[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ",") {
        cur.push(field);
        field = "";
      } else if (ch === "\n") {
        cur.push(field);
        rows.push(cur);
        cur = [];
        field = "";
      } else if (ch === "\r") {
        // swallow; handled by the following \n in CRLF, or treat lone \r as line end
        if (content[i + 1] !== "\n") {
          cur.push(field);
          rows.push(cur);
          cur = [];
          field = "";
        }
      } else {
        field += ch;
      }
    }
  }
  // flush trailing field/row
  if (field.length > 0 || cur.length > 0) {
    cur.push(field);
    rows.push(cur);
  }

  if (rows.length === 0) return [];
  const headers = rows[0]!.map((h) => h.trim());
  const out: Record<string, string>[] = [];
  for (let r = 1; r < rows.length; r++) {
    const row = rows[r]!;
    // skip blank trailing lines
    if (row.length === 1 && row[0] === "") continue;
    const obj: Record<string, string> = {};
    for (let c = 0; c < headers.length; c++) {
      obj[headers[c]!] = row[c] ?? "";
    }
    out.push(obj);
  }
  return out;
}

export class Client {
  apiKey: string | undefined;
  baseUrl: string;
  tenant: string;

  constructor(opts: ClientOptions = {}) {
    // Resolution order for each field: explicit opts → env var → ~/.cograph/config.json
    // (written by `cograph login`) → built-in default. Reading the config eagerly
    // is cheap (small JSON file) and lets users skip env vars entirely after login.
    const cfg = readConfig();
    this.apiKey = opts.apiKey ?? envVar("API_KEY") ?? cfg.apiKey;
    const url =
      opts.baseUrl ?? envVar("API_URL") ?? cfg.apiUrl ?? "https://api.cograph.cloud";
    this.baseUrl = url.replace(/\/+$/, "");
    this.tenant = opts.tenant ?? envVar("TENANT") ?? cfg.tenant ?? "demo-tenant";
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) h["X-API-Key"] = this.apiKey;
    return h;
  }

  private base(): string {
    return `${this.baseUrl}/graphs/${this.tenant}`;
  }

  /**
   * Probe the backend to determine reachability and whether endpoints
   * require an X-API-Key header. Used at shell startup to distinguish
   * cloud (auth required) from self-hosted open-access deployments.
   */
  async healthCheck(): Promise<{
    ok: boolean;
    requiresAuth: boolean;
    url: string;
  }> {
    const healthUrl = `${this.baseUrl}/health`;
    try {
      const res = await fetch(healthUrl, {
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) return { ok: false, requiresAuth: false, url: this.baseUrl };
    } catch {
      return { ok: false, requiresAuth: false, url: this.baseUrl };
    }
    // Probe whether endpoints require auth by hitting /kgs without X-API-Key.
    // 401 = requires auth; 200/empty = open access; anything else = treat as
    // auth-required to be safe.
    try {
      const res = await fetch(`${this.base()}/kgs`, {
        headers: { "Content-Type": "application/json" },
        signal: AbortSignal.timeout(5000),
      });
      return {
        ok: true,
        requiresAuth: res.status === 401,
        url: this.baseUrl,
      };
    } catch {
      return { ok: true, requiresAuth: true, url: this.baseUrl };
    }
  }

  private async request<T = unknown>(
    method: string,
    url: string,
    body?: unknown,
    timeoutMs: number = 120_000,
  ): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    let res: Response;
    try {
      res = await fetch(url, {
        method,
        headers: this.headers(),
        body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal,
      });
    } catch (err) {
      clearTimeout(timer);
      if (err instanceof Error && err.name === "AbortError") {
        throw new CographError(`Request to ${url} timed out after ${timeoutMs}ms`);
      }
      throw new CographError(
        `Network error contacting ${url}: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
    clearTimeout(timer);

    if (!res.ok) {
      let text = "";
      try {
        text = await res.text();
      } catch {
        // ignore
      }
      throw new CographError(`HTTP ${res.status}: ${text}`, {
        status: res.status,
        body: text,
      });
    }

    // 204 No Content
    if (res.status === 204) return undefined as T;

    const ct = res.headers.get("content-type") ?? "";
    if (ct.includes("application/json")) {
      return (await res.json()) as T;
    }
    // fall back to text
    const text = await res.text();
    try {
      return JSON.parse(text) as T;
    } catch {
      return text as unknown as T;
    }
  }

  /**
   * Ingest a file path or raw text into a knowledge graph.
   *
   * If `pathOrText` points to an existing file, its contents are read and the
   * format is inferred from the extension (.csv, .json, .txt) unless
   * `contentType` is given. CSV files use the two-step schema-inference + row
   * mapping flow.
   */
  async ingest(
    pathOrText: string,
    opts: IngestOptions = {},
  ): Promise<Record<string, unknown>> {
    let content: string;
    let fmt: string;

    let isFile = false;
    try {
      isFile = existsSync(pathOrText) && statSync(pathOrText).isFile();
    } catch {
      isFile = false;
    }

    if (isFile) {
      const ext = extname(pathOrText).toLowerCase();
      if (ext === ".pdf") {
        throw new CographError(
          "PDF ingest not yet supported in the Node CLI; use the Python CLI or POST raw bytes to the API.",
        );
      }
      content = readFileSync(pathOrText, "utf-8");
      fmt = opts.contentType ?? EXT_FORMAT[ext] ?? "text";
      if (fmt === "csv") {
        return this.ingestCsv(content, opts);
      }
    } else {
      content = pathOrText;
      fmt = opts.contentType ?? "text";
    }

    const body: Record<string, unknown> = {
      content,
      content_type: fmt,
      source: "client",
    };
    if (opts.kg) body.kg_name = opts.kg;
    return this.request("POST", `${this.base()}/ingest`, body, 120_000);
  }

  private async ingestCsv(
    content: string,
    opts: IngestOptions,
  ): Promise<Record<string, unknown>> {
    const kgName = opts.kg;
    const batchSize = opts.batchSize ?? 200;
    const concurrency = opts.concurrency ?? 4;

    const rows = parseCsv(content);
    if (rows.length === 0) throw new CographError("CSV is empty");
    const headers = Object.keys(rows[0]!);

    // Send the whole file to the profiler, evenly strided across it (never the
    // head — head-of-file bias, e.g. a key column that goes sparse later, is
    // exactly what evidence-grounded inference fixes). Profile fidelity =
    // decision quality. Mirrors the Explorer's upload flow.
    const sampleRows = stridedSample(rows);

    const schemaBody = {
      headers,
      sample_rows: sampleRows,
      total_rows: rows.length,
    };
    const mapping = await this.request<Record<string, unknown>>(
      "POST",
      `${this.base()}/ingest/csv/schema`,
      schemaBody,
      300_000,
    );

    // Confirm/override gate (same contract as the Explorer's review step): the
    // caller inspects the inferred mapping and returns what to ingest, or null
    // to cancel before any rows are written. /ingest/csv/rows applies exactly
    // what we post back. When no hook is given, apply the inference as-is.
    let mappingToPost: Record<string, unknown> = mapping;
    if (opts.onSchemaInferred) {
      const reviewed = await opts.onSchemaInferred(mapping, {
        totalRows: rows.length,
        rowsProfiled: sampleRows.length,
      });
      if (reviewed == null) {
        return { cancelled: true, message: "Ingest cancelled before any rows were written." };
      }
      mappingToPost = reviewed;
    }

    // Slice rows into batches up front so we can fire them off in a
    // bounded worker pool. Sequential 50-row batches over 891 rows took
    // ~60s end-to-end (18 round-trips); 200-row batches × 4 in flight
    // brings that to ~5s on the same backend.
    const batches: Array<Record<string, string>[]> = [];
    for (let i = 0; i < rows.length; i += batchSize) {
      batches.push(rows.slice(i, i + batchSize));
    }

    let totalEntities = 0;
    let totalTriples = 0;
    let rowsProcessed = 0;
    let nextBatch = 0;

    const postBatch = async (batch: Record<string, string>[]) => {
      const body: Record<string, unknown> = {
        mapping: mappingToPost,
        rows: batch,
        source: "client",
      };
      if (kgName) body.kg_name = kgName;
      const result = await this.request<{
        entities_resolved?: number;
        triples_inserted?: number;
      }>("POST", `${this.base()}/ingest/csv/rows`, body, 300_000);
      return {
        entities: result.entities_resolved ?? 0,
        triples: result.triples_inserted ?? 0,
        size: batch.length,
      };
    };

    const worker = async (): Promise<void> => {
      while (true) {
        const idx = nextBatch++;
        if (idx >= batches.length) return;
        const r = await postBatch(batches[idx]!);
        totalEntities += r.entities;
        totalTriples += r.triples;
        rowsProcessed += r.size;
        opts.onProgress?.({
          rowsProcessed,
          totalRows: rows.length,
          entitiesResolved: totalEntities,
          triplesInserted: totalTriples,
        });
      }
    };

    const workers: Array<Promise<void>> = [];
    for (let i = 0; i < Math.min(concurrency, batches.length); i++) {
      workers.push(worker());
    }
    await Promise.all(workers);

    // All batches are in — kick off a background recompute of the Explorer
    // type-stats for this KG so type-detail views load instantly. The endpoint
    // returns immediately (the scan runs server-side in the background); this
    // is best-effort and never fails the ingest.
    if (kgName) {
      try {
        await this.request(
          "POST",
          `${this.base()}/explore/kgs/${encodeURIComponent(kgName)}/recompute-stats`,
          {},
          15_000,
        );
      } catch {
        // non-fatal — stats fall back to a live scan until the next recompute
      }
    }

    return {
      entities_resolved: totalEntities,
      triples_inserted: totalTriples,
      mapping,
    };
  }

  /** Ask a natural language question and return the parsed response. */
  async ask(
    question: string,
    opts: AskOptions = {},
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = { question };
    if (opts.kg) body.kg_name = opts.kg;
    if (opts.model) body.model = opts.model;
    return this.request("POST", `${this.base()}/ask`, body, 60_000);
  }

  /**
   * One turn of the unified Ask-AI agent — the SINGLE conversational surface
   * (`POST /graphs/{tenant}/agent`, COG-118). Mirrors the HTTP contract exactly:
   *
   *  - `confirmPlanId` set → the server runs `execute_plan` (the only mutating
   *    path) and returns `{kind:"result", steps}`.
   *  - otherwise → the server runs `planner.handle(message)` and returns one of
   *    `{kind:"answer"}` / `{kind:"clarify"}` / `{kind:"plan"}`.
   *
   * The agent classifies intent server-side and drives the underlying engines
   * through its capability registry — the client never talks to `/ask`,
   * `/enrich/*` etc. for an agent turn. ENTITLEMENT for any paid step a plan
   * contains is enforced server-side at execute time (the same authorization the
   * direct paid routes apply), so confirming a plan here cannot bypass a gate the
   * direct path enforces — the gate lives behind the endpoint, not in this client.
   */
  async agent(opts: AgentTurnOptions): Promise<AgentResult> {
    const body: Record<string, unknown> = {
      message: opts.message ?? "",
      context: {
        kg_name: opts.kgName ?? "",
        type_name: opts.typeName ?? null,
      },
    };
    if (opts.sessionId) body.session_id = opts.sessionId;
    // confirm.plan_id present → the server routes to execute_plan (mutating).
    if (opts.confirmPlanId) body.confirm = { plan_id: opts.confirmPlanId };
    return this.request<AgentResult>(
      "POST",
      `${this.base()}/agent`,
      body,
      // Generous: a confirmed plan can kick off enrichment/dedup work, and a
      // question turn runs an LLM round-trip server-side.
      120_000,
    );
  }

  /** List the tenants the authenticated user can access (GET /v1/me/tenants).
   *  Keyed by the API key (X-API-Key → user), so it's independent of the active
   *  tenant. Throws CographError with status 501 on deployments without a tenant
   *  provider (e.g. OSS-only). */
  async listTenants(): Promise<Array<{ id: string; label: string }>> {
    return this.request<Array<{ id: string; label: string }>>(
      "GET",
      `${this.baseUrl}/v1/me/tenants`,
      undefined,
      15_000,
    );
  }

  /** List all knowledge graphs for the current tenant. */
  async listKgs(): Promise<Array<Record<string, unknown>>> {
    const data = await this.request<unknown>(
      "GET",
      `${this.base()}/kgs`,
      undefined,
      15_000,
    );
    if (Array.isArray(data)) return data as Array<Record<string, unknown>>;
    if (data && typeof data === "object" && "kgs" in data) {
      const kgs = (data as { kgs?: unknown }).kgs;
      if (Array.isArray(kgs)) return kgs as Array<Record<string, unknown>>;
    }
    return [];
  }

  /** Create a knowledge graph. */
  async createKg(
    name: string,
    description?: string,
  ): Promise<Record<string, unknown>> {
    const body: Record<string, unknown> = { name };
    if (description) body.description = description;
    return this.request("POST", `${this.base()}/kgs`, body, 15_000);
  }

  /** Delete a knowledge graph by name. */
  async deleteKg(name: string): Promise<Record<string, unknown>> {
    return this.request(
      "DELETE",
      `${this.base()}/kgs/${encodeURIComponent(name)}`,
      undefined,
      30_000,
    );
  }

  /** List ontology types. */
  async ontologyTypes(): Promise<Array<Record<string, unknown>>> {
    const data = await this.request<unknown>(
      "GET",
      `${this.base()}/ontology/types`,
      undefined,
      15_000,
    );
    return Array.isArray(data) ? (data as Array<Record<string, unknown>>) : [];
  }

  /**
   * Resolve a natural-language ontology change against the existing ontology.
   * The caller does not need to know exact type/attribute/relationship names —
   * the server matches the plain-language `ask` to the current schema and
   * returns auto-applied changes plus proposals that need confirmation.
   */
  async ontologyResolve(
    ask: string,
    opts: { knowledge_graph?: string } = {},
  ): Promise<OntologyResolveResult> {
    const body: Record<string, unknown> = { ask };
    if (opts.knowledge_graph) body.knowledge_graph = opts.knowledge_graph;
    return this.request<OntologyResolveResult>(
      "POST",
      `${this.base()}/ontology/resolve`,
      body,
      60_000,
    );
  }

  /**
   * Apply a single resolved ontology change — one of the `proposals` returned
   * by {@link ontologyResolve}. Pass the proposal object through unchanged.
   */
  async ontologyApply(
    proposal: ResolvedChange,
  ): Promise<OntologyApplyResult> {
    return this.request<OntologyApplyResult>(
      "POST",
      `${this.base()}/ontology/apply`,
      proposal,
      60_000,
    );
  }

  /**
   * Second-pass entity resolution: re-run ER over an already-ingested KG to
   * collapse intra-batch fragments. Synchronous on the server; returns a
   * per-type before/after report. Generous timeout — it rewrites triples.
   */
  async erRebuild(kg: string): Promise<Record<string, unknown>> {
    return this.request<Record<string, unknown>>(
      "POST",
      `${this.base()}/explore/kgs/${encodeURIComponent(kg)}/er-rebuild`,
      {},
      300_000,
    );
  }

  /** Per-KG type counts: every type with ≥1 instance, sorted desc. */
  async typeCounts(kg: string): Promise<TypeCount[]> {
    const data = await this.request<unknown>(
      "GET",
      `${this.base()}/kgs/${encodeURIComponent(kg)}/type-counts`,
      undefined,
      30_000,
    );
    return Array.isArray(data) ? (data as TypeCount[]) : [];
  }

  /** Plan + run an enrichment job. Returns immediately with the job id. */
  async enrichRun(req: EnrichRequest): Promise<EnrichJobCreate> {
    return this.request<EnrichJobCreate>(
      "POST",
      `${this.base()}/enrich/jobs`,
      req,
      30_000,
    );
  }

  /** List recent enrichment jobs for the current tenant. */
  async enrichJobs(): Promise<JobSummary[]> {
    const data = await this.request<unknown>(
      "GET",
      `${this.base()}/enrich/jobs`,
      undefined,
      15_000,
    );
    return Array.isArray(data) ? (data as JobSummary[]) : [];
  }

  /** Fetch a single enrichment job (with truncated results). */
  async enrichJob(jobId: string): Promise<EnrichJob> {
    return this.request<EnrichJob>(
      "GET",
      `${this.base()}/enrich/jobs/${encodeURIComponent(jobId)}`,
      undefined,
      15_000,
    );
  }

  /** Fetch the conflict review queue for a job. */
  async enrichConflicts(jobId: string): Promise<ConflictReview[]> {
    const data = await this.request<unknown>(
      "GET",
      `${this.base()}/enrich/jobs/${encodeURIComponent(jobId)}/conflicts`,
      undefined,
      30_000,
    );
    return Array.isArray(data) ? (data as ConflictReview[]) : [];
  }

  /** Apply a set of conflict review decisions to a job. */
  async enrichApply(
    jobId: string,
    decisions: ConflictReview[],
  ): Promise<{ applied: number }> {
    return this.request<{ applied: number }>(
      "POST",
      `${this.base()}/enrich/jobs/${encodeURIComponent(jobId)}/apply`,
      { decisions },
      60_000,
    );
  }

  /** Cancel an enrichment job. */
  async enrichCancel(jobId: string): Promise<void> {
    await this.request<void>(
      "DELETE",
      `${this.base()}/enrich/jobs/${encodeURIComponent(jobId)}`,
      undefined,
      15_000,
    );
  }

  /** Per-type breakdown for one type in one KG: definition + counts + samples.
   *
   * System predicates (rdfs:label, ingested_at, source) are hidden by default
   * — they're attached to every entity at 100% and drown out the columns the
   * user cares about. Pass `includeSystem: true` to see them. */
  async typeUsage(
    kg: string,
    typeName: string,
    opts: { includeSystem?: boolean } = {},
  ): Promise<TypeUsage> {
    const qs = opts.includeSystem ? "?include_system=true" : "";
    return this.request<TypeUsage>(
      "GET",
      `${this.base()}/kgs/${encodeURIComponent(kg)}/types/${encodeURIComponent(typeName)}/usage${qs}`,
      undefined,
      30_000,
    );
  }

  /** Explorer summary for a type — like typeUsage but adds coverage_pct + avg_degree. */
  async typeSummary(kg: string, typeName: string): Promise<TypeSummary> {
    return this.request<TypeSummary>(
      "GET",
      `${this.base()}/explore/kgs/${encodeURIComponent(kg)}/types/${encodeURIComponent(typeName)}/summary`,
      undefined,
      30_000,
    );
  }

  /** Search types or attributes by name substring within a KG. */
  async exploreSearch(
    kg: string,
    q: string,
    kind: "type" | "attr" = "type",
  ): Promise<Array<Record<string, unknown>>> {
    const qs = new URLSearchParams({ kg, q, kind }).toString();
    const data = await this.request<unknown>(
      "GET",
      `${this.base()}/explore/search?${qs}`,
      undefined,
      15_000,
    );
    return Array.isArray(data) ? (data as Array<Record<string, unknown>>) : [];
  }
}

/**
 * A single ontology change resolved against the existing schema. The same
 * shape is returned under `applied`/`proposals` by {@link Client.ontologyResolve}
 * and accepted as the request body by {@link Client.ontologyApply}.
 */
export interface ResolvedChange {
  kind: "attribute" | "relationship";
  subject_type: string;
  name: string;
  datatype_or_target: string;
  action: "reuse" | "extend" | "create";
  confidence: number;
  reason: string;
}

export interface OntologyResolveResult {
  applied: ResolvedChange[];
  proposals: ResolvedChange[];
  summary: string;
}

export interface OntologyApplyResult {
  applied: ResolvedChange;
  operations: number;
  summary: string;
}

export interface TypeCount {
  name: string;
  entity_count: number;
}

export interface AttributeUsage {
  name: string;
  datatype: string;
  count: number;
}

export interface RelationshipUsage {
  name: string;
  target_type: string | null;
  count: number;
}

export interface EntitySample {
  uri: string;
  label: string;
}

export interface TypeUsage {
  name: string;
  description: string;
  parent_type: string | null;
  entity_count: number;
  attributes: AttributeUsage[];
  relationships: RelationshipUsage[];
  samples: EntitySample[];
}

export interface AttributeSummary {
  name: string;
  predicate_uri: string;
  datatype: string;
  count: number;
  coverage_pct: number;
}

export interface RelationshipSummary {
  name: string;
  predicate_uri: string;
  target_type: string | null;
  count: number;
  coverage_pct: number;
  avg_degree: number;
}

export interface TypeSummary {
  name: string;
  description: string;
  parent_type: string | null;
  entity_count: number;
  attributes: AttributeSummary[];
  relationships: RelationshipSummary[];
}

export type EnrichmentTier = "lite" | "base" | "core" | "pro";
export type JobStatus =
  | "queued"
  | "running"
  | "review"
  | "applied"
  | "cancelled"
  | "failed";
export type ConflictPolicy = "skip" | "verify" | "overwrite" | "stage";
export type RowAction =
  | "filled"
  | "verified"
  | "conflict"
  | "skipped"
  | "no_match";
export type ReviewDecision = "accept" | "reject" | "skip";

export interface EnrichRequest {
  type_name: string;
  attributes: string[];
  tier?: EnrichmentTier;
  kg_name: string;
  conflict_policy?: ConflictPolicy;
  confidence_min?: number;
  limit?: number;
}

export interface EnrichJobCreate {
  job_id: string;
  status: JobStatus;
  estimated_cost_usd: number;
  total_entities: number;
}

export interface Verdict {
  value: string;
  confidence: number;
  source: string;
  source_url?: string | null;
  reasoning?: string | null;
}

export interface JobProgress {
  total: number;
  processed: number;
  filled: number;
  verified: number;
  conflicts: number;
  skipped: number;
  cache_hits: number;
}

export interface RowResult {
  entity_uri: string;
  attribute: string;
  existing_value: string | null;
  verdict: Verdict | null;
  action: RowAction;
}

export interface JobSummary {
  id: string;
  tenant_id: string;
  kg_name: string;
  type_name: string;
  attributes: string[];
  tier: EnrichmentTier;
  status: JobStatus;
  progress: JobProgress;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  conflict_policy: ConflictPolicy;
  confidence_min: number;
  error?: string | null;
}

export interface EnrichJob extends JobSummary {
  results?: RowResult[];
  limit?: number | null;
}

export interface ConflictReview {
  entity_uri: string;
  attribute: string;
  existing_value: string;
  proposed: Verdict;
  decision?: ReviewDecision | null;
}

// --- Unified Ask-AI agent (COG-118 / COG-125) -------------------------------- #

/** Inputs to {@link Client.agent} — mirror the `/agent` HTTP body. */
export interface AgentTurnOptions {
  /** The user's natural-language message. Optional when `confirmPlanId` is set
   *  (a confirm turn carries no new message). */
  message?: string;
  /** Knowledge graph the turn operates within. */
  kgName?: string;
  /** Optional active type scope (needed for enrich/clean/dedup planning). */
  typeName?: string;
  /** Optional conversation/session id for multi-turn continuity. */
  sessionId?: string;
  /** When set, the server CONFIRMS + EXECUTES this previously-proposed plan
   *  (the only mutating path) instead of classifying a new message. */
  confirmPlanId?: string;
}

/**
 * The kind-tagged result of one agent turn. The server returns exactly one of:
 *  - `answer`  — a read-only answer (questions; an ontology INSPECT) with SPARQL.
 *  - `clarify` — the agent needs more detail; ask the user `question`.
 *  - `plan`    — a proposed (un-executed) plan with `plan_id` + `steps`; confirm
 *                by calling `agent({ confirmPlanId: plan_id })`.
 *  - `result`  — the outcome of executing a confirmed plan, per-step.
 *  - `error`   — e.g. an unknown/expired plan_id on confirm.
 * Extra fields vary by kind (answer/sparql/rows; question; plan_id/steps;
 * steps), so this is intentionally open beyond the discriminant.
 */
export interface AgentResult {
  kind: "answer" | "clarify" | "plan" | "result" | "error";
  [key: string]: unknown;
}
