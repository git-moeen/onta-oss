/**
 * Shared terminal renderer for unified Ask-AI agent turns (COG-129).
 *
 * One function — {@link renderAgentResult} — turns a kind-tagged
 * {@link AgentResult} (the response of {@link Client.agent}) into terminal
 * output, used by BOTH the non-interactive `agent` command in `cli.ts` and the
 * interactive `/agent` slash command in `shell.ts`. Keeping it here means the
 * answer/clarify/plan/result rendering lives in exactly one place and the two
 * entry points can never drift.
 *
 * The renderer writes to a caller-supplied sink (defaults to stdout) so it is
 * trivially testable: tests pass a string-collecting sink and assert on what
 * came out, without stubbing `process.stdout`.
 *
 * It treats {@link AgentResult} as the open, kind-tagged shape the SDK exports
 * (the discriminant `kind` plus loosely-typed extras), reading fields
 * defensively — a missing `sparql`/`rows`/`cost` is simply omitted, never throws.
 */

import type { AgentResult } from "./client.js";

// --- ANSI helpers ----------------------------------------------------------- #
// Self-contained (no dependency on cli.ts/shell.ts) so this module can be
// imported by either. Color is gated on a TTY + NO_COLOR, matching cli.ts; a
// caller can force-disable with `{ color: false }` (tests do this so assertions
// are on plain text).

const ttyColor = Boolean(process.stdout.isTTY) && !process.env.NO_COLOR;

function sgr(code: string, s: string, color: boolean): string {
  return color ? `\x1b[${code}m${s}\x1b[0m` : s;
}

/** Where a render writes. The default targets stdout; tests pass a collector. */
export interface RenderSink {
  write(s: string): void;
}

export interface RenderOptions {
  /** Override color. Defaults to "on when stdout is a TTY and NO_COLOR unset". */
  color?: boolean;
  /** Output sink. Defaults to process.stdout. */
  sink?: RenderSink;
}

interface Pen {
  bold: (s: string) => string;
  dim: (s: string) => string;
  cyan: (s: string) => string;
  green: (s: string) => string;
  red: (s: string) => string;
  yellow: (s: string) => string;
}

function makePen(color: boolean): Pen {
  return {
    bold: (s) => sgr("1", s, color),
    dim: (s) => sgr("2", s, color),
    cyan: (s) => sgr("36", s, color),
    green: (s) => sgr("32", s, color),
    red: (s) => sgr("31", s, color),
    yellow: (s) => sgr("33", s, color),
  };
}

// --- small shape helpers ---------------------------------------------------- #
// AgentResult is intentionally open beyond `kind`, so we read fields with
// runtime guards rather than casts that could throw on an unexpected payload.

function str(v: unknown): string | undefined {
  return typeof v === "string" && v.length > 0 ? v : undefined;
}

function asArray(v: unknown): unknown[] {
  return Array.isArray(v) ? v : [];
}

function asRecord(v: unknown): Record<string, unknown> {
  return v && typeof v === "object" && !Array.isArray(v)
    ? (v as Record<string, unknown>)
    : {};
}

/** Indent every line of a (possibly multi-line) block by `pad` spaces. */
function indent(text: string, pad: string): string {
  return text
    .split("\n")
    .map((l) => (l.length ? pad + l : l))
    .join("\n");
}

/**
 * Render a 0.0–1.0 confidence as a short bar + percentage, colored by band:
 * green ≥ 0.75, yellow ≥ 0.4, dim/red below. Returns "" when there's no usable
 * number so a step with no confidence renders cleanly.
 */
function confidenceIndicator(value: unknown, pen: Pen): string {
  const n = typeof value === "number" ? value : Number(value);
  if (!Number.isFinite(n)) return "";
  const clamped = Math.max(0, Math.min(1, n));
  const filled = Math.round(clamped * 5);
  const bar = "●".repeat(filled) + "○".repeat(5 - filled);
  const pct = `${Math.round(clamped * 100)}%`;
  const label = `${bar} ${pct}`;
  if (clamped >= 0.75) return pen.green(label);
  if (clamped >= 0.4) return pen.yellow(label);
  return pen.dim(label);
}

/**
 * Render a step's cost dict. Highlights paid work — a nonzero `paid_calls` and
 * any `estimated_usd` are surfaced in yellow so a plan that will spend money
 * stands out; free work renders dim. Returns "" for an empty/zero cost.
 */
function costIndicator(cost: unknown, pen: Pen): string {
  const c = asRecord(cost);
  const paidRaw = c.paid_calls ?? c.paidCalls;
  const usdRaw = c.estimated_usd ?? c.estimatedUsd ?? c.usd;
  const paid = typeof paidRaw === "number" ? paidRaw : Number(paidRaw);
  const usd = typeof usdRaw === "number" ? usdRaw : Number(usdRaw);
  const parts: string[] = [];
  if (Number.isFinite(paid) && paid > 0) {
    parts.push(`${paid.toLocaleString()} paid call${paid === 1 ? "" : "s"}`);
  }
  if (Number.isFinite(usd) && usd > 0) {
    parts.push(`$${usd.toFixed(usd < 0.01 ? 4 : 2)}`);
  }
  if (parts.length) return pen.yellow(parts.join(" · "));
  // Explicitly-free work (a cost dict present but zero) reads as "free".
  if (Object.keys(c).length) return pen.dim("free");
  return "";
}

// --- table rendering (mirrors the existing CLI results-table style) --------- #

/**
 * Render `rows` (array of objects) as an aligned table, columns ordered by
 * `columns` when given else by first-row key order. Mirrors the column-padding
 * style used elsewhere in the CLI (`/types`, enrich jobs). Caps at 20 rows with
 * a "… N more" footer so a large result set doesn't flood the terminal.
 */
function renderTable(
  rows: unknown[],
  columns: string[] | undefined,
  pen: Pen,
  write: (s: string) => void,
): void {
  const objRows = rows.map(asRecord);
  if (objRows.length === 0) return;
  const cols =
    columns && columns.length
      ? columns
      : Array.from(
          objRows.reduce<Set<string>>((set, r) => {
            for (const k of Object.keys(r)) set.add(k);
            return set;
          }, new Set<string>()),
        );
  if (cols.length === 0) return;

  const cell = (r: Record<string, unknown>, c: string): string => {
    const v = r[c];
    if (v == null) return "";
    return typeof v === "object" ? JSON.stringify(v) : String(v);
  };

  const MAX = 20;
  const shown = objRows.slice(0, MAX);
  const widths = cols.map((c) =>
    Math.max(c.length, ...shown.map((r) => cell(r, c).length)),
  );

  write(
    "  " +
      pen.bold(cols.map((c, i) => c.padEnd(widths[i]!)).join("  ")) +
      "\n",
  );
  for (const r of shown) {
    write(
      "  " + cols.map((c, i) => cell(r, c).padEnd(widths[i]!)).join("  ") + "\n",
    );
  }
  if (objRows.length > MAX) {
    write("  " + pen.dim(`… ${objRows.length - MAX} more row(s)`) + "\n");
  }
}

// --- per-kind renderers ----------------------------------------------------- #

function renderAnswer(
  r: AgentResult,
  pen: Pen,
  write: (s: string) => void,
): void {
  // Prefer the narrative, then the formatted answer; either may be present.
  const narrative = str(r.narrative) ?? str(r.answer) ?? "No answer.";
  write("\n  " + narrative + "\n");

  const sparql = str(r.sparql);
  if (sparql) {
    write("\n  " + pen.dim("SPARQL") + "\n");
    write(indent(pen.dim(sparql), "    ") + "\n");
  }

  const rows = asArray(r.rows);
  if (rows.length) {
    const columns = asArray(r.columns).filter(
      (c): c is string => typeof c === "string",
    );
    write("\n");
    renderTable(rows, columns.length ? columns : undefined, pen, write);
  }
  write("\n");
}

function renderClarify(
  r: AgentResult,
  pen: Pen,
  write: (s: string) => void,
): void {
  const q =
    str(r.question) ?? str(r.clarify) ?? "Could you clarify what you'd like?";
  write("\n  " + pen.yellow("?") + " " + q + "\n\n");
}

function renderPlan(
  r: AgentResult,
  pen: Pen,
  write: (s: string) => void,
): void {
  const planId = str(r.plan_id) ?? str(r.planId) ?? "?";
  const steps = asArray(r.steps).map(asRecord);
  write(
    "\n  " +
      pen.bold("Plan") +
      "  " +
      pen.dim(`${steps.length} step${steps.length === 1 ? "" : "s"}`) +
      pen.dim(`  ·  plan_id ${planId}`) +
      "\n",
  );
  steps.forEach((s, i) => {
    const cap = str(s.capability) ?? "?";
    const action = str(s.action) ?? "?";
    const conf = confidenceIndicator(s.confidence, pen);
    const cost = costIndicator(s.cost, pen);
    const tags = [conf, cost].filter(Boolean).join("  ");
    write(
      `\n  ${pen.dim(`${i + 1}.`)} ${pen.cyan(cap)} ${pen.dim("→")} ${pen.bold(action)}` +
        (tags ? `   ${tags}` : "") +
        "\n",
    );
    const rationale = str(s.rationale);
    if (rationale) write("     " + pen.dim(rationale) + "\n");
  });
  write("\n");
}

function renderResult(
  r: AgentResult,
  pen: Pen,
  write: (s: string) => void,
): void {
  const steps = asArray(r.steps).map(asRecord);
  write("\n  " + pen.bold("Result") + "\n");
  if (steps.length === 0) {
    write("  " + pen.dim("(no steps)") + "\n\n");
    return;
  }
  for (const s of steps) {
    // The planner stamps status "ok" on success, "failed" on a raised step, and
    // "skipped" for an unregistered capability.
    const status = str(s.status) ?? "ok";
    const ok = status === "ok";
    const failed = status === "failed";
    const mark = ok ? pen.green("✓") : failed ? pen.red("✗") : pen.dim("–");
    const cap = str(s.capability) ?? str(s.action) ?? "step";
    const summary =
      str(s.message) ?? str(s.error) ?? (ok ? "done" : status);
    write(`  ${mark} ${pen.bold(cap)}  ${summary}\n`);
    // Job reference, when the step kicked off background work.
    const jobId = str(s.job_id) ?? str(s.jobId);
    if (jobId) {
      const jobStatus = str(s.job_status) ?? str(s.jobStatus);
      write(
        "     " +
          pen.dim(`job ${jobId}${jobStatus ? ` · ${jobStatus}` : ""}`) +
          "\n",
      );
    }
  }
  write("\n");
}

function renderError(
  r: AgentResult,
  pen: Pen,
  write: (s: string) => void,
): void {
  const msg = str(r.error) ?? "The agent returned an error.";
  const planId = str(r.plan_id) ?? str(r.planId);
  write(
    "\n  " +
      pen.red("✗") +
      " " +
      msg +
      (planId ? pen.dim(`  (plan_id ${planId})`) : "") +
      "\n\n",
  );
}

/**
 * Render one agent turn to the terminal. Dispatches on `result.kind`:
 *  - `answer`  → narrative, optional SPARQL (dim), optional rows table.
 *  - `clarify` → the clarifying question.
 *  - `plan`    → each step (capability → action, confidence, cost, rationale)
 *                plus the plan_id.
 *  - `result`  → per-step done/failed marker + summary + any job reference.
 *  - `error`   → the error message (+ plan_id if present).
 * An unknown `kind` falls back to a compact JSON dump so nothing is swallowed.
 */
export function renderAgentResult(
  result: AgentResult,
  opts: RenderOptions = {},
): void {
  const color = opts.color ?? ttyColor;
  const sink = opts.sink ?? process.stdout;
  const write = (s: string): void => {
    sink.write(s);
  };
  const pen = makePen(color);

  switch (result.kind) {
    case "answer":
      return renderAnswer(result, pen, write);
    case "clarify":
      return renderClarify(result, pen, write);
    case "plan":
      return renderPlan(result, pen, write);
    case "result":
      return renderResult(result, pen, write);
    case "error":
      return renderError(result, pen, write);
    default:
      // Unknown discriminant — never throw, never silently drop. Surface it.
      write("\n  " + JSON.stringify(result) + "\n\n");
  }
}
