import { describe, expect, it } from "vitest";
import { renderAgentResult, type RenderSink } from "../src/agentRender.js";
import type { AgentResult } from "../src/client.js";

// --- string-collecting sink -------------------------------------------------- #
// The renderer writes to a caller-supplied sink, so tests capture output without
// touching process.stdout. Color is forced off so assertions are on plain text
// (no ANSI escapes to match around).

function collect(result: AgentResult): string {
  let out = "";
  const sink: RenderSink = { write: (s) => (out += s) };
  renderAgentResult(result, { color: false, sink });
  return out;
}

describe("renderAgentResult — answer", () => {
  it("renders the narrative, SPARQL, and a rows table without throwing", () => {
    const out = collect({
      kind: "answer",
      answer: "There are 42 people.",
      narrative: "I found 42 people in the graph.",
      sparql: "SELECT (COUNT(?p) AS ?n) WHERE { ?p a :Person }",
      columns: ["name", "age"],
      rows: [
        { name: "Ada", age: "36" },
        { name: "Alan", age: "41" },
      ],
    });
    expect(out).toContain("I found 42 people in the graph.");
    expect(out).toContain("SPARQL");
    expect(out).toContain("SELECT (COUNT(?p) AS ?n)");
    // table header + cells
    expect(out).toContain("name");
    expect(out).toContain("Ada");
    expect(out).toContain("Alan");
  });

  it("falls back to `answer` when no narrative, and omits SPARQL/table when absent", () => {
    const out = collect({ kind: "answer", answer: "Plain answer." });
    expect(out).toContain("Plain answer.");
    expect(out).not.toContain("SPARQL");
  });

  it("handles a missing answer gracefully", () => {
    const out = collect({ kind: "answer" });
    expect(out).toContain("No answer.");
  });
});

describe("renderAgentResult — clarify", () => {
  it("renders the clarifying question", () => {
    const out = collect({
      kind: "clarify",
      question: "Which field do you mean — title or role?",
    });
    expect(out).toContain("Which field do you mean — title or role?");
  });

  it("falls back when no question text is present", () => {
    const out = collect({ kind: "clarify" });
    expect(out).toContain("clarify");
  });
});

describe("renderAgentResult — plan", () => {
  it("renders each step (capability → action, rationale, confidence, cost) + plan_id", () => {
    const out = collect({
      kind: "plan",
      plan_id: "plan-123",
      steps: [
        {
          id: "s1",
          capability: "enrich",
          action: "enrich_attribute",
          rationale: "Fill missing emails from the web.",
          confidence: 0.9,
          cost: { paid_calls: 1200, estimated_usd: 3.6 },
        },
        {
          id: "s2",
          capability: "normalize",
          action: "clean_field",
          rationale: "Strip emoji first.",
          confidence: 0.5,
          cost: {},
        },
      ],
    });
    expect(out).toContain("Plan");
    expect(out).toContain("plan-123");
    expect(out).toContain("enrich");
    expect(out).toContain("enrich_attribute");
    expect(out).toContain("Fill missing emails from the web.");
    expect(out).toContain("normalize");
    // cost highlight — paid calls + dollar amount surfaced
    expect(out).toContain("1,200 paid calls");
    expect(out).toContain("$3.60");
    // confidence percentage rendered
    expect(out).toContain("90%");
    expect(out).toContain("50%");
  });

  it("renders a plan with no steps without throwing", () => {
    const out = collect({ kind: "plan", plan_id: "p0", steps: [] });
    expect(out).toContain("p0");
    expect(out).toContain("0 steps");
  });
});

describe("renderAgentResult — result", () => {
  it("renders per-step ok/failed markers, summary, and a job reference", () => {
    const out = collect({
      kind: "result",
      plan_id: "plan-123",
      steps: [
        {
          step_id: "s1",
          capability: "enrich",
          status: "ok",
          message: "Enriching email on Person in the background.",
          job_id: "job-abc",
          job_status: "queued",
        },
        {
          step_id: "s2",
          capability: "normalize",
          status: "failed",
          error: "no rule matched",
        },
        {
          step_id: "s3",
          capability: "dedup",
          status: "skipped",
          error: "capability not registered",
        },
      ],
    });
    expect(out).toContain("Result");
    expect(out).toContain("enrich");
    expect(out).toContain("Enriching email on Person in the background.");
    // job reference
    expect(out).toContain("job-abc");
    expect(out).toContain("queued");
    // failed step surfaces its error
    expect(out).toContain("no rule matched");
    // skipped step surfaces its reason
    expect(out).toContain("capability not registered");
  });

  it("renders an empty result without throwing", () => {
    const out = collect({ kind: "result", steps: [] });
    expect(out).toContain("Result");
    expect(out).toContain("(no steps)");
  });
});

describe("renderAgentResult — error + unknown", () => {
  it("renders an error kind with its message and plan_id", () => {
    const out = collect({
      kind: "error",
      error: "plan not found",
      plan_id: "missing-plan",
    });
    expect(out).toContain("plan not found");
    expect(out).toContain("missing-plan");
  });

  it("does not throw or swallow an unknown kind — dumps it", () => {
    // Cast through unknown: the renderer must tolerate a discriminant outside the
    // documented union (forward-compat with a new server kind).
    const weird = { kind: "future_kind", detail: "x" } as unknown as AgentResult;
    const out = collect(weird);
    expect(out).toContain("future_kind");
    expect(out).toContain("detail");
  });
});

describe("renderAgentResult — color toggle", () => {
  it("emits ANSI escapes when color is on, none when off", () => {
    let withColor = "";
    renderAgentResult(
      { kind: "clarify", question: "hi?" },
      { color: true, sink: { write: (s) => (withColor += s) } },
    );
    let without = "";
    renderAgentResult(
      { kind: "clarify", question: "hi?" },
      { color: false, sink: { write: (s) => (without += s) } },
    );
    // eslint-disable-next-line no-control-regex
    expect(withColor).toMatch(/\x1b\[/);
    // eslint-disable-next-line no-control-regex
    expect(without).not.toMatch(/\x1b\[/);
  });
});
