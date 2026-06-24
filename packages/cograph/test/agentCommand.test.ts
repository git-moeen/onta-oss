import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { runAgentCommand } from "../src/cli.js";
import type { AgentResult, AgentTurnOptions, Client } from "../src/client.js";

// --- harness ----------------------------------------------------------------- #
// runAgentCommand takes a Client, so we hand it a fake whose `agent` is a spy
// returning scripted responses. We never construct a real Client and never hit
// the network. The renderer writes to process.stdout, so we silence it here and
// assert purely on the `agent` call sequence.

function fakeClient(responses: AgentResult[]): {
  client: Client;
  calls: AgentTurnOptions[];
} {
  const calls: AgentTurnOptions[] = [];
  let i = 0;
  const agent = vi.fn(async (opts: AgentTurnOptions) => {
    calls.push(opts);
    const r = responses[i++];
    if (!r) throw new Error("fakeClient: no scripted response left");
    return r;
  });
  // Only `agent` is exercised by runAgentCommand; cast the partial as Client.
  return { client: { agent } as unknown as Client, calls };
}

let stdoutSpy: ReturnType<typeof vi.spyOn>;
beforeEach(() => {
  stdoutSpy = vi.spyOn(process.stdout, "write").mockImplementation(() => true);
});
afterEach(() => {
  stdoutSpy.mockRestore();
  vi.restoreAllMocks();
});

/** What the CLI passes as the agent turn's context (camelCase SDK shape). */
const ANSWER: AgentResult = { kind: "answer", answer: "42" };
const CLARIFY: AgentResult = { kind: "clarify", question: "which?" };
function plan(id: string): AgentResult {
  return { kind: "plan", plan_id: id, steps: [] };
}
function result(id: string): AgentResult {
  return { kind: "result", plan_id: id, steps: [] };
}

describe("runAgentCommand — single turn", () => {
  it("calls client.agent once with {message, context} for a plain answer", async () => {
    const { client, calls } = fakeClient([ANSWER]);
    await runAgentCommand(client, "how many people?", {
      kg: "people-kg",
      type: "Person",
    });
    expect(calls).toHaveLength(1);
    expect(calls[0]).toEqual({
      message: "how many people?",
      kgName: "people-kg",
      typeName: "Person",
    });
  });

  it("passes undefined kg/type through (backend default) when flags are omitted", async () => {
    const { client, calls } = fakeClient([CLARIFY]);
    await runAgentCommand(client, "do something", {});
    expect(calls).toHaveLength(1);
    expect(calls[0]).toEqual({
      message: "do something",
      kgName: undefined,
      typeName: undefined,
    });
  });

  it("does NOT auto-confirm a plan without --yes (single call, prints a hint)", async () => {
    const { client, calls } = fakeClient([plan("plan-1")]);
    await runAgentCommand(client, "enrich emails", { kg: "k" });
    // Only the planning turn — no confirm follow-up.
    expect(calls).toHaveLength(1);
    expect(calls[0]!.confirmPlanId).toBeUndefined();
    // The confirm hint was printed.
    const printed = stdoutSpy.mock.calls.map((c) => String(c[0])).join("");
    expect(printed).toContain("--confirm plan-1");
  });
});

describe("runAgentCommand — --yes (confirm-and-execute)", () => {
  it("follows a returned plan with a {confirm:{plan_id}} turn", async () => {
    const { client, calls } = fakeClient([plan("plan-9"), result("plan-9")]);
    await runAgentCommand(client, "merge duplicates", { kg: "k", yes: true });
    expect(calls).toHaveLength(2);
    // First turn: the message (no confirm).
    expect(calls[0]).toEqual({
      message: "merge duplicates",
      kgName: "k",
      typeName: undefined,
    });
    // Second turn: confirm the returned plan id, carrying the same context.
    expect(calls[1]).toEqual({
      confirmPlanId: "plan-9",
      kgName: "k",
      typeName: undefined,
    });
  });

  it("does NOT confirm when --yes is set but the response is an answer (no plan)", async () => {
    const { client, calls } = fakeClient([ANSWER]);
    await runAgentCommand(client, "how many?", { yes: true });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.confirmPlanId).toBeUndefined();
  });
});

describe("runAgentCommand — --confirm (direct execute)", () => {
  it("routes straight to {confirm:{plan_id}} and skips planning", async () => {
    const { client, calls } = fakeClient([result("plan-77")]);
    await runAgentCommand(client, "ignored message", {
      confirm: "plan-77",
      kg: "k",
      type: "Person",
    });
    expect(calls).toHaveLength(1);
    expect(calls[0]).toEqual({
      confirmPlanId: "plan-77",
      kgName: "k",
      typeName: "Person",
    });
    // The message is NOT sent as a planning turn on the --confirm path.
    expect(calls[0]!.message).toBeUndefined();
  });
});
