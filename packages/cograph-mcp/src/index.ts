import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { Client, CographError } from "cograph";
import type { AgentResult, ResolvedChange } from "cograph";
import { z } from "zod";

const VERSION = "0.1.0";

const server = new McpServer(
  {
    name: "cograph",
    version: VERSION,
  },
  {
    instructions:
      "Cograph is a knowledge graph platform. Use these tools to query " +
      "structured data across multiple knowledge graphs using natural language.",
  },
);

function client(): Client {
  return new Client();
}

function textResult(text: string) {
  return {
    content: [{ type: "text" as const, text }],
  };
}

function errorResult(err: unknown) {
  const msg =
    err instanceof CographError
      ? `Cograph error: ${err.message}`
      : err instanceof Error
        ? err.message
        : String(err);
  return {
    content: [{ type: "text" as const, text: msg }],
    isError: true,
  };
}

server.registerTool(
  "list_knowledge_graphs",
  {
    description:
      "List all available knowledge graphs and their descriptions.",
    inputSchema: {},
  },
  async () => {
    try {
      const kgs = await client().listKgs();
      if (!kgs.length) return textResult("No knowledge graphs found.");
      const lines = kgs.map((kg) => {
        const name = String(kg.name ?? "?");
        const desc = kg.description ? `: ${kg.description}` : "";
        return `- ${name}${desc}`;
      });
      return textResult(lines.join("\n"));
    } catch (err) {
      return errorResult(err);
    }
  },
);

server.registerTool(
  "ask",
  {
    description:
      "Ask a natural language question against a knowledge graph. " +
      'Use list_knowledge_graphs to see available KGs first.',
    inputSchema: {
      question: z
        .string()
        .describe(
          'The natural language question to ask (e.g., "How many events are in San Francisco?")',
        ),
      kg_name: z
        .string()
        .optional()
        .describe(
          "Name of the knowledge graph to query. Use list_knowledge_graphs to see available KGs.",
        ),
    },
  },
  async ({ question, kg_name }) => {
    try {
      const data = await client().ask(question, { kg: kg_name });
      const answer = data.answer ?? "No answer";
      const explanation = data.explanation;
      let out = `Answer: ${answer}`;
      if (explanation) out += `\nExplanation: ${explanation}`;
      return textResult(out);
    } catch (err) {
      return errorResult(err);
    }
  },
);

server.registerTool(
  "ingest_csv",
  {
    description:
      "Ingest a CSV file into a knowledge graph. The schema is automatically inferred.",
    inputSchema: {
      file_path: z
        .string()
        .describe("Absolute path to the CSV file to ingest."),
      kg_name: z
        .string()
        .describe(
          'Name for the knowledge graph (e.g., "sales-data", "customer-records").',
        ),
    },
  },
  async ({ file_path, kg_name }) => {
    try {
      const result = await client().ingest(file_path, { kg: kg_name });
      const entities = Number(result.entities_resolved ?? 0);
      const triples = Number(result.triples_inserted ?? 0);
      return textResult(
        `Ingestion complete: ${entities} entities resolved, ${triples} triples inserted into "${kg_name}".`,
      );
    } catch (err) {
      return errorResult(err);
    }
  },
);

server.registerTool(
  "create_knowledge_graph",
  {
    description:
      "Create a new, empty knowledge graph in the current tenant. Use this " +
      "before ingesting data into a fresh graph (ingest_csv also auto-creates a " +
      "graph, so this is for setting one up explicitly / with a description).",
    inputSchema: {
      name: z
        .string()
        .describe('Name for the new knowledge graph (e.g. "sales-2026").'),
      description: z
        .string()
        .optional()
        .describe("Optional human-readable description of the graph."),
    },
  },
  async ({ name, description }) => {
    try {
      const kg = await client().createKg(name, description);
      return textResult(
        `Created knowledge graph "${String(kg.name ?? name)}".`,
      );
    } catch (err) {
      return errorResult(err);
    }
  },
);

server.registerTool(
  "delete_knowledge_graph",
  {
    description:
      "Delete a knowledge graph and ALL of its data. This is irreversible — " +
      "confirm with the user before calling it.",
    inputSchema: {
      name: z.string().describe("Name of the knowledge graph to delete."),
    },
  },
  async ({ name }) => {
    try {
      await client().deleteKg(name);
      return textResult(`Deleted knowledge graph "${name}".`);
    } catch (err) {
      return errorResult(err);
    }
  },
);

server.registerTool(
  "view_ontology",
  {
    description:
      "View the ontology (types, attributes, relationships) across all knowledge graphs.",
    inputSchema: {},
  },
  async () => {
    try {
      const types = await client().ontologyTypes();
      if (!types.length) return textResult("No ontology types defined yet.");
      const lines: string[] = [];
      for (const t of types) {
        const name = String(t.name ?? "?");
        lines.push(`Type: ${name}`);
        const attrs = (t.attributes ?? []) as Array<Record<string, unknown>>;
        if (attrs.length) {
          lines.push(
            `  Attributes: ${attrs.map((a) => String(a.name ?? "?")).join(", ")}`,
          );
        }
        const rels = (t.relationships ?? []) as Array<Record<string, unknown>>;
        if (rels.length) {
          lines.push(
            `  Relationships: ${rels
              .map(
                (r) =>
                  `${String(r.predicate ?? "?")} -> ${String(r.target_type ?? "?")}`,
              )
              .join(", ")}`,
          );
        }
      }
      return textResult(lines.join("\n"));
    } catch (err) {
      return errorResult(err);
    }
  },
);

function describeChange(c: ResolvedChange): string {
  const verb =
    c.kind === "relationship"
      ? `relationship "${c.name}" from ${c.subject_type} -> ${c.datatype_or_target}`
      : `attribute "${c.name}" (${c.datatype_or_target}) on ${c.subject_type}`;
  return `[${c.action}] ${verb} — confidence ${c.confidence.toFixed(2)}: ${c.reason}`;
}

server.registerTool(
  "evolve_ontology",
  {
    description:
      "Evolve the knowledge-graph ontology from a plain-language description of " +
      "the change you want. You do NOT need to know exact type, attribute, or " +
      'relationship names — just describe the change in natural language (e.g. ' +
      '"track which company a person works for" or "people should have a birth ' +
      'date") and the server resolves it against the existing ontology. ' +
      "High-confidence changes are applied automatically; lower-confidence ones " +
      "are returned as proposals for you to confirm by passing them to " +
      "apply_ontology_change.",
    inputSchema: {
      ask: z
        .string()
        .describe(
          "A plain-language description of the ontology change to make " +
            '(e.g. "track which company a person works for"). No exact schema ' +
            "names required.",
        ),
      knowledge_graph: z
        .string()
        .optional()
        .describe(
          "Optional name of the knowledge graph to scope the change to. " +
            "Use list_knowledge_graphs to see available KGs.",
        ),
    },
  },
  async ({ ask, knowledge_graph }) => {
    try {
      const result = await client().ontologyResolve(ask, { knowledge_graph });
      const lines: string[] = [result.summary];

      if (result.applied.length) {
        lines.push("", "Auto-applied:");
        for (const c of result.applied) lines.push(`  ${describeChange(c)}`);
      } else {
        lines.push("", "Auto-applied: none");
      }

      if (result.proposals.length) {
        lines.push(
          "",
          "Proposals needing confirmation (pass one straight to apply_ontology_change):",
        );
        for (const c of result.proposals) lines.push(`  ${describeChange(c)}`);
        lines.push(
          "",
          "Raw proposal objects:",
          JSON.stringify(result.proposals, null, 2),
        );
      } else {
        lines.push("", "Proposals needing confirmation: none");
      }

      return textResult(lines.join("\n"));
    } catch (err) {
      return errorResult(err);
    }
  },
);

server.registerTool(
  "apply_ontology_change",
  {
    description:
      "Confirm and apply a single ontology change proposal returned by " +
      "evolve_ontology. Pass one of the raw proposal objects through unchanged " +
      "as `proposal`.",
    inputSchema: {
      proposal: z
        .object({
          kind: z.enum(["attribute", "relationship"]),
          subject_type: z.string(),
          name: z.string(),
          datatype_or_target: z.string(),
          action: z.enum(["reuse", "extend", "create"]),
          confidence: z.number(),
          reason: z.string(),
        })
        .describe(
          "A ResolvedChange proposal object exactly as returned by " +
            "evolve_ontology.",
        ),
    },
  },
  async ({ proposal }) => {
    try {
      const result = await client().ontologyApply(proposal as ResolvedChange);
      const lines = [result.summary];
      lines.push("", `Operations applied: ${result.operations}`);
      lines.push(describeChange(result.applied));
      return textResult(lines.join("\n"));
    } catch (err) {
      return errorResult(err);
    }
  },
);

/**
 * Render a kind-tagged agent result (the shape returned by `/agent`) as readable
 * text plus the raw JSON, so an MCP client can both read a summary and act on the
 * machine-readable fields (e.g. carry a `plan_id` back into a confirm call).
 */
function describeAgentResult(r: AgentResult): string {
  const lines: string[] = [];
  switch (r.kind) {
    case "answer": {
      const answer = (r.answer as string | undefined) ?? "(no answer)";
      lines.push(`Answer: ${answer}`);
      if (r.narrative) lines.push(`\n${String(r.narrative)}`);
      if (r.sparql) lines.push(`\nSPARQL:\n${String(r.sparql)}`);
      break;
    }
    case "clarify":
      lines.push(
        `Clarification needed: ${String(r.question ?? "Could you clarify?")}`,
      );
      break;
    case "plan": {
      const steps = Array.isArray(r.steps) ? r.steps : [];
      lines.push(
        `Proposed plan (${steps.length} step${steps.length === 1 ? "" : "s"}) — ` +
          `NOT yet executed. Review, then confirm by calling agent again with ` +
          `confirm_plan_id="${String(r.plan_id ?? "")}".`,
      );
      for (const s of steps as Array<Record<string, unknown>>) {
        const cap = String(s.capability ?? "?");
        const action = String(s.action ?? "?");
        const rationale = s.rationale ? ` — ${String(s.rationale)}` : "";
        lines.push(`  • [${cap}] ${action}${rationale}`);
        const cost = s.cost as Record<string, unknown> | undefined;
        if (cost?.note) lines.push(`      cost: ${String(cost.note)}`);
      }
      break;
    }
    case "result": {
      const steps = Array.isArray(r.steps) ? r.steps : [];
      lines.push(`Executed plan ${String(r.plan_id ?? "")}:`);
      for (const s of steps as Array<Record<string, unknown>>) {
        const status = String(s.status ?? "?");
        const msg = s.message ? ` — ${String(s.message)}` : "";
        lines.push(`  • [${String(s.capability ?? "?")}] ${status}${msg}`);
      }
      break;
    }
    case "error":
      lines.push(`Agent error: ${String(r.error ?? "unknown error")}`);
      break;
    default:
      lines.push(`Agent returned: ${String(r.kind)}`);
  }
  // Always append the raw JSON so the caller can read structured fields
  // (plan_id, steps, rows, …) it needs to drive the next turn.
  lines.push("", "Raw result:", JSON.stringify(r, null, 2));
  return lines.join("\n");
}

server.registerTool(
  "agent",
  {
    description:
      "Talk to the Cograph Ask-AI agent — the single conversational front door " +
      "to a knowledge graph. Send a natural-language message and the agent " +
      "classifies your intent and either ANSWERS a question directly, asks a " +
      "CLARIFYing question, or proposes a PLAN of actions (enrich attributes, " +
      "clean/normalize values, merge duplicates, inspect/extend the ontology). " +
      "A plan is NOT executed until you confirm it: call this tool again with " +
      "the returned plan_id as `confirm_plan_id`. Planning is free; any paid " +
      "step a plan contains (e.g. web enrichment) is authorized server-side at " +
      "execute time, so confirming honors your tenant's entitlements. Prefer " +
      "this over the lower-level tools for conversational, multi-step work.",
    inputSchema: {
      message: z
        .string()
        .optional()
        .describe(
          "Your natural-language message to the agent (e.g. 'how many mentors " +
            "speak Persian?' or 'enrich the company for managers'). Optional " +
            "when confirm_plan_id is set (a confirm turn carries no new message).",
        ),
      kg_name: z
        .string()
        .optional()
        .describe(
          "Knowledge graph to operate within. Use list_knowledge_graphs to see " +
            "available KGs.",
        ),
      type_name: z
        .string()
        .optional()
        .describe(
          "Optional active type to scope the turn to (needed for enrich / clean " +
            "/ dedup planning, e.g. 'Mentor').",
        ),
      urls: z
        .array(z.string())
        .optional()
        .describe(
          "Optional explicit web page links to parse for this turn. When the " +
            "message asks to fill in attributes on existing records, the agent " +
            "extracts those values from these pages; otherwise it pulls a new " +
            "set of records from them. Plain http(s) URLs.",
        ),
      session_id: z
        .string()
        .optional()
        .describe(
          "Optional conversation id to keep multi-turn context across calls.",
        ),
      confirm_plan_id: z
        .string()
        .optional()
        .describe(
          "When set, CONFIRM and EXECUTE the previously-proposed plan with this " +
            "id (the only mutating path) instead of sending a new message. Use " +
            "the plan_id from a prior 'plan' result.",
        ),
    },
  },
  async ({ message, kg_name, type_name, urls, session_id, confirm_plan_id }) => {
    try {
      const result = await client().agent({
        message,
        kgName: kg_name,
        typeName: type_name,
        urls,
        sessionId: session_id,
        confirmPlanId: confirm_plan_id,
      });
      return textResult(describeAgentResult(result));
    } catch (err) {
      return errorResult(err);
    }
  },
);

server.registerTool(
  "list_jobs",
  {
    description:
      "List background jobs (enrichment, dedupe/merge, reconciliation) for the " +
      "tenant, newest first. Use this to check on async work the `agent` tool " +
      "kicked off (e.g. after confirming an enrich or find-duplicates plan): a " +
      "plan's steps run as background jobs, and this is how you see their status.",
    inputSchema: {
      category: z
        .enum(["enrichment", "dedupe", "reconciliation"])
        .optional()
        .describe("Optional filter to a single job category."),
    },
  },
  async ({ category }) => {
    try {
      const jobs = await client().jobs(category ? { category } : {});
      if (!jobs.length) return textResult("No jobs found.");
      const lines = jobs.map((j) => {
        const rec = j as unknown as Record<string, unknown>;
        const id = String(rec.id ?? "?");
        const cat = String(rec.category ?? "?");
        const status = String(rec.status ?? "?");
        const label = rec.label ?? rec.type_name ?? "";
        return `- ${id} [${cat}] ${status}${label ? ` — ${String(label)}` : ""}`;
      });
      return textResult(lines.join("\n"));
    } catch (err) {
      return errorResult(err);
    }
  },
);

server.registerTool(
  "get_job",
  {
    description:
      "Get the full record + progress of a single enrichment job by id (as " +
      "listed by list_jobs). Returns status, tier, per-entity progress and, when " +
      "finished, the applied/staged counts.",
    inputSchema: {
      job_id: z.string().describe("The job id (from list_jobs)."),
    },
  },
  async ({ job_id }) => {
    try {
      const job = (await client().enrichJob(job_id)) as unknown as Record<
        string,
        unknown
      >;
      const status = String(job.status ?? "?");
      const lines = [`Job ${String(job.id ?? job_id)} — ${status}`];
      for (const k of [
        "type_name",
        "resolved_tier",
        "processed",
        "total_entities",
        "applied",
        "staged",
      ]) {
        if (job[k] !== undefined && job[k] !== null)
          lines.push(`  ${k}: ${String(job[k])}`);
      }
      lines.push("", "Raw job:", JSON.stringify(job, null, 2));
      return textResult(lines.join("\n"));
    } catch (err) {
      return errorResult(err);
    }
  },
);

async function main(): Promise<void> {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  process.stderr.write(
    `cograph-mcp failed to start: ${err instanceof Error ? err.message : String(err)}\n`,
  );
  process.exit(1);
});
