import { createInterface } from "node:readline";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { Command } from "commander";
import { Client, CographError } from "./client.js";

// Read version from package.json at runtime so we never drift again.
// dist/cli.js sits next to package.json once published; in dev (`npm link`)
// dist/cli.js sits inside packages/cograph/dist/, so the parent dir is the
// package root either way.
function pkgVersion(): string {
  try {
    const here = dirname(fileURLToPath(import.meta.url));
    const pkg = JSON.parse(readFileSync(join(here, "..", "package.json"), "utf-8"));
    return typeof pkg.version === "string" ? pkg.version : "0.0.0";
  } catch {
    return "0.0.0";
  }
}

function client(): Client {
  return new Client();
}

function printJson(data: unknown): void {
  process.stdout.write(JSON.stringify(data, null, 2) + "\n");
}

function fail(msg: string, code = 1): never {
  process.stderr.write(msg.endsWith("\n") ? msg : msg + "\n");
  process.exit(code);
}

async function withErrors<T>(fn: () => Promise<T>): Promise<T | void> {
  try {
    return await fn();
  } catch (err) {
    if (err instanceof CographError) {
      fail(`Error: ${err.message}`);
    }
    fail(`Error: ${err instanceof Error ? err.message : String(err)}`);
  }
}

async function confirm(prompt: string): Promise<boolean> {
  const rl = createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve) => {
    rl.question(`${prompt} [y/N] `, (ans) => {
      rl.close();
      resolve(ans.trim().toLowerCase() === "y");
    });
  });
}

const program = new Command();
program
  .name("cograph")
  .description("Cograph Knowledge Graph CLI")
  .version(pkgVersion())
  // Default action when no subcommand is given: drop into the interactive
  // shell. So `cograph` (or `npx cograph`) Just Works for the common case;
  // subcommands like `cograph ingest <file>` still route to their own
  // actions because commander dispatches subcommands first.
  .option("--local", "Use http://localhost:8000 and skip login (self-hosted)")
  .option("--no-login", "Skip browser login (assume open-access backend)")
  .action(async (opts: { local?: boolean; login?: boolean }) => {
    const { runShell } = await import("./shell.js");
    await runShell({
      local: opts.local,
      // commander's --no-login inverts: opts.login === false when flag passed.
      noLogin: opts.login === false,
    });
  });

// ---------------------------------------------------------------------------
// kg
// ---------------------------------------------------------------------------

const kg = program.command("kg").description("Manage knowledge graphs");

kg.command("list")
  .description("List knowledge graphs")
  .action(async () => {
    await withErrors(async () => {
      const kgs = await client().listKgs();
      if (!kgs.length) {
        process.stdout.write(
          "No knowledge graphs. Create one with: cograph kg create <name>\n",
        );
        return;
      }
      for (const k of kgs) {
        const name = String(k.name ?? "?");
        const triples = Number(k.triple_count ?? 0);
        const desc = k.description ? ` — ${k.description}` : "";
        const padName = name.padEnd(20, " ");
        const padTriples = String(triples).padStart(6, " ");
        process.stdout.write(`  ${padName} ${padTriples} triples${desc}\n`);
      }
    });
  });

kg.command("create <name>")
  .description("Create a knowledge graph")
  .option("-d, --description <text>", "Description")
  .action(async (name: string, opts: { description?: string }) => {
    await withErrors(async () => {
      const created = await client().createKg(name, opts.description);
      process.stdout.write(`Created knowledge graph: ${created.name ?? name}\n`);
    });
  });

kg.command("delete <name>")
  .description("Delete a knowledge graph")
  .action(async (name: string) => {
    await withErrors(async () => {
      await client().deleteKg(name);
      process.stdout.write(`Deleted knowledge graph: ${name}\n`);
    });
  });

// ---------------------------------------------------------------------------
// ingest
// ---------------------------------------------------------------------------

program
  .command("ingest [file]")
  .description("Ingest data from a file or --text")
  .option("-t, --text <text>", "Inline text to ingest")
  .option("--kg <name>", "Target knowledge graph name")
  .option(
    "-f, --format <fmt>",
    "Override format detection (text|csv|json)",
  )
  .action(
    async (
      file: string | undefined,
      opts: { text?: string; kg?: string; format?: string },
    ) => {
      await withErrors(async () => {
        const c = client();
        if (opts.text) {
          process.stdout.write(
            `Ingesting text (${opts.text.length.toLocaleString()} chars)...\n`,
          );
          const result = await c.ingest(opts.text, {
            kg: opts.kg,
            contentType: opts.format ?? "text",
          });
          printIngestResult(result);
          return;
        }
        if (!file) {
          fail("Provide a file or --text");
        }
        // ingest() handles file reading + format detection + CSV two-step flow.
        process.stdout.write(`Ingesting ${file}...\n`);
        const result = await c.ingest(file, {
          kg: opts.kg,
          contentType: opts.format,
        });
        printIngestResult(result);
      });
    },
  );

function printIngestResult(result: Record<string, unknown>): void {
  const num = (k: string) => Number(result[k] ?? 0);
  process.stdout.write(`  Entities extracted: ${num("entities_extracted")}\n`);
  process.stdout.write(`  Entities resolved:  ${num("entities_resolved")}\n`);
  process.stdout.write(`  Triples inserted:   ${num("triples_inserted")}\n`);
  const types = result.types_created;
  if (Array.isArray(types) && types.length) {
    process.stdout.write(`  Types created:      ${types.join(", ")}\n`);
  }
  const rejections = result.rejections;
  if (Array.isArray(rejections) && rejections.length) {
    process.stdout.write(`  Rejections:         ${rejections.length}\n`);
  }
}

// ---------------------------------------------------------------------------
// ask
// ---------------------------------------------------------------------------

program
  .command("ask <question>")
  .description("Ask a natural language question")
  .option("--kg <name>", "Knowledge graph to query")
  .option("-d, --debug", "Show SPARQL and latency breakdown")
  .option("-m, --model <model>", "Override query model")
  .action(
    async (
      question: string,
      opts: { kg?: string; debug?: boolean; model?: string },
    ) => {
      await withErrors(async () => {
        if (opts.model) process.stdout.write(`Model: ${opts.model}\n`);
        process.stdout.write(`Q: ${question}\n`);
        process.stdout.write("Generating answer...\n");
        const t0 = Date.now();
        const result = await client().ask(question, {
          kg: opts.kg,
          model: opts.model,
        });
        const roundtripMs = Date.now() - t0;
        process.stdout.write(`\nA: ${result.answer ?? "No answer"}\n`);
        if (opts.debug) {
          process.stdout.write(`\nSPARQL:\n${result.sparql ?? ""}\n`);
          const timing = (result.timing ?? {}) as Record<string, unknown>;
          if (Object.keys(timing).length) {
            process.stdout.write(`\n${"─".repeat(40)}\n`);
            process.stdout.write(
              `${"Stage".padEnd(25)} ${"Time".padStart(10)}\n`,
            );
            process.stdout.write(`${"─".repeat(40)}\n`);
            for (const [key, val] of Object.entries(timing)) {
              if (key === "attempts") {
                process.stdout.write(
                  `${"Attempts".padEnd(25)} ${String(val).padStart(10)}\n`,
                );
              } else if (typeof val === "string") {
                const label = key
                  .replace(/_/g, " ")
                  .replace(/\b\w/g, (c) => c.toUpperCase());
                process.stdout.write(
                  `${label.padEnd(25)} ${val.padStart(10)}\n`,
                );
              } else {
                const label = key
                  .replace(/_ms$/, "")
                  .replace(/_/g, " ")
                  .replace(/\b\w/g, (c) => c.toUpperCase());
                const num = typeof val === "number" ? val : Number(val);
                process.stdout.write(
                  `${label.padEnd(25)} ${num.toFixed(1).padStart(8)}ms\n`,
                );
              }
            }
            process.stdout.write(`${"─".repeat(40)}\n`);
            process.stdout.write(
              `${"Client roundtrip".padEnd(25)} ${roundtripMs.toFixed(1).padStart(8)}ms\n`,
            );
          }
        }
      });
    },
  );

// ---------------------------------------------------------------------------
// ontology
// ---------------------------------------------------------------------------

const onto = program.command("ontology").description("View ontology");

onto
  .command("types")
  .description("List ontology types")
  .action(async () => {
    await withErrors(async () => {
      const types = await client().ontologyTypes();
      if (!types.length) {
        process.stdout.write("No ontology types defined.\n");
        return;
      }
      for (const t of types) {
        const parent = t.parent_type
          ? ` (subClassOf ${t.parent_type})`
          : "";
        const desc = t.description ? ` — ${t.description}` : "";
        process.stdout.write(`  ${t.name}${parent}${desc}\n`);
        const attrs = (t.attributes ?? []) as Array<Record<string, unknown>>;
        for (const a of attrs) {
          process.stdout.write(
            `    .${a.name} (${a.datatype ?? "string"})\n`,
          );
        }
      }
    });
  });

// ---------------------------------------------------------------------------
// vis
// ---------------------------------------------------------------------------

program
  .command("vis <type>")
  .description("Visualise a type — instance count, attribute coverage, top relations")
  .option("--kg <name>", "Knowledge graph to inspect")
  .action(async (typeName: string, opts: { kg?: string }) => {
    await withErrors(async () => {
      const c = client();

      // Resolve KG: use --kg flag, or pick first available KG.
      let kg = opts.kg;
      if (!kg) {
        const kgs = await c.listKgs();
        if (!kgs.length) {
          fail("No knowledge graphs found. Run 'cograph ingest' first.");
        }
        kg = String(kgs[0].name ?? "");
      }

      let summary: import("./client.js").TypeSummary;
      try {
        summary = await c.typeSummary(kg, typeName);
      } catch {
        fail(`Type '${typeName}' not found in KG '${kg}'.`);
      }

      const { entity_count, attributes, relationships, description, parent_type } = summary;
      const header = `${typeName}${parent_type ? ` (subClassOf ${parent_type})` : ""} — ${entity_count.toLocaleString()} instances`;
      process.stdout.write(`\n${header}\n${"─".repeat(header.length)}\n`);
      if (description) process.stdout.write(`${description}\n`);

      // Attributes table
      if (attributes.length) {
        process.stdout.write(`\nAttributes (${attributes.length}):\n`);
        const sorted = [...attributes].sort((a, b) => b.coverage_pct - a.coverage_pct);
        for (const a of sorted.slice(0, 10)) {
          const bar = "█".repeat(Math.round(a.coverage_pct / 10));
          const pct = `${a.coverage_pct}%`.padStart(6);
          process.stdout.write(`  ${a.name.padEnd(24)} ${pct}  ${bar}\n`);
        }
        if (attributes.length > 10) {
          process.stdout.write(`  … and ${attributes.length - 10} more\n`);
        }
      }

      // Relations table
      if (relationships.length) {
        process.stdout.write(`\nRelationships (${relationships.length}):\n`);
        for (const r of relationships.slice(0, 8)) {
          const target = r.target_type ? ` → ${r.target_type}` : "";
          const pct = `${r.coverage_pct}%`.padStart(6);
          const avg = r.avg_degree ? ` (avg ${r.avg_degree})` : "";
          process.stdout.write(`  ${(r.name + target).padEnd(36)} ${pct}${avg}\n`);
        }
      }

      const tenant = c.tenant;
      const explorerUrl = `https://app.cograph.cloud/${tenant}/explore/${encodeURIComponent(typeName)}?kg=${encodeURIComponent(kg)}`;
      process.stdout.write(`\n→ Open visually at ${explorerUrl}\n`);
      process.stdout.write("  (Sign in for interactive viz, search, and click-to-enrich.)\n\n");
    });
  });

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

program
  .command("clear")
  .description("Clear data")
  .option("--kg <name>", "Clear a specific knowledge graph")
  .option(
    "--include-ontology",
    "Also clear the ontology (only meaningful when --kg is omitted)",
    false,
  )
  .option("-y, --yes", "Skip confirmation", false)
  .action(
    async (opts: { kg?: string; includeOntology?: boolean; yes?: boolean }) => {
      await withErrors(async () => {
        let msg: string;
        if (opts.kg) {
          msg = `Clear KG '${opts.kg}'?`;
        } else if (opts.includeOntology) {
          msg = "Clear EVERYTHING including ontology?";
        } else {
          msg = "Clear all instance data (ontology preserved)?";
        }

        if (!opts.yes) {
          const ok = await confirm(msg);
          if (!ok) {
            process.stdout.write("Cancelled.\n");
            return;
          }
        }

        const c = client();
        if (opts.kg) {
          await c.deleteKg(opts.kg);
          process.stdout.write(`Cleared KG: ${opts.kg}\n`);
          return;
        }

        // Bulk-clear via /query + DELETE /triples — same loop the Python CLI uses.
        const tenant = c.tenant;
        const baseUrl = `${c.baseUrl}/graphs/${tenant}`;
        const headers: Record<string, string> = {
          "Content-Type": "application/json",
        };
        if (c.apiKey) headers["X-API-Key"] = c.apiKey;

        const filters = opts.includeOntology
          ? ""
          : `FILTER(CONTAINS(STR(?s), '/entities/') || CONTAINS(STR(?s), '/onto/') || CONTAINS(STR(?s), '/kgs/'))`;
        const query = `SELECT ?s ?p ?o FROM <https://cograph.tech/graphs/${tenant}> WHERE { ?s ?p ?o . ${filters} } LIMIT 1000`;

        process.stdout.write("Clearing...\n");
        let deleted = 0;
        for (let i = 0; i < 50; i++) {
          const fetchRes = await fetch(`${baseUrl}/query`, {
            method: "POST",
            headers,
            body: JSON.stringify({ query }),
          });
          if (!fetchRes.ok) break;
          const data = (await fetchRes.json()) as {
            bindings?: Array<Record<string, unknown>>;
          };
          const bindings = data.bindings ?? [];
          if (!bindings.length) break;
          const triples = bindings
            .filter((b) => b.s)
            .map((b) => ({
              subject: b.s,
              predicate: b.p,
              object: b.o,
            }));
          for (let j = 0; j < triples.length; j += 100) {
            await fetch(`${baseUrl}/triples`, {
              method: "DELETE",
              headers,
              body: JSON.stringify({ triples: triples.slice(j, j + 100) }),
            });
          }
          deleted += triples.length;
        }
        process.stdout.write(`Deleted ${deleted} triples\n`);
      });
    },
  );

// ---------------------------------------------------------------------------
// login
// ---------------------------------------------------------------------------

program
  .command("login")
  .description("Sign in via your browser and save an API key")
  .action(async () => {
    const { runLogin } = await import("./login.js");
    await runLogin();
  });

// ---------------------------------------------------------------------------
// shell
// ---------------------------------------------------------------------------

program
  .command("shell")
  .description("Start an interactive REPL")
  .option("--kg <name>", "Knowledge graph to use")
  .option("--local", "Use http://localhost:8000 and skip login (self-hosted)")
  .option("--no-login", "Skip browser login (assume open-access backend)")
  .action(
    async (opts: { kg?: string; local?: boolean; login?: boolean }) => {
      // Parent program also accepts --local/--no-login (so `cograph --local`
      // works without a subcommand). When commander parses
      // `cograph shell --local`, the parent sees --local first and the
      // subcommand never gets it — so merge from program.opts() too.
      const parentOpts = program.opts() as {
        local?: boolean;
        login?: boolean;
      };
      const { runShell } = await import("./shell.js");
      await runShell({
        kg: opts.kg,
        local: opts.local || parentOpts.local,
        noLogin: opts.login === false || parentOpts.login === false,
      });
    },
  );

// ---------------------------------------------------------------------------

program.parseAsync(process.argv).catch((err) => {
  fail(`Error: ${err instanceof Error ? err.message : String(err)}`);
});

// silence unused import warning if ever needed
void printJson;
