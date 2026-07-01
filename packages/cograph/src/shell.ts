import * as readline from "node:readline";
import { stdin, stdout } from "node:process";
import { randomUUID } from "node:crypto";
import {
  Client,
  CographError,
  type TypeCount,
  type EnrichJob,
  type ConflictReview,
  type JobSummary,
  type EnrichmentTier,
} from "./client.js";
import { renderAgentResult } from "./agentRender.js";
import { writeConfig } from "./config.js";

const CYAN = "\x1b[36m";
const CYAN_BOLD = "\x1b[1;36m";
const DIM = "\x1b[2m";
const RED = "\x1b[31m";
const GREEN = "\x1b[32m";
const YELLOW = "\x1b[33m";
const BOLD = "\x1b[1m";
const RESET = "\x1b[0m";

function fmtNum(n: number): string {
  return n.toLocaleString("en-US");
}

function canRenderBlockArt(): boolean {
  // Apple_Terminal (macOS Terminal.app) treats the block-shade chars (█░)
  // we use in the banner as East Asian Ambiguous Width = 2 cells, so each
  // banner row renders at double width and wraps mid-letter. iTerm,
  // WezTerm, Kitty, VS Code, Cursor, etc. all treat them as 1 cell and
  // render the art correctly. Skip the banner on Apple_Terminal and show
  // a plain header instead. Force on/off via COGRAPH_BANNER=on|off.
  const force = process.env.COGRAPH_BANNER;
  if (force === "on") return true;
  if (force === "off") return false;
  if (!process.stdout.isTTY) return false;
  if (process.env.TERM_PROGRAM === "Apple_Terminal") return false;
  return true;
}

function showBanner(): void {
  if (canRenderBlockArt()) {
    const lines = [
      "",
      `${CYAN}       ███████    ██████   █████ ███████████   █████████${RESET}`,
      `${CYAN}     ███░░░░░███ ░░██████ ░░███ ░█░░░███░░░█  ███░░░░░███${RESET}`,
      `${CYAN}    ███     ░░███ ░███░███ ░███ ░   ░███  ░  ░███    ░███${RESET}`,
      `${CYAN}    ░███      ░███ ░███░░███░███     ░███     ░███████████${RESET}`,
      `${CYAN}    ░███      ░███ ░███ ░░██████     ░███     ░███░░░░░███${RESET}`,
      `${CYAN}    ░░███     ███  ░███  ░░█████     ░███     ░███    ░███${RESET}`,
      `${CYAN}     ░░░███████░   █████  ░░█████    █████    █████   █████${RESET}`,
      `${CYAN}       ░░░░░░░    ░░░░░    ░░░░░    ░░░░░    ░░░░░   ░░░░░${RESET}`,
      "",
      `${DIM}    The object graph for AI agents${RESET}`,
      "",
    ];
    for (const l of lines) stdout.write(l + "\n");
  } else {
    stdout.write(`\n  ${CYAN_BOLD}ONTA${RESET}\n`);
    stdout.write(`  ${DIM}The object graph for AI agents${RESET}\n\n`);
  }
  showCommands();
}

function showCommands(): void {
  const rows: Array<[string, string]> = [
    ["/ingest <file> ...", "Ingest a CSV/JSON/text file"],
    ["/ask <question>", "Ask in natural language"],
    ["/agent <message>", "Unified Ask-AI agent — answers, plans, runs actions"],
    ["/kg list", "List your knowledge graphs"],
    ["/kg switch <name>", "Switch to a different KG"],
    ["/kg create <name>", "Create a new KG and switch to it"],
    ["/kg delete <name>", "Delete a KG (irreversible)"],
    ["/tenant list", "List tenants you can access"],
    ["/tenant use <id>", "Switch tenant (then pick a KG)"],
    ["/types [query]", "List types in the current KG (with entity counts)"],
    ["/type <name>", "Drill into one type — attributes, relationships, samples"],
    ["/type <name> --system", "…also include auto-attached system attributes"],
    ["/enrich <Type> <attr> ...", "Plan + run an enrichment job (interactive)"],
    ["/enrich watch <job_id>", "Live progress for a running job"],
    ["/enrich jobs", "List recent enrichment jobs"],
    ["/enrich review <job_id>", "Walk through conflicts and accept/reject"],
    ["/login", "Re-authenticate (browser)"],
    ["/status", "Show graph stats"],
    ["/reset", "Clear the current KG"],
    ["/help", "Show this command list"],
    ["/quit", "Exit"],
  ];
  const colWidth = Math.max(...rows.map((r) => r[0].length));
  for (const [cmd, desc] of rows) {
    const pad = " ".repeat(colWidth - cmd.length);
    stdout.write(`    ${CYAN_BOLD}${cmd}${RESET}${pad}   ${DIM}${desc}${RESET}\n`);
  }
  stdout.write("\n");
}

function printError(msg: string): void {
  stdout.write(`  ${RED}✗${RESET} ${msg}\n`);
}

interface KgInfo {
  name: string;
  triple_count: number;
}

async function fetchKg(client: Client, name: string): Promise<KgInfo | null> {
  try {
    const kgs = await client.listKgs();
    const found = kgs.find((k) => (k as { name?: string }).name === name);
    if (!found) return null;
    const tc = (found as { triple_count?: number }).triple_count ?? 0;
    return { name, triple_count: typeof tc === "number" ? tc : 0 };
  } catch {
    return null;
  }
}

function ask(rl: readline.Interface, prompt: string): Promise<string> {
  return new Promise((resolve) => {
    rl.question(prompt, (answer) => resolve(answer));
  });
}

async function selectKg(
  client: Client,
  rl: readline.Interface,
): Promise<string | null> {
  let kgs: Array<Record<string, unknown>> = [];
  try {
    kgs = await client.listKgs();
  } catch (err) {
    printError(
      `Could not list knowledge graphs: ${err instanceof Error ? err.message : String(err)}`,
    );
    return null;
  }

  if (kgs.length === 0) {
    stdout.write(
      `  ${DIM}No knowledge graphs found. Enter a name to create your first KG.${RESET}\n`,
    );
    const name = (await ask(rl, "  KG name: ")).trim();
    if (!name) return null;
    // Persist immediately. Without this, the name only existed as a local
    // string until the user ran /ingest, so quitting before ingesting lost
    // the KG entirely — and the next shell session showed "No KGs found"
    // again.
    try {
      await client.createKg(name);
      stdout.write(`  ${GREEN}✓${RESET} Created ${BOLD}${name}${RESET}\n`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      // 409 / "already exists" is fine — someone created it between listKgs
      // and now, or the user retried. Anything else is a real failure.
      if (!/already exists|409/i.test(msg)) {
        printError(`Could not create knowledge graph: ${msg}`);
        return null;
      }
    }
    return name;
  }

  if (kgs.length === 1) {
    const only = (kgs[0] as { name?: string }).name;
    if (only) {
      stdout.write(`  ${DIM}Using only available KG: ${BOLD}${only}${RESET}\n`);
      return only;
    }
  }

  stdout.write(`  ${BOLD}Available knowledge graphs:${RESET}\n`);
  kgs.forEach((kg, i) => {
    const n = (kg as { name?: string }).name ?? "?";
    const tc = (kg as { triple_count?: number }).triple_count ?? 0;
    stdout.write(`    ${CYAN}${i + 1}${RESET}. ${n} ${DIM}(${fmtNum(tc)} triples)${RESET}\n`);
  });
  const pick = (await ask(rl, "  Select KG [1]: ")).trim() || "1";
  const idx = Number.parseInt(pick, 10);
  if (Number.isFinite(idx) && idx >= 1 && idx <= kgs.length) {
    const name = (kgs[idx - 1] as { name?: string }).name;
    if (name) return name;
  }
  // Allow typing a name directly
  if (pick && !/^\d+$/.test(pick)) return pick;
  printError("Invalid selection.");
  return null;
}

/**
 * Tiny live-line spinner. Returns handles to update the trailing text and
 * stop. We use \r + clear-line escape so the line redraws in place.
 */
function startSpinner(initial: string): {
  setText: (text: string) => void;
  stop: () => void;
} {
  const frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
  let frame = 0;
  let text = initial;
  let stopped = false;

  const draw = (): void => {
    if (stopped) return;
    // \x1b[2K = clear entire line; \r = carriage return
    stdout.write(`\r\x1b[2K  ${CYAN}${frames[frame]}${RESET} ${text}`);
    frame = (frame + 1) % frames.length;
  };
  draw();
  const tick = setInterval(draw, 80);

  return {
    setText(t: string) {
      text = t;
    },
    stop() {
      stopped = true;
      clearInterval(tick);
      stdout.write("\r\x1b[2K");
    },
  };
}

async function cmdIngest(
  client: Client,
  kg: string,
  args: string[],
): Promise<void> {
  if (args.length === 0) {
    stdout.write(`  ${YELLOW}Usage:${RESET} /ingest <file> [<file>...]\n`);
    return;
  }
  for (const file of args) {
    const sp = startSpinner(`Inferring schema from ${file}...`);
    try {
      const result = await client.ingest(file, {
        kg,
        onProgress: ({
          rowsProcessed,
          totalRows,
          entitiesResolved,
          triplesInserted,
        }) => {
          const pct = Math.round((rowsProcessed / totalRows) * 100);
          sp.setText(
            `Ingesting ${file} ${DIM}·${RESET} ${BOLD}${pct}%${RESET} ` +
              `${DIM}(${fmtNum(rowsProcessed)}/${fmtNum(totalRows)} rows · ` +
              `${fmtNum(entitiesResolved)} entities · ${fmtNum(triplesInserted)} triples)${RESET}`,
          );
        },
      });
      sp.stop();
      const ents =
        (result as { entities_resolved?: number }).entities_resolved ?? 0;
      const trip =
        (result as { triples_inserted?: number }).triples_inserted ?? 0;
      stdout.write(
        `  ${GREEN}✓${RESET} ${file} ${DIM}·${RESET} ${fmtNum(ents)} entities · ${fmtNum(trip)} triples\n`,
      );
    } catch (err) {
      sp.stop();
      if (err instanceof CographError) printError(err.message);
      else printError(err instanceof Error ? err.message : String(err));
    }
  }
}

async function cmdAsk(
  client: Client,
  kg: string,
  question: string,
): Promise<void> {
  const q = question.trim();
  if (!q) {
    stdout.write(`  ${YELLOW}Usage:${RESET} /ask <your question>\n`);
    return;
  }
  try {
    const result = await client.ask(q, { kg });
    const answer =
      (result as { narrative_answer?: string }).narrative_answer ||
      (result as { answer?: string }).answer ||
      "No answer generated.";
    stdout.write("\n");
    stdout.write(`  ${answer}\n`);
    stdout.write("\n");
  } catch (err) {
    if (err instanceof CographError) printError(err.message);
    else printError(err instanceof Error ? err.message : String(err));
  }
}

/**
 * `/agent <message>` — one turn of the unified Ask-AI agent inside the REPL.
 *
 * Sends the message (threading the per-session `sessionId` for multi-turn
 * continuity), renders the kind-tagged response with the shared renderer, and —
 * because the shell IS interactive — when the response is a `plan`, prompts
 * `Confirm & run? [y/N]`. On `y` it confirms the plan (the only mutating path)
 * and renders the `result`. Mirrors the cli.ts agent command, but the confirm
 * is an inline prompt rather than --yes/--confirm.
 */
async function cmdAgent(
  client: Client,
  kg: string,
  rl: readline.Interface,
  sessionId: string,
  message: string,
): Promise<void> {
  const msg = message.trim();
  if (!msg) {
    stdout.write(`  ${YELLOW}Usage:${RESET} /agent <your message>\n`);
    return;
  }
  const context = { kgName: kg, sessionId };
  const sp = startSpinner("Thinking...");
  let result;
  try {
    result = await client.agent({ message: msg, ...context });
  } catch (err) {
    sp.stop();
    if (err instanceof CographError) printError(err.message);
    else printError(err instanceof Error ? err.message : String(err));
    return;
  }
  sp.stop();
  renderAgentResult(result);

  // Only a plan awaits confirmation. Prompt inline; on "y", confirm + execute.
  if (result.kind === "plan") {
    const planId =
      typeof result.plan_id === "string" ? result.plan_id : undefined;
    if (!planId) return;
    const ans = (await ask(rl, `  ${YELLOW}Confirm & run?${RESET} [y/N]: `))
      .trim()
      .toLowerCase();
    if (ans !== "y" && ans !== "yes") {
      stdout.write(`  ${DIM}Not run. Plan ${planId} kept.${RESET}\n`);
      return;
    }
    const sp2 = startSpinner("Running plan...");
    let executed;
    try {
      executed = await client.agent({ confirmPlanId: planId, ...context });
    } catch (err) {
      sp2.stop();
      if (err instanceof CographError) printError(err.message);
      else printError(err instanceof Error ? err.message : String(err));
      return;
    }
    sp2.stop();
    renderAgentResult(executed);
  }
}

async function cmdStatus(client: Client, kg: string): Promise<void> {
  try {
    const info = await fetchKg(client, kg);
    stdout.write("\n");
    stdout.write(`  ${BOLD}KG${RESET}       ${kg}\n`);
    if (info) {
      stdout.write(`  ${BOLD}Triples${RESET}  ${fmtNum(info.triple_count)}\n`);
    } else {
      stdout.write(`  ${BOLD}Triples${RESET}  ${DIM}(empty)${RESET}\n`);
    }
    try {
      const types = await client.ontologyTypes();
      const names = types
        .map((t) => (t as { name?: string }).name)
        .filter((n): n is string => Boolean(n));
      if (names.length > 0) {
        stdout.write(`  ${BOLD}Types${RESET}    ${names.join(", ")}\n`);
      } else {
        stdout.write(`  ${BOLD}Types${RESET}    ${DIM}(none)${RESET}\n`);
      }
    } catch (err) {
      printError(
        `Could not list ontology types: ${err instanceof Error ? err.message : String(err)}`,
      );
    }
    stdout.write("\n");
  } catch (err) {
    if (err instanceof CographError) printError(err.message);
    else printError(err instanceof Error ? err.message : String(err));
  }
}

async function cmdReset(
  client: Client,
  kg: string,
  rl: readline.Interface,
): Promise<boolean> {
  const confirm = (
    await ask(rl, `  ${YELLOW}Delete KG "${kg}"?${RESET} [y/N]: `)
  )
    .trim()
    .toLowerCase();
  if (confirm !== "y" && confirm !== "yes") {
    stdout.write(`  ${DIM}Cancelled.${RESET}\n`);
    return false;
  }
  try {
    await client.deleteKg(kg);
    stdout.write(`  ${GREEN}✓${RESET} Graph cleared.\n`);
    return true;
  } catch (err) {
    if (err instanceof CographError) printError(err.message);
    else printError(err instanceof Error ? err.message : String(err));
    return false;
  }
}

async function cmdTypes(
  client: Client,
  kg: string,
  query: string,
): Promise<void> {
  const sp = startSpinner(
    query ? `Searching types matching "${query}"...` : "Loading types...",
  );
  let types: TypeCount[];
  try {
    types = await client.typeCounts(kg);
  } catch (err) {
    sp.stop();
    if (err instanceof CographError) printError(err.message);
    else printError(err instanceof Error ? err.message : String(err));
    return;
  }
  sp.stop();

  const q = query.trim().toLowerCase();
  const filtered = q
    ? types.filter((t) => t.name.toLowerCase().includes(q))
    : types;

  if (filtered.length === 0) {
    if (types.length === 0) {
      stdout.write(
        `  ${DIM}No types yet in ${BOLD}${kg}${RESET}${DIM}. Try ${RESET}/ingest <file>${DIM} first.${RESET}\n`,
      );
    } else {
      stdout.write(
        `  ${DIM}No types match "${query}". Try ${RESET}/types${DIM} for the full list.${RESET}\n`,
      );
    }
    return;
  }

  // Right-align counts; leave room for the longest name we'll print.
  const nameWidth = Math.max(
    "Type".length,
    ...filtered.map((t) => t.name.length),
  );
  const countWidth = Math.max(
    "Entities".length,
    ...filtered.map((t) => fmtNum(t.entity_count).length),
  );
  stdout.write("\n");
  stdout.write(
    `  ${BOLD}${"Type".padEnd(nameWidth)}   ${"Entities".padStart(countWidth)}${RESET}\n`,
  );
  let total = 0;
  for (const t of filtered) {
    total += t.entity_count;
    stdout.write(
      `  ${CYAN}${t.name.padEnd(nameWidth)}${RESET}   ${fmtNum(t.entity_count).padStart(countWidth)}\n`,
    );
  }
  stdout.write("\n");
  const summary = q
    ? `${filtered.length} match${filtered.length === 1 ? "" : "es"}.`
    : `${filtered.length} type${filtered.length === 1 ? "" : "s"}, ${fmtNum(total)} entities total.`;
  stdout.write(`  ${DIM}${summary}${RESET}\n`);
  stdout.write(
    `  ${DIM}Drill in:  ${RESET}/type <name>${DIM}   Filter:  ${RESET}/types <query>${DIM}${RESET}\n\n`,
  );
}

/**
 * Resolve a user-supplied type name to a canonical type. Case-insensitive
 * exact match wins; otherwise we fall back to prefix match. If multiple
 * types share a prefix, prompt the user to pick from a numbered list.
 */
async function resolveTypeName(
  client: Client,
  kg: string,
  rl: readline.Interface,
  input: string,
): Promise<string | null> {
  const types = await client.typeCounts(kg);
  if (types.length === 0) {
    printError(`No types in ${kg} yet. Try /ingest <file> first.`);
    return null;
  }
  const q = input.trim().toLowerCase();
  const exact = types.find((t) => t.name.toLowerCase() === q);
  if (exact) return exact.name;
  const prefix = types.filter((t) => t.name.toLowerCase().startsWith(q));
  const matches = prefix.length > 0
    ? prefix
    : types.filter((t) => t.name.toLowerCase().includes(q));
  if (matches.length === 0) {
    printError(
      `No type matches "${input}". Try /types to see what's available.`,
    );
    return null;
  }
  if (matches.length === 1) return matches[0]!.name;
  stdout.write(`  ${DIM}Multiple types match "${input}":${RESET}\n`);
  matches.forEach((t, i) => {
    stdout.write(
      `    ${CYAN}${i + 1}${RESET}. ${BOLD}${t.name}${RESET} ${DIM}(${fmtNum(t.entity_count)} entities)${RESET}\n`,
    );
  });
  const pick = (await ask(rl, `  Pick [1]: `)).trim() || "1";
  const idx = Number.parseInt(pick, 10);
  if (Number.isFinite(idx) && idx >= 1 && idx <= matches.length) {
    return matches[idx - 1]!.name;
  }
  printError("Invalid selection.");
  return null;
}

async function cmdType(
  client: Client,
  kg: string,
  rl: readline.Interface,
  input: string,
): Promise<void> {
  // Pull off any --system flag so the rest can be treated as the type name.
  // Conservative parse: only the literal flag, anywhere in the input.
  const tokens = splitArgs(input.trim());
  const includeSystem = tokens.includes("--system");
  const nameTokens = tokens.filter((t) => t !== "--system");
  const nameInput = nameTokens.join(" ").trim();
  if (!nameInput) {
    stdout.write(`  ${YELLOW}Usage:${RESET} /type <name> [--system]\n`);
    return;
  }
  const name = await resolveTypeName(client, kg, rl, nameInput);
  if (!name) return;

  const sp = startSpinner(`Loading ${name}...`);
  let usage;
  try {
    usage = await client.typeUsage(kg, name, { includeSystem });
  } catch (err) {
    sp.stop();
    if (err instanceof CographError) printError(err.message);
    else printError(err instanceof Error ? err.message : String(err));
    return;
  }
  sp.stop();

  const total = usage.entity_count;
  const pct = (n: number): string =>
    total > 0 ? `${Math.round((n / total) * 100).toString().padStart(3)}%` : "  —";

  // Dedup: when the resolver produces both a literal attribute and a typed
  // relationship for the same column (e.g. .title literal + .title→JobTitle),
  // we collapse to a single relationship row and surface the literal count
  // as a "(+775 string)" annotation. The relationship row "wins" because
  // its count is the union upper bound (every entity with a typed link)
  // and it's the richer fact. Pure literals and pure relationships are
  // unaffected.
  const relNames = new Set(usage.relationships.map((r) => r.name));
  const attrLitByName = new Map(usage.attributes.map((a) => [a.name, a]));
  const litOnlyAttrs = usage.attributes.filter((a) => !relNames.has(a.name));

  stdout.write("\n");
  stdout.write(
    `  ${BOLD}${usage.name}${RESET}  ${DIM}${fmtNum(total)} entities${RESET}\n`,
  );
  if (usage.description) {
    stdout.write(`  ${DIM}${usage.description}${RESET}\n`);
  }
  if (usage.parent_type) {
    stdout.write(`  ${DIM}subClassOf  ${usage.parent_type}${RESET}\n`);
  }

  if (litOnlyAttrs.length > 0) {
    stdout.write(
      `\n  ${BOLD}Attributes (${litOnlyAttrs.length})${RESET}\n`,
    );
    const nameW = Math.max(
      ...litOnlyAttrs.map((a) => a.name.length + 1),
      8,
    );
    const typeW = Math.max(
      ...litOnlyAttrs.map((a) => a.datatype.length),
      8,
    );
    const cntW = Math.max(
      ...litOnlyAttrs.map((a) => fmtNum(a.count).length),
      4,
    );
    for (const a of litOnlyAttrs) {
      const dotName = `.${a.name}`;
      stdout.write(
        `    ${CYAN}${dotName.padEnd(nameW)}${RESET}  ${DIM}${a.datatype.padEnd(typeW)}${RESET}  ${fmtNum(a.count).padStart(cntW)}  ${DIM}(${pct(a.count)})${RESET}\n`,
      );
    }
  }

  if (usage.relationships.length > 0) {
    stdout.write(
      `\n  ${BOLD}Relationships (${usage.relationships.length})${RESET}\n`,
    );
    const nameW = Math.max(
      ...usage.relationships.map((r) => r.name.length + 1),
      8,
    );
    const tgtW = Math.max(
      ...usage.relationships.map((r) => (r.target_type ?? "?").length),
      6,
    );
    for (const r of usage.relationships) {
      const dotName = `.${r.name}`;
      const tgt = r.target_type ?? "?";
      const lit = attrLitByName.get(r.name);
      const litNote = lit
        ? ` ${DIM}(+${fmtNum(lit.count)} ${lit.datatype})${RESET}`
        : "";
      stdout.write(
        `    ${CYAN}${dotName.padEnd(nameW)}${RESET}  ${DIM}→${RESET} ${BOLD}${tgt.padEnd(tgtW)}${RESET}  ${fmtNum(r.count).padStart(6)}  ${DIM}(${pct(r.count)})${RESET}${litNote}\n`,
      );
    }
  }

  if (usage.samples.length > 0) {
    stdout.write(`\n  ${BOLD}Sample entities${RESET}\n`);
    usage.samples.forEach((s, i) => {
      const label = s.label || s.uri.split("/").pop() || s.uri;
      stdout.write(`    ${DIM}${i + 1}.${RESET} ${label}\n`);
    });
  }

  if (
    usage.attributes.length === 0 &&
    usage.relationships.length === 0 &&
    total === 0
  ) {
    stdout.write(
      `\n  ${DIM}Type defined in the ontology but no instances yet in ${kg}.${RESET}\n`,
    );
  }
  stdout.write("\n");
}

function lastUriSegment(uri: string): string {
  if (!uri) return uri;
  const hash = uri.lastIndexOf("#");
  if (hash >= 0 && hash < uri.length - 1) return uri.slice(hash + 1);
  const slash = uri.lastIndexOf("/");
  if (slash >= 0 && slash < uri.length - 1) return uri.slice(slash + 1);
  return uri;
}

function relativeTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const t = Date.parse(iso);
  if (!Number.isFinite(t)) return "—";
  const diffMs = Date.now() - t;
  const s = Math.max(0, Math.floor(diffMs / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  const d = Math.floor(h / 24);
  return `${d}d ago`;
}

function progressBar(processed: number, total: number, width = 20): string {
  if (!total || total <= 0) return "[" + " ".repeat(width) + "]";
  const ratio = Math.max(0, Math.min(1, processed / total));
  const filled = Math.round(ratio * width);
  return "[" + "█".repeat(filled) + "░".repeat(width - filled) + "]";
}

function statusColor(status: string): string {
  switch (status) {
    case "applied":
      return GREEN;
    case "failed":
      return RED;
    case "review":
      return YELLOW;
    case "cancelled":
      return DIM;
    default:
      return CYAN;
  }
}

async function cmdEnrichRun(
  client: Client,
  kg: string,
  rl: readline.Interface,
  args: string[],
): Promise<void> {
  if (args.length < 2) {
    stdout.write(
      `  ${YELLOW}Usage:${RESET} /enrich <Type> <attr1> [<attr2> ...]\n`,
    );
    return;
  }
  const typeInput = args[0]!;
  const attrs = args.slice(1).map((a) => a.replace(/^\./, ""));
  const typeName = await resolveTypeName(client, kg, rl, typeInput);
  if (!typeName) return;

  const policy: "stage" = "stage";
  // The backend now picks the source ("auto"): free (Wikidata) vs paid web
  // search, asking us to clarify only when it's genuinely unsure. So we no
  // longer prompt up front or claim a fixed tier here.
  stdout.write(
    `\n  ${BOLD}Plan:${RESET} enrich ${CYAN}${typeName}${RESET}.${attrs
      .map((a) => `${CYAN}${a}${RESET}`)
      .join(`, .`)} in ${BOLD}${kg}${RESET}  ${DIM}·${RESET} ${DIM}auto-routing source…${RESET}\n\n`,
  );

  // Queue at tier "auto" and let the backend route. If it needs us to clarify,
  // re-queue with the explicit tier the user picks.
  let created = await queueEnrich(client, typeName, attrs, kg, policy, "auto");
  if (!created) return;

  if (created.needs_clarification || created.status === "needs_clarification") {
    if (created.routing_note) {
      stdout.write(`  ${DIM}${created.routing_note}${RESET}\n`);
    }
    const candidates = created.candidates ?? ["lite", "core"];
    const offersWeb = candidates.includes("core");
    const offersFree = candidates.includes("lite");
    const prompt = offersWeb && offersFree
      ? `  Source unclear — [w]eb search (paid) or [f]ree (Wikidata)? [w/f/c]: `
      : `  Pick a source — ${candidates.join(" / ")} [c]ancel: `;
    const ans = (await ask(rl, prompt)).trim().toLowerCase();
    let chosen: EnrichmentTier | null = null;
    if (ans === "c" || ans === "cancel") {
      stdout.write(`  ${DIM}Cancelled.${RESET}\n`);
      return;
    } else if (ans === "w" || ans === "web") {
      chosen = "core";
    } else if (ans === "f" || ans === "free") {
      chosen = "lite";
    } else if (candidates.includes(ans)) {
      chosen = ans as EnrichmentTier;
    } else {
      stdout.write(`  ${DIM}Cancelled.${RESET}\n`);
      return;
    }
    created = await queueEnrich(client, typeName, attrs, kg, policy, chosen);
    if (!created) return;
  } else {
    // A job was created — surface the routing decision so the user sees which
    // source ran and why.
    const sourceLabel =
      created.resolved_tier === "lite" ? "Wikidata (free)" : "live web search";
    stdout.write(
      `  ${DIM}Source:${RESET} ${sourceLabel}${created.routing_note ? ` — ${created.routing_note}` : ""}\n`,
    );
  }

  if (!created.job_id) {
    printError("Backend did not return a job id.");
    return;
  }

  const cost = (created.estimated_cost_usd ?? 0).toFixed(4);
  stdout.write(
    `  ${GREEN}✓${RESET} Job queued: ${CYAN_BOLD}${created.job_id}${RESET} ${DIM}·${RESET} estimated cost ${BOLD}$${cost}${RESET} ${DIM}·${RESET} ${fmtNum(created.total_entities ?? 0)} entities\n`,
  );

  const resolvedTier: EnrichmentTier = created.resolved_tier ?? "core";
  const watch = (await ask(rl, `  Watch progress? [Y/n]: `)).trim().toLowerCase();
  if (watch === "" || watch === "y" || watch === "yes") {
    const finished = await watchJob(client, created.job_id);
    await maybeEscalateToWeb(client, rl, typeName, attrs, kg, policy, resolvedTier, finished);
  } else {
    stdout.write(
      `  ${DIM}Tip: /enrich watch ${created.job_id} to follow it.${RESET}\n`,
    );
  }
}

/**
 * Queue one enrichment job, with a spinner and error rendering. Returns the
 * create-response (which may carry a routing decision or a needs_clarification
 * flag) or null when the call failed.
 */
async function queueEnrich(
  client: Client,
  typeName: string,
  attrs: string[],
  kg: string,
  policy: "stage",
  tier: EnrichmentTier,
): Promise<import("./client.js").EnrichJobCreate | null> {
  const sp = startSpinner(`Queueing enrichment for ${typeName}...`);
  try {
    const created = await client.enrichRun({
      type_name: typeName,
      attributes: attrs,
      tier,
      kg_name: kg,
      conflict_policy: policy,
    });
    sp.stop();
    return created;
  } catch (err) {
    sp.stop();
    if (err instanceof CographError) printError(err.message);
    else printError(err instanceof Error ? err.message : String(err));
    return null;
  }
}

// ALL-MISS FALLBACK: if the backend routed to FREE (Wikidata, resolved_tier
// "lite") and that run found nothing — nothing filled/verified/conflicting AND
// at least one miss — offer to escalate to live web search ("core"). When the
// backend already chose "core", web search has run, so we never offer it again.
async function maybeEscalateToWeb(
  client: Client,
  rl: readline.Interface,
  typeName: string,
  attrs: string[],
  kg: string,
  policy: "stage",
  resolvedTier: EnrichmentTier,
  finished: EnrichJob | null,
): Promise<void> {
  if (!finished) return;
  if (finished.status !== "applied" && finished.status !== "review") return;
  if (resolvedTier !== "lite") return;
  const p = finished.progress;
  if (p.filled + p.verified + p.conflicts > 0) return;
  if (p.no_match <= 0) return;

  const ans = (
    await ask(
      rl,
      `  Nothing found in Wikidata. Try live web search? [Y/n]: `,
    )
  )
    .trim()
    .toLowerCase();
  if (ans !== "" && ans !== "y" && ans !== "yes") {
    stdout.write(
      `  ${DIM}Tip: re-run /enrich ${typeName} ${attrs.join(" ")} to try again.${RESET}\n`,
    );
    return;
  }

  const created = await queueEnrich(client, typeName, attrs, kg, policy, "core");
  if (!created || !created.job_id) return;

  const cost = (created.estimated_cost_usd ?? 0).toFixed(4);
  stdout.write(
    `  ${GREEN}✓${RESET} Job queued: ${CYAN_BOLD}${created.job_id}${RESET} ${DIM}·${RESET} estimated cost ${BOLD}$${cost}${RESET} ${DIM}·${RESET} ${fmtNum(created.total_entities ?? 0)} entities\n`,
  );
  await watchJob(client, created.job_id);
}

async function watchJob(
  client: Client,
  jobId: string,
): Promise<EnrichJob | null> {
  const startedAt = Date.now();
  let lastJob: EnrichJob | null = null;
  // Render in place
  const draw = (job: EnrichJob): void => {
    const p = job.progress;
    const bar = progressBar(p.processed, p.total);
    const elapsed = Math.max(1, Math.floor((Date.now() - startedAt) / 1000));
    const rate = p.processed / elapsed;
    let etaStr = "—";
    if (rate > 0 && p.total > p.processed) {
      const remaining = Math.ceil((p.total - p.processed) / rate);
      etaStr =
        remaining < 60
          ? `${remaining}s`
          : remaining < 3600
            ? `${Math.floor(remaining / 60)}m`
            : `${Math.floor(remaining / 3600)}h`;
    }
    const sc = statusColor(job.status);
    stdout.write(
      `\r\x1b[2K  ${sc}${job.status}${RESET} ${bar} ${fmtNum(p.processed)}/${fmtNum(p.total)} ` +
        `${DIM}·${RESET} filled ${GREEN}${fmtNum(p.filled)}${RESET} ` +
        `${DIM}·${RESET} verified ${CYAN}${fmtNum(p.verified)}${RESET} ` +
        `${DIM}·${RESET} conflicts ${YELLOW}${fmtNum(p.conflicts)}${RESET} ` +
        `${DIM}·${RESET} not found ${DIM}${fmtNum(p.no_match)}${RESET} ` +
        `${DIM}·${RESET} ETA ${etaStr}`,
    );
  };

  while (true) {
    let job: EnrichJob;
    try {
      job = await client.enrichJob(jobId);
    } catch (err) {
      stdout.write("\r\x1b[2K");
      if (err instanceof CographError) printError(err.message);
      else printError(err instanceof Error ? err.message : String(err));
      return null;
    }
    lastJob = job;
    draw(job);
    if (job.status !== "running" && job.status !== "queued") break;
    await new Promise((r) => setTimeout(r, 1500));
  }

  // Final newline after the live line.
  stdout.write("\n");
  if (!lastJob) return null;
  const p = lastJob.progress;
  if (lastJob.status === "review") {
    stdout.write(
      `  ${YELLOW}✦${RESET} ${fmtNum(p.conflicts)} conflict${p.conflicts === 1 ? "" : "s"} need review ` +
        `${DIM}·${RESET} filled ${fmtNum(p.filled)}, verified ${fmtNum(p.verified)}, not found ${fmtNum(p.no_match)}. ` +
        `${DIM}Run${RESET} /enrich review ${lastJob.id}${DIM} to walk through them.${RESET}\n`,
    );
  } else if (lastJob.status === "applied") {
    stdout.write(
      `  ${GREEN}✓${RESET} Applied ${DIM}·${RESET} filled ${fmtNum(p.filled)}, verified ${fmtNum(p.verified)}, not found ${fmtNum(p.no_match)}\n`,
    );
  } else if (lastJob.status === "failed") {
    printError(`Job failed: ${lastJob.error ?? "(no error message)"}`);
  } else if (lastJob.status === "cancelled") {
    stdout.write(`  ${DIM}Job cancelled.${RESET}\n`);
  }
  return lastJob;
}

async function cmdEnrichJobs(client: Client): Promise<void> {
  const sp = startSpinner("Loading enrichment jobs...");
  let jobs: JobSummary[];
  try {
    jobs = await client.enrichJobs();
  } catch (err) {
    sp.stop();
    if (err instanceof CographError) printError(err.message);
    else printError(err instanceof Error ? err.message : String(err));
    return;
  }
  sp.stop();

  if (jobs.length === 0) {
    stdout.write(`  ${DIM}No enrichment jobs yet.${RESET}\n`);
    return;
  }

  const truncAttrs = (attrs: string[]): string => {
    const max = 30;
    const joined = attrs.join(", ");
    if (joined.length <= max) return joined;
    return joined.slice(0, max - 1) + "…";
  };

  const rows = jobs.map((j) => ({
    id: j.id,
    type: j.type_name,
    attrs: truncAttrs(j.attributes ?? []),
    status: j.status,
    progress: `${fmtNum(j.progress?.processed ?? 0)}/${fmtNum(j.progress?.total ?? 0)}`,
    created: relativeTime(j.created_at),
  }));

  const w = {
    id: Math.max("ID".length, ...rows.map((r) => r.id.length)),
    type: Math.max("Type".length, ...rows.map((r) => r.type.length)),
    attrs: Math.max("Attrs".length, ...rows.map((r) => r.attrs.length)),
    status: Math.max("Status".length, ...rows.map((r) => r.status.length)),
    progress: Math.max("Progress".length, ...rows.map((r) => r.progress.length)),
  };

  stdout.write("\n");
  stdout.write(
    `  ${BOLD}${"ID".padEnd(w.id)}  ${"Type".padEnd(w.type)}  ${"Attrs".padEnd(w.attrs)}  ${"Status".padEnd(w.status)}  ${"Progress".padEnd(w.progress)}  Created${RESET}\n`,
  );
  for (const r of rows) {
    const sc = statusColor(r.status);
    stdout.write(
      `  ${CYAN}${r.id.padEnd(w.id)}${RESET}  ${r.type.padEnd(w.type)}  ${DIM}${r.attrs.padEnd(w.attrs)}${RESET}  ${sc}${r.status.padEnd(w.status)}${RESET}  ${r.progress.padEnd(w.progress)}  ${DIM}${r.created}${RESET}\n`,
    );
  }
  stdout.write("\n");
}

async function cmdEnrichReview(
  client: Client,
  rl: readline.Interface,
  jobId: string,
): Promise<void> {
  if (!jobId) {
    stdout.write(`  ${YELLOW}Usage:${RESET} /enrich review <job_id>\n`);
    return;
  }
  const sp = startSpinner(`Loading conflicts for ${jobId}...`);
  let conflicts: ConflictReview[];
  try {
    conflicts = await client.enrichConflicts(jobId);
  } catch (err) {
    sp.stop();
    if (err instanceof CographError) printError(err.message);
    else printError(err instanceof Error ? err.message : String(err));
    return;
  }
  sp.stop();

  if (conflicts.length === 0) {
    stdout.write(`  ${DIM}No conflicts to review.${RESET}\n`);
    return;
  }

  const decisions: ConflictReview[] = [];
  let acceptAll = false;
  let quitEarly = false;

  for (let i = 0; i < conflicts.length; i++) {
    const c = conflicts[i]!;
    const entity = lastUriSegment(c.entity_uri);
    const conf = (c.proposed?.confidence ?? 0).toFixed(2);
    stdout.write("\n");
    stdout.write(
      `  ${DIM}[${i + 1}/${conflicts.length}]${RESET} ${BOLD}${entity}${RESET}.${CYAN}${c.attribute}${RESET}\n`,
    );
    stdout.write(
      `    ${DIM}existing →${RESET} ${c.existing_value}\n` +
        `    ${DIM}proposed →${RESET} ${BOLD}${c.proposed?.value ?? ""}${RESET} ${DIM}(conf ${conf}, src ${c.proposed?.source ?? "?"})${RESET}\n`,
    );
    if (c.proposed?.source_url) {
      stdout.write(`    ${DIM}url      →${RESET} ${c.proposed.source_url}\n`);
    }

    let decision: "accept" | "reject" | "skip";
    if (acceptAll) {
      decision = "accept";
      stdout.write(`    ${GREEN}auto-accepted${RESET}\n`);
    } else {
      const ans = (
        await ask(
          rl,
          `    [a]ccept / [r]eject / [s]kip / [A]ccept all remaining / [q]uit (saves progress) [s]: `,
        )
      ).trim();
      if (ans === "A") {
        acceptAll = true;
        decision = "accept";
      } else if (ans === "a") {
        decision = "accept";
      } else if (ans === "r") {
        decision = "reject";
      } else if (ans === "q") {
        quitEarly = true;
        break;
      } else {
        decision = "skip";
      }
    }
    decisions.push({ ...c, decision });
  }

  if (quitEarly) {
    if (decisions.length === 0) {
      stdout.write(`  ${DIM}No decisions made — nothing to save.${RESET}\n`);
      return;
    }
    const save = (
      await ask(rl, `  Save ${decisions.length} decision(s) so far? [Y/n]: `)
    )
      .trim()
      .toLowerCase();
    if (save !== "" && save !== "y" && save !== "yes") {
      stdout.write(`  ${DIM}Discarded.${RESET}\n`);
      return;
    }
  }

  if (decisions.length === 0) {
    stdout.write(`  ${DIM}No decisions to apply.${RESET}\n`);
    return;
  }

  const sp2 = startSpinner(`Applying ${decisions.length} decision(s)...`);
  try {
    const res = await client.enrichApply(jobId, decisions);
    sp2.stop();
    stdout.write(
      `  ${GREEN}✓${RESET} Applied ${BOLD}${fmtNum(res.applied)}${RESET} change${res.applied === 1 ? "" : "s"}.\n`,
    );
  } catch (err) {
    sp2.stop();
    if (err instanceof CographError) printError(err.message);
    else printError(err instanceof Error ? err.message : String(err));
  }
}

function urlHost(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url.replace(/^https?:\/\//, "").replace(/\/+$/, "");
  }
}

function makePrompt(
  kg: string,
  triples: number,
  mode: "cloud" | "self-hosted" = "cloud",
  baseUrl?: string,
): string {
  const kgPart = `${DIM}(${kg})${RESET}`;
  const triplePart = triples > 0 ? `${DIM}[${fmtNum(triples)}]${RESET} ` : "";
  if (mode === "self-hosted" && baseUrl) {
    const host = urlHost(baseUrl);
    return `  ${CYAN_BOLD}cograph${RESET}${DIM}@${host}${RESET} ${kgPart} ${triplePart}${CYAN_BOLD}▸${RESET} `;
  }
  return `  ${CYAN_BOLD}cograph${RESET} ${kgPart} ${triplePart}${CYAN_BOLD}▸${RESET} `;
}

/**
 * Split a command-line style argument string. Supports double-quoted args.
 */
function splitArgs(s: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQ = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i];
    if (inQ) {
      if (c === '"') inQ = false;
      else cur += c;
    } else {
      if (c === '"') inQ = true;
      else if (c === " " || c === "\t") {
        if (cur) {
          out.push(cur);
          cur = "";
        }
      } else cur += c;
    }
  }
  if (cur) out.push(cur);
  return out;
}

export async function runShell(opts: {
  kg?: string;
  local?: boolean;
  noLogin?: boolean;
}): Promise<void> {
  const CLOUD_DEFAULT = "https://api.cograph.cloud";
  // Detection precedence: --local > --no-login > COGRAPH_API_URL pointing
  // anywhere besides the cloud default. When self-hosted we never trigger
  // login and tenant defaults to "default" (open-access backend behavior).
  const envUrl = process.env.COGRAPH_API_URL || process.env.OMNIX_API_URL;
  const envIsSelfHosted = !!envUrl && envUrl !== CLOUD_DEFAULT;
  const selfHostedHint = !!opts.local || !!opts.noLogin || envIsSelfHosted;

  // `let` rather than `const` so /login can swap in a fresh Client after
  // ~/.cograph/config.json is rewritten with the new key.
  let client = opts.local
    ? new Client({ baseUrl: "http://localhost:8000", tenant: "default" })
    : selfHostedHint
      ? new Client({ tenant: "default" })
      : new Client();

  // Probe the backend before deciding whether to trigger login. This lets
  // us distinguish "cloud, needs auth" from "self-hosted, open access" and
  // also surfaces an unreachable server with a clear error rather than a
  // confusing browser-login attempt.
  const health = await client.healthCheck();
  if (!health.ok) {
    printError(
      `Could not reach ${health.url}. Is the server running?`,
    );
    return;
  }

  const selfHosted = selfHostedHint || !health.requiresAuth;
  const mode: "cloud" | "self-hosted" = selfHosted ? "self-hosted" : "cloud";

  // Cloud / auth-required path: behave as before — if no key, log in.
  if (!selfHosted && health.requiresAuth && !client.apiKey) {
    stdout.write(
      `\n  ${DIM}Not signed in — opening your browser to log in...${RESET}\n`,
    );
    const { runLogin } = await import("./login.js");
    await runLogin();
    client = new Client();
    if (!client.apiKey) {
      // runLogin already exits the process on hard failures, so reaching
      // here means it returned without writing a key (rare). Bail rather
      // than continue into a broken shell.
      printError("Login did not produce an API key. Aborting.");
      return;
    }
  }
  const rl = readline.createInterface({
    input: stdin,
    output: stdout,
    terminal: true,
  });

  showBanner();

  if (selfHosted) {
    stdout.write(
      `${DIM}  Self-hosted mode · ${client.baseUrl} · tenant=${client.tenant}${RESET}\n\n`,
    );
  }

  // One agent session id per shell session — threaded across every /agent turn
  // for multi-turn continuity (the server keys conversation state on it).
  const agentSessionId = randomUUID();

  let kg = opts.kg;
  if (!kg) {
    const picked = await selectKg(client, rl);
    if (!picked) {
      rl.close();
      return;
    }
    kg = picked;
  }

  let triples = 0;
  const info = await fetchKg(client, kg);
  if (info && info.triple_count > 0) {
    triples = info.triple_count;
    stdout.write(
      `  ${DIM}Connected to${RESET} ${BOLD}${kg}${RESET}${DIM}: ${fmtNum(triples)} triples${RESET}\n\n`,
    );
  } else {
    stdout.write(
      `  ${DIM}Connected — ${kg} is empty (use /ingest to add data)${RESET}\n\n`,
    );
  }

  const refresh = async (): Promise<void> => {
    const fresh = await fetchKg(client, kg!);
    triples = fresh?.triple_count ?? 0;
  };

  let running = true;
  rl.on("close", () => {
    running = false;
  });

  while (running) {
    let line: string;
    try {
      line = (
        await ask(rl, makePrompt(kg, triples, mode, client.baseUrl))
      ).trim();
    } catch {
      break;
    }
    if (!running) break;
    if (!line) continue;

    if (line === "/quit" || line === "/exit" || line === "/q") {
      stdout.write(`  ${DIM}Bye.${RESET}\n`);
      break;
    }

    if (line === "/help") {
      showCommands();
      continue;
    }

    try {
      if (line.startsWith("/ingest")) {
        const args = splitArgs(line.slice("/ingest".length).trim());
        await cmdIngest(client, kg, args);
        await refresh();
      } else if (line.startsWith("/ask ")) {
        await cmdAsk(client, kg, line.slice("/ask ".length));
      } else if (line === "/ask") {
        await cmdAsk(client, kg, "");
      } else if (line.startsWith("/agent ")) {
        await cmdAgent(
          client,
          kg,
          rl,
          agentSessionId,
          line.slice("/agent ".length),
        );
        await refresh();
      } else if (line === "/agent") {
        await cmdAgent(client, kg, rl, agentSessionId, "");
      } else if (line === "/types" || line.startsWith("/types ")) {
        const query = line === "/types" ? "" : line.slice("/types ".length);
        await cmdTypes(client, kg, query);
      } else if (line.startsWith("/type ") || line === "/type") {
        const arg = line === "/type" ? "" : line.slice("/type ".length);
        await cmdType(client, kg, rl, arg);
      } else if (line === "/enrich" || line.startsWith("/enrich ")) {
        const args = splitArgs(line.slice("/enrich".length).trim());
        if (args.length === 0) {
          stdout.write(
            `  ${YELLOW}Usage:${RESET} /enrich <Type> <attr> ... | /enrich watch <id> | /enrich jobs | /enrich review <id>\n`,
          );
        } else if (args[0] === "jobs") {
          await cmdEnrichJobs(client);
        } else if (args[0] === "watch") {
          const jid = args[1];
          if (!jid) {
            stdout.write(`  ${YELLOW}Usage:${RESET} /enrich watch <job_id>\n`);
          } else {
            await watchJob(client, jid);
          }
        } else if (args[0] === "review") {
          await cmdEnrichReview(client, rl, args[1] ?? "");
        } else {
          await cmdEnrichRun(client, kg, rl, args);
          await refresh();
        }
      } else if (line === "/status") {
        await cmdStatus(client, kg);
        await refresh();
      } else if (line === "/reset") {
        const did = await cmdReset(client, kg, rl);
        if (did) await refresh();
      } else if (line === "/login") {
        const { runLogin } = await import("./login.js");
        await runLogin();
        // Pick up the new key from ~/.cograph/config.json for subsequent calls.
        client = new Client();
        await refresh();
      } else if (line === "/tenant" || line.startsWith("/tenant ")) {
        const args = splitArgs(line.slice("/tenant".length).trim());
        const sub = args[0] ?? "list";
        const target = args.slice(1).join(" ");

        if (sub === "use" || sub === "switch") {
          if (!target) {
            stdout.write(`  ${YELLOW}Usage:${RESET} /tenant use <id>\n`);
          } else {
            writeConfig({ tenant: target });
            // Rebuild the client so it picks up the new tenant; preserve the
            // current base URL (self-hosted/local) and key.
            client = new Client({ baseUrl: client.baseUrl });
            stdout.write(
              `  ${GREEN}✓${RESET} Switched to tenant ${BOLD}${target}${RESET}\n`,
            );
            // KGs are per-tenant — the old current KG may not exist here, so
            // pick one from the new tenant.
            const picked = await selectKg(client, rl);
            if (picked) {
              kg = picked;
            } else {
              stdout.write(
                `  ${DIM}No KGs in ${target} yet — /kg create <name>${RESET}\n`,
              );
            }
            await refresh();
          }
        } else if (sub === "current") {
          stdout.write(`  ${BOLD}${client.tenant}${RESET}\n`);
        } else {
          try {
            const tenants = await client.listTenants();
            if (!tenants.length) {
              stdout.write(`  ${DIM}No tenants found for your account.${RESET}\n`);
            } else {
              for (const t of tenants) {
                const marker =
                  t.id === client.tenant ? `${CYAN_BOLD}*${RESET}` : " ";
                stdout.write(
                  `  ${marker} ${BOLD}${t.id}${RESET} ${DIM}${t.label}${RESET}\n`,
                );
              }
              stdout.write(`  ${DIM}/tenant use <id> to switch${RESET}\n`);
            }
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            if (err instanceof CographError && err.status === 501) {
              printError("Tenant management isn't configured on this backend.");
            } else {
              printError(msg);
            }
          }
        }
      } else if (line === "/kg" || line.startsWith("/kg ")) {
        const args = splitArgs(line.slice("/kg".length).trim());
        const sub = args[0] ?? "list";
        const target = args.slice(1).join(" ");

        if (sub === "list") {
          const list = await client.listKgs();
          if (!list.length) {
            stdout.write(
              `  ${DIM}No knowledge graphs yet. /kg create <name>${RESET}\n`,
            );
          } else {
            for (const k of list) {
              const n = String((k as { name?: string }).name ?? "?");
              const tc = Number((k as { triple_count?: number }).triple_count ?? 0);
              const marker = n === kg ? `${CYAN_BOLD}*${RESET}` : " ";
              stdout.write(
                `  ${marker} ${BOLD}${n}${RESET} ${DIM}(${fmtNum(tc)} triples)${RESET}\n`,
              );
            }
          }
        } else if (sub === "switch") {
          if (!target) {
            stdout.write(`  ${YELLOW}Usage:${RESET} /kg switch <name>\n`);
          } else {
            const list = await client.listKgs();
            const found = list.find(
              (k) => (k as { name?: string }).name === target,
            );
            if (!found) {
              printError(`KG not found: ${target}. Try /kg list.`);
            } else {
              kg = target;
              triples = Number(
                (found as { triple_count?: number }).triple_count ?? 0,
              );
              stdout.write(
                `  ${GREEN}✓${RESET} Switched to ${BOLD}${kg}${RESET}\n`,
              );
            }
          }
        } else if (sub === "create") {
          if (!target) {
            stdout.write(`  ${YELLOW}Usage:${RESET} /kg create <name>\n`);
          } else {
            try {
              await client.createKg(target);
              kg = target;
              triples = 0;
              stdout.write(
                `  ${GREEN}✓${RESET} Created and switched to ${BOLD}${kg}${RESET}\n`,
              );
            } catch (err) {
              const msg = err instanceof Error ? err.message : String(err);
              if (/already exists|409/i.test(msg)) {
                kg = target;
                await refresh();
                stdout.write(
                  `  ${DIM}${target} already exists — switched to it.${RESET}\n`,
                );
              } else {
                printError(`Could not create: ${msg}`);
              }
            }
          }
        } else if (sub === "delete") {
          if (!target) {
            stdout.write(`  ${YELLOW}Usage:${RESET} /kg delete <name>\n`);
          } else {
            const isActive = target === kg;
            const tag = isActive ? " (the active KG)" : "";
            const confirm = (
              await ask(
                rl,
                `  ${YELLOW}Delete KG "${target}"${tag}?${RESET} [y/N]: `,
              )
            )
              .trim()
              .toLowerCase();
            if (confirm === "y" || confirm === "yes") {
              try {
                await client.deleteKg(target);
                stdout.write(`  ${GREEN}✓${RESET} Deleted ${BOLD}${target}${RESET}\n`);
                if (isActive) {
                  // Active KG is gone; let the user pick (or create) a new one
                  // before any further commands try to use it.
                  const picked = await selectKg(client, rl);
                  if (!picked) {
                    running = false;
                    break;
                  }
                  kg = picked;
                  await refresh();
                }
              } catch (err) {
                const msg = err instanceof Error ? err.message : String(err);
                printError(`Could not delete: ${msg}`);
              }
            } else {
              stdout.write(`  ${DIM}Cancelled.${RESET}\n`);
            }
          }
        } else {
          stdout.write(
            `  ${YELLOW}Unknown /kg subcommand: ${sub}.${RESET} Try /kg list, /kg switch <name>, /kg create <name>, /kg delete <name>.\n`,
          );
        }
      } else if (line.startsWith("/")) {
        stdout.write(
          `  ${YELLOW}Unknown command.${RESET} Try /ingest, /ask, /agent, /kg, /types, /type, /enrich, /login, /status, /reset, /help, /quit\n`,
        );
      } else {
        // Bare line — auto-route to /ask
        await cmdAsk(client, kg, line);
      }
    } catch (err) {
      if (err instanceof CographError) printError(err.message);
      else printError(err instanceof Error ? err.message : String(err));
    }
  }

  rl.close();
}
