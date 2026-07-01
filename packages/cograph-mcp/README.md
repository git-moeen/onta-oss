# cograph-mcp

MCP (Model Context Protocol) server for [Cograph](https://cograph.cloud). Gives AI agents tools to query and ingest data into your knowledge graphs.

## Install / run

No install needed — use `npx`:

```bash
npx -y cograph-mcp
```

## Claude Desktop / Cursor / Claude Code

```json
{
  "mcpServers": {
    "cograph": {
      "command": "npx",
      "args": ["-y", "cograph-mcp"],
      "env": {
        "COGRAPH_API_KEY": "your-key",
        "COGRAPH_API_URL": "https://api.cograph.cloud"
      }
    }
  }
}
```

## Tools exposed

- `agent` — the single conversational front door to the Ask-AI agent. Send a natural-language message; the agent classifies intent and either answers a question, asks a clarifying question, or proposes a multi-step plan (enrich attributes, clean/normalize values, merge duplicates, inspect/extend the ontology). A plan is **not executed** until you confirm it by calling `agent` again with the returned `plan_id` as `confirm_plan_id`. Planning is free; any paid step a plan contains (e.g. web enrichment) is authorized server-side at execute time, so confirming honors your tenant's entitlements.
- `list_knowledge_graphs` — list available KGs and descriptions
- `create_knowledge_graph` — create a new, empty KG (optionally with a description)
- `delete_knowledge_graph` — delete a KG and all of its data (irreversible)
- `ask` — ask a natural language question; returns the answer
- `ingest_csv` — ingest a CSV file by absolute path into a named KG
- `view_ontology` — show types, attributes, relationships across KGs
- `evolve_ontology` — resolve a fuzzy natural-language ontology-evolution ask (no exact names needed); auto-applies high-confidence changes and returns a summary plus any proposals to confirm
- `apply_ontology_change` — confirm and commit a single proposal returned by `evolve_ontology`
- `list_jobs` — list background jobs (enrichment / dedupe / reconciliation); use it to check on async work the `agent` tool kicked off
- `get_job` — full record + progress of a single enrichment job by id

> Enrichment, cleaning/normalization and duplicate-merging are reached **through
> the `agent` tool** — it plans them and, on confirm, runs them as background
> jobs, so any paid step stays authorized server-side at execute time. Use
> `list_jobs` / `get_job` to watch those jobs finish.

## Environment

- `COGRAPH_API_KEY` — required
- `COGRAPH_API_URL` — default `https://api.cograph.cloud`
- `COGRAPH_TENANT` — default `demo-tenant`

Legacy `OMNIX_*` vars are also accepted.

## License

MIT
