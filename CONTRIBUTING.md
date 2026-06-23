# Contributing to Cograph

## What can ship here (OSS boundary — read first)

`cograph-oss` is published publicly to npm (and PyPI). **Public publication is a
one-way door** — once code ships, it's in mirrors, archives, and forks within
hours. Everything in this repo must be OSS-safe.

**Ships here (OSS):**
- `cograph_client/` — ingest, resolver, **core ER engine** (normalize, block,
  score, merge), REST API surface, embedding service
- `packages/cograph` (TS SDK + CLI) and `packages/cograph-mcp` (MCP server)
- Plugin **protocols**: `register_external_verifier` (auth),
  `register_adapter` (enrichment)
- Default OSS adapters: Wikidata enrichment, static-keys auth
- Tests for all of the above

**Does NOT ship here (proprietary — lives in the parent `cograph/` repo):**
- Paid enrichment adapters (Exa, Perplexity, GS1, Anthropic web_search)
- Production Clerk auth integration (`cograph-auth-clerk`)
- Cograph Explorer web app, AWS/SAM infra, deploy workflows
- Entitlement / billing / rate-limit logic
- Advanced ER tooling (review-queue UI, embedding matchers, active learning)

The canonical, fuller table with reasoning lives in the parent repo at
[`docs/oss_proprietary_boundary.md`](https://github.com/git-moeen/cograph/blob/main/docs/oss_proprietary_boundary.md).
When in doubt, surface the question before writing code.

**Entitlement gating is NOT done in OSS (incl. the MCP server).** The MCP server
and its `agent` tool are OSS and are advertised freely — planning a turn is free.
A plan the agent executes may contain a *paid* step (e.g. web enrichment), but the
authorization for that step is enforced **server-side, behind the HTTP API**, by
the proprietary backend (a 4xx on `POST /graphs/{tenant}/agent` confirm, the same
way the direct paid routes are gated). The MCP `agent` tool reaches the backend
through the exact same authenticated HTTP client (`X-API-Key` → tenant) as every
other tool, so confirming a plan via the agent **cannot bypass** a gate the direct
path enforces — there is deliberately no entitlement check to duplicate in OSS
(per the proprietary list above). Do **not** add billing/entitlement logic here to
"gate" the agent: that belongs in the parent repo.

**This is mechanically enforced** (MOE-21). Run the same checks CI runs:

```bash
bash scripts/check_boundary.sh      # static: no proprietary imports/hosts/paths/secrets
bash scripts/check_npm_bundle.sh    # inspect published tarballs for forbidden paths
```

CI runs `check_boundary.sh` on every PR (`.github/workflows/boundary.yml`), and
the publish workflow runs `check_npm_bundle.sh` before any `npm publish`. A PR
that adds `from cograph.<anything>` under `cograph_client/` or `packages/` fails.

## Dev Setup

```bash
# Clone
git clone https://github.com/git-moeen/cograph-oss.git
cd cograph-oss

# Start graph DB
docker compose up -d

# Install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env

# Run tests
pytest tests/ -v --tb=short
```

## Running Locally

```bash
source .env
uvicorn cograph_client.api.app:create_app --factory --port 8000
```

## Project Structure

```
cograph_client/
  api/          FastAPI routes and middleware
  auth/         API key authentication
  graph/        SPARQL client and query builders
  nlp/          Query pipeline, prompts, example bank, embeddings
  resolver/     Schema inference, type matching, CSV mapping
  models/       Pydantic data models
  functions/    Compute function registry
  cli.py        CLI entry point
  config.py     Settings (OMNIX_ env prefix)
  eval.py       Eval framework
  mcp_server.py MCP server for AI agents
```

## Code Style

- Python 3.12+
- Type hints on all function signatures
- snake_case for functions and variables, PascalCase for classes
- No print statements in library code, use structlog
- Keep functions short. If it needs a comment explaining what it does, it's too long.

## Making Changes

1. Fork the repo
2. Create a branch: `git checkout -b my-change`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Commit with a clear message: `git commit -m "fix: description of what and why"`
6. Open a PR against `main`

## Commit Messages

Format: `type: description`

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `perf`

Examples:
- `feat: add Blazegraph backend support`
- `fix: handle empty CSV columns in schema inference`
- `docs: add Ollama configuration guide`

## Tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_validator.py -v

# Run with coverage
pytest tests/ --cov=cograph_client --cov-report=term-missing
```

Tests mock the Neptune/Fuseki client. No running graph DB needed for unit tests.

## Areas We'd Love Help With

- Additional graph DB backends (Blazegraph, Oxigraph, RDFLib)
- More LLM provider support (Ollama, vLLM, Together)
- Better eval question generation (more natural language, less attribute-name references)
- Entity resolution ("TX" = "Texas")
- Documentation improvements
