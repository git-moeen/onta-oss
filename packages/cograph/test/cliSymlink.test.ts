import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, rmSync, symlinkSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";

// ---------------------------------------------------------------------------
// Regression guard for the symlink'd-bin bug (COG-129).
//
// npm installs the `cograph` bin as a SYMLINK (node_modules/.bin/cograph →
// dist/cli.js). Node then sets import.meta.url to the *realpath* of the entry
// while process.argv[1] keeps the *symlink* path. A naive
// `import.meta.url === pathToFileURL(process.argv[1]).href` guard therefore
// never matches when launched via the symlink, so program.parseAsync() never
// runs and the published CLI silently does nothing for every command.
//
// This test reproduces the real `.bin` layout: it builds dist/cli.js, drops a
// symlink to it in a temp dir, and runs `node <symlink> --version`. Before the
// fix this printed nothing (empty stdout, exit 0); after it must print the
// package version.
// ---------------------------------------------------------------------------

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = join(here, "..");
const cliPath = join(pkgRoot, "dist", "cli.js");

beforeAll(() => {
  // Build the bin once so the test exercises the real published artifact and
  // never depends on stale dist/. tsup is fast (~tens of ms); the generous
  // timeout covers a cold first build.
  if (!existsSync(cliPath)) {
    execFileSync("npm", ["run", "build"], { cwd: pkgRoot, stdio: "inherit" });
  }
  if (!existsSync(cliPath)) {
    throw new Error(`expected built CLI at ${cliPath} after build`);
  }
}, 120_000);

describe("cli — symlinked bin (npm .bin layout)", () => {
  it("runs --version when invoked through a symlink to dist/cli.js", () => {
    const dir = mkdtempSync(join(tmpdir(), "cograph-bin-"));
    const link = join(dir, "cograph");
    try {
      // Mimic node_modules/.bin/cograph -> ../cograph/dist/cli.js
      symlinkSync(cliPath, link);
      const res = spawnSync("node", [link, "--version"], { encoding: "utf-8" });

      expect(res.status).toBe(0);
      // The bug manifested as EMPTY stdout (guard false → parse never runs).
      const out = res.stdout.trim();
      expect(out).not.toBe("");
      expect(out).toMatch(/^\d+\.\d+\.\d+/);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("still runs --version when invoked directly (no symlink)", () => {
    const res = spawnSync("node", [cliPath, "--version"], { encoding: "utf-8" });
    expect(res.status).toBe(0);
    expect(res.stdout.trim()).toMatch(/^\d+\.\d+\.\d+/);
  });

  it("dispatches subcommands through the symlink (agent --help)", () => {
    const dir = mkdtempSync(join(tmpdir(), "cograph-bin-"));
    const link = join(dir, "cograph");
    try {
      symlinkSync(cliPath, link);
      const res = spawnSync("node", [link, "agent", "--help"], {
        encoding: "utf-8",
      });
      expect(res.status).toBe(0);
      expect(res.stdout).toContain("agent");
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
