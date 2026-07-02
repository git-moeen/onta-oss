import { describe, expect, it } from "vitest";
import { USER_SCHEDULABLE_ACTIONS } from "../src/index.js";
import type { Schedule, ScheduleAction, UserSchedulableAction } from "../src/index.js";

// ONTA-173 — the schedule action vocabulary is split in two:
//  * UserSchedulableAction — what create/update accept (mirrors the backend's
//    USER_SCHEDULABLE_ACTIONS allowlist in scheduling/models.py);
//  * ScheduleAction — the full READ union, which additionally carries the
//    system-managed semantic maintenance rows the backend creates internally
//    (they appear in a tenant's list/get responses).
// These tests pin the runtime allowlist and prove the read union covers the
// system-managed arms, so exhaustive consumers don't break when a
// `semantic-reconcile` row shows up in a schedules listing.

describe("schedule action vocabulary (ONTA-173)", () => {
  it("user-schedulable allowlist mirrors the backend: exactly the three action-endpoint actions", () => {
    expect([...USER_SCHEDULABLE_ACTIONS].sort()).toEqual([
      "enrich",
      "find-merge-duplicates",
      "suggest-relationships",
    ]);
    // The system-managed semantic actions must never be user-schedulable.
    const asStrings: readonly string[] = USER_SCHEDULABLE_ACTIONS;
    expect(asStrings).not.toContain("semantic-embed-fill");
    expect(asStrings).not.toContain("semantic-reconcile");
  });

  it("Schedule.action covers backend system-managed rows (full read union)", () => {
    // Compile-time: a `semantic-reconcile` row from a list response is a valid
    // Schedule, and an exhaustive Record over ScheduleAction must include the
    // two system-managed arms (missing keys would fail `npm run typecheck` if
    // this shape lived in src — at runtime we assert the lookup resolves).
    const row: Schedule = {
      id: "semantic-reconcile:acme:kg",
      tenant_id: "acme",
      kg_name: "kg",
      category: "reconciliation",
      action: "semantic-reconcile",
      params: {},
      interval_seconds: 900,
      enabled: true,
      created_at: "2026-01-01T00:00:00Z",
    };
    const handled: Record<ScheduleAction, "user" | "system"> = {
      "find-merge-duplicates": "user",
      enrich: "user",
      "suggest-relationships": "user",
      "semantic-embed-fill": "system",
      "semantic-reconcile": "system",
    };
    expect(handled[row.action]).toBe("system");
    // Every user-schedulable action is (by construction) a valid ScheduleAction.
    for (const action of USER_SCHEDULABLE_ACTIONS) {
      const widened: ScheduleAction = action satisfies UserSchedulableAction;
      expect(handled[widened]).toBe("user");
    }
  });
});
