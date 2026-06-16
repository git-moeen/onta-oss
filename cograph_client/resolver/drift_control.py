"""ADR 0004 — ontology drift-control decision logic (pure, side-effect-free).

This module is the *foundation* of ADR 0004: the support-floor gate, quarantine
split, and drift report that decide whether a tenant-layer type-level
relationship/attribute is a real schema declaration or low-support drift (the
literal ``ManufacturerPartNumber.issuedby -> Retailer`` shape that production
surfaced on 2026-06-15).

It is intentionally:

  - **Pure** — no Neptune, no LLM, no I/O. Just arithmetic over counts. The
    calibration in ``scripts/adr4_drift_threshold_experiment.py`` validated the
    exact rule implemented here.
  - **Stdlib-only** and importable with **zero dependency on the rest of the
    package** — so it can be unit-tested and reused without dragging in the
    resolver/graph stack.
  - **Flag-gated**: every NEW behavior in ADR 0004 sits behind
    ``OMNIX_DRIFT_CONTROL``. With the flag OFF (default), nothing in the ingest
    path calls into this module, so runtime behavior is byte-identical to today.
    The functions here are still importable and testable regardless of the flag
    — the flag governs whether *callers* wire them in, not whether the math runs.

Rule (ADR 0004 §1, calibrated at floor=20% + support>=5 + core-slot exemption):

  A non-core type-level relationship/attribute is **DECLARED** iff
      ``coverage >= FLOOR_COV`` AND ``support >= FLOOR_COUNT``.
  A **core slot** (ADR 0003 §3) is **EXEMPT** — always declared, even at 0
  support (it is an enrichment target).
  Anything below the floor is **QUARANTINED** (held for review), never silently
  deleted — the gray zone (10–30%) holds real sparse edges, so the floor must be
  recall-safe (ADR 0004 §2).

Env (read at *call time* via the helpers below, so tests can monkeypatch
``os.environ`` without re-importing):

  - ``OMNIX_DRIFT_CONTROL``   -> ``DRIFT_CONTROL_ENABLED`` ("1" => True, else False)
  - ``OMNIX_DRIFT_FLOOR_COV`` -> ``FLOOR_COV`` (float percent, default 20.0)
  - ``OMNIX_DRIFT_FLOOR_COUNT`` -> ``FLOOR_COUNT`` (int, default 5)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# --- Defaults (validated in scripts/adr4_drift_threshold_experiment.py) -------
DEFAULT_FLOOR_COV: float = 20.0
DEFAULT_FLOOR_COUNT: int = 5

# Env var names (single source of truth so callers/tests don't hardcode strings).
ENV_ENABLED = "OMNIX_DRIFT_CONTROL"
ENV_FLOOR_COV = "OMNIX_DRIFT_FLOOR_COV"
ENV_FLOOR_COUNT = "OMNIX_DRIFT_FLOOR_COUNT"


# --- Env helpers (read at CALL time, never cached at import) ------------------
def drift_control_enabled() -> bool:
    """True iff the ADR 0004 drift-control feature flag is ON.

    Gate for every NEW ADR 0004 behavior. ``OMNIX_DRIFT_CONTROL=="1"`` enables;
    anything else (unset, "0", "true", ...) is OFF, so the default is a no-op and
    today's behavior is preserved exactly. Read live from ``os.environ`` so a
    test can flip it with ``monkeypatch.setenv`` mid-process.
    """
    return os.environ.get(ENV_ENABLED, "0") == "1"


def env_floor_cov() -> float:
    """Coverage floor (percent) from ``OMNIX_DRIFT_FLOOR_COV``, default 20.0.

    A malformed value falls back to the validated default rather than raising —
    a bad env var must never break ingest.
    """
    raw = os.environ.get(ENV_FLOOR_COV)
    if raw is None or raw == "":
        return DEFAULT_FLOOR_COV
    try:
        return float(raw)
    except (TypeError, ValueError):
        return DEFAULT_FLOOR_COV


def env_floor_count() -> int:
    """Support-count floor from ``OMNIX_DRIFT_FLOOR_COUNT``, default 5.

    Malformed value falls back to the validated default (see ``env_floor_cov``).
    """
    raw = os.environ.get(ENV_FLOOR_COUNT)
    if raw is None or raw == "":
        return DEFAULT_FLOOR_COUNT
    try:
        return int(raw)
    except (TypeError, ValueError):
        return DEFAULT_FLOOR_COUNT


def _resolve_floors(floor_cov: float | None, floor_count: int | None) -> tuple[float, int]:
    """Resolve effective floors: explicit arg wins, else the live env value."""
    cov = env_floor_cov() if floor_cov is None else float(floor_cov)
    count = env_floor_count() if floor_count is None else int(floor_count)
    return cov, count


# --- Core decision logic ------------------------------------------------------
def coverage(support: int, source_count: int) -> float:
    """Percent of the source type's instances that actually carry this slot.

    ``0`` when ``source_count == 0`` (no instances => no coverage, never a
    divide-by-zero). Mirrors ``Decl.coverage`` in the calibration script exactly.
    """
    if not source_count:
        return 0.0
    return support / source_count * 100.0


def should_declare(
    support: int,
    source_count: int,
    is_core_slot: bool,
    floor_cov: float | None = None,
    floor_count: int | None = None,
) -> bool:
    """Decide whether a type-level declaration clears the ADR 0004 floor.

    Core slots are EXEMPT (ADR 0003 §3) — always declared. Otherwise a
    declaration is kept iff ``coverage >= floor_cov`` AND ``support >=
    floor_count``. ``floor_cov``/``floor_count`` default to the live env values
    when ``None``, so callers can override per-type (ADR 0004 §6) without
    touching the env.
    """
    if is_core_slot:
        return True
    cov, count = _resolve_floors(floor_cov, floor_count)
    return coverage(support, source_count) >= cov and support >= count


@dataclass
class ReconcileResult:
    """Split of declarations into kept (declared) vs quarantined (held).

    Each list holds the ORIGINAL declaration dicts, unmodified — this module
    never mutates its input (pure, side-effect-free). A caller writes the kept
    set to the ontology and routes the quarantine set to the review store.
    """

    keep: list[dict] = field(default_factory=list)
    quarantine: list[dict] = field(default_factory=list)


def _is_core(decl: dict) -> bool:
    """Read the optional ``is_core_slot`` flag, defaulting to False."""
    return bool(decl.get("is_core_slot", False))


def reconcile(
    declarations: list[dict],
    floor_cov: float | None = None,
    floor_count: int | None = None,
) -> ReconcileResult:
    """Partition declarations into kept vs quarantined per the ADR 0004 floor.

    ``declarations`` is a list of dicts with keys ``key`` (str), ``support``
    (int), ``source_count`` (int), and optional ``is_core_slot`` (bool, default
    False). Below-floor declarations go to ``.quarantine`` (held for review),
    never dropped — the recall-safe trade in ADR 0004 §2. Pure: the input list
    and its dicts are not mutated.
    """
    result = ReconcileResult()
    for decl in declarations:
        if should_declare(
            int(decl["support"]),
            int(decl["source_count"]),
            _is_core(decl),
            floor_cov=floor_cov,
            floor_count=floor_count,
        ):
            result.keep.append(decl)
        else:
            result.quarantine.append(decl)
    return result


def drift_report(
    declarations: list[dict],
    floor_cov: float | None = None,
    floor_count: int | None = None,
) -> dict:
    """Summarize a reconciliation pass into the ADR 0004 §5 drift report.

    Returns a dict with the effective floors, kept/quarantined counts, and the
    quarantine list (each as ``{key, coverage, support}``) — the payload the
    drift dashboard and tenant changelog read from. Coverage is rounded to a
    stable 2 decimals so the report is deterministic and snapshot-friendly.
    """
    cov, count = _resolve_floors(floor_cov, floor_count)
    split = reconcile(declarations, floor_cov=cov, floor_count=count)
    quarantine = [
        {
            "key": d["key"],
            "coverage": round(coverage(int(d["support"]), int(d["source_count"])), 2),
            "support": int(d["support"]),
        }
        for d in split.quarantine
    ]
    return {
        "floor_cov": cov,
        "floor_count": count,
        "kept": len(split.keep),
        "quarantined": len(split.quarantine),
        "quarantine": quarantine,
    }
