"""Next-run computation for a :class:`Schedule` (COG-135).

Interval is the v1 path and has no extra dependency. Cron is best-effort: it
uses ``croniter`` IF it is importable, but ``croniter`` is NOT a hard
dependency of the OSS package — the import is guarded and a clear
``NotImplementedError`` is raised when a cron schedule is computed without it.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from cograph_client.scheduling.models import Schedule


def compute_next_run(schedule: Schedule, after: datetime) -> datetime:
    """Return the next firing time of ``schedule`` strictly relative to ``after``.

    - ``interval_seconds`` set → ``after + interval`` (the v1 path; no extra
      dependency).
    - ``cron`` set → the next cron occurrence after ``after``, computed with
      ``croniter`` if it is importable. ``croniter`` is intentionally optional
      (not a hard OSS dependency); if it cannot be imported a
      ``NotImplementedError`` is raised with a clear message so the caller can
      surface "cron schedules need the optional croniter extra".

    The model guarantees exactly one of cron / interval_seconds is set, so the
    final ``ValueError`` is defensive only.
    """
    if schedule.interval_seconds is not None:
        return after + timedelta(seconds=schedule.interval_seconds)

    if schedule.cron is not None and schedule.cron.strip():
        try:
            from croniter import croniter  # optional; guarded on purpose
        except ImportError as exc:  # pragma: no cover - depends on env
            raise NotImplementedError(
                "cron schedules require the optional 'croniter' package; "
                "install it (pip install croniter) or use interval_seconds"
            ) from exc
        return croniter(schedule.cron, after).get_next(datetime)

    # Unreachable given the model validator, but fail loudly if it ever is.
    raise ValueError(
        "schedule has neither interval_seconds nor cron; cannot compute next_run"
    )
