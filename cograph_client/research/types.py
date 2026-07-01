"""Typed values passed between the research-harness stages (ADR 0006).

These dataclasses are the CONTRACT the stages agree on: the planner emits a
:class:`ResearchPlan` (a :class:`TargetSchema` + discovery queries); discovery +
the fetch ladder produce :class:`FetchedPage`\\ s; extraction turns pages into
:class:`ResearchRow`\\ s; verification filters them into a
:class:`VerifyOutcome`; synthesis composes a :class:`ResearchResult` with
:class:`Citation`\\ s. A :class:`Budget` threads through the whole run and bounds
it (tool calls + iterations + wall-clock).

Everything is a plain dataclass with ``to_dict`` where a serialized form is
needed (the capability returns these over the agent JSON contract). No LLM / no
network here — pure data.
"""

from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass, field
from typing import Optional

# Recognized field types for a target-schema column. The extractor returns bare
# string values; these are advisory hints for the extraction prompt + downstream
# formatting, never a hard validation gate (a research answer is best-effort).
FIELD_TYPES = ("string", "number", "boolean", "date", "url")

# Characters that make a spreadsheet treat a cell as a formula. Research rows come
# from untrusted web pages / LLM extraction, so a value like ``=cmd|'…'!A1`` would
# execute when the exported CSV is opened in Excel/Sheets. Prefix such cells.
_CSV_FORMULA_LEADERS = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(value: object) -> str:
    """Neutralize CSV formula injection: prefix a leading formula-trigger char
    with an apostrophe so a spreadsheet treats the cell as text, not a formula."""
    s = "" if value is None else str(value)
    if s and s[0] in _CSV_FORMULA_LEADERS:
        return "'" + s
    return s


@dataclass
class SchemaField:
    """One column of the caller-defined / planner-derived output schema."""

    name: str
    description: str = ""
    type: str = "string"
    required: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SchemaField":
        t = str(d.get("type", "string") or "string").lower()
        if t not in FIELD_TYPES:
            t = "string"
        return cls(
            name=str(d.get("name", "")).strip(),
            description=str(d.get("description", "") or ""),
            type=t,
            required=bool(d.get("required", False)),
        )


@dataclass
class TargetSchema:
    """The output shape the harness extracts into.

    ``entity`` names what each row represents ("TTS model", "company"); ``fields``
    are its columns. Schema-first is the highest-leverage stage: fixing the shape
    up front makes extraction and verification tractable (ADR 0006 §Decision).
    """

    entity: str = "item"
    fields: list[SchemaField] = field(default_factory=list)

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields if f.name]

    def required_names(self) -> list[str]:
        return [f.name for f in self.fields if f.name and f.required]

    def is_empty(self) -> bool:
        return not self.field_names()

    def to_dict(self) -> dict:
        return {"entity": self.entity, "fields": [f.to_dict() for f in self.fields]}

    @classmethod
    def from_dict(cls, d: dict) -> "TargetSchema":
        raw = d.get("fields") or []
        fields_out: list[SchemaField] = []
        seen: set[str] = set()
        for item in raw:
            if isinstance(item, str):
                item = {"name": item}
            if not isinstance(item, dict):
                continue
            sf = SchemaField.from_dict(item)
            key = sf.name.lower()
            if not sf.name or key in seen:
                continue
            seen.add(key)
            fields_out.append(sf)
        return cls(entity=str(d.get("entity", "item") or "item"), fields=fields_out)


@dataclass
class Citation:
    """A source consulted, surfaced to the user as a clickable citation."""

    url: str
    title: str = ""
    snippet: str = ""

    def to_dict(self) -> dict:
        return {"url": self.url, "title": self.title, "snippet": self.snippet}


@dataclass
class ResearchRow:
    """One extracted record: field values plus the URLs that support it.

    ``citations`` is the cite-or-abstain substrate — a row with no supporting URL
    is dropped by the default verifier rather than presented as an unsourced
    claim (ADR 0006 §Verify).
    """

    values: dict[str, str] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return {
            "values": dict(self.values),
            "citations": list(self.citations),
            "confidence": self.confidence,
        }


@dataclass
class FetchedPage:
    """The result of fetching one URL through some rung of the fetch ladder.

    ``tier`` records which rung produced it (``static`` / ``render`` /
    ``structured``) for observability + escalation decisions. ``ok=False`` with an
    ``error`` means the fetch failed (timeout, non-200, blocked) as distinct from
    fetching successfully but finding little text.
    """

    url: str
    text: str = ""
    title: str = ""
    tier: str = ""
    ok: bool = True
    error: Optional[str] = None
    truncated: bool = False

    def has_content(self) -> bool:
        return self.ok and bool(self.text and self.text.strip())

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "tier": self.tier,
            "ok": self.ok,
            "error": self.error,
            "truncated": self.truncated,
            "chars": len(self.text or ""),
        }


@dataclass
class ResearchPlan:
    """The planner's output: the target schema + how to go get it.

    ``needs_web`` is False when the question can be answered without the web (the
    harness then abstains rather than fabricating). ``fast_path`` marks a trivial
    question that a single cited answer covers — the harness may skip the full
    ladder. ``queries`` are discovery queries; ``seed_urls`` are pages the planner
    already knows are authoritative (or the user supplied).
    """

    question: str
    schema: TargetSchema = field(default_factory=TargetSchema)
    needs_web: bool = True
    fast_path: bool = False
    queries: list[str] = field(default_factory=list)
    seed_urls: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "schema": self.schema.to_dict(),
            "needs_web": self.needs_web,
            "fast_path": self.fast_path,
            "queries": list(self.queries),
            "seed_urls": list(self.seed_urls),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchPlan":
        return cls(
            question=str(d.get("question", "") or ""),
            schema=TargetSchema.from_dict(d.get("schema") or {}),
            needs_web=bool(d.get("needs_web", True)),
            fast_path=bool(d.get("fast_path", False)),
            queries=[str(q) for q in (d.get("queries") or []) if str(q).strip()],
            seed_urls=[str(u) for u in (d.get("seed_urls") or []) if str(u).strip()],
            rationale=str(d.get("rationale", "") or ""),
        )


@dataclass
class ResearchResult:
    """The final artifact returned to the caller.

    ``answer`` is prose (with inline citations where useful); ``rows`` are the
    structured records behind it (may be empty for a prose-only answer);
    ``citations`` are the distinct sources. ``abstained`` is True when
    cite-or-abstain found nothing supportable — the harness returns an honest "I
    couldn't verify an answer" rather than an unsourced guess.
    """

    question: str
    answer: str = ""
    rows: list[ResearchRow] = field(default_factory=list)
    schema: TargetSchema = field(default_factory=TargetSchema)
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    is_complete: bool = False
    abstained: bool = False
    iterations: int = 0
    sources_consulted: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "rows": [r.to_dict() for r in self.rows],
            "schema": self.schema.to_dict(),
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "is_complete": self.is_complete,
            "abstained": self.abstained,
            "iterations": self.iterations,
            "sources_consulted": list(self.sources_consulted),
            "notes": self.notes,
        }

    def to_csv(self) -> str:
        """Serialize ``rows`` to CSV using the schema's columns (+ a trailing
        ``sources`` column). Returns an empty string when there are no rows."""
        cols = self.schema.field_names()
        if not cols and self.rows:
            # Fall back to the union of keys actually present, order-preserving.
            seen: list[str] = []
            for r in self.rows:
                for k in r.values:
                    if k not in seen:
                        seen.append(k)
            cols = seen
        if not self.rows or not cols:
            return ""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([_csv_safe(c) for c in cols] + ["sources"])
        for r in self.rows:
            writer.writerow(
                [_csv_safe(r.values.get(c, "")) for c in cols]
                + [_csv_safe("; ".join(r.citations))]
            )
        return buf.getvalue()


@dataclass
class Budget:
    """A per-request spend cap: bounds the reflect loop, page fetches, and LLM
    calls, plus a wall-clock ceiling. The harness checks ``can_*`` before each
    metered action and stops cleanly (returns what it has) when a limit is hit —
    a partial cited answer beats an unbounded run (ADR 0006 §cross-cutting
    "budget caps per request").

    The counters are live state mutated by the harness via ``note_*``. Construct a
    fresh Budget per run (do not share).
    """

    max_iterations: int = 2
    max_fetches: int = 6
    max_llm_calls: int = 12
    max_wall_clock_s: float = 90.0

    fetches_used: int = 0
    llm_calls_used: int = 0
    _started_at: Optional[float] = None

    def start(self) -> "Budget":
        if self._started_at is None:
            self._started_at = time.monotonic()
        return self

    def elapsed_s(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.monotonic() - self._started_at

    def timed_out(self) -> bool:
        return self.elapsed_s() >= self.max_wall_clock_s

    def remaining_fetches(self) -> int:
        return max(0, self.max_fetches - self.fetches_used)

    def can_fetch(self) -> bool:
        return self.remaining_fetches() > 0 and not self.timed_out()

    def can_call_llm(self) -> bool:
        return self.llm_calls_used < self.max_llm_calls and not self.timed_out()

    def note_fetch(self, n: int = 1) -> None:
        self.fetches_used += max(0, n)

    def note_llm(self, n: int = 1) -> None:
        self.llm_calls_used += max(0, n)

    def to_dict(self) -> dict:
        return {
            "max_iterations": self.max_iterations,
            "max_fetches": self.max_fetches,
            "max_llm_calls": self.max_llm_calls,
            "max_wall_clock_s": self.max_wall_clock_s,
            "fetches_used": self.fetches_used,
            "llm_calls_used": self.llm_calls_used,
            "elapsed_s": round(self.elapsed_s(), 2),
        }


__all__ = [
    "Budget",
    "Citation",
    "FIELD_TYPES",
    "FetchedPage",
    "ResearchPlan",
    "ResearchResult",
    "ResearchRow",
    "SchemaField",
    "TargetSchema",
]
