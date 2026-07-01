"""Synthesis — turn verified rows into the answer artifact (ADR 0006 §Synthesize).

Pure, no LLM / no network. Takes the verifier's surviving rows + the pages that
were read and assembles the :class:`ResearchResult` the caller receives: a short
prose answer, the structured rows, the distinct citations (with page titles when
we have them), a confidence, and the honest ``abstained`` flag. The CSV/JSON
artifact itself is produced by :meth:`ResearchResult.to_csv` / ``to_dict``.
"""

from __future__ import annotations

from cograph_client.research.types import (
    Citation,
    FetchedPage,
    ResearchResult,
    ResearchRow,
    TargetSchema,
    normalize_clarifying_questions,
)
from cograph_client.research.verify import VerifyOutcome


def _distinct_citations(
    rows: list[ResearchRow],
    pages: list[FetchedPage],
    extra_sources: list[str],
) -> list[Citation]:
    """Distinct source URLs, order-preserving, titled from fetched pages where
    known. Row citations come first (they directly support an answer), then any
    additional sources consulted."""
    titles = {p.url: p.title for p in pages if p.url}
    ordered: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for url in r.citations:
            u = str(url).strip()
            if u and u not in seen:
                seen.add(u)
                ordered.append(u)
    for url in extra_sources:
        u = str(url).strip()
        if u and u not in seen:
            seen.add(u)
            ordered.append(u)
    return [Citation(url=u, title=titles.get(u, "")) for u in ordered]


def _prose_answer(
    question: str,
    schema: TargetSchema,
    rows: list[ResearchRow],
    *,
    abstained: bool,
    n_sources: int,
) -> str:
    if abstained or not rows:
        return (
            "I couldn't verify an answer from the sources I was able to reach. "
            "Every candidate result was either unsourced or empty, so rather than "
            "guess I'm returning nothing. Try giving me a specific source URL, or "
            "enable a web-discovery provider for open-web search."
        )
    entity = schema.entity or "result"
    plural = entity if entity.endswith("s") else f"{entity}s"
    lead = f"Found {len(rows)} {plural} across {n_sources} source"
    lead += "s." if n_sources != 1 else "."
    # Preview the first few rows inline so the answer is useful without opening
    # the CSV; the full set is in the artifact.
    cols = schema.field_names()
    preview_bits: list[str] = []
    for r in rows[:5]:
        if cols:
            label = str(r.values.get(cols[0], "")).strip() or "(unnamed)"
            if len(cols) > 1:
                rest = ", ".join(
                    f"{c}={r.values.get(c, '')}" for c in cols[1:4] if r.values.get(c)
                )
                preview_bits.append(f"{label} ({rest})" if rest else label)
            else:
                preview_bits.append(label)
        else:
            preview_bits.append("; ".join(f"{k}={v}" for k, v in r.values.items()))
    preview = "; ".join(b for b in preview_bits if b)
    more = "" if len(rows) <= 5 else f" …and {len(rows) - 5} more."
    tail = f" e.g. {preview}.{more}" if preview else ""
    return lead + tail


def clarification_result(question: str, questions: list) -> ResearchResult:
    """A no-spend result that asks the user to disambiguate rather than guessing.

    Returned by the harness when the planner flags the question as genuinely
    ambiguous (ADR 0006 §Plan — ask only on true ambiguity). Carries no rows and
    is NOT an abstain: ``abstained`` means "searched, found nothing supportable",
    while this means "I haven't searched yet — pick a reading first".

    ``questions`` entries may be bare strings or ``{"question", "options"}``
    shapes (see :func:`normalize_clarifying_questions`); options are rendered
    inline as suggested answers and ride the result structurally for clients
    that show reply chips."""
    qs = normalize_clarifying_questions(questions)
    lines = [
        f"- {q.question}" + (f" ({' / '.join(q.options)})" if q.options else "")
        for q in qs
    ]
    answer = (
        "Before I search the web for this, I need one clarification — the question "
        "has more than one reasonable reading:\n" + "\n".join(lines)
        if qs
        else "Could you clarify what exactly you're looking for?"
    )
    return ResearchResult(
        question=question,
        answer=answer,
        needs_clarification=True,
        clarifying_questions=qs,
        confidence=0.0,
        is_complete=False,
        abstained=False,
        iterations=0,
    )


def synthesize_result(
    question: str,
    schema: TargetSchema,
    outcome: VerifyOutcome,
    pages: list[FetchedPage],
    *,
    iterations: int,
    sources_consulted: list[str],
    complete: bool | None = None,
) -> ResearchResult:
    """Assemble the final :class:`ResearchResult` from a verify outcome."""
    rows = list(outcome.rows)
    citations = _distinct_citations(rows, pages, sources_consulted)
    n_sources = len(citations)
    answer = _prose_answer(
        question, schema, rows, abstained=outcome.abstained, n_sources=n_sources
    )
    if complete is None:
        complete = (
            not outcome.abstained and bool(rows) and outcome.confidence >= 0.5
        )
    return ResearchResult(
        question=question,
        answer=answer,
        rows=rows,
        schema=schema,
        citations=citations,
        confidence=outcome.confidence,
        is_complete=bool(complete),
        abstained=outcome.abstained,
        iterations=iterations,
        sources_consulted=list(dict.fromkeys(sources_consulted)),
        notes=outcome.notes,
    )


__all__ = ["clarification_result", "synthesize_result"]
