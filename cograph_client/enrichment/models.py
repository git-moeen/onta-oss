"""Pydantic models and enums for the auto-enrichment feature."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class EnrichmentTier(str, Enum):
    lite = "lite"
    base = "base"
    core = "core"
    pro = "pro"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    review = "review"
    applied = "applied"
    cancelled = "cancelled"
    failed = "failed"


class JobCategory(str, Enum):
    """The kind of work a job performs.

    The unified Jobs page lists jobs across all three categories. Existing
    enrichment jobs default to ``enrichment`` for backward compatibility.
    """

    dedupe = "dedupe"
    enrichment = "enrichment"
    reconciliation = "reconciliation"


class JobTrigger(str, Enum):
    """How a job was kicked off.

    Today everything is ``manual`` (a user clicked an action). ``scheduled``
    and ``webhook`` are reserved for future automation — populated by callers,
    no scheduling logic lives here yet (TODO).
    """

    manual = "manual"
    scheduled = "scheduled"
    webhook = "webhook"


class ConflictPolicy(str, Enum):
    skip = "skip"
    verify = "verify"
    overwrite = "overwrite"
    stage = "stage"


class EnrichScope(BaseModel):
    """Value filter restricting an enrich job to a subset of a type's entities (COG-112).

    ``predicate`` is an attribute OR relationship **local-name** (e.g.
    ``haslevel``, ``title``) of the enriched ``type_name``. ``value`` is matched
    case-insensitively:

    - For a **literal attribute** the value is matched against the literal's
      string value.
    - For a **relationship to another node** (object property, e.g.
      ``haslevel → Level``) the value is matched against the target node's
      display label/name — so value ``"Manager"`` selects entities related to
      the Level node whose ``rdfs:label`` / name is "Manager". The target IRI's
      local-name is accepted as a fallback.

    The same local-name may be stored on instance triples under either the
    attribute-URI form (``…/types/<Type>/attrs/<name>``) or the relationship
    predicate form (``…/onto/<name>``); the executor matches both, so callers
    never need to know which.
    """

    predicate: str
    value: str


class EnrichRequest(BaseModel):
    type_name: str
    attributes: list[str]
    tier: EnrichmentTier = EnrichmentTier.lite
    kg_name: str
    conflict_policy: ConflictPolicy = ConflictPolicy.stage
    confidence_min: float = 0.85
    limit: Optional[int] = None
    # COG-112 scoped enrichment. Both optional; default None → unchanged
    # whole-type behavior. If BOTH are set, ``entity_uris`` wins (the explicit
    # subset is the lower-level primitive and takes precedence over ``scope``).
    scope: Optional[EnrichScope] = None
    entity_uris: Optional[list[str]] = None


class Verdict(BaseModel):
    """A single enrichment candidate value with provenance (ADR-0005 §5).

    Two distinct confidence signals are intentionally kept separate:

    - ``confidence`` is the CALIBRATED score. It is the only value the
      tier-chain threshold (e.g. ``confidence_min``) compares against. A
      calibrated score is meant to approximate the probability that the
      value is correct.
    - ``raw_confidence`` is an untrusted, relevance-ish signal straight from
      a source (e.g. an Exa neural relevance score). It is NEVER compared to
      a threshold; it exists only for diagnostics/debugging and as input to a
      calibration step that produces ``confidence``.

    All provenance fields are optional with a ``None`` default so legacy
    construction ``Verdict(value=..., confidence=..., source=...)`` keeps
    working unchanged.
    """

    value: str
    confidence: float
    source: str
    source_url: Optional[str] = None
    reasoning: Optional[str] = None
    raw_confidence: Optional[float] = None
    retrieved_at: Optional[datetime] = None
    source_published_at: Optional[datetime] = None
    grounding_score: Optional[float] = None
    extraction_method: Optional[str] = None
    calibration_method: Optional[str] = None


RowAction = Literal["filled", "verified", "conflict", "skipped", "no_match"]


class RowResult(BaseModel):
    entity_uri: str
    attribute: str
    existing_value: Optional[str] = None
    verdict: Optional[Verdict] = None
    action: RowAction


class JobProgress(BaseModel):
    total: int = 0
    processed: int = 0
    filled: int = 0
    verified: int = 0
    conflicts: int = 0
    skipped: int = 0
    cache_hits: int = 0


class EnrichJob(BaseModel):
    id: str
    tenant_id: str
    kg_name: str
    type_name: str
    attributes: list[str]
    tier: EnrichmentTier
    status: JobStatus
    progress: JobProgress = Field(default_factory=JobProgress)
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    conflict_policy: ConflictPolicy
    confidence_min: float = 0.85
    error: Optional[str] = None
    limit: Optional[int] = None
    results: list[RowResult] = Field(default_factory=list)
    # COG-112 scoped enrichment. Both optional / default None so existing
    # enrichment-job construction keeps working unchanged (whole-type behavior).
    # If both are set, ``entity_uris`` wins (see EnrichRequest).
    scope: Optional[EnrichScope] = None
    entity_uris: Optional[list[str]] = None
    # COG-101: unified-jobs fields. All optional with safe defaults so existing
    # enrichment-job construction keeps working unchanged.
    category: JobCategory = JobCategory.enrichment
    trigger: JobTrigger = JobTrigger.manual
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    cost: Optional[float] = None
    cost_note: Optional[str] = None


class JobSummary(BaseModel):
    id: str
    tenant_id: str
    kg_name: str
    type_name: str
    attributes: list[str]
    tier: EnrichmentTier
    status: JobStatus
    progress: JobProgress
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    conflict_policy: ConflictPolicy
    confidence_min: float = 0.85
    error: Optional[str] = None
    # COG-101: unified-jobs fields.
    category: JobCategory = JobCategory.enrichment
    trigger: JobTrigger = JobTrigger.manual
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    cost: Optional[float] = None
    cost_note: Optional[str] = None
    # Derived 0-100 completion percentage from progress.processed/total.
    progress_pct: int = 0


ReviewDecision = Literal["accept", "reject", "skip"]


class ConflictReview(BaseModel):
    entity_uri: str
    attribute: str
    existing_value: str
    proposed: Verdict
    decision: Optional[ReviewDecision] = None


def _progress_pct(progress: JobProgress) -> int:
    """Derive a 0-100 completion percentage from processed/total.

    Returns 0 when total is unknown (0) to avoid division-by-zero; clamps to
    [0, 100] so a stray over-count can never report >100.
    """
    if not progress.total:
        return 0
    pct = round(progress.processed / progress.total * 100)
    return max(0, min(100, pct))


def job_to_summary(job: EnrichJob) -> JobSummary:
    return JobSummary(
        id=job.id,
        tenant_id=job.tenant_id,
        kg_name=job.kg_name,
        type_name=job.type_name,
        attributes=job.attributes,
        tier=job.tier,
        status=job.status,
        progress=job.progress,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        conflict_policy=job.conflict_policy,
        confidence_min=job.confidence_min,
        error=job.error,
        category=job.category,
        trigger=job.trigger,
        last_run=job.last_run,
        next_run=job.next_run,
        cost=job.cost,
        cost_note=job.cost_note,
        progress_pct=_progress_pct(job.progress),
    )
