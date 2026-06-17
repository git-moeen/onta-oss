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


class ConflictPolicy(str, Enum):
    skip = "skip"
    verify = "verify"
    overwrite = "overwrite"
    stage = "stage"


class EnrichRequest(BaseModel):
    type_name: str
    attributes: list[str]
    tier: EnrichmentTier = EnrichmentTier.lite
    kg_name: str
    conflict_policy: ConflictPolicy = ConflictPolicy.stage
    confidence_min: float = 0.85
    limit: Optional[int] = None


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


ReviewDecision = Literal["accept", "reject", "skip"]


class ConflictReview(BaseModel):
    entity_uri: str
    attribute: str
    existing_value: str
    proposed: Verdict
    decision: Optional[ReviewDecision] = None


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
    )
