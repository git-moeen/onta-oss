"""Shared types for the cross-file entity resolution pipeline.

The pipeline shape is:
    ExtractedEntity (from LLM extraction)
      └─► extract_signals()  → EntitySignals
              └─► Normalizer  → NormalizedSignals
                      └─► Blocker.candidates()  → list[CanonicalURI]
                              └─► Scorer.score()  → MatchScore (per candidate)
                                      └─► Decide  → MergeDecision

Every module in cograph_client.resolver.er should import its types from here
so the contract stays single-sourced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol


# ---------------------------------------------------------------------------
# Signals — the inputs to ER
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntitySignals:
    """Raw signal values extracted from an entity's attributes.

    Values are the literal strings as they came off the row (pre-normalize).
    Missing signals are None, not "" — so the scorer can correctly skip them
    in weight renormalization.

    `email_aliases` covers secondary/work/personal emails the entity is also
    known by. Real-world CRMs almost always have a primary + secondary; ignoring
    aliases is the main reason naive ER misses obvious matches.
    """

    name: str | None = None
    email: str | None = None
    email_aliases: tuple[str, ...] = ()
    phone: str | None = None
    address: str | None = None
    dob: str | None = None  # any date-ish string; normalizer canonicalizes

    def has_any(self) -> bool:
        return any(v is not None and v != "" for v in (
            self.name, self.email, self.phone, self.address, self.dob,
        )) or bool(self.email_aliases)


@dataclass(frozen=True)
class NormalizedSignals:
    """Canonicalized signal values, ready for comparison."""

    name: str | None = None             # lowercase, diacritics stripped, nickname expanded
    name_tokens: tuple[str, ...] = ()   # sorted set of name tokens (for token-set scoring)
    email: str | None = None            # local-part normalized, gmail dots stripped, +tag dropped
    email_local: str | None = None      # the local part only (useful for blocking)
    email_aliases: tuple[str, ...] = ()       # normalized alternate emails (full form)
    email_locals: tuple[str, ...] = ()        # normalized local parts of all known emails (incl. primary)
    phone_e164: str | None = None       # +CC + digits only
    address: str | None = None          # USPS-abbreviated lowercase
    address_tokens: tuple[str, ...] = ()
    dob_iso: str | None = None          # YYYY-MM-DD

    def has_any(self) -> bool:
        return any((self.name, self.email, self.phone_e164, self.address, self.dob_iso)) \
            or bool(self.email_aliases)


# ---------------------------------------------------------------------------
# Blocking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockKey:
    """A small key string used to look up candidate matches in O(1)-ish time.

    Each entity emits multiple BlockKeys (one per blocking strategy). A
    candidate match is any existing entity that shares at least one BlockKey
    with the new entity.
    """

    kind: str   # e.g. "email_local", "lastname3_phone4", "soundex_finit"
    value: str  # the actual key string


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalContribution:
    """Why this signal contributed what it did to the score.

    Surfaced in the audit log so a reviewer can see WHY two records matched.
    """

    signal: str          # "email", "phone", "name", ...
    weight: float        # the (possibly re-normalized) weight applied
    similarity: float    # comparator output in [0, 1]
    contribution: float  # weight * similarity (the actual addend)


@dataclass
class MatchScore:
    """Result of scoring one candidate against the incoming entity."""

    candidate_uri: str
    score: float                                   # final weighted similarity in [0, 1]
    contributions: list[SignalContribution] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


class MergeAction(str, Enum):
    AUTO_MERGE = "AUTO_MERGE"   # score >= auto_threshold → write owl:sameAs, reuse URI
    REVIEW = "REVIEW"           # score in [review_threshold, auto_threshold) → enqueue, no merge
    SKIP = "SKIP"               # score < review_threshold or no candidates → new entity


@dataclass
class MergeDecision:
    """The output of the ER pipeline for one new entity."""

    action: MergeAction
    canonical_uri: str | None = None  # set when AUTO_MERGE — the URI to reuse
    best_match: MatchScore | None = None
    all_scores: list[MatchScore] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-type configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ERConfig:
    """Per-type ER configuration. Loaded from ontology metadata.

    `decisive_signals` lists signals whose exact match on both sides is
    sufficient on its own to merge — bypassing the weighted-sum pass. This
    is the standard "deterministic rules on top of probabilistic scoring"
    pattern in ER literature.

    Use ONLY for signals that are genuinely globally-unique identifiers in
    the real world (email, GTIN, tax_id, DUNS, ORCID, ISBN). Do NOT mark
    phone or address decisive — family-shared phones and roommate-shared
    addresses cause wrong merges.
    """

    type_name: str
    signals: tuple[str, ...]          # e.g. ("email", "phone", "name", "address", "dob")
    weights: tuple[float, ...]         # parallel to signals; must sum to 1.0 within float tolerance
    auto_merge_threshold: float = 0.90
    review_threshold: float = 0.70
    decisive_signals: tuple[str, ...] = ()  # exact match on any → AUTO_MERGE

    def weight_for(self, signal: str) -> float:
        try:
            idx = self.signals.index(signal)
        except ValueError:
            return 0.0
        return self.weights[idx]

    def is_decisive(self, signal: str) -> bool:
        return signal in self.decisive_signals


DEFAULT_GUEST_CONFIG = ERConfig(
    type_name="Guest",
    signals=("email", "phone", "name", "address", "dob"),
    weights=(0.45, 0.25, 0.20, 0.05, 0.05),
    auto_merge_threshold=0.90,
    review_threshold=0.70,
    # Email is genuinely globally unique (modulo family-shared accounts,
    # which are rare in commercial contexts). Phone is NOT — shared family
    # landlines and reused work mobiles cause wrong merges.
    decisive_signals=("email",),
)

DEFAULT_CUSTOMER_CONFIG = ERConfig(
    type_name="Customer",
    signals=("email", "phone", "name", "address"),
    weights=(0.50, 0.25, 0.20, 0.05),
    auto_merge_threshold=0.90,
    review_threshold=0.70,
    decisive_signals=("email",),
)

DEFAULT_PROPERTY_CONFIG = ERConfig(
    type_name="Property",
    signals=("address", "name"),
    weights=(0.70, 0.30),
    auto_merge_threshold=0.95,
    review_threshold=0.80,
    # Property has no globally-unique identifier in our default signal set.
    # Tenants who ingest a stable property_id should override the config
    # to mark it decisive.
    decisive_signals=(),
)

DEFAULTS_BY_TYPE: dict[str, ERConfig] = {
    "Guest": DEFAULT_GUEST_CONFIG,
    "Customer": DEFAULT_CUSTOMER_CONFIG,
    "Contact": DEFAULT_CUSTOMER_CONFIG,
    "LoyaltyMember": DEFAULT_GUEST_CONFIG,
    "Member": DEFAULT_GUEST_CONFIG,
    "Property": DEFAULT_PROPERTY_CONFIG,
    "Hotel": DEFAULT_PROPERTY_CONFIG,
}


def config_for(type_name: str) -> ERConfig | None:
    """Look up the default ERConfig for a type. Returns None if the type
    isn't in our ER-enabled set — in which case ER is skipped and the
    exact-URI-match path is used as today."""
    return DEFAULTS_BY_TYPE.get(type_name)


# ---------------------------------------------------------------------------
# Audit / merge-event log
# ---------------------------------------------------------------------------


@dataclass
class MergeEvent:
    """Recorded for every AUTO_MERGE so a merge can be audited or reversed."""

    canonical_uri: str
    merged_uri: str          # the URI that got absorbed (loser)
    score: float
    contributions: list[SignalContribution]
    triggered_at: datetime
    triggered_by: str        # batch_id or user id


# ---------------------------------------------------------------------------
# Protocols (so plugins / proprietary impls can swap pieces)
# ---------------------------------------------------------------------------


class Normalizer(Protocol):
    def normalize(self, signals: EntitySignals) -> NormalizedSignals: ...


class Blocker(Protocol):
    """Looks up candidate matches.

    Implementations may be SPARQL-backed (default) or Redis-backed (high-scale).
    """

    async def block_keys(self, normalized: NormalizedSignals) -> list[BlockKey]: ...
    async def candidates(self, tenant_id: str, type_name: str, keys: list[BlockKey]) -> list[str]: ...
    async def index(self, tenant_id: str, type_name: str, entity_uri: str, keys: list[BlockKey]) -> None: ...


class Scorer(Protocol):
    def score(
        self,
        incoming: NormalizedSignals,
        candidate: NormalizedSignals,
        config: ERConfig,
    ) -> MatchScore: ...
