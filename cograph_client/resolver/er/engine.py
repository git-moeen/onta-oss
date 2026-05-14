"""ER engine — orchestrates signal extraction, blocking, scoring, decision.

Single entry point: `ERPipeline.find_match(entity, type_uri, instance_graph)`.
Returns a MergeDecision the caller acts on.

Signal extraction from `ExtractedEntity.attributes` is heuristic — attribute
names vary across CSVs (`email`, `guest_email`, `email_address`, `e-mail`),
so we substring-match against a small set of known patterns.
"""

from __future__ import annotations

import logging

from cograph_client.resolver.er.blocking import SparqlBlocker, generate_block_keys
from cograph_client.resolver.er.normalize import DefaultNormalizer
from cograph_client.resolver.er.scoring import DefaultScorer
from cograph_client.resolver.er.types import (
    ERConfig,
    EntitySignals,
    MatchScore,
    MergeAction,
    MergeDecision,
    NormalizedSignals,
    config_for,
)
from cograph_client.resolver.models import ExtractedEntity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal extraction — map ExtractedEntity attributes → EntitySignals
# ---------------------------------------------------------------------------


def _attr_value(entity: ExtractedEntity, patterns: tuple[str, ...]) -> str | None:
    """Find the first attribute whose name matches any of the patterns
    (case-insensitive substring) and return its value. None if not found
    or value is empty."""
    if not entity.attributes:
        return None
    for attr in entity.attributes:
        n = (attr.name or "").lower()
        for p in patterns:
            if p in n:
                v = (attr.value or "").strip()
                return v or None
    return None


# Note: `id` is used as the canonical name fallback when no name-shaped
# attribute exists — many CSV ingest paths put the human-readable label there.
NAME_PATTERNS = ("guest_name", "full_name", "member_name", "contact_name", "customer_name", "name")
EMAIL_PATTERNS = ("email", "e-mail", "e_mail", "mail_address")
PHONE_PATTERNS = ("phone", "mobile", "tel", "contact_number")
ADDRESS_PATTERNS = ("address", "street", "addr")
DOB_PATTERNS = ("dob", "birth", "date_of_birth")


def extract_signals(entity: ExtractedEntity) -> EntitySignals:
    """Pull ER signals out of an extracted entity's attributes.

    For Guest-shape entities most CSVs split name into first/last. We check
    first/last splits BEFORE generic name patterns, because the generic
    pattern "name" substring-matches "guest_first_name" / "guest_last_name"
    and would incorrectly return only one half of the name.
    """
    first = _attr_value(entity, ("first_name", "given_name", "firstname", "forename"))
    last = _attr_value(entity, ("last_name", "family_name", "surname", "lastname"))
    if first and last:
        name = f"{first} {last}"
    elif first or last:
        name = first or last
    else:
        name = _attr_value(entity, NAME_PATTERNS)
        if not name:
            # Fall back to the entity id (often the CSV-row label)
            name = entity.id or None

    # Collect ALL email-shaped attributes — primary plus any aliases.
    # CRMs almost always have primary + secondary (work + personal). Loyalty
    # systems often have a registration email distinct from booking email.
    # Treating only the "first" email loses the majority of cross-system
    # matches in practice.
    emails: list[str] = []
    seen_emails: set[str] = set()
    for attr in entity.attributes or ():
        n = (attr.name or "").lower()
        if any(p in n for p in EMAIL_PATTERNS):
            v = (attr.value or "").strip()
            if v and v.lower() not in seen_emails:
                emails.append(v)
                seen_emails.add(v.lower())

    primary_email = emails[0] if emails else None
    aliases = tuple(emails[1:])

    return EntitySignals(
        name=name,
        email=primary_email,
        email_aliases=aliases,
        phone=_attr_value(entity, PHONE_PATTERNS),
        address=_attr_value(entity, ADDRESS_PATTERNS),
        dob=_attr_value(entity, DOB_PATTERNS),
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ERPipeline:
    """Per-resolver instance. Stateless across calls; shares Neptune client."""

    def __init__(self, neptune):
        self._normalizer = DefaultNormalizer()
        self._scorer = DefaultScorer()
        self._blocker = SparqlBlocker(neptune)

    async def find_match(
        self,
        entity: ExtractedEntity,
        type_name: str,
        type_uri: str,
        instance_graph: str,
        config: ERConfig | None = None,
    ) -> MergeDecision:
        """Run ER for one entity. Return a MergeDecision the caller acts on.

        If `config` is None, look up defaults by type_name. If still None,
        ER is skipped (the caller falls back to exact-URI-match dedup).
        """
        if config is None:
            config = config_for(type_name)
        if config is None:
            return MergeDecision(action=MergeAction.SKIP)

        raw = extract_signals(entity)
        if not raw.has_any():
            return MergeDecision(action=MergeAction.SKIP)

        normalized = self._normalizer.normalize(raw)
        if not normalized.has_any():
            return MergeDecision(action=MergeAction.SKIP)

        keys = generate_block_keys(normalized)
        if not keys:
            return MergeDecision(action=MergeAction.SKIP)

        try:
            candidates = await self._blocker.candidates_with_signals(
                instance_graph, type_uri, keys,
            )
        except Exception as e:
            # ER is best-effort; never block ingest. Log and fall through.
            logger.warning("er_blocker_failed", error=str(e), type=type_name)
            return MergeDecision(action=MergeAction.SKIP)

        if not candidates:
            return MergeDecision(action=MergeAction.SKIP)

        scores: list[MatchScore] = []
        for uri, cand_signals in candidates.items():
            ms = self._scorer.score(normalized, cand_signals, config)
            # Caller-fills candidate_uri (per scorer contract), do it here
            ms = MatchScore(candidate_uri=uri, score=ms.score, contributions=ms.contributions)
            scores.append(ms)

        scores.sort(key=lambda s: s.score, reverse=True)
        best = scores[0]

        if best.score >= config.auto_merge_threshold:
            action = MergeAction.AUTO_MERGE
            canonical = best.candidate_uri
        elif best.score >= config.review_threshold:
            action = MergeAction.REVIEW
            canonical = None
        else:
            action = MergeAction.SKIP
            canonical = None

        logger.info(
            "er_decision",
            type=type_name,
            action=action.value,
            score=round(best.score, 3),
            canonical=canonical,
            candidates_evaluated=len(scores),
        )

        return MergeDecision(
            action=action,
            canonical_uri=canonical,
            best_match=best,
            all_scores=scores,
        )

    @staticmethod
    def signals_and_keys(entity: ExtractedEntity) -> tuple[NormalizedSignals | None, list]:
        """Helper for the caller: extract + normalize + generate block keys.
        Used when the caller wants to write the block-index triples for a
        newly-minted (non-merged) entity. Returns (normalized, keys) or
        (None, []) when ER doesn't apply."""
        raw = extract_signals(entity)
        if not raw.has_any():
            return None, []
        normalizer = DefaultNormalizer()
        normalized = normalizer.normalize(raw)
        if not normalized.has_any():
            return None, []
        keys = generate_block_keys(normalized)
        return normalized, keys
