"""Type matching — resolves proposed type names against the existing ontology.

Cascade architecture (fast → slow):
  1. Verdict cache (instant, free)
  2. Embedding similarity via OntologyEmbeddingService (fast, cheap)
     - cosine > 0.92 → SAME (no LLM)
     - cosine < 0.55 → DIFFERENT (no LLM)
     - 0.55–0.92 → pass top-3 candidates to LLM
  3. LLM judgment for ambiguous cases (~5% of decisions)
  4. 3-judge fan-out for deadlocks (~1% of decisions)
"""

from __future__ import annotations

import os

import numpy as np

import anthropic
import structlog

from cograph_client.resolver.models import MatchVerdict, TypeMatch
from cograph_client.resolver.verdict_cache import JsonVerdictCache, VerdictEntry

logger = structlog.stdlib.get_logger("cograph.resolver.type_matcher")

# Type-matching decision model (reuse-vs-expand verdict + ambiguous judge
# fan-out) — env-overridable; default preserves prior behavior.
MATCH_MODEL = os.environ.get("OMNIX_MATCH_MODEL", "claude-sonnet-4-6")

# Embedding similarity thresholds
EMBEDDING_SAME_THRESHOLD = 0.92
EMBEDDING_SUBTYPE_THRESHOLD = 0.78
EMBEDDING_DIFFERENT_THRESHOLD = 0.55

MATCH_SYSTEM_PROMPT = """\
You are a schema matching expert for a knowledge graph. Your job is to decide
whether a proposed entity type matches an existing type in the ontology.

For each proposed type, you must return one of:
- SAME: The proposed type is semantically identical to an existing type (just a different name).
- SUBTYPE: The proposed type is a more specific version of an existing type (is-a relationship).
- DIFFERENT: The proposed type is genuinely new and does not match any existing type.

Hierarchy-first principle:
When in doubt between SUBTYPE and DIFFERENT, prefer SUBTYPE. A connected ontology \
is far more useful than a flat collection of unrelated types. If the proposed type \
could plausibly be described as "a kind of" an existing type, it IS a subtype. \
Only choose DIFFERENT when the proposed type is truly unrelated to ALL existing types.

CRITICAL: subClassOf means "is a kind of" (type hierarchy), NOT "is contained in" \
or "is part of" (spatial/geographic/compositional relationships). Geographic \
containment is a RELATIONSHIP, not a subtype:
- State is NOT a subtype of City (State contains cities, not "is a kind of" city)
- City is NOT a subtype of State (cities are located in states, not "a kind of" state)
- ZipCode is NOT a subtype of City (zip codes are within cities)
- Country is NOT a subtype of State
When two types have a containment/location relationship, return DIFFERENT.

Examples:
- "Broker" with existing "Person" → SUBTYPE (a broker is a kind of person/agent)
- "City" with existing "Place" → SUBTYPE (a city is a kind of place)
- "RealEstateBroker" with existing "Person" → SUBTYPE (a real estate broker is a person)
- "Invoice" with existing "Person", "Place" → DIFFERENT (genuinely unrelated)
- "State" with existing "City" → DIFFERENT (geographic containment, not a subtype)
- "City" with existing "State" → DIFFERENT (geographic containment, not a subtype)
- "ZipCode" with existing "City" → DIFFERENT (geographic containment, not a subtype)

Respond with valid JSON only. No markdown, no explanation."""

MATCH_USER_TEMPLATE = """\
Existing types in the ontology:
{existing_types}

Proposed type: "{proposed_type}"
{proposed_description}

Compare the proposed type against each existing type. Return JSON:
{{
  "verdict": "SAME" | "SUBTYPE" | "DIFFERENT",
  "matched_type": "<name of matched existing type or null>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence>"
}}"""

JUDGE_SYSTEM_PROMPT = """\
You are one of three independent judges resolving an ambiguous type match in a
knowledge graph ontology. You must make a forced choice.

When in doubt between SUBTYPE and DIFFERENT, prefer SUBTYPE. A connected graph \
is more valuable than isolated nodes. If it could be "a kind of" the existing \
type, it is a SUBTYPE. But geographic/spatial containment (State contains City, \
City contains ZipCode) is NOT a subtype relationship — return DIFFERENT for those.

Respond with valid JSON only. No markdown, no explanation."""

JUDGE_USER_TEMPLATE = """\
Existing type: "{existing_type}" — {existing_description}
Proposed type: "{proposed_type}" — {proposed_description}

Are these the SAME concept, is the proposed type a SUBTYPE of the existing one,
or are they DIFFERENT concepts?

Return JSON:
{{
  "verdict": "SAME" | "SUBTYPE" | "DIFFERENT",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<one sentence>"
}}"""


class TypeMatcher:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic,
        cache: JsonVerdictCache,
        embedding_service: object | None = None,
        graph_uri: str = "",
    ):
        self._client = client
        self._cache = cache
        self._embedding_service = embedding_service
        self._graph_uri = graph_uri

    async def match(
        self,
        proposed_type: str,
        proposed_description: str,
        existing_types: dict[str, str],
    ) -> TypeMatch:
        """Match a proposed type against the existing ontology.

        Cascade: cache → embeddings → LLM → 3-judge fan-out.
        """
        if not existing_types:
            logger.info("type_match_auto_new", proposed=proposed_type, reason="empty_ontology")
            return TypeMatch(
                proposed=proposed_type,
                resolved=proposed_type,
                verdict=MatchVerdict.DIFFERENT,
                confidence=1.0,
                is_new=True,
            )

        # Layer 0: exact case-insensitive name match. Short-circuits LLM calls
        # for the common case where the extractor proposes the same type name
        # that already exists (e.g. "Company" re-seen row after row).
        proposed_norm = proposed_type.strip().lower()
        for existing_name in existing_types:
            if existing_name.strip().lower() == proposed_norm:
                logger.info("type_match_exact_name", proposed=proposed_type, resolved=existing_name)
                return TypeMatch(
                    proposed=proposed_type,
                    resolved=existing_name,
                    verdict=MatchVerdict.SAME,
                    confidence=1.0,
                    is_new=False,
                )

        # Layer 1: Verdict cache (instant)
        for existing_name in existing_types:
            cached = await self._cache.get(proposed_type, existing_name)
            if cached and cached.verdict in (MatchVerdict.SAME, MatchVerdict.SUBTYPE):
                logger.info(
                    "type_match_cached",
                    proposed=proposed_type,
                    resolved=existing_name,
                    verdict=cached.verdict.value,
                )
                return TypeMatch(
                    proposed=proposed_type,
                    resolved=existing_name if cached.verdict == MatchVerdict.SAME else proposed_type,
                    verdict=cached.verdict,
                    confidence=cached.confidence,
                    is_new=cached.verdict != MatchVerdict.SAME,
                    parent_type=existing_name if cached.verdict == MatchVerdict.SUBTYPE else None,
                )

        all_different = True
        for existing_name in existing_types:
            cached = await self._cache.get(proposed_type, existing_name)
            if cached is None:
                all_different = False
                break
        if all_different:
            logger.info("type_match_cached_all_different", proposed=proposed_type)
            return TypeMatch(
                proposed=proposed_type,
                resolved=proposed_type,
                verdict=MatchVerdict.DIFFERENT,
                confidence=1.0,
                is_new=True,
            )

        # Layer 2: Embedding similarity (fast, cheap)
        embedding_result = await self._embedding_pre_filter(proposed_type, existing_types)
        if embedding_result is not None:
            return embedding_result

        # Layer 3: LLM judgment (for the ambiguous band). If the LLM provider
        # is unavailable (quota, outage), fall back to treating the proposed
        # type as new rather than failing the whole ingest.
        try:
            initial = await self._initial_match(proposed_type, proposed_description, existing_types)
        except Exception as exc:
            logger.warning("type_match_llm_unavailable", proposed=proposed_type, error=str(exc))
            for existing_name in existing_types:
                await self._cache.put(VerdictEntry(
                    proposed_type, existing_name, MatchVerdict.DIFFERENT, 0.5,
                ))
            return TypeMatch(
                proposed=proposed_type,
                resolved=proposed_type,
                verdict=MatchVerdict.DIFFERENT,
                confidence=0.5,
                is_new=True,
                inconclusive=True,
            )

        if initial["confidence"] > 0.90 and initial["verdict"] == "SAME":
            match = TypeMatch(
                proposed=proposed_type,
                resolved=initial["matched_type"],
                verdict=MatchVerdict.SAME,
                confidence=initial["confidence"],
                is_new=False,
            )
            await self._cache.put(VerdictEntry(
                proposed_type, initial["matched_type"], MatchVerdict.SAME, initial["confidence"],
            ))
            return match

        if initial["confidence"] < 0.40 or (initial["verdict"] == "DIFFERENT" and initial["confidence"] > 0.80):
            match = TypeMatch(
                proposed=proposed_type,
                resolved=proposed_type,
                verdict=MatchVerdict.DIFFERENT,
                confidence=initial["confidence"],
                is_new=True,
            )
            for existing_name in existing_types:
                await self._cache.put(VerdictEntry(
                    proposed_type, existing_name, MatchVerdict.DIFFERENT, initial["confidence"],
                ))
            return match

        if initial["confidence"] > 0.70 and initial["verdict"] == "SUBTYPE":
            match = TypeMatch(
                proposed=proposed_type,
                resolved=proposed_type,
                verdict=MatchVerdict.SUBTYPE,
                confidence=initial["confidence"],
                is_new=True,
                parent_type=initial["matched_type"],
            )
            await self._cache.put(VerdictEntry(
                proposed_type, initial["matched_type"], MatchVerdict.SUBTYPE, initial["confidence"],
            ))
            return match

        # Layer 4: 3-judge fan-out for ambiguous cases
        matched_type = initial.get("matched_type") or list(existing_types.keys())[0]
        matched_desc = existing_types.get(matched_type, "")
        return await self._judge_ambiguous(
            proposed_type, proposed_description, matched_type, matched_desc,
        )

    async def _embedding_pre_filter(
        self,
        proposed_type: str,
        existing_types: dict[str, str],
    ) -> TypeMatch | None:
        """Use embedding similarity to resolve obvious matches without LLM calls.

        Returns a TypeMatch for high-confidence decisions, or None to fall through to LLM.
        """
        if self._embedding_service is None or not self._graph_uri:
            return None

        store = self._embedding_service._stores.get(self._graph_uri)
        if store is None or not store.chunks:
            return None

        # Embed the proposed type name
        try:
            embeddings = await self._embedding_service._embed_texts([proposed_type])
            proposed_vec = np.array(embeddings[0], dtype=np.float32)
        except Exception:
            logger.warning("embedding_pre_filter_failed", proposed=proposed_type, exc_info=True)
            return None

        # Compare against all existing type embeddings that are in the store
        candidates: list[tuple[str, float]] = []
        for type_name in existing_types:
            chunk = store.chunks.get(type_name)
            if chunk is None:
                continue
            sim = float(np.dot(proposed_vec, chunk.embedding) / (
                np.linalg.norm(proposed_vec) * np.linalg.norm(chunk.embedding) + 1e-9
            ))
            candidates.append((type_name, sim))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[1], reverse=True)
        best_name, best_sim = candidates[0]

        logger.info(
            "embedding_pre_filter",
            proposed=proposed_type,
            best_match=best_name,
            similarity=round(best_sim, 4),
        )

        # High similarity → SAME (no LLM needed)
        if best_sim >= EMBEDDING_SAME_THRESHOLD:
            verdict = MatchVerdict.SAME
            await self._cache.put(VerdictEntry(
                proposed_type, best_name, verdict, best_sim,
            ))
            logger.info("embedding_resolved_same", proposed=proposed_type, resolved=best_name, sim=round(best_sim, 4))
            return TypeMatch(
                proposed=proposed_type,
                resolved=best_name,
                verdict=verdict,
                confidence=best_sim,
                is_new=False,
            )

        # Low similarity → DIFFERENT (no LLM needed)
        if best_sim < EMBEDDING_DIFFERENT_THRESHOLD:
            verdict = MatchVerdict.DIFFERENT
            for existing_name in existing_types:
                await self._cache.put(VerdictEntry(
                    proposed_type, existing_name, verdict, 1.0 - best_sim,
                ))
            logger.info("embedding_resolved_different", proposed=proposed_type, best_sim=round(best_sim, 4))
            return TypeMatch(
                proposed=proposed_type,
                resolved=proposed_type,
                verdict=verdict,
                confidence=1.0 - best_sim,
                is_new=True,
            )

        # Mid-range with subtype signal
        if EMBEDDING_DIFFERENT_THRESHOLD <= best_sim < EMBEDDING_SAME_THRESHOLD:
            # Fall through to LLM, but narrow candidates to top-3
            # Store candidates so _initial_match can use them
            self._narrowed_candidates = {
                name: existing_types.get(name, "")
                for name, _ in candidates[:3]
            }
            logger.info(
                "embedding_ambiguous_narrowed",
                proposed=proposed_type,
                candidates=[c[0] for c in candidates[:3]],
                sims=[round(c[1], 4) for c in candidates[:3]],
            )

        return None  # Fall through to LLM

    async def _initial_match(
        self,
        proposed_type: str,
        proposed_description: str,
        existing_types: dict[str, str],
    ) -> dict:
        # Use narrowed candidates from embedding pre-filter if available
        candidates = getattr(self, "_narrowed_candidates", None)
        if candidates:
            target_types = candidates
            self._narrowed_candidates = None  # consume once
            logger.info("llm_using_narrowed_candidates", proposed=proposed_type, candidates=list(target_types.keys()))
        else:
            target_types = existing_types

        types_text = "\n".join(
            f'- "{name}": {desc}' if desc else f'- "{name}"'
            for name, desc in target_types.items()
        )
        desc_line = f'Description: "{proposed_description}"' if proposed_description else ""

        msg = await self._client.messages.create(
            model=MATCH_MODEL,
            max_tokens=256,
            system=MATCH_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": MATCH_USER_TEMPLATE.format(
                    existing_types=types_text,
                    proposed_type=proposed_type,
                    proposed_description=desc_line,
                ),
            }],
        )

        import json
        try:
            result = json.loads(msg.content[0].text)
            return {
                "verdict": result.get("verdict", "DIFFERENT"),
                "matched_type": result.get("matched_type"),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", ""),
            }
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.warning("type_match_parse_error", error=str(e), raw=msg.content[0].text)
            return {"verdict": "DIFFERENT", "matched_type": None, "confidence": 0.5, "reasoning": "parse error"}

    async def _judge_ambiguous(
        self,
        proposed_type: str,
        proposed_description: str,
        existing_type: str,
        existing_description: str,
    ) -> TypeMatch:
        """Fan out to 3 independent LLM judges for ambiguous matches."""
        import asyncio
        import json

        async def single_judge() -> dict:
            msg = await self._client.messages.create(
                model=MATCH_MODEL,
                max_tokens=256,
                temperature=0.7,  # diversity between judges
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": JUDGE_USER_TEMPLATE.format(
                        existing_type=existing_type,
                        existing_description=existing_description,
                        proposed_type=proposed_type,
                        proposed_description=proposed_description,
                    ),
                }],
            )
            try:
                return json.loads(msg.content[0].text)
            except (json.JSONDecodeError, IndexError):
                return {"verdict": "DIFFERENT", "confidence": 0.5}

        results = await asyncio.gather(single_judge(), single_judge(), single_judge())
        verdicts = [r.get("verdict", "DIFFERENT") for r in results]

        logger.info(
            "judge_votes",
            proposed=proposed_type,
            existing=existing_type,
            verdicts=verdicts,
        )

        # Majority vote
        from collections import Counter
        counts = Counter(verdicts)
        winner, count = counts.most_common(1)[0]

        if count == 1:
            # 3-way split — flag for user
            logger.warning("judge_deadlock", proposed=proposed_type, existing=existing_type)
            await self._cache.put(VerdictEntry(
                proposed_type, existing_type, MatchVerdict.FLAGGED, 0.5,
            ))
            return TypeMatch(
                proposed=proposed_type,
                resolved=proposed_type,
                verdict=MatchVerdict.FLAGGED,
                confidence=0.5,
                is_new=True,
            )

        avg_confidence = sum(
            r.get("confidence", 0.5) for r in results if r.get("verdict") == winner
        ) / count

        verdict = MatchVerdict(winner)
        await self._cache.put(VerdictEntry(
            proposed_type, existing_type, verdict, avg_confidence,
        ))

        if verdict == MatchVerdict.SAME:
            return TypeMatch(
                proposed=proposed_type,
                resolved=existing_type,
                verdict=verdict,
                confidence=avg_confidence,
                is_new=False,
            )
        elif verdict == MatchVerdict.SUBTYPE:
            return TypeMatch(
                proposed=proposed_type,
                resolved=proposed_type,
                verdict=verdict,
                confidence=avg_confidence,
                is_new=True,
                parent_type=existing_type,
            )
        else:
            return TypeMatch(
                proposed=proposed_type,
                resolved=proposed_type,
                verdict=verdict,
                confidence=avg_confidence,
                is_new=True,
            )
