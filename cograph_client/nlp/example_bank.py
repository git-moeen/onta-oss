"""SPARQL example bank with semantic retrieval for few-shot prompting.

Stores (question, SPARQL) pairs from successful evaluations. At query time,
retrieves the most relevant examples via embedding similarity with anti-cheat
filtering, cross-dataset preference, and pattern diversity.

Uses the same OpenRouter text-embedding-3-small embeddings as ontology_embeddings.py.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Shared embed client (ONTA-174) — model/batching/errors live in ONE place.
# Constants are re-exported for backward compatibility with existing importers.
from cograph_client.nlp.embed_client import (  # noqa: F401 — re-exports
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    OPENROUTER_EMBEDDINGS_URL,
    embed_texts,
)
from cograph_client.nlp.embed_client import cosine_similarity as _cosine_similarity  # noqa: F401

logger = logging.getLogger(__name__)

# Bank limits
MAX_BANK_SIZE = 500

# Similarity thresholds
ANTI_CHEAT_THRESHOLD = 0.90  # Exclude examples too similar to excluded questions
SAME_DATASET_MAX_SIM = 0.75  # Same-KG examples must be below this to avoid near-cheating

# Pattern tags detected from SPARQL text
PATTERN_DETECTORS: list[tuple[str, str]] = [
    ("count", r"COUNT\s*\("),
    ("avg", r"AVG\s*\("),
    ("max", r"MAX\s*\("),
    ("sum", r"SUM\s*\("),
    ("filter", r"FILTER\s*\("),
    ("contains", r"CONTAINS\s*\("),
    ("date_filter", r"xsd:dateTime"),
    ("group_by", r"GROUP\s+BY"),
]

# Default file paths
DEFAULT_BANK_PATH = Path(__file__).resolve().parent.parent.parent / "eval_reports" / "example_bank.jsonl"
EVAL_REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "eval_reports"

# Holdout v2 KG exclusion list (spec §6.1): holdout-v2 KGs must never be
# indexed into the example bank, regardless of whether they appear in
# eval_reports. The manifest is the source of truth; we fall back to a
# hardcoded list if the manifest isn't reachable (e.g. prod deploys that
# don't ship eval_holdout_v2/). Drift between the fallback and manifest
# is logged at import time so it gets noticed.
_HOLDOUT_V2_KGS_FALLBACK: frozenset[str] = frozenset({
    # healthcare
    "cms-nursing-home-compare",
    "samhsa-n-ssats",
    "medicare-part-d-pricing",
    "hrsa-hpsa",
    "cdc-fluview",
    "cdc-wonder-mortality",
    "npi-registry",
    # finance
    "sec-edgar-10k",
    "fdic-call-reports",
    "treasury-fiscaldata-securities",
    "cftc-swap-data",
    "ncua-credit-union-call-reports",
    "finra-trace-corporate-bonds",
    "ofr-financial-stability",
    # legal
    "patentsview",
    "scdb-supreme-court",
    "doj-enforcement-actions",
    "ftc-consent-decrees",
    "uspto-trademarks",
    "pacer-federal-dockets",
    "fec-enforcement",
    # scientific_public_sector
    "nsf-awards",
    "nih-reporter-non-clinical",
    "fema-disaster-declarations",
    "epa-water-quality-portal",
    "noaa-storm-events",
    "usda-agricultural-statistics",
    "doe-energy-research-grants",
})


def _load_holdout_v2_kgs() -> frozenset[str]:
    """Load holdout-v2 KG IDs from eval_holdout_v2/HOLDOUT_V2_MANIFEST.json.

    Searches a few plausible locations (the omnix-oss submodule lives inside
    the parent cograph repo, so the manifest is typically two or three
    parents up from this file). On any failure, returns the hardcoded
    fallback set and logs a warning so drift gets noticed.
    """
    # __file__ = .../omnix-oss/omnix/nlp/example_bank.py
    here = Path(__file__).resolve()
    candidates = [
        # cograph parent (submodule layout): .../cograph/omnix-oss/omnix/nlp/
        here.parent.parent.parent.parent / "eval_holdout_v2" / "HOLDOUT_V2_MANIFEST.json",
        # alt: one level up (standalone)
        here.parent.parent.parent / "eval_holdout_v2" / "HOLDOUT_V2_MANIFEST.json",
        # cwd fallback
        Path.cwd() / "eval_holdout_v2" / "HOLDOUT_V2_MANIFEST.json",
    ]
    for path in candidates:
        try:
            if not path.is_file():
                continue
            with open(path) as f:
                manifest = json.load(f)
            ids = frozenset(
                kg["id"] for kg in manifest.get("kgs", []) if kg.get("id")
            )
            if not ids:
                continue
            # Drift check vs fallback
            missing_from_fallback = ids - _HOLDOUT_V2_KGS_FALLBACK
            extra_in_fallback = _HOLDOUT_V2_KGS_FALLBACK - ids
            if missing_from_fallback or extra_in_fallback:
                logger.warning(
                    "HOLDOUT_V2_KGS fallback drift vs manifest %s: "
                    "missing_from_fallback=%s extra_in_fallback=%s",
                    path, sorted(missing_from_fallback), sorted(extra_in_fallback),
                )
            logger.info("Loaded %d holdout-v2 KGs from %s", len(ids), path)
            return ids
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Failed to read holdout-v2 manifest at %s: %s", path, exc)
            continue
    logger.warning(
        "HOLDOUT_V2_MANIFEST.json not found in any candidate path; "
        "using hardcoded HOLDOUT_V2_KGS fallback (%d entries). "
        "This is OK for prod deploys that don't ship eval_holdout_v2/.",
        len(_HOLDOUT_V2_KGS_FALLBACK),
    )
    return _HOLDOUT_V2_KGS_FALLBACK


HOLDOUT_V2_KGS: frozenset[str] = _load_holdout_v2_kgs()


@dataclass
class Example:
    """A single (question, SPARQL) example with metadata."""

    question: str
    sparql: str
    kg_name: str
    ontology_context: str
    pattern_tags: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "sparql": self.sparql,
            "kg_name": self.kg_name,
            "ontology_context": self.ontology_context,
            "pattern_tags": self.pattern_tags,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Example":
        return cls(
            question=d["question"],
            sparql=d["sparql"],
            kg_name=d.get("kg_name", ""),
            ontology_context=d.get("ontology_context", ""),
            pattern_tags=d.get("pattern_tags", []),
            embedding=d.get("embedding", []),
        )


def detect_pattern_tags(sparql: str) -> list[str]:
    """Auto-detect query pattern tags from SPARQL text.

    Detects aggregation functions (COUNT, AVG, MAX, SUM), filtering patterns
    (FILTER, CONTAINS, date), structural patterns (JOIN, GROUP BY, multi-hop).
    """
    tags: list[str] = []

    for tag, pattern in PATTERN_DETECTORS:
        if re.search(pattern, sparql, re.IGNORECASE):
            tags.append(tag)

    # "join" — 2+ triple patterns with different subjects
    subjects = set(re.findall(r"\?\w+\s+<", sparql))
    if len(subjects) >= 2:
        tags.append("join")

    # "multi_hop" — 3+ triple patterns (lines ending with ' .')
    triple_count = len(re.findall(r"\.\s*(?:\n|$|\})", sparql))
    # Also count triples separated by ' . '
    triple_count += sparql.count(" . ")
    if triple_count >= 3:
        tags.append("multi_hop")

    return sorted(set(tags))


class ExampleBank:
    """Persistent bank of (question, SPARQL) examples with semantic retrieval.

    Stores examples as JSONL on disk. Embeddings are generated via OpenRouter
    text-embedding-3-small (1536 dims). Retrieval uses cosine similarity with
    anti-cheat exclusion, cross-dataset preference, and pattern diversity.

    Usage:
        bank = ExampleBank(openrouter_api_key="sk-...")
        bank.load()
        await bank.add("How many events?", "SELECT ...", "events-kg", "Type: Event...")
        examples = await bank.retrieve("Count the events", "Type: Event...", top_k=3)
    """

    def __init__(self, openrouter_api_key: str, bank_path: str | Path | None = None):
        self._api_key = openrouter_api_key
        self._bank_path = Path(bank_path) if bank_path else DEFAULT_BANK_PATH
        self._examples: list[Example] = []

    @property
    def size(self) -> int:
        """Number of examples in the bank."""
        return len(self._examples)

    # ── Persistence ──────────────────────────────────────────────────────

    def load(self) -> int:
        """Load examples from JSONL file. Returns number loaded."""
        self._examples = []
        if not self._bank_path.exists():
            logger.info("Example bank file not found, starting empty: %s", self._bank_path)
            return 0

        with open(self._bank_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._examples.append(Example.from_dict(json.loads(line)))
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Skipping malformed example bank line: %s", exc)

        logger.info("Loaded %d examples from %s", len(self._examples), self._bank_path)
        return len(self._examples)

    def save(self) -> None:
        """Persist all examples to JSONL file."""
        self._bank_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._bank_path, "w") as f:
            for ex in self._examples:
                f.write(json.dumps(ex.to_dict()) + "\n")
        logger.info("Saved %d examples to %s", len(self._examples), self._bank_path)

    # ── Add examples ─────────────────────────────────────────────────────

    async def add(
        self,
        question: str,
        sparql: str,
        kg_name: str,
        ontology_context: str,
    ) -> bool:
        """Embed and store a new example. Returns True if added, False if duplicate or bank full.

        Deduplicates by checking if an example with the exact same question text
        already exists. Enforces MAX_BANK_SIZE cap.
        """
        # Dedup by exact question match
        for ex in self._examples:
            if ex.question.strip().lower() == question.strip().lower():
                logger.debug("Skipping duplicate question: %s", question[:80])
                return False

        if len(self._examples) >= MAX_BANK_SIZE:
            logger.warning("Example bank at capacity (%d), skipping add", MAX_BANK_SIZE)
            return False

        pattern_tags = detect_pattern_tags(sparql)
        embedding = await self._embed_single(question)

        self._examples.append(
            Example(
                question=question,
                sparql=sparql,
                kg_name=kg_name,
                ontology_context=ontology_context,
                pattern_tags=pattern_tags,
                embedding=embedding,
            )
        )
        return True

    async def add_batch(
        self,
        items: list[dict],
    ) -> int:
        """Bulk-add examples. Each dict must have: question, sparql, kg_name, ontology_context.

        Deduplicates, embeds in batches, and appends. Returns count of newly added examples.
        """
        # Filter out duplicates and existing
        existing_questions = {ex.question.strip().lower() for ex in self._examples}
        new_items: list[dict] = []
        for item in items:
            q = item["question"].strip().lower()
            if q in existing_questions:
                continue
            existing_questions.add(q)
            new_items.append(item)

        # Enforce cap
        capacity = MAX_BANK_SIZE - len(self._examples)
        if capacity <= 0:
            logger.warning("Example bank at capacity, skipping batch add")
            return 0
        new_items = new_items[:capacity]

        if not new_items:
            return 0

        # Batch embed all questions
        questions = [item["question"] for item in new_items]
        embeddings = await self._embed_texts(questions)

        for item, emb in zip(new_items, embeddings):
            self._examples.append(
                Example(
                    question=item["question"],
                    sparql=item["sparql"],
                    kg_name=item.get("kg_name", ""),
                    ontology_context=item.get("ontology_context", ""),
                    pattern_tags=detect_pattern_tags(item["sparql"]),
                    embedding=emb,
                )
            )

        logger.info("Added %d examples (batch), bank now has %d", len(new_items), len(self._examples))
        return len(new_items)

    # ── Retrieval ────────────────────────────────────────────────────────

    async def retrieve(
        self,
        question: str,
        ontology_context: str = "",
        exclude_questions: list[str] | None = None,
        kg_name: str = "",
        top_k: int = 3,
    ) -> list[Example]:
        """Retrieve the best few-shot examples for a query.

        Algorithm:
        1. Embed the incoming question.
        2. Cosine similarity against all examples -> top-10 candidates.
        3. EXCLUDE any example whose question similarity > 0.95 to any excluded question (anti-cheat).
        4. EXCLUDE same-dataset examples with similarity > 0.85 (too close).
        5. PREFER cross-dataset examples (different kg_name scores higher).
        6. DIVERSIFY: pick top_k examples with different pattern_tags when possible.

        Args:
            question: The natural language query.
            ontology_context: Current ontology summary (used for re-ranking).
            exclude_questions: Questions to exclude from results (anti-cheat).
            kg_name: The current KG name (for cross-dataset preference).
            top_k: Number of examples to return.

        Returns:
            List of up to top_k Example objects, pattern-diverse and relevant.
        """
        if not self._examples:
            return []

        exclude_questions = exclude_questions or []

        # Step 1: Embed the question
        q_embedding = await self._embed_single(question)
        q_vec = np.array(q_embedding, dtype=np.float32)

        # Build embedding matrix
        bank_matrix = np.stack(
            [np.array(ex.embedding, dtype=np.float32) for ex in self._examples]
        )
        similarities = _cosine_similarity(q_vec, bank_matrix)

        # Step 2: Top-10 candidates by raw similarity
        candidate_indices = np.argsort(similarities)[::-1][:10].tolist()

        # Step 3: Anti-cheat — embed excluded questions and filter
        exclude_vecs: list[np.ndarray] = []
        if exclude_questions:
            exclude_embeddings = await self._embed_texts(exclude_questions)
            exclude_vecs = [np.array(e, dtype=np.float32) for e in exclude_embeddings]

        filtered: list[tuple[int, float]] = []  # (index, adjusted_score)
        for idx in candidate_indices:
            ex = self._examples[idx]
            sim = float(similarities[idx])

            # Anti-cheat: check against excluded questions
            if exclude_vecs:
                ex_vec = np.array(ex.embedding, dtype=np.float32)
                excluded = False
                for ev in exclude_vecs:
                    excl_sim = float(np.dot(ex_vec, ev) / (np.linalg.norm(ex_vec) * np.linalg.norm(ev) + 1e-9))
                    if excl_sim > ANTI_CHEAT_THRESHOLD:
                        excluded = True
                        break
                if excluded:
                    continue

            # Step 4: Same-dataset anti-cheat — EVAL ONLY.
            # Gated on exclude_questions so this only runs during eval/benchmark
            # harness calls (which always pass exclude_questions). In production
            # /ask we WANT to reuse a near-identical prior answer on the same KG:
            # that's the best possible signal, not cheating. Dropping/penalizing
            # it here would actively hurt real users. Keep this in sync with the
            # anti-cheat gate above (line ~321).
            if exclude_vecs and kg_name and ex.kg_name == kg_name:
                if sim > SAME_DATASET_MAX_SIM:
                    continue  # Too similar within same dataset
                # Penalize same-dataset slightly to prefer cross-dataset
                sim *= 0.9

            filtered.append((idx, sim))

        if not filtered:
            return []

        # Sort by adjusted score
        filtered.sort(key=lambda x: x[1], reverse=True)

        # Step 5 & 6: Diversify by pattern_tags
        selected: list[Example] = []
        used_tag_sets: list[set[str]] = []

        for idx, _score in filtered:
            if len(selected) >= top_k:
                break
            ex = self._examples[idx]
            ex_tags = set(ex.pattern_tags)

            # Check if this example's tags are too similar to already-selected ones
            if selected and ex_tags:
                too_similar = False
                for used in used_tag_sets:
                    if used and ex_tags == used:
                        too_similar = True
                        break
                if too_similar:
                    # Still consider it if we haven't filled slots
                    continue

            selected.append(ex)
            used_tag_sets.append(ex_tags)

        # If diversity filtering was too aggressive, backfill from remaining
        if len(selected) < top_k:
            selected_set = {id(ex) for ex in selected}
            for idx, _score in filtered:
                if len(selected) >= top_k:
                    break
                ex = self._examples[idx]
                if id(ex) not in selected_set:
                    selected.append(ex)
                    selected_set.add(id(ex))

        return selected[:top_k]

    # ── Populate from eval reports ───────────────────────────────────────

    async def populate_from_eval_reports(self, reports_dir: str | Path | None = None) -> int:
        """Scan eval_reports/*.json for correct answers and bulk-add them.

        Also reads finetune_pairs.jsonl if present. Returns total examples added.

        Each eval report JSON has structure:
            {
                "kg_name": str,
                "ontology": str,
                "queries": {
                    "results": [{"question", "sparql", "verdict", ...}, ...]
                }
            }
        """
        reports_path = Path(reports_dir) if reports_dir else EVAL_REPORTS_DIR
        items: list[dict] = []
        seen_questions: set[str] = set()

        # 1. Scan eval report JSON files
        for json_file in sorted(reports_path.glob("eval-*.json")):
            try:
                with open(json_file) as f:
                    report = json.load(f)

                kg_name = report.get("kg_name", "")
                if kg_name in HOLDOUT_V2_KGS:
                    logger.debug(
                        "example_bank: skipping holdout-v2 KG %s from eval report %s",
                        kg_name, json_file,
                    )
                    continue
                ontology = report.get("ontology", "")
                results = report.get("queries", {}).get("results", [])

                for result in results:
                    if result.get("verdict") != "correct":
                        continue
                    question = result.get("question", "").strip()
                    sparql = result.get("sparql", "").strip()
                    if not question or not sparql:
                        continue
                    q_key = question.lower()
                    if q_key in seen_questions:
                        continue
                    seen_questions.add(q_key)
                    items.append({
                        "question": question,
                        "sparql": sparql,
                        "kg_name": kg_name,
                        "ontology_context": ontology,
                    })
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping eval report %s: %s", json_file, exc)

        # 2. Read finetune_pairs.jsonl
        finetune_path = reports_path / "finetune_pairs.jsonl"
        if finetune_path.exists():
            try:
                with open(finetune_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            pair = json.loads(line)
                            question = pair.get("question", "").strip()
                            sparql = pair.get("sparql", "").strip()
                            if not question or not sparql:
                                continue
                            q_key = question.lower()
                            if q_key in seen_questions:
                                continue
                            seen_questions.add(q_key)
                            # Extract kg_name from graph_uri if available
                            graph_uri = pair.get("graph_uri", "")
                            kg_name = graph_uri.split("/kg/")[-1] if "/kg/" in graph_uri else ""
                            if kg_name in HOLDOUT_V2_KGS:
                                logger.debug(
                                    "example_bank: skipping holdout-v2 KG %s from finetune pair",
                                    kg_name,
                                )
                                continue
                            items.append({
                                "question": question,
                                "sparql": sparql,
                                "kg_name": kg_name,
                                "ontology_context": pair.get("ontology", ""),
                            })
                        except json.JSONDecodeError:
                            continue
            except OSError as exc:
                logger.warning("Skipping finetune pairs: %s", exc)

        if not items:
            logger.info("No correct examples found in eval reports")
            return 0

        # Balance across KGs: cap per-KG to ensure representation from all datasets
        from collections import defaultdict
        by_kg: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            by_kg[item.get("kg_name", "")].append(item)

        num_kgs = max(len(by_kg), 1)
        per_kg_cap = MAX_BANK_SIZE // num_kgs
        balanced: list[dict] = []
        for kg, kg_items in by_kg.items():
            balanced.extend(kg_items[:per_kg_cap])

        # Fill remaining capacity with extras from any KG
        remaining = MAX_BANK_SIZE - len(balanced)
        if remaining > 0:
            extras = [item for item in items if item not in balanced]
            balanced.extend(extras[:remaining])

        logger.info("Found %d correct examples, balanced to %d across %d KGs", len(items), len(balanced), num_kgs)
        added = await self.add_batch(balanced)
        self.save()
        return added

    # ── Embedding API ────────────────────────────────────────────────────

    async def _embed_single(self, text: str) -> list[float]:
        """Embed a single text string."""
        results = await self._embed_texts([text])
        return results[0]

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Delegate to the shared embed client (kept as a method: test seam).

        Raises :class:`~cograph_client.nlp.embed_client.EmbeddingError` — a
        ``RuntimeError`` subclass, so historical ``except RuntimeError``
        callers are unaffected.
        """
        return await embed_texts(texts, api_key=self._api_key)


# ── Prompt formatting ────────────────────────────────────────────────────


def format_examples_for_prompt(examples: list[Example]) -> str:
    """Format retrieved examples into a string for injection into the SPARQL generation prompt.

    Output format:
        Similar queries that worked:

        Example 1 (count + join):
          Q: How many events are in the Mission District?
          SPARQL: SELECT (COUNT(DISTINCT ?event) AS ?count) FROM <graph> WHERE { ... }

        Example 2 (avg + filter):
          Q: What is the average price of condos?
          SPARQL: SELECT (AVG(?price) AS ?avg) FROM <graph> WHERE { ... }
    """
    if not examples:
        return ""

    lines = ["Similar queries that worked:"]

    for i, ex in enumerate(examples, 1):
        tag_str = " + ".join(ex.pattern_tags) if ex.pattern_tags else "basic"
        # Compact the SPARQL — collapse excessive whitespace but keep it readable
        sparql_compact = " ".join(ex.sparql.split())
        lines.append("")
        lines.append(f"Example {i} ({tag_str}):")
        lines.append(f"  Q: {ex.question}")
        lines.append(f"  SPARQL: {sparql_compact}")

    return "\n".join(lines)


# ── Singleton accessor ───────────────────────────────────────────────────

_example_bank: ExampleBank | None = None


def get_example_bank() -> ExampleBank | None:
    """Lazy-init singleton for the example bank. Returns None if no API key."""
    global _example_bank
    if _example_bank is None:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            try:
                from cograph_client.config import settings
                api_key = settings.openrouter_api_key or ""
            except Exception:
                pass
        if not api_key:
            return None
        _example_bank = ExampleBank(openrouter_api_key=api_key)
        _example_bank.load()
    return _example_bank
