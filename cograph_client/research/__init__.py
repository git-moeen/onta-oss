"""Web-research harness — the staged agent-loop that answers a natural-language
question from the open web and returns a **cited answer / artifact** (ADR 0006).

This is the read-only counterpart to web *discovery* ingest. Discovery
(:mod:`cograph_client.web_sources`, :class:`WebIngestCapability`) turns a query
into rows and WRITES them into a knowledge graph. The research harness answers a
question — "list every TTS model on the VAPI Humanness Index with its score" —
by **planning** a target schema, **discovering** sources, **fetching** them via a
cheap→expensive ladder, **extracting** schema-valid rows, **verifying** them
(cite-or-abstain), **synthesizing** an answer + artifact (CSV/JSON) with
citations, and **reflecting** to close gaps — bounded by a per-request budget. It
does NOT write to the graph.

The design is provider-portable: discovery reuses the existing
:class:`~cograph_client.web_sources.base.WebSourceProvider` seam; the fetch
ladder and the verifier are their own OSS protocols with default OSS
implementations (a static HTTP fetcher, a deterministic cite-or-abstain
verifier) so the package works standalone. Premium tiers — a JS-render fetcher,
an LLM-judge verifier, semantic-escalation discovery — plug in through the same
``register_*`` hooks, key-gated and dormant without their keys.

Boundary: OSS. Every module here imports only stdlib / ``cograph_client.*`` /
``httpx``. No ``from cograph.*`` and no proprietary identifiers.
"""

from __future__ import annotations

from cograph_client.research.fetch import (
    FetchedPage,
    PageFetcher,
    StaticHttpFetcher,
    fetcher_cost,
    get_page_fetchers,
    register_page_fetcher,
    reset_page_fetchers,
)
from cograph_client.research.types import (
    Budget,
    Citation,
    ClarifyingQuestion,
    ResearchPlan,
    ResearchResult,
    ResearchRow,
    SchemaField,
    TargetSchema,
    normalize_clarifying_questions,
)
from cograph_client.research.verify import (
    CiteOrAbstainVerifier,
    ResearchVerifier,
    VerifyOutcome,
    get_research_verifier,
    register_research_verifier,
    reset_research_verifier,
)

__all__ = [
    "Budget",
    "Citation",
    "CiteOrAbstainVerifier",
    "ClarifyingQuestion",
    "FetchedPage",
    "PageFetcher",
    "ResearchPlan",
    "ResearchResult",
    "ResearchRow",
    "ResearchVerifier",
    "SchemaField",
    "StaticHttpFetcher",
    "TargetSchema",
    "VerifyOutcome",
    "fetcher_cost",
    "get_page_fetchers",
    "get_research_verifier",
    "normalize_clarifying_questions",
    "register_page_fetcher",
    "register_research_verifier",
    "reset_page_fetchers",
    "reset_research_verifier",
]
