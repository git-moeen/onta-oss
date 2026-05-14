"""Cross-file entity resolution for Cograph.

Auto-fires during ingest at the URI-minting step in `schema_resolver.py`.
For ER-enabled types (Guest, Customer, Contact, Property, ...), incoming
entities are matched against existing entities in the tenant's graph via:

    extract_signals → normalize → block → score → decide

If a match is found at >= auto_merge_threshold, the incoming entity is
rerouted to the canonical URI (all its triples flow into the existing
entity). If the score is in the review band, a MergeReview is enqueued.
Otherwise the entity is minted as new.

See docs/specs/entity_resolution_spec.md for the full design.
"""

from cograph_client.resolver.er.engine import ERPipeline, extract_signals
from cograph_client.resolver.er.types import (
    DEFAULTS_BY_TYPE,
    BlockKey,
    ERConfig,
    EntitySignals,
    MatchScore,
    MergeAction,
    MergeDecision,
    NormalizedSignals,
    SignalContribution,
    config_for,
)

__all__ = [
    "DEFAULTS_BY_TYPE",
    "BlockKey",
    "ERConfig",
    "ERPipeline",
    "EntitySignals",
    "MatchScore",
    "MergeAction",
    "MergeDecision",
    "NormalizedSignals",
    "SignalContribution",
    "config_for",
    "extract_signals",
]
