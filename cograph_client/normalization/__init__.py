"""Inferred, human-confirmed data-normalization subsystem (OSS core).

Round 1: an agent infers per-predicate normalization rules (v1: list_explode —
splitting collapsed multi-value cells into atomic values), ranks them by
confidence, a human confirms, then the rule is applied to the KG. Auto-apply to
new inserts is a follow-up round.

Public surface:
- :mod:`rules` — :class:`NormalizationRule` + :class:`NormalizationRuleStore`
- :mod:`inference` — :func:`suggest_rules`
- :mod:`execute` — :func:`apply_rule`
"""
