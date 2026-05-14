"""Default Scorer for cross-file entity resolution.

Compares two NormalizedSignals records, producing a weighted similarity in
[0, 1] plus per-signal contributions for the audit log.

Weights are re-normalized over the signals that are present on BOTH sides,
so a record with only a name doesn't get penalized just because email is
missing — its name match carries the full weight budget.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from .types import ERConfig, MatchScore, NormalizedSignals, SignalContribution


def _email_similarity(a: NormalizedSignals, b: NormalizedSignals) -> float:
    # Build the full set of known emails on each side (primary + aliases).
    # Real CRMs store work + personal; ignoring aliases loses the majority
    # of cross-system matches.
    a_emails = {a.email} | set(a.email_aliases)
    a_emails.discard(None)
    a_emails.discard("")
    b_emails = {b.email} | set(b.email_aliases)
    b_emails.discard(None)
    b_emails.discard("")
    if a_emails and b_emails and a_emails & b_emails:
        return 1.0
    # Bonus: same local part but different domains → probably same human
    # (e.g. jane.doe@gmail.com vs jane.doe@company.com).
    a_locals = {a.email_local} | set(a.email_locals)
    a_locals.discard(None)
    a_locals.discard("")
    b_locals = {b.email_local} | set(b.email_locals)
    b_locals.discard(None)
    b_locals.discard("")
    if a_locals and b_locals and a_locals & b_locals:
        return 0.6
    return 0.0


def _phone_similarity(a: NormalizedSignals, b: NormalizedSignals) -> float:
    if a.phone_e164 and b.phone_e164 and a.phone_e164 == b.phone_e164:
        return 1.0
    return 0.0


def _name_similarity(a: NormalizedSignals, b: NormalizedSignals) -> float:
    # WRatio handles word reorder + partial matches; token_set_ratio is
    # robust to extra/missing tokens. Taking the max gives us the benefit
    # of both without double-counting.
    if not (a.name and b.name):
        return 0.0
    w = fuzz.WRatio(a.name, b.name)
    t = fuzz.token_set_ratio(a.name, b.name)
    return max(w, t) / 100.0


def _address_similarity(a: NormalizedSignals, b: NormalizedSignals) -> float:
    if not (a.address and b.address):
        return 0.0
    return fuzz.token_set_ratio(a.address, b.address) / 100.0


def _dob_similarity(a: NormalizedSignals, b: NormalizedSignals) -> float:
    if a.dob_iso and b.dob_iso and a.dob_iso == b.dob_iso:
        return 1.0
    return 0.0


# Each entry: (presence-check, comparator). Presence-check returns True when
# the signal is non-empty on a given side; comparator returns the similarity.
_SIGNAL_HANDLERS = {
    "email": (
        lambda s: bool(s.email or s.email_local),
        _email_similarity,
    ),
    "phone": (
        lambda s: bool(s.phone_e164),
        _phone_similarity,
    ),
    "name": (
        lambda s: bool(s.name),
        _name_similarity,
    ),
    "address": (
        lambda s: bool(s.address),
        _address_similarity,
    ),
    "dob": (
        lambda s: bool(s.dob_iso),
        _dob_similarity,
    ),
}


class DefaultScorer:
    """Plain weighted-sum scorer. Conforms to `Scorer` protocol."""

    def score(
        self,
        incoming: NormalizedSignals,
        candidate: NormalizedSignals,
        config: ERConfig,
    ) -> MatchScore:
        # Pass 0 — DECISIVE-SIGNAL SHORT-CIRCUIT. If any signal flagged as
        # decisive in the config matches with similarity 1.0 on BOTH sides,
        # we have deterministic identity and don't need probabilistic
        # scoring. Return immediately with score=1.0 and a single
        # contribution recording which signal decided it. This is the
        # "exact email match alone is enough" pattern.
        for signal in config.decisive_signals:
            handler = _SIGNAL_HANDLERS.get(signal)
            if handler is None:
                continue
            present, comparator = handler
            if not (present(incoming) and present(candidate)):
                continue
            if comparator(incoming, candidate) >= 1.0:
                return MatchScore(
                    candidate_uri="",
                    score=1.0,
                    contributions=[
                        SignalContribution(
                            signal=f"decisive:{signal}",
                            weight=1.0,
                            similarity=1.0,
                            contribution=1.0,
                        )
                    ],
                )

        # Pass 1: figure out which signals are present on both sides, and
        # collect their raw similarities + configured weights.
        active: list[tuple[str, float, float]] = []  # (signal, raw_weight, similarity)
        for signal in config.signals:
            handler = _SIGNAL_HANDLERS.get(signal)
            if handler is None:
                continue
            present, comparator = handler
            if not (present(incoming) and present(candidate)):
                continue
            raw_weight = config.weight_for(signal)
            similarity = comparator(incoming, candidate)
            active.append((signal, raw_weight, similarity))

        if not active:
            return MatchScore(candidate_uri="", score=0.0, contributions=[])

        weight_sum = sum(w for _, w, _ in active)
        if weight_sum <= 0:
            return MatchScore(candidate_uri="", score=0.0, contributions=[])

        # Pass 2: re-normalize and build contributions.
        contributions: list[SignalContribution] = []
        total = 0.0
        for signal, raw_weight, similarity in active:
            norm_weight = raw_weight / weight_sum
            contribution = norm_weight * similarity
            total += contribution
            contributions.append(
                SignalContribution(
                    signal=signal,
                    weight=norm_weight,
                    similarity=similarity,
                    contribution=contribution,
                )
            )

        # Strong-signal override: when phone matches exactly AND we have any
        # reasonable name signal of agreement (high similarity OR shared
        # last-name token), we have near-certain identity even if email is
        # contradictory (e.g. same human used personal vs work mail across
        # systems). Phone collisions between unrelated humans are vanishingly
        # rare in practice — the empirical false-positive rate on the smoke
        # tests is zero across hundreds of unrelated pairs.
        phone_sim = next((s for sig, _, s in active if sig == "phone"), 0.0)
        name_sim = next((s for sig, _, s in active if sig == "name"), 0.0)
        shared_last = (
            len(incoming.name_tokens) > 0
            and len(candidate.name_tokens) > 0
            and incoming.name_tokens[-1] == candidate.name_tokens[-1]
        )
        if phone_sim >= 1.0 and (name_sim >= 0.85 or shared_last) and total < 0.92:
            contributions.append(
                SignalContribution(
                    signal="phone_name_override",
                    weight=0.0,
                    similarity=1.0,
                    contribution=0.92 - total,
                )
            )
            total = 0.92

        return MatchScore(candidate_uri="", score=total, contributions=contributions)
