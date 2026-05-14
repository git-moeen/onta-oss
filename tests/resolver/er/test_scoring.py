"""Tests for DefaultScorer."""

from __future__ import annotations

import pytest

from cograph_client.resolver.er.scoring import DefaultScorer
from cograph_client.resolver.er.types import (
    DEFAULT_GUEST_CONFIG,
    ERConfig,
    NormalizedSignals,
)


@pytest.fixture
def scorer() -> DefaultScorer:
    return DefaultScorer()


def _ns(**kwargs) -> NormalizedSignals:
    return NormalizedSignals(**kwargs)


def test_perfect_match_all_signals(scorer: DefaultScorer) -> None:
    a = _ns(
        name="jane doe",
        email="jane@x.com",
        email_local="jane",
        phone_e164="+14155550101",
        address="100 main st sf ca",
        dob_iso="1990-01-01",
    )
    result = scorer.score(a, a, DEFAULT_GUEST_CONFIG)
    assert result.score == pytest.approx(1.0)
    # With email marked decisive in DEFAULT_GUEST_CONFIG, an exact email
    # match short-circuits to score=1.0 with a single decisive contribution.
    assert len(result.contributions) == 1
    assert result.contributions[0].signal == "decisive:email"


def test_perfect_match_no_decisive_signals(scorer: DefaultScorer) -> None:
    """When the config has no decisive signals, full weighted-sum runs."""
    cfg_no_decisive = ERConfig(
        type_name="Guest",
        signals=("email", "phone", "name", "address", "dob"),
        weights=(0.45, 0.25, 0.20, 0.05, 0.05),
        decisive_signals=(),
    )
    a = _ns(
        name="jane doe",
        email="jane@x.com",
        email_local="jane",
        phone_e164="+14155550101",
        address="100 main st sf ca",
        dob_iso="1990-01-01",
    )
    result = scorer.score(a, a, cfg_no_decisive)
    assert result.score == pytest.approx(1.0)
    assert len(result.contributions) == 5


def test_decisive_email_overrides_contradicting_signals(scorer: DefaultScorer) -> None:
    """Email exact match alone is enough — even if phone, name, address differ."""
    a = _ns(
        name="jane doe",
        email="shared@example.com",
        email_local="shared",
        phone_e164="+14155550101",
        address="100 main st sf",
    )
    b = _ns(
        name="janet smith",           # totally different name
        email="shared@example.com",   # same email
        email_local="shared",
        phone_e164="+12025559999",    # different phone
        address="500 oak st chicago", # different address
    )
    result = scorer.score(a, b, DEFAULT_GUEST_CONFIG)
    assert result.score == pytest.approx(1.0)
    assert result.contributions[0].signal == "decisive:email"


def test_decisive_skipped_when_signal_missing(scorer: DefaultScorer) -> None:
    """If a decisive signal is missing on either side, fall through to weighted sum."""
    a = _ns(name="jane doe", phone_e164="+14155550101")  # no email
    b = _ns(name="jane doe", phone_e164="+14155550101")
    result = scorer.score(a, b, DEFAULT_GUEST_CONFIG)
    # Email isn't present, decisive pass skips. Weighted sum runs on phone+name.
    assert result.score == pytest.approx(1.0)
    # Multiple contributions (not the single decisive one)
    assert all(not c.signal.startswith("decisive:") for c in result.contributions)


def test_decisive_email_local_bonus_does_not_trigger_shortcut(scorer: DefaultScorer) -> None:
    """The email_local bonus path (similarity=0.6) is NOT a decisive match.
    Decisive requires similarity >= 1.0 — i.e., full email equality."""
    a = _ns(email="jane.doe@gmail.com", email_local="janedoe", name="jane doe")
    b = _ns(email="jane.doe@work.com", email_local="janedoe", name="jane doe")
    result = scorer.score(a, b, DEFAULT_GUEST_CONFIG)
    # Should NOT short-circuit; weighted sum runs (no "decisive:" contribution).
    assert all(not c.signal.startswith("decisive:") for c in result.contributions)
    # Email similarity is 0.6 (local match, different domain bonus), name is 1.0.
    email_contrib = next(c for c in result.contributions if c.signal == "email")
    assert email_contrib.similarity == pytest.approx(0.6)


def test_email_mismatch_different_local(scorer: DefaultScorer) -> None:
    a = _ns(email="alice@x.com", email_local="alice")
    b = _ns(email="bob@x.com", email_local="bob")
    cfg = ERConfig(type_name="T", signals=("email",), weights=(1.0,))
    result = scorer.score(a, b, cfg)
    assert result.score == 0.0
    assert result.contributions[0].signal == "email"
    assert result.contributions[0].similarity == 0.0


def test_email_local_match_domain_differs_gives_bonus(scorer: DefaultScorer) -> None:
    a = _ns(email="jane.doe@gmail.com", email_local="jane.doe")
    b = _ns(email="jane.doe@work.com", email_local="jane.doe")
    cfg = ERConfig(type_name="T", signals=("email",), weights=(1.0,))
    result = scorer.score(a, b, cfg)
    assert result.score == pytest.approx(0.6)
    assert result.contributions[0].similarity == pytest.approx(0.6)


def test_name_typo_high_similarity(scorer: DefaultScorer) -> None:
    a = _ns(name="john smith")
    b = _ns(name="jon smith")
    cfg = ERConfig(type_name="T", signals=("name",), weights=(1.0,))
    result = scorer.score(a, b, cfg)
    # rapidfuzz handles single-char typos easily; expect well above 0.85.
    assert 0.85 <= result.score <= 1.0
    # Spot check: "approximately 0.9".
    assert result.score == pytest.approx(0.9, abs=0.1)


def test_sparse_record_renormalizes_weights(scorer: DefaultScorer) -> None:
    # Only name on both sides; full Guest config has 5 signals.
    a = _ns(name="jane doe")
    b = _ns(name="jane doe")
    result = scorer.score(a, b, DEFAULT_GUEST_CONFIG)
    assert result.score == pytest.approx(1.0)
    assert len(result.contributions) == 1
    assert result.contributions[0].signal == "name"
    # Re-normalized to 1.0 since it's the only present signal.
    assert result.contributions[0].weight == pytest.approx(1.0)


def test_empty_incoming(scorer: DefaultScorer) -> None:
    a = _ns()
    b = _ns(name="jane doe", email="j@x.com", email_local="j")
    result = scorer.score(a, b, DEFAULT_GUEST_CONFIG)
    assert result.score == 0.0
    assert result.contributions == []


def test_empty_candidate(scorer: DefaultScorer) -> None:
    a = _ns(name="jane doe", email="j@x.com", email_local="j")
    b = _ns()
    result = scorer.score(a, b, DEFAULT_GUEST_CONFIG)
    assert result.score == 0.0
    assert result.contributions == []


def test_realistic_guest_pair(scorer: DefaultScorer) -> None:
    # Two records of "Jane Doe" — same person, slight noise.
    incoming = _ns(
        name="jane doe",
        email="jane.doe@gmail.com",
        email_local="jane.doe",
        phone_e164="+14155550101",
        address="100 main st sf ca",
        dob_iso="1990-01-01",
    )
    candidate = _ns(
        name="jane d doe",
        email="jane.doe@work.com",  # different domain (bonus path: 0.6)
        email_local="jane.doe",
        phone_e164="+14155550101",
        address="100 main street san francisco ca",
        dob_iso="1990-01-01",
    )
    result = scorer.score(incoming, candidate, DEFAULT_GUEST_CONFIG)
    # Email contributes 0.6 (bonus), phone/dob 1.0, name+address high.
    # Phone-exact + strong-name triggers the strong-signal override → score == 0.92.
    assert result.score == pytest.approx(0.92)
    # 5 real signal contributions + 1 override marker
    real_contribs = [c for c in result.contributions if c.signal != "phone_name_override"]
    assert len(real_contribs) == 5
    email_contrib = next(c for c in real_contribs if c.signal == "email")
    assert email_contrib.similarity == pytest.approx(0.6)


def test_weights_renormalize_partial(scorer: DefaultScorer) -> None:
    # email + phone only — both should split the weight budget proportionally
    # to the original Guest weights (0.45 / 0.25 → ~0.643 / ~0.357).
    # Use a config WITHOUT decisive_signals so the weighted-sum path runs
    # (DEFAULT_GUEST_CONFIG has email decisive, which would short-circuit
    # when emails match exactly).
    cfg = ERConfig(
        type_name="Guest",
        signals=("email", "phone", "name", "address", "dob"),
        weights=(0.45, 0.25, 0.20, 0.05, 0.05),
        decisive_signals=(),
    )
    a = _ns(email="j@x.com", email_local="j", phone_e164="+14155550101")
    b = _ns(email="j@x.com", email_local="j", phone_e164="+14155550101")
    result = scorer.score(a, b, cfg)
    assert result.score == pytest.approx(1.0)
    assert sum(c.weight for c in result.contributions) == pytest.approx(1.0)
    email_w = next(c.weight for c in result.contributions if c.signal == "email")
    phone_w = next(c.weight for c in result.contributions if c.signal == "phone")
    assert email_w == pytest.approx(0.45 / 0.70)
    assert phone_w == pytest.approx(0.25 / 0.70)
