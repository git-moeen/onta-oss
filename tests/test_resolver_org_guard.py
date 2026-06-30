"""ONTA-155 — the extraction prompt must guard against over-minting Organizations
from the data source / benchmark name itself or from baseline/placeholder values.

This is a *prompt* guard: its real effect is on live LLM extraction, which is
non-deterministic and can't be unit-asserted here. What we CAN lock down
deterministically is that the guard's intent stays present in
``EXTRACTION_SYSTEM`` — so a future edit can't silently drop it and regress the
Humanness-Index-style failure (the benchmark name 'Humanness Index' and the
baseline 'Human' getting minted as Organizations, with the publisher split
across duplicate orgs).

The asserts are intentionally concept-level (not exact-string) so benign
rewording of the prompt doesn't break them, but removing the guard does.
"""

from __future__ import annotations

from cograph_client.resolver.schema_resolver import EXTRACTION_SYSTEM


def _lift_block() -> str:
    """The 'Lift providers / organizations' section, lowercased."""
    text = EXTRACTION_SYSTEM
    start = text.index("Lift providers / organizations")
    # Up to the next titled section (each ends with a trailing ':').
    end = text.index("Subtypes with a description", start)
    return text[start:end].lower()


def test_guard_present_source_or_benchmark_name_is_not_an_org():
    """The guard must tell the extractor that the data source / benchmark /
    dataset name itself is NOT an Organization (the operator is)."""
    block = _lift_block()
    # Names the artifact-kinds that must not be lifted.
    assert any(k in block for k in ("benchmark", "leaderboard", "dataset", "index", "publication")), (
        "org-lift guard must name the source/benchmark/dataset artifact"
    )
    # Says it is NOT an actor / the publisher is the operating company.
    assert "operates" in block or "operating" in block, (
        "guard must attribute publication to the company that OPERATES the source, "
        "not to the source/benchmark name itself"
    )
    assert "itself" in block, "guard must call out the source name ITSELF as not-an-org"


def test_guard_present_placeholders_are_not_minted():
    """The guard must tell the extractor not to mint baseline / placeholder /
    null-like values ('Human', 'Unknown', 'N/A', '-', …) as entities."""
    block = _lift_block()
    assert "placeholder" in block or "baseline" in block, (
        "guard must mention baseline/placeholder values"
    )
    # A representative null-like token list is present.
    assert any(tok in block for tok in ("unknown", "n/a", "none")), (
        "guard should enumerate null-like tokens to omit"
    )


def test_guard_says_omit_not_invent():
    """When only the dataset name / a placeholder is available, the guard must say
    to OMIT the organization rather than invent one."""
    block = _lift_block()
    assert "omit" in block, "guard must instruct to OMIT rather than invent a spurious org"
