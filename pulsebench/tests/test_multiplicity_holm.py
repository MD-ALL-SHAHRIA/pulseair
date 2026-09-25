"""Holm-Bonferroni: the properties that make it safe to offer alongside Bonferroni.

Issue #3 asked for Holm as an option, not a replacement. The tests below encode the
three things that has to mean: the default must not move, Holm must never be less
powerful than Bonferroni, and the resampling resolution floor must survive either
choice — a correction changes which claims stand, not what the test could resolve.
"""

from __future__ import annotations

import random

import pytest

from pulsebench import bonferroni_report, format_markdown


def _family(ps):
    return {chr(ord("A") + i): {"p": p, "delta": 0.01 * (i + 1)}
            for i, p in enumerate(ps)}


# ------------------------------------------------- the default must not move


def test_default_is_bonferroni_and_unchanged():
    fam = _family([0.001, 0.004, 0.02, 0.3])
    assert bonferroni_report(fam)["method"] == "bonferroni"
    assert (bonferroni_report(fam)["survivors"]
            == bonferroni_report(fam, method="bonferroni")["survivors"])


def test_default_matches_the_hand_computed_threshold():
    """alpha/k, applied to every comparison regardless of rank."""
    fam = _family([0.001, 0.004, 0.02, 0.3])
    r = bonferroni_report(fam, alpha=0.05)
    assert r["corrected_alpha"] == pytest.approx(0.05 / 4)
    assert r["survivors"] == [n for n, v in fam.items() if v["p"] < 0.05 / 4]


# ------------------------------------------------------ the ordering property


def test_holm_never_rejects_fewer_than_bonferroni():
    """The property that makes Holm strictly preferable on power grounds.

    Checked over random families rather than one example, because the claim is a
    universal one and a single case would not establish it.
    """
    rng = random.Random(0)
    for _ in range(400):
        k = rng.randint(1, 12)
        fam = _family([rng.choice([rng.uniform(0, 0.06), rng.uniform(0, 1)])
                       for _ in range(k)])
        b = set(bonferroni_report(fam)["survivors"])
        h = set(bonferroni_report(fam, method="holm")["survivors"])
        assert b <= h, f"Bonferroni rejected something Holm did not: {fam}"


def test_the_two_agree_when_only_the_smallest_p_is_significant():
    """If exactly one hypothesis clears alpha/k, both procedures reject exactly it.

    Holm's first threshold *is* alpha/k, so the smallest p is tested identically;
    the step-down then stops at the second comparison.
    """
    fam = _family([0.001, 0.40, 0.55, 0.90])
    b = bonferroni_report(fam)
    h = bonferroni_report(fam, method="holm")
    assert b["survivors"] == h["survivors"] == ["A"]


def test_holm_rejects_strictly_more_on_a_family_designed_for_it():
    fam = {"A": {"p": 0.004}, "B": {"p": 0.02}, "C": {"p": 0.9}}
    assert bonferroni_report(fam)["survivors"] == ["A"]
    assert bonferroni_report(fam, method="holm")["survivors"] == ["A", "B"]


def test_holm_stops_at_the_first_failure():
    """A later p that would pass its own threshold is still not rejected.

    This is the step-down rule, and skipping it would break error control.
    """
    # k=4: thresholds are 0.0125, 0.0167, 0.025, 0.05.
    # B fails its threshold, so D must not be rejected even though 0.03 < 0.05.
    fam = {"A": {"p": 0.001}, "B": {"p": 0.9}, "C": {"p": 0.95}, "D": {"p": 0.03}}
    h = bonferroni_report(fam, method="holm")
    assert h["survivors"] == ["A"]
    assert "D" not in h["survivors"]


# -------------------------------------------------- the floor is method-agnostic


def test_resolution_floor_is_reported_under_both_methods():
    fam = _family([0.0, 0.002, 0.3])
    for method in ("bonferroni", "holm"):
        r = bonferroni_report(fam, n_resamples=1000, method=method)
        assert r["resolution_floor"] == pytest.approx(0.002)
        at_floor = [row for row in r["rows"] if row["at_resolution_floor"]]
        assert at_floor, method
        for row in at_floor:
            assert row["p_display"] == "<0.0020"


def test_signed_rank_floor_is_unchanged_by_holm():
    """2 ** (1 - n) is a property of the test, not of the correction."""
    n_folds = 5
    floor = 2.0 ** (1 - n_folds)
    fam = _family([floor, 0.2, 0.5])
    for method in ("bonferroni", "holm"):
        r = bonferroni_report(fam, n_resamples=int(2 / floor), method=method)
        assert r["resolution_floor"] == pytest.approx(floor)


# ------------------------------------------------------------------ rendering


def test_format_markdown_names_the_method():
    fam = _family([0.001, 0.02, 0.4])
    assert "Bonferroni" in format_markdown(bonferroni_report(fam))
    holm_md = format_markdown(bonferroni_report(fam, method="holm"))
    assert "Holm" in holm_md
    assert "Holm threshold" in holm_md


def test_format_markdown_states_the_floor_when_there_is_one():
    md = format_markdown(bonferroni_report(_family([0.001]), n_resamples=1000))
    assert "cannot resolve a two-sided p below 0.0020" in md
    assert "unaffected by the choice of correction" in md


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="bonferroni.*holm"):
        bonferroni_report(_family([0.01]), method="benjamini-hochberg")


def test_summary_names_the_method():
    assert "Bonferroni" in bonferroni_report(_family([0.001, 0.4]))["summary"]
    assert "Holm" in bonferroni_report(_family([0.001, 0.4]),
                                       method="holm")["summary"]
