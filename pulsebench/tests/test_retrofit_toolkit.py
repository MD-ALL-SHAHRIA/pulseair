"""Regression test: completing the pulsebench retrofit changed no numbers.

Three call sites in ``src/`` carried their own copy of a rule that pulsebench also
implements — the persistence floor in ``dhaka_ground_truth.py``, the Bonferroni
correction in ``compile_results.py`` and ``generate_figures.py``, and the
disqualification rule in ``gan/ablation.py``. All now call the package. That is only
worth doing if the package computes the same thing, so the pre-retrofit output of each
duplicate is frozen in ``retrofit_toolkit_expected.json`` and compared here.

Same purpose as ``test_retrofit_phase9.py``, which pins the rolling-origin extraction.
Together they mean every pulsebench function the project relies on is checked against a
number the project actually published, rather than against synthetic data alone.

One difference was found rather than assumed, and is asserted as a difference: see
``test_bonferroni_display_difference_is_the_known_one``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pulsebench import advisory_disqualification, bonferroni_report, persistence_floor

HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[2]
EXPECTED = json.loads((HERE / "retrofit_toolkit_expected.json").read_text())
METRICS = ROOT / "reports" / "metrics"


def _metrics(name: str) -> dict:
    path = METRICS / name
    if not path.exists():
        pytest.skip(f"{name} not present — run the phase that writes it first")
    return json.loads(path.read_text())


# ------------------------------------------------- persistence floor (Phase 11)


@pytest.fixture(scope="module")
def dhaka_floor():
    """The retrofitted function, run on the real Embassy series."""
    raw_dir = ROOT / "data" / "raw" / "dhaka_embassy"
    if not raw_dir.exists():
        pytest.skip("data/raw/dhaka_embassy not present")
    import yaml
    from src.models import dhaka_ground_truth as gt

    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    emb, _ = gt.load_embassy()
    return gt.persistence_floor(emb,
                                list(cfg["data"]["pm25_breakpoints"]),
                                list(cfg["data"]["pm25_labels"]),
                                int(cfg["preprocessing"]["horizon"]))


def test_dhaka_persistence_floor_is_unchanged_by_the_retrofit(dhaka_floor):
    want = EXPECTED["dhaka_persistence_floor"]
    for field in ("n_pairs", "macro_f1", "accuracy", "label_unchanged_pct"):
        assert dhaka_floor[field] == want[field], field


def test_dhaka_per_class_values_are_unchanged(dhaka_floor):
    want = EXPECTED["dhaka_persistence_floor"]
    assert dhaka_floor["per_class_f1"] == want["per_class_f1"]
    assert dhaka_floor["support"] == want["support"]


def test_dhaka_floor_still_matches_the_published_metrics_file(dhaka_floor):
    """The committed JSON, the fixture and the live function must all agree."""
    published = _metrics("dhaka_ground_truth.json")["persistence"]
    for field in ("n_pairs", "macro_f1", "accuracy", "label_unchanged_pct",
                  "per_class_f1", "support"):
        assert dhaka_floor[field] == published[field], field


def test_persistence_floor_keeps_absent_classes_in_support():
    """Passing explicit labels must keep a zero-support class in the output.

    The Dhaka retrofit relies on this: per-class tables downstream index every label,
    so a class the series never reaches has to appear with a zero rather than vanish.
    """
    idx = pd.date_range("2024-01-01", periods=50, freq="h")
    d = pd.DataFrame({"y": [0, 1] * 25}, index=idx)
    out = persistence_floor(d, "y", horizon=1, labels=[0, 1, 2])
    assert out["support"][2] == 0
    assert set(out["support"]) == {0, 1, 2}


# --------------------------------------------------- Bonferroni (sections 6 + fig 12)


@pytest.fixture(scope="module")
def family():
    from src.reporting.compile_results import collect
    d = collect()
    tested = [v for v in d["variants"] if v["vs_persistence"] not in (None, "untested")]
    return bonferroni_report(
        {v["name"].split(" —")[0]: {"p": v["vs_persistence"]["p_two_sided"],
                                    "delta": v["vs_persistence"]["observed_diff"]}
         for v in tested},
        alpha=0.05, n_resamples=d["n_boot"])


def test_bonferroni_family_size_and_threshold_unchanged(family):
    want = EXPECTED["bonferroni_family"]
    assert family["k"] == want["k"]
    assert family["alpha"] == want["alpha"]
    assert family["corrected_alpha"] == want["corrected_alpha"]
    assert family["resolution_floor"] == want["resolution_floor"]


def test_bonferroni_survivors_and_losses_unchanged(family):
    want = EXPECTED["bonferroni_family"]
    assert sorted(family["survivors"]) == sorted(want["survivors"])
    assert sorted(family["lost"]) == sorted(want["lost"])


def test_bonferroni_display_difference_is_the_known_one(family):
    """pulsebench marks a p *at* the resolution floor as '<floor'; the inline code did not.

    Asserted rather than silently accepted. A 1,000-resample bootstrap cannot resolve a
    two-sided p below 2/1000, so rendering an exact 0.0020 overstates what the procedure
    can produce. This is the one behavioural difference the retrofit introduced, it is
    display-only, and if it ever grows to touch a verdict the tests above will fail first.
    """
    at_floor = [r for r in family["rows"] if r["at_resolution_floor"]]
    assert at_floor, "expected at least one p at the bootstrap resolution floor"
    for r in at_floor:
        assert r["p"] <= family["resolution_floor"]
        assert r["p_display"] == f"<{family['resolution_floor']:.4f}"


# -------------------------------------------------- disqualification (Phase 4)


def test_disqualification_verdicts_unchanged():
    ab = _metrics("ablation_h6.json")
    want = EXPECTED["disqualification"]
    tests = ab["tests"]
    for variant in ("broad-4", "targeted-2"):
        got = advisory_disqualification(
            {m: {"delta": tests[variant][m]["observed_diff"],
                 "significant": tests[variant][m]["significant"]}
             for m in tests[variant] if m != "macro_f1"},
            protected_classes=want["protected_classes"],
            practical_threshold=want["material_threshold"],
            aggregate_delta=tests[variant]["macro_f1"]["observed_diff"],
            aggregate_significant=tests[variant]["macro_f1"]["significant"])
        exp = want[variant]
        assert got["disqualified"] == exp["disqualified"], variant
        assert sorted(got["harmed"]) == sorted(exp["harmed"]), variant
        assert (got["verdict"] == "accepted") == exp["accepted"], variant


def test_every_pulsebench_export_the_project_relies_on_is_exercised_here():
    """The point of the retrofit, stated as a test.

    Before it, four of pulsebench's headline functions were checked only against
    synthetic data while ``src/`` ran its own copies. Each one named below is now
    covered by a comparison against a number this project published.
    """
    import pulsebench
    covered = {"persistence_floor", "bonferroni_report", "advisory_disqualification"}
    here = (HERE / "test_retrofit_toolkit.py").read_text()
    phase9 = (HERE / "test_retrofit_phase9.py").read_text()
    for name in covered:
        assert name in here, name
    for name in ("make_folds", "aggregate_folds"):
        assert name in phase9, name
    assert covered <= set(pulsebench.__all__)
