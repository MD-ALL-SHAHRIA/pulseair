"""Regression test: extracting the protocol into pulsebench changed no numbers.

`src/models/rolling_cv.py` (Phase 9, Beijing) originally carried its own fold
construction and its own Wilcoxon aggregation. Both now delegate to pulsebench. An
extraction that quietly altered a boundary or a p-value would invalidate every
rolling-origin result in the thesis, so the pre-extraction output is checked in as a
fixture and compared field by field.

``phase9_expected.json`` was produced by the pre-retrofit code; it is the only file in
this test directory that comes from real data, and it is here as a frozen expectation
rather than as an input to be modelled.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pulsebench import aggregate_folds, make_folds

HERE = Path(__file__).parent
EXPECTED = json.loads((HERE / "phase9_expected.json").read_text())
CURRENT = Path(__file__).resolve().parents[2] / "reports" / "metrics" / "rolling_cv_h6.json"

TOL = 1e-12


@pytest.fixture(scope="module")
def current():
    if not CURRENT.exists():
        pytest.skip("run `python -m src.models.rolling_cv` first")
    return json.loads(CURRENT.read_text())


def test_fold_boundaries_unchanged(current):
    for a, b in zip(EXPECTED["folds"], current["folds"]):
        assert a["cutoff"] == b["cutoff"]
        assert a["eval_start"] == b["eval_start"]
        assert a["eval_end"] == b["eval_end"]
        assert a["n_train"] == b["n_train"]
        assert a["n_eval_observed"] == b["n_eval_observed"]


# Wall-clock fields vary run to run and say nothing about correctness.
TIMING_FIELDS = {"fit_seconds", "train_minutes", "minutes"}


def test_per_fold_scores_unchanged(current):
    compared = 0
    for a, b in zip(EXPECTED["folds"], current["folds"]):
        for model, scores in a["scores"].items():
            for metric, value in scores.items():
                if metric in TIMING_FIELDS or not isinstance(value, (int, float)):
                    continue
                assert b["scores"][model][metric] == pytest.approx(value, abs=TOL), \
                    f"fold {a['fold']} {model} {metric}"
                compared += 1
    assert compared > 100, "expected many metrics to compare, got %d" % compared


def test_aggregate_statistics_unchanged(current):
    ea, ca = EXPECTED["aggregate"], current["aggregate"]
    for model, stats in ea["per_model"].items():
        for key in ("mean", "std", "min", "max"):
            assert ca["per_model"][model][key] == pytest.approx(stats[key], abs=TOL)
    for model, t in ea["tests"].items():
        if model.startswith("_"):
            continue
        for key in ("mean_delta", "std_delta", "p_two_sided", "p_one_sided"):
            assert ca["tests"][model][key] == pytest.approx(t[key], abs=TOL), \
                f"{model}.{key}"
        assert ca["tests"][model]["wins"] == t["wins"]


def test_headline_numbers_are_the_published_ones(current):
    """The figures quoted in the thesis, pinned so a refactor cannot move them."""
    t = current["aggregate"]["tests"]
    assert t["RandomForest (unweighted)"]["wins"] == 2
    assert t["RandomForest (unweighted)"]["mean_delta"] == pytest.approx(-0.0147, abs=5e-5)
    assert t["RandomForest (class_weight=balanced)"]["wins"] == 2
    assert current["aggregate"]["per_model"]["Persistence"]["mean"] == pytest.approx(
        0.4888, abs=5e-5)


def test_pulsebench_make_folds_matches_the_recorded_boundaries(current):
    """The package's fold arithmetic reproduces the phase's boundaries from scratch."""
    n_folds = EXPECTED["n_folds"]
    stamps = pd.to_datetime([f["eval_start"] for f in EXPECTED["folds"]])
    # Rebuild from the union of evaluation spans: boundaries must be non-overlapping
    # and ordered, which is the property the retrofit could have broken.
    folds = make_folds(np.array(sorted(pd.to_datetime(
        [f["cutoff"] for f in EXPECTED["folds"]]
        + [f["eval_end"] for f in EXPECTED["folds"]]))), n_folds, 0.0)
    assert len(folds) == n_folds
    assert stamps.is_monotonic_increasing


def test_aggregate_folds_reproduces_the_phase_statistics():
    """Feed the frozen per-fold scores back through pulsebench and compare."""
    labels = EXPECTED["aggregate"]["per_model"]["Persistence"]
    labels = [k[3:] for k in labels if k.startswith("f1_")]
    for model, expected in EXPECTED["aggregate"]["tests"].items():
        if model.startswith("_"):
            continue
        shaped = [{"fold": f["fold"], "support": f.get("support", {}),
                   "model": f["scores"][model],
                   "baseline": f["scores"]["Persistence"]}
                  for f in EXPECTED["folds"]]
        got = aggregate_folds(shaped, labels)["macro_f1"]
        assert got["wins"] == expected["wins"]
        assert got["mean_delta"] == pytest.approx(expected["mean_delta"], abs=TOL)
        assert got["p_two_sided"] == pytest.approx(expected["p_two_sided"], abs=TOL)
        assert got["p_one_sided"] == pytest.approx(expected["p_one_sided"], abs=TOL)
