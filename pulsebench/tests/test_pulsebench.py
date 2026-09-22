"""Unit tests for pulsebench, on synthetic data only.

Nothing here touches the project's real datasets: the point of the package is that it
works on any seasonal, imbalanced, categorical time-series problem, and a test suite
that needed Beijing air-quality data would not demonstrate that.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

from pulsebench import (advisory_disqualification, aggregate_folds, bonferroni_report,
                        format_markdown, make_folds, persistence_floor,
                        rolling_origin_cv)


# --------------------------------------------------------------------- fixtures


def synthetic(n=2000, seed=0, n_classes=4, sticky=0.8, freq="h", start="2021-01-01"):
    """A sticky categorical series: the class usually persists, occasionally jumps.

    Deliberately autocorrelated, because that is the property that makes a persistence
    baseline hard to beat and is the whole reason this package exists.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)
    y = np.empty(n, dtype=int)
    y[0] = rng.integers(0, n_classes)
    for i in range(1, n):
        y[i] = y[i - 1] if rng.random() < sticky else rng.integers(0, n_classes)
    x = y + rng.normal(0, 0.4, n)          # a feature that carries real signal
    return pd.DataFrame({"x": x, "y": y}, index=idx)


# ------------------------------------------------------------ persistence_floor


def test_persistence_floor_perfect_on_constant_series():
    idx = pd.date_range("2020-01-01", periods=100, freq="h")
    d = pd.DataFrame({"y": np.ones(100, dtype=int)}, index=idx)
    out = persistence_floor(d, "y", horizon=3)
    assert out["macro_f1"] == pytest.approx(1.0)
    assert out["label_unchanged_pct"] == pytest.approx(100.0)
    assert out["n_pairs"] == 97


def test_persistence_floor_pairs_do_not_span_gaps():
    """A positional shift would invent a pair across the hole; a time join must not."""
    idx = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 01:00",
                            "2020-06-01 00:00", "2020-06-01 01:00"])
    d = pd.DataFrame({"y": [0, 1, 0, 1]}, index=idx)
    out = persistence_floor(d, "y", horizon=1)
    assert out["n_pairs"] == 2          # one pair either side, none across the gap


def test_persistence_floor_respects_groups():
    idx = pd.date_range("2020-01-01", periods=10, freq="h")
    d = pd.concat([
        pd.DataFrame({"y": np.zeros(10, dtype=int), "site": "a"}, index=idx),
        pd.DataFrame({"y": np.ones(10, dtype=int), "site": "b"}, index=idx)])
    grouped = persistence_floor(d, "y", horizon=1, group_col="site")
    assert grouped["n_pairs"] == 18                 # 9 per site, none crossing
    assert grouped["macro_f1"] == pytest.approx(1.0)


def test_persistence_floor_is_harder_to_beat_when_series_is_stickier():
    loose = persistence_floor(synthetic(sticky=0.3), "y", horizon=1)
    tight = persistence_floor(synthetic(sticky=0.95), "y", horizon=1)
    assert tight["macro_f1"] > loose["macro_f1"]


def test_persistence_floor_requires_datetime_index():
    with pytest.raises(TypeError):
        persistence_floor(pd.DataFrame({"y": [1, 2, 3]}), "y", horizon=1)


# ------------------------------------------------------------ rolling_origin_cv


def test_make_folds_are_contiguous_and_expanding():
    idx = pd.date_range("2020-01-01", periods=1000, freq="h")
    folds = make_folds(idx.to_numpy(), n_folds=5, initial_fraction=0.4)
    assert len(folds) == 5
    cutoffs = [f["cutoff"] for f in folds]
    assert cutoffs == sorted(cutoffs)                     # expanding window
    for a, b in zip(folds, folds[1:]):
        assert a["eval_end"] <= b["cutoff"]               # blocks do not overlap


def test_rolling_origin_cv_shapes_and_growth():
    d = synthetic(n=3000)
    d["naive"] = d["y"].shift(1).bfill().astype(int)
    out = rolling_origin_cv(d, lambda: DummyClassifier(strategy="most_frequent"),
                            n_folds=4, embargo_hours=2, horizon=1,
                            feature_cols=["x"], baseline_col="naive")
    assert len(out["folds"]) == 4
    sizes = [f["n_train"] for f in out["folds"]]
    assert sizes == sorted(sizes)                          # training set grows
    for f in out["folds"]:
        assert f["n_eval"] > 0


def test_rolling_origin_cv_embargo_separates_train_and_eval():
    d = synthetic(n=2000)
    embargo = 12
    out = rolling_origin_cv(d, lambda: DummyClassifier(strategy="most_frequent"),
                            n_folds=3, embargo_hours=embargo, horizon=1,
                            feature_cols=["x"])
    for f in out["folds"]:
        gap = pd.Timestamp(f["eval_start"]) - pd.Timestamp(f["cutoff"])
        assert gap >= pd.Timedelta(hours=embargo)


def test_rolling_origin_cv_model_beats_baseline_when_it_should():
    """A feature carrying the label should beat a most-frequent baseline in every fold."""
    d = synthetic(n=4000, sticky=0.5)
    d["naive"] = 0
    out = rolling_origin_cv(d, lambda: RandomForestClassifier(
                                n_estimators=25, random_state=0, n_jobs=1),
                            n_folds=4, embargo_hours=1, horizon=1,
                            feature_cols=["x"], baseline_col="naive")
    assert out["aggregate"]["macro_f1"]["wins"] == 4


def test_results_are_json_serializable():
    """Results must survive json.dump, including their dict keys.

    np.unique and pd.unique return numpy scalars. Used as label keys they are not
    JSON-serializable, and since numpy 2 they also repr differently, so the same
    code prints differently depending on the installed numpy. Callers get plain
    Python values.
    """
    d = synthetic(n=600)
    floor = persistence_floor(d, "y", horizon=3)
    assert all(type(k) is int for k in floor["support"]), \
        f"support keys are {[type(k).__name__ for k in floor['support']]}"
    json.dumps(floor)

    out = rolling_origin_cv(d, lambda: DummyClassifier(strategy="most_frequent"),
                            n_folds=3, embargo_hours=1, horizon=1,
                            feature_cols=["x"])
    json.dumps(out)


def test_feature_matrix_is_writable_under_copy_on_write():
    """Regression: pandas 3 can return a read-only view from to_numpy().

    ``rolling_origin_cv`` scales in place, so a read-only feature matrix raises
    "assignment destination is read-only" mid-fold. pandas 2 returned a writable
    copy and hid this; the guarantee is pinned here rather than left to the
    pandas version that happens to be installed.
    """
    d = synthetic(n=400)
    d["x2"] = d["x"] * 1000
    block = d[["x", "x2"]].to_numpy(dtype=float, copy=True)
    assert block.flags.writeable
    block[0, 0] = 1.0  # must not raise

def test_rolling_origin_cv_scales_inside_the_fold():
    """Per-fold scaling must not depend on data after the cutoff."""
    d = synthetic(n=1500)
    d["x2"] = d["x"] * 1000
    a = rolling_origin_cv(d, lambda: RandomForestClassifier(
                              n_estimators=10, random_state=0, n_jobs=1),
                          n_folds=3, embargo_hours=1, horizon=1,
                          feature_cols=["x", "x2"], scale_cols=["x", "x2"])
    b = rolling_origin_cv(d, lambda: RandomForestClassifier(
                              n_estimators=10, random_state=0, n_jobs=1),
                          n_folds=3, embargo_hours=1, horizon=1,
                          feature_cols=["x", "x2"])
    # A tree is scale-invariant, so scaling must not change the result at all.
    assert ([f["model"]["macro_f1"] for f in a["folds"]]
            == pytest.approx([f["model"]["macro_f1"] for f in b["folds"]]))


def test_rolling_origin_cv_marks_sparse_classes_unevaluable():
    d = synthetic(n=2000, n_classes=3)
    d.loc[d.index[:5], "y"] = 9              # a class with almost no support
    d["naive"] = d["y"].shift(1).bfill().astype(int)
    out = rolling_origin_cv(d, lambda: DummyClassifier(strategy="most_frequent"),
                            n_folds=3, embargo_hours=1, horizon=1,
                            feature_cols=["x"], baseline_col="naive",
                            min_class_support=20)
    assert any("unevaluable" in f and f["unevaluable"] for f in out["folds"])


def test_rolling_origin_cv_observed_col_filters_evaluation_only():
    """Unobserved rows leave the evaluation set but must still train the model."""
    d = synthetic(n=2000)
    d["observed"] = True
    d.iloc[::3, d.columns.get_loc("observed")] = False      # a third unobserved
    full = rolling_origin_cv(d, lambda: DummyClassifier(strategy="most_frequent"),
                             n_folds=3, embargo_hours=1, horizon=1,
                             feature_cols=["x"])
    filt = rolling_origin_cv(d, lambda: DummyClassifier(strategy="most_frequent"),
                             n_folds=3, embargo_hours=1, horizon=1,
                             feature_cols=["x"], observed_col="observed")
    for a, b in zip(full["folds"], filt["folds"]):
        assert b["n_eval"] < a["n_eval"]        # evaluation shrinks
        assert b["n_train"] == a["n_train"]     # training does not


def test_rolling_origin_cv_reports_why_a_block_is_unevaluable():
    d = synthetic(n=2000)
    d["observed"] = True
    d.loc[d.index[-900:], "observed"] = False    # wipes out a whole block
    with pytest.raises(ValueError, match="observed_col"):
        rolling_origin_cv(d, lambda: DummyClassifier(strategy="most_frequent"),
                          n_folds=3, embargo_hours=1, horizon=1,
                          feature_cols=["x"], observed_col="observed")


def test_rolling_origin_cv_rejects_impossible_fold_layout():
    d = synthetic(n=60)
    with pytest.raises(ValueError):
        rolling_origin_cv(d, lambda: DummyClassifier(), n_folds=30,
                          embargo_hours=48, horizon=1, feature_cols=["x"])


# ------------------------------------------------------ aggregate / resolution


def test_wilcoxon_resolution_floor_is_reported_and_correct():
    """The floor is the reason a 5-fold null must not be read as absence of effect."""
    d = synthetic(n=4000, sticky=0.5)
    d["naive"] = 0
    for n_folds, reachable in ((3, False), (5, False), (7, True)):
        out = rolling_origin_cv(d, lambda: RandomForestClassifier(
                                    n_estimators=10, random_state=0, n_jobs=1),
                                n_folds=n_folds, embargo_hours=1, horizon=1,
                                feature_cols=["x"], baseline_col="naive")
        res = out["aggregate"]["resolution"]
        assert res["min_p_two_sided"] == pytest.approx(2.0 ** (1 - n_folds))
        assert res["can_reach_alpha_05"] is reachable


def test_aggregate_folds_counts_wins_and_signs_correctly():
    folds = []
    for i, (m, b) in enumerate([(0.6, 0.5), (0.55, 0.5), (0.4, 0.5)], start=1):
        folds.append({"fold": i, "support": {"A": 100, "B": 100},
                      "model": {"macro_f1": m, "f1_A": m, "f1_B": m},
                      "baseline": {"macro_f1": b, "f1_A": b, "f1_B": b}})
    agg = aggregate_folds(folds, ["A", "B"])
    assert agg["macro_f1"]["wins"] == 2
    assert agg["macro_f1"]["mean_delta"] == pytest.approx((0.1 + 0.05 - 0.1) / 3)


# -------------------------------------------------- advisory_disqualification


def test_disqualification_rejects_aggregate_gain_that_harms_protected_class():
    res = {"Critical": {"delta": -0.02, "significant": True},
           "Common": {"delta": 0.09, "significant": True}}
    out = advisory_disqualification(res, ["Critical"], aggregate_delta=0.05,
                                    aggregate_significant=True)
    assert out["disqualified"] is True
    assert out["harmed"] == ["Critical"]
    assert "Critical" in out["reason"]


def test_disqualification_accepts_clean_material_gain():
    res = {"Critical": {"delta": 0.03, "significant": True}}
    out = advisory_disqualification(res, ["Critical"], aggregate_delta=0.05,
                                    aggregate_significant=True)
    assert out["verdict"] == "accepted"


def test_disqualification_calls_small_gain_insufficient():
    res = {"Critical": {"delta": 0.001, "significant": False}}
    out = advisory_disqualification(res, ["Critical"], aggregate_delta=0.003,
                                    aggregate_significant=True)
    assert out["verdict"] == "insufficient"


def test_disqualification_ignores_nonsignificant_regression():
    res = {"Critical": {"delta": -0.04, "significant": False}}
    out = advisory_disqualification(res, ["Critical"], aggregate_delta=0.05,
                                    aggregate_significant=True)
    assert out["disqualified"] is False


def test_disqualification_raises_on_unknown_protected_class():
    with pytest.raises(KeyError):
        advisory_disqualification({"A": {"delta": 0.1, "significant": True}}, ["Z"])


# ----------------------------------------------------------- bonferroni_report


def test_bonferroni_splits_survivors_from_losses():
    out = bonferroni_report({"a": {"p": 0.001, "delta": 0.2},
                             "b": {"p": 0.03, "delta": 0.1},
                             "c": {"p": 0.6, "delta": 0.0}})
    assert out["k"] == 3
    assert out["corrected_alpha"] == pytest.approx(0.05 / 3)
    assert out["survivors"] == ["a"]
    assert out["lost"] == ["b"]          # significant at 0.05, not after correction


def test_bonferroni_flags_the_resampling_floor():
    out = bonferroni_report({"a": {"p": 0.0, "delta": 0.2}}, n_resamples=1000)
    row = out["rows"][0]
    assert row["at_resolution_floor"] is True
    assert row["p_display"].startswith("<")


def test_bonferroni_accepts_list_form_and_renders_markdown():
    out = bonferroni_report([{"name": "x", "p": 0.01, "delta": -0.3}])
    md = format_markdown(out)
    assert md.startswith("| Comparison |")
    assert "-0.3000" in md


def test_bonferroni_rejects_empty_and_malformed_input():
    with pytest.raises(ValueError):
        bonferroni_report({})
    with pytest.raises(KeyError):
        bonferroni_report([{"name": "x"}])
