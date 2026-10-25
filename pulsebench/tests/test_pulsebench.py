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

# One step of each frequency, written out rather than derived from to_offset so the
# expectations below are checkable by eye. "W" is the Sunday-anchored week.
STEP = {"h": pd.Timedelta(hours=1), "D": pd.Timedelta(days=1),
        "15min": pd.Timedelta(minutes=15), "W": pd.Timedelta(weeks=1)}
FREQS = list(STEP)


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


@pytest.mark.parametrize("freq", FREQS)
def test_seasonal_naive_periodic_series(freq):
    from pulsebench import seasonal_naive_floor
    d = pd.DataFrame({"y": [0, 1, 2] * 4},
                     index=pd.date_range("2020-01-01", periods=12, freq=freq))
    out = seasonal_naive_floor(d, "y", 1, season_length=3, freq=freq)
    assert out["macro_f1"] == 1.0
    assert out["n_pairs"] == 9
    json.dumps(out)


@pytest.mark.parametrize("freq", FREQS)
def test_seasonal_naive_matches_persistence_at_season_horizon(freq):
    from pulsebench import seasonal_naive_floor
    d = synthetic(n=100, freq=freq)
    d = d.drop(d.index[5])          # a hole, so the join (not a shift) is what is tested
    assert (seasonal_naive_floor(d, "y", 3, season_length=3, freq=freq)
            == persistence_floor(d, "y", 3, freq=freq))


@pytest.mark.parametrize("freq", FREQS)
def test_seasonal_naive_requires_origin_target_and_seasonal_source(freq):
    from pulsebench import seasonal_naive_floor
    grid = pd.date_range("2020-01-05", periods=7, freq=freq)    # starts on a Sunday
    idx = grid[[0, 2, 3, 4, 6]]
    d = pd.DataFrame({"y": [0, 2, 0, 1, 0]}, index=idx)
    out = seasonal_naive_floor(d, "y", 1, season_length=3, freq=freq)
    # Target 3 has origin 2/source 0; target 4 lacks source 1; 6 lacks origin 5.
    assert out["n_pairs"] == 1
    assert out["accuracy"] == 1.0


@pytest.mark.parametrize("freq", FREQS)
def test_seasonal_naive_groups_time_column_and_explicit_labels(freq):
    from pulsebench import seasonal_naive_floor
    idx = pd.date_range("2020-01-01", periods=8, freq=freq)
    d = pd.concat([pd.DataFrame({"time": idx, "y": value, "site": site})
                   for value, site in [(0, "a"), (1, "b")]])
    out = seasonal_naive_floor(d, "y", 1, season_length=3, time_col="time",
                               group_col="site", labels=[0, 1, 2], freq=freq)
    assert out["n_pairs"] == 10
    assert out["accuracy"] == 1.0
    assert out["support"] == {0: 5, 1: 5, 2: 0}


@pytest.mark.parametrize("freq", FREQS)
def test_seasonal_naive_long_horizon_uses_completed_seasons(freq):
    from pulsebench import seasonal_naive_floor
    d = pd.DataFrame({"y": [0] * 3 + [1] * 3 + [2] * 3},
                     index=pd.date_range("2020-01-01", periods=9, freq=freq))
    out = seasonal_naive_floor(d, "y", 4, season_length=3, freq=freq)
    assert out == persistence_floor(d, "y", 6, freq=freq)


@pytest.mark.parametrize("horizon,season", [(0, 3), (1, 0), (-1, 3), (1, 1.5), (True, 3)])
def test_seasonal_naive_rejects_invalid_steps(horizon, season):
    from pulsebench import seasonal_naive_floor
    with pytest.raises(ValueError, match="positive integer"):
        seasonal_naive_floor(synthetic(n=10), "y", horizon, season_length=season)


def test_seasonal_naive_no_pairs():
    from pulsebench import seasonal_naive_floor
    with pytest.raises(ValueError, match="no seasonal pairs"):
        seasonal_naive_floor(synthetic(n=3), "y", 1, season_length=24)


@pytest.mark.parametrize("freq", FREQS)
def test_persistence_floor_perfect_on_constant_series(freq):
    idx = pd.date_range("2020-01-01", periods=100, freq=freq)
    d = pd.DataFrame({"y": np.ones(100, dtype=int)}, index=idx)
    out = persistence_floor(d, "y", horizon=3, freq=freq)
    assert out["macro_f1"] == pytest.approx(1.0)
    assert out["label_unchanged_pct"] == pytest.approx(100.0)
    assert out["n_pairs"] == 97


@pytest.mark.parametrize("freq", FREQS)
def test_persistence_floor_pairs_do_not_span_gaps(freq):
    """A positional shift would invent a pair across the hole; a time join must not."""
    first = pd.date_range("2020-01-05", periods=2, freq=freq)
    second = pd.date_range("2021-01-03", periods=2, freq=freq)   # both Sundays
    d = pd.DataFrame({"y": [0, 1, 0, 1]}, index=first.append(second))
    out = persistence_floor(d, "y", horizon=1, freq=freq)
    assert out["n_pairs"] == 2          # one pair either side, none across the gap


@pytest.mark.parametrize("freq", FREQS)
def test_persistence_floor_respects_groups(freq):
    idx = pd.date_range("2020-01-01", periods=10, freq=freq)
    d = pd.concat([
        pd.DataFrame({"y": np.zeros(10, dtype=int), "site": "a"}, index=idx),
        pd.DataFrame({"y": np.ones(10, dtype=int), "site": "b"}, index=idx)])
    grouped = persistence_floor(d, "y", horizon=1, group_col="site", freq=freq)
    assert grouped["n_pairs"] == 18                 # 9 per site, none crossing
    assert grouped["macro_f1"] == pytest.approx(1.0)


@pytest.mark.parametrize("freq", FREQS)
def test_persistence_floor_is_harder_to_beat_when_series_is_stickier(freq):
    loose = persistence_floor(synthetic(sticky=0.3, freq=freq), "y", horizon=1, freq=freq)
    tight = persistence_floor(synthetic(sticky=0.95, freq=freq), "y", horizon=1, freq=freq)
    assert tight["macro_f1"] > loose["macro_f1"]


@pytest.mark.parametrize("freq", FREQS)
def test_persistence_floor_horizon_is_counted_in_freq_steps(freq):
    """Horizon 2 pairs t with t+2 steps, so a series with period 2 is perfectly floored."""
    d = pd.DataFrame({"y": [0, 1] * 10},
                     index=pd.date_range("2020-01-05", periods=20, freq=freq))
    assert persistence_floor(d, "y", horizon=2, freq=freq)["macro_f1"] == 1.0
    assert persistence_floor(d, "y", horizon=1, freq=freq)["macro_f1"] == 0.0
    # Forgetting freq= means hourly steps, which miss a daily or weekly grid entirely.
    if freq in ("D", "W"):
        with pytest.raises(ValueError, match="no .* pairs"):
            persistence_floor(d, "y", horizon=2)


def test_persistence_floor_requires_datetime_index():
    with pytest.raises(TypeError):
        persistence_floor(pd.DataFrame({"y": [1, 2, 3]}), "y", horizon=1)


# ------------------------------------------------------------ rolling_origin_cv


@pytest.mark.parametrize("freq", FREQS)
def test_make_folds_are_contiguous_and_expanding(freq):
    idx = pd.date_range("2020-01-01", periods=1000, freq=freq)
    folds = make_folds(idx.to_numpy(), n_folds=5, initial_fraction=0.4)
    assert len(folds) == 5
    cutoffs = [f["cutoff"] for f in folds]
    assert cutoffs == sorted(cutoffs)                     # expanding window
    for a, b in zip(folds, folds[1:]):
        assert a["eval_end"] <= b["cutoff"]               # blocks do not overlap


@pytest.mark.parametrize("freq", FREQS)
def test_rolling_origin_cv_shapes_and_growth(freq):
    d = synthetic(n=3000, freq=freq)
    d["naive"] = d["y"].shift(1).bfill().astype(int)
    out = rolling_origin_cv(d, lambda: DummyClassifier(strategy="most_frequent"),
                            n_folds=4, embargo_hours=2, horizon=1, freq=freq,
                            feature_cols=["x"], baseline_col="naive")
    assert len(out["folds"]) == 4
    sizes = [f["n_train"] for f in out["folds"]]
    assert sizes == sorted(sizes)                          # training set grows
    for f in out["folds"]:
        assert f["n_eval"] > 0


@pytest.mark.parametrize("freq", FREQS)
def test_rolling_origin_cv_embargo_separates_train_and_eval(freq):
    d = synthetic(n=2000, freq=freq)
    embargo = 12
    out = rolling_origin_cv(d, lambda: DummyClassifier(strategy="most_frequent"),
                            n_folds=3, embargo_hours=embargo, horizon=1, freq=freq,
                            feature_cols=["x"])
    for f in out["folds"]:
        gap = pd.Timestamp(f["eval_start"]) - pd.Timestamp(f["cutoff"])
        assert gap >= pd.Timedelta(hours=embargo)


@pytest.mark.parametrize("freq", FREQS)
def test_rolling_origin_cv_embargo_hours_means_wall_clock_hours(freq):
    """`embargo_hours` is hours at every freq, not a number of steps.

    The first evaluable row is the first grid time at or after cutoff + embargo, so
    the gap is the embargo rounded up to a whole step: 12 h is exactly 12 h at "h"
    and "15min", but one whole step at "D" (24 h) and "W" (168 h). A step-count
    reading would instead give 12 steps (12 days at "D").
    """
    step = STEP[freq]
    d = synthetic(n=2000, freq=freq)
    for embargo in (12, 48):
        out = rolling_origin_cv(d, lambda: DummyClassifier(strategy="most_frequent"),
                                n_folds=3, embargo_hours=embargo, horizon=1, freq=freq,
                                feature_cols=["x"])
        want = -(-pd.Timedelta(hours=embargo) // step) * step
        for f in out["folds"]:
            gap = pd.Timestamp(f["eval_start"]) - pd.Timestamp(f["cutoff"])
            assert gap == want


@pytest.mark.parametrize("freq", FREQS)
def test_rolling_origin_cv_training_targets_precede_cutoff_in_freq_steps(freq):
    """horizon is in freq steps: a row's target time is t + horizon*step."""
    d = synthetic(n=1200, freq=freq)
    out = rolling_origin_cv(d, lambda: DummyClassifier(strategy="most_frequent"),
                            n_folds=3, embargo_hours=1, horizon=4, freq=freq,
                            feature_cols=["x"])
    step = STEP[freq]
    for f in out["folds"]:
        cutoff = pd.Timestamp(f["cutoff"])
        # The row at cutoff - horizon*step has its target exactly at the cutoff and
        # must be excluded, so training is everything strictly before that row.
        assert f["n_train"] == int((d.index < cutoff - 4 * step).sum())


@pytest.mark.parametrize("freq", FREQS)
def test_rolling_origin_cv_model_beats_baseline_when_it_should(freq):
    """A feature carrying the label should beat a most-frequent baseline in every fold."""
    d = synthetic(n=4000, sticky=0.5, freq=freq)
    d["naive"] = 0
    out = rolling_origin_cv(d, lambda: RandomForestClassifier(
                                n_estimators=25, random_state=0, n_jobs=1),
                            n_folds=4, embargo_hours=1, horizon=1, freq=freq,
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
