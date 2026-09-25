"""dataset_audit on synthetic series with known provenance.

Two properties matter and pull against each other. The audit has to flag a
deliberately fabricated segment, and it has to stay quiet on data that is merely
noisy, seasonal and gappy — which is what real environmental data looks like. A
detector that fires on both is useless, so both directions are tested here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pulsebench import dataset_audit

HOURS = 24 * 365


def realistic(n=HOURS * 2, seed=0, start="2019-01-01"):
    """Noisy, seasonal, autocorrelated, occasionally extreme — but genuine.

    Built as an AR(1) around a seasonal mean with log-normal spikes, which is the
    shape hourly particulate data actually has. Nothing here should trip the audit.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = 60 + 35 * np.sin(2 * np.pi * t / HOURS - 1.2) + 12 * np.sin(2 * np.pi * t / 24)
    y = np.empty(n)
    y[0] = seasonal[0]
    for i in range(1, n):
        y[i] = 0.92 * y[i - 1] + 0.08 * seasonal[i] + rng.normal(0, 6)
    y += rng.lognormal(0, 1.1, n) * (rng.random(n) < 0.02) * 40   # episodic spikes
    # a soft floor, not a clip: real readings approach zero without piling on it
    y = np.maximum(y, 0.5 + rng.random(n))
    return pd.DataFrame({"t": pd.date_range(start, periods=n, freq="h"),
                         "v": y})


def fabricated(n=HOURS * 5, seed=1, start="2013-01-01", clip_at=250.0):
    """A generator's output: near-linear drift, a hard ceiling, weak persistence."""
    rng = np.random.default_rng(seed)
    y = np.linspace(20, 300, n) + rng.normal(0, 6, n)
    return pd.DataFrame({"t": pd.date_range(start, periods=n, freq="h"),
                         "v": np.minimum(y, clip_at)})


def spliced(unit_change=False, weak_persistence=False):
    """Fabricated prefix followed by genuine data, as a published dataset had.

    The published case carried four defects at once: a near-linear trend, a hard
    ceiling, carbon monoxide in different units either side of the join, and lag-1
    autocorrelation of 0.905 before against 0.983 after. The flags are separable here
    so each check can be exercised on its own and in combination.
    """
    fake = fabricated()
    if weak_persistence:                      # generated noise is less persistent
        rng = np.random.default_rng(11)
        fake["v"] = np.minimum(
            np.linspace(20, 300, len(fake)) + rng.normal(0, 45, len(fake)), 250.0)
    if unit_change:                           # the prefix is recorded in another unit
        fake["v"] = fake["v"] * 40
    real = realistic(start="2018-01-01")
    return pd.concat([fake, real], ignore_index=True), real["t"].iloc[0]


# ------------------------------------------------------- it must flag the fake


def test_flags_a_fabricated_segment():
    """All four defects together, as the published dataset carried them."""
    df, _ = spliced(unit_change=True, weak_persistence=True)
    out = dataset_audit(df, time_col="t", value_col="v")
    assert out["verdict"] == "fabrication suspected", out["flagged"]
    assert out["n_flagged"] >= 3


def test_two_defects_alone_are_reported_as_inconclusive():
    """A trend and a clip without a splice signature is suspicious, not conclusive.

    The module does not overclaim on partial evidence, and the boundary is still
    located even when the verdict stops short.
    """
    df, true_boundary = spliced()
    out = dataset_audit(df, time_col="t", value_col="v")
    assert out["verdict"] == "inconclusive"
    assert sorted(out["flagged"]) == ["hard_clip", "trend_linearity"]
    assert abs((pd.Timestamp(out["suspected_boundary"]) - true_boundary).days) <= 120


def test_locates_the_splice_boundary():
    """The proposed boundary must land near the real one, not anywhere."""
    df, true_boundary = spliced()
    out = dataset_audit(df, time_col="t", value_col="v")
    assert out["suspected_boundary"] is not None
    got = pd.Timestamp(out["suspected_boundary"])
    off_days = abs((got - true_boundary).days)
    assert off_days <= 120, f"boundary off by {off_days} days ({got} vs {true_boundary})"


def test_detects_the_hard_clip():
    df, _ = spliced()
    c = dataset_audit(df, time_col="t", value_col="v")["checks"]["hard_clip"]
    assert c["flagged"]
    assert c["clip_value"] == pytest.approx(250.0)


def test_detects_the_linear_prefix():
    df, _ = spliced()
    c = dataset_audit(df, time_col="t", value_col="v")["checks"]["trend_linearity"]
    assert c["flagged"]
    assert c["r2"] >= 0.90
    assert 0.2 < c["prefix_fraction"] < 0.8


# ------------------------------------------------ it must stay quiet on real data


def test_does_not_flag_genuine_noisy_data():
    """The property that makes the audit usable rather than alarmist."""
    out = dataset_audit(realistic(), time_col="t", value_col="v")
    assert out["verdict"] == "no structural anomaly", (
        f"flagged {out['flagged']} on realistic data:\n"
        + json.dumps({k: v["note"] for k, v in out["checks"].items()}, indent=1))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_stays_quiet_across_several_realistic_series(seed):
    out = dataset_audit(realistic(seed=seed), time_col="t", value_col="v")
    assert out["n_flagged"] <= 1, f"seed {seed} flagged {out['flagged']}"


def test_gaps_and_missing_values_do_not_trigger_it():
    df = realistic()
    rng = np.random.default_rng(7)
    df.loc[rng.random(len(df)) < 0.05, "v"] = np.nan          # 5% missing
    df = df.drop(df.index[5000:5600])                          # a three-week gap
    out = dataset_audit(df, time_col="t", value_col="v")
    assert out["n_flagged"] <= 1, out["flagged"]


# -------------------------------------------------------------------- grouping


def test_group_mode_isolates_the_affected_group():
    """A fabricated backfill often affects one group only, which is diagnostic."""
    fake, _ = spliced(unit_change=True, weak_persistence=True)
    fake["city"] = "Backfilled"
    clean = realistic(start="2018-01-01")
    clean["city"] = "Genuine"
    out = dataset_audit(pd.concat([fake, clean], ignore_index=True),
                        time_col="t", value_col="v", group_col="city")
    assert out["focus_group"] == "Backfilled", out["focus_group"]
    assert out["per_group"]["Backfilled"]["n_flagged"] >= 3
    assert out["per_group"]["Genuine"]["n_flagged"] <= 1


# ------------------------------------------------------------------- interface


def test_missing_columns_raise_with_a_useful_message():
    with pytest.raises(KeyError, match="need columns"):
        dataset_audit(pd.DataFrame({"a": [1, 2, 3]}), time_col="t", value_col="v")


def test_result_is_json_serializable():
    out = dataset_audit(realistic(), time_col="t", value_col="v")
    json.dumps(out)


def test_short_series_does_not_crash():
    df = realistic(n=50)
    out = dataset_audit(df, time_col="t", value_col="v")
    assert out["verdict"] in ("no structural anomaly", "inconclusive",
                              "fabrication suspected")


def test_a_segment_shorter_than_the_aggregation_window_is_not_detectable():
    """The trend test has a stated resolution limit, and it is tested rather than hidden.

    A linear fit needs enough aggregated points to mean anything, so a fabricated
    segment shorter than that many periods cannot be found by this check. A caller
    who knows the segment is short should pass a finer `period`.
    """
    short_fake = fabricated(n=24 * 60, start="2017-11-01")    # two months
    real = realistic(start="2018-01-01")
    df = pd.concat([short_fake, real], ignore_index=True)
    out = dataset_audit(df, time_col="t", value_col="v")
    assert not out["checks"]["trend_linearity"]["flagged"]
    # and with a finer aggregation it becomes visible again
    finer = dataset_audit(df, time_col="t", value_col="v", period="W")
    assert finer["checks"]["trend_linearity"]["n_periods"] > 8
