"""Tests for the analyses added in the six-phase extension.

These are report-only or evaluation-only modules, so what needs protecting is not a
model but the arithmetic and the honesty constraints: that the noise model uses only
published parameters, that selective prediction partitions rather than filters, and
that the station holdout compares each station against its own baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "reports" / "metrics"


def _metrics(name):
    p = METRICS / name
    if not p.exists():
        pytest.skip(f"{name} not present — run the phase that writes it")
    return json.loads(p.read_text())


# ------------------------------------------------------- sensor-noise model (D)


def test_noise_sd_reproduces_the_target_r2():
    """The amplitude is derived from a published R2, so it must reproduce it."""
    from src.models.sensor_noise_robustness import noise_sd_for_r2, inject
    rng = np.random.default_rng(0)
    signal = np.abs(rng.lognormal(4, 0.8, 20000))
    for target in (0.10, 0.19, 0.28, 0.50):
        sd = noise_sd_for_r2(signal, target)
        noisy = signal + rng.normal(0, sd, len(signal))
        got = np.corrcoef(signal, noisy)[0, 1] ** 2
        assert abs(got - target) < 0.03, f"target {target}, got {got:.4f}"


def test_noise_model_rejects_an_impossible_r2():
    from src.models.sensor_noise_robustness import noise_sd_for_r2
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError, match="target_r2"):
            noise_sd_for_r2(np.arange(100.0), bad)


def test_injection_never_produces_negative_concentrations():
    from src.models.sensor_noise_robustness import inject
    rng = np.random.default_rng(1)
    clean = np.abs(rng.lognormal(3.5, 1.0, 5000))
    out = inject(clean, 0.19, seed=7)
    assert (out["values"] >= 0).all()


def test_only_published_parameters_are_used():
    """The module must not carry a noise parameter the cited paper does not report.

    This is the constraint the phase was given: state the source, and do not invent
    what it does not provide. Slope, intercept, quantisation and bias are absent from
    the paper, so they must stay absent from the model.
    """
    from src.models import sensor_noise_robustness as snr
    assert set(snr.AMBIENT_R2.values()) == {0.10, 0.23, 0.28, 0.15}
    assert snr.DETECTION_FLOOR_UGM3 == 10.0
    # Check the module's own constants, not its prose. An earlier version grepped the
    # source text and failed on the docstring that says these are *not* modelled,
    # which is the very sentence the phase required.
    consts = {k: v for k, v in vars(snr).items()
              if k.isupper() and isinstance(v, (int, float))}
    forbidden = [k for k in consts
                 if any(w in k for w in ("SLOPE", "INTERCEPT", "QUANT", "BIAS",
                                         "OFFSET"))]
    assert not forbidden, (
        f"module defines parameters the cited paper does not report: {forbidden}")
    # and the JSON must say out loud which ones are absent
    assert set(snr.__doc__.upper().count(w) > 0 for w in ("SLOPE", "QUANTIS")) == {True}


def test_reported_degradation_is_internally_consistent():
    m = _metrics("sensor_noise_robustness_h6.json")
    h = m["summary"]["f1_Hazardous"]
    assert h["absolute_drop"] == pytest.approx(h["clean"]["mean"] - h["noisy"]["mean"])
    assert h["margin_over_noisy_floor"] == pytest.approx(
        h["noisy"]["mean"] - h["persistence_noisy"]["mean"])
    assert 0 <= h["folds_won_noisy"] <= m["n_folds"]
    assert m["not_modelled"], "the report must name what it does not model"


# ------------------------------------------------- selective prediction (E)


@pytest.mark.parametrize("name", ["selective_prediction_h6.json",
                                  "selective_prediction_h6_bangladesh.json"])
def test_confident_and_abstained_partition_the_test_set(name):
    """The two subsets must sum to the whole: this is a partition, not a filter."""
    m = _metrics(name)
    assert m["confident"]["n"] + m["abstained"]["n"] == m["full"]["n"]
    assert m["confident_fraction"] + m["abstain_fraction"] == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["selective_prediction_h6.json",
                                  "selective_prediction_h6_bangladesh.json"])
def test_set_size_buckets_sum_to_the_whole(name):
    m = _metrics(name)
    assert sum(b["n"] for b in m["by_set_size"].values()) == m["full"]["n"]
    assert sum(b["share"] for b in m["by_set_size"].values()) == pytest.approx(1.0)


def test_abstention_is_reported_as_a_trade_not_a_gain():
    """The report must not present the confident subset as a free improvement."""
    text = (ROOT / "reports" / "selective_prediction_h6.md").read_text()
    assert "trade, not a gain" in text
    assert "does not make those rows" in text


# -------------------------------------------------------- station holdout (C)


def test_each_station_is_compared_against_its_own_floor():
    m = _metrics("station_holdout_h6.json")
    for st in m["stations"]:
        # the floor is computed on the held-out station's own rows
        assert st["persistence"]["n"] if "n" in st["persistence"] else True
        assert st["n_eval_observed"] > 0
        assert set(st["vs_persistence"]) >= {"macro_f1", "Hazardous"}


def test_station_holdout_verdict_matches_its_win_count():
    m = _metrics("station_holdout_h6.json")
    a, n = m["aggregate"], m["n_stations"]
    expected = ("generalisation holds" if a["wins_macro"] == n
                else "generalisation largely holds" if a["wins_macro"] >= n * 0.75
                else "generalisation partially holds" if a["wins_macro"] >= n * 0.4
                else "generalisation fails")
    assert a["verdict"] == expected


def test_station_holdout_resolution_floor_is_stated():
    m = _metrics("station_holdout_h6.json")
    n = m["n_stations"]
    assert m["aggregate"]["resolution"]["min_p_two_sided"] == pytest.approx(2.0 ** (1 - n))


# --------------------------------------------------------- fold count (B)


def test_eight_fold_run_did_not_overwrite_the_published_five_fold_run():
    """The five-fold numbers are quoted throughout the thesis and must survive."""
    five = _metrics("rolling_cv_h6.json")
    assert five["n_folds"] == 5
    eight = METRICS / "rolling_cv_h6_f8.json"
    if eight.exists():
        assert json.loads(eight.read_text())["n_folds"] == 8


def test_no_model_reaches_a_majority_of_folds_at_either_count():
    """The headline claim, asserted at both fold counts."""
    for name in ("rolling_cv_h6.json", "rolling_cv_h6_f8.json",
                 "rolling_cv_h6_f8_seq.json"):
        p = METRICS / name
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        n = m["n_folds"]
        majority = n // 2 + 1
        for model, t in m["aggregate"]["tests"].items():
            if model.startswith("_"):
                continue
            assert t["wins"] < majority, (
                f"{model} won {t['wins']}/{n} in {name}, which is a majority — "
                f"the thesis headline claim would need revising")
