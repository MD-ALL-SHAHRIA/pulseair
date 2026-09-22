"""Unit tests for the preprocessing pipeline.

These run against a small synthetic frame with the same *schema* as the Beijing table,
not the 420,768-row CSV -- the invariants under test (shapes, leakage, scaler fit) are
structural, and a fixture makes them fast and deterministic. The handful of checks that
need the real artifacts are skipped unless the pipeline has been run.

    pytest src/preprocessing/test_pipeline.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.preprocessing import pipeline as pl

RNG = np.random.default_rng(0)

STATIONS = ["Alpha", "Bravo", "Charlie"]
HOURS = 600          # per station -- enough for a real 70/15/15 chronological cut


# ------------------------------------------------------------------------- fixtures


def _synthetic(n_hours: int = HOURS, stations=tuple(STATIONS), with_nans: bool = True):
    """A miniature Beijing-shaped table: same columns, same dtypes, same NaN behaviour."""
    stamps = pd.date_range("2013-03-01", periods=n_hours, freq="h")
    frames = []
    for s_i, station in enumerate(stations):
        f = pd.DataFrame({
            "year": stamps.year, "month": stamps.month,
            "day": stamps.day, "hour": stamps.hour,
        })
        # Station-separated magnitudes: a value alone identifies its station, which is
        # what lets the leakage tests below detect a window that crossed a boundary.
        base = 1000.0 * (s_i + 1)
        f["PM2.5"] = base + RNG.uniform(1, 300, n_hours)
        f["PM10"] = base + RNG.uniform(1, 300, n_hours)
        f["SO2"] = RNG.uniform(1, 50, n_hours)
        f["NO2"] = RNG.uniform(1, 80, n_hours)
        f["CO"] = base + RNG.uniform(100, 3000, n_hours)
        f["O3"] = RNG.uniform(1, 200, n_hours)
        f["TEMP"] = base + RNG.uniform(-10, 35, n_hours)
        f["PRES"] = RNG.uniform(990, 1040, n_hours)
        f["DEWP"] = base + RNG.uniform(-25, 25, n_hours)
        f["RAIN"] = 0.0
        f["wd"] = "NNW"
        f["WSPM"] = RNG.uniform(0, 8, n_hours)
        f["station"] = station
        if with_nans:
            # Leading NaNs (unfillable -> dropped) and interior NaNs (ffill-able).
            f.loc[:2, "CO"] = np.nan
            f.loc[100:104, "PM2.5"] = np.nan
            f.loc[200:201, "DEWP"] = np.nan
            # Out-of-scope channel: neither the leading nor the interior gap may cost
            # us rows, because O3 is not a wearable channel and is not modelled.
            f.loc[:5, "O3"] = np.nan
            f.loc[50:60, "O3"] = np.nan
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "No", np.arange(1, len(out) + 1))
    return out


@pytest.fixture(scope="module")
def cfg():
    return pl.load_config()


@pytest.fixture(scope="module")
def raw():
    return _synthetic()


@pytest.fixture(scope="module")
def result(cfg, raw):
    return pl.run(cfg, raw=raw, write=False, verbose=False)


# --------------------------------------------------------------------------- config


def test_config_matches_repo_yaml(cfg):
    """The pipeline must take the split from the config, not define its own."""
    assert cfg.train_size + cfg.val_size + cfg.test_size == pytest.approx(1.0)
    assert cfg.train_size == pytest.approx(0.70)
    assert cfg.val_size == pytest.approx(0.15)
    assert cfg.test_size == pytest.approx(0.15)
    assert cfg.group_key == "station"
    assert cfg.window == 24
    # h=1 was retired as primary: persistence scored 0.7933 macro-F1 there against
    # RandomForest's 0.7994. See reports/horizon_comparison.md.
    assert cfg.horizon == 6, "primary horizon should be 6 h"
    assert set(cfg.horizons) == {1, 6, 12, 24}


def test_feature_set_is_the_wearable_subset(cfg):
    assert cfg.features == ["PM2.5", "PM10", "CO", "TEMP", "DEWP"]
    assert cfg.feature_columns == [
        "PM2.5", "PM10", "CO", "TEMP", "DEWP",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
    ]
    # Every out-of-scope channel is excluded *and* documented.
    for channel in ("SO2", "NO2", "O3", "PRES", "RAIN", "wd", "WSPM"):
        assert channel not in cfg.feature_columns
        assert channel in pl.DROPPED_FEATURES
        assert len(pl.DROPPED_FEATURES[channel]) > 40, "needs a real justification"


# ----------------------------------------------------------------- missing values


def test_ffill_respects_station_boundaries(cfg):
    """A station's leading NaN must not be filled from the previous station's tail."""
    raw = _synthetic(n_hours=100, with_nans=False)
    raw["datetime"] = pd.to_datetime(raw[["year", "month", "day", "hour"]])
    raw = raw.sort_values(["station", "datetime"]).reset_index(drop=True)

    # Blank the first hour of Bravo and Charlie entirely for a modelled channel.
    for station in ("Bravo", "Charlie"):
        first = raw.index[raw["station"] == station][0]
        raw.loc[first, "PM2.5"] = np.nan

    filled, stats = pl.fill_missing(raw, cfg)
    # Those two rows had nothing to fill from, so they are dropped, not back-filled
    # with Alpha's / Bravo's values.
    assert stats["dropped"] == 2
    for station in ("Bravo", "Charlie"):
        kept = filled[filled["station"] == station]
        expected_base = 1000.0 * (STATIONS.index(station) + 1)
        assert kept["PM2.5"].min() > expected_base, "value bled across a station boundary"


def test_ffill_fills_interior_gaps_and_drops_leading(cfg, raw):
    df = raw.copy()
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)
    filled, stats = pl.fill_missing(df, cfg)

    assert filled[cfg.features].isna().sum().sum() == 0
    # Only the 3 leading CO NaNs per station are unfillable.
    assert stats["dropped"] == 3 * len(STATIONS)
    assert stats["rows_out"] == stats["rows_in"] - stats["dropped"]


def test_out_of_scope_nulls_do_not_drop_rows(cfg, raw):
    """O3 is missing in the fixture but is not modelled -- it must not cost us rows."""
    df = raw.copy()
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)
    _, stats = pl.fill_missing(df, cfg)
    assert stats["dropped"] < stats["dropped_if_all_columns"]


# ------------------------------------------------------------------------ features


def test_cyclical_encoding_is_on_the_unit_circle(cfg, raw):
    df = pl.add_cyclical(raw, cfg)
    for col in cfg.cyclical:
        r = df[f"{col}_sin"] ** 2 + df[f"{col}_cos"] ** 2
        np.testing.assert_allclose(r, 1.0, atol=1e-9)
    # Hour 23 must sit adjacent to hour 0, not 23 units away.
    h = df.drop_duplicates("hour").set_index("hour")[["hour_sin", "hour_cos"]]
    d_wrap = np.hypot(*(h.loc[23] - h.loc[0]))
    d_step = np.hypot(*(h.loc[1] - h.loc[0]))
    assert d_wrap == pytest.approx(d_step)


def test_aqi_labels_match_config_breakpoints(cfg):
    df = pd.DataFrame({"PM2.5": [5.0, 20.0, 45.0, 100.0, 200.0, 400.0]})
    out = pl.add_aqi_category(df, cfg)
    assert list(out[cfg.target]) == cfg.labels
    assert list(out[f"{cfg.target}_idx"]) == [0, 1, 2, 3, 4, 5]


# --------------------------------------------------------------------------- shapes


def test_sequence_shapes(result, cfg):
    n_feat = len(cfg.feature_columns)
    total = 0
    for split in pl.SPLITS:
        seq = result.sequences[split]
        n = seq["X"].shape[0]
        total += n
        assert seq["X"].shape == (n, cfg.window, n_feat)
        assert seq["X"].dtype == np.float32
        for key in ("y_category", "y_pm25", "y_pm25_raw", "station", "target_time"):
            assert seq[key].shape == (n,), f"{split}/{key} misaligned with X"
        assert not np.isnan(seq["X"]).any()
    assert total > 0


def test_tabular_matches_sequences_sample_for_sample(result, cfg):
    """The flat baseline view must cover exactly the same samples as the windows."""
    for split in pl.SPLITS:
        seq, tab = result.sequences[split], result.tabular[split]
        assert len(tab) == seq["X"].shape[0]
        np.testing.assert_array_equal(tab["y_category"].to_numpy(), seq["y_category"])
        np.testing.assert_array_equal(
            tab["target_time"].to_numpy(dtype="datetime64[ns]"), seq["target_time"]
        )
        # The flat row is the window's final timestep.
        np.testing.assert_allclose(
            tab[cfg.feature_columns].to_numpy(dtype=np.float32), seq["X"][:, -1, :], rtol=1e-6
        )


def test_horizon_target_sits_exactly_horizon_hours_after_the_window(result, cfg):
    for split in pl.SPLITS:
        tab = result.tabular[split]
        delta = pd.to_datetime(tab["target_time"]) - pd.to_datetime(tab["datetime"])
        assert (delta == pd.Timedelta(hours=cfg.horizon)).all()


def test_windows_are_contiguous_hours(result, cfg):
    """No window may silently span a hole left by a dropped row."""
    for split in pl.SPLITS:
        seq = result.sequences[split]
        if seq["X"].shape[0] == 0:
            continue
        # hour_sin/hour_cos let us recover the hour of every timestep in the window.
        i_sin = cfg.feature_columns.index("hour_sin")
        i_cos = cfg.feature_columns.index("hour_cos")
        theta = np.arctan2(seq["X"][:, :, i_sin], seq["X"][:, :, i_cos])
        hours = np.round(theta / (2 * np.pi) * 24) % 24
        step = np.diff(hours, axis=1) % 24
        assert (step == 1).all(), f"{split}: a window skipped an hour"


# -------------------------------------------------------------------------- leakage


def test_no_window_spans_two_stations(result, cfg):
    """Every timestep of a window must decode back to the station the window is labelled with.

    The fixture puts each station's PM2.5 in its own 1000-wide magnitude band, so a
    window that had been stitched across a station boundary would contain two bands.
    """
    band_of = {s: i + 1 for i, s in enumerate(STATIONS)}
    i_pm = cfg.feature_columns.index("PM2.5")
    for split in pl.SPLITS:
        seq = result.sequences[split]
        if seq["X"].shape[0] == 0:
            continue
        # Undo the scaling on the PM2.5 channel to get back to the fixture's bands.
        pm = seq["X"][:, :, i_pm] * result.scaler.scale_[i_pm] + result.scaler.mean_[i_pm]
        bands = np.floor(pm / 1000.0).astype(int)                     # (n_samples, window)
        assert (bands == bands[:, :1]).all(), f"{split}: a window mixes two stations"
        expected = np.array([band_of[s] for s in seq["station"]])
        np.testing.assert_array_equal(bands[:, 0], expected)
        # The t+horizon label belongs to the same station as its window.
        np.testing.assert_array_equal((seq["y_pm25_raw"] // 1000).astype(int), expected)


def test_no_timestamp_appears_in_two_splits(result, cfg):
    """Per station, the three splits' input timestamps must be pairwise disjoint."""
    for station in STATIONS:
        stamps = {
            split: set(pd.to_datetime(
                result.tabular[split].loc[
                    result.tabular[split]["station"] == station, "datetime"
                ]
            ))
            for split in pl.SPLITS
        }
        assert not (stamps["train"] & stamps["val"]), f"{station}: train/val overlap"
        assert not (stamps["train"] & stamps["test"]), f"{station}: train/test overlap"
        assert not (stamps["val"] & stamps["test"]), f"{station}: val/test overlap"


def test_splits_are_chronologically_ordered(result, cfg):
    """Every train timestamp precedes every val timestamp, which precedes every test one."""
    ends, starts = {}, {}
    for split in pl.SPLITS:
        t = pd.to_datetime(result.tabular[split]["target_time"])
        starts[split], ends[split] = t.min(), t.max()
    assert ends["train"] < starts["val"]
    assert ends["val"] < starts["test"]


def test_every_split_holds_every_station(result, cfg):
    for split in pl.SPLITS:
        assert set(result.tabular[split]["station"]) == set(STATIONS)


def test_split_proportions_follow_the_config(result, cfg):
    per_split = result.stats["split_boundaries"]["timestamps_per_split"]
    total = sum(per_split.values())
    assert per_split["train"] / total == pytest.approx(cfg.train_size, abs=0.01)
    assert per_split["val"] / total == pytest.approx(cfg.val_size, abs=0.01)
    assert per_split["test"] / total == pytest.approx(cfg.test_size, abs=0.01)


# --------------------------------------------------------------------------- scaler


def test_scaler_is_fit_on_train_rows_only(result, cfg, raw):
    """Re-fit a scaler on the train slice alone; it must equal the pipeline's."""
    df = raw.copy()
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)
    df, _ = pl.fill_missing(df, cfg)
    df = pl.add_cyclical(df, cfg)
    df = pl.add_aqi_category(df, cfg)
    df, _ = pl.assign_splits(df, cfg)

    train_only = StandardScaler().fit(df.loc[df["split"] == "train", cfg.features].to_numpy())
    np.testing.assert_allclose(result.scaler.mean_, train_only.mean_, rtol=1e-12)
    np.testing.assert_allclose(result.scaler.scale_, train_only.scale_, rtol=1e-12)
    assert result.scaler.n_samples_seen_ == (df["split"] == "train").sum()

    # And it must NOT match a scaler that saw everything.
    all_rows = StandardScaler().fit(df[cfg.features].to_numpy())
    assert not np.allclose(result.scaler.mean_, all_rows.mean_), "scaler saw val/test"


def test_train_features_are_standardised(result, cfg):
    """Train is ~N(0,1) by construction; val/test are transformed with those stats, not re-fit."""
    train = result.tabular["train"][cfg.features].to_numpy()
    np.testing.assert_allclose(train.mean(axis=0), 0.0, atol=0.15)
    np.testing.assert_allclose(train.std(axis=0), 1.0, atol=0.15)


def test_cyclical_columns_are_left_unscaled(result, cfg):
    for split in pl.SPLITS:
        block = result.tabular[split][cfg.cyclical_columns].to_numpy()
        assert block.min() >= -1.0 - 1e-9 and block.max() <= 1.0 + 1e-9


def test_scaler_is_invertible_back_to_raw_units(result, cfg):
    tab = result.tabular["test"]
    recovered = result.scaler.inverse_transform(tab[cfg.features].to_numpy())
    i_pm = cfg.features.index("PM2.5")
    # The window's last step at t, inverse-scaled, must be a plausible raw PM2.5.
    assert recovered[:, i_pm].min() > 0


# ------------------------------------------------------------------- class balance


def test_class_distribution_is_reported_per_split(result, cfg):
    dist = result.stats["class_distribution"]
    assert set(dist) == set(pl.SPLITS)
    for split in pl.SPLITS:
        assert set(dist[split]) == set(cfg.labels)
        assert sum(v["n"] for v in dist[split].values()) == len(result.tabular[split])
        assert sum(v["pct"] for v in dist[split].values()) == pytest.approx(100.0, abs=0.1)


def test_report_names_the_rare_classes(result):
    report = pl.build_report(result)
    assert "Hazardous" in report and "Very unhealthy" in report
    assert "Phase 4" in report or "CTGAN" in report


# --------------------------------------------------------- real artifacts (if built)


PROCESSED = pl.load_config().processed_dir      # the primary horizon's directory
requires_artifacts = pytest.mark.skipif(
    not (PROCESSED / "metadata.json").exists(),
    reason="run `python -m src.preprocessing.pipeline` first",
)


@requires_artifacts
def test_every_configured_horizon_was_generated(cfg):
    """h1/h6/h12/h24 must all exist, and differ only in sample count."""
    shapes = {}
    for h in cfg.horizons:
        d = cfg.processed_root / f"h{h}"
        assert (d / "metadata.json").exists(), f"missing {d}"
        meta = json.loads((d / "metadata.json").read_text())
        assert meta["horizon"] == h
        assert meta["window"] == cfg.window
        assert meta["feature_columns"] == cfg.feature_columns
        # Same split boundaries across horizons -- that is what makes them comparable.
        assert meta["stats"]["split_boundaries"]["val_start"] == \
               json.loads((cfg.processed_root / f"h{cfg.horizon}" / "metadata.json").read_text()
                          )["stats"]["split_boundaries"]["val_start"]
        shapes[h] = meta["stats"]["sequence_shapes"]["train"][0]
    # A longer horizon can only cost samples, never add them.
    ordered = sorted(shapes)
    assert all(shapes[a] > shapes[b] for a, b in zip(ordered, ordered[1:]))


@requires_artifacts
def test_real_artifacts_are_consistent(cfg):
    meta = json.loads((PROCESSED / "metadata.json").read_text())
    assert meta["feature_columns"] == cfg.feature_columns
    assert meta["window"] == cfg.window and meta["horizon"] == cfg.horizon

    n_feat = len(cfg.feature_columns)
    for split in pl.SPLITS:
        with np.load(PROCESSED / f"sequences_{split}.npz", allow_pickle=False) as z:
            assert z["X"].shape[1:] == (cfg.window, n_feat)
            assert z["X"].shape[0] == z["y_category"].shape[0]


@requires_artifacts
def test_real_scaler_round_trips(cfg):
    import joblib

    scaler = joblib.load(PROCESSED / "scaler.pkl")
    assert scaler.n_features_in_ == len(cfg.features)
    meta = json.loads((PROCESSED / "metadata.json").read_text())
    assert scaler.n_samples_seen_ == meta["stats"]["scaler"]["n_samples_seen"]
    assert meta["stats"]["scaler"]["fit_on"] == "train"


# ------------------------------------------------- imputation provenance side-cars


def test_sidecars_are_not_model_inputs(result, cfg):
    """The flags must never reach the model: not in feature_columns, not in X."""
    for col in pl.SIDECAR_COLUMNS:
        assert col not in cfg.feature_columns
        assert col not in cfg.features
    n_feat = len(cfg.feature_columns)
    for split in pl.SPLITS:
        assert result.sequences[split]["X"].shape[2] == n_feat
        # The flat view keeps them, but strictly after the label columns.
        tab = result.tabular[split]
        assert set(pl.SIDECAR_COLUMNS) <= set(tab.columns)
        assert tab.columns.get_loc(pl.IMPUTED) > tab.columns.get_loc("y_category")


def test_sidecars_present_in_both_formats_and_aligned(result, cfg):
    for split in pl.SPLITS:
        seq, tab = result.sequences[split], result.tabular[split]
        n = seq["X"].shape[0]
        for col in pl.SIDECAR_COLUMNS:
            assert seq[col].shape == (n,)
            assert seq[col].dtype == np.bool_
            np.testing.assert_array_equal(seq[col], tab[col].to_numpy(dtype=bool))
        assert seq["is_imputed_steps"].shape == (n, cfg.window)


def test_is_imputed_counts_match_the_ffill_counts_exactly(cfg, raw):
    """Per channel, the number of flagged cells must equal the raw NaN count."""
    df = raw.copy()
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)
    filled, stats = pl.fill_missing(df, cfg)

    for channel, n_flagged in stats["imputed_per_channel"].items():
        raw_nan = stats["nulls_before_ffill"][channel]
        unfillable = int(df.groupby("station")[channel].apply(
            lambda s: s.isna().cummin().sum()          # leading NaNs, nothing to fill from
        ).sum())
        assert n_flagged == raw_nan - unfillable, channel

    # PM2.5 in this fixture is only missing mid-series, so every NaN is flagged.
    assert stats["imputed_per_channel"]["PM2.5"] == stats["nulls_before_ffill"]["PM2.5"]
    # CO's leading NaNs were dropped, so they are not flagged as imputed.
    assert stats["imputed_per_channel"]["CO"] == 0


def test_is_imputed_marks_exactly_the_filled_cells(cfg):
    """Hand-built case: one interior PM2.5 gap, flagged on that row and nowhere else."""
    df = _synthetic(n_hours=200, stations=("Alpha",), with_nans=False)
    df.loc[[50, 51], "PM2.5"] = np.nan
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    out, stats = pl.fill_missing(df, cfg)

    assert stats["dropped"] == 0
    assert out[pl.IMPUTED_PM25].sum() == 2
    assert out[pl.IMPUTED].sum() == 2
    assert list(np.flatnonzero(out[pl.IMPUTED_PM25].to_numpy())) == [50, 51]
    # The filled value is the last observation carried forward, not a neighbour average.
    assert out.loc[50, "PM2.5"] == out.loc[49, "PM2.5"] == out.loc[51, "PM2.5"]


def test_is_imputed_input_flags_any_timestep_in_the_window(result, cfg):
    """`is_imputed_input` must equal the row-wise OR of the per-timestep mask."""
    for split in pl.SPLITS:
        seq = result.sequences[split]
        if seq["X"].shape[0] == 0:
            continue
        np.testing.assert_array_equal(
            seq[pl.IMPUTED_INPUT], seq["is_imputed_steps"].any(axis=1)
        )
        # An imputed label row does not by itself imply an imputed input window.
        assert seq[pl.IMPUTED_INPUT].sum() >= 0


def test_imputed_label_flag_implies_the_channel_flag(result, cfg):
    """PM2.5-imputed is a subset of any-channel-imputed."""
    for split in pl.SPLITS:
        seq = result.sequences[split]
        assert not (seq[pl.IMPUTED_PM25] & ~seq[pl.IMPUTED]).any()


def test_imputation_stats_reconcile_row_and_sample_level(result, cfg):
    imp = result.stats["imputation"]
    # Every sample-level total sits at or below its row-level total, and the shortfall
    # cannot exceed the boundary rows that never become a label row.
    for row_key, sample_key in (
        ("rows_imputed_any", "label_rows_imputed_any"),
        ("rows_imputed_pm25", "label_rows_imputed_pm25"),
    ):
        row_n, sample_n = imp["row_level"][row_key], imp["sample_level"][sample_key]
        assert sample_n <= row_n
        assert row_n - sample_n <= imp["boundary_rows_excluded"]

    for split in pl.SPLITS:
        st = imp["per_split"][split]
        assert st["label_row_imputed_pm25"] <= st["label_row_imputed_any"] <= st["samples"]


def test_report_states_the_imputation_limitation_and_val_caveat(result):
    report = pl.build_report(result)
    assert "must be disclosed as a limitation in the thesis" in report
    assert "### Validation set caveat" in report
    assert "Hazardous prevalence is inflated by imputation" in report
    assert "is_imputed_pm25" in report
    # The caveat must name both splits it is contrasting.
    caveat = report.split("### Validation set caveat")[1]
    assert "validation" in caveat.lower() and "test" in caveat.lower()
    assert "model selection" in caveat.lower()
