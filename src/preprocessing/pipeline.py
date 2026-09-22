"""Turn the raw Beijing multi-site table into model-ready splits.

Reads ``data/raw/beijing_multisite.csv`` (420,768 hourly rows x 12 stations) and writes
to ``data/processed``:

* ``scaler.pkl``               -- StandardScaler, **fit on the training split only**
* ``sequences_{split}.npz``    -- sliding windows (window=24 h, horizon=1 h)
* ``tabular_{split}.csv``      -- the same samples, one flat row each (baseline + CTGAN)
* ``metadata.json``            -- feature order, class names, shapes, provenance

Everything that defines the split -- chronological cut, station grouping, the
train/val/test fractions, the PM2.5 AQI breakpoints -- is read from
``configs/default.yaml``. This module does not define its own.

    python -m src.preprocessing.pipeline

Design notes
------------
**Station boundaries are hard boundaries.** Forward-fill runs per station, and no
sliding window ever spans two stations -- windows are built inside
``(station, split)`` partitions.

**Time gaps are hard boundaries too.** Rows dropped after forward-fill leave holes in
the hourly axis. A window is only emitted over a run of genuinely consecutive hours,
so a 24-step "day of history" is always a real day.

**The split cut is global, not per station.** All 12 stations share the identical
hourly index (2013-03-01 00:00 .. 2017-02-28 23:00), so the two cut timestamps are
taken from that shared axis. Every split therefore holds all 12 stations over the same
date range, and no timestamp appears in two splits.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
# Reports are horizon-suffixed (see PipelineConfig.report_path); there is no
# unsuffixed summary, because "the" summary stopped being well-defined at 4 horizons.

SPLITS = ("train", "val", "test")

# Side-car provenance columns. Evaluation-only: they are never part of
# ``PipelineConfig.feature_columns`` and never enter the model input tensor.
IMPUTED = "is_imputed"                    # any modelled channel forward-filled on this row
IMPUTED_PM25 = "is_imputed_pm25"          # PM2.5 specifically -- this is what sets the label
IMPUTED_INPUT = "is_imputed_input"        # any timestep of the sample's 24 h window imputed
SIDECAR_COLUMNS = (IMPUTED, IMPUTED_PM25, IMPUTED_INPUT)

# Channels present in the raw table but deliberately not carried into the feature set.
# Keyed by channel -> why a low-cost wearable cannot or should not use it.
DROPPED_FEATURES = {
    "SO2": "No low-cost SO2 sensor exists at neckband price/power. Electrochemical SO2 "
           "cells are ~$50-100, need a stable bias supply, and drift badly below 20 ug/m3 "
           "-- which is most of this dataset.",
    "NO2": "Low-cost NO2 electrochemical cells cross-sensitise strongly with O3 and need "
           "per-unit lab calibration; an uncalibrated channel would inject noise the model "
           "would learn as signal.",
    "O3":  "Same cross-sensitivity problem as NO2, and the pairing needed to separate them "
           "doubles the analog front end. Out of the ESP32 board's power budget.",
    "PRES": "The neckband has no barometer. Pressure is also near-constant over the minutes "
            "of a walking exposure episode, so it carries little within-session signal.",
    "RAIN": "Not a sensor channel -- it is a station observation. A wearable would have to "
            "infer it, which makes it a label-adjacent leak rather than an input.",
    "wd":  "Wind direction is meaningless for a body-worn device: the neckband moves and "
           "rotates with the wearer, so any fixed-station bearing does not transfer.",
    "WSPM": "Same as wd -- station anemometer readings do not describe the air moving past "
            "a walking person, and the neckband carries no anemometer.",
}


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class PipelineConfig:
    """The subset of ``configs/default.yaml`` this pipeline needs."""

    raw_file: Path
    processed_root: Path
    features: list[str]
    cyclical: list[str]
    breakpoints: list[float]
    labels: list[str]
    target: str
    group_key: str
    val_size: float
    test_size: float
    window: int
    horizon: int
    expected_rows: int
    seed: int
    horizons: tuple[int, ...] = ()

    @property
    def processed_dir(self) -> Path:
        """Per-horizon output directory: ``data/processed/h6`` and friends.

        Horizons share a raw table, a feature set and a split, but produce different
        samples. Writing them to one directory would silently overwrite.
        """
        return self.processed_root / f"h{self.horizon}"

    @property
    def report_path(self) -> Path:
        return REPO_ROOT / "reports" / f"preprocessing_summary_h{self.horizon}.md"

    @property
    def train_size(self) -> float:
        return 1.0 - self.val_size - self.test_size

    @property
    def cyclical_columns(self) -> list[str]:
        """Encoded cyclical column names, in order: hour_sin, hour_cos, month_sin, ..."""
        return [f"{c}_{fn}" for c in self.cyclical for fn in ("sin", "cos")]

    @property
    def feature_columns(self) -> list[str]:
        """Full model input order: scaled sensor channels, then cyclical encodings."""
        return [*self.features, *self.cyclical_columns]


def load_config(path: Path | str = DEFAULT_CONFIG, root: Path | None = None,
                horizon: int | None = None) -> PipelineConfig:
    """Read the YAML config. Paths in it are relative to the repo root.

    ``horizon`` overrides ``preprocessing.horizon`` so the same config can generate the
    secondary horizons; everything else -- features, split, breakpoints -- stays fixed,
    which is what makes the horizons comparable.
    """
    path = Path(path)
    root = root or REPO_ROOT
    raw = yaml.safe_load(path.read_text())
    data, prep = raw["data"], raw["preprocessing"]

    split_kind = data["split"]
    if split_kind != "chronological":
        raise ValueError(
            f"config sets data.split={split_kind!r}; this pipeline only implements "
            "'chronological' (a random split leaks future information -- see README)"
        )

    return PipelineConfig(
        raw_file=root / data["raw_file"],
        processed_root=root / data["processed_dir"],
        features=list(prep["features"]),
        cyclical=list(prep["cyclical"]),
        breakpoints=list(data["pm25_breakpoints"]),
        labels=list(data["pm25_labels"]),
        target=data["target"],
        group_key=data["group_key"],
        val_size=float(data["val_size"]),
        test_size=float(data["test_size"]),
        window=int(prep["window"]),
        horizon=int(horizon if horizon is not None else prep["horizon"]),
        horizons=tuple(int(h) for h in prep.get("horizons", [])),
        expected_rows=int(data["expected_rows"]),
        seed=int(raw["seed"]),
    )


# ----------------------------------------------------------------------------- steps


def load_raw(cfg: PipelineConfig, path: Path | None = None) -> pd.DataFrame:
    """Load the combined raw CSV and attach a proper datetime index column."""
    path = Path(path) if path is not None else cfg.raw_file
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    return df.sort_values([cfg.group_key, "datetime"], kind="mergesort").reset_index(drop=True)


def fill_missing(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    """Forward-fill within each station, then drop rows still missing a modelling column.

    The station boundary is respected: the groupby means a station's first hours are
    never filled from the previous station's last hours.

    Only the columns this pipeline actually models (``features`` + PM2.5, which is the
    label source) gate the drop. Dropping a row because O3 is missing would throw away
    usable wearable data for a channel the neckband does not even carry.
    """
    channels = [c for c in df.columns if df[c].dtype.kind in "fi" and c != "No"]
    was_nan = df["PM2.5"].isna().to_numpy()
    filled = df.copy()
    filled[channels] = (
        filled.groupby(cfg.group_key, sort=False, observed=True)[channels].ffill()
    )

    needed = sorted(set(cfg.features) | {"PM2.5"})
    before = len(filled)

    # Provenance: a row is "imputed" on a channel when that channel was NaN in the raw
    # table and forward-fill supplied a value. These are side-car columns for evaluation
    # -- they are deliberately NOT in cfg.feature_columns and never reach the model.
    per_channel = {}
    imputed_any = np.zeros(len(filled), dtype=bool)
    for c in needed:
        mask = df[c].isna().to_numpy() & filled[c].notna().to_numpy()
        per_channel[c] = int(mask.sum())
        imputed_any |= mask
        if c == "PM2.5":
            filled[IMPUTED_PM25] = mask
    filled[IMPUTED] = imputed_any

    out = filled.dropna(subset=needed).reset_index(drop=True)

    # Forward-fill is not label-neutral: the hours the reference monitors drop out are
    # not a random sample of hours. Quantify the shift so the report can state it.
    hazard_floor = 250.4
    pm_filled = filled.loc[was_nan & filled["PM2.5"].notna(), "PM2.5"]
    pm_original = df.loc[~was_nan, "PM2.5"]
    fill_bias = {
        "n_filled": int(len(pm_filled)),
        "filled_mean": round(float(pm_filled.mean()), 2) if len(pm_filled) else None,
        "original_mean": round(float(pm_original.mean()), 2) if len(pm_original) else None,
        "filled_hazardous_pct": round(float((pm_filled > hazard_floor).mean() * 100), 2)
        if len(pm_filled) else None,
        "original_hazardous_pct": round(float((pm_original > hazard_floor).mean() * 100), 2)
        if len(pm_original) else None,
    }

    stats = {
        "rows_in": before,
        "pm25_fill_bias": fill_bias,
        "rows_out": len(out),
        "dropped": before - len(out),
        "nulls_before_ffill": {c: int(df[c].isna().sum()) for c in needed},
        "nulls_after_ffill": {c: int(filled[c].isna().sum()) for c in needed},
        # Row-level imputation provenance. Because nothing is dropped on this dataset,
        # imputed_per_channel must equal nulls_before_ffill channel for channel.
        "imputed_per_channel": per_channel,
        "imputed_rows_any_channel": int(out[IMPUTED].sum()),
        "imputed_rows_pm25": int(out[IMPUTED_PM25].sum()),
        # For the report: what a strict dropna() over every column would have cost.
        "dropped_if_all_columns": before - len(filled.dropna()),
    }
    return out, stats


def add_cyclical(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Encode hour-of-day and month-of-year as (sin, cos) pairs.

    Hour 23 and hour 0 are one hour apart, and December and January are one month
    apart; a raw integer column tells a model the opposite.
    """
    # (period, offset) -- month is 1..12, hour is 0..23; both are normalised to start at 0.
    encoding = {"hour": (24, 0), "month": (12, 1)}
    out = df.copy()
    for col in cfg.cyclical:
        if col not in encoding:
            raise ValueError(f"no cyclical encoding defined for {col!r}")
        period, offset = encoding[col]
        theta = 2.0 * np.pi * (out[col] - offset) / period
        out[f"{col}_sin"] = np.sin(theta)
        out[f"{col}_cos"] = np.cos(theta)
    return out


def add_aqi_category(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    """Bin PM2.5 onto the config's AQI breakpoints -> an ordinal risk class.

    ``pm25_breakpoints`` holds the six lower edges; the top bin is open-ended.
    """
    bins = [*cfg.breakpoints, np.inf]
    if len(bins) - 1 != len(cfg.labels):
        raise ValueError(
            f"{len(cfg.breakpoints)} breakpoints imply {len(bins) - 1} bins but "
            f"{len(cfg.labels)} labels were given in data.pm25_labels"
        )
    out = df.copy()
    cat = pd.cut(out["PM2.5"], bins=bins, labels=cfg.labels, right=True)
    out[cfg.target] = cat
    out[f"{cfg.target}_idx"] = cat.cat.codes.astype("int16")
    if (out[f"{cfg.target}_idx"] < 0).any():
        raise ValueError("PM2.5 values fell outside the breakpoint range (negative PM2.5?)")
    return out


def assign_splits(df: pd.DataFrame, cfg: PipelineConfig) -> tuple[pd.DataFrame, dict]:
    """Label every row train/val/test by a chronological cut on the shared time axis.

    The cut points come from the *distinct timestamps* rather than row positions, so
    the 12 stations are split at the same two moments and one timestamp can never land
    in two splits.
    """
    stamps = np.sort(df["datetime"].unique())
    n = len(stamps)
    i_val = int(round(n * cfg.train_size))
    i_test = int(round(n * (cfg.train_size + cfg.val_size)))
    val_start, test_start = stamps[i_val], stamps[i_test]

    out = df.copy()
    out["split"] = np.where(
        out["datetime"] < val_start, "train",
        np.where(out["datetime"] < test_start, "val", "test"),
    )
    boundaries = {
        "val_start": pd.Timestamp(val_start).isoformat(),
        "test_start": pd.Timestamp(test_start).isoformat(),
        "n_timestamps": int(n),
        "timestamps_per_split": {
            "train": i_val, "val": i_test - i_val, "test": n - i_test,
        },
    }
    return out, boundaries


def fit_scaler(df: pd.DataFrame, cfg: PipelineConfig) -> StandardScaler:
    """Fit StandardScaler on the training rows only -- never on val or test.

    Only the physical sensor channels are scaled. The cyclical encodings are already
    bounded in [-1, 1] and standardising them would stretch the circle into an ellipse,
    destroying the equal-spacing property that is the whole point of the encoding.
    """
    train = df.loc[df["split"] == "train", cfg.features]
    if train.empty:
        raise ValueError("training split is empty -- cannot fit the scaler")
    return StandardScaler().fit(train.to_numpy())


def apply_scaler(df: pd.DataFrame, cfg: PipelineConfig, scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    out[cfg.features] = scaler.transform(out[cfg.features].to_numpy())
    return out


def _contiguous_runs(stamps: np.ndarray, step: np.timedelta64) -> list[tuple[int, int]]:
    """Split positions 0..len(stamps) into [start, stop) runs of consecutive stamps."""
    if len(stamps) == 0:
        return []
    breaks = np.flatnonzero(np.diff(stamps) != step) + 1
    edges = [0, *breaks.tolist(), len(stamps)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:])]


def make_sequences(df: pd.DataFrame, cfg: PipelineConfig, split: str) -> dict[str, np.ndarray]:
    """Build (window, horizon) sliding windows inside one split.

    A sample is ``window`` consecutive hours of features from one station, labelled by
    the AQI class and PM2.5 value ``horizon`` hours after the window's last step. Windows
    are built per ``(station, contiguous time run)``, so they never cross a station
    boundary, a split boundary, or a hole left by dropped rows.
    """
    cols = cfg.feature_columns
    w, h = cfg.window, cfg.horizon
    hour = np.timedelta64(1, "h")

    xs, y_idx, y_pm, y_pm_raw, meta_station, meta_time = [], [], [], [], [], []
    imp_tgt, imp_pm_tgt, imp_in, imp_steps = [], [], [], []
    part = df[df["split"] == split]

    for station, grp in part.groupby(cfg.group_key, sort=True, observed=True):
        grp = grp.sort_values("datetime", kind="mergesort")
        values = grp[cols].to_numpy(dtype=np.float32)
        labels = grp[f"{cfg.target}_idx"].to_numpy(dtype=np.int16)
        pm_scaled = grp["PM2.5"].to_numpy(dtype=np.float32)
        pm_raw = grp["PM2.5_raw"].to_numpy(dtype=np.float32)
        stamps = grp["datetime"].to_numpy(dtype="datetime64[ns]")
        imputed = grp[IMPUTED].to_numpy(dtype=bool)
        imputed_pm = grp[IMPUTED_PM25].to_numpy(dtype=bool)

        for start, stop in _contiguous_runs(stamps, hour):
            run = stop - start
            n_samples = run - w - h + 1
            if n_samples <= 0:
                continue
            # Vectorised window view: sample i covers [start+i, start+i+w).
            idx = start + np.arange(n_samples)[:, None] + np.arange(w)[None, :]
            xs.append(values[idx])
            tgt = start + np.arange(n_samples) + w + h - 1
            y_idx.append(labels[tgt])
            y_pm.append(pm_scaled[tgt])
            y_pm_raw.append(pm_raw[tgt])
            meta_station.append(np.repeat(station, n_samples))
            meta_time.append(stamps[tgt])
            # Provenance, aligned to the same sample axis as X.
            steps = imputed[idx]                       # (n_samples, window)
            imp_steps.append(steps)
            imp_in.append(steps.any(axis=1))
            imp_tgt.append(imputed[tgt])
            imp_pm_tgt.append(imputed_pm[tgt])

    if not xs:
        empty = np.empty((0, w, len(cols)), dtype=np.float32)
        return {
            "X": empty,
            "y_category": np.empty(0, dtype=np.int16),
            "y_pm25": np.empty(0, dtype=np.float32),
            "y_pm25_raw": np.empty(0, dtype=np.float32),
            "station": np.empty(0, dtype=object),
            "target_time": np.empty(0, dtype="datetime64[ns]"),
            IMPUTED: np.empty(0, dtype=bool),
            IMPUTED_PM25: np.empty(0, dtype=bool),
            IMPUTED_INPUT: np.empty(0, dtype=bool),
            "is_imputed_steps": np.empty((0, w), dtype=bool),
        }

    return {
        "X": np.concatenate(xs).astype(np.float32),
        "y_category": np.concatenate(y_idx).astype(np.int16),
        "y_pm25": np.concatenate(y_pm).astype(np.float32),
        "y_pm25_raw": np.concatenate(y_pm_raw).astype(np.float32),
        "station": np.concatenate(meta_station),
        "target_time": np.concatenate(meta_time),
        IMPUTED: np.concatenate(imp_tgt),
        IMPUTED_PM25: np.concatenate(imp_pm_tgt),
        IMPUTED_INPUT: np.concatenate(imp_in),
        "is_imputed_steps": np.concatenate(imp_steps),
    }


def make_tabular(df: pd.DataFrame, cfg: PipelineConfig, split: str) -> pd.DataFrame:
    """The flat, one-row-per-sample view of the same samples the sequences cover.

    Row = the features at the window's *last* step, labelled with the same t+horizon
    target. Keeping the sample set identical to ``make_sequences`` is what makes the
    baseline and the sequence model comparable -- they see the same rows, the baseline
    just sees one timestep of them instead of 24. This is also the shape CTGAN wants
    in Phase 4.
    """
    cols = cfg.feature_columns
    w, h = cfg.window, cfg.horizon
    hour = np.timedelta64(1, "h")

    frames = []
    part = df[df["split"] == split]

    for station, grp in part.groupby(cfg.group_key, sort=True, observed=True):
        grp = grp.sort_values("datetime", kind="mergesort").reset_index(drop=True)
        stamps = grp["datetime"].to_numpy(dtype="datetime64[ns]")
        imputed = grp[IMPUTED].to_numpy(dtype=bool)
        for start, stop in _contiguous_runs(stamps, hour):
            n_samples = (stop - start) - w - h + 1
            if n_samples <= 0:
                continue
            last = start + np.arange(n_samples) + w - 1   # window's final step
            tgt = last + h
            idx = start + np.arange(n_samples)[:, None] + np.arange(w)[None, :]
            block = grp.loc[last, cols].reset_index(drop=True)
            block.insert(0, "station", station)
            block.insert(1, "datetime", grp.loc[last, "datetime"].to_numpy())
            block["target_time"] = grp.loc[tgt, "datetime"].to_numpy()
            block["y_pm25"] = grp.loc[tgt, "PM2.5"].to_numpy()
            block["y_pm25_raw"] = grp.loc[tgt, "PM2.5_raw"].to_numpy()
            block[cfg.target] = grp.loc[tgt, cfg.target].to_numpy()
            block["y_category"] = grp.loc[tgt, f"{cfg.target}_idx"].to_numpy()
            # Side-car provenance -- evaluation only, never a model input.
            block[IMPUTED] = grp.loc[tgt, IMPUTED].to_numpy()
            block[IMPUTED_PM25] = grp.loc[tgt, IMPUTED_PM25].to_numpy()
            block[IMPUTED_INPUT] = imputed[idx].any(axis=1)
            frames.append(block)

    if not frames:
        return pd.DataFrame(
            columns=["station", "datetime", *cols, "target_time", "y_pm25",
                     "y_pm25_raw", cfg.target, "y_category", *SIDECAR_COLUMNS]
        )
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------------ orchestrate


@dataclass
class PipelineResult:
    config: PipelineConfig
    sequences: dict[str, dict[str, np.ndarray]]
    tabular: dict[str, pd.DataFrame]
    scaler: StandardScaler
    stats: dict = field(default_factory=dict)


def run(
    cfg: PipelineConfig | None = None,
    *,
    raw: pd.DataFrame | None = None,
    write: bool = True,
    report_path: Path | None = None,
    verbose: bool = True,
) -> PipelineResult:
    """Run the whole pipeline. ``raw`` lets tests inject a small frame instead of the CSV."""
    cfg = cfg or load_config()
    report_path = cfg.report_path if report_path is None else report_path
    say = print if verbose else (lambda *a, **k: None)
    stats: dict = {}

    df = raw.copy() if raw is not None else load_raw(cfg)
    if raw is None:
        if len(df) != cfg.expected_rows:
            raise ValueError(f"expected {cfg.expected_rows:,} raw rows, got {len(df):,}")
    else:
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        df = df.sort_values([cfg.group_key, "datetime"], kind="mergesort").reset_index(drop=True)

    stats["raw_rows"] = len(df)
    stats["stations"] = int(df[cfg.group_key].nunique())
    say(f"[1/6] loaded {len(df):,} rows x {stats['stations']} stations "
        f"(horizon={cfg.horizon}h, window={cfg.window}h)")

    df, fill_stats = fill_missing(df, cfg)
    stats["fill"] = fill_stats
    say(f"[2/6] ffill within station -> dropped {fill_stats['dropped']:,} rows "
        f"({fill_stats['dropped'] / max(fill_stats['rows_in'], 1) * 100:.3f}%), "
        f"{len(df):,} remain")

    df = add_cyclical(df, cfg)
    df = add_aqi_category(df, cfg)
    df["PM2.5_raw"] = df["PM2.5"]          # keep ug/m3 alongside the scaled column
    stats["features"] = cfg.feature_columns
    stats["dropped_features"] = DROPPED_FEATURES
    say(f"[3/6] features: {', '.join(cfg.feature_columns)}")

    df, boundaries = assign_splits(df, cfg)
    stats["split_boundaries"] = boundaries
    stats["rows_per_split"] = df["split"].value_counts().reindex(SPLITS).to_dict()
    say(f"[4/6] chronological split at {boundaries['val_start']} / {boundaries['test_start']}")

    scaler = fit_scaler(df, cfg)
    df = apply_scaler(df, cfg, scaler)
    stats["scaler"] = {
        "fit_on": "train",
        "n_samples_seen": int(scaler.n_samples_seen_),
        "mean": dict(zip(cfg.features, scaler.mean_.round(6).tolist())),
        "scale": dict(zip(cfg.features, scaler.scale_.round(6).tolist())),
    }
    say(f"[5/6] StandardScaler fit on {scaler.n_samples_seen_:,} train rows")

    sequences = {s: make_sequences(df, cfg, s) for s in SPLITS}
    tabular = {s: make_tabular(df, cfg, s) for s in SPLITS}
    stats["sequence_shapes"] = {s: list(sequences[s]["X"].shape) for s in SPLITS}
    stats["imputation"] = _imputation_stats(df, tabular, cfg)
    stats["class_distribution"] = _class_distribution(tabular, cfg)
    stats["class_distribution_overall"] = _overall_distribution(df, cfg)
    say("[6/6] sequences " + ", ".join(f"{s}={sequences[s]['X'].shape}" for s in SPLITS))

    result = PipelineResult(cfg, sequences, tabular, scaler, stats)

    if write:
        _write_outputs(result)
        say(f"      wrote artifacts to {cfg.processed_dir}")
    if write and report_path is not None:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(build_report(result))
        say(f"      wrote report to {report_path}")
    return result


def _imputation_stats(df: pd.DataFrame, tabular: dict[str, pd.DataFrame],
                      cfg: PipelineConfig) -> dict:
    """Reconcile the side-car flags against the raw NaN counts, and per split.

    Row level and sample level differ by construction: a row becomes a *label row* for
    at most one sample, and the first ``window + horizon - 1`` rows of each
    ``(station, split)`` partition never become one. So the sample-level totals sit
    below the row-level totals, and the gap is exactly those boundary rows.
    """
    hazardous = cfg.labels[-1]
    per_split = {}
    for split, frame in tabular.items():
        n = len(frame)
        imp = frame[IMPUTED_PM25].to_numpy(dtype=bool)
        haz = (frame[cfg.target].to_numpy() == hazardous)
        per_split[split] = {
            "samples": n,
            "label_row_imputed_any": int(frame[IMPUTED].sum()),
            "label_row_imputed_pm25": int(imp.sum()),
            "input_window_imputed": int(frame[IMPUTED_INPUT].sum()),
            "input_window_imputed_pct": round(float(frame[IMPUTED_INPUT].mean() * 100), 2),
            # The thesis limitation, measured per split rather than asserted.
            "hazardous_pct_imputed_labels": round(float(haz[imp].mean() * 100), 2)
            if imp.any() else None,
            "hazardous_pct_observed_labels": round(float(haz[~imp].mean() * 100), 2)
            if (~imp).any() else None,
        }

    row_level = {
        "rows_imputed_any": int(df[IMPUTED].sum()),
        "rows_imputed_pm25": int(df[IMPUTED_PM25].sum()),
    }
    sample_level = {
        "label_rows_imputed_any": sum(v["label_row_imputed_any"] for v in per_split.values()),
        "label_rows_imputed_pm25": sum(v["label_row_imputed_pm25"] for v in per_split.values()),
    }
    return {
        "per_split": per_split,
        "row_level": row_level,
        "sample_level": sample_level,
        "boundary_rows_excluded": (cfg.window + cfg.horizon - 1)
        * int(df[cfg.group_key].nunique()) * len(SPLITS),
    }


def _class_distribution(tabular: dict[str, pd.DataFrame], cfg: PipelineConfig) -> dict:
    out = {}
    for split, frame in tabular.items():
        counts = frame[cfg.target].value_counts().reindex(cfg.labels).fillna(0).astype(int)
        total = max(int(counts.sum()), 1)
        out[split] = {
            label: {"n": int(counts[label]), "pct": round(counts[label] / total * 100, 2)}
            for label in cfg.labels
        }
    return out


def _overall_distribution(df: pd.DataFrame, cfg: PipelineConfig) -> dict:
    counts = df[cfg.target].value_counts().reindex(cfg.labels).fillna(0).astype(int)
    total = max(int(counts.sum()), 1)
    return {
        label: {"n": int(counts[label]), "pct": round(counts[label] / total * 100, 2)}
        for label in cfg.labels
    }


def _write_outputs(result: PipelineResult) -> None:
    cfg = result.config
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(result.scaler, cfg.processed_dir / "scaler.pkl")
    for split in SPLITS:
        np.savez_compressed(cfg.processed_dir / f"sequences_{split}.npz", **result.sequences[split])
        result.tabular[split].to_csv(cfg.processed_dir / f"tabular_{split}.csv", index=False)

    meta = {
        "feature_columns": cfg.feature_columns,
        "scaled_columns": cfg.features,
        "unscaled_columns": cfg.cyclical_columns,
        # Present in every split, in both formats, and excluded from feature_columns:
        # evaluation/analysis side-cars, not model inputs.
        "sidecar_columns": list(SIDECAR_COLUMNS),
        "sidecar_note": (
            "is_imputed / is_imputed_pm25 flag the sample's LABEL row (t+horizon); "
            "is_imputed_input flags whether any timestep of the 24 h input window was "
            "forward-filled. sequences_*.npz additionally carries is_imputed_steps "
            "(n, window) per-timestep. None of these are model inputs."
        ),
        "class_labels": cfg.labels,
        "window": cfg.window,
        "horizon": cfg.horizon,
        "split_fractions": {"train": cfg.train_size, "val": cfg.val_size, "test": cfg.test_size},
        "dropped_features": DROPPED_FEATURES,
        "stats": result.stats,
    }
    (cfg.processed_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))


# ---------------------------------------------------------------------------- report


def build_report(result: PipelineResult) -> str:
    cfg, st = result.config, result.stats
    fill, bounds = st["fill"], st["split_boundaries"]
    bias = fill["pm25_fill_bias"]
    imp = st["imputation"]
    imp_split, imp_row, imp_sample = imp["per_split"], imp["row_level"], imp["sample_level"]

    # Row-level flags must reconcile channel-for-channel against the raw NaN counts.
    recon_rows = [
        [f"`{c}`", f"{fill['nulls_before_ffill'][c]:,}", f"{fill['imputed_per_channel'][c]:,}",
         "match" if fill["nulls_before_ffill"][c] == fill["imputed_per_channel"][c] else "MISMATCH"]
        for c in sorted(fill["imputed_per_channel"])
    ]
    sidecar_rows = [
        [f"`{IMPUTED}`", "label row (t+h) had any modelled channel forward-filled",
         *[f"{imp_split[s]['label_row_imputed_any']:,}" for s in SPLITS]],
        [f"`{IMPUTED_PM25}`", "label row had **PM2.5** forward-filled -- this sets the class",
         *[f"{imp_split[s]['label_row_imputed_pm25']:,}" for s in SPLITS]],
        [f"`{IMPUTED_INPUT}`", "any of the 24 input hours was forward-filled",
         *[f"{imp_split[s]['input_window_imputed']:,} "
           f"({imp_split[s]['input_window_imputed_pct']:.1f}%)" for s in SPLITS]],
    ]
    def _pct(v) -> str:
        return "n/a" if v is None else f"{v:.2f}%"

    def _ratio(a, b) -> str:
        return "n/a" if (a is None or not b) else f"{a / b:.1f}x"

    haz_rows = [
        [s,
         _pct(imp_split[s]["hazardous_pct_imputed_labels"]),
         _pct(imp_split[s]["hazardous_pct_observed_labels"]),
         _ratio(imp_split[s]["hazardous_pct_imputed_labels"],
                imp_split[s]["hazardous_pct_observed_labels"])]
        for s in SPLITS
    ]

    # The limitation only applies if forward-fill actually supplied PM2.5 values.
    if bias["n_filled"]:
        limitation = f"""**This must be disclosed as a limitation in the thesis.**

The Hazardous class is over-represented in the data relative to what was actually
*measured*, because forward-fill is not label-neutral. Across the dataset,
**{bias['filled_hazardous_pct']:.2f}% of forward-filled PM2.5 readings fall in Hazardous
against {bias['original_hazardous_pct']:.2f}% of observed readings** -- {_ratio(bias['filled_hazardous_pct'], bias['original_hazardous_pct'])} the rate.
The carried-forward values average {bias['filled_mean']:.1f} ug/m3 versus
{bias['original_mean']:.1f} for observed hours."""
    else:
        limitation = (
            "Forward-fill supplied no PM2.5 values on this run, so the class "
            "distribution carries no imputation bias."
        )

    L = cfg.labels

    def table(header: list[str], rows: list[list[str]]) -> str:
        # A literal '|' in a cell would silently split it into two columns.
        esc = lambda cells: [str(c).replace("|", "\\|") for c in cells]
        sep = ["---"] * len(header)
        body = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return f"| {' | '.join(esc(header))} |\n| {' | '.join(sep)} |\n{body}"

    # --- row counts at each step
    steps = [
        ["0. raw CSV", f"{st['raw_rows']:,}", "as downloaded, 12 stations x 35,064 h"],
        ["1. forward-fill (per station)", f"{fill['rows_in']:,}", "ffill never crosses a station boundary"],
        ["2. drop remaining nulls", f"{fill['rows_out']:,}",
         f"-{fill['dropped']:,} rows ({fill['dropped'] / max(fill['rows_in'], 1) * 100:.3f}%)"],
        ["3. + cyclical / AQI label", f"{fill['rows_out']:,}", "columns added, no rows lost"],
    ]
    for s in SPLITS:
        steps.append([f"4. split -> {s}", f"{st['rows_per_split'][s]:,}", "chronological, all 12 stations"])
    for s in SPLITS:
        shape = st["sequence_shapes"][s]
        steps.append([f"5. windows -> {s}", f"{shape[0]:,}", f"shape {tuple(shape)}"])

    # --- per-split class distribution
    dist = st["class_distribution"]
    overall = st["class_distribution_overall"]
    v_val = dist["val"]["Very unhealthy"]["pct"]
    v_test = dist["test"]["Very unhealthy"]["pct"]
    v_gap = _ratio(max(v_val, v_test), min(v_val, v_test))
    dist_rows = []
    for label in L:
        row = [f"**{label}**" if label in ("Very unhealthy", "Hazardous") else label]
        row.append(f"{overall[label]['pct']:.2f}%")
        for s in SPLITS:
            row.append(f"{dist[s][label]['n']:,} ({dist[s][label]['pct']:.2f}%)")
        dist_rows.append(row)

    dropped_rows = [[f"`{k}`", v] for k, v in st["dropped_features"].items()]
    scaler_rows = [
        [f"`{f}`", f"{st['scaler']['mean'][f]:.4f}", f"{st['scaler']['scale'][f]:.4f}"]
        for f in cfg.features
    ]

    primary_h = yaml.safe_load(DEFAULT_CONFIG.read_text())["preprocessing"]["horizon"]
    if cfg.horizon == primary_h:
        role = ("> **This is the primary dataset.** Every later phase -- CTGAN, the "
                "sequence model, SHAP, the LLM advisory -- trains and reports on it.")
    else:
        role = (f"> **Secondary horizon, comparison only.** The primary dataset is "
                f"h{primary_h} (`reports/preprocessing_summary_h{primary_h}.md`). This one "
                f"exists for the multi-horizon figure in Phase 9; do not select models or "
                f"report headline results on it.")

    n_feat = len(cfg.feature_columns)
    artifact_rows = [
        ["`data/processed/scaler.pkl`", "StandardScaler, fit on train only"],
        ["`data/processed/sequences_<split>.npz`",
         f"`X` (n, {cfg.window}, {n_feat}), `y_category`, `y_pm25` (scaled), "
         f"`y_pm25_raw` (ug/m3), `station`, `target_time`, and the side-cars "
         f"`{IMPUTED}`, `{IMPUTED_PM25}`, `{IMPUTED_INPUT}`, "
         f"`is_imputed_steps` (n, {cfg.window})"],
        ["`data/processed/tabular_<split>.csv`",
         f"same samples, flat -- the window's last step plus the t+{cfg.horizon} h label; "
         f"baseline model and CTGAN input. Carries `{IMPUTED}`, `{IMPUTED_PM25}`, "
         f"`{IMPUTED_INPUT}` as trailing columns"],
        ["`data/processed/metadata.json`", "feature order, class labels, shapes, scaler stats"],
    ]

    if fill["dropped"]:
        drop_note = (
            f"**{fill['dropped']:,} rows dropped** -- the leading NaNs at the start of a "
            "station's series, which forward-fill has nothing to fill from."
        )
    else:
        drop_note = (
            "**No rows were dropped.** Forward-fill within each station resolved every "
            "NaN in the five modelled channels: each station's first hour already carries "
            "all five, so there is no unfillable leading gap, and every later gap has a "
            "prior in-station observation to carry forward. The row count is therefore "
            "unchanged from the raw table through to the split."
        )

    haz = {s: dist[s]["Hazardous"] for s in SPLITS}
    vun = {s: dist[s]["Very unhealthy"] for s in SPLITS}

    return f"""# Preprocessing summary -- horizon {cfg.horizon} h

{role}

Generated by `src/preprocessing/pipeline.py` from `configs/default.yaml`.
Source: `{cfg.raw_file.relative_to(REPO_ROOT)}` -> `{cfg.processed_dir.relative_to(REPO_ROOT)}/`.

Window **{cfg.window} h**, horizon **{cfg.horizon} h**, split
**{cfg.train_size:.0%} / {cfg.val_size:.0%} / {cfg.test_size:.0%}** chronological,
grouped by `{cfg.group_key}`.

Everything except the horizon is identical across
`h{"`, `h".join(str(h) for h in cfg.horizons)}` -- same raw table, same five wearable
channels, same chronological split boundaries -- so the horizons differ only in how far
ahead the label sits. That is what makes them comparable.

---

## 1. Row counts at each step

{table(["Step", "Rows", "Note"], steps)}

### Missingness handled

Missing values are true `NaN` (no sentinel encoding). Forward-fill runs inside each
station group, so a station's opening hours are never filled from the previous
station's closing hours.

{table(["Channel", "NaN before ffill", "NaN after ffill"],
       [[f"`{c}`", f"{fill['nulls_before_ffill'][c]:,}", f"{fill['nulls_after_ffill'][c]:,}"]
        for c in sorted(fill['nulls_before_ffill'])])}

{drop_note}

Only the modelled columns gate the drop. A strict `dropna()` across *every* raw column
would have cost **{fill['dropped_if_all_columns']:,} rows** instead, discarding usable
wearable data over channels (SO2, NO2, O3, ...) the neckband does not carry.

### Imputation provenance (`is_imputed` side-cars)

Every split carries provenance flags, in **both** the tabular and the sequence format.
They are **evaluation-only**: none of them appears in `feature_columns`, and the model
input tensor stays `(n, {cfg.window}, {len(cfg.feature_columns)})`. Feeding them would let
the model condition on monitor downtime, which the neckband cannot observe.

{table(["Column", "Marks", *[f"{s}" for s in SPLITS]], sidecar_rows)}

`sequences_*.npz` additionally carries `is_imputed_steps`, an
`(n, {cfg.window})` per-timestep mask, for anyone who needs finer granularity than
"the window contains an imputed hour".

**Reconciliation against the raw NaN counts** -- because no rows were dropped, every
forward-filled cell must show up in exactly one flagged row:

{table(["Channel", "NaN in raw", "Flagged imputed", "Status"], recon_rows)}

Row level vs sample level: **{imp_row['rows_imputed_pm25']:,}** rows carry an imputed
PM2.5, of which **{imp_sample['label_rows_imputed_pm25']:,}** become the label row of a
sample (**{imp_row['rows_imputed_any']:,}** / **{imp_sample['label_rows_imputed_any']:,}**
for any modelled channel). The gap is the
{imp['boundary_rows_excluded']:,} boundary rows per station per split that can never
complete a window -- it is not loss.

Note the third row of that table: roughly a quarter to a third of all input windows
touch at least one forward-filled hour, even though only ~2-5% of individual cells were
filled. A 24-hour window is a wide net.

---

## 2. Feature set

Model input order ({len(cfg.feature_columns)} channels):
{', '.join(f'`{c}`' for c in cfg.feature_columns)}

`DEWP` is the humidity proxy -- the neckband carries a DHT-class temperature/humidity
sensor, and dew point converts to relative humidity given `TEMP`.

`hour` and `month` are encoded as (sin, cos) pairs because hour 23 and hour 0 are one
hour apart, and December and January are one month apart -- an integer column asserts
the opposite.

### Dropped features and why

{table(["Channel", "Why it is out of scope for a low-cost wearable"], dropped_rows)}

---

## 3. Split

Chronological cut on the shared hourly axis ({bounds['n_timestamps']:,} distinct
timestamps), so all 12 stations are cut at the same two moments and no timestamp
appears in two splits.

- `train` -- up to **{bounds['val_start']}** ({bounds['timestamps_per_split']['train']:,} hours)
- `val`   -- **{bounds['val_start']}** to **{bounds['test_start']}** ({bounds['timestamps_per_split']['val']:,} hours)
- `test`  -- **{bounds['test_start']}** onward ({bounds['timestamps_per_split']['test']:,} hours)

Sliding windows are built inside `(station, contiguous-hour run, split)` partitions, so
no window spans two stations, two splits, or a gap left by a dropped row.

### Scaler

`StandardScaler` fit on **train rows only** ({st['scaler']['n_samples_seen']:,} samples),
saved to `data/processed/scaler.pkl`. Val and test are transformed with those statistics,
never re-fit. The cyclical (sin, cos) columns are left unscaled -- standardising them
would stretch the unit circle into an ellipse and destroy the equal-spacing that is the
point of the encoding.

{table(["Channel", "train mean", "train scale"], scaler_rows)}

---

## 4. Class distribution per split

AQI risk category from the `pm25_breakpoints` in `configs/default.yaml`, assigned at the
prediction target (t + {cfg.horizon} h) of every sample.

{table(["Class", "Overall", *[f"{s} n (%)" for s in SPLITS]], dist_rows)}

The *Overall* column is measured after forward-fill, over all {fill['rows_out']:,} rows,
so it sits slightly above the raw-data figures in
`notebooks/01_eda_beijing_multisite.ipynb` (Hazardous 4.47%, Very unhealthy 10.74%), which
exclude the {bias['n_filled']:,} rows where PM2.5 was NaN. That gap is not rounding:
**forward-fill is not label-neutral here.** The carried-forward values average
{bias['filled_mean']:.1f} ug/m3 against {bias['original_mean']:.1f} for observed hours, and
{bias['filled_hazardous_pct']:.2f}% of them land in Hazardous against
{bias['original_hazardous_pct']:.2f}% of observed hours -- roughly double. Reference monitors
drop out disproportionately during severe episodes, so imputation concentrates in exactly
the class Phase 4 cares about. Treat the Hazardous counts below as carrying imputed mass,
and if that becomes a problem for the GAN, the fix is to flag imputed rows rather than to
stop filling.

The split columns are measured over sequence samples, so they exclude the
{cfg.window + cfg.horizon - 1} boundary hours per station per split that cannot complete a
window ({(cfg.window + cfg.horizon - 1) * 12 * 3:,} samples in total).

### What Phase 4's GAN has to target

The two classes that trigger an actual health advisory are also the two the model will
see least:

- **Hazardous** -- {overall['Hazardous']['pct']:.2f}% overall;
  train {haz['train']['pct']:.2f}% ({haz['train']['n']:,}),
  val {haz['val']['pct']:.2f}% ({haz['val']['n']:,}),
  test {haz['test']['pct']:.2f}% ({haz['test']['n']:,}).
- **Very unhealthy** -- {overall['Very unhealthy']['pct']:.2f}% overall;
  train {vun['train']['pct']:.2f}% ({vun['train']['n']:,}),
  val {vun['val']['pct']:.2f}% ({vun['val']['n']:,}),
  test {vun['test']['pct']:.2f}% ({vun['test']['n']:,}).

Because the split is chronological rather than stratified, these two classes are **not**
evenly spread across the splits -- Beijing's severe-pollution episodes are seasonal, so
which winters land in which split moves the rare-class share substantially. The per-split
numbers above, not the overall rate, are what CTGAN rebalancing should be sized against,
and CTGAN must be fit on `tabular_train.csv` alone.

### Limitation: Hazardous prevalence is inflated by imputation

{limitation}

The mechanism is not incidental: reference monitors drop out disproportionately *during*
severe episodes, so the hours with no reading are precisely the hours most likely to have
been hazardous. Forward-fill then propagates the last pre-outage value across them.

The effect reproduces inside every split, measured on each sample's label row:

{table(["Split", "Hazardous, imputed labels", "Hazardous, observed labels", "Ratio"], haz_rows)}

What this means concretely:

- Some fraction of the Hazardous labels the model trains on and is scored against are
  **imputed, not observed**. Reported Hazardous recall is therefore partly recall on
  carried-forward values.
- **CTGAN will inherit this.** Fitting on `tabular_train.csv` without filtering means the
  generator learns a Hazardous conditional that is itself partly an imputation artifact,
  and synthesising more of it amplifies the bias rather than correcting the imbalance.
- Report headline metrics twice -- all samples, and `is_imputed_pm25 == False` only. The
  side-car column exists for exactly this. A large gap between the two is a finding, not
  a bug.

The honest framing for the write-up is that imputation was chosen over deletion to keep
the hourly sequence contiguous (deletion would fragment the 24 h windows), and the cost
is a measured upward bias in the rarest and most safety-relevant class.

### Validation set caveat

**Very-unhealthy prevalence differs substantially between validation
({v_val:.2f}%) and test ({v_test:.2f}%)** -- a {v_gap}
gap, and validation also sits below its own training share
({dist['train']['Very unhealthy']['pct']:.2f}%). This is a direct consequence of the
chronological split: the validation window
({bounds['val_start'][:10]} to {bounds['test_start'][:10]}) covers a different slice of the
seasonal cycle than the test window ({bounds['test_start'][:10]} onward), and severe
episodes in Beijing are winter-loaded.

Any model selection done on validation metrics should be interpreted with this in mind:

- Validation is **not** an unbiased preview of test performance for the rare classes.
  Macro-F1, per-class recall, and anything else that weights Very-unhealthy or Hazardous
  heavily will move between the two splits for reasons that have nothing to do with the
  model.
- Early stopping and hyperparameter choices driven by validation macro-F1 are implicitly
  tuned to a validation-specific class mix. Prefer selection criteria that are stable
  under prevalence shift (balanced accuracy, per-class recall read individually, or a
  threshold-free measure such as AUC-PR per class).
- Do not read a val -> test drop in rare-class metrics as overfitting without first
  checking it against this prevalence gap.

A prevalence-matched or rolling-origin validation scheme would remove the ambiguity, at
the cost of departing from the single chronological split the project has standardised
on. That trade-off is left open rather than decided here.

---

## 5. Artifacts

{table(["File", "Contents"], artifact_rows)}

`<split>` is one of `train`, `val`, `test`.

The side-car columns are **not** model inputs. Load features with
`metadata.json -> feature_columns`, never by taking every numeric column in the CSV.

Reproduce this horizon with `python -m src.preprocessing.pipeline --horizon {cfg.horizon}`,
or all of them with `--all-horizons`; verify with
`pytest src/preprocessing/test_pipeline.py`.
"""


# ------------------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--horizon", type=int, default=None,
                    help="override preprocessing.horizon (default: the config's primary)")
    ap.add_argument("--all-horizons", action="store_true",
                    help="generate every horizon in preprocessing.horizons")
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--no-write", action="store_true", help="compute but write nothing")
    args = ap.parse_args(argv)

    if args.all_horizons:
        primary = load_config(args.config)
        targets = primary.horizons or (primary.horizon,)
        raw = load_raw(primary)                     # read the 420k-row CSV once
        if len(raw) != primary.expected_rows:
            raise ValueError(f"expected {primary.expected_rows:,} raw rows, got {len(raw):,}")
        for h in targets:
            cfg = load_config(args.config, horizon=h)
            marker = "  <- PRIMARY" if h == primary.horizon else ""
            print(f"\n=== horizon {h}h ==={marker}")
            run(cfg, raw=raw, write=not args.no_write)
        return 0

    cfg = load_config(args.config, horizon=args.horizon)
    run(cfg, write=not args.no_write, report_path=args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
