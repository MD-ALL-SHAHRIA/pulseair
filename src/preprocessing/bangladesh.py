"""Phase 10 — prepare the Bangladesh AQI data using the Beijing pipeline's conventions.

External validation set for the wearable's actual target population. Mendeley Data
`9j447cynb9` v2, downloaded manually to ``data/raw/bangladesh_aqi.csv`` (see the
README's Data section); ``audit()`` re-derives every integrity number from that file.

**Read the data audit in ``reports/bangladesh_validation.md`` before using this.** The
published dataset covers 2000-2025, but only the portion from **2022-08-05** onward
survives inspection; the earlier 87% of Dhaka's rows carry a near-linear synthetic
trend, a hard clip at exactly 250.0 ug/m3, and carbon monoxide in different units.
``CLEAN_START`` is where this module begins reading, and the audit numbers are
recomputed by ``audit()`` rather than asserted.

Conventions carried over from Phase 2 unchanged: per-group forward fill with an
``is_imputed`` provenance flag, cyclical hour/month encodings, the same EPA PM2.5
breakpoints, a chronological split on the shared hourly axis, and window/horizon from
``configs/default.yaml``.

One constraint the Beijing pipeline did not have: **this dataset has no temperature or
dew point**, so the nine-channel Beijing feature set cannot be reproduced. The shared
subset is PM2.5, PM10, CO plus the four cyclical encodings -- seven features.

    python -m src.preprocessing.bangladesh

Writes ``data/processed/bd_h6/``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
RAW = REPO_ROOT / "data" / "raw" / "bangladesh_aqi.csv"

# Everything before this timestamp fails the audit -- see module docstring.
CLEAN_START = pd.Timestamp("2022-08-05")

# Four major metros, each with complete hourly coverage over the clean window.
CITIES = ("Dhaka", "Chittagong", "Comilla", "Barisāl")

# Column names in the Mendeley file -> the names the Beijing pipeline uses.
RENAME = {"pm2_5": "PM2.5", "pm10": "PM10", "carbon_monoxide": "CO",
          "carbon_dioxide": "CO2", "nitrogen_dioxide": "NO2",
          "sulphur_dioxide": "SO2", "ozone": "O3", "city_name": "station"}

# The Beijing model's nine channels minus the two this dataset does not carry.
SHARED_FEATURES = ["PM2.5", "PM10", "CO"]
CYCLICAL = ["hour", "month"]


@dataclass(frozen=True)
class BDConfig:
    out_dir: Path
    labels: list[str]
    breakpoints: list[float]
    window: int
    horizon: int
    val_size: float
    test_size: float
    seed: int

    @property
    def train_size(self) -> float:
        return 1.0 - self.val_size - self.test_size

    @property
    def cyclical_columns(self) -> list[str]:
        return [f"{c}_{fn}" for c in CYCLICAL for fn in ("sin", "cos")]

    @property
    def feature_columns(self) -> list[str]:
        return [*SHARED_FEATURES, *self.cyclical_columns]


def load_config(path: Path | str = DEFAULT_CONFIG) -> BDConfig:
    raw = yaml.safe_load(Path(path).read_text())
    data, prep = raw["data"], raw["preprocessing"]
    return BDConfig(
        out_dir=REPO_ROOT / data["processed_dir"] / f"bd_h{prep['horizon']}",
        labels=list(data["pm25_labels"]),
        breakpoints=list(data["pm25_breakpoints"]),
        window=int(prep["window"]), horizon=int(prep["horizon"]),
        val_size=float(data["val_size"]), test_size=float(data["test_size"]),
        seed=int(raw["seed"]),
    )


# ------------------------------------------------------------------ data audit


def audit(path: Path = RAW) -> dict:
    """Recompute the evidence that dates the usable window. Nothing here is asserted."""
    df = pd.read_csv(path, parse_dates=["datetime"])
    dhaka = df[df.city_name == "Dhaka"].sort_values("datetime")
    dhaka = dhaka.assign(yr=dhaka.datetime.dt.year)

    pre = dhaka[dhaka.datetime < CLEAN_START]
    post = dhaka[dhaka.datetime >= CLEAN_START]

    med = pre[pre.yr <= 2021].groupby("yr")["pm2_5"].median()
    x, y = med.index.values.astype(float), med.values
    coef = np.polyfit(x, y, 1)
    r2 = 1 - ((y - np.polyval(coef, x)) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    clean = df[df.datetime >= CLEAN_START]
    return {
        "file_rows": int(len(df)),
        "stated_cities": 103,
        "actual_cities": int(df.city_name.nunique()),
        "stated_range": "2000-2025",
        "actual_range": [str(df.datetime.min()), str(df.datetime.max())],
        "clean_start": str(CLEAN_START),
        "clean_rows": int(len(clean)),
        "clean_pct": round(100 * len(clean) / len(df), 1),
        "clean_cities": int(clean.city_name.nunique()),
        "clean_range": [str(clean.datetime.min()), str(clean.datetime.max())],
        "dhaka_pm25_linear_trend_r2": round(float(r2), 6),
        "dhaka_pm25_trend_slope": round(float(coef[0]), 3),
        "pre_clip_at_250_pct": round(
            float(100 * (pre[pre.yr >= 2018]["pm2_5"] >= 249.99).mean()), 2),
        "co_median_pre": round(float(pre.carbon_monoxide.median()), 2),
        "co_median_post": round(float(post.carbon_monoxide.median()), 1),
        "pm25_lag1_pre": round(float(pre.pm2_5.autocorr(1)), 4),
        "pm25_lag1_post": round(float(post.pm2_5.autocorr(1)), 4),
        "co2_present_pct_clean": round(
            float(100 * (1 - clean.carbon_dioxide.isna().mean())), 2),
        "has_temperature": bool(any("temp" in c.lower() for c in df.columns)),
        "has_dewpoint": bool(any("dew" in c.lower() for c in df.columns)),
        "channels_present": [c for c in df.columns],
    }


# ------------------------------------------------------------------- pipeline


def build(cfg: BDConfig, cities=CITIES, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    df = pd.read_csv(RAW, parse_dates=["datetime"])
    df = df.rename(columns=RENAME)
    df = df[(df.datetime >= CLEAN_START) & (df.station.isin(cities))]
    df = df.sort_values(["station", "datetime"], kind="mergesort").reset_index(drop=True)
    say(f"{len(df):,} rows, {df.station.nunique()} cities, "
        f"{df.datetime.min().date()} .. {df.datetime.max().date()}")

    stats = {"rows_in": int(len(df)), "cities": sorted(df.station.unique())}

    # --- forward fill within city, with provenance (Phase 2 convention) ---------
    needed = [*SHARED_FEATURES]
    nulls_before = {c: int(df[c].isna().sum()) for c in needed}
    filled = df.copy()
    filled[needed] = filled.groupby("station", sort=False)[needed].ffill()
    imputed_any = np.zeros(len(filled), dtype=bool)
    per_channel = {}
    for c in needed:
        m = df[c].isna().to_numpy() & filled[c].notna().to_numpy()
        per_channel[c] = int(m.sum())
        imputed_any |= m
        if c == "PM2.5":
            filled["is_imputed_pm25"] = m
    filled["is_imputed"] = imputed_any
    out = filled.dropna(subset=needed).reset_index(drop=True)
    stats.update(nulls_before_ffill=nulls_before, imputed_per_channel=per_channel,
                 rows_dropped=int(len(filled) - len(out)), rows_out=int(len(out)))
    say(f"ffill within city -> {stats['rows_dropped']:,} rows dropped, "
        f"{len(out):,} remain")

    # --- cyclical encodings (identical formula to Phase 2) ----------------------
    for col, (period, offset) in {"hour": (24, 0), "month": (12, 1)}.items():
        v = out.datetime.dt.hour if col == "hour" else out.datetime.dt.month
        theta = 2.0 * np.pi * (v - offset) / period
        out[f"{col}_sin"], out[f"{col}_cos"] = np.sin(theta), np.cos(theta)

    # --- AQI category from the same EPA breakpoints -----------------------------
    bins = [*cfg.breakpoints, np.inf]
    cat = pd.cut(out["PM2.5"], bins=bins, labels=cfg.labels, right=True)
    out["pm25_category"] = cat
    out["y_category_now"] = cat.cat.codes.astype("int16")
    out["PM2.5_raw"] = out["PM2.5"]

    # --- chronological split on the shared hourly axis --------------------------
    stamps = np.sort(out.datetime.unique())
    n = len(stamps)
    i_val = int(round(n * cfg.train_size))
    i_test = int(round(n * (cfg.train_size + cfg.val_size)))
    val_start, test_start = stamps[i_val], stamps[i_test]
    out["split"] = np.where(out.datetime < val_start, "train",
                            np.where(out.datetime < test_start, "val", "test"))
    stats["split_boundaries"] = {"val_start": str(pd.Timestamp(val_start)),
                                 "test_start": str(pd.Timestamp(test_start)),
                                 "n_timestamps": int(n)}
    say(f"split at {pd.Timestamp(val_start).date()} / {pd.Timestamp(test_start).date()}")

    # --- scaler on train rows only ----------------------------------------------
    scaler = StandardScaler().fit(
        out.loc[out.split == "train", SHARED_FEATURES].to_numpy(dtype=np.float64))
    out[SHARED_FEATURES] = scaler.transform(
        out[SHARED_FEATURES].to_numpy(dtype=np.float64))

    # --- windowed samples, per city, per contiguous run --------------------------
    w, h, hour = cfg.window, cfg.horizon, np.timedelta64(1, "h")
    frames = []
    for city, grp in out.groupby("station", sort=True):
        grp = grp.sort_values("datetime", kind="mergesort").reset_index(drop=True)
        ts = grp.datetime.to_numpy(dtype="datetime64[ns]")
        breaks = np.flatnonzero(np.diff(ts) != hour) + 1
        edges = [0, *breaks.tolist(), len(ts)]
        for a, b in zip(edges[:-1], edges[1:]):
            n_s = (b - a) - w - h + 1
            if n_s <= 0:
                continue
            last = a + np.arange(n_s) + w - 1
            tgt = last + h
            blk = grp.loc[last, ["station", "datetime", *cfg.feature_columns,
                                 "is_imputed", "is_imputed_pm25", "split"]].reset_index(drop=True)
            blk["target_time"] = grp.loc[tgt, "datetime"].to_numpy()
            blk["y_pm25_raw"] = grp.loc[tgt, "PM2.5_raw"].to_numpy()
            blk["pm25_category"] = grp.loc[tgt, "pm25_category"].to_numpy()
            blk["y_category"] = grp.loc[tgt, "y_category_now"].to_numpy()
            # provenance follows the LABEL row, as in Phase 2
            blk["is_imputed_pm25"] = grp.loc[tgt, "is_imputed_pm25"].to_numpy()
            frames.append(blk)
    samples = pd.concat(frames, ignore_index=True)
    stats["samples"] = {s: int((samples.split == s).sum())
                        for s in ("train", "val", "test")}
    say("samples: " + ", ".join(f"{k}={v:,}" for k, v in stats["samples"].items()))

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    for s in ("train", "val", "test"):
        samples[samples.split == s].drop(columns=["split"]).to_csv(
            cfg.out_dir / f"tabular_{s}.csv", index=False)
    joblib.dump(scaler, cfg.out_dir / "scaler.pkl")
    meta = {"feature_columns": cfg.feature_columns, "scaled_columns": SHARED_FEATURES,
            "class_labels": cfg.labels, "window": cfg.window, "horizon": cfg.horizon,
            "cities": stats["cities"], "clean_start": str(CLEAN_START),
            "source": "Mendeley 9j447cynb9 v2", "stats": stats}
    (cfg.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    say(f"wrote {cfg.out_dir.relative_to(REPO_ROOT)}/")
    return {"stats": stats, "meta": meta}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args(argv)
    if args.audit_only:
        print(json.dumps(audit(), indent=2, default=str))
        return 0
    build(load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
