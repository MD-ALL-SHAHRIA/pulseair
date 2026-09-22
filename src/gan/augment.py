"""CTGAN augmentation of the rare AQI risk classes.

Fits one SDV ``CTGANSynthesizer`` per minority class on the primary horizon's training
split, samples enough rows to lift each minority class to a target fraction of the
majority class, and writes an augmented training table with a ``source`` column so
later phases can strip the synthetic rows back out.

Three constraints shape this, all of them inherited from earlier phases:

* **Observed labels only.** CTGAN is fit on ``is_imputed_pm25 == False`` rows. Forward
  fill roughly doubles Hazardous prevalence (8.38% of imputed readings vs 4.47% of
  observed ones, see ``reports/preprocessing_summary_h6.md``), so fitting on unfiltered
  rows would teach the generator an imputation artifact and then amplify it.
* **Nine feature columns, nothing else.** The side-cars and every label-derived column
  are excluded from the modelled set. One synthesizer per class carries the class
  identity, so the label never has to be a modelled column.
* **Partial rebalancing.** Minority classes are lifted to ``gan.target_ratio`` of the
  majority count, not to parity. A wearable does not encounter hazardous air half the
  time, and a generator asked to pretend otherwise distorts the base rate the advisory
  layer depends on.

    python -m src.gan.augment

Writes ``data/processed/h<primary>/tabular_train_augmented.csv`` and
``reports/gan_quality_report_h<primary>.md``.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

SOURCE_COL = "source"
REAL, SYNTHETIC = "real", "synthetic"

# Degrees C. Separates float64 round-trip noise from a genuine DEWP > TEMP violation;
# see validity_checks for why a bare `>` is the wrong test on standardized storage.
DEWP_TOLERANCE = 1e-6


@dataclass(frozen=True)
class GanConfig:
    processed_root: Path
    reports_dir: Path
    horizon: int
    labels: list[str]
    epochs: int
    batch_size: int
    minority_threshold: float
    target_ratio: float
    quality_floor: float
    seed: int
    variant: str = "broad"          # "broad" = every minority class; "targeted" = advisory only

    @property
    def processed_dir(self) -> Path:
        return self.processed_root / f"h{self.horizon}"

    @property
    def train_path(self) -> Path:
        return self.processed_dir / "tabular_train.csv"

    @property
    def suffix(self) -> str:
        return "" if self.variant == "broad" else f"_{self.variant}"

    @property
    def augmented_path(self) -> Path:
        return self.processed_dir / f"tabular_train_augmented{self.suffix}.csv"

    @property
    def report_path(self) -> Path:
        return self.reports_dir / f"gan_quality_report_h{self.horizon}{self.suffix}.md"

    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / "metrics" / f"gan_h{self.horizon}{self.suffix}.json"


# The two classes that actually trigger a health advisory. The "targeted" variant
# augments only these; "broad" augments everything the 50%-of-majority rule catches,
# which on this dataset also sweeps in Good and Unhealthy (sensitive).
ADVISORY_CLASSES = ("Very unhealthy", "Hazardous")


def load_config(path: Path | str = DEFAULT_CONFIG, root: Path | None = None,
                horizon: int | None = None, variant: str = "broad") -> GanConfig:
    root = root or REPO_ROOT
    raw = yaml.safe_load(Path(path).read_text())
    data, gan, prep = raw["data"], raw["gan"], raw["preprocessing"]
    if gan["model"] != "ctgan":
        raise ValueError(f"gan.model is {gan['model']!r}; this module implements ctgan")
    return GanConfig(
        processed_root=root / data["processed_dir"],
        reports_dir=root / "reports",
        horizon=int(horizon if horizon is not None else prep["horizon"]),
        labels=list(data["pm25_labels"]),
        epochs=int(gan["epochs"]),
        batch_size=int(gan["batch_size"]),
        minority_threshold=float(gan["minority_threshold"]),
        target_ratio=float(gan["target_ratio"]),
        quality_floor=float(gan["quality_floor"]),
        seed=int(raw["seed"]),
        variant=variant,
    )


# ------------------------------------------------------------------ class balance


def class_balance(cfg: GanConfig) -> dict:
    """The h6 observed-only training distribution, and what each class needs.

    Deliberately recomputed here rather than reused from the h1 or all-rows figures:
    both the horizon and the observed-only filter change the counts.
    """
    meta = json.loads((cfg.processed_dir / "metadata.json").read_text())
    train = pd.read_csv(cfg.train_path)
    observed = train[~train["is_imputed_pm25"]].reset_index(drop=True)

    counts_all = train["pm25_category"].value_counts().reindex(cfg.labels).fillna(0).astype(int)
    counts_obs = observed["pm25_category"].value_counts().reindex(cfg.labels).fillna(0).astype(int)
    majority_label = counts_obs.idxmax()
    majority_n = int(counts_obs.max())
    target_n = int(round(cfg.target_ratio * majority_n))

    plan = {}
    for label in cfg.labels:
        n = int(counts_obs[label])
        ratio = n / majority_n if majority_n else 0.0
        is_minority = ratio < cfg.minority_threshold
        if cfg.variant == "targeted" and label not in ADVISORY_CLASSES:
            is_minority = False
        plan[label] = {
            "n_all": int(counts_all[label]),
            "n_observed": n,
            "pct_observed": round(n / max(int(counts_obs.sum()), 1) * 100, 2),
            "ratio_to_majority": round(ratio, 4),
            "is_minority": bool(is_minority),
            "n_synthetic": max(target_n - n, 0) if is_minority else 0,
        }

    return {
        "variant": cfg.variant,
        "features": list(meta["feature_columns"]),
        "scaled_columns": list(meta["scaled_columns"]),
        "modelled_columns": [*meta["scaled_columns"], "hour", "month"],
        "sidecars": list(meta["sidecar_columns"]),
        "n_train_all": len(train),
        "n_train_observed": len(observed),
        "majority_label": majority_label,
        "majority_n": majority_n,
        "target_n": target_n,
        "plan": plan,
        "minority_classes": [l for l in cfg.labels if plan[l]["is_minority"]],
        "total_synthetic": sum(p["n_synthetic"] for p in plan.values()),
    }


# ------------------------------------------------------------------------ CTGAN


def modelled_frame(part: pd.DataFrame, scaler, scaled_cols: list[str]) -> pd.DataFrame:
    """The columns CTGAN actually models: raw sensor channels + integer hour/month.

    Two deliberate departures from feeding the model's own feature matrix straight in:

    **Raw units, not standardized.** ``DEWP <= TEMP`` is a fact about degrees Celsius.
    After StandardScaler it is not a column comparison at all -- the two channels have
    different means and scales -- so an Inequality constraint on the scaled values would
    enforce the wrong thing. CTGAN models degrees and ug/m3; the scaler is reapplied
    afterwards.

    **Integer hour and month, not their sin/cos.** The cyclical pair is a deterministic
    function of an integer with 24 (or 12) possible values. Handed to CTGAN as two free
    continuous columns it produced points off the unit circle -- times of day that do
    not exist. Here the integer is modelled as a categorical and the pair is computed
    after sampling.
    """
    out = pd.DataFrame(index=part.index)
    for col in scaled_cols:
        i = scaled_cols.index(col)
        out[col] = part[col] * scaler.scale_[i] + scaler.mean_[i]
    stamps = pd.to_datetime(part["datetime"])
    out["hour"] = stamps.dt.hour.astype(int)
    out["month"] = stamps.dt.month.astype(int)
    out = out.reset_index(drop=True)

    # A small fraction of real rows record DEWP marginally above TEMP -- instrument
    # tolerance, since the two are measured to 0.1 degC by separate sensors. Physically
    # that state is saturation, so those rows are clipped to DEWP == TEMP. Without this
    # the Inequality constraint cannot be fit at all: SDV refuses training data that
    # already violates it, which is the correct behaviour.
    violating = out["DEWP"] > out["TEMP"]
    n_clipped = int(violating.sum())
    out.loc[violating, "DEWP"] = out.loc[violating, "TEMP"]
    out.attrs["n_dewp_clipped"] = n_clipped
    return out


def _metadata_for(frame: pd.DataFrame, scaled_cols: list[str]):
    """Explicit metadata -- detection would type hour/month as numerical."""
    from sdv.metadata import Metadata

    columns = {c: {"sdtype": "numerical"} for c in scaled_cols}
    columns["hour"] = {"sdtype": "categorical"}
    columns["month"] = {"sdtype": "categorical"}
    return Metadata.load_from_dict({"tables": {"wearable": {"columns": columns}}})


def to_feature_space(sampled: pd.DataFrame, scaler, scaled_cols: list[str],
                     pcfg) -> pd.DataFrame:
    """Turn CTGAN's raw output into the nine scaled feature columns.

    The cyclical encoding is computed by the *pipeline's own* ``add_cyclical``, not a
    local copy, so the synthetic rows are guaranteed to use the identical formula the
    real rows were built with.
    """
    from src.preprocessing import pipeline as pl

    frame = sampled.copy()
    frame["hour"] = frame["hour"].astype(int)
    frame["month"] = frame["month"].astype(int)
    frame = pl.add_cyclical(frame, pcfg)

    scaled = frame[scaled_cols].to_numpy(dtype=np.float64)
    frame[scaled_cols] = scaler.transform(scaled)
    return frame[pcfg.feature_columns]


def _fit_one(real_modelled: pd.DataFrame, cfg: GanConfig, n_samples: int,
             scaled_cols: list[str], verbose: bool = True):
    """Fit a constrained CTGAN on one class's modelled frame and sample from it."""
    from sdv.single_table import CTGANSynthesizer
    from sdv.evaluation.single_table import evaluate_quality
    from sdv.cag import Inequality

    metadata = _metadata_for(real_modelled, scaled_cols)
    synth = CTGANSynthesizer(
        metadata, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=False, cuda=False,
    )
    # Enforced during training and sampling, not checked afterwards. Note this must be
    # the sdv.cag.Inequality object: passing the equivalent dict is accepted silently
    # and has no effect (verified -- 51% violations with the dict, 0% with the object).
    synth.add_constraints(constraints=[
        Inequality(low_column_name="DEWP", high_column_name="TEMP",
                   strict_boundaries=False),
    ])

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        synth.fit(real_modelled)
    fit_s = time.perf_counter() - t0

    sampled = synth.sample(num_rows=n_samples)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quality = evaluate_quality(real_modelled, sampled, metadata, verbose=False)

    if verbose:
        print(f"    fit {fit_s / 60:.1f} min | sampled {len(sampled):,} | "
              f"overall quality {quality.get_score():.4f}")
    return sampled, quality, fit_s


def _quality_dict(quality, cfg: GanConfig) -> dict:
    """Flatten an SDV QualityReport into something JSON- and Markdown-friendly."""
    props = quality.get_properties().set_index("Property")["Score"].to_dict()
    shapes = quality.get_details("Column Shapes")
    try:
        pairs = quality.get_details("Column Pair Trends")
    except Exception:
        pairs = pd.DataFrame()

    col_scores = {
        str(r["Column"]): (None if pd.isna(r["Score"]) else float(r["Score"]))
        for _, r in shapes.iterrows()
    }
    poor = {c: s for c, s in col_scores.items()
            if s is not None and s < cfg.quality_floor}

    pair_rows, poor_pairs = [], []
    if len(pairs):
        for _, r in pairs.iterrows():
            score = None if pd.isna(r["Score"]) else float(r["Score"])
            entry = {"a": str(r["Column 1"]), "b": str(r["Column 2"]), "score": score}
            pair_rows.append(entry)
            if score is not None and score < cfg.quality_floor:
                poor_pairs.append(entry)

    return {
        "overall": float(quality.get_score()),
        "column_shapes": (None if pd.isna(props.get("Column Shapes", np.nan))
                          else float(props["Column Shapes"])),
        "column_pair_trends": (None if pd.isna(props.get("Column Pair Trends", np.nan))
                               else float(props["Column Pair Trends"])),
        "per_column": col_scores,
        "poor_columns": poor,
        "pairs": pair_rows,
        "poor_pairs": poor_pairs,
    }


# ------------------------------------------------------------- validity checks


def validity_checks(augmented: pd.DataFrame, cfg: GanConfig) -> dict:
    """Constraint checks SDV's quality score does not perform.

    SDV scores marginal shapes and pairwise correlation. Neither notices that a row is
    *impossible* -- that its cyclical encoding lies off the unit circle, or that its dew
    point exceeds its air temperature. Those are the failures that matter for data a
    downstream model will treat as real sensor readings, so they are checked here.
    """
    import joblib

    meta = json.loads((cfg.processed_dir / "metadata.json").read_text())
    scaled_cols = list(meta["scaled_columns"])
    scaler = joblib.load(cfg.processed_dir / "scaler.pkl")

    def unscale(frame: pd.DataFrame, col: str) -> pd.Series:
        i = scaled_cols.index(col)
        return frame[col] * scaler.scale_[i] + scaler.mean_[i]

    out = {}
    for name, part in (("real", augmented[augmented[SOURCE_COL] == REAL]),
                       ("synthetic", augmented[augmented[SOURCE_COL] == SYNTHETIC])):
        if part.empty:
            continue
        checks = {"n": int(len(part))}

        # 1. Cyclical encodings must satisfy sin^2 + cos^2 == 1, and a real hour takes
        #    only 24 distinct values (12 for month).
        for base, period in (("hour", 24), ("month", 12)):
            sin_c, cos_c = f"{base}_sin", f"{base}_cos"
            if sin_c not in part.columns:
                continue
            radius = part[sin_c] ** 2 + part[cos_c] ** 2
            checks[base] = {
                "on_unit_circle_pct": round(float((radius - 1).abs().lt(0.01).mean() * 100), 2),
                "radius_min": round(float(radius.min()), 4),
                "radius_max": round(float(radius.max()), 4),
                "distinct_sin": int(part[sin_c].nunique()),
                "expected_distinct": period,
            }

        # 2. Dew point cannot exceed air temperature. This is thermodynamics, not a
        #    statistical tendency, so any material rate is a defect.
        #
        #    The tolerance is not slack: the stored table holds *standardized* values,
        #    so reconstructing degrees costs a multiply and an add, and a row that was
        #    generated at exactly DEWP == TEMP (saturation, which the constraint allows)
        #    can come back a few float64 ulps the wrong side of it. Measured excess on
        #    such rows is ~1e-15 degC. DEWP_TOLERANCE is nine orders of magnitude above
        #    that and nine orders below the 0.1 degC the instruments resolve, so it
        #    separates arithmetic noise from a real violation without hiding one.
        if {"TEMP", "DEWP"} <= set(part.columns):
            excess = unscale(part, "DEWP") - unscale(part, "TEMP")
            material = excess > DEWP_TOLERANCE
            checks["dewp_above_temp_pct"] = round(float(material.mean() * 100), 4)
            checks["dewp_above_temp_n"] = int(material.sum())
            checks["dewp_excess_max_degc"] = (
                float(f"{excess.max():.3g}") if len(excess) else 0.0)
            checks["dewp_within_float_noise_n"] = int(
                ((excess > 0) & ~material).sum())

        # 3. Concentrations cannot be negative once unscaled.
        neg = {}
        for col in ("PM2.5", "PM10", "CO"):
            if col in part.columns:
                neg[col] = round(float((unscale(part, col) < 0).mean() * 100), 2)
        checks["negative_concentration_pct"] = neg
        out[name] = checks
    return out


# --------------------------------------------------------------------- pipeline


@dataclass
class AugmentResult:
    config: GanConfig
    balance: dict
    quality: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    augmented: pd.DataFrame | None = None
    final_counts: dict = field(default_factory=dict)
    validity: dict = field(default_factory=dict)


def run(cfg: GanConfig | None = None, *, write: bool = True, verbose: bool = True,
        epochs: int | None = None) -> AugmentResult:
    """Fit one CTGAN per minority class, sample, and assemble the augmented table."""
    cfg = cfg or load_config()
    if epochs is not None:
        cfg = replace(cfg, epochs=epochs)
    say = print if verbose else (lambda *a, **k: None)

    bal = class_balance(cfg)
    features, sidecars = bal["features"], bal["sidecars"]

    # The modelled set is the nine feature columns. Assert it, do not assume it.
    forbidden = set(sidecars) | {"y_category", "y_pm25", "y_pm25_raw", "pm25_category",
                                 "target_time", "datetime", "station"}
    leaked = set(features) & forbidden
    if leaked:
        raise ValueError(f"modelled columns would include non-features: {sorted(leaked)}")

    say(f"h{cfg.horizon} train: {bal['n_train_all']:,} rows, {bal['n_train_observed']:,} "
        f"observed ({bal['n_train_observed'] / bal['n_train_all'] * 100:.1f}%)")
    say(f"majority: {bal['majority_label']} @ {bal['majority_n']:,} -> target "
        f"{bal['target_n']:,} ({cfg.target_ratio:.0%} of majority)")
    say(f"minority classes ({len(bal['minority_classes'])}): "
        f"{', '.join(bal['minority_classes'])}")
    say(f"synthetic rows to generate: {bal['total_synthetic']:,}")
    say(f"modelled columns ({len(features)}): {', '.join(features)}")

    import joblib
    from src.preprocessing import pipeline as pl

    scaled_cols = bal["scaled_columns"]
    scaler = joblib.load(cfg.processed_dir / "scaler.pkl")
    pcfg = pl.load_config(horizon=cfg.horizon)
    say(f"modelled by CTGAN ({len(bal['modelled_columns'])}): "
        f"{', '.join(bal['modelled_columns'])}  "
        f"[raw units; hour/month categorical; DEWP <= TEMP constrained]")

    train = pd.read_csv(cfg.train_path)
    observed = train[~train["is_imputed_pm25"]]

    quality, timings, synthetic_frames = {}, {}, []
    for i, label in enumerate(bal["minority_classes"], 1):
        need = bal["plan"][label]["n_synthetic"]
        rows = observed.loc[observed["pm25_category"] == label]
        real = modelled_frame(rows, scaler, scaled_cols)
        say(f"\n  [{i}/{len(bal['minority_classes'])}] {label}: CTGAN on {len(real):,} "
            f"real rows, {cfg.epochs} epochs -> {need:,} synthetic")

        sampled_raw, report, fit_s = _fit_one(real, cfg, need, scaled_cols, verbose=verbose)
        sampled = to_feature_space(sampled_raw, scaler, scaled_cols, pcfg)

        q = _quality_dict(report, cfg)
        q.update(n_real=len(real), n_synthetic=len(sampled),
                 fit_minutes=round(fit_s / 60, 2),
                 n_dewp_clipped=int(real.attrs.get("n_dewp_clipped", 0)))
        quality[label] = q
        timings[label] = round(fit_s / 60, 2)
        poor = sorted(q["poor_columns"])
        say(f"    shapes {q['column_shapes']!r} | pairs {q['column_pair_trends']!r} | "
            f"below floor: {poor or 'none'}")

        sampled = sampled.copy()
        sampled["pm25_category"] = label
        sampled["y_category"] = cfg.labels.index(label)
        # Synthetic rows are neither observed nor imputed. `source` is the real
        # discriminator; the side-cars are set False so an observed-only filter
        # inherited from an earlier phase does not quietly drop them.
        for c in sidecars:
            sampled[c] = False
        sampled[SOURCE_COL] = SYNTHETIC
        synthetic_frames.append(sampled)

    real_marked = train.copy()
    real_marked[SOURCE_COL] = REAL
    augmented = pd.concat([real_marked, *synthetic_frames], ignore_index=True)

    total = augmented["pm25_category"].value_counts().reindex(cfg.labels).fillna(0).astype(int)
    real_n = (augmented.loc[augmented[SOURCE_COL] == REAL, "pm25_category"]
              .value_counts().reindex(cfg.labels).fillna(0).astype(int))
    final_counts = {
        label: {"real": int(real_n[label]),
                "synthetic": int(total[label] - real_n[label]),
                "total": int(total[label]),
                "ratio_to_majority": round(float(total[label] / total.max()), 4)}
        for label in cfg.labels
    }

    validity = validity_checks(augmented, cfg)
    result = AugmentResult(cfg, bal, quality, timings, augmented, final_counts, validity)

    if write:
        augmented.to_csv(cfg.augmented_path, index=False)
        say(f"\nwrote {cfg.augmented_path.relative_to(REPO_ROOT)} ({len(augmented):,} rows "
            f"= {len(train):,} real + {len(augmented) - len(train):,} synthetic)")
        cfg.report_path.write_text(build_report(result))
        say(f"wrote {cfg.report_path.relative_to(REPO_ROOT)}")
        cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.metrics_path.write_text(json.dumps(
            {"balance": bal, "quality": quality, "final_counts": final_counts,
             "validity": validity, "epochs": cfg.epochs, "fit_minutes": timings},
            indent=2, default=str))
    return result


# ---------------------------------------------------------------------- report


def build_report(result: AugmentResult) -> str:
    cfg, bal, quality, final = result.config, result.balance, result.quality, result.final_counts

    def table(header, rows):
        esc = lambda cs: [str(c).replace("|", "\\|") for c in cs]
        body = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return f"| {' | '.join(esc(header))} |\n| {' | '.join(['---'] * len(header))} |\n{body}"

    def fmt(v, nd=4):
        return "n/a" if v is None else f"{v:.{nd}f}"

    # --- the filtered h6 distribution and the plan
    plan_rows = []
    for label in cfg.labels:
        p = bal["plan"][label]
        tag = " **(minority)**" if p["is_minority"] else ""
        plan_rows.append([
            f"{label}{tag}", f"{p['n_all']:,}", f"{p['n_observed']:,}",
            f"{p['pct_observed']:.2f}%", f"{p['ratio_to_majority']:.3f}",
            f"{p['n_synthetic']:,}" if p["n_synthetic"] else "—",
        ])

    q_rows = [[
        label, f"{quality[label]['n_real']:,}", f"{quality[label]['n_synthetic']:,}",
        fmt(quality[label]["overall"]), fmt(quality[label]["column_shapes"]),
        fmt(quality[label]["column_pair_trends"]),
        f"{quality[label]['fit_minutes']:.1f}",
    ] for label in bal["minority_classes"]]

    all_cols = bal.get("modelled_columns", bal["features"])
    col_rows = [[f"`{c}`", *[fmt(quality[l]["per_column"].get(c)) for l in bal["minority_classes"]]]
                for c in all_cols]

    final_rows = [[
        label, f"{final[label]['real']:,}", f"{final[label]['synthetic']:,}",
        f"{final[label]['total']:,}", f"{final[label]['ratio_to_majority']:.3f}",
    ] for label in cfg.labels]

    # --- flag poor columns explicitly rather than proceeding silently
    flagged = []
    for label in bal["minority_classes"]:
        for col, score in sorted(quality[label]["poor_columns"].items(),
                                 key=lambda kv: kv[1]):
            flagged.append([label, f"`{col}`", f"{score:.4f}",
                            f"{cfg.quality_floor - score:.4f} below floor"])
    pair_flagged = []
    for label in bal["minority_classes"]:
        for pr in sorted(quality[label]["poor_pairs"], key=lambda d: d["score"] or 1.0)[:5]:
            pair_flagged.append([label, f"`{pr['a']}` x `{pr['b']}`", fmt(pr["score"])])

    if flagged or pair_flagged:
        col_block = (
            f"**{len(flagged)} column(s) and {len(pair_flagged)} pair(s) scored below the "
            f"{cfg.quality_floor:.2f} floor set in `configs/default.yaml`.** They are "
            f"listed here rather than passed over:\n\n"
        )
        if flagged:
            col_block += table(["Class", "Column", "Score", "Shortfall"], flagged) + "\n\n"
        if pair_flagged:
            col_block += ("Worst correlation pairs:\n\n"
                          + table(["Class", "Pair", "Score"], pair_flagged) + "\n\n")
        col_block += (
            "A low **column shape** score means the synthetic marginal distribution for "
            "that channel does not match the real one — the generator has the wrong "
            "histogram. A low **pair trend** score means the marginals may be fine but "
            "the joint structure is not: the synthetic rows break a correlation the real "
            "sensor data has. For a wearable that is the more damaging failure, because "
            "PM2.5 and PM10 moving together is most of what distinguishes a real reading "
            "from noise."
        )
    else:
        col_block = (
            f"**No column or pair scored below the {cfg.quality_floor:.2f} floor.** "
            f"Every modelled channel's marginal distribution and every pairwise "
            f"correlation is reproduced at or above the threshold set in "
            f"`configs/default.yaml`."
        )

    # --- constraint checks: the part SDV's score is blind to
    val = result.validity
    vrows, verdicts = [], []
    if val:
        syn_v, real_v = val.get("synthetic", {}), val.get("real", {})
        for base in ("hour", "month"):
            if base in syn_v:
                sv, rv = syn_v[base], real_v.get(base, {})
                vrows.append([
                    f"`{base}_sin` / `{base}_cos` on the unit circle",
                    f"{rv.get('on_unit_circle_pct', float('nan')):.1f}%",
                    f"{sv['on_unit_circle_pct']:.1f}%",
                    "must be 100%",
                ])
                vrows.append([
                    f"distinct `{base}_sin` values",
                    f"{rv.get('distinct_sin', 0):,}", f"{sv['distinct_sin']:,}",
                    f"~{sv['expected_distinct']} possible",
                ])
        if "dewp_above_temp_pct" in syn_v:
            vrows.append([
                f"DEWP > TEMP by more than {DEWP_TOLERANCE:g} °C",
                f"{real_v.get('dewp_above_temp_pct', float('nan')):.4f}% "
                f"({real_v.get('dewp_above_temp_n', 0):,} rows)",
                f"{syn_v['dewp_above_temp_pct']:.4f}% "
                f"({syn_v.get('dewp_above_temp_n', 0):,} rows)",
                "0%",
            ])
            vrows.append([
                "largest DEWP − TEMP excess",
                f"{real_v.get('dewp_excess_max_degc', 0):.3g} °C",
                f"{syn_v.get('dewp_excess_max_degc', 0):.3g} °C",
                "≤ float noise",
            ])
        for col, pct in syn_v.get("negative_concentration_pct", {}).items():
            vrows.append([f"negative `{col}` after unscaling",
                          f"{real_v.get('negative_concentration_pct', {}).get(col, 0):.2f}%",
                          f"{pct:.2f}%", "must be 0%"])

        circ = syn_v.get("hour", {}).get("on_unit_circle_pct")
        dewp = syn_v.get("dewp_above_temp_pct")
        dewp_real = real_v.get("dewp_above_temp_pct", 0.0)
        if circ is not None and circ < 99:
            verdicts.append(
                f"**The cyclical encodings are broken.** Only {circ:.1f}% of synthetic "
                f"rows have `hour_sin² + hour_cos² ≈ 1`, against 100% of real rows, and "
                f"the generator produced {syn_v['hour']['distinct_sin']:,} distinct "
                f"`hour_sin` values where only {syn_v['hour']['expected_distinct']} hours "
                f"exist. CTGAN modelled sin and cos as two unrelated continuous columns, "
                f"so most synthetic rows encode a time of day that does not exist. The "
                f"fix is not more epochs: these columns are a deterministic function of "
                f"an integer hour and should not be generated at all. Sample the hour, "
                f"then compute the pair.")
        if dewp is not None and dewp > max(0.5, 3 * dewp_real):
            verdicts.append(
                f"**{dewp:.1f}% of synthetic rows have a dew point above their air "
                f"temperature**, against {dewp_real:.2f}% of real rows. That is "
                f"thermodynamically impossible — DEWP ≤ TEMP always. CTGAN reproduced "
                f"the TEMP and DEWP marginals well (both scored above the floor) and "
                f"still broke the constraint between them, because a correlation score "
                f"does not encode an inequality. SDV supports an `Inequality` constraint "
                f"for exactly this.")

    if vrows:
        validity_block = (
            table(["Check", "Real", "Synthetic", "Expected"], vrows) + "\n\n"
            + ("\n\n".join(f"- {v}" for v in verdicts) if verdicts else
               "All constraint checks are within the range set by the real data.")
        )
        if verdicts:
            validity_block += (
                "\n\n**Why the quality score missed this.** SDV's Column Shapes metric "
                "compares one marginal distribution at a time, and Column Pair Trends "
                "compares linear correlation. Neither asks whether a row is *possible*. "
                "A synthetic row can match every marginal and every correlation and "
                "still be a reading no sensor could produce. Scores in the 0.89–0.93 "
                "range are not evidence that the synthetic data is usable."
            )
    else:
        validity_block = "No constraint checks were run."

    worst = min(
        ((l, c, s) for l in bal["minority_classes"]
         for c, s in quality[l]["per_column"].items() if s is not None),
        key=lambda t: t[2], default=(None, None, None))

    total_min = sum(quality[l]["fit_minutes"] for l in bal["minority_classes"])
    clipped_total = sum(quality[l].get("n_dewp_clipped", 0) for l in bal["minority_classes"])
    clipped_base = sum(quality[l]["n_real"] for l in bal["minority_classes"]) or 1
    clipped_pct = clipped_total / clipped_base * 100

    return f"""# CTGAN quality report — horizon {cfg.horizon} h

One `CTGANSynthesizer` per minority class, {cfg.epochs} epochs, batch size
{cfg.batch_size}, seed {cfg.seed}. Total fit time {total_min:.0f} min.

Fit on `{cfg.train_path.relative_to(REPO_ROOT)}` restricted to
**`is_imputed_pm25 == False`** — {bal['n_train_observed']:,} of {bal['n_train_all']:,}
training rows ({bal['n_train_observed'] / bal['n_train_all'] * 100:.1f}%). Forward-fill
roughly doubles Hazardous prevalence, so fitting on unfiltered rows would have taught
the generator an imputation artifact and then amplified it.

Modelled columns ({len(all_cols)}): {', '.join(f'`{c}`' for c in all_cols)} — the five
sensor channels **in raw units** plus the **integer** hour and month as categoricals.
The cyclical `sin`/`cos` pairs are *not* generated; they are computed from the sampled
integer afterwards by the pipeline's own `add_cyclical`, so a synthetic row's encoding
is identical to a real one's. `DEWP ≤ TEMP` is enforced by an `sdv.cag.Inequality`
constraint during fitting and sampling. Raw units matter for that: after StandardScaler
the two channels have different means and scales, so the inequality is not a column
comparison at all. See section 5 for why the first design failed.

The class identity is carried by *which* synthesizer produced a row, so the label never
becomes a modelled column and cannot leak.

---

## 1. The h{cfg.horizon} observed-only training distribution

These counts are recomputed here, not carried over. Both the horizon change and the
observed-only filter move them, so the h1 and all-rows percentages reported in earlier
phases do not apply.

Majority class: **{bal['majority_label']}** at {bal['majority_n']:,} observed rows.
A class counts as minority below **{cfg.minority_threshold:.0%}** of that, and is topped
up to **{cfg.target_ratio:.0%}** of it ({bal['target_n']:,} rows).

{table(["Class", "All train rows", "Observed rows", "Observed %", "vs majority",
        "Synthetic needed"], plan_rows)}

{_minority_note(bal, cfg)}

---

## 2. Synthetic data quality

SDV's quality report compares each class's synthetic rows against the real rows that
class was fit on. 1.00 is a perfect match; the floor configured for this project is
**{cfg.quality_floor:.2f}**.

{table(["Class", "Real rows", "Synthetic", "Overall", "Column shapes", "Pair trends",
        "Fit (min)"], q_rows)}

### Per-column shape scores

{table(["Column", *bal["minority_classes"]], col_rows)}

### Flagged

{col_block}

---

## 2b. Constraint validity — what the quality score does not measure

{validity_block}

---

## 3. Resulting training distribution

{table(["Class", "Real", "Synthetic", "Total", "vs majority"], final_rows)}

Written to `{cfg.augmented_path.relative_to(REPO_ROOT)}`
({len(result.augmented):,} rows), with a **`source`** column holding `real` or
`synthetic`.

**Later phases must filter on `source`, not on `is_imputed_pm25`.** Synthetic rows carry
`is_imputed_pm25 == False` because they are not imputed, so an observed-only filter
inherited from Phase 2 will happily include them. Any reporting-only metric needs
`source == "real"` as well.

Note the real rows include the {bal['n_train_all'] - bal['n_train_observed']:,}
imputed-label rows: CTGAN did not learn from them, but dropping them from the training
table would discard real data for a reason that only applies to the generator.

---

## 4. Read this before using the output

- The rebalance is **partial by design** ({cfg.target_ratio:.0%} of majority, not parity).
  A neckband does not meet hazardous air half the time, and a classifier trained to
  expect that will over-warn in the field. The base rate matters to the advisory layer.
- Quality scores measure whether synthetic rows *look like* real ones. They do not
  measure whether adding them helps a downstream model, and as section 2b shows they do
  not measure whether a row is physically possible either. The downstream question is
  answered in `reports/baseline_metrics_h{cfg.horizon}_augmented.md`.
{_worst_note(worst, cfg)}

---

## 5. Methodological note — synthesizer quality metrics are not validity checks

**Keep this in the thesis as a caveat.** It is a finding about evaluating generative
models, not an incident report.

The first version of this step handed CTGAN the model's own nine feature columns
directly, including the four cyclical encodings, in standardized units. SDV's quality
report scored that output **0.89–0.93 overall**, with Column Shapes 0.88–0.91 and Column
Pair Trends 0.88–0.95. On those numbers the synthetic data looked usable.

It was not. Two defects, both invisible to the score:

1. **Invalid cyclical encodings.** `hour_sin` and `hour_cos` are a deterministic
   function of one integer with 24 possible values, and every real row satisfies
   `hour_sin² + hour_cos² = 1`. CTGAN modelled them as two unrelated continuous
   columns. Only **19.5%** of synthetic rows landed on the unit circle, and the
   generator emitted **82,885 distinct `hour_sin` values** where 24 hours exist. Four
   fifths of the synthetic rows encoded a time of day that cannot occur. Each column's
   *marginal* was reproduced acceptably, which is all Column Shapes measures.

2. **A violated physical constraint.** Dew point cannot exceed air temperature.
   **13.1%** of synthetic rows had `DEWP > TEMP`, against 0.18% in the real data. TEMP
   and DEWP each scored above the quality floor individually, and their linear
   correlation was reproduced well enough to pass Column Pair Trends — but a
   correlation coefficient does not encode an inequality.

The general lesson: **Column Shapes compares one marginal at a time, Column Pair Trends
compares linear association, and neither asks whether a row is possible.** A synthetic
record can match every marginal and every pairwise correlation in the training data and
still be a reading no instrument could produce. A high SDV score is evidence that the
synthetic data resembles the real data distributionally. It is not evidence that the
data is valid, and for engineered features or physically constrained channels the two
come apart completely.

What this implies for anyone using a tabular synthesizer:

- **Do not hand a synthesizer engineered features.** Model the underlying variable and
  recompute the derived ones. Here: model the integer hour, then compute sin/cos.
- **Encode hard constraints as constraints, not as hope.** SDV's `Inequality` enforces
  `DEWP ≤ TEMP` during fitting and sampling. Checking afterwards only tells you how
  much of the output to throw away.
- **Write domain validity checks and run them every time.** Section 2b of this report
  is four assertions. They caught what a 0.93 quality score missed, and they cost
  nothing to run.
- One SDV-specific trap worth recording: passing the constraint as a plain dict to
  `add_constraints` is accepted **silently and has no effect** — measured at 51%
  violations with the dict versus 0% with the `sdv.cag.Inequality` object. An API that
  accepts a no-op without complaint is exactly the kind of thing a validity check
  catches and a quality score does not.

A further wrinkle worth recording: {clipped_total:,} real training rows
({clipped_pct:.2f}%) have `DEWP` marginally above `TEMP` — instrument tolerance, since
the two are measured to 0.1 °C by separate sensors. SDV correctly refuses to fit an
`Inequality` on data that already violates it, so those rows are clipped to
`DEWP == TEMP` (saturation) before fitting. The refusal is the right behaviour: a
constraint the training data contradicts is a modelling error, not a detail.

Regenerate with `python -m src.gan.augment --variant {cfg.variant}`.
"""


def _minority_note(bal: dict, cfg: GanConfig) -> str:
    mins = bal["minority_classes"]
    odd = [l for l in mins if l in ("Good", "Moderate")]
    base = (f"{len(mins)} of {len(cfg.labels)} classes fall below the threshold: "
            f"{', '.join(mins)}.")
    if odd:
        base += (
            f"\n\n> **Worth a second look:** {', '.join(odd)} qualifies as a \"minority\" "
            f"only because *Unhealthy* is unusually dominant in this dataset — Beijing "
            f"2013–2017 spends most of its hours there. {', '.join(odd)} is not a rare "
            f"or safety-critical regime, and synthesising more of it serves the "
            f"balancing rule rather than the thesis. The rule in `configs/default.yaml` "
            f"was applied as specified; narrowing `gan.minority_threshold`, or "
            f"restricting augmentation to the two advisory classes, would target the "
            f"classes the wearable actually needs to get right."
        )
    return base


def _worst_note(worst, cfg: GanConfig) -> str:
    label, col, score = worst
    if label is None:
        return ""
    verdict = "comfortably above" if score >= cfg.quality_floor else "**below**"
    return (f"- The weakest single column across all classes is `{col}` for "
            f"*{label}* at {score:.4f}, which is {verdict} the {cfg.quality_floor:.2f} "
            f"floor.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--variant", choices=("broad", "targeted"), default="broad",
                    help="broad = every minority class; targeted = advisory classes only")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override gan.epochs (for smoke runs)")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from the saved CSV + metrics JSON, no refit")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the class balance and the synthesis plan, then stop")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, horizon=args.horizon, variant=args.variant)
    if args.plan_only:
        print(json.dumps(class_balance(cfg), indent=2, default=str))
        return 0

    if args.report_only:
        saved = json.loads(cfg.metrics_path.read_text())
        augmented = pd.read_csv(cfg.augmented_path, low_memory=False)
        cfg = replace(cfg, epochs=int(saved["epochs"]))
        res = AugmentResult(cfg, saved["balance"], saved["quality"],
                            saved.get("fit_minutes", {}), augmented,
                            saved["final_counts"], validity_checks(augmented, cfg))
        cfg.report_path.write_text(build_report(res))
        cfg.metrics_path.write_text(
            json.dumps({**saved, "validity": res.validity}, indent=2, default=str))
        print(f"rebuilt {cfg.report_path.relative_to(REPO_ROOT)}")
        return 0
    run(cfg, write=not args.no_write, epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
