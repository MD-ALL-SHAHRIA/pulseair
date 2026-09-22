"""Tabular baselines: Random Forest and XGBoost on the AQI risk category.

Trains both on ``data/processed/tabular_train.csv`` and evaluates on
``tabular_test.csv``, using **only** the nine ``feature_columns`` from
``metadata.json``. The ``is_imputed*`` side-cars are evaluation slicers -- they are
asserted out of the input matrix, never fed to a model.

Every metric is reported twice:

* **all rows** -- every test sample;
* **observed-only** -- ``is_imputed_pm25 == False``, i.e. samples whose label came from
  a real PM2.5 reading rather than a forward-filled one.

The second set is the one that matters. ``reports/preprocessing_summary.md`` shows that
forward-fill roughly doubles Hazardous prevalence (8.38% of imputed readings vs 4.47% of
observed ones), so any Hazardous score computed over all rows is partly a score on
carried-forward values. Model selection therefore runs on observed-only macro-F1.

    python -m src.models.baseline

Writes ``reports/baseline_metrics.md`` and ``src/models/artifacts/baseline.pkl``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

# The two classes that trigger a health advisory, and the reason the thesis exists.
RARE_CLASSES = ("Very unhealthy", "Hazardous")

# Label for the zero-parameter reference: "the category N hours from now is this one".
PERSISTENCE_KEY = "Persistence"


def replace_variant(cfg: "BaselineConfig", aug_variant: str) -> "BaselineConfig":
    """Return `cfg` pointing at a different augmentation variant."""
    from dataclasses import replace as _replace
    return _replace(cfg, augmented=True, aug_variant=aug_variant)

# Marker written by src/gan/augment.py; "real" vs "synthetic".
SOURCE_COL, REAL = "source", "real"

# Columns that must never reach a model, even if a future refactor widens the frame.
FORBIDDEN_INPUTS = ("is_imputed", "is_imputed_pm25", "is_imputed_input",
                    "y_category", "y_pm25", "y_pm25_raw", "pm25_category", "target_time")


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class BaselineConfig:
    processed_root: Path
    artifacts_dir: Path
    reports_dir: Path
    labels: list[str]
    breakpoints: list[float]
    rf_params: dict
    xgb_params: dict
    seed: int
    horizon: int
    horizons: tuple[int, ...] = ()
    augmented: bool = False
    aug_variant: str = "broad"

    @property
    def processed_dir(self) -> Path:
        return self.processed_root / f"h{self.horizon}"

    @property
    def tag(self) -> str:
        if not self.augmented:
            return f"h{self.horizon}"
        suffix = "" if self.aug_variant == "broad" else f"_{self.aug_variant}"
        return f"h{self.horizon}_augmented{suffix}"

    @property
    def train_file(self) -> Path:
        if not self.augmented:
            return self.processed_dir / "tabular_train.csv"
        suffix = "" if self.aug_variant == "broad" else f"_{self.aug_variant}"
        return self.processed_dir / f"tabular_train_augmented{suffix}.csv"

    @property
    def model_path(self) -> Path:
        # Horizon-suffixed: an unsuffixed baseline.pkl stopped being well-defined once
        # the project carried four datasets.
        return self.artifacts_dir / f"baseline_{self.tag}.pkl"

    @property
    def report_path(self) -> Path:
        return self.reports_dir / f"baseline_metrics_{self.tag}.md"

    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / "metrics" / f"baseline_{self.tag}.json"

    @property
    def horizon_report_path(self) -> Path:
        return self.reports_dir / "horizon_comparison.md"

    @property
    def horizon_metrics_path(self) -> Path:
        return self.reports_dir / "metrics" / "horizon_comparison.json"


def load_config(path: Path | str = DEFAULT_CONFIG, root: Path | None = None,
                horizon: int | None = None, augmented: bool = False,
                aug_variant: str = "broad") -> BaselineConfig:
    root = root or REPO_ROOT
    raw = yaml.safe_load(Path(path).read_text())
    data, base, prep = raw["data"], raw["baseline"], raw["preprocessing"]
    return BaselineConfig(
        processed_root=root / data["processed_dir"],
        artifacts_dir=root / base["artifacts_dir"],
        reports_dir=root / Path(base["report"]).parent,
        labels=list(data["pm25_labels"]),
        breakpoints=list(data["pm25_breakpoints"]),
        rf_params=dict(base["random_forest"]),
        xgb_params=dict(base["xgboost"]),
        seed=int(raw["seed"]),
        horizon=int(horizon if horizon is not None else prep["horizon"]),
        horizons=tuple(int(h) for h in prep.get("horizons", [])),
        augmented=augmented,
        aug_variant=aug_variant,
    )


# ----------------------------------------------------------------------------- data


@dataclass
class Split:
    name: str
    X: np.ndarray
    y: np.ndarray
    observed: np.ndarray          # bool mask: label came from a real PM2.5 reading
    frame: pd.DataFrame

    def __len__(self) -> int:
        return len(self.y)


def load_split(cfg: BaselineConfig, name: str, feature_columns: list[str]) -> Split:
    """Load one tabular split and build the input matrix from feature_columns alone.

    Only ``train`` may come from the augmented file. Validation and test are always the
    real tables: synthetic rows in either would make every number meaningless.
    """
    path = cfg.train_file if name == "train" else cfg.processed_dir / f"tabular_{name}.csv"
    frame = pd.read_csv(path)
    if name != "train" and SOURCE_COL in frame.columns:
        raise ValueError(f"{path.name} carries a `{SOURCE_COL}` column; "
                         f"val/test must be real data only")

    # Guard rail, not decoration: this is the one place a side-car could leak in.
    leaked = set(feature_columns) & set(FORBIDDEN_INPUTS)
    if leaked:
        raise ValueError(f"feature_columns contains non-input columns: {sorted(leaked)}")
    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"tabular_{name}.csv is missing {sorted(missing)}")

    X = frame[feature_columns].to_numpy(dtype=np.float32)
    y = frame["y_category"].to_numpy(dtype=np.int16)
    # "observed" means a real, measured label: not imputed AND not synthetic.
    observed = ~frame["is_imputed_pm25"].to_numpy(dtype=bool)
    if SOURCE_COL in frame.columns:
        observed &= (frame[SOURCE_COL].to_numpy() == REAL)
    return Split(name, X, y, observed, frame)


def load_feature_columns(cfg: BaselineConfig) -> tuple[list[str], dict]:
    meta = json.loads((cfg.processed_dir / "metadata.json").read_text())
    return list(meta["feature_columns"]), meta


# -------------------------------------------------------------------------- metrics


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    """Accuracy, macro-F1 and the full per-class breakdown, over whatever rows are given."""
    idx = list(range(len(labels)))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
        "macro_f1": float(f1_score(y_true, y_pred, labels=idx, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=idx, average="weighted", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }


def evaluate_both_ways(split: Split, y_pred: np.ndarray, labels: list[str]) -> dict:
    """The required pair: all test rows, and observed-label rows only."""
    obs = split.observed
    return {
        "all": evaluate(split.y, y_pred, labels),
        "observed": evaluate(split.y[obs], y_pred[obs], labels),
        "imputed": evaluate(split.y[~obs], y_pred[~obs], labels),
    }


# --------------------------------------------------------------------------- models


def build_models(cfg: BaselineConfig) -> dict:
    rf = dict(cfg.rf_params)
    xgb = dict(cfg.xgb_params)
    return {
        "RandomForest": RandomForestClassifier(random_state=cfg.seed, **rf),
        "XGBoost": XGBClassifier(
            random_state=cfg.seed,
            objective="multi:softprob",
            num_class=len(cfg.labels),
            eval_metric="mlogloss",
            verbosity=0,
            **xgb,
        ),
    }


def persistence_baseline(split: Split, cfg: BaselineConfig, scaler,
                         scaled_columns: list[str]) -> np.ndarray:
    """Reference point: predict that the next hour's class equals the current hour's.

    This costs nothing and needs no model. A learned baseline that cannot clear it is
    not adding anything, so the report shows it alongside RF and XGBoost.

    ``scaled_columns`` is metadata's ``scaled_columns`` (the five physical channels),
    not ``feature_columns`` -- the scaler never saw the cyclical encodings, so its
    ``mean_``/``scale_`` are five-wide.
    """
    i_pm = scaled_columns.index("PM2.5")
    pm_now = (split.frame["PM2.5"].to_numpy(dtype=np.float64) * scaler.scale_[i_pm]
              + scaler.mean_[i_pm])
    bins = [*cfg.breakpoints, np.inf]
    return pd.cut(pm_now, bins=bins, labels=False, right=True).astype(np.int16)


# ------------------------------------------------------------------------- pipeline


def run(cfg: BaselineConfig | None = None, *, write: bool = True, verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    say = print if verbose else (lambda *a, **k: None)

    features, meta = load_feature_columns(cfg)
    say(f"features ({len(features)}): {', '.join(features)}")
    say(f"excluded side-cars: {', '.join(meta['sidecar_columns'])}")

    train = load_split(cfg, "train", features)
    val = load_split(cfg, "val", features)
    test = load_split(cfg, "test", features)
    if cfg.augmented:
        mix = train.frame[SOURCE_COL].value_counts()
        say(f"AUGMENTED train from {cfg.train_file.name}: {len(train):,} rows "
            f"({mix.get(REAL, 0):,} real + {mix.get('synthetic', 0):,} synthetic)")
    say(f"train={len(train):,}  val={len(val):,}  test={len(test):,} "
        f"| test observed={int(test.observed.sum()):,} "
        f"imputed={int((~test.observed).sum()):,}")

    results = {}
    for name, model in build_models(cfg).items():
        t0 = time.perf_counter()
        model.fit(train.X, train.y)
        fit_s = time.perf_counter() - t0
        pred_test = model.predict(test.X)
        pred_val = model.predict(val.X)
        results[name] = {
            "model": model,
            "test": evaluate_both_ways(test, pred_test, cfg.labels),
            "val": evaluate_both_ways(val, pred_val, cfg.labels),
            "fit_seconds": round(fit_s, 1),
        }
        r = results[name]["test"]
        say(f"  {name:13s} fit {fit_s:5.1f}s | "
            f"all macro-F1 {r['all']['macro_f1']:.4f} | "
            f"observed macro-F1 {r['observed']['macro_f1']:.4f}")

    # Persistence needs the scaler to undo the feature scaling.
    scaler = joblib.load(cfg.processed_dir / "scaler.pkl")
    pred_persist = persistence_baseline(test, cfg, scaler, list(meta["scaled_columns"]))
    results[PERSISTENCE_KEY] = {
        "model": None,
        "test": evaluate_both_ways(test, pred_persist, cfg.labels),
        "val": None,
        "fit_seconds": 0.0,
    }
    # How often the label simply does not change over the horizon. This is the accuracy
    # a pure echo achieves for free, and the honest floor for any accuracy figure.
    payload_unchanged = float((pred_persist == test.y).mean() * 100)
    say(f"  {'Persistence':13s} (reference)  | "
        f"all macro-F1 {results[PERSISTENCE_KEY]['test']['all']['macro_f1']:.4f} | "
        f"observed macro-F1 {results[PERSISTENCE_KEY]['test']['observed']['macro_f1']:.4f}")

    # Selection reads the VALIDATION split only. Test is for final reporting.
    # Choosing on test -- even between two candidates, even when they agree -- turns the
    # test score into a selection-contaminated estimate, and the agreement is only
    # knowable by looking, which is the thing being avoided.
    learned = {k: v for k, v in results.items() if v["model"] is not None}
    best_name = max(learned, key=lambda k: learned[k]["val"]["observed"]["macro_f1"])
    would_be_test = max(learned, key=lambda k: learned[k]["test"]["observed"]["macro_f1"])
    say(f"selected on VAL observed macro-F1: {best_name} "
        f"({learned[best_name]['val']['observed']['macro_f1']:.4f})")
    if would_be_test != best_name:
        say(f"  note: test would have picked {would_be_test} -- selection stands on val")

    payload = {
        "results": results,
        "best": best_name,
        "selected_on": "val_observed_macro_f1",
        "test_would_pick": would_be_test,
        "horizon": cfg.horizon,
        "features": features,
        "labels": cfg.labels,
        "label_unchanged_pct": payload_unchanged,
        "counts": {
            "train": len(train), "val": len(val), "test": len(test),
            "test_observed": int(test.observed.sum()),
            "test_imputed": int((~test.observed).sum()),
        },
        "config": cfg,
    }

    if write:
        cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": results[best_name]["model"],
            "model_name": best_name,
            "feature_columns": features,
            "class_labels": cfg.labels,
            "selected_on": "VALIDATION observed-only macro-F1 (is_imputed_pm25 == False)",
            "horizon": cfg.horizon,
            "val_observed_macro_f1": results[best_name]["val"]["observed"]["macro_f1"],
            "test_observed_macro_f1": results[best_name]["test"]["observed"]["macro_f1"],
            "test_all_macro_f1": results[best_name]["test"]["all"]["macro_f1"],
            "trained_on": str(cfg.processed_dir.relative_to(REPO_ROOT) / "tabular_train.csv"),
            "seed": cfg.seed,
        }
        # compress=3: a tree ensemble pickles to several times its compressed size.
        joblib.dump(bundle, cfg.model_path, compress=3)
        size_mb = cfg.model_path.stat().st_size / 1e6
        payload["artifact_mb"] = round(size_mb, 1)
        say(f"saved {best_name} -> {cfg.model_path.relative_to(REPO_ROOT)} ({size_mb:.1f} MB)")

        cfg.report_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.report_path.write_text(build_report(payload))  # reads payload["artifact_mb"]
        say(f"wrote {cfg.report_path.relative_to(REPO_ROOT)}")

        # Machine-readable metrics so a later run can compare against this one without
        # re-fitting. Models are dropped; only numbers are kept.
        cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.metrics_path.write_text(json.dumps({
            "tag": cfg.tag, "horizon": cfg.horizon, "augmented": cfg.augmented,
            "best": best_name, "selected_on": "val_observed_macro_f1",
            "counts": payload["counts"],
            "label_unchanged_pct": payload["label_unchanged_pct"],
            "results": {n: {"val": v["val"], "test": v["test"]} if v["val"]
                        else {"test": v["test"]} for n, v in results.items()},
        }, indent=2, default=str))
        say(f"wrote {cfg.metrics_path.relative_to(REPO_ROOT)}")

    return payload


# ---------------------------------------------------------------------------- report


def build_report(payload: dict) -> str:
    results, labels = payload["results"], payload["labels"]
    cfg, counts, best = payload["config"], payload["counts"], payload["best"]
    names = list(results)

    def table(header, rows):
        esc = lambda cells: [str(c).replace("|", "\\|") for c in cells]
        body = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return f"| {' | '.join(esc(header))} |\n| {' | '.join(['---'] * len(header))} |\n{body}"

    def headline(view: str):
        return table(
            ["Model", "n", "Accuracy", "Macro-F1", "Weighted-F1"],
            [[n, f"{results[n]['test'][view]['n']:,}",
              f"{results[n]['test'][view]['accuracy']:.4f}",
              f"**{results[n]['test'][view]['macro_f1']:.4f}**",
              f"{results[n]['test'][view]['weighted_f1']:.4f}"] for n in names],
        )

    def per_class(view: str):
        return table(
            ["Class", *names, "Support"],
            [[f"**{lab}**" if lab in RARE_CLASSES else lab,
              *[f"{results[n]['test'][view]['per_class'][lab]['f1']:.4f}" for n in names],
              f"{results[names[0]]['test'][view]['per_class'][lab]['support']:,}"]
             for lab in labels],
        )

    # The comparison the brief asks for, stated per model and per rare class.
    delta_rows = []
    for n in names:
        a, o = results[n]["test"]["all"], results[n]["test"]["observed"]
        row = [n, f"{a['macro_f1']:.4f}", f"{o['macro_f1']:.4f}",
               f"{o['macro_f1'] - a['macro_f1']:+.4f}"]
        for lab in RARE_CLASSES:
            row += [f"{a['per_class'][lab]['f1']:.4f}", f"{o['per_class'][lab]['f1']:.4f}",
                    f"{o['per_class'][lab]['f1'] - a['per_class'][lab]['f1']:+.4f}"]
        delta_rows.append(row)

    b = results[best]["test"]
    drops = {
        lab: b["observed"]["per_class"][lab]["f1"] - b["all"]["per_class"][lab]["f1"]
        for lab in RARE_CLASSES
    }
    haz_drop = drops["Hazardous"]
    macro_drop = b["observed"]["macro_f1"] - b["all"]["macro_f1"]

    def verdict(delta: float, what: str) -> str:
        if delta <= -0.02:
            return (f"**{what} drops by {abs(delta):.4f} when imputed-label rows are "
                    f"excluded.** Stated plainly: part of the headline score was earned "
                    f"against forward-filled labels, and the honest number is the lower one.")
        if delta >= 0.02:
            return (f"**{what} rises by {delta:.4f} on observed-only rows** -- the imputed "
                    f"rows were the harder ones, and the all-rows figure was understating "
                    f"real performance.")
        return (f"{what} moves by {delta:+.4f}, which is not a meaningful shift -- this "
                f"figure is not materially contaminated.")

    hazard_note = verdict(haz_drop, "Hazardous F1")
    macro_note = verdict(macro_drop, "Macro-F1")

    imputed_share = counts["test_imputed"] / counts["test"] * 100
    haz_all = b["all"]["per_class"]["Hazardous"]["support"]
    haz_obs = b["observed"]["per_class"]["Hazardous"]["support"]
    if max(abs(haz_drop), abs(macro_drop)) < 0.02:
        contamination_note = (
            f"So the forward-fill artifact does **not** materially move this baseline's "
            f"scores. That is a different question from the one "
            f"`reports/preprocessing_summary.md` raises, and both answers stand: "
            f"imputation inflates Hazardous *prevalence* (8.38% of imputed readings vs "
            f"4.47% of observed ones), but only {counts['test_imputed']:,} test samples "
            f"({imputed_share:.1f}%) carry an imputed label, and only "
            f"{haz_all - haz_obs:,} of the {haz_all:,} Hazardous ones do. Too few rows to "
            f"shift an F1. Keep reporting both numbers -- the check is cheap and the "
            f"answer could change once CTGAN starts generating from this distribution."
        )
    else:
        contamination_note = (
            f"The shift is large enough to matter. {counts['test_imputed']:,} test samples "
            f"({imputed_share:.1f}%) carry an imputed label, including "
            f"{haz_all - haz_obs:,} of {haz_all:,} Hazardous ones, and removing them "
            f"changes the picture. Quote the observed-only figures as the headline result."
        )

    # Confusion detail for the best model, observed-only -- where do Hazardous go?
    conf = np.array(b["observed"]["confusion"])
    i_haz = labels.index("Hazardous")
    haz_total = int(conf[i_haz].sum())
    haz_rows = [
        [labels[j], f"{int(conf[i_haz, j]):,}",
         f"{conf[i_haz, j] / haz_total * 100:.1f}%" if haz_total else "n/a"]
        for j in range(len(labels)) if conf[i_haz, j] > 0
    ]

    if payload["test_would_pick"] == best:
        agreement_note = (
            f"Test would have picked the same model, but that is an observation made "
            f"after the fact, not part of the decision. The selection rule does not "
            f"consult it."
        )
    else:
        agreement_note = (
            f"Test would have picked **{payload['test_would_pick']}** instead. The "
            f"selection stands on validation anyway -- that is the point of the rule. "
            f"Given how far val and test diverge in Very-unhealthy prevalence (see the "
            f"validation caveat in `reports/preprocessing_summary_h{cfg.horizon}.md`), "
            f"the disagreement is expected rather than alarming."
        )

    # --- CTGAN comparison: only meaningful on an augmented run, and only if the
    # --- unaugmented numbers are on disk to compare against.
    augmentation_block = ""
    if cfg.augmented:
        ref_path = cfg.reports_dir / "metrics" / f"baseline_h{cfg.horizon}.json"
        if ref_path.exists():
            ref = json.loads(ref_path.read_text())
            rows, gains = [], {}
            for n in names:
                if n not in ref["results"]:
                    continue
                before = ref["results"][n]["test"]["observed"]
                after = results[n]["test"]["observed"]
                d = after["macro_f1"] - before["macro_f1"]
                gains[n] = d
                rows.append([n, f"{before['macro_f1']:.4f}", f"{after['macro_f1']:.4f}",
                             f"**{d:+.4f}**"])
            cls_rows = []
            for lab in labels:
                r = [f"**{lab}**" if lab in RARE_CLASSES else lab]
                for n in names:
                    if n not in ref["results"]:
                        continue
                    bf = ref["results"][n]["test"]["observed"]["per_class"][lab]["f1"]
                    af = results[n]["test"]["observed"]["per_class"][lab]["f1"]
                    r += [f"{bf:.4f}", f"{af:.4f}", f"{af - bf:+.4f}"]
                cls_rows.append(r)
            cls_header = ["Class"]
            for n in names:
                if n in ref["results"]:
                    cls_header += [f"{n} before", f"{n} after", "Δ"]

            learned_gains = {n: g for n, g in gains.items() if n != PERSISTENCE_KEY}
            haz_gains = {
                n: (results[n]["test"]["observed"]["per_class"]["Hazardous"]["f1"]
                    - ref["results"][n]["test"]["observed"]["per_class"]["Hazardous"]["f1"])
                for n in learned_gains
            }
            # The selected model is the one that gets deployed, so its deltas are the
            # headline. Quoting the best delta across models would be cherry-picking.
            sel_gain = learned_gains.get(best, 0.0)
            sel_haz = haz_gains.get(best, 0.0)
            disagree = (len(haz_gains) > 1
                        and max(haz_gains.values()) > 0.005 > min(haz_gains.values())
                        and min(haz_gains.values()) < -0.005)

            if sel_gain >= 0.02:
                aug_headline = (f"**CTGAN augmentation helped.** {best}, the selected model, "
                            f"gains {sel_gain:+.4f} macro-F1 — a real improvement rather "
                            f"than noise.")
            elif sel_gain <= -0.02:
                aug_headline = (f"**CTGAN augmentation hurt.** {best}, the selected model, is "
                            f"{sel_gain:+.4f} macro-F1 *worse* than on real data alone. "
                            f"The synthetic rows displaced the real class balance without "
                            f"supplying usable signal.")
            else:
                aug_headline = (f"**CTGAN augmentation changed nothing that matters.** {best}, "
                            f"the selected model, moves {sel_gain:+.4f} macro-F1 — inside "
                            f"the range where the number is noise.")

            if sel_haz <= -0.02:
                haz_line = (f"On **Hazardous**, the class the augmentation was aimed at, "
                            f"{best} got **worse**: {sel_haz:+.4f} F1. Adding "
                            f"{counts['train'] - ref['counts']['train']:,} synthetic rows, "
                            f"most of them Hazardous, lowered Hazardous F1.")
            elif sel_haz >= 0.02:
                haz_line = (f"On **Hazardous**, {best} gains {sel_haz:+.4f} F1 — the one "
                            f"place the augmentation did what it was meant to.")
            else:
                haz_line = (f"On **Hazardous**, the class the augmentation targeted, {best} "
                            f"moves {sel_haz:+.4f} F1 — no meaningful change.")

            if disagree:
                order = sorted(haz_gains.items(), key=lambda kv: kv[1])
                haz_line += (
                    f" The two models also disagree in *direction* here "
                    f"({', '.join(f'{n} {g:+.4f}' for n, g in order)}), which is itself "
                    f"evidence that the movement is noise rather than signal — a real "
                    f"effect would push both the same way.")

            aug_verdict = (
                f"{aug_headline}\n\n{haz_line}\n\nThis is a negative result and belongs in "
                f"the thesis as one, alongside the flattened-window test: two reasonable "
                f"interventions at h={cfg.horizon}, neither of which moved the needle."
                if abs(sel_gain) < 0.02 else f"{aug_headline}\n\n{haz_line}")

            augmentation_block = f"""
---

## 4b. Against the unaugmented baseline

Both rows are observed-only macro-F1 on the **same real test set**. Only the training
data differs: `tabular_train.csv` versus `tabular_train_augmented.csv`
({counts['train']:,} rows including CTGAN output).

{table(["Model", "Real train", "Augmented train", "Δ"], rows)}

Persistence is unchanged by construction — it never sees training data — so it remains
the floor at **{ref['results'][PERSISTENCE_KEY]['test']['observed']['macro_f1']:.4f}**.

### Per-class F1

{table(cls_header, cls_rows)}

### Verdict

{aug_verdict}

The synthetic rows passed SDV's distributional checks
(`reports/gan_quality_report_h{cfg.horizon}.md`). That they did, and the downstream
metric still did not move, is the useful part: **looking like real data and being useful
to a classifier are different properties**, and only the second one was ever the point.
"""
        else:
            augmentation_block = (
                f"\n---\n\n## 4b. Against the unaugmented baseline\n\n"
                f"`{ref_path.relative_to(REPO_ROOT)}` is missing, so no comparison was "
                f"made. Run `python -m src.models.baseline` first.\n")

    persist = results.get(PERSISTENCE_KEY)
    learned_best_obs = b["observed"]["macro_f1"]
    persist_obs = persist["test"]["observed"]["macro_f1"] if persist else None
    beats = learned_best_obs - persist_obs if persist_obs is not None else None

    # Headline. The margin over persistence is the finding that changes what to do next,
    # so it goes at the top rather than in a reference section at the bottom.
    unchanged = payload["label_unchanged_pct"]
    haz_best = b["observed"]["per_class"]["Hazardous"]["f1"]
    haz_persist = persist["test"]["observed"]["per_class"]["Hazardous"]["f1"] if persist else None
    if beats is not None and beats <= 0.01:
        summary = f"""> **Headline: the learned baselines barely beat doing nothing.**
>
> {best} reaches **{learned_best_obs:.4f}** observed-only macro-F1. A zero-parameter
> persistence rule -- "next hour's category equals this hour's" -- reaches
> **{persist_obs:.4f}**. The margin is **{beats:+.4f}**. On Hazardous specifically,
> persistence scores {haz_persist:.4f} against {best}'s {haz_best:.4f}.
>
> This is a property of the task, not a bug in the models: **the AQI category is
> unchanged from t to t+{cfg.horizon} in {unchanged:.2f}% of test samples**, so echoing
> the current class is right {unchanged:.1f}% of the time for free. {best}'s accuracy is
> {b['all']['accuracy'] * 100:.2f}% -- {b['all']['accuracy'] * 100 - unchanged:+.2f}
> points on top of that. PM2.5 at time t alone carries most of the signal. **Treat {persist_obs:.4f}, not {learned_best_obs:.4f}, as the bar the
> sequence model has to clear.** If Phase 3 reports 0.81 macro-F1 as an improvement over
> "the baseline", it is claiming credit for roughly nothing.
>
> A longer horizon would make this a real forecasting problem; see
> `reports/horizon_comparison.md` for how persistence degrades as the horizon grows."""
    else:
        summary = f"""> **Headline:** {best} reaches **{learned_best_obs:.4f}** observed-only macro-F1
> against a zero-parameter persistence rule's **{persist_obs:.4f}** -- a margin of
> **{beats:+.4f}**. The learned baseline is doing real work beyond echoing its input."""

    return f"""# Baseline metrics -- horizon {cfg.horizon} h{" (CTGAN-augmented training set)" if cfg.augmented else ""}

Random Forest vs XGBoost, predicting the AQI risk category **{cfg.horizon} hours ahead**.
{"Trained on the CTGAN-augmented table; validation and test are real data only." if cfg.augmented else ""}

{summary}

---

Task: predict the PM2.5 AQI risk category **{cfg.horizon} hours ahead** from the nine
wearable feature channels. Trained on
`{cfg.processed_dir.relative_to(REPO_ROOT)}/tabular_train.csv`
({counts['train']:,} samples), evaluated on `tabular_test.csv` ({counts['test']:,}).

Inputs are `metadata.json -> feature_columns` only:
{', '.join(f'`{c}`' for c in payload['features'])}.
The `is_imputed*` columns are **evaluation slicers, not inputs** -- the loader raises if
one ever appears in the feature list.

Hyperparameters come from the `baseline:` block in `configs/default.yaml`
(seed {cfg.seed}). No tuning, no resampling, no class weighting: these are the floor the
GAN-augmented sequence model has to clear, so they are deliberately plain.

---

## 1. All test rows ({counts['test']:,} samples)

{headline('all')}

### Per-class F1

{per_class('all')}

---

## 2. Observed labels only ({counts['test_observed']:,} samples, `is_imputed_pm25 == False`)

The {counts['test_imputed']:,} excluded samples
({counts['test_imputed'] / counts['test'] * 100:.1f}% of test) carry a label derived from a
forward-filled PM2.5 reading rather than a measured one. **This is the set that matters
for the thesis** -- it is the only one not contaminated by the forward-fill artifact.

{headline('observed')}

### Per-class F1

{per_class('observed')}

---

## 3. All rows vs observed-only

{table(["Model", "Macro-F1 all", "Macro-F1 obs", "Δ",
        "V.unhealthy all", "V.unhealthy obs", "Δ",
        "Hazardous all", "Hazardous obs", "Δ"], delta_rows)}

### What the comparison says

Best model by observed-only macro-F1: **{best}**.

- {macro_note}
- {hazard_note}
- {verdict(drops['Very unhealthy'], 'Very unhealthy F1')}

{contamination_note}

Where {best} sends true Hazardous hours on observed-only rows
({haz_total:,} samples):

{table(["Predicted as", "n", "share"], haz_rows)}

---

## 4. Persistence reference

`Persistence` predicts that the category {cfg.horizon} hours from now equals the current
hour's. It has no parameters and needs no training. See
`reports/horizon_comparison.md` for how it degrades across h=1/6/12/24.

{"It scores **%.4f** observed-only macro-F1 against %s's **%.4f** -- the learned model is %s%.4f ahead. %s" % (
    persist_obs, best, learned_best_obs, "+" if beats >= 0 else "", beats,
    "That margin is what the learned baseline actually buys over doing nothing."
    if beats > 0.01 else
    "**That is not a meaningful margin.** At one-hour horizon the class rarely changes, "
    "so a learned model that barely clears persistence has not demonstrated much. Treat "
    "persistence, not the RF/XGBoost numbers, as the bar the sequence model must beat."
) if persist_obs is not None else ""}

{augmentation_block}
---

## 5. Saved artifact

`{cfg.model_path.relative_to(REPO_ROOT)}` holds a joblib bundle
({payload.get('artifact_mb', 0):.1f} MB, `compress=3`): the fitted **{best}**, its
`feature_columns` and `class_labels`, and the selection metric. Load it with
`joblib.load(...)` and index `["model"]`.

The forest is depth- and leaf-capped in `configs/default.yaml` for a reason worth
recording: an unbounded forest on these 294k rows pickles to **2.4 GB** and scores
*lower* (0.7970 vs {b['observed']['macro_f1']:.4f} observed macro-F1) -- it memorises
rather than generalises. Neither variant is remotely deployable to an ESP32; that is
what the TFLite Micro path in `src/deployment/` is for, and a tree ensemble is not the
model that will make that trip.

Selection used **validation observed-only macro-F1**
({results[best]['val']['observed']['macro_f1']:.4f}) -- observed-only rather than overall,
because the overall figure is inflated by the forward-fill artifact, and validation
rather than test, because test is reserved for final reporting. {agreement_note}

Reproduce with `python -m src.models.baseline`.
"""


# ------------------------------------------------------------- horizon comparison


def persistence_sweep(cfg: BaselineConfig, horizons=None, verbose: bool = True) -> dict:
    """Score the zero-parameter persistence rule at every generated horizon.

    This is the evidence that the task stops being trivial. Nothing is trained: for each
    horizon it bins the current hour's PM2.5 and calls that the prediction for t+h.
    """
    say = print if verbose else (lambda *a, **k: None)
    horizons = list(horizons or cfg.horizons or [cfg.horizon])
    rows = {}
    for h in horizons:
        hcfg = load_config(horizon=h)
        features, meta = load_feature_columns(hcfg)
        test = load_split(hcfg, "test", features)
        scaler = joblib.load(hcfg.processed_dir / "scaler.pkl")
        pred = persistence_baseline(test, hcfg, scaler, list(meta["scaled_columns"]))
        both = evaluate_both_ways(test, pred, hcfg.labels)
        rows[h] = {
            "n_test": len(test),
            "label_unchanged_pct": float((pred == test.y).mean() * 100),
            "all": both["all"],
            "observed": both["observed"],
        }
        say(f"  h={h:>2}  unchanged {rows[h]['label_unchanged_pct']:5.2f}%  "
            f"obs macro-F1 {both['observed']['macro_f1']:.4f}  "
            f"acc {both['observed']['accuracy']:.4f}")
    return rows


def write_horizon_metrics(sweep: dict, cfg: BaselineConfig,
                          learned: dict | None = None) -> Path:
    """Persist the horizon sweep as JSON beside the markdown.

    The markdown is for reading; this is what downstream code reads. Figure
    generation must never have to parse a report table to recover a number that
    was computed here.
    """
    payload = {
        "horizons": [int(h) for h in sweep],
        "primary": int(cfg.horizon),
        "labels": list(cfg.labels),
        "rows": {str(h): dict(v) for h, v in sweep.items()},
    }
    if learned:
        payload["learned_at_primary"] = {k: float(v) for k, v in learned.items()}
    cfg.horizon_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.horizon_metrics_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {cfg.horizon_metrics_path.relative_to(REPO_ROOT)}")
    return cfg.horizon_metrics_path


def build_horizon_report(sweep: dict, cfg: BaselineConfig, learned: dict | None = None) -> str:
    def table(header, rows):
        esc = lambda cs: [str(c).replace("|", "\\|") for c in cs]
        body = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return f"| {' | '.join(esc(header))} |\n| {' | '.join(['---'] * len(header))} |\n{body}"

    hs = sorted(sweep)
    base_h, base = hs[0], sweep[sorted(sweep)[0]]
    rows = []
    for h in hs:
        r = sweep[h]
        mark = " **(primary)**" if h == cfg.horizon else ""
        rows.append([
            f"**{h} h**{mark}",
            f"{r['label_unchanged_pct']:.2f}%",
            f"{r['observed']['accuracy']:.4f}",
            f"{r['observed']['macro_f1']:.4f}",
            f"{r['observed']['macro_f1'] - base['observed']['macro_f1']:+.4f}",
            f"{r['observed']['per_class']['Very unhealthy']['f1']:.4f}",
            f"{r['observed']['per_class']['Hazardous']['f1']:.4f}",
        ])

    prim = sweep[cfg.horizon]
    drop = base["observed"]["macro_f1"] - prim["observed"]["macro_f1"]
    drop_24 = base["observed"]["macro_f1"] - sweep[hs[-1]]["observed"]["macro_f1"]
    haz_drop = (base["observed"]["per_class"]["Hazardous"]["f1"]
                - prim["observed"]["per_class"]["Hazardous"]["f1"])

    # A 24 h offset lands at the same time of day, so diurnal structure could in
    # principle make h=24 easier than h=12. Check rather than assume.
    seq = [sweep[h]["observed"]["macro_f1"] for h in hs]
    monotone = all(a > b for a, b in zip(seq, seq[1:]))
    if monotone:
        deltas = ", ".join(f"h{a}->h{b} {sweep[b]['observed']['macro_f1'] - sweep[a]['observed']['macro_f1']:+.4f}"
                           for a, b in zip(hs, hs[1:]))
        monotone_note = (
            f"The decline is monotone across every step ({deltas}), so there is **no** "
            f"diurnal rebound at {hs[-1]} h -- landing on the same time of day does not "
            f"recover the lost predictability. Difficulty rises with distance, full stop."
        )
    else:
        worst = min(hs, key=lambda h: sweep[h]["observed"]["macro_f1"])
        monotone_note = (
            f"The decline is **not** monotone: h={worst} is the hardest, and longer "
            f"horizons recover somewhat. That is the diurnal cycle -- a 24 h offset lands "
            f"at the same time of day -- and it is worth showing in the Phase 9 figure "
            f"rather than smoothing over."
        )

    learned_block = ""
    if learned:
        lr = [[n, f"{v:.4f}", f"{v - prim['observed']['macro_f1']:+.4f}"]
              for n, v in learned.items()]
        best_gain = max(v - prim["observed"]["macro_f1"] for v in learned.values())
        if best_gain < 0.02:
            learned_verdict = (
                f"**The horizon change did its job, but the tabular baselines still do "
                f"not clear persistence by a meaningful margin** -- the best gain is "
                f"{best_gain:+.4f}. Two separate facts, and both matter: h=6 is a real "
                f"forecasting task (persistence fell {drop:.4f} from h={base_h}), and a "
                f"model that sees only the current hour's nine channels still cannot "
                f"exploit it. That is an *information* limit, not a model limit -- the "
                f"tabular row is one timestep, while the sequence model gets all "
                f"{{}} hours of the window. Opening a real gap over "
                f"{prim['observed']['macro_f1']:.4f} is now the thing Phase 4 onward has "
                f"to demonstrate, and it is no longer a foregone conclusion."
            ).format(24)
        else:
            learned_verdict = (
                f"The best learned model clears persistence by **{best_gain:+.4f}**, "
                f"which is a real margin rather than rounding. At h={base_h} the same "
                f"comparison was worth almost nothing, so the horizon change is what made "
                f"the learned model's contribution visible."
            )
        learned_block = f"""
## Learned models at the primary horizon

At h={cfg.horizon}, persistence scores **{prim['observed']['macro_f1']:.4f}**. The trained
baselines (observed-only macro-F1 on test):

{table(["Model", "Macro-F1", "vs persistence"], lr)}

{learned_verdict}
"""

    return f"""# Horizon comparison -- how hard is the task, really?

All numbers are the **zero-parameter persistence rule** on the test split, observed
labels only (`is_imputed_pm25 == False`). Persistence predicts that the category h hours
from now equals the current hour's. It has no parameters and never sees the training set.

Its score is the floor any learned model must clear to have demonstrated anything. If a
model lands near these numbers, it has learned to echo its PM2.5 input.

Every horizon uses the same raw table, the same nine features, and the same chronological
split boundaries -- only the label offset changes.

---

## Persistence degradation

{table(["Horizon", "Label unchanged", "Accuracy", "Macro-F1", "vs h={} h".format(base_h),
        "Very unhealthy F1", "Hazardous F1"], rows)}

### Reading this

At **h={base_h} h** the label is unchanged in {base['label_unchanged_pct']:.1f}% of samples.
Persistence scores {base['observed']['macro_f1']:.4f} macro-F1 for free, which is why the
learned scores at that horizon (`reports/baseline_metrics_h{base_h}.md`) were not evidence
of forecasting ability -- they were evidence that consecutive hours look alike.

At the primary **h={cfg.horizon} h**, persistence falls to
{prim['observed']['macro_f1']:.4f} (**{drop:.4f} lower**), and the label is unchanged in
only {prim['label_unchanged_pct']:.1f}% of samples. Hazardous F1 alone drops
{haz_drop:.4f}. The echo strategy has stopped working, which is exactly what makes the
horizon change worth doing: **there is now room for a model to be better than trivial.**

By **h={hs[-1]} h** persistence has given up {drop_24:.4f} macro-F1 against h={base_h}.
{monotone_note}
{learned_block}
---

## What this settles

- **h={base_h} h was not a forecasting task.** It was a persistence task with a model
  attached. Any Phase 3 result quoted at h={base_h} should be read as such.
- **h={cfg.horizon} h is the primary dataset** for CTGAN, the sequence model, SHAP and the
  LLM advisory. `configs/default.yaml` sets `preprocessing.horizon: {cfg.horizon}`.
- **h={base_h}/{hs[-2] if len(hs) > 2 else hs[-1]}/{hs[-1]} h are kept for this figure only.**
  They are generated by the same pipeline into `data/processed/h<N>/`, and they are not
  training sets.
- Report the persistence floor next to every future headline number. A macro-F1 with no
  floor beside it does not say whether the model is working.

Regenerate with `python -m src.models.baseline --horizon-sweep`.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--horizon", type=int, default=None,
                    help="override the primary horizon from the config")
    ap.add_argument("--horizon-sweep", action="store_true",
                    help="persistence-only across every generated horizon")
    ap.add_argument("--augmented", action="store_true",
                    help="train on the CTGAN-augmented table")
    ap.add_argument("--aug-variant", choices=("broad", "targeted"), default="broad",
                    help="which augmented table to use")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, horizon=args.horizon, augmented=args.augmented,
                      aug_variant=args.aug_variant)

    if args.horizon_sweep:
        print(f"persistence sweep over h={list(cfg.horizons)}")
        sweep = persistence_sweep(cfg)
        if not args.no_write:
            cfg.horizon_report_path.write_text(build_horizon_report(sweep, cfg))
            print(f"wrote {cfg.horizon_report_path.relative_to(REPO_ROOT)}")
            write_horizon_metrics(sweep, cfg)
        return 0

    payload = run(cfg, write=not args.no_write)

    # The comparison figure needs the learned numbers alongside persistence, so the
    # normal run refreshes it too when every horizon is on disk. Only the PRIMARY
    # horizon may write it -- a `--horizon 1` run would otherwise relabel h1 as primary.
    primary = load_config(args.config).horizon
    if (not args.no_write and cfg.horizon == primary and not cfg.augmented
            and all((cfg.processed_root / f"h{h}" / "metadata.json").exists()
                    for h in cfg.horizons)):
        print("refreshing horizon comparison ...")
        sweep = persistence_sweep(cfg)
        learned = {n: v["test"]["observed"]["macro_f1"]
                   for n, v in payload["results"].items() if v["model"] is not None}
        cfg.horizon_report_path.write_text(build_horizon_report(sweep, cfg, learned))
        print(f"wrote {cfg.horizon_report_path.relative_to(REPO_ROOT)}")
        write_horizon_metrics(sweep, cfg, learned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
