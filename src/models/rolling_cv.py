"""Rolling-origin (walk-forward) cross-validation at the primary horizon.

The decisive experiment. Every earlier phase evaluated on one chronological split, and
that split disagreed with validation about whether any model beats persistence: the
unweighted forest wins on test and loses on validation, the class-weighted forest does
the reverse. A single split cannot settle that, because the answer depends on which
winter happens to land in which block.

Five expanding-window folds rotate the evaluation block across the four-year span, so
each fold is scored on a different season. The question "how many folds does this model
beat persistence in?" is then answerable, and it is the number that resolves the
disagreement.

Protocol, per fold:

* **Expanding window.** Fold k trains on everything before its cutoff and evaluates on
  the contiguous block after it.
* **Embargo.** Training uses samples whose *target* falls before the cutoff; evaluation
  uses samples whose 24-hour input window starts at or after it. Without the gap, an
  evaluation sample's input window would contain hours the model saw as training
  labels.
* **Scaler refit inside the fold.** The published splits are scaled with statistics
  fitted on the global training block, which would leak future information into every
  fold. Features are un-scaled back to physical units and re-standardised per fold.
* **Observed labels only** (`is_imputed_pm25 == False`) in evaluation, as everywhere
  else in this project.

    python -m src.models.rolling_cv

Writes ``reports/rolling_origin_cv_h6.md``.
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
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
RARE = ("Very unhealthy", "Hazardous")

N_FOLDS = 5
INITIAL_TRAIN_FRACTION = 0.40    # first 40% of the timeline is never evaluated on
EMBARGO_HOURS = 24               # = window length; see module docstring


@dataclass(frozen=True)
class CVConfig:
    processed_dir: Path
    reports_dir: Path
    horizon: int
    window: int
    labels: list[str]
    breakpoints: list[float]
    rf: dict
    xgb: dict
    seed: int
    dataset: str = "beijing"
    n_folds: int = N_FOLDS
    initial_fraction: float = INITIAL_TRAIN_FRACTION

    @property
    def tag(self) -> str:
        return "" if self.dataset == "beijing" else f"_{self.dataset}"

    @property
    def report_path(self) -> Path:
        if self.dataset == "beijing":
            return self.reports_dir / f"rolling_origin_cv_h{self.horizon}.md"
        return self.reports_dir / f"{self.dataset}_rolling_cv.md"

    @property
    def metrics_path(self) -> Path:
        return (self.reports_dir / "metrics"
                / f"rolling_cv_h{self.horizon}{self.tag}.json")


def load_config(path: Path | str = DEFAULT_CONFIG, dataset: str = "beijing",
                n_folds: int = N_FOLDS,
                initial_fraction: float = INITIAL_TRAIN_FRACTION) -> CVConfig:
    raw = yaml.safe_load(Path(path).read_text())
    data, prep = raw["data"], raw["preprocessing"]
    h = int(prep["horizon"])
    sub = f"h{h}" if dataset == "beijing" else f"bd_h{h}"
    return CVConfig(
        dataset=dataset, n_folds=n_folds, initial_fraction=initial_fraction,
        processed_dir=REPO_ROOT / data["processed_dir"] / sub,
        reports_dir=REPO_ROOT / "reports",
        horizon=h,
        window=int(prep["window"]),
        labels=list(data["pm25_labels"]),
        breakpoints=list(data["pm25_breakpoints"]),
        xgb=dict(raw["baseline"]["xgboost"]), rf=dict(raw["baseline"]["random_forest"]),
        seed=int(raw["seed"]),
    )


# ----------------------------------------------------------------------- data


def load_unscaled(cfg: CVConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    """The full h6 sample set in physical units, ordered by time.

    The published CSVs are standardised with statistics fitted on the global training
    block. Using them directly would leak future information into every fold, so the
    five physical channels are inverted back to raw units here and re-standardised
    inside each fold.
    """
    meta = json.loads((cfg.processed_dir / "metadata.json").read_text())
    scaler = joblib.load(cfg.processed_dir / "scaler.pkl")
    scaled_cols = list(meta["scaled_columns"])
    features = list(meta["feature_columns"])

    frames = [pd.read_csv(cfg.processed_dir / f"tabular_{s}.csv")
              for s in ("train", "val", "test")]
    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["target_time"] = pd.to_datetime(df["target_time"])

    # exact inverse of the Phase 2 transform
    raw = scaler.inverse_transform(df[scaled_cols].to_numpy(dtype=np.float64))
    for i, c in enumerate(scaled_cols):
        df[c] = raw[:, i]

    df = df.sort_values(["datetime", "station"], kind="mergesort").reset_index(drop=True)
    return df, features, scaled_cols


def make_folds(df: pd.DataFrame, n_folds: int = N_FOLDS,
               initial_fraction: float = INITIAL_TRAIN_FRACTION) -> list[dict]:
    """Expanding-window folds with cutoffs spread across the span.

    The first ``INITIAL_TRAIN_FRACTION`` of the timeline is training-only; the rest is
    divided into ``n_folds`` contiguous evaluation blocks, so each fold is scored on a
    different part of the year.
    """
    stamps = np.sort(df["datetime"].unique())
    n = len(stamps)
    start = int(n * initial_fraction)
    edges = np.linspace(start, n, n_folds + 1).astype(int)

    folds = []
    for k in range(n_folds):
        lo, hi = edges[k], edges[k + 1]
        cutoff = pd.Timestamp(stamps[lo])
        end = pd.Timestamp(stamps[hi - 1])
        folds.append({
            "fold": k + 1,
            "cutoff": cutoff,
            "eval_end": end,
            "eval_start_embargoed": cutoff + pd.Timedelta(hours=EMBARGO_HOURS),
        })
    return folds


def split_fold(df: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training = targets strictly before the cutoff. Evaluation = windows after it."""
    train = df[df["target_time"] < fold["cutoff"]]
    ev = df[(df["datetime"] >= fold["eval_start_embargoed"])
            & (df["datetime"] <= fold["eval_end"])]
    return train, ev


# -------------------------------------------------------------------- scoring


def persistence(cfg: CVConfig, frame: pd.DataFrame) -> np.ndarray:
    """Predict that the category h hours out equals the current hour's."""
    edges = [*cfg.breakpoints, np.inf]
    return pd.cut(frame["PM2.5"].to_numpy(dtype=np.float64), bins=edges,
                  labels=False, right=True).astype(np.int64)


def score(y: np.ndarray, pred: np.ndarray, labels: list[str]) -> dict:
    return {
        "macro_f1": float(f1_score(y, pred, labels=list(range(len(labels))),
                                   average="macro", zero_division=0)),
        "accuracy": float((y == pred).mean()),
        **{f"f1_{lab}": float(f1_score(y == i, pred == i, zero_division=0))
           for i, lab in enumerate(labels)},
    }


def build_rf(cfg: CVConfig, class_weight) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=cfg.rf["n_estimators"], max_depth=cfg.rf["max_depth"],
        min_samples_leaf=cfg.rf["min_samples_leaf"],
        max_features=cfg.rf["max_features"], class_weight=class_weight,
        random_state=cfg.seed, n_jobs=-1,
    )


def build_xgb(cfg: CVConfig):
    """XGBoost with the project's configured parameters, per Appendix A."""
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=cfg.seed, objective="multi:softprob",
                         num_class=len(cfg.labels), eval_metric="mlogloss",
                         verbosity=0, **cfg.xgb)


def load_sequences(cfg: CVConfig) -> dict:
    """The windowed arrays in RAW units, aligned to load_unscaled's row order.

    The committed .npz files are standardised with the global training scaler, which
    is exactly the leak the tabular path inverts away. The same inversion is applied
    here, and the arrays are then re-ordered to match the frame the folds index into,
    so a fold mask selects the same samples in both representations.
    """
    import joblib
    meta = json.loads((cfg.processed_dir / "metadata.json").read_text())
    scaler = joblib.load(cfg.processed_dir / "scaler.pkl")
    features = list(meta["feature_columns"])
    scaled_cols = list(meta["scaled_columns"])
    idx = [features.index(c) for c in scaled_cols]

    Xs, ts, sts = [], [], []
    for split in ("train", "val", "test"):
        z = np.load(cfg.processed_dir / f"sequences_{split}.npz", allow_pickle=True)
        Xs.append(z["X"].astype(np.float32))
        ts.append(pd.to_datetime(z["target_time"]))
        sts.append(z["station"])
    X = np.concatenate(Xs)
    target_time = np.concatenate([t.to_numpy() for t in ts])
    station = np.concatenate(sts)

    # invert the global standardisation on the physical channels only
    flat = X[:, :, idx].reshape(-1, len(idx)).astype(np.float64)
    X[:, :, idx] = scaler.inverse_transform(flat).reshape(
        X.shape[0], X.shape[1], len(idx)).astype(np.float32)

    order = np.lexsort((station, target_time))
    return {"X": X[order], "idx": idx, "n": len(order)}


def fit_predict_sequence(cfg: CVConfig, arch: str, tr_idx, ev_idx, ytr,
                         tr_observed, cache) -> np.ndarray:
    """Train one sequence model inside a fold and predict its evaluation block.

    Three things are kept consistent with the tabular path. Scaling is fitted on the
    fold's training windows alone. Early stopping uses a validation slice carved from
    the *end* of the training block, never from the evaluation block, so the stopping
    decision cannot see what it is scored on. And the stopping metric is observed-only
    macro-F1, as everywhere else in this project.
    """
    import torch
    from src.models import dl_forecast as dl

    seq = cache["seq"]
    X, idx = seq["X"], seq["idx"]
    tr_idx, ev_idx = np.asarray(tr_idx), np.asarray(ev_idx)

    # last 15% of the training block becomes the early-stopping set
    n_val = max(1, int(0.15 * len(tr_idx)))
    fit_idx, val_idx = tr_idx[:-n_val], tr_idx[-n_val:]
    ytr = np.asarray(ytr)
    y_fit, y_val = ytr[:-n_val], ytr[-n_val:]
    obs = np.asarray(tr_observed)
    obs_val = obs[-n_val:]

    Xf, Xv, Xe = (X[fit_idx].copy(), X[val_idx].copy(), X[ev_idx].copy())
    flat = Xf[:, :, idx].reshape(-1, len(idx))
    mu, sd = flat.mean(axis=0), flat.std(axis=0)
    sd[sd == 0] = 1.0
    for A in (Xf, Xv, Xe):
        A[:, :, idx] = (A[:, :, idx] - mu) / sd

    mk = lambda name, A, y, o: dl.SeqSplit(
        name, torch.from_numpy(A), torch.from_numpy(np.asarray(y, dtype=np.int64)), o)
    device = dl.pick_device()
    out = dl.train_model(arch, cache["dlcfg"],
                         mk("fold-train", Xf, y_fit, np.ones(len(Xf), bool)),
                         mk("fold-val", Xv, y_val, obs_val),
                         device, verbose=False)
    model = out["model"] if isinstance(out, dict) else out
    logits = dl._predict_logits(model, torch.from_numpy(Xe), device,
                                cache["dlcfg"].batch_size)
    return logits.argmax(dim=1).cpu().numpy()


# ------------------------------------------------------------------ the run


# name -> (kind, class_weight). "seq" entries are trained on the windowed arrays
# rather than the tabular rows, and are only run when --with-sequence-models is given,
# because each one costs minutes per fold rather than seconds.
MODELS = {
    "Persistence": ("persistence", None),
    "RandomForest (unweighted)": ("rf", None),
    "RandomForest (class_weight=balanced)": ("rf", "balanced"),
    "XGBoost": ("xgb", None),
}
SEQ_MODELS = {
    "LSTM": ("seq", "lstm"),
    "Transformer": ("seq", "transformer"),
}


def run(cfg: CVConfig | None = None, *, write: bool = True,
        verbose: bool = True, with_sequence_models: bool = False) -> dict:
    cfg = cfg or load_config()
    say = print if verbose else (lambda *a, **k: None)
    labels = cfg.labels

    df, features, scaled_cols = load_unscaled(cfg)
    say(f"{len(df):,} samples, {df['station'].nunique()} stations, "
        f"{df['datetime'].min().date()} .. {df['datetime'].max().date()}")

    # Fold boundaries come from pulsebench, the extracted protocol package. This is
    # the same arithmetic that used to live in make_folds() below; delegating it keeps
    # one implementation, and the regression test in
    # pulsebench/tests/test_retrofit_phase9.py pins the numbers.
    from pulsebench import make_folds as pb_make_folds
    pb_folds = pb_make_folds(df["datetime"].to_numpy(), cfg.n_folds,
                             cfg.initial_fraction)
    folds = [{"fold": f["fold"], "cutoff": f["cutoff"], "eval_end": f["eval_end"],
              "eval_start_embargoed": f["cutoff"] + pd.Timedelta(hours=EMBARGO_HOURS)}
             for f in pb_folds]
    active_models = dict(MODELS)
    seq_cache: dict = {}
    if with_sequence_models:
        from src.models import dl_forecast as dl
        active_models.update(SEQ_MODELS)
        say("loading windowed sequences for the sequence models ...")
        seq_cache["seq"] = load_sequences(cfg)
        seq_cache["dlcfg"] = dl.load_config(horizon=cfg.horizon)
        if seq_cache["seq"]["n"] != len(df):
            raise ValueError(
                f"sequence array has {seq_cache['seq']['n']:,} rows but the tabular "
                f"frame has {len(df):,}; a fold mask would select different samples "
                f"in the two representations")
    active = list(active_models)

    results = []
    for fold in folds:
        tr, ev = split_fold(df, fold)
        obs = ~ev["is_imputed_pm25"].to_numpy(dtype=bool)
        y = ev["y_category"].to_numpy()[obs]

        say(f"\nfold {fold['fold']}: cutoff {fold['cutoff'].date()} | "
            f"train {len(tr):,} -> eval {len(ev):,} "
            f"({int(obs.sum()):,} observed) | eval block "
            f"{ev['datetime'].min().date()} .. {ev['datetime'].max().date()}")

        # Scaler fitted inside the fold, on training rows only.
        scaler = StandardScaler().fit(tr[scaled_cols].to_numpy(dtype=np.float64))

        def prep(frame: pd.DataFrame) -> np.ndarray:
            out = frame[features].copy()
            out[scaled_cols] = scaler.transform(
                frame[scaled_cols].to_numpy(dtype=np.float64))
            return out.to_numpy(dtype=np.float32)

        Xtr, ytr = prep(tr), tr["y_category"].to_numpy()
        Xev = prep(ev)

        entry = {
            "fold": fold["fold"],
            "cutoff": str(fold["cutoff"]),
            "eval_start": str(ev["datetime"].min()),
            "eval_end": str(ev["datetime"].max()),
            "n_train": int(len(tr)), "n_eval": int(len(ev)),
            "n_eval_observed": int(obs.sum()),
            "eval_months": sorted({int(m) for m in ev["datetime"].dt.month.unique()}),
            # Per-class support in the evaluation block. Without it an F1 of 0.0 on a
            # class with no samples reads like a model failure rather than an absence
            # of data.
            "support": {lab: int((y == i).sum()) for i, lab in enumerate(labels)},
            "scores": {},
        }

        for name, (kind, arg) in active_models.items():
            t0 = time.perf_counter()
            if kind == "persistence":
                pred = persistence(cfg, ev)[obs]
            elif kind == "rf":
                pred = build_rf(cfg, arg).fit(Xtr, ytr).predict(Xev)[obs]
            elif kind == "xgb":
                pred = build_xgb(cfg).fit(Xtr, ytr).predict(Xev)[obs]
            elif kind == "seq":
                pred = fit_predict_sequence(cfg, arg, tr.index, ev.index, ytr,
                                            tr["is_imputed_pm25"].to_numpy() == False,
                                            seq_cache)[obs]
            else:
                raise ValueError(f"unknown model kind {kind!r}")
            sc = score(y, pred, labels)
            sc["fit_seconds"] = round(time.perf_counter() - t0, 1)
            entry["scores"][name] = sc
            say(f"  {name:38s} macro-F1 {sc['macro_f1']:.4f}  "
                f"Haz {sc['f1_Hazardous']:.4f}  VU {sc['f1_Very unhealthy']:.4f}  "
                f"({sc['fit_seconds']:.0f}s)")
        results.append(entry)

    agg = aggregate(results, labels)
    say("\n" + "=" * 72)
    for name in active:
        a = agg["per_model"][name]
        say(f"{name:38s} macro-F1 {a['mean']:.4f} +/- {a['std']:.4f}")
    for name, t in agg["tests"].items():
        if name.startswith("_"):      # _resolution is metadata, not a comparison
            continue
        say(f"\n{name} vs Persistence:")
        say(f"  mean delta {t['mean_delta']:+.4f} | wins {t['wins']}/{t['n_folds']} folds")
        say(f"  Wilcoxon two-sided p={t['p_two_sided']:.4f} "
            f"one-sided p={t['p_one_sided']:.4f}")

    payload = {"config": cfg, "folds": results, "aggregate": agg,
               "n_folds": len(results), "embargo_hours": EMBARGO_HOURS,
               "initial_train_fraction": cfg.initial_fraction,
               "dataset": cfg.dataset}

    if write:
        cfg.report_path.write_text(build_report(payload))
        say(f"\nwrote {cfg.report_path.relative_to(REPO_ROOT)}")
        cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.metrics_path.write_text(json.dumps(
            {k: v for k, v in payload.items() if k != "config"},
            indent=2, default=str))
        say(f"wrote {cfg.metrics_path.relative_to(REPO_ROOT)}")
    return payload


def aggregate(results: list[dict], labels: list[str]) -> dict:
    """Thin adapter over ``pulsebench.aggregate_folds``.

    The per-model shape this phase reports predates the package, so the fold records
    are reshaped into pulsebench's ``model``/``baseline`` form once per comparison and
    the statistics come from there.
    """
    from pulsebench import aggregate_folds as pb_aggregate

    # Derived from the results rather than from a module-level constant, so the
    # aggregation follows whichever models were actually run -- the sequence models
    # are optional, and an earlier version read a name that only existed inside run().
    names = list(results[0]["scores"]) if results else []

    per_model = {}
    for name in names:
        vals = np.array([r["scores"][name]["macro_f1"] for r in results])
        per_model[name] = {
            "mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
            "min": float(vals.min()), "max": float(vals.max()),
            "per_fold": [float(v) for v in vals]}
        for lab in labels:
            lv = np.array([r["scores"][name][f"f1_{lab}"] for r in results])
            per_model[name][f"f1_{lab}"] = {
                "mean": float(lv.mean()), "std": float(lv.std(ddof=1)),
                "per_fold": [float(v) for v in lv]}

    tests = {}
    for name in names:
        if name == "Persistence":
            continue
        shaped = [{"fold": r["fold"], "support": r.get("support", {}),
                   "model": r["scores"][name],
                   "baseline": r["scores"]["Persistence"]} for r in results]
        pb = pb_aggregate(shaped, labels)
        m = pb["macro_f1"]
        tests[name] = {
            "deltas": m["per_fold_delta"], "mean_delta": m["mean_delta"],
            "std_delta": m["std_delta"], "wins": m["wins"], "n_folds": m["n_folds"],
            "p_two_sided": m["p_two_sided"], "p_one_sided": m["p_one_sided"],
            "significant_two_sided": m["sig_two_sided"],
            "significant_one_sided": m["sig_one_sided"]}
        for lab in RARE:
            k = pb[f"f1_{lab}"]
            tests[name][f"delta_{lab}"] = {
                "mean": k["mean_delta"], "wins": k["wins"],
                "per_fold": k["per_fold_delta"]}
        res = pb["resolution"]
    tests["_resolution"] = {"n_pairs": len(results),
                            "min_p_two_sided": float(2 ** -(len(results) - 1))
                            if len(results) <= 20 else 0.0,
                            "min_p_one_sided": float(2 ** -len(results))
                            if len(results) <= 20 else 0.0}
    return {"per_model": per_model, "tests": tests}


# ----------------------------------------------------------------- report


MONTH = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
         7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def build_report(p: dict) -> str:
    cfg, folds, agg = p["config"], p["folds"], p["aggregate"]
    per, tests = agg["per_model"], agg["tests"]
    res = tests["_resolution"]
    n = p["n_folds"]
    labels = cfg.labels

    def table(header, rows):
        esc = lambda cs: [str(c).replace("|", "\\|") for c in cs]
        b = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return (f"| {' | '.join(esc(header))} |\n"
                f"| {' | '.join(['---'] * len(header))} |\n{b}")

    fold_rows = [[f"{f['fold']}", f["cutoff"][:10],
                  f"{f['eval_start'][:10]} .. {f['eval_end'][:10]}",
                  ", ".join(MONTH[m] for m in f["eval_months"]),
                  f"{f['n_train']:,}", f"{f['n_eval_observed']:,}"] for f in folds]

    names = list(per)
    score_rows = []
    for f in folds:
        row = [f"{f['fold']}"]
        for nm in names:
            row.append(f"{f['scores'][nm]['macro_f1']:.4f}")
        for nm in names[1:]:
            d = f["scores"][nm]["macro_f1"] - f["scores"]["Persistence"]["macro_f1"]
            row.append(f"**{d:+.4f}**" if d > 0 else f"{d:+.4f}")
        score_rows.append(row)
    score_rows.append(["**mean**", *[f"{per[nm]['mean']:.4f}" for nm in names],
                       *[f"{tests[nm]['mean_delta']:+.4f}" for nm in names[1:]]])
    score_rows.append(["**std**", *[f"{per[nm]['std']:.4f}" for nm in names],
                       *[f"{tests[nm]['std_delta']:.4f}" for nm in names[1:]]])

    supports = {lab: [f["support"][lab] for f in folds] for lab in labels}
    sup_rows = [[f"**{lab}**" if lab in RARE else lab,
                 *[f"{f['support'][lab]:,}" for f in folds],
                 f"{sum(supports[lab]):,}"] for lab in labels]

    rare_rows = []
    for lab in RARE:
        for nm in names:
            cells = []
            for f in folds:
                nsup = f["support"][lab]
                v = f["scores"][nm][f"f1_{lab}"]
                cells.append("n/a" if nsup == 0 else f"{v:.3f}")
            usable = [f["scores"][nm][f"f1_{lab}"] for f in folds
                      if f["support"][lab] > 0]
            mean = f"{np.mean(usable):.4f}" if usable else "n/a"
            rare_rows.append([lab, nm, mean, f"{len(usable)}/{len(folds)}",
                              "  ".join(cells)])

    test_rows = []
    for nm in names[1:]:
        t = tests[nm]
        test_rows.append([
            nm, f"{t['mean_delta']:+.4f}", f"{t['std_delta']:.4f}",
            f"**{t['wins']}/{t['n_folds']}**",
            f"{t['p_two_sided']:.4f}", f"{t['p_one_sided']:.4f}",
            "yes" if t["significant_one_sided"] else "no",
        ])

    empty = {lab: sum(1 for f in folds if f["support"][lab] == 0) for lab in RARE}
    tot = {lab: sum(f["support"][lab] for f in folds) for lab in RARE}
    bad = [l for l in RARE if empty[l] >= len(folds) - 1 or tot[l] < 50]
    if bad:
        support_note = (
            "> **" + ", ".join(bad) + " cannot be evaluated on this dataset.** "
            + "; ".join(f"{l} has {tot[l]} samples across all {len(folds)} evaluation "
                        f"blocks combined ({empty[l]} blocks contain none)"
                        for l in bad)
            + ". No number of folds fixes this — the class is not in the data. Any "
              "F1 or coverage figure for these classes below is reported for "
              "completeness and should not be cited as a result. **Phase 11b "
              "validates these classes on reference-grade ground-truth data instead** "
              "(`reports/dhaka_ground_truth_model_h6.md`).")
    else:
        support_note = (
            f"Every advisory class appears in at least "
            f"{len(folds) - max(empty.values())} of {len(folds)} evaluation blocks, so "
            f"the rare-class comparison below is interpretable.")

    verdict = _verdict(p)

    title = ("Rolling-origin cross-validation — horizon {} h".format(cfg.horizon)
             if cfg.dataset == "beijing" else
             "Rolling-origin cross-validation — Bangladesh, horizon {} h".format(cfg.horizon))
    return f"""# {title}

{n} expanding-window folds over the full {len(folds[0]['cutoff'][:4])}-year span, built
to answer one question the single chronological split could not: **how often does a
learned model actually beat persistence?**

Every earlier phase evaluated on one split, and validation and test disagreed — the
unweighted forest wins on test and loses on validation, the class-weighted forest does
the reverse. That disagreement is unresolvable with one split, because the answer
depends on which season lands in the evaluation block.

---

## 1. Protocol

- **Expanding window.** Fold *k* trains on everything before its cutoff and evaluates
  on the contiguous block after it. Training data grows with *k*.
- **Cutoffs** are spread across the timeline: the first
  {p['initial_train_fraction']:.0%} is training-only, the remainder split into {n}
  equal contiguous evaluation blocks. Each block therefore covers a different part of
  the year, which is the point.
- **Embargo of {p['embargo_hours']} hours** between train and eval. Training uses
  samples whose *target* precedes the cutoff; evaluation uses samples whose
  {cfg.window}-hour input window begins at or after it. Without the gap, an evaluation
  sample's input window would contain hours the model was trained on as labels.
- **Scaler refit inside every fold**, on that fold's training rows only. The published
  splits are standardised with global-train statistics, which would leak future
  information into every fold; features are inverted to physical units and
  re-standardised per fold.
- **Station grouping** is unchanged: all 12 stations share one hourly axis, so a
  cutoff divides every station at the same moment, and no sample's window spans two
  stations (guaranteed at construction in Phase 2).
- **Observed labels only** in evaluation (`is_imputed_pm25 == False`).

{table(["Fold", "Cutoff", "Evaluation block", "Months covered", "Train n",
        "Eval n (observed)"], fold_rows)}

---

## 2. Per-fold macro-F1

{table(["Fold", *names, *[f"Δ {nm.split(' (')[0]} {('(cw)' if 'balanced' in nm else '')}".strip()
                          for nm in names[1:]]], score_rows)}

---

## 3. Fold-level comparison against persistence

The unit of analysis is the **fold**, not the sample. A percentile bootstrap is
inappropriate here: it assumes exchangeable draws within one comparison, and with
{n} folds it can only ever resample the same {n} numbers. A **Wilcoxon signed-rank
test** on the {n} paired deltas is the right instrument — it asks whether the fold-level
differences are consistently one-signed without assuming normality.

{table(["Model", "Mean Δ", "Std Δ", "Folds won", "Wilcoxon p (2-sided)",
        "p (1-sided)", "p₁ < 0.05"], test_rows)}

> **A {n}-pair signed-rank test has a hard resolution floor.** The smallest two-sided
> p it can return is **{res['min_p_two_sided']:.4f}** and the smallest one-sided p is
> **{res['min_p_one_sided']:.5f}** — reached only when all {n} deltas share a sign. So
> **no two-sided result here can ever reach α = 0.05, regardless of the data.** That
> is a property of the sample size, not evidence of absence. The one-sided test can
> reach significance, and the **folds-won count is the more informative statistic** at
> this n.

### Class support per evaluation block

An F1 of 0.0 on a class with no samples is not a model result. Support is reported
first so the next table can be read correctly.

{table(["Class", *[f"F{f['fold']}" for f in folds], "Total"], sup_rows)}

{support_note}

### Rare-class F1 across folds

Means are taken over **folds with non-zero support only**; `n/a` marks a block where
the class does not occur.

{table(["Class", "Model", "Mean (usable folds)", "Usable folds", "Per fold"], rare_rows)}

---

## 4. Verdict

{verdict}

Reproduce with `python -m src.models.rolling_cv`.
"""


def _verdict(p: dict) -> str:
    agg, n = p["aggregate"], p["n_folds"]
    per, tests = agg["per_model"], agg["tests"]
    names = [k for k in per if k != "Persistence"]
    parts = []

    for nm in names:
        t = tests[nm]
        w, md = t["wins"], t["mean_delta"]
        if w == n and t["significant_one_sided"]:
            parts.append(
                f"**{nm} beats persistence in all {n}/{n} folds** (mean "
                f"{md:+.4f}, one-sided Wilcoxon p = {t['p_one_sided']:.5f}). Every "
                f"evaluation block, every season. This is the consistency a single "
                f"split could not demonstrate.")
        elif w == 0:
            parts.append(
                f"**{nm} loses to persistence in all {n}/{n} folds** (mean "
                f"{md:+.4f}). Not a split artifact: it is worse everywhere.")
        elif w >= n - 1:
            parts.append(
                f"**{nm} beats persistence in {w}/{n} folds** (mean {md:+.4f}, "
                f"one-sided p = {t['p_one_sided']:.5f}) — consistent, with one "
                f"exception worth naming in the write-up.")
        else:
            parts.append(
                f"**{nm} beats persistence in only {w}/{n} folds** (mean "
                f"{md:+.4f}, std {t['std_delta']:.4f}). The sign of the effect "
                f"changes with the evaluation block, which is precisely the "
                f"instability the single-split analysis suspected.")

    best = max(names, key=lambda k: tests[k]["wins"])
    bw = tests[best]["wins"]
    if bw == n:
        closing = (
            f"**The open question from the single-split analysis is resolved.** "
            f"`{best}` clears the persistence floor in every fold, so the earlier "
            f"validation/test disagreement was an artifact of one arbitrary "
            f"chronological cut rather than a property of the model. The thesis can "
            f"state that a learned model beats persistence at h=6, with {n}-fold "
            f"rolling-origin evidence behind it — and should report the per-fold "
            f"spread rather than a single number.")
    elif bw >= (n + 1) // 2:
        closing = (
            f"**Partially resolved, and the honest reading is 'usually but not "
            f"always'.** The best model (`{best}`) clears the floor in {bw} of {n} "
            f"folds. That is stronger evidence than a single split, and weaker than "
            f"a clean win. The thesis should report the fold count directly — "
            f"\"{bw}/{n} folds\" is a more honest headline than any mean, and the "
            f"folds where it loses should be characterised by season.")
    else:
        closing = (
            f"**The instability is confirmed, and it is not a single-split "
            f"artifact.** No model clears the persistence floor in a majority of "
            f"folds (best: `{best}` at {bw}/{n}). Five evaluation blocks spanning "
            f"different seasons give the same answer the single split hinted at: at a "
            f"6-hour horizon on these nine channels, a learned model is not reliably "
            f"better than assuming the next six hours look like now. This upgrades "
            f"the finding from a possible artifact to a robust multi-fold negative "
            f"result, which is the publishable version of it.")
    parts.append(closing)
    return "\n\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="beijing",
                    choices=("beijing", "bangladesh"))
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--initial-fraction", type=float, default=INITIAL_TRAIN_FRACTION)
    ap.add_argument("--with-sequence-models", action="store_true",
                    help="also run the LSTM and Transformer in every fold "
                         "(minutes per fold rather than seconds)")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    run(load_config(dataset=args.dataset, n_folds=args.folds,
                    initial_fraction=args.initial_fraction),
        write=not args.no_write,
        with_sequence_models=args.with_sequence_models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
