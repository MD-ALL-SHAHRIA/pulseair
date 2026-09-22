"""Split conformal prediction on top of the Phase 3 RandomForest.

Turns the forest's point prediction into a **prediction set** with a finite-sample
coverage guarantee: over the test distribution, the set contains the true AQI category
at least ``1 - alpha`` of the time. No distributional assumption, no retraining — the
forest is used exactly as saved.

Phase 5 established that the LSTM and Transformer do not beat the forest and are in
fact significantly worse on Hazardous. Rather than keep a sequence model around purely
for MC-dropout confidence, uncertainty is added to the model that actually won.

The procedure (split conformal, inverse-probability score):

1. Score the **validation** split, observed rows only. Nonconformity is
   ``s = 1 - P(true class)`` — high when the forest put little mass on the truth.
2. Take ``q`` = the ``ceil((n+1)(1-alpha)) / n`` empirical quantile of those scores.
   The ``(n+1)`` is not a rounding detail: it is what makes the guarantee hold in
   finite samples rather than asymptotically.
3. For a test point, the set is every class with ``1 - P(class) <= q``, i.e. every
   class the forest gives at least ``1 - q`` probability.

**What the guarantee covers.** Coverage is *marginal* — averaged over all test points.
It says nothing about any particular class, and the per-class table in the report is
there because marginal coverage routinely hides a class that is badly under-covered.
For an advisory device the classes most likely to be under-covered are exactly the ones
that matter.

    python -m src.models.conformal

Writes ``reports/conformal_h6.md``.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
ADVISORY = ("Very unhealthy", "Hazardous")


@dataclass(frozen=True)
class ConformalConfig:
    processed_root: Path
    artifacts_dir: Path
    reports_dir: Path
    horizon: int
    labels: list[str]
    target_coverage: float
    calibration_split: str

    @property
    def alpha(self) -> float:
        return 1.0 - self.target_coverage

    @property
    def processed_dir(self) -> Path:
        return self.processed_root / f"h{self.horizon}"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / f"baseline_h{self.horizon}.pkl"

    @property
    def report_path(self) -> Path:
        return self.reports_dir / f"conformal_h{self.horizon}.md"

    @property
    def artifact_path(self) -> Path:
        return self.artifacts_dir / f"conformal_h{self.horizon}.pkl"


def load_config(path: Path | str = DEFAULT_CONFIG, root: Path | None = None,
                horizon: int | None = None) -> ConformalConfig:
    root = root or REPO_ROOT
    raw = yaml.safe_load(Path(path).read_text())
    data, cf, prep = raw["data"], raw["conformal"], raw["preprocessing"]
    return ConformalConfig(
        processed_root=root / data["processed_dir"],
        artifacts_dir=root / raw["baseline"]["artifacts_dir"],
        reports_dir=root / "reports",
        horizon=int(horizon if horizon is not None else prep["horizon"]),
        labels=list(data["pm25_labels"]),
        target_coverage=float(cf["target_coverage"]),
        calibration_split=str(cf["calibration_split"]),
    )


# ------------------------------------------------------------------------- core


def load_probs(cfg: ConformalConfig, split: str, bundle: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Class probabilities, labels and the observed mask for one split."""
    frame = pd.read_csv(cfg.processed_dir / f"tabular_{split}.csv")
    X = frame[bundle["feature_columns"]].to_numpy(dtype=np.float32)
    y = frame["y_category"].to_numpy()
    observed = ~frame["is_imputed_pm25"].to_numpy(dtype=bool)
    probs = bundle["model"].predict_proba(X)

    # predict_proba columns follow model.classes_, which need not be 0..K-1 if a class
    # were absent from training. Re-index into label order rather than assume.
    classes = list(bundle["model"].classes_)
    full = np.zeros((len(frame), len(cfg.labels)), dtype=np.float64)
    for col, cls in enumerate(classes):
        full[:, int(cls)] = probs[:, col]
    return full, y, observed, frame


def calibrate(probs: np.ndarray, y: np.ndarray, alpha: float) -> dict:
    """The conformal threshold from calibration scores.

    ``q`` is the ``ceil((n+1)(1-alpha))``-th smallest score. If that index exceeds n --
    which happens when the calibration set is too small for the requested coverage --
    no finite threshold exists and the guarantee cannot be met; that is reported rather
    than silently clipped.
    """
    n = len(y)
    scores = 1.0 - probs[np.arange(n), y]
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        raise ValueError(
            f"calibration set of {n} is too small for {1 - alpha:.0%} coverage: "
            f"needs at least {int(np.ceil(1 / alpha)) - 1} points")
    q = float(np.sort(scores)[k - 1])
    return {
        "n_calibration": n, "alpha": alpha, "q": q, "rank": k,
        "score_mean": float(scores.mean()), "score_median": float(np.median(scores)),
        "prob_floor": 1.0 - q,
    }


def calibrate_mondrian(probs: np.ndarray, y: np.ndarray, alpha: float,
                       n_classes: int) -> dict:
    """One threshold per class -- Mondrian (class-conditional) conformal.

    Marginal conformal pools every calibration point into one quantile, so a class
    that is rare and hard absorbs the slack: on this data Hazardous came out at 84.3%
    against a 90% target. Mondrian calibrates within each class instead, which turns
    the marginal guarantee into a per-class one.

    The cost is real and worth stating: each threshold is estimated from only that
    class's calibration points, so a thin class gets a noisier threshold, and the sets
    get larger because the hard classes now need a looser cut to reach their target.
    """
    thresholds, detail = {}, {}
    for c in range(n_classes):
        m = y == c
        n = int(m.sum())
        if n == 0:
            thresholds[c] = 1.0
            detail[c] = {"n": 0, "q": 1.0, "rank": None,
                         "note": "no calibration points; class always included"}
            continue
        scores = 1.0 - probs[m, c]
        k = int(np.ceil((n + 1) * (1.0 - alpha)))
        if k > n:
            # Too few points to certify this class at the requested level. Including
            # it unconditionally is the conservative choice -- it cannot under-cover,
            # it only inflates set size -- and it is reported rather than hidden.
            thresholds[c] = 1.0
            detail[c] = {"n": n, "q": 1.0, "rank": None,
                         "note": f"n={n} too small for {1 - alpha:.0%}; always included"}
            continue
        q = float(np.sort(scores)[k - 1])
        thresholds[c] = q
        detail[c] = {"n": n, "q": q, "rank": k, "note": ""}
    return {"thresholds": thresholds, "detail": detail, "alpha": alpha}


def prediction_sets(probs: np.ndarray, q: float) -> np.ndarray:
    """Boolean (n, n_classes) membership matrix: every class with 1 - P <= q."""
    return (1.0 - probs) <= q + 1e-12


def prediction_sets_mondrian(probs: np.ndarray, thresholds: dict) -> np.ndarray:
    """Membership under per-class thresholds: class c joins when 1 - P_c <= q_c."""
    out = np.zeros(probs.shape, dtype=bool)
    for c, q in thresholds.items():
        out[:, int(c)] = (1.0 - probs[:, int(c)]) <= q + 1e-12
    return out


def evaluate_sets(sets: np.ndarray, y: np.ndarray, labels: list[str]) -> dict:
    """Coverage and set size, overall and conditioned on the true class."""
    n = len(y)
    covered = sets[np.arange(n), y]
    sizes = sets.sum(axis=1)

    per_class = {}
    for i, lab in enumerate(labels):
        m = y == i
        per_class[lab] = {
            "n": int(m.sum()),
            "coverage": float(covered[m].mean()) if m.any() else None,
            "mean_set_size": float(sizes[m].mean()) if m.any() else None,
            "median_set_size": float(np.median(sizes[m])) if m.any() else None,
        }

    size_hist = {int(s): int((sizes == s).sum()) for s in range(0, len(labels) + 1)
                 if (sizes == s).any()}
    return {
        "n": n,
        "coverage": float(covered.mean()),
        "mean_set_size": float(sizes.mean()),
        "median_set_size": float(np.median(sizes)),
        "singleton_rate": float((sizes == 1).mean()),
        "empty_rate": float((sizes == 0).mean()),
        "size_histogram": size_hist,
        "per_class": per_class,
    }


def run(cfg: ConformalConfig | None = None, *, write: bool = True,
        verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    say = print if verbose else (lambda *a, **k: None)

    bundle = joblib.load(cfg.model_path)
    say(f"model: {bundle['model_name']} from {cfg.model_path.name} "
        f"(h{cfg.horizon}, selected on {bundle['selected_on'][:24]}...)")

    cal_probs, cal_y, cal_obs, _ = load_probs(cfg, cfg.calibration_split, bundle)
    test_probs, test_y, test_obs, test_frame = load_probs(cfg, "test", bundle)
    say(f"calibration ({cfg.calibration_split}): {int(cal_obs.sum()):,} observed rows "
        f"of {len(cal_y):,}")
    say(f"test: {int(test_obs.sum()):,} observed rows of {len(test_y):,}")

    calib = calibrate(cal_probs[cal_obs], cal_y[cal_obs], cfg.alpha)
    say(f"threshold q = {calib['q']:.4f} (rank {calib['rank']} of "
        f"{calib['n_calibration']:,}) -> a class joins the set at "
        f"P >= {calib['prob_floor']:.4f}")

    sets = prediction_sets(test_probs[test_obs], calib["q"])
    ev = evaluate_sets(sets, test_y[test_obs], cfg.labels)

    mond = calibrate_mondrian(cal_probs[cal_obs], cal_y[cal_obs], cfg.alpha,
                              len(cfg.labels))
    mond_sets = prediction_sets_mondrian(test_probs[test_obs], mond["thresholds"])
    mond_ev = evaluate_sets(mond_sets, test_y[test_obs], cfg.labels)
    say(f"empirical coverage {ev['coverage']:.4f} (target {cfg.target_coverage:.2f}) | "
        f"mean set size {ev['mean_set_size']:.3f} | singletons "
        f"{ev['singleton_rate']:.1%} | empty {ev['empty_rate']:.2%}")
    for lab in ADVISORY:
        c = ev["per_class"][lab]
        say(f"  marginal  {lab:16s} coverage {c['coverage']:.4f}  "
            f"mean set {c['mean_set_size']:.2f}")
    say(f"mondrian: coverage {mond_ev['coverage']:.4f} | mean set size "
        f"{mond_ev['mean_set_size']:.3f} | singletons {mond_ev['singleton_rate']:.1%}")
    for lab in ADVISORY:
        c = mond_ev["per_class"][lab]
        say(f"  mondrian  {lab:16s} coverage {c['coverage']:.4f}  "
            f"mean set {c['mean_set_size']:.2f}")

    # A sanity check that the calibration itself is sound: coverage measured back on
    # the calibration set should sit at the target by construction.
    cal_sets = prediction_sets(cal_probs[cal_obs], calib["q"])
    cal_cov = float(cal_sets[np.arange(cal_obs.sum()), cal_y[cal_obs]].mean())

    payload = {
        "config": cfg, "calibration": calib, "test": ev,
        "mondrian": {"calibration": mond, "test": mond_ev},
        "calibration_selfcheck": cal_cov,
        "model_name": bundle["model_name"],
        "counts": {"calibration_observed": int(cal_obs.sum()),
                   "test_observed": int(test_obs.sum()),
                   "test_total": len(test_y)},
    }

    if write:
        joblib.dump({
            # Marginal (pooled) threshold.
            "q": calib["q"], "alpha": cfg.alpha,
            "target_coverage": cfg.target_coverage,
            "prob_floor": calib["prob_floor"],
            "empirical_test_coverage": ev["coverage"],
            # Mondrian (class-conditional) thresholds. This is what the advisory layer
            # uses, because it is the one that certifies Hazardous specifically.
            "mondrian_thresholds": {int(k): float(v)
                                    for k, v in mond["thresholds"].items()},
            "mondrian_test_coverage": mond_ev["coverage"],
            "mondrian_per_class_coverage": {
                l: mond_ev["per_class"][l]["coverage"] for l in cfg.labels},
            "default_method": "mondrian",
            "class_labels": cfg.labels,
            "feature_columns": bundle["feature_columns"],
            "base_model": cfg.model_path.name,
            "calibrated_on": f"{cfg.calibration_split} split, observed rows "
                             f"({calib['n_calibration']:,})",
        }, cfg.artifact_path)
        say(f"saved threshold -> {cfg.artifact_path.relative_to(REPO_ROOT)}")

        cfg.report_path.write_text(build_report(payload))
        say(f"wrote {cfg.report_path.relative_to(REPO_ROOT)}")
        mpath = cfg.reports_dir / "metrics" / f"conformal_h{cfg.horizon}.json"
        mpath.parent.mkdir(parents=True, exist_ok=True)
        mpath.write_text(json.dumps({k: v for k, v in payload.items() if k != "config"},
                                    indent=2, default=str))
        say(f"wrote {mpath.relative_to(REPO_ROOT)}")
    return payload


# ----------------------------------------------------------------------- report


def build_report(payload: dict) -> str:
    cfg, calib, ev = payload["config"], payload["calibration"], payload["test"]
    counts, labels = payload["counts"], cfg.labels

    def table(header, rows):
        esc = lambda cs: [str(c).replace("|", "\\|") for c in cs]
        b = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return f"| {' | '.join(header)} |\n| {' | '.join(['---'] * len(header))} |\n{b}"

    target = cfg.target_coverage
    cls_rows = []
    for lab in labels:
        c = ev["per_class"][lab]
        short = c["coverage"] - target
        flag = "**UNDER**" if short < -0.01 else ("over" if short > 0.01 else "on target")
        cls_rows.append([
            f"**{lab}**" if lab in ADVISORY else lab, f"{c['n']:,}",
            f"{c['coverage']:.4f}", f"{short:+.4f}", flag,
            f"{c['mean_set_size']:.2f}", f"{c['median_set_size']:.0f}",
        ])

    meanings = {0: "no category qualifies — the reading is unlike calibration data",
                1: "confident: a single category", 2: "two plausible categories",
                3: "three categories", 4: "four", 5: "five",
                6: "all six — effectively no information"}
    size_rows = [[str(s), f"{n:,}", f"{n / ev['n'] * 100:.2f}%", meanings.get(s, str(s))]
                 for s, n in sorted(ev["size_histogram"].items())]

    under = [l for l in labels if ev["per_class"][l]["coverage"] is not None
             and ev["per_class"][l]["coverage"] < target - 0.01]
    under_adv = [l for l in under if l in ADVISORY]

    if under_adv:
        worst = min(under_adv, key=lambda l: ev["per_class"][l]["coverage"])
        w = ev["per_class"][worst]
        thin = min(ev["per_class"][l]["n"] for l in under_adv)
        conditional_note = f"""> **The advisory classes are under-covered. State this in the thesis.**

{', '.join(f'**{l}**' for l in under_adv)} fall below the {target:.0%} target. The worst is
**{worst}** at **{w['coverage']:.4f}** — {abs(w['coverage'] - target):.4f} short, across
{w['n']:,} test cases. In plain terms: when the true category is {worst}, the prediction
set misses it about {(1 - w['coverage']) * 100:.0f}% of the time, not {(1 - target) * 100:.0f}%.

**This is not a calibration bug — it is exactly what the guarantee does and does not
promise.** Split conformal delivers *marginal* coverage: averaged over all test points
the set contains the truth {ev['coverage']:.1%} of the time, and it does. Nothing in the
procedure equalises coverage across classes, and when a class is both rare and hard, the
slack lands there. For a device whose purpose is warning about hazardous air, the
marginal number flatters and the per-class number is the one that matters.

Two honest options, neither free:

- **Mondrian (class-conditional) conformal** — a separate threshold per class. Buys
  per-class coverage at the cost of larger sets and a thinner effective calibration set
  for the rare classes ({thin:,} points for the worst one here, workable but not
  generous).
- **Keep marginal coverage and state the limitation**, ensuring the advisory copy never
  implies a per-category guarantee that does not exist.

Until one is chosen: **the advisory layer must not claim "{target:.0%} confident" for a
{worst} prediction.** The {target:.0%} is a fleet-wide average, not a promise about that
reading."""
    elif under:
        conditional_note = (
            f"Both advisory classes clear the target. {', '.join(under)} sit below it, "
            f"but those are not the classes the device warns on.")
    else:
        conditional_note = (
            f"**Every class clears the {target:.0%} target.** Marginal coverage does not "
            f"imply conditional coverage in general, so this is a property of this "
            f"calibration rather than a guarantee — but no class is under-covered here.")

    empty_note = ""
    if ev["empty_rate"] > 0:
        empty_note = f"""
**{ev['empty_rate']:.2%} of sets are empty** ({round(ev['empty_rate'] * ev['n']):,} cases).
No category cleared the {calib['prob_floor']:.4f} floor. That is information, not a
failure — it says the reading is unlike the calibration data. The advisory layer should
surface it as *no reliable prediction* rather than silently falling back to the argmax,
which would quietly void the coverage guarantee."""

    # --- Mondrian comparison -------------------------------------------------------
    mo = payload["mondrian"]["test"]
    mc = payload["mondrian"]["calibration"]
    mond_rows = []
    for i, lab in enumerate(labels):
        d = mc["detail"][i]
        marg = ev["per_class"][lab]
        mon = mo["per_class"][lab]
        mond_rows.append([
            f"**{lab}**" if lab in ADVISORY else lab,
            f"{d['n']:,}", "—" if d["rank"] is None else f"{d['q']:.4f}",
            f"{marg['coverage']:.4f}", f"{mon['coverage']:.4f}",
            f"{mon['coverage'] - marg['coverage']:+.4f}",
            f"{marg['mean_set_size']:.2f}", f"{mon['mean_set_size']:.2f}",
        ])
    still_under = [l for l in labels
                   if mo["per_class"][l]["coverage"] is not None
                   and mo["per_class"][l]["coverage"] < target - 0.01]
    adv_fixed = all(mo["per_class"][l]["coverage"] >= target - 0.01 for l in ADVISORY)
    mond_verdict = (
        f"**Mondrian fixes the advisory classes.** "
        + ", ".join(f"{l} {ev['per_class'][l]['coverage']:.4f} → "
                    f"**{mo['per_class'][l]['coverage']:.4f}**" for l in ADVISORY)
        + f". The cost is set size: mean {ev['mean_set_size']:.3f} → "
          f"**{mo['mean_set_size']:.3f}**, singletons {ev['singleton_rate']:.1%} → "
          f"**{mo['singleton_rate']:.1%}**. That is the trade in plain terms — a "
          f"genuine per-class guarantee is paid for with wider, less decisive sets."
        if adv_fixed else
        f"**Mondrian does not fully fix it.** "
        + ", ".join(f"{l} {mo['per_class'][l]['coverage']:.4f}" for l in ADVISORY)
        + ". Thin calibration classes give noisy thresholds; see the per-class n below.")
    if still_under:
        mond_verdict += (f" Still below target after Mondrian: "
                         f"{', '.join(still_under)}.")

    mondrian_section = f"""---

## 4. Mondrian (class-conditional) conformal

The marginal threshold above is a single number pooled over every calibration point,
so a class that is both rare and hard absorbs the slack — which is what put Hazardous
at {ev['per_class']['Hazardous']['coverage']:.4f}. Mondrian calibrates a separate
quantile *within each class*, converting the marginal guarantee into a per-class one.

{table(["Class", "Calib n", "q (per class)", "Coverage: marginal", "Coverage: Mondrian",
        "Δ", "Set size: marginal", "Set size: Mondrian"], mond_rows)}

| | Marginal | Mondrian |
|---|---|---|
| Overall coverage | {ev['coverage']:.4f} | **{mo['coverage']:.4f}** |
| Mean set size | {ev['mean_set_size']:.3f} | **{mo['mean_set_size']:.3f}** |
| Median set size | {ev['median_set_size']:.0f} | **{mo['median_set_size']:.0f}** |
| Singletons | {ev['singleton_rate']:.1%} | **{mo['singleton_rate']:.1%}** |
| Empty sets | {ev['empty_rate']:.2%} | **{mo['empty_rate']:.2%}** |

{mond_verdict}

**Mondrian is what the advisory layer uses** (`default_method: "mondrian"` in
`{cfg.artifact_path.name}`). The reason is specific: an advisory that says "this could
be Hazardous" needs the Hazardous guarantee to hold for Hazardous readings, not on
average across a fleet where the common classes carry the average. Larger sets are the
honest price, and the advisory copy names the ambiguity rather than hiding it.

"""

    return f"""# Conformal prediction — horizon {cfg.horizon} h

Split conformal prediction wrapped around **{payload['model_name']}**
(`{cfg.model_path.name}`), the Phase 3 baseline, used exactly as saved — no retraining.

> **Scope note (added after Phase 10).** This report documents the conformal method as
> developed on **Beijing** data. It is **not** the deployed system. Rolling-origin CV
> later showed no Beijing-trained model beats persistence in more than 2 of 5 folds
> (`reports/rolling_origin_cv_h{cfg.horizon}.md`), so nothing here ships. The deployed
> candidate is the Bangladesh-native model in `reports/bangladesh_deployment.md`, which
> re-derives these thresholds on its own data. Read this as the calibration
> methodology, which transferred; not as a predictor, which did not.

Phase 5 found the LSTM and Transformer significantly worse than this forest, and worse
on Hazardous specifically (`reports/dl_metrics_h{cfg.horizon}.md`). So the uncertainty
machinery is attached to the model that won rather than keeping a sequence model around
for MC dropout; no deep model is carried forward.

---

## 1. Calibration

| | |
|---|---|
| Calibration set | `{cfg.calibration_split}` split, observed rows only |
| Calibration points | {calib['n_calibration']:,} |
| Nonconformity score | `1 − P(true class)` |
| Target coverage | **{target:.0%}** (α = {cfg.alpha:.2f}) |
| Quantile rank | {calib['rank']:,} of {calib['n_calibration']:,} — `⌈(n+1)(1−α)⌉` |
| **Threshold q** | **{calib['q']:.4f}** |
| Set rule | include every class with `P ≥ {calib['prob_floor']:.4f}` |

Calibration uses **observed rows only** (`is_imputed_pm25 == False`), consistent with
every metric in this project: a threshold tuned against forward-filled labels would
inherit the Hazardous inflation documented in Phase 2.

The `(n+1)` in the rank is what makes this a finite-sample guarantee rather than an
asymptotic one. Measured back on the calibration set, coverage is
{payload['calibration_selfcheck']:.4f} — at target by construction, which checks the
quantile was taken correctly and says nothing about generalisation.

---

## 2. Empirical coverage on test

{counts['test_observed']:,} observed test rows of {counts['test_total']:,}.

| | Target | Empirical | Difference |
|---|---|---|---|
| **Marginal coverage** | {target:.4f} | **{ev['coverage']:.4f}** | {ev['coverage'] - target:+.4f} |

### Per-class conditional coverage

{table(["Class", "n", "Coverage", "vs target", "Status", "Mean set size", "Median"],
       cls_rows)}

{conditional_note}

---

## 3. Prediction set size

The actionable number. A set of one is a confident call; anything larger is genuine
ambiguity the advisory layer has to communicate rather than hide.

| | |
|---|---|
| Mean set size | **{ev['mean_set_size']:.3f}** |
| Median set size | **{ev['median_set_size']:.0f}** |
| Singletons (size 1) | **{ev['singleton_rate']:.1%}** |
| Empty (size 0) | {ev['empty_rate']:.2%} |

{table(["Set size", "n", "Share", "Meaning"], size_rows)}
{empty_note}

At {target:.0%} coverage this model is confident — a single category — only
{ev['singleton_rate']:.1%} of the time. The median reading yields
{ev['median_set_size']:.0f} categories. **That is the honest operating point**, and it
follows directly from the Phase 3–5 finding that this task is hard six hours out: a
model that cannot separate adjacent AQI bands cannot produce small sets, and forcing
smaller ones would only move the error from "visibly ambiguous" to "quietly wrong".

### By true class

{table(["Class", "n", "Mean set size", "Median"],
       [[f"**{l}**" if l in ADVISORY else l, f"{ev['per_class'][l]['n']:,}",
         f"{ev['per_class'][l]['mean_set_size']:.2f}",
         f"{ev['per_class'][l]['median_set_size']:.0f}"] for l in labels])}

{mondrian_section}---

## 5. What this replaces

| Before | After this phase |
|---|---|
| Point prediction, no uncertainty | Prediction set with {target:.0%} marginal coverage |
| MC dropout on a sequence model that lost to the forest | Conformal wrapper on the forest that won |
| Confidence as a softmax number with no operational meaning | Set size, directly interpretable |

`{cfg.artifact_path.name}` holds the threshold, probability floor, feature order and
class labels — everything needed to build sets at inference. It is ~1 KB and pairs with
the existing `{cfg.model_path.name}`.

**No LSTM or Transformer is carried forward as a deployed artifact.**
`dl_model_h{cfg.horizon}.pt` stays in `src/models/artifacts/` as the record of the
Phase 5 experiment and should not be loaded by the advisory layer.

Reproduce with `python -m src.models.conformal`.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--coverage", type=float, default=None,
                    help="override conformal.target_coverage")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, horizon=args.horizon)
    if args.coverage is not None:
        from dataclasses import replace
        cfg = replace(cfg, target_coverage=args.coverage)
    run(cfg, write=not args.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
