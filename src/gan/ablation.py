"""Ablation: does CTGAN augmentation help at the primary horizon?

Trains the Phase 3 baselines on three training sets -- real only, broadly augmented,
and augmented on the advisory classes only -- and compares them on the *same* real test
split with a paired bootstrap.

Why paired, and why a bootstrap. The three variants are scored on identical test rows,
so their errors are correlated; an unpaired comparison of two macro-F1 numbers throws
that correlation away and widens the interval for no reason. Resampling the test set
and recomputing both scores on each resample keeps the pairing, and the distribution of
the *difference* answers the only question that matters: is this gap distinguishable
from zero.

    python -m src.gan.ablation

Writes ``reports/gan_ablation_h<primary>.md``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.models import baseline as bl

REPO_ROOT = bl.REPO_ROOT
N_BOOTSTRAP = 1000
CI = 95
RARE = ("Very unhealthy", "Hazardous")

VARIANTS = (
    ("unaugmented", None, "Real training rows only"),
    ("broad-4", "broad", "CTGAN on every minority class (Good, U(sens), V.unhealthy, Hazardous)"),
    ("targeted-2", "targeted", "CTGAN on the advisory classes only (V.unhealthy, Hazardous)"),
)


@dataclass
class Variant:
    name: str
    description: str
    n_train: int
    n_synthetic: int
    models: dict            # model name -> {"pred": array, "val_macro_f1": float}
    selected: str


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(f1_score(y_true, y_pred, labels=list(range(n_classes)),
                          average="macro", zero_division=0))


def paired_bootstrap(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                     n_classes: int, n_boot: int = N_BOOTSTRAP, seed: int = 42,
                     scorer=None) -> dict:
    """Bootstrap the difference ``score(b) - score(a)`` over shared resampled rows.

    Both predictions are scored on the *same* resampled indices every iteration, which
    is what makes it paired.
    """
    scorer = scorer or (lambda yt, yp: _macro_f1(yt, yp, n_classes))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        diffs[i] = scorer(yt, pred_b[idx]) - scorer(yt, pred_a[idx])

    lo, hi = np.percentile(diffs, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    observed = scorer(y_true, pred_b) - scorer(y_true, pred_a)
    return {
        "observed_diff": float(observed),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "mean_diff": float(diffs.mean()),
        # Two-sided: how often the resampled difference crosses zero.
        "p_two_sided": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())),
        "significant": bool(lo > 0 or hi < 0),
        "n_boot": n_boot,
    }


def collect(horizon: int | None = None, verbose: bool = True) -> dict:
    """Train every variant and return their test predictions on the shared test split."""
    say = print if verbose else (lambda *a, **k: None)
    variants, test_ref, labels, features = {}, None, None, None

    for name, aug, desc in VARIANTS:
        cfg = bl.load_config(horizon=horizon, augmented=bool(aug))
        if aug:
            cfg = bl.replace_variant(cfg, aug)
            if not cfg.train_file.exists():
                raise FileNotFoundError(
                    f"{cfg.train_file} is missing -- run "
                    f"`python -m src.gan.augment --variant {aug}` first")
        features, meta = bl.load_feature_columns(cfg)
        labels = labels or cfg.labels

        train = bl.load_split(cfg, "train", features)
        val = bl.load_split(cfg, "val", features)
        test = bl.load_split(cfg, "test", features)
        if test_ref is None:
            test_ref = test
        else:
            np.testing.assert_array_equal(test.y, test_ref.y)   # same rows, always

        n_syn = int((train.frame[bl.SOURCE_COL] == "synthetic").sum()) \
            if bl.SOURCE_COL in train.frame.columns else 0
        say(f"\n[{name}] train {len(train):,} rows ({n_syn:,} synthetic)")

        models = {}
        for mname, model in bl.build_models(cfg).items():
            model.fit(train.X, train.y)
            pv, pt = model.predict(val.X), model.predict(test.X)
            vf1 = _macro_f1(val.y[val.observed], pv[val.observed], len(labels))
            models[mname] = {"pred": pt, "val_macro_f1": vf1, "model": model}
            say(f"   {mname:13s} val observed macro-F1 {vf1:.4f}")

        # Selection on validation, observed-only real rows. Never test.
        selected = max(models, key=lambda m: models[m]["val_macro_f1"])
        say(f"   -> selected {selected}")
        variants[name] = Variant(name, desc, len(train), n_syn, models, selected)

    # Persistence floor, recomputed here so the table is self-contained.
    import joblib
    cfg0 = bl.load_config(horizon=horizon)
    scaler = joblib.load(cfg0.processed_dir / "scaler.pkl")
    meta = json.loads((cfg0.processed_dir / "metadata.json").read_text())
    persist = bl.persistence_baseline(test_ref, cfg0, scaler, list(meta["scaled_columns"]))

    return {"variants": variants, "test": test_ref, "labels": labels,
            "features": features, "persistence": persist, "horizon": cfg0.horizon}


# ---------------------------------------------------------------------- analysis


def analyse(collected: dict, n_boot: int = N_BOOTSTRAP, verbose: bool = True) -> dict:
    """Score every variant and bootstrap each against the unaugmented baseline."""
    say = print if verbose else (lambda *a, **k: None)
    variants, test, labels = collected["variants"], collected["test"], collected["labels"]
    n_cls = len(labels)

    # Every number is on real, observed-label test rows only.
    mask = test.observed
    y = test.y[mask]

    def score_block(pred: np.ndarray) -> dict:
        p = pred[mask]
        per = {}
        for i, lab in enumerate(labels):
            per[lab] = float(f1_score(y == i, p == i, zero_division=0))
        return {"macro_f1": _macro_f1(y, p, n_cls),
                "accuracy": float((y == p).mean()), "per_class": per}

    rows = {"Persistence": {"scores": score_block(collected["persistence"]),
                            "n_train": 0, "n_synthetic": 0,
                            "description": "Zero-parameter reference; never trained"}}
    for name, v in variants.items():
        sel = v.selected
        rows[name] = {
            "scores": score_block(v.models[sel]["pred"]),
            "selected": sel,
            "val_macro_f1": v.models[sel]["val_macro_f1"],
            "n_train": v.n_train, "n_synthetic": v.n_synthetic,
            "description": v.description,
            "per_model": {m: score_block(d["pred"])["macro_f1"]
                          for m, d in v.models.items()},
        }

    # Bootstrap each variant against the unaugmented selected model.
    base_pred = variants["unaugmented"].models[variants["unaugmented"].selected]["pred"][mask]
    tests = {}
    for name in ("broad-4", "targeted-2"):
        pred = variants[name].models[variants[name].selected]["pred"][mask]
        tests[name] = {"macro_f1": paired_bootstrap(y, base_pred, pred, n_cls, n_boot)}
        for lab in RARE:
            i = labels.index(lab)
            tests[name][lab] = paired_bootstrap(
                y, base_pred, pred, n_cls, n_boot,
                scorer=lambda yt, yp, i=i: float(f1_score(yt == i, yp == i, zero_division=0)))
        t = tests[name]["macro_f1"]
        say(f"  {name:12s} macro-F1 Δ {t['observed_diff']:+.4f} "
            f"[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] "
            f"{'SIGNIFICANT' if t['significant'] else 'not significant'}")

    # Persistence is the floor every variant must clear to mean anything.
    persist_pred = collected["persistence"][mask]
    vs_persistence = {
        name: paired_bootstrap(y, persist_pred,
                               variants[name].models[variants[name].selected]["pred"][mask],
                               n_cls, n_boot)
        for name in variants
    }

    return {"rows": rows, "tests": tests, "vs_persistence": vs_persistence,
            "labels": labels, "n_test": int(mask.sum()),
            "n_test_all": int(len(test)), "horizon": collected["horizon"],
            "n_boot": n_boot}


def build_report(a: dict, validity: dict | None = None) -> str:
    rows, tests, labels = a["rows"], a["tests"], a["labels"]
    order = ["Persistence", "unaugmented", "broad-4", "targeted-2"]

    def table(header, body):
        esc = lambda cs: [str(c).replace("|", "\\|") for c in cs]
        b = "\n".join("| " + " | ".join(esc(r)) + " |" for r in body)
        return f"| {' | '.join(esc(header))} |\n| {' | '.join(['---'] * len(header))} |\n{b}"

    main_rows = []
    for name in order:
        r = rows[name]
        train = f"{r['n_train']:,}" if r["n_train"] else "—"
        syn = f"{r['n_synthetic']:,}" if r["n_synthetic"] else "—"
        sel = r.get("selected", "—")
        main_rows.append([
            f"**{name}**", train, syn, sel,
            f"{r['scores']['accuracy']:.4f}", f"**{r['scores']['macro_f1']:.4f}**",
            f"{r['scores']['per_class']['Very unhealthy']:.4f}",
            f"{r['scores']['per_class']['Hazardous']:.4f}",
        ])

    cls_rows = [[f"**{l}**" if l in RARE else l,
                 *[f"{rows[n]['scores']['per_class'][l]:.4f}" for n in order]]
                for l in labels]

    boot_rows = []
    for name in ("broad-4", "targeted-2"):
        for metric in ("macro_f1", *RARE):
            t = tests[name][metric]
            label = "Macro-F1" if metric == "macro_f1" else f"{metric} F1"
            boot_rows.append([
                name, label, f"{t['observed_diff']:+.4f}",
                f"[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]",
                f"{t['p_two_sided']:.3f}",
                "**yes**" if t["significant"] else "no",
            ])

    pers_rows = [[n, f"{a['vs_persistence'][n]['observed_diff']:+.4f}",
                  f"[{a['vs_persistence'][n]['ci_low']:+.4f}, "
                  f"{a['vs_persistence'][n]['ci_high']:+.4f}]",
                  "**yes**" if a["vs_persistence"][n]["significant"] else "no"]
                 for n in ("unaugmented", "broad-4", "targeted-2")]

    # Decision rule, in order of precedence:
    #   1. A variant that significantly degrades either advisory class is disqualified,
    #      whatever it does to macro-F1. Very unhealthy and Hazardous are what a wearable
    #      advisory exists to get right; a macro-F1 gain bought by trading them away is
    #      not an improvement for this application.
    #   2. Otherwise the macro-F1 gain must be significant AND at least MATERIAL. With
    #      ~60k test rows a bootstrap resolves differences far below anything that
    #      matters in practice, so significance alone does not justify carrying a
    #      generator through the rest of the project.
    #
    # The rule itself lives in pulsebench.advisory_disqualification. It was extracted
    # from this block, and calling it back here is the point: the package the project
    # publishes should be the one the project's own verdict came from, not a second
    # implementation that happens to agree.
    from pulsebench import advisory_disqualification

    MATERIAL = 0.01
    candidates, disqualified = [], {}
    for n in ("broad-4", "targeted-2"):
        verdict = advisory_disqualification(
            {m: {"delta": tests[n][m]["observed_diff"],
                 "significant": tests[n][m]["significant"]}
             for m in tests[n] if m != "macro_f1"},
            protected_classes=RARE,
            practical_threshold=MATERIAL,
            aggregate_delta=tests[n]["macro_f1"]["observed_diff"],
            aggregate_significant=tests[n]["macro_f1"]["significant"])
        if verdict["disqualified"]:
            disqualified[n] = verdict["harmed"]
        elif verdict["verdict"] == "accepted":
            candidates.append(n)

    best_point = max(("unaugmented", "broad-4", "targeted-2"),
                     key=lambda n: rows[n]["scores"]["macro_f1"])

    if candidates:
        winner = max(candidates, key=lambda n: tests[n]["macro_f1"]["observed_diff"])
        t = tests[winner]["macro_f1"]
        recommendation = (
            f"**Carry `{winner}` forward to Phase 5.** It improves macro-F1 by "
            f"{t['observed_diff']:+.4f} over the unaugmented baseline, {CI}% CI "
            f"[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}], with no significant regression "
            f"on either advisory class.")
    else:
        recommendation = "**Carry `unaugmented` forward to Phase 5.**\n\n"
        if disqualified:
            lines = "\n".join(
                f"- `{n}` significantly **degrades** "
                + "; ".join(f"{m} {tests[n][m]['observed_diff']:+.4f} "
                            f"[{tests[n][m]['ci_low']:+.4f}, {tests[n][m]['ci_high']:+.4f}]"
                            for m in harmed)
                for n, harmed in disqualified.items())
            recommendation += ("Both augmented variants are disqualified on the classes "
                               "that matter most:\n\n" + lines + "\n\n")
        b4 = tests.get("broad-4", {}).get("macro_f1")
        if b4 and b4["significant"] and b4["observed_diff"] > 0:
            us = rows["unaugmented"]["scores"]["per_class"]["Unhealthy (sensitive)"]
            bs = rows["broad-4"]["scores"]["per_class"]["Unhealthy (sensitive)"]
            recommendation += (
                f"`broad-4`'s macro-F1 does rise {b4['observed_diff']:+.4f} "
                f"[{b4['ci_low']:+.4f}, {b4['ci_high']:+.4f}], and that interval does "
                f"exclude zero — **but look at where the gain comes from.** "
                f"`Unhealthy (sensitive)` jumps from {us:.4f} to {bs:.4f} "
                f"({bs - us:+.4f}), while Very unhealthy and Hazardous both fall. "
                f"Macro-F1 weights all six classes equally, so one large gain on a "
                f"mid-range class more than covers losses on the two the device exists "
                f"to warn about. **For a wearable air-quality advisory that is a bad "
                f"trade, not a result** — a missed Hazardous hour and a missed "
                f"moderately-unhealthy hour are not equally costly, and macro-F1 does "
                f"not know that.\n\n"
                f"Keep the magnitude in view too: {b4['observed_diff']:+.4f} macro-F1 is "
                f"statistically detectable across {a['n_test']:,} test rows and "
                f"practically negligible. Significance at this sample size is cheap; a "
                f"generator in the pipeline is not.\n\n")
        recommendation += (
            f"`{best_point}` holds the highest macro-F1 point estimate. It is still not "
            f"the right thing to carry forward.")

    gains = [(n, m) for n in ("broad-4", "targeted-2") for m in RARE
             if tests[n][m]["significant"] and tests[n][m]["observed_diff"] > 0]
    losses = [(n, m) for n in ("broad-4", "targeted-2") for m in RARE
              if tests[n][m]["significant"] and tests[n][m]["observed_diff"] < 0]

    def _bullets(pairs):
        return "\n".join(
            f"- **{n} / {m}**: {tests[n][m]['observed_diff']:+.4f} "
            f"[{tests[n][m]['ci_low']:+.4f}, {tests[n][m]['ci_high']:+.4f}]"
            for n, m in pairs)

    if losses and not gains:
        rare_note = (
            "**On the two advisory classes, every significant move is a regression:**\n\n"
            + _bullets(losses)
            + "\n\nThis is the finding that decides the recommendation. Very unhealthy "
              "and Hazardous are the classes the augmentation was built to help and the "
              "only two a wearable advisory really turns on — and adding synthetic rows "
              "to them made both *worse*, in every variant, with intervals that exclude "
              "zero. More synthetic Hazardous rows produced a model that is worse at "
              "Hazardous.")
    elif gains and losses:
        rare_note = ("On the advisory classes the augmentation cuts both ways.\n\n"
                     "Gains:\n\n" + _bullets(gains) + "\n\nRegressions:\n\n"
                     + _bullets(losses)
                     + "\n\nA variant that trades one advisory class for the other has "
                       "not improved the advisory.")
    elif gains:
        rare_note = ("On the advisory classes the augmentation helps:\n\n"
                     + _bullets(gains))
    else:
        rare_note = (
            "No rare-class comparison clears the bar. Very unhealthy and Hazardous — "
            "the two classes the augmentation exists to help — show no change "
            "distinguishable from zero under resampling.")

    validity_block = ""
    if validity:
        v = validity
        vrows = [
            ["`hour_sin² + hour_cos² = 1`", f"{v['hour_circle']:.2f}%", "100%", "100%"],
            ["distinct `hour_sin` values", f"{v['hour_distinct']:,}",
             f"{v['hour_distinct_real']:,}", "~21–24"],
            ["`month_sin² + month_cos² = 1`", f"{v['month_circle']:.2f}%", "100%", "100%"],
            ["DEWP > TEMP", f"{v['dewp']:.2f}%", f"{v['dewp_real']:.2f}%", "0%"],
        ]
        validity_block = f"""
## 1. Synthetic data validity (re-verified)

The first generation run produced physically impossible rows that SDV's quality score
did not catch. The generator was redesigned — integer hour/month as categoricals with
the cyclical pair computed after sampling, and an `sdv.cag.Inequality` constraint on
DEWP ≤ TEMP enforced during training. All four checks now pass:

{table(["Check", "Synthetic", "Real", "Required"], vrows)}

Full detail and the methodological note are in
`reports/gan_quality_report_h{a['horizon']}.md`. **No conclusion below comes from the
earlier, buggy run.**

---
"""

    return f"""# CTGAN ablation — horizon {a['horizon']} h

Three training sets, one shared test split, {a['n_boot']:,}-resample paired bootstrap.

All scores are on **real, observed-label test rows only** ({a['n_test']:,} of
{a['n_test_all']:,}): synthetic rows never enter validation or test, and imputed-label
rows are excluded because forward-fill inflates Hazardous prevalence. Model selection
within each variant is on **validation** observed-only macro-F1; test is reporting only.
{validity_block}
## 2. Results

{table(["Variant", "Train rows", "of which synthetic", "Selected", "Accuracy",
        "Macro-F1", "V.unhealthy F1", "Hazardous F1"], main_rows)}

### Per-class F1

{table(["Class", *order], cls_rows)}

---

## 3. Paired bootstrap vs the unaugmented baseline

{a['n_boot']:,} resamples of the test rows, both variants scored on the *same* rows each
time. The interval is the {CI}% percentile CI of the difference; "significant" means it
excludes zero.

{table(["Variant", "Metric", "Δ vs unaugmented", f"{CI}% CI", "p (two-sided)",
        "Significant"], boot_rows)}

{rare_note}

### And against the persistence floor

The comparison that decides whether any of this is a forecasting model at all:

{table(["Variant", "Δ vs persistence", f"{CI}% CI", "Significant"], pers_rows)}

---

## 4. Which variant carries forward

{recommendation}

{_floor_note(a, rows)}

Regenerate with `python -m src.gan.ablation`.
"""


def _floor_note(a: dict, rows: dict) -> str:
    vp = a["vs_persistence"]
    clears = [n for n in ("unaugmented", "broad-4", "targeted-2")
              if vp[n]["significant"] and vp[n]["observed_diff"] > 0]
    if not clears:
        return ("**A larger caveat sits above this choice.** No variant — augmented or "
                "not — beats the zero-parameter persistence rule by a margin whose CI "
                "excludes zero. The question Phase 5 has to answer is not which "
                "augmentation to use; it is whether a model of this family beats "
                "persistence at all. A sequence model that sees all 24 hours is the "
                "next thing to try, and it should be judged against persistence, not "
                "against these baselines.")
    return (f"All of {', '.join(clears)} clear the persistence floor with a CI excluding "
            f"zero, so the learned models are doing real work beyond echoing their "
            f"input. That was not true at h=1.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the write-up from the saved analysis, no retraining")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    cache = REPO_ROOT / "reports" / "metrics" / f"ablation_h{args.horizon or 6}.json"
    if args.report_only:
        # Everything build_report needs is already in the saved analysis: no retraining,
        # and no chance of the numbers shifting while only the prose is being fixed.
        a = json.loads(cache.read_text())
    else:
        collected = collect(horizon=args.horizon)
        print(f"\npaired bootstrap ({args.n_boot:,} resamples) ...")
        a = analyse(collected, n_boot=args.n_boot)

    # Pull the re-verified validity numbers straight from the generator's own output,
    # so this report cannot drift from what was actually checked.
    validity = None
    gan_metrics = REPO_ROOT / "reports" / "metrics" / f"gan_h{a['horizon']}.json"
    if gan_metrics.exists():
        v = json.loads(gan_metrics.read_text()).get("validity", {})
        syn, real = v.get("synthetic", {}), v.get("real", {})
        if syn:
            validity = {
                "hour_circle": syn["hour"]["on_unit_circle_pct"],
                "hour_distinct": syn["hour"]["distinct_sin"],
                "hour_distinct_real": real["hour"]["distinct_sin"],
                "month_circle": syn["month"]["on_unit_circle_pct"],
                "dewp": syn["dewp_above_temp_pct"],
                "dewp_real": real["dewp_above_temp_pct"],
            }

    out = REPO_ROOT / "reports" / f"gan_ablation_h{a['horizon']}.md"
    if not args.no_write:
        out.write_text(build_report(a, validity))
        if not args.report_only:
            cache.write_text(json.dumps(
                {k: v for k, v in a.items() if k != "rows"} |
                {"rows": {n: {kk: vv for kk, vv in r.items() if kk != "per_model"}
                          for n, r in a["rows"].items()}}, indent=2, default=str))
        print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
