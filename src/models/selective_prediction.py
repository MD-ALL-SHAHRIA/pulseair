"""Phase E — what does the model get right when it says it is sure?

A conformal prediction set is a statement about confidence that a device can act on:
a two-category set is an ambiguous forecast, a five-category set is barely a forecast
at all. That suggests an obvious operating mode — answer when the set is small,
abstain when it is not — and the question is what accuracy that buys and at what
coverage cost.

This is a report-only analysis. Nothing is retrained. The saved model and the saved
Mondrian thresholds are applied to the test split exactly as Phase 6 applied them,
and the resulting predictions are partitioned by set size.

The honest framing matters here. Restricting to confident predictions is **not** a
free accuracy gain: it is a trade of coverage for reliability, and a device that
abstains on the hard cases has not solved them. Both sides are reported.

    python -m src.models.selective_prediction                # Beijing
    python -m src.models.selective_prediction --dataset bangladesh

Writes ``reports/selective_prediction_h6[_bangladesh].md`` and the matching JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
CONFIDENT_MAX_SET = 2      # "confident" = the set holds at most this many categories


def _score(y: np.ndarray, pred: np.ndarray, labels: list[str]) -> dict:
    idx = list(range(len(labels)))
    out = {"n": int(len(y)),
           "accuracy": float((y == pred).mean()) if len(y) else None,
           "macro_f1": float(f1_score(y, pred, labels=idx, average="macro",
                                      zero_division=0)) if len(y) else None}
    for i, lab in enumerate(labels):
        out[f"f1_{lab}"] = (float(f1_score(y == i, pred == i, zero_division=0))
                            if len(y) else None)
        out[f"n_{lab}"] = int((y == i).sum())
    return out


def load_beijing(cfg: dict) -> dict:
    """Rebuild Phase 6's Mondrian sets on the Beijing test split. No retraining."""
    from src.models import conformal as cf

    ccfg = cf.load_config()
    bundle = joblib.load(ccfg.artifact_path)
    # The bundle stores the base model by reference, not the fitted estimator, so the
    # Phase 3 forest is loaded and attached exactly as conformal.run does.
    if "model" not in bundle:
        bundle = {**bundle, "model": joblib.load(ccfg.model_path)["model"]}
    probs, y, observed, _ = cf.load_probs(ccfg, "test", bundle)
    thresholds = {int(k): float(v) for k, v in bundle["mondrian_thresholds"].items()}
    sets = cf.prediction_sets_mondrian(probs, thresholds)
    return {"probs": probs[observed], "y": y[observed], "sets": sets[observed],
            "labels": list(ccfg.labels),
            "source": "conformal_h6.pkl Mondrian thresholds + baseline_h6.pkl",
            "model": str(bundle.get("base_model", "RandomForest"))}


def load_bangladesh(cfg: dict) -> dict:
    """The deployed Bangladesh predictor and its committed Mondrian thresholds."""
    from src.preprocessing import bangladesh as bd

    bcfg = bd.load_config()
    proc = REPO_ROOT / "data" / "processed" / "bd_h6"
    meta = json.loads((proc / "metadata.json").read_text())
    features = list(meta["feature_columns"])
    test = pd.read_csv(proc / "tabular_test.csv")
    saved = joblib.load(REPO_ROOT / "src" / "models" / "artifacts"
                        / "bangladesh_rf_h6.pkl")
    model = saved["model"] if isinstance(saved, dict) else saved
    if isinstance(saved, dict) and saved.get("feature_columns"):
        features = list(saved["feature_columns"])
    thr = json.loads((REPO_ROOT / "reports" / "deployment"
                      / "conformal_thresholds_h6_bd.json").read_text())
    # The deployment artifact keys thresholds by class NAME, the Phase 6 bundle by
    # index. Accept either rather than assuming one.
    names = list(meta["class_labels"])
    thresholds = {(names.index(k) if isinstance(k, str) and not k.isdigit() else int(k)):
                  float(v) for k, v in thr["thresholds"].items()}

    probs = model.predict_proba(test[features].to_numpy(dtype=np.float32))
    y = test["y_category"].to_numpy()
    observed = ~test["is_imputed_pm25"].to_numpy().astype(bool)
    sets = np.zeros_like(probs, dtype=bool)
    for k, t in thresholds.items():
        sets[:, k] = probs[:, k] >= (1.0 - t)
    return {"probs": probs[observed], "y": y[observed], "sets": sets[observed],
            "labels": list(meta["class_labels"]),
            "source": "bangladesh_rf_h6.pkl + conformal_thresholds_h6_bd.json",
            "model": "RandomForest (class_weight=balanced)"}


def analyse(data: dict, max_set: int = CONFIDENT_MAX_SET) -> dict:
    labels, y, sets, probs = data["labels"], data["y"], data["sets"], data["probs"]
    sizes = sets.sum(axis=1)
    pred = probs.argmax(axis=1)          # the point prediction the device would show
    covered = sets[np.arange(len(y)), y]

    confident = sizes <= max_set
    frac = float(confident.mean())

    by_size = {}
    for s in sorted(set(int(v) for v in sizes)):
        m = sizes == s
        by_size[str(s)] = {**_score(y[m], pred[m], labels),
                           "share": float(m.mean()),
                           "coverage": float(covered[m].mean())}

    return {
        "max_set_for_confident": max_set,
        "n_total": int(len(y)),
        "confident_fraction": frac,
        "abstain_fraction": float(1 - frac),
        "full": {**_score(y, pred, labels), "coverage": float(covered.mean())},
        "confident": {**_score(y[confident], pred[confident], labels),
                      "coverage": float(covered[confident].mean()) if confident.any() else None},
        "abstained": {**_score(y[~confident], pred[~confident], labels),
                      "coverage": float(covered[~confident].mean()) if (~confident).any() else None},
        "by_set_size": by_size,
        "source": data["source"], "model": data["model"], "labels": labels,
    }


def _tbl(h: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(h) + " |",
                      "| " + " | ".join(["---"] * len(h)) + " |"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _f(x, spec=".4f"):
    return "n/a" if x is None else format(x, spec)


def build_report(a: dict, place: str) -> str:
    labels = a["labels"]
    k = a["max_set_for_confident"]
    head = ["Subset", "Share of predictions", "n", "Accuracy", "Macro-F1"] + \
           [f"{l} F1" for l in ("Very unhealthy", "Hazardous") if l in labels] + \
           ["Coverage"]

    def row(name, blk, share):
        cells = [name, share, f"{blk['n']:,}", _f(blk["accuracy"]), _f(blk["macro_f1"])]
        for l in ("Very unhealthy", "Hazardous"):
            if l in labels:
                cells.append(_f(blk.get(f"f1_{l}")))
        cells.append(_f(blk.get("coverage")))
        return cells

    main = _tbl(head, [
        row("**All predictions**", a["full"], "100%"),
        row(f"Confident (set ≤ {k})", a["confident"], f"{a['confident_fraction']:.1%}"),
        row(f"Abstained (set > {k})", a["abstained"], f"{a['abstain_fraction']:.1%}"),
    ])

    size_rows = []
    for s, blk in sorted(a["by_set_size"].items(), key=lambda kv: int(kv[0])):
        size_rows.append([s, f"{blk['share']:.1%}", f"{blk['n']:,}",
                          _f(blk["accuracy"]), _f(blk["macro_f1"]),
                          _f(blk.get("coverage"))])
    by_size = _tbl(["Set size", "Share", "n", "Accuracy", "Macro-F1", "Coverage"],
                   size_rows)

    d_acc = ((a["confident"]["accuracy"] or 0) - (a["full"]["accuracy"] or 0))
    d_f1 = ((a["confident"]["macro_f1"] or 0) - (a["full"]["macro_f1"] or 0))
    haz_full = a["full"].get("f1_Hazardous")
    haz_conf = a["confident"].get("f1_Hazardous")
    haz_n_conf = a["confident"].get("n_Hazardous", 0)

    haz_note = (
        f"**The Hazardous class is where the trade bites.** Restricting to confident "
        f"predictions leaves {haz_n_conf:,} Hazardous samples of "
        f"{a['full'].get('n_Hazardous', 0):,}, and Hazardous F1 moves from "
        f"{_f(haz_full)} to {_f(haz_conf)}. A device that abstains on the readings it "
        f"finds hard is abstaining disproportionately on the readings that matter, "
        f"which is the opposite of what an advisory is for."
        if haz_full is not None and haz_conf is not None else
        f"**Hazardous cannot be assessed on this subset**: it has "
        f"{a['full'].get('n_Hazardous', 0)} samples in the full test split.")

    return f"""# Selective prediction — {place}, horizon 6 h

Phase 6 turned point predictions into conformal prediction **sets**. A set with one or
two categories is a usable forecast; a set with four is barely a forecast. That
suggests an operating mode a device could actually adopt — answer when the set is
small, say nothing when it is not — and this section measures what that buys.

**Report-only. Nothing is retrained.** The saved model and the committed Mondrian
thresholds are applied to the test split exactly as Phase 6 applied them
(`{a['source']}`), and the predictions are partitioned by set size. "Confident" means
the conformal set holds at most **{k}** of the {len(labels)} categories.

---

## 1. Confident against full

{main}

Coverage is the share of rows whose conformal set contained the true category.
Accuracy and macro-F1 are computed on the **point** prediction the device would show.

## 2. By set size

{by_size}

## 3. What this establishes

- **{a['confident_fraction']:.1%} of predictions qualify as confident** at a set size
  of {k} or fewer. The remaining {a['abstain_fraction']:.1%} would be withheld.
- On that subset accuracy moves **{d_acc:+.4f}** and macro-F1 **{d_f1:+.4f}** against
  the full test split.

{haz_note}

**This is a trade, not a gain.** Abstention improves the reported metric on the rows
that remain by removing the rows the model finds hard; it does not make those rows
easier. The relevant question for a wearable is whether a forecast withheld is better
than a forecast hedged, and the answer depends on what the wearer does with silence.
The advisory layer in Phase 6 takes the other option — it reports the ambiguity rather
than suppressing the reading — and this analysis quantifies what the alternative would
have cost.

Regenerate with `python -m src.models.selective_prediction`.
"""


def run(dataset: str = "beijing", *, write: bool = True, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    data = load_beijing(cfg) if dataset == "beijing" else load_bangladesh(cfg)
    a = analyse(data)
    a["dataset"] = dataset

    say(f"{dataset}: {a['n_total']:,} observed test rows")
    say(f"  confident (set <= {a['max_set_for_confident']}): "
        f"{a['confident_fraction']:.1%}")
    say(f"  accuracy  full {_f(a['full']['accuracy'])} -> "
        f"confident {_f(a['confident']['accuracy'])}")
    say(f"  macro-F1  full {_f(a['full']['macro_f1'])} -> "
        f"confident {_f(a['confident']['macro_f1'])}")
    say(f"  Hazardous full {_f(a['full'].get('f1_Hazardous'))} -> "
        f"confident {_f(a['confident'].get('f1_Hazardous'))}")

    if write:
        suffix = "" if dataset == "beijing" else f"_{dataset}"
        rep = REPO_ROOT / "reports" / f"selective_prediction_h6{suffix}.md"
        met = REPO_ROOT / "reports" / "metrics" / f"selective_prediction_h6{suffix}.json"
        met.parent.mkdir(parents=True, exist_ok=True)
        met.write_text(json.dumps(a, indent=2, default=str))
        rep.write_text(build_report(a, "Beijing" if dataset == "beijing"
                                    else "Bangladesh (deployed model)"))
        say(f"  wrote {rep.relative_to(REPO_ROOT)}")
        say(f"  wrote {met.relative_to(REPO_ROOT)}")
    return a


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="beijing",
                    choices=("beijing", "bangladesh", "both"))
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    targets = ("beijing", "bangladesh") if args.dataset == "both" else (args.dataset,)
    for d in targets:
        run(d, write=not args.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
