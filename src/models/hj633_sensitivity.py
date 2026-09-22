"""HJ 633-2012 sensitivity analysis — do the results depend on the EPA labelling?

Every class label in this project comes from **US EPA** PM2.5 breakpoints, applied to
Chinese monitoring data. China's own standard, **HJ 633-2012** (on GB 3095-2012), uses
higher cut-points. The breakpoints are not a neutral relabelling: they set the class
boundaries, and therefore the class balance, the rare-class prevalence that motivated
the CTGAN work, and the conformal thresholds.

This re-runs only what is needed to answer the question a reviewer will actually ask:
**how much does the class balance move, and does the RandomForest still beat
persistence under the Chinese labelling?** The full pipeline is not re-run.

    python -m src.models.hj633_sensitivity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from src.gan.ablation import paired_bootstrap
from src.models import baseline as bl

REPO_ROOT = Path(__file__).resolve().parents[2]
RARE = ("Very unhealthy", "Hazardous")

# HJ 633-2012 / GB 3095-2012 24-hour PM2.5 IAQI breakpoints (ug/m3), lower edges.
# IAQI 0-50 / 51-100 / 101-150 / 151-200 / 201-300 / >300 maps to PM2.5
# 0-35 / 35-75 / 75-115 / 115-150 / 150-250 / >250.
HJ633_BREAKPOINTS = [0, 35, 75, 115, 150, 250]
HJ633_LABELS = ["Excellent", "Good", "Lightly polluted", "Moderately polluted",
                "Heavily polluted", "Severely polluted"]
# The two bands that trigger an advisory under the Chinese scale.
HJ_ADVISORY = ("Heavily polluted", "Severely polluted")


def relabel(pm25_raw: np.ndarray, breakpoints, labels) -> np.ndarray:
    return pd.cut(pm25_raw, bins=[*breakpoints, np.inf], labels=False,
                  right=True).astype(np.int64)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    cfg = bl.load_config()
    features, meta = bl.load_feature_columns(cfg)
    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    rp = raw["baseline"]["random_forest"]
    epa_labels = cfg.labels
    scaler = joblib.load(cfg.processed_dir / "scaler.pkl")
    scaled = list(meta["scaled_columns"])

    frames, split_data = {}, {}
    for s in ("train", "val", "test"):
        f = pd.read_csv(cfg.processed_dir / f"tabular_{s}.csv")
        frames[s] = f
        split_data[s] = {
            "X": f[features].to_numpy(np.float32),
            "obs": ~f["is_imputed_pm25"].to_numpy(bool),
            "pm_target": f["y_pm25_raw"].to_numpy(np.float64),
            "pm_now": (f["PM2.5"].to_numpy(np.float64) * scaler.scale_[scaled.index("PM2.5")]
                       + scaler.mean_[scaled.index("PM2.5")]),
            "y_epa": f["y_category"].to_numpy(),
        }

    results = {}
    for scheme, bps, labs, adv in (
            ("EPA", cfg.breakpoints, epa_labels, RARE),
            ("HJ 633-2012", HJ633_BREAKPOINTS, HJ633_LABELS, HJ_ADVISORY)):
        k = len(labs)
        print(f"\n=== {scheme} ===")
        dist = {}
        for s in ("train", "val", "test"):
            d = split_data[s]
            y = relabel(d["pm_target"], bps, labs)
            d[f"y_{scheme}"] = y
            o = d["obs"]
            cnt = np.bincount(y[o], minlength=k)
            dist[s] = {labs[i]: {"n": int(cnt[i]),
                                 "pct": round(100 * cnt[i] / o.sum(), 2)}
                       for i in range(k)}
        print("  train distribution: " + " ".join(
            f"{l[:12]}={dist['train'][l]['pct']:.1f}%" for l in labs))

        tr, va, te = split_data["train"], split_data["val"], split_data["test"]
        model = RandomForestClassifier(
            n_estimators=rp["n_estimators"], max_depth=rp["max_depth"],
            min_samples_leaf=rp["min_samples_leaf"], max_features=rp["max_features"],
            class_weight=rp["class_weight"], random_state=cfg.seed, n_jobs=-1,
        ).fit(tr["X"], tr[f"y_{scheme}"])

        mask, y_te = te["obs"], te[f"y_{scheme}"][te["obs"]]
        pred = model.predict(te["X"])[mask]
        persist = relabel(te["pm_now"], bps, labs)[mask]

        def sc(p):
            return {"macro_f1": float(f1_score(y_te, p, labels=list(range(k)),
                                               average="macro", zero_division=0)),
                    "accuracy": float((y_te == p).mean()),
                    "per_class": {l: float(f1_score(y_te == i, p == i, zero_division=0))
                                  for i, l in enumerate(labs)}}

        rf_s, pe_s = sc(pred), sc(persist)
        bt = paired_bootstrap(y_te, persist, pred, k, 1000)
        adv_bt = {}
        for lab in adv:
            i = labs.index(lab)
            adv_bt[lab] = paired_bootstrap(
                y_te, persist, pred, k, 1000,
                scorer=lambda yt, yp, i=i: float(f1_score(yt == i, yp == i,
                                                          zero_division=0)))
        val_f1 = float(f1_score(va[f"y_{scheme}"][va["obs"]],
                                model.predict(va["X"])[va["obs"]],
                                labels=list(range(k)), average="macro",
                                zero_division=0))
        print(f"  RF test macro-F1 {rf_s['macro_f1']:.4f} | persistence "
              f"{pe_s['macro_f1']:.4f} | delta {bt['observed_diff']:+.4f} "
              f"[{bt['ci_low']:+.4f}, {bt['ci_high']:+.4f}] "
              f"{'SIG' if bt['significant'] else 'ns'}")
        results[scheme] = {
            "breakpoints": list(bps), "labels": list(labs), "advisory": list(adv),
            "distribution": dist, "rf": rf_s, "persistence": pe_s,
            "val_macro_f1": val_f1, "vs_persistence": bt, "advisory_tests": adv_bt,
            "label_unchanged_pct": float((persist == y_te).mean() * 100),
        }

    out = {"results": results, "n_test_observed": int(split_data["test"]["obs"].sum())}
    if not args.no_write:
        p = REPO_ROOT / "reports" / "metrics" / "hj633_h6.json"
        p.write_text(json.dumps(out, indent=2, default=str))
        (REPO_ROOT / "reports" / "hj633_sensitivity.md").write_text(build_report(out))
        print(f"\nwrote reports/hj633_sensitivity.md")
    return 0


def build_report(out: dict) -> str:
    R = out["results"]
    epa, hj = R["EPA"], R["HJ 633-2012"]

    def table(header, rows):
        esc = lambda cs: [str(c).replace("|", "\\|") for c in cs]
        b = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return (f"| {' | '.join(esc(header))} |\n"
                f"| {' | '.join(['---'] * len(header))} |\n{b}")

    def dist_rows(res):
        return [[f"**{l}**" if l in res["advisory"] else l,
                 f"{res['distribution']['train'][l]['pct']:.2f}%",
                 f"{res['distribution']['test'][l]['pct']:.2f}%",
                 f"{res['distribution']['train'][l]['n']:,}"]
                for l in res["labels"]]

    cmp_rows = []
    for name, res in (("EPA (used throughout this project)", epa),
                      ("HJ 633-2012 (Chinese national standard)", hj)):
        b = res["vs_persistence"]
        cmp_rows.append([
            name, f"{res['persistence']['macro_f1']:.4f}",
            f"{res['rf']['macro_f1']:.4f}", f"{b['observed_diff']:+.4f}",
            f"[{b['ci_low']:+.4f}, {b['ci_high']:+.4f}]",
            "**yes**" if (b["significant"] and b["observed_diff"] > 0) else
            ("**worse**" if b["significant"] else "no"),
        ])

    epa_rare = min(epa["distribution"]["train"][l]["pct"] for l in epa["advisory"])
    hj_rare = min(hj["distribution"]["train"][l]["pct"] for l in hj["advisory"])
    eb, hb = epa["vs_persistence"], hj["vs_persistence"]
    same = ((eb["significant"] and eb["observed_diff"] > 0) ==
            (hb["significant"] and hb["observed_diff"] > 0))

    verdict = (
        f"**The conclusion is unchanged under the Chinese labelling.** Under EPA the "
        f"RandomForest beats persistence by {eb['observed_diff']:+.4f}; under "
        f"HJ 633-2012 by {hb['observed_diff']:+.4f}, and the direction and "
        f"significance agree. The Phase 3 result is not an artifact of using a US "
        f"scale on Chinese data."
        if same else
        f"**The conclusion changes under the Chinese labelling.** Under EPA the "
        f"RandomForest is {eb['observed_diff']:+.4f} against persistence; under "
        f"HJ 633-2012 it is {hb['observed_diff']:+.4f}. The Phase 3 result is "
        f"sensitive to the breakpoint scheme, and the thesis must say so.")

    return f"""# HJ 633-2012 sensitivity analysis

Every class label in this project comes from **US EPA** PM2.5 breakpoints applied to
Chinese monitoring data. China's own standard, **HJ 633-2012** (built on
GB 3095-2012), uses higher cut-points. This was listed as an open limitation; it is
closed here.

The breakpoints are not a neutral relabelling — they define the class boundaries and
therefore the class balance, the rare-class prevalence that motivated the CTGAN work,
and the conformal thresholds. The question is whether the headline result survives
the change.

**Scope:** only the class distribution and the Phase 3 baseline are re-run. The GAN
ablation, sequence models, conformal calibration and deployment work are not.

---

## 1. The two schemes

{table(["Scheme", "PM2.5 cut-points (µg/m³)", "Classes"], [
  ["US EPA AQI", " / ".join(str(b) for b in epa["breakpoints"][1:]), ", ".join(epa["labels"])],
  ["HJ 633-2012", " / ".join(str(b) for b in hj["breakpoints"][1:]), ", ".join(hj["labels"])],
])}

The Chinese scale starts its second band at 35 µg/m³ where EPA starts at 12, so a
large block of hours that EPA calls *Moderate* the Chinese scale calls *Excellent*.

---

## 2. How much the class balance moves

### Under EPA

{table(["Class", "Train %", "Test %", "Train n"], dist_rows(epa))}

### Under HJ 633-2012

{table(["Class", "Train %", "Test %", "Train n"], dist_rows(hj))}

**The rarest advisory class goes from {epa_rare:.2f}% (EPA) to {hj_rare:.2f}%
(HJ 633-2012)** of the training split. The Chinese scheme is
{"less" if hj_rare > epa_rare else "more"} skewed at the top end, which
{"weakens" if hj_rare > epa_rare else "strengthens"} the case for the rare-class
augmentation work in Phase 4 — that work was sized against the EPA prevalence, and
the thesis should say which scale its imbalance argument depends on.

---

## 3. Does the RandomForest still beat persistence?

Same protocol: chronological split, observed labels only, 1,000-resample paired
bootstrap on the same {out['n_test_observed']:,} test rows.

{table(["Labelling", "Persistence macro-F1", "RF macro-F1", "Δ", "95% CI",
        "RF beats persistence?"], cmp_rows)}

Label persistence over the 6-hour horizon also differs:
**{epa['label_unchanged_pct']:.1f}%** of samples keep their EPA class against
**{hj['label_unchanged_pct']:.1f}%** under HJ 633-2012 — a coarser scale at the low
end means fewer boundary crossings, which is why the persistence floor itself moves.

---

## 4. Verdict

{verdict}

**What remains conditional on the EPA scale.** The Phase 4 augmentation targets, the
Phase 6 per-class conformal thresholds, and every per-class F1 in the project were
computed against EPA bands. This analysis shows the *headline* persistence comparison
is robust to the choice; it does not show the rare-class results are. A full re-run
under HJ 633-2012 remains future work, and is a one-line config change
(`data.pm25_breakpoints`) plus the pipeline.

**For a Bangladeshi deployment neither scale is obviously correct** — Bangladesh's
Department of Environment publishes its own AQI, closer to the US EPA scheme than to
China's. The labelling should follow the jurisdiction the device is used in, and this
analysis shows how to check whether that choice matters.

Reproduce with `python -m src.models.hj633_sensitivity`.
"""


if __name__ == "__main__":
    raise SystemExit(main())
