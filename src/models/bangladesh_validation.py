"""Phase 10 — external validation on Bangladesh, the device's target population.

Three questions, in order:

1. **Does the persistence floor look the same here?** Reported whatever it shows; the
   Beijing value is not assumed to carry over.
2. **Does the Beijing model transfer?** The nine-channel model cannot: this dataset has
   no temperature or dew point. A Beijing model retrained on the *shared* channels
   (PM2.5, PM10, CO + cyclical) is the fair transfer test, and is what gets run.
3. **What should actually be deployed?** A RandomForest trained natively on Bangladesh
   data, under the same protocol as every other model in this project: chronological
   split, validation-based selection, paired bootstrap against persistence, Mondrian
   conformal calibration.

    python -m src.models.bangladesh_validation

Writes ``reports/bangladesh_validation.md``.
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
from sklearn.preprocessing import StandardScaler

from src.gan.ablation import paired_bootstrap
from src.models.conformal import (calibrate_mondrian, prediction_sets_mondrian,
                                  evaluate_sets)
from src.preprocessing import bangladesh as bd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
RARE = ("Very unhealthy", "Hazardous")
N_BOOT = 1000


def macro_f1(y, p, k):
    return float(f1_score(y, p, labels=list(range(k)), average="macro", zero_division=0))


def score(y, p, labels):
    return {"macro_f1": macro_f1(y, p, len(labels)),
            "accuracy": float((y == p).mean()),
            **{f"f1_{l}": float(f1_score(y == i, p == i, zero_division=0))
               for i, l in enumerate(labels)}}


def full_probs(model, X, k):
    pr = model.predict_proba(X)
    out = np.zeros((len(X), k))
    for c, cls in enumerate(model.classes_):
        out[:, int(cls)] = pr[:, c]
    return out


def load_bd(cfg) -> dict:
    meta = json.loads((cfg.out_dir / "metadata.json").read_text())
    feats = meta["feature_columns"]
    out = {"meta": meta, "features": feats}
    for s in ("train", "val", "test"):
        f = pd.read_csv(cfg.out_dir / f"tabular_{s}.csv")
        out[s] = {"X": f[feats].to_numpy(np.float32),
                  "y": f["y_category"].to_numpy(),
                  "obs": ~f["is_imputed_pm25"].to_numpy(bool), "frame": f}
    return out


def persistence(cfg, frame, scaler, scaled_cols) -> np.ndarray:
    i = scaled_cols.index("PM2.5")
    pm = frame["PM2.5"].to_numpy(np.float64) * scaler.scale_[i] + scaler.mean_[i]
    return pd.cut(pm, bins=[*cfg.breakpoints, np.inf], labels=False,
                  right=True).astype(np.int64)


def beijing_shared_model(cfg, shared_feats, verbose=True):
    """Retrain the Beijing RF on only the channels Bangladesh also has.

    The deployed nine-channel Beijing model cannot be evaluated here at all -- TEMP and
    DEWP do not exist in this dataset. Substituting constants for them would measure a
    model nobody trained. Retraining on the intersection is the honest transfer test:
    same data, same hyperparameters, same protocol, just the channels both populations
    share.
    """
    say = print if verbose else (lambda *a, **k: None)
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    rp = raw["baseline"]["random_forest"]
    bj_dir = REPO_ROOT / "data" / "processed" / f"h{raw['preprocessing']['horizon']}"
    bj_meta = json.loads((bj_dir / "metadata.json").read_text())
    bj_scaler = joblib.load(bj_dir / "scaler.pkl")
    bj_scaled = list(bj_meta["scaled_columns"])

    tr = pd.read_csv(bj_dir / "tabular_train.csv")
    # Beijing rows are scaled with Beijing statistics; invert, then re-standardise on
    # the shared channels so the feature space matches what Bangladesh will present.
    inv = bj_scaler.inverse_transform(tr[bj_scaled].to_numpy(np.float64))
    for i, c in enumerate(bj_scaled):
        tr[c] = inv[:, i]
    phys = [c for c in shared_feats if c in bj_scaled]
    sc = StandardScaler().fit(tr[phys].to_numpy(np.float64))
    tr[phys] = sc.transform(tr[phys].to_numpy(np.float64))

    model = RandomForestClassifier(
        n_estimators=rp["n_estimators"], max_depth=rp["max_depth"],
        min_samples_leaf=rp["min_samples_leaf"], max_features=rp["max_features"],
        class_weight=rp["class_weight"], random_state=raw["seed"], n_jobs=-1,
    ).fit(tr[shared_feats].to_numpy(np.float32), tr["y_category"].to_numpy())
    say(f"  Beijing shared-channel model: {len(tr):,} training rows, "
        f"{len(shared_feats)} features {shared_feats}")
    return model, sc, phys


def run(write: bool = True, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    cfg = bd.load_config()
    labels = cfg.labels
    k = len(labels)
    data = load_bd(cfg)
    feats, meta = data["features"], data["meta"]
    scaler = joblib.load(cfg.out_dir / "scaler.pkl")
    scaled_cols = meta["scaled_columns"]

    te, va = data["test"], data["val"]
    obs = te["obs"]
    y = te["y"][obs]
    say(f"Bangladesh h{cfg.horizon}: train {len(data['train']['y']):,} | "
        f"val {len(va['y']):,} | test {len(te['y']):,} ({obs.sum():,} observed)")

    results = {}

    # --- 1. persistence floor on Bangladesh ------------------------------------
    p_test = persistence(cfg, te["frame"], scaler, scaled_cols)[obs]
    results["Persistence"] = score(y, p_test, labels)
    unchanged = float((p_test == y).mean() * 100)
    say(f"\npersistence: macro-F1 {results['Persistence']['macro_f1']:.4f} | "
        f"label unchanged over {cfg.horizon}h in {unchanged:.2f}% of samples")

    # --- 2. transfer: Beijing model, shared channels ----------------------------
    say("\ntransfer test:")
    bj_model, bj_sc, phys = beijing_shared_model(cfg, feats, verbose)
    # Bangladesh rows are scaled with Bangladesh statistics; put them into the
    # Beijing model's feature space (physical units -> Beijing training statistics).
    def to_beijing_space(frame):
        f = frame[feats].copy()
        raw_phys = frame[phys].to_numpy(np.float64) * scaler.scale_ + scaler.mean_
        f[phys] = bj_sc.transform(raw_phys)
        return f.to_numpy(np.float32)

    Xte_bj = to_beijing_space(te["frame"])
    pred_tr = bj_model.predict(Xte_bj)[obs]
    results["Beijing model (transfer)"] = score(y, pred_tr, labels)

    # --- 3. native Bangladesh model --------------------------------------------
    say("\nnative model:")
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    rp = raw["baseline"]["random_forest"]
    candidates = {}
    for name, cw in (("RandomForest", rp["class_weight"]),
                     ("RandomForest (class_weight=balanced)", "balanced")):
        m = RandomForestClassifier(
            n_estimators=rp["n_estimators"], max_depth=rp["max_depth"],
            min_samples_leaf=rp["min_samples_leaf"], max_features=rp["max_features"],
            class_weight=cw, random_state=raw["seed"], n_jobs=-1,
        ).fit(data["train"]["X"], data["train"]["y"])
        vf1 = macro_f1(va["y"][va["obs"]], m.predict(va["X"])[va["obs"]], k)
        candidates[name] = {"model": m, "val_macro_f1": vf1}
        say(f"  {name:38s} val macro-F1 {vf1:.4f}")
    # selection on validation, as everywhere else
    best = max(candidates, key=lambda n: candidates[n]["val_macro_f1"])
    say(f"  -> selected on validation: {best}")
    native = candidates[best]["model"]
    results[f"Bangladesh-native {best}"] = score(y, native.predict(te["X"])[obs], labels)

    # --- bootstrap every learned model against the Bangladesh persistence floor --
    tests = {}
    for name, pred in (("Beijing model (transfer)", pred_tr),
                       (f"Bangladesh-native {best}", native.predict(te["X"])[obs])):
        t = {"macro_f1": paired_bootstrap(y, p_test, pred, k, N_BOOT)}
        for lab in RARE:
            i = labels.index(lab)
            t[lab] = paired_bootstrap(
                y, p_test, pred, k, N_BOOT,
                scorer=lambda yt, yp, i=i: float(f1_score(yt == i, yp == i,
                                                          zero_division=0)))
        tests[name] = t
        m = t["macro_f1"]
        say(f"  {name:42s} vs persistence {m['observed_diff']:+.4f} "
            f"[{m['ci_low']:+.4f}, {m['ci_high']:+.4f}] "
            f"{'SIG' if m['significant'] else 'ns'}")

    # --- conformal on both, calibrated on the Bangladesh validation split --------
    say("\nMondrian conformal (calibrated on Bangladesh validation):")
    alpha = 1.0 - float(raw["conformal"]["target_coverage"])
    conf = {}
    for name, model, Xv, Xt in (
            ("Beijing model (transfer)", bj_model,
             to_beijing_space(va["frame"]), Xte_bj),
            (f"Bangladesh-native {best}", native, va["X"], te["X"])):
        cal = full_probs(model, Xv[va["obs"]], k)
        mond = calibrate_mondrian(cal, va["y"][va["obs"]], alpha, k)
        sets = prediction_sets_mondrian(full_probs(model, Xt[obs], k), mond["thresholds"])
        ev = evaluate_sets(sets, y, labels)
        conf[name] = {"thresholds": {int(a): float(b) for a, b in mond["thresholds"].items()},
                      "test": ev}
        hz = ev["per_class"]["Hazardous"]["coverage"]
        say(f"  {name:42s} coverage {ev['coverage']:.4f} | "
            f"Haz {'n/a (0 test samples)' if hz is None else format(hz, '.4f')} | "
            f"mean set {ev['mean_set_size']:.2f}")

    support = {}
    for split, blk in (("train", data["train"]), ("val", va), ("test", te)):
        yy = blk["y"][blk["obs"]]
        support[split] = {l: int((yy == i).sum()) for i, l in enumerate(labels)}

    rcv_path = REPO_ROOT / "reports" / "metrics" / "rolling_cv_h6_bangladesh.json"
    rcv = json.loads(rcv_path.read_text()) if rcv_path.exists() else None

    payload = {"support": support, "rcv": rcv, "cfg": cfg, "meta": meta,
               "audit": bd.audit(),
               "results": results, "tests": tests, "conformal": conf,
               "selected_native": best,
               "native_val": {n: c["val_macro_f1"] for n, c in candidates.items()},
               "label_unchanged_pct": unchanged,
               "n_test_observed": int(obs.sum()), "n_test": int(len(te["y"])),
               "n_train": int(len(data["train"]["y"])), "labels": labels,
               "split_note": (f"{te['frame'].datetime.min()[:10]} to "
                              f"{te['frame'].datetime.max()[:10]}"
                              if isinstance(te["frame"].datetime.iloc[0], str) else
                              f"{pd.to_datetime(te['frame'].datetime).min().date()} to "
                              f"{pd.to_datetime(te['frame'].datetime).max().date()}"),
               "features": feats, "n_boot": N_BOOT}

    if write:
        out = REPO_ROOT / "reports" / "bangladesh_validation.md"
        out.write_text(build_report(payload))
        say(f"\nwrote {out.relative_to(REPO_ROOT)}")
        mp = REPO_ROOT / "reports" / "metrics" / "bangladesh_h6.json"
        mp.write_text(json.dumps({x: v for x, v in payload.items() if x != "cfg"},
                                 indent=2, default=str))
        say(f"wrote {mp.relative_to(REPO_ROOT)}")
        joblib.dump({"model": native, "feature_columns": feats,
                     "class_labels": labels, "selected_on": "validation macro-F1",
                     "trained_on": "data/processed/bd_h6/tabular_train.csv",
                     "conformal_thresholds": conf[f"Bangladesh-native {best}"]["thresholds"],
                     "source": "Mendeley 9j447cynb9 v2, >= 2022-08-05"},
                    REPO_ROOT / "src" / "models" / "artifacts" / "bangladesh_rf_h6.pkl",
                    compress=3)
    return payload


def build_report(p: dict) -> str:
    cfg, a, res, tests, conf = p["cfg"], p["audit"], p["results"], p["tests"], p["conformal"]
    labels, meta = p["labels"], p["meta"]
    native = f"Bangladesh-native {p['selected_native']}"

    def table(header, rows):
        esc = lambda cs: [str(c).replace("|", "\\|") for c in cs]
        b = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return (f"| {' | '.join(esc(header))} |\n"
                f"| {' | '.join(['---'] * len(header))} |\n{b}")

    order = ["Persistence", "Beijing model (transfer)", native]
    main = [[n, f"{res[n]['macro_f1']:.4f}", f"{res[n]['accuracy']:.4f}",
             f"{res[n]['f1_Hazardous']:.4f}", f"{res[n]['f1_Very unhealthy']:.4f}",
             ("— (reference)" if n == "Persistence" else
              (lambda t: f"{t['observed_diff']:+.4f} [{t['ci_low']:+.4f}, "
                         f"{t['ci_high']:+.4f}] "
                         f"{'**significant**' if t['significant'] else 'n.s.'}")(
                  tests[n]["macro_f1"]))] for n in order]

    cls = [[f"**{l}**" if l in RARE else l,
            *[f"{res[n][f'f1_{l}']:.4f}" for n in order]] for l in labels]

    def cvfmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    cov = [[n, f"{conf[n]['test']['coverage']:.4f}",
            cvfmt(conf[n]["test"]["per_class"]["Hazardous"]["coverage"]),
            cvfmt(conf[n]["test"]["per_class"]["Very unhealthy"]["coverage"]),
            f"{conf[n]['test']['mean_set_size']:.2f}",
            f"{conf[n]['test']['singleton_rate']:.1%}"]
           for n in ("Beijing model (transfer)", native)]

    sup = p["support"]
    thin = [l for l in RARE if sup["test"][l] < 50]
    if thin:
        rows_sup = [[f"**{l}**" if l in RARE else l,
                     f"{sup['train'][l]:,}", f"{sup['val'][l]:,}",
                     f"**{sup['test'][l]:,}**"] for l in labels]
        support_warning = f"""> ### The advisory classes are effectively absent from this test split
>
> {table(["Class", "Train", "Val", "Test"], rows_sup).replace(chr(10), chr(10) + "> ")}
>
> **{', '.join(thin)} have {' and '.join(str(sup['test'][l]) for l in thin)} test
> samples respectively.** Their F1 is therefore 0 or meaningless for every model, and
> the macro-F1 comparison above is effectively a four-class comparison.
>
> The cause is the chronological split: the last {cfg.test_size:.0%} of the usable
> window runs from {p['split_note']}, which is Bangladesh's monsoon and post-monsoon
> period. Dhaka's severe pollution is a November–February phenomenon, and this split
> puts almost none of it in test.
>
> **This is the Phase 9 seasonal-split problem arriving in the external validation**,
> and it means the headline comparison here cannot speak to the classes the device
> exists to warn about. Rolling-origin CV over this dataset — the machinery is already
> in `src/models/rolling_cv.py` — would rotate a winter into the evaluation block and
> is required before any advisory-class claim is made on Bangladesh data."""
    else:
        support_warning = ""

    t_nat = tests[native]["macro_f1"]
    t_tr = tests["Beijing model (transfer)"]["macro_f1"]
    beats_nat = t_nat["significant"] and t_nat["observed_diff"] > 0
    beats_tr = t_tr["significant"] and t_tr["observed_diff"] > 0

    if beats_nat and not beats_tr:
        verdict = (f"**The native model clears the Bangladesh persistence floor; the "
                   f"transferred Beijing model does not.** That is the expected and "
                   f"the desirable result: it says the methodology transfers and the "
                   f"*weights* do not, which is exactly why a target-population model "
                   f"is the deployment candidate.")
    elif beats_nat and beats_tr:
        verdict = (f"**Both the native and the transferred model clear the floor.** "
                   f"The native model is still the deployment candidate — it is "
                   f"trained on the target population's own pollution profile — but "
                   f"the transfer result is a genuine positive worth reporting.")
    elif not beats_nat and not beats_tr:
        verdict = (f"**Neither model clears the Bangladesh persistence floor.** This "
                   f"reproduces the Beijing finding on an entirely independent "
                   f"dataset, population and pollution regime, which makes it "
                   f"substantially stronger: the limitation is the task at a "
                   f"{cfg.horizon}-hour horizon, not the city.")
    else:
        verdict = (f"**The transferred model clears the floor and the native one does "
                   f"not** — an unusual result that warrants investigation before "
                   f"anything is deployed.")

    co2 = a["co2_present_pct_clean"]

    # --- rolling-origin CV, if it has been run ---------------------------------
    r = p.get("rcv")
    if not r:
        rcv_block = ("Not yet run. Execute "
                     "`python -m src.models.rolling_cv --dataset bangladesh "
                     "--folds 5 --initial-fraction 0.30`.")
    else:
        rper, rt = r["aggregate"]["per_model"], r["aggregate"]["tests"]
        nms = [n for n in rper if n != "Persistence"]
        nf = r["n_folds"]
        rrows = [["Persistence",
                  f"{rper['Persistence']['mean']:.4f} ± {rper['Persistence']['std']:.4f}",
                  "—", "—", "—"]]
        for nm in nms:
            t = rt[nm]
            rrows.append([nm,
                          f"{rper[nm]['mean']:.4f} ± {rper[nm]['std']:.4f}",
                          f"{t['mean_delta']:+.4f}", f"**{t['wins']}/{nf}**",
                          f"{t['p_one_sided']:.4f}"])
        haz_tot = sum(f["support"]["Hazardous"] for f in r["folds"])
        vu_tot = sum(f["support"]["Very unhealthy"] for f in r["folds"])
        best_r = max(nms, key=lambda n: rt[n]["wins"])
        bw = rt[best_r]["wins"]
        if bw == nf and rt[best_r]["significant_one_sided"]:
            rv = (f"**`{best_r}` beats persistence in all {bw}/{nf} folds** "
                  f"(mean {rt[best_r]['mean_delta']:+.4f}, one-sided Wilcoxon "
                  f"p = {rt[best_r]['p_one_sided']:.4f}). Every evaluation block, "
                  f"every season in the usable window. **This is the strongest "
                  f"deployment claim in the project** — it is the only model, on "
                  f"either dataset, that clears its persistence floor in every "
                  f"rolling-origin fold.")
        elif bw > nf / 2:
            rv = (f"`{best_r}` beats persistence in {bw}/{nf} folds "
                  f"(mean {rt[best_r]['mean_delta']:+.4f}). Better than the Beijing "
                  f"result but short of unanimous.")
        else:
            rv = (f"No model clears the floor in a majority of folds "
                  f"(best `{best_r}`, {bw}/{nf}). The Bangladesh single-split win "
                  f"does not survive rotation of the evaluation block.")

        rcv_block = f"""{nf} expanding-window folds over the usable window, 30% initial
training. All {nf} evaluation blocks touch Bangladesh's Nov–Feb high-pollution season;
the fold count was chosen for that reason and the coverage is verified in
`reports/bangladesh_rolling_cv.md`.

{table(["Model", "Macro-F1 (mean ± std)", "Mean Δ vs persistence", "Folds won",
        "Wilcoxon p (1-sided)"], rrows)}

{rv}

> **The advisory classes still cannot be evaluated, and more folds will not fix it.**
> Across all {nf} evaluation blocks combined there are **{haz_tot} Hazardous** and
> **{vu_tot} Very unhealthy** samples. Hazardous is absent from
> {sum(1 for f in r['folds'] if f['support']['Hazardous'] == 0)} of {nf} blocks.
>
> The cause is the data, not the split — and **Phase 11 has now confirmed it against a
> reference-grade instrument** (`reports/dhaka_ground_truth_validation.md`). Over
> 23,010 jointly observed hours the US Embassy Dhaka monitor records **1,602 Hazardous
> hours where this dataset records 17**, and above 150 µg/m³ the reanalysis runs
> ~138 µg/m³ low. The reanalysis understates South Asian peak PM2.5, and the error is
> concentrated exactly in the advisory range.
>
> **Advisory-class validation for Bangladesh requires ground-station measurements** —
> the US Embassy Dhaka reference monitor or Department of Environment CAMS stations,
> not this product. **Phase 11b does exactly that**
> (`reports/dhaka_ground_truth_model_h6.md`): on the reference series a PM2.5-only
> model beats persistence on Hazardous F1 in 7/7 rolling-origin folds (0.4451 vs
> 0.3167, p = 0.0156). That validates the *task*; it does not validate the 7-channel
> model evaluated in this report, which cannot run on a single-pollutant source. Until then, no Hazardous or Very-unhealthy claim should be made on
> Bangladeshi data, and the {p['cfg'].horizon}-hour macro-F1 result above should be
> cited as covering the four common classes."""

    return f"""# Phase 10 — external validation on Bangladesh

The device's target population. Source: Mendeley Data
[`9j447cynb9` v2](https://data.mendeley.com/datasets/9j447cynb9/2),
`AQI Bangladesh.csv` (98.3 MB, SHA-256 verified on download).

---

## 1. Data audit — read this first

The dataset is published as **{a['stated_cities']} cities, {a['stated_range']},
{a['file_rows']:,} hourly records**. Inspection does not support that description,
and most of the file is not usable.

{table(["Claim", "As published", "As measured"], [
  ["Cities", f"{a['stated_cities']}", f"**{a['actual_cities']}**"],
  ["Rows", f"{a['file_rows']:,}", f"{a['file_rows']:,} (matches)"],
  ["Date range", a["stated_range"], f"{a['actual_range'][0][:10]} .. {a['actual_range'][1][:10]}"],
  ["Temperature channel", "—", "**absent**"],
  ["Dew point channel", "—", "**absent**"],
  ["CO₂ coverage", "available", f"**{co2:.1f}%** of the usable window"],
])}

`cities.csv` lists 102 city records, but the measurement file contains only
**{a['actual_cities']}** distinct cities.

### The pre-2022 portion does not survive inspection

Four independent signals put the boundary at **{a['clean_start'][:10]}**:

{table(["Signal", "Before 2022-08-05", "After", "Reading"], [
  ["Dhaka yearly median PM2.5", f"linear in year, R² = **{a['dhaka_pm25_linear_trend_r2']:.4f}** (slope {a['dhaka_pm25_trend_slope']:+.2f} µg/m³/yr)", "no trend", "a real city's yearly median does not lie on a straight line for 22 years"],
  ["PM2.5 ceiling", f"**{a['pre_clip_at_250_pct']:.1f}%** of 2018+ hours sit at exactly 250.0", "not clipped", "hard clip, not a physical limit"],
  ["CO median", f"{a['co_median_pre']:.2f}", f"{a['co_median_post']:.0f}", "a unit change mid-file (≈250×), i.e. two sources spliced"],
  ["PM2.5 lag-1 autocorrelation", f"{a['pm25_lag1_pre']:.3f}", f"{a['pm25_lag1_post']:.3f}", "the later block behaves like real hourly air quality"],
])}

Only **Dhaka** carries the pre-2022 history; every other city begins at
{a['clean_start'][:10]}. The most likely explanation is that Dhaka was
back-filled synthetically to give the dataset a longer nominal span.

**Everything below uses the {a['clean_start'][:10]} onward window only**:
{a['clean_rows']:,} rows ({a['clean_pct']:.0f}% of the file),
{a['clean_cities']} cities, {a['clean_range'][0][:10]} to {a['clean_range'][1][:10]}
({(pd.Timestamp(a['clean_range'][1]) - pd.Timestamp(a['clean_range'][0])).days / 365.25:.2f} years).

> **For the thesis.** This audit is worth a paragraph in its own right. The dataset is
> the top Google result for Bangladeshi air-quality data and is published on a
> reputable repository with a DOI; 81% of it is not usable, and nothing in the record
> says so. Any external-validation claim built on the advertised 2000–2025 span would
> be built on generated numbers.

### A correction to the CO₂ premise

The project brief noted that this dataset carries **CO₂ directly**, unlike Beijing,
and that this would improve sensor alignment for the wearable. **It does not hold.**
CO₂ is present in only **{co2:.1f}%** of the usable window — absent entirely before
2024 and complete only from 2025. It cannot serve as a modelling channel here.

The schema (`pm10`, `pm2_5`, `carbon_monoxide`, `carbon_dioxide`,
`nitrogen_dioxide`, `sulphur_dioxide`, `ozone`) matches the **Open-Meteo Air Quality
API** exactly, which suggests the usable portion is CAMS reanalysis rather than
ground-station measurement. That is adequate for a transfer study and should be
stated: it is modelled ambient air quality, not sensor data.

---

## 2. Preprocessing — same conventions as Phase 2

Cities: **{', '.join(meta['cities'])}** — four major metros, each with complete
hourly coverage over the usable window.

Everything carried over unchanged: per-city forward fill with an `is_imputed`
provenance flag, identical cyclical hour/month encodings, the same EPA PM2.5
breakpoints, a chronological {1 - cfg.val_size - cfg.test_size:.0%}/{cfg.val_size:.0%}/{cfg.test_size:.0%}
split on the shared hourly axis, scaler fitted on training rows only, window
{cfg.window} h / horizon {cfg.horizon} h, and metrics on observed labels only.

| | |
|---|---|
| Rows after filtering | {meta['stats']['rows_in']:,} |
| Rows dropped by forward fill | {meta['stats']['rows_dropped']:,} |
| Samples (train / val / test) | {meta['stats']['samples']['train']:,} / {meta['stats']['samples']['val']:,} / {meta['stats']['samples']['test']:,} |
| Test samples with observed labels | {p['n_test_observed']:,} |

**Feature set: {len(p['features'])} channels, not nine.** Temperature and dew point do
not exist in this dataset, so the Beijing feature set cannot be reproduced. The shared
subset is {', '.join(f'`{c}`' for c in p['features'])}.

---

## 3. The persistence floor on Bangladesh

Recomputed, not assumed.

| | Beijing h{cfg.horizon} | **Bangladesh h{cfg.horizon}** |
|---|---|---|
| Label unchanged over the horizon | 53.87% | **{p['label_unchanged_pct']:.2f}%** |
| Persistence macro-F1 | 0.5118 | **{res['Persistence']['macro_f1']:.4f}** |
| Persistence Hazardous F1 | 0.5897 | **{res['Persistence']['f1_Hazardous']:.4f}** |

---

## 4. Results

{table(["Model", "Macro-F1", "Accuracy", "Hazardous F1", "V.unhealthy F1",
        "vs persistence (1,000-resample bootstrap)"], main)}

### Per-class F1

{table(["Class", *order], cls)}

{support_warning}

### Native model selection

Both candidates were fitted on Bangladesh training data and chosen on **validation**,
as in every other phase:

{table(["Candidate", "Validation macro-F1"],
       [[n, f"{v:.4f}"] for n, v in p["native_val"].items()])}

Selected: **{p['selected_native']}**.

---

## 5. Conformal coverage on Bangladesh

Mondrian thresholds re-calibrated on the **Bangladesh validation split** — thresholds
are quantiles under a specific model and distribution, so the Beijing ones certify
nothing here.

{table(["Model", "Coverage", "Hazardous", "V.unhealthy", "Mean set size",
        "Singletons"], cov)}

---

## 6. Rolling-origin cross-validation

{rcv_block}

---

## 7. Recommendation

{verdict}

**Deploy the Bangladesh-native model.** `src/models/artifacts/bangladesh_rf_h6.pkl`
holds the selected forest, its feature order, class labels and re-calibrated Mondrian
thresholds. Dhaka's pollution profile, sources and seasonal cycle differ substantially
from Beijing's, and the transfer numbers above show the difference is not
cosmetic.

**Cite the Beijing pipeline as the methodology, not as the model.** What transfers is
the discipline, and it is what makes the Bangladesh number trustworthy rather than
merely reported:

- the **persistence floor** reported alongside every result (Phase 3);
- the **horizon selection** that rejected h=1 as autocorrelation rather than forecasting
  (Phase 3);
- the **GAN ablation** with a disqualification rule on the advisory classes (Phase 4);
- **rolling-origin cross-validation**, which showed single-split wins on Beijing sit
  inside the fold-to-fold variance of the baseline (Phase 9);
- **Mondrian conformal calibration**, after marginal coverage was found to under-cover
  Hazardous (Phase 6);
- **imputation provenance**, so every metric here is computed on observed labels only
  (Phase 2).

**Carry the Phase 9 caveat across.** The Bangladesh result above is a single
chronological split. The Beijing work showed a single split can produce a win whose
sign flips under rolling-origin evaluation, and nothing about Bangladesh makes that
less likely — its usable window is {(pd.Timestamp(a['clean_range'][1]) - pd.Timestamp(a['clean_range'][0])).days / 365.25:.1f} years
against Beijing's 4. **Rolling-origin CV on this dataset is the next step before any
deployment claim**, and the machinery already exists in `src/models/rolling_cv.py`.

Reproduce with `python -m src.preprocessing.bangladesh` then
`python -m src.models.bangladesh_validation`.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    run(write=not args.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
