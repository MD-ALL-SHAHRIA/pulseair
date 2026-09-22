"""Phase 11b — PM2.5-only model on the US Embassy Dhaka ground truth.

Closes the evidence gap the whole project has been circling. Every Bangladesh-side
result so far covers the four common AQI classes only, because the Mendeley reanalysis
contains 2 Hazardous hours in 3.3 years. Phase 11 showed the reference instrument
records 3,788 of them. This trains and validates a model on that data.

**This does not replace the deployed model.** Two models, two jobs:

* the **7-channel Bangladesh-native model** (Phase 10) is the deployment candidate —
  four cities, PM2.5/PM10/CO + time, validated 5/5 rolling-origin folds on the four
  common classes;
* **this PM2.5-only model** is the advisory-class validation evidence — one station,
  nine years, real Hazardous support.

Neither subsumes the other: this one has the rare classes but one pollutant and one
site; that one has the channels and the spatial spread but no rare classes.

Features, given a single pollutant: PM2.5 at *t*, cyclical hour/month, and lagged
PM2.5 at t−1, t−3, t−6, t−12, t−24. Without other species, recent history is the only
signal available.

    python -m src.models.dhaka_pm25_model

Writes ``reports/dhaka_ground_truth_model_h6.md``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.gan.ablation import paired_bootstrap
from src.models.dhaka_ground_truth import load_embassy, classify

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
RARE = ("Very unhealthy", "Hazardous")

LAGS = (1, 3, 6, 12, 24)
MAX_FILL_GAP = 3          # hours; longer downtime is dropped, not interpolated
WINTER = {11, 12, 1, 2}


@dataclass(frozen=True)
class P11Config:
    labels: list[str]
    breakpoints: list[float]
    horizon: int
    val_size: float
    test_size: float
    rf: dict
    seed: int

    @property
    def train_size(self) -> float:
        return 1.0 - self.val_size - self.test_size

    @property
    def features(self) -> list[str]:
        return (["pm25", "hour_sin", "hour_cos", "month_sin", "month_cos"]
                + [f"pm25_lag{h}" for h in LAGS])


def load_config() -> P11Config:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    d, p = raw["data"], raw["preprocessing"]
    return P11Config(labels=list(d["pm25_labels"]),
                     breakpoints=list(d["pm25_breakpoints"]),
                     horizon=int(p["horizon"]),
                     val_size=float(d["val_size"]), test_size=float(d["test_size"]),
                     rf=dict(raw["baseline"]["random_forest"]), seed=int(raw["seed"]))


# --------------------------------------------------------------------- features


def build_samples(cfg: P11Config, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Lagged PM2.5 features on a gap-aware hourly grid.

    The station has ~5% instrument downtime. Short gaps are forward-filled and flagged
    with the project's ``is_imputed`` convention; **gaps longer than
    ``MAX_FILL_GAP`` hours are not filled at all**, and any sample whose lags or target
    would reach across one is dropped. Carrying a value forward across a multi-day
    outage would manufacture the very autocorrelation the persistence floor measures.
    """
    say = print if verbose else (lambda *a, **k: None)
    emb, audit = load_embassy()

    grid = pd.date_range(emb.datetime.min(), emb.datetime.max(), freq="h")
    s = emb.set_index("datetime")["pm25"].reindex(grid)
    missing = s.isna()

    # Label each missing run with its length so short ones can be filled selectively.
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size").where(missing, 0)
    fillable = missing & (run_len <= MAX_FILL_GAP)

    filled = s.copy()
    filled[fillable] = s.ffill()[fillable]
    is_imputed = fillable.copy()
    long_gap = missing & ~fillable

    df = pd.DataFrame({"pm25": filled, "is_imputed": is_imputed,
                       "long_gap": long_gap}, index=grid)
    for h in LAGS:
        df[f"pm25_lag{h}"] = df["pm25"].shift(h)
        df[f"imp_lag{h}"] = df["is_imputed"].shift(h).fillna(False)
    df["pm25_target"] = df["pm25"].shift(-cfg.horizon)
    df["imp_target"] = df["is_imputed"].shift(-cfg.horizon).fillna(False)

    theta_h = 2 * np.pi * df.index.hour / 24
    theta_m = 2 * np.pi * (df.index.month - 1) / 12
    df["hour_sin"], df["hour_cos"] = np.sin(theta_h), np.cos(theta_h)
    df["month_sin"], df["month_cos"] = np.sin(theta_m), np.cos(theta_m)

    need = ["pm25", "pm25_target"] + [f"pm25_lag{h}" for h in LAGS]
    before = int(len(df))
    ok = df[need].notna().all(axis=1)
    samples = df[ok].copy()

    samples["y"] = classify(samples["pm25_target"].to_numpy(),
                            cfg.breakpoints, cfg.labels)
    samples["y_now"] = classify(samples["pm25"].to_numpy(),
                                cfg.breakpoints, cfg.labels)
    # Provenance follows the LABEL row, as everywhere else in this project.
    samples["is_imputed_pm25"] = samples["imp_target"].astype(bool)

    stats = {
        "audit": audit,
        "grid_hours": before,
        "observed_hours": int((~missing).sum()),
        "missing_hours": int(missing.sum()),
        "short_gap_filled": int(fillable.sum()),
        "long_gap_unfilled": int(long_gap.sum()),
        "max_fill_gap_hours": MAX_FILL_GAP,
        "samples": int(len(samples)),
        "windows_dropped": before - int(len(samples)),
        "windows_dropped_pct": round(100 * (before - len(samples)) / before, 2),
        "label_imputed": int(samples["is_imputed_pm25"].sum()),
    }
    say(f"grid {before:,} h | observed {stats['observed_hours']:,} | "
        f"short gaps filled {stats['short_gap_filled']:,} | "
        f"long gaps left empty {stats['long_gap_unfilled']:,}")
    say(f"usable samples {stats['samples']:,} "
        f"({stats['windows_dropped']:,} windows dropped, "
        f"{stats['windows_dropped_pct']:.1f}%)")
    return samples.reset_index(names="datetime"), stats


def fit_rf(cfg: P11Config, X, y):
    return RandomForestClassifier(
        n_estimators=cfg.rf["n_estimators"], max_depth=cfg.rf["max_depth"],
        min_samples_leaf=cfg.rf["min_samples_leaf"],
        max_features=cfg.rf["max_features"], class_weight="balanced",
        random_state=cfg.seed, n_jobs=-1).fit(X, y)


def score(y, p, labels) -> dict:
    k = len(labels)
    return {"macro_f1": float(f1_score(y, p, labels=list(range(k)), average="macro",
                                       zero_division=0)),
            "accuracy": float((y == p).mean()),
            **{f"f1_{l}": float(f1_score(y == i, p == i, zero_division=0))
               for i, l in enumerate(labels)},
            **{f"n_{l}": int((y == i).sum()) for i, l in enumerate(labels)}}


# ------------------------------------------------------- single split + rolling CV


N_FOLDS = 7
INITIAL_FRACTION = 0.30
EMBARGO_HOURS = 24 + max(LAGS)      # widest reach of any feature window


def single_split(cfg: P11Config, s: pd.DataFrame, verbose=True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    ts = np.sort(s.datetime.unique()); n = len(ts)
    v0, t0 = ts[int(round(n * cfg.train_size))], ts[int(round(n * (cfg.train_size + cfg.val_size)))]
    part = np.where(s.datetime < v0, "train", np.where(s.datetime < t0, "val", "test"))
    s = s.assign(split=part)

    sc = StandardScaler().fit(s.loc[s.split == "train", cfg.features].to_numpy(float))
    def X(df): return sc.transform(df[cfg.features].to_numpy(float)).astype(np.float32)

    tr = s[s.split == "train"]
    va, te = s[s.split == "val"], s[s.split == "test"]
    model = fit_rf(cfg, X(tr), tr.y.to_numpy())

    vo, to = ~va.is_imputed_pm25.to_numpy(), ~te.is_imputed_pm25.to_numpy()
    val_f1 = float(f1_score(va.y.to_numpy()[vo], model.predict(X(va))[vo],
                            labels=list(range(len(cfg.labels))), average="macro",
                            zero_division=0))
    y, pred = te.y.to_numpy()[to], model.predict(X(te))[to]
    per = te.y_now.to_numpy()[to]
    k = len(cfg.labels)
    out = {"val_macro_f1": val_f1, "n_test": int(to.sum()),
           "split_dates": {"val_start": str(pd.Timestamp(v0)),
                           "test_start": str(pd.Timestamp(t0))},
           "model": score(y, pred, cfg.labels),
           "persistence": score(y, per, cfg.labels),
           "tests": {"macro_f1": paired_bootstrap(y, per, pred, k, 1000)}}
    for lab in RARE:
        i = cfg.labels.index(lab)
        out["tests"][lab] = paired_bootstrap(
            y, per, pred, k, 1000,
            scorer=lambda yt, yp, i=i: float(f1_score(yt == i, yp == i, zero_division=0)))
    say(f"single split: val {val_f1:.4f} | test macro-F1 {out['model']['macro_f1']:.4f} "
        f"vs persistence {out['persistence']['macro_f1']:.4f} "
        f"({out['tests']['macro_f1']['observed_diff']:+.4f})")
    say(f"  Hazardous F1 {out['model']['f1_Hazardous']:.4f} "
        f"(persistence {out['persistence']['f1_Hazardous']:.4f}, "
        f"n={out['model']['n_Hazardous']:,})")
    return out


def rolling_cv(cfg: P11Config, s: pd.DataFrame, n_folds=N_FOLDS,
               initial=INITIAL_FRACTION, verbose=True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    ts = np.sort(s.datetime.unique()); n = len(ts)
    edges = np.linspace(int(n * initial), n, n_folds + 1).astype(int)
    k = len(cfg.labels)
    folds = []

    for f in range(n_folds):
        cut = pd.Timestamp(ts[edges[f]])
        end = pd.Timestamp(ts[edges[f + 1] - 1])
        tr = s[s.datetime + pd.Timedelta(hours=cfg.horizon) < cut]
        ev = s[(s.datetime >= cut + pd.Timedelta(hours=EMBARGO_HOURS))
               & (s.datetime <= end)]
        obs = ~ev.is_imputed_pm25.to_numpy()
        y = ev.y.to_numpy()[obs]

        sc = StandardScaler().fit(tr[cfg.features].to_numpy(float))
        Xtr = sc.transform(tr[cfg.features].to_numpy(float)).astype(np.float32)
        Xev = sc.transform(ev[cfg.features].to_numpy(float)).astype(np.float32)
        model = fit_rf(cfg, Xtr, tr.y.to_numpy())
        pred = model.predict(Xev)[obs]
        per = ev.y_now.to_numpy()[obs]

        e = {"fold": f + 1, "cutoff": str(cut),
             "eval_start": str(ev.datetime.min()), "eval_end": str(ev.datetime.max()),
             "months": sorted({int(m) for m in ev.datetime.dt.month.unique()}),
             "n_train": int(len(tr)), "n_eval": int(obs.sum()),
             "model": score(y, pred, cfg.labels),
             "persistence": score(y, per, cfg.labels)}
        e["winter"] = bool(set(e["months"]) & WINTER)
        folds.append(e)
        say(f"  F{f+1} {e['eval_start'][:7]}..{e['eval_end'][:7]} "
            f"train {len(tr):>6,} eval {obs.sum():>6,} | "
            f"macro {e['model']['macro_f1']:.4f} vs {e['persistence']['macro_f1']:.4f} | "
            f"Haz {e['model']['f1_Hazardous']:.4f} vs "
            f"{e['persistence']['f1_Hazardous']:.4f} (n={e['model']['n_Hazardous']:,})")

    agg = {}
    for metric in ["macro_f1"] + [f"f1_{l}" for l in cfg.labels]:
        m = np.array([f["model"][metric] for f in folds])
        p = np.array([f["persistence"][metric] for f in folds])
        d = m - p
        try:
            p2 = float(wilcoxon(d).pvalue)
            p1 = float(wilcoxon(d, alternative="greater").pvalue)
        except ValueError:
            p2 = p1 = 1.0
        agg[metric] = {
            "model_mean": float(m.mean()), "model_std": float(m.std(ddof=1)),
            "persistence_mean": float(p.mean()),
            "mean_delta": float(d.mean()), "std_delta": float(d.std(ddof=1)),
            "wins": int((d > 0).sum()), "n_folds": n_folds,
            "p_two_sided": p2, "p_one_sided": p1,
            "sig_two_sided": bool(p2 < 0.05), "sig_one_sided": bool(p1 < 0.05),
            "per_fold_model": [float(x) for x in m],
            "per_fold_delta": [float(x) for x in d]}
    agg["_resolution"] = {"n_pairs": n_folds,
                          "min_p_two_sided": float(2 ** (1 - n_folds)),
                          "min_p_one_sided": float(2 ** -n_folds)}
    say(f"\naggregate over {n_folds} folds:")
    for metric in ("macro_f1", "f1_Very unhealthy", "f1_Hazardous"):
        a = agg[metric]
        say(f"  {metric:20s} {a['model_mean']:.4f}+/-{a['model_std']:.4f} vs "
            f"{a['persistence_mean']:.4f} | delta {a['mean_delta']:+.4f} | "
            f"wins {a['wins']}/{n_folds} | p2={a['p_two_sided']:.4f}")
    return {"folds": folds, "aggregate": agg, "n_folds": n_folds,
            "initial_fraction": initial, "embargo_hours": EMBARGO_HOURS}


# ------------------------------------------------------------------- report


def build_report(cfg: P11Config, stats: dict, ss: dict, cv: dict) -> str:
    a, agg, folds = stats["audit"], cv["aggregate"], cv["folds"]
    labels, res = cfg.labels, agg["_resolution"]
    nf = cv["n_folds"]

    def table(header, rows):
        esc = lambda cs: [str(x).replace("|", "\\|") for x in cs]
        b = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return (f"| {' | '.join(esc(header))} |\n"
                f"| {' | '.join(['---'] * len(header))} |\n{b}")

    fold_rows = [[f"{f['fold']}", f"{f['eval_start'][:7]}..{f['eval_end'][:7]}",
                  "yes" if f["winter"] else "no", f"{f['n_train']:,}",
                  f"{f['n_eval']:,}", f"{f['model']['n_Hazardous']:,}",
                  f"{f['model']['macro_f1']:.4f}",
                  f"{f['persistence']['macro_f1']:.4f}",
                  f"{f['model']['macro_f1'] - f['persistence']['macro_f1']:+.4f}"]
                 for f in folds]

    haz_rows = [[f"{f['fold']}", f"{f['model']['n_Hazardous']:,}",
                 f"{f['model']['f1_Hazardous']:.4f}",
                 f"{f['persistence']['f1_Hazardous']:.4f}",
                 f"{f['model']['f1_Hazardous'] - f['persistence']['f1_Hazardous']:+.4f}",
                 f"{f['model']['f1_Very unhealthy']:.4f}",
                 f"{f['persistence']['f1_Very unhealthy']:.4f}",
                 f"{f['model']['f1_Very unhealthy'] - f['persistence']['f1_Very unhealthy']:+.4f}"]
                for f in folds]

    agg_rows = []
    for metric, name in [("macro_f1", "Macro-F1")] + \
                        [(f"f1_{l}", l) for l in labels]:
        m = agg[metric]
        agg_rows.append([
            f"**{name}**" if name in RARE else name,
            f"{m['model_mean']:.4f} ± {m['model_std']:.4f}",
            f"{m['persistence_mean']:.4f}", f"{m['mean_delta']:+.4f}",
            f"**{m['wins']}/{nf}**", f"{m['p_two_sided']:.4f}",
            "**yes**" if m["sig_two_sided"] else "no"])

    hz, mf, vu = agg["f1_Hazardous"], agg["macro_f1"], agg["f1_Very unhealthy"]
    haz_ok = hz["wins"] == nf and hz["sig_two_sided"]
    haz_most = hz["wins"] >= (nf + 1) // 2

    if haz_ok:
        verdict = f"""**Yes — the Hazardous-class claim is validated, at the level stated below.**

The model beats persistence on **Hazardous F1 in {hz['wins']}/{nf} rolling-origin
folds**, mean {hz['model_mean']:.4f} against persistence's
{hz['persistence_mean']:.4f} (Δ {hz['mean_delta']:+.4f}, two-sided Wilcoxon
p = {hz['p_two_sided']:.4f}). Every evaluation block contains winter and carries real
Hazardous support ({min(f['model']['n_Hazardous'] for f in folds):,}–{max(f['model']['n_Hazardous'] for f in folds):,}
hours). Macro-F1 is {mf['model_mean']:.4f} ± {mf['model_std']:.4f}, beating persistence
in {mf['wins']}/{nf} folds.

Unlike the 5-fold runs elsewhere in this project, **{nf} folds let the two-sided test
actually reach significance** — its floor here is {res['min_p_two_sided']:.4f} rather
than 0.0625."""
    elif haz_most:
        verdict = f"""**Partially — the model beats persistence on Hazardous in
{hz['wins']} of {nf} folds**, mean F1 {hz['model_mean']:.4f} against
{hz['persistence_mean']:.4f} (Δ {hz['mean_delta']:+.4f}, two-sided p =
{hz['p_two_sided']:.4f}). That is a majority but not unanimous, and at
{nf} folds the test {"does" if hz['sig_two_sided'] else "does not"} clear
α = 0.05. Report the fold count, not the mean alone."""
    else:
        verdict = f"""**No.** Even with real Hazardous support in every fold
({min(f['model']['n_Hazardous'] for f in folds):,}–{max(f['model']['n_Hazardous'] for f in folds):,}
hours per block), the model beats persistence on Hazardous F1 in only
{hz['wins']}/{nf} folds (mean {hz['model_mean']:.4f} against
{hz['persistence_mean']:.4f}, Δ {hz['mean_delta']:+.4f}, p =
{hz['p_two_sided']:.4f}). **The advisory-class claim is not validated.** The evidence
gap is closed in the sense that the question is now answerable — and the answer is
negative."""

    ssm, ssp, sst = ss["model"], ss["persistence"], ss["tests"]

    return f"""# Phase 11b — PM2.5-only model on Dhaka ground truth

The experiment the project has been building toward. Every Bangladesh-side result
until now covered the four common AQI classes only, because the Mendeley reanalysis
contains 2 Hazardous hours in 3.3 years. Phase 11 showed the US Embassy reference
monitor records {a['pm25']['max']:.0f} µg/m³ peaks and thousands of Hazardous hours.
This trains and validates a model on it.

> **Two models, two jobs. This one does not replace the deployed model.**
>
> | | 7-channel Bangladesh-native (Phase 10) | **PM2.5-only (this phase)** |
> |---|---|---|
> | Role | **deployment candidate** | **advisory-class validation evidence** |
> | Data | 4 cities, reanalysis, 3.3 yr | 1 station, reference-grade, 9.1 yr |
> | Channels | PM2.5, PM10, CO + time | PM2.5 + lags + time |
> | Validated on | four common classes, 5/5 folds | all six classes, {nf} folds |
> | Hazardous support | 2 hours total | {sum(f['model']['n_Hazardous'] for f in folds):,} hours across folds |
>
> Neither subsumes the other: this one has the rare classes but one pollutant and one
> site; that one has the channels and the spatial spread but no rare classes.

---

## 1. Features and gap handling

Without other pollutants, recent history is the only signal. Features
({len(cfg.features)}): PM2.5 at *t*, cyclical hour/month, and lagged PM2.5 at
t−1, t−3, t−6, t−12, t−24. Target: the EPA class at t+{cfg.horizon} h, same
breakpoints as everywhere else.

{table(["", "Hours"], [
  ["Complete hourly grid", f"{stats['grid_hours']:,}"],
  ["Observed readings", f"{stats['observed_hours']:,}"],
  ["Missing (instrument downtime)", f"{stats['missing_hours']:,}"],
  [f"Short gaps (≤{stats['max_fill_gap_hours']} h) forward-filled + flagged", f"{stats['short_gap_filled']:,}"],
  ["Long gaps left unfilled", f"**{stats['long_gap_unfilled']:,}**"],
  ["**Windows dropped**", f"**{stats['windows_dropped']:,}** ({stats['windows_dropped_pct']:.1f}%)"],
  ["**Usable samples**", f"**{stats['samples']:,}**"],
  ["Samples with an imputed label", f"{stats['label_imputed']:,}"],
])}

Gaps longer than {stats['max_fill_gap_hours']} hours are **not** filled, and any
sample whose lags or target reach across one is dropped. Carrying a value forward
across a multi-day outage would manufacture exactly the autocorrelation the
persistence floor measures, which would flatter every model in this report.

---

## 2. Single chronological split

{cfg.train_size:.0%}/{cfg.val_size:.0%}/{cfg.test_size:.0%}, RandomForest with
`class_weight='balanced'` — the only imbalance intervention that did not damage the
advisory classes in Phases 4, 2.10 and 2.11. Validation macro-F1
{ss['val_macro_f1']:.4f}; {ss['n_test']:,} observed test samples.

{table(["Class", "Model F1", "Persistence F1", "Δ", "Test n"],
       [[f"**{l}**" if l in RARE else l, f"{ssm[f'f1_{l}']:.4f}",
         f"{ssp[f'f1_{l}']:.4f}", f"{ssm[f'f1_{l}'] - ssp[f'f1_{l}']:+.4f}",
         f"{ssm[f'n_{l}']:,}"] for l in labels]
       + [["**Macro-F1**", f"**{ssm['macro_f1']:.4f}**", f"{ssp['macro_f1']:.4f}",
           f"**{ssm['macro_f1'] - ssp['macro_f1']:+.4f}**", ""]])}

Paired bootstrap (1,000 resamples) against persistence:
macro-F1 **{sst['macro_f1']['observed_diff']:+.4f}**
[{sst['macro_f1']['ci_low']:+.4f}, {sst['macro_f1']['ci_high']:+.4f}];
Hazardous **{sst['Hazardous']['observed_diff']:+.4f}**
[{sst['Hazardous']['ci_low']:+.4f}, {sst['Hazardous']['ci_high']:+.4f}];
Very unhealthy **{sst['Very unhealthy']['observed_diff']:+.4f}**
[{sst['Very unhealthy']['ci_low']:+.4f}, {sst['Very unhealthy']['ci_high']:+.4f}].

---

## 3. Rolling-origin cross-validation ({nf} folds)

Nine years of data supports more folds than the 3.3-year Bangladesh set. **{nf} folds**
at {cv['initial_fraction']:.0%} initial training: every evaluation block contains
Nov–Feb, every block carries real Hazardous support, and — the reason for going above
five — **the two-sided Wilcoxon floor falls to {res['min_p_two_sided']:.4f}**, so
unlike the 5-fold runs elsewhere in this project the test can actually reach α = 0.05.
Embargo {cv['embargo_hours']} h (the widest feature reach, t−24 plus the horizon).

{table(["Fold", "Eval block", "Winter", "Train n", "Eval n", "Hazardous n",
        "Model macro-F1", "Persistence", "Δ"], fold_rows)}

### The advisory classes, per fold

{table(["Fold", "Haz n", "Haz F1 model", "Haz F1 persist", "Δ",
        "VU F1 model", "VU F1 persist", "Δ"], haz_rows)}

### Aggregate

{table(["Metric", "Model (mean ± std)", "Persistence", "Mean Δ", "Folds won",
        "Wilcoxon p (2-sided)", "Significant"], agg_rows)}

---

## 4. Is the Hazardous-class claim validated?

{verdict}

### What this does and does not license

- **Does:** the advisory-class evidence gap that blocked Phases 10 and 10b is closed.
  Hazardous is measurable on this data ({min(f['model']['n_Hazardous'] for f in folds):,}–{max(f['model']['n_Hazardous'] for f in folds):,}
  hours per evaluation block against 2 in the entire Mendeley record), and the
  question has an answer.
- **Does not:** license a deployment claim for the neckband. This model reads **one
  pollutant at one station**. The device carries PM2.5, PM10 and CO sensors and is
  meant to work across Bangladesh. This is validation evidence about the *task*, not
  a shippable predictor.
- **Next step, stated concretely:** join this reference PM2.5 series to co-located
  meteorology and the other pollutant channels — or deploy reference-grade monitors
  at the other three cities — so the 7-channel model can be validated on data that
  actually contains the classes it is meant to warn about.

Reproduce with `python -m src.models.dhaka_pm25_model`.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--initial-fraction", type=float, default=INITIAL_FRACTION)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    s, stats = build_samples(cfg)
    ss = single_split(cfg, s)
    print(f"\nrolling-origin CV, {args.folds} folds:")
    cv = rolling_cv(cfg, s, args.folds, args.initial_fraction)

    if not args.no_write:
        out = REPO_ROOT / "reports" / "dhaka_ground_truth_model_h6.md"
        out.write_text(build_report(cfg, stats, ss, cv))
        print(f"\nwrote {out.relative_to(REPO_ROOT)}")
        mp = REPO_ROOT / "reports" / "metrics" / "dhaka_pm25_model_h6.json"
        mp.write_text(json.dumps({"stats": stats, "single_split": ss, "cv": cv,
                                  "features": cfg.features}, indent=2, default=str))
        print(f"wrote {mp.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
