"""Phase C — does a Beijing model generalise to a station it never saw?

Rolling-origin CV answers "does this model hold up over time". It says nothing about
space. A wearable is carried somewhere the training data does not cover, so the
question that matters for deployment is whether a model trained on eleven monitoring
stations works at a twelfth.

Twelve leave-one-station-out folds. Train on the other eleven, evaluate on the held-out
one, and compare against **that station's own persistence floor** rather than a global
one — a station whose air is stickier is easier to predict by doing nothing, and
scoring it against someone else's baseline would flatter or punish it for that alone.

The model is the class-weighted RandomForest, which was the only imbalance
intervention in Phase 4 that did not degrade the advisory classes.

    python -m src.models.station_holdout_cv

Writes ``reports/station_holdout_h6.md`` and ``reports/metrics/station_holdout_h6.json``.
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
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
N_BOOT = 1000


@dataclass
class HoldoutConfig:
    processed_dir: Path
    reports_dir: Path
    horizon: int
    labels: list[str]
    breakpoints: list[float]
    rf: dict
    seed: int

    @property
    def report_path(self) -> Path:
        return self.reports_dir / f"station_holdout_h{self.horizon}.md"

    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / "metrics" / f"station_holdout_h{self.horizon}.json"


def load_config(path: Path | str = DEFAULT_CONFIG) -> HoldoutConfig:
    raw = yaml.safe_load(Path(path).read_text())
    h = int(raw["preprocessing"]["horizon"])
    return HoldoutConfig(
        processed_dir=REPO_ROOT / "data" / "processed" / f"h{h}",
        reports_dir=REPO_ROOT / "reports",
        horizon=h,
        labels=list(raw["data"]["pm25_labels"]),
        breakpoints=list(raw["data"]["pm25_breakpoints"]),
        rf=dict(raw["baseline"]["random_forest"]),
        seed=int(raw.get("seed", 42)),
    )


def load_all(cfg: HoldoutConfig) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Every h6 sample in physical units.

    The committed CSVs are standardised with statistics fitted on the global training
    block. Holding out a station and re-fitting inside the fold requires raw units, so
    the Phase 2 transform is inverted here exactly as the rolling-origin module does.
    """
    meta = json.loads((cfg.processed_dir / "metadata.json").read_text())
    scaler = joblib.load(cfg.processed_dir / "scaler.pkl")
    scaled_cols, features = list(meta["scaled_columns"]), list(meta["feature_columns"])

    df = pd.concat([pd.read_csv(cfg.processed_dir / f"tabular_{s}.csv")
                    for s in ("train", "val", "test")], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    raw = scaler.inverse_transform(df[scaled_cols].to_numpy(dtype=np.float64))
    for i, c in enumerate(scaled_cols):
        df[c] = raw[:, i]
    return df, features, scaled_cols


def station_persistence(cfg: HoldoutConfig, block: pd.DataFrame) -> np.ndarray:
    """Predict that the category in h hours equals the category now, for this station."""
    bins = [*cfg.breakpoints, np.inf]
    return pd.cut(block["PM2.5"].to_numpy(dtype=float), bins=bins,
                  labels=False, right=True).astype(np.int16)


def score(y: np.ndarray, pred: np.ndarray, labels: list[str]) -> dict:
    idx = list(range(len(labels)))
    out = {"macro_f1": float(f1_score(y, pred, labels=idx, average="macro",
                                      zero_division=0)),
           "accuracy": float((y == pred).mean())}
    for i, lab in enumerate(labels):
        out[f"f1_{lab}"] = float(f1_score(y == i, pred == i, zero_division=0))
    return out


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, n_classes: int,
                     seed: int, n_boot: int = N_BOOT, metric: str = "macro") -> dict:
    """Difference between two predictors on identical rows, with a percentile CI.

    Pairing matters: both predictors are resampled on the same row indices, so the
    shared difficulty of those rows cancels rather than adding noise to the interval.
    """
    rng = np.random.default_rng(seed)
    idx_all = list(range(n_classes))

    def m(yy, pp):
        if metric == "macro":
            return f1_score(yy, pp, labels=idx_all, average="macro", zero_division=0)
        cls = int(metric)
        return f1_score(yy == cls, pp == cls, zero_division=0)

    observed = m(y, a) - m(y, b)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        diffs[i] = m(y[s], a[s]) - m(y[s], b[s])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"observed_diff": float(observed), "ci_low": float(lo), "ci_high": float(hi),
            "p_two_sided": float(min(1.0, p)), "significant": bool(min(1.0, p) < 0.05),
            "n_boot": n_boot}


def run(cfg: HoldoutConfig | None = None, *, write: bool = True,
        verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    say = print if verbose else (lambda *a, **k: None)

    df, features, scaled_cols = load_all(cfg)
    stations = sorted(df["station"].unique())
    say(f"h{cfg.horizon}: {len(df):,} samples across {len(stations)} stations")
    say(f"leave-one-station-out, class-weighted RandomForest, observed labels only\n")

    labels = cfg.labels
    results = []
    for i, station in enumerate(stations, 1):
        t0 = time.perf_counter()
        tr = df[df["station"] != station]
        ev = df[df["station"] == station]
        obs = (~ev["is_imputed_pm25"].to_numpy()).astype(bool)
        y = ev["y_category"].to_numpy()[obs]

        # Scaler fitted on the eleven training stations only. Fitting on all twelve
        # would leak the held-out station's distribution into its own evaluation.
        scaler = StandardScaler().fit(tr[scaled_cols].to_numpy(dtype=np.float64))

        def prep(frame):
            out = frame[features].copy()
            out[scaled_cols] = scaler.transform(
                frame[scaled_cols].to_numpy(dtype=np.float64))
            return out.to_numpy(dtype=np.float32)

        model = RandomForestClassifier(
            n_estimators=cfg.rf["n_estimators"], max_depth=cfg.rf["max_depth"],
            min_samples_leaf=cfg.rf["min_samples_leaf"],
            max_features=cfg.rf["max_features"], class_weight="balanced",
            random_state=cfg.seed, n_jobs=-1).fit(prep(tr), tr["y_category"].to_numpy())

        pred = model.predict(prep(ev))[obs]
        floor = station_persistence(cfg, ev)[obs]

        m_model, m_floor = score(y, pred, labels), score(y, floor, labels)
        entry = {
            "station": station,
            "n_train": int(len(tr)), "n_eval": int(len(ev)),
            "n_eval_observed": int(obs.sum()),
            "support": {lab: int((y == k).sum()) for k, lab in enumerate(labels)},
            "pm25_median": float(ev["PM2.5"].median()),
            "pm25_p95": float(ev["PM2.5"].quantile(0.95)),
            "label_unchanged_pct": float((floor == y).mean() * 100),
            "model": m_model, "persistence": m_floor,
            "vs_persistence": {
                "macro_f1": paired_bootstrap(y, pred, floor, len(labels), cfg.seed),
                "Hazardous": paired_bootstrap(y, pred, floor, len(labels), cfg.seed,
                                              metric=str(labels.index("Hazardous"))),
            },
            "fit_seconds": round(time.perf_counter() - t0, 1),
        }
        results.append(entry)
        d = entry["vs_persistence"]["macro_f1"]
        say(f"[{i:>2}/{len(stations)}] {station:<14} "
            f"macro-F1 {m_model['macro_f1']:.4f} vs floor {m_floor['macro_f1']:.4f} "
            f"({d['observed_diff']:+.4f}{'*' if d['significant'] else ' '}) | "
            f"Haz {m_model['f1_Hazardous']:.4f} vs {m_floor['f1_Hazardous']:.4f} "
            f"| {entry['fit_seconds']:.0f}s")

    agg = aggregate(results, labels)
    say("\n" + "=" * 74)
    say(f"stations where the model beats its own floor on macro-F1: "
        f"{agg['wins_macro']}/{len(results)}  "
        f"(significantly: {agg['wins_macro_significant']})")
    say(f"stations where it beats the floor on Hazardous F1:        "
        f"{agg['wins_hazardous']}/{len(results)}  "
        f"(significantly: {agg['wins_hazardous_significant']})")
    say(f"mean delta macro-F1 {agg['mean_delta_macro']:+.4f}  "
        f"(sd {agg['sd_delta_macro']:.4f})")
    say(f"Wilcoxon over stations: two-sided p = {agg['p_two_sided']:.4f} "
        f"(floor {agg['resolution']['min_p_two_sided']:.4f})")
    say(f"\nverdict: {agg['verdict']}")
    if agg["outlier"]:
        say(f"outlier: {agg['outlier']['station']} — {agg['outlier']['why']}")

    payload = {"horizon": cfg.horizon, "n_stations": len(results),
               "labels": labels, "stations": results, "aggregate": agg,
               "model": "RandomForest (class_weight=balanced)", "n_boot": N_BOOT}
    if write:
        cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.metrics_path.write_text(json.dumps(payload, indent=2, default=str))
        cfg.report_path.write_text(build_report(payload))
        say(f"\nwrote {cfg.report_path.relative_to(REPO_ROOT)}")
        say(f"wrote {cfg.metrics_path.relative_to(REPO_ROOT)}")
    return payload


def aggregate(results: list[dict], labels: list[str]) -> dict:
    """Summarise across stations, and look for one that behaves unlike the rest."""
    from scipy.stats import wilcoxon

    d_macro = np.array([r["vs_persistence"]["macro_f1"]["observed_diff"]
                        for r in results])
    d_haz = np.array([r["vs_persistence"]["Hazardous"]["observed_diff"]
                      for r in results])
    n = len(results)
    try:
        p = float(wilcoxon(d_macro, alternative="two-sided").pvalue)
    except ValueError:                      # all differences identical
        p = 1.0

    wins = int((d_macro > 0).sum())
    wins_sig = int(sum(r["vs_persistence"]["macro_f1"]["significant"]
                       and r["vs_persistence"]["macro_f1"]["observed_diff"] > 0
                       for r in results))

    if wins == n:
        verdict = "generalisation holds"
    elif wins >= n * 0.75:
        verdict = "generalisation largely holds"
    elif wins >= n * 0.4:
        verdict = "generalisation partially holds"
    else:
        verdict = "generalisation fails"

    return {
        "wins_macro": wins, "wins_macro_significant": wins_sig,
        "wins_hazardous": int((d_haz > 0).sum()),
        "wins_hazardous_significant": int(sum(
            r["vs_persistence"]["Hazardous"]["significant"]
            and r["vs_persistence"]["Hazardous"]["observed_diff"] > 0
            for r in results)),
        "mean_delta_macro": float(d_macro.mean()),
        "sd_delta_macro": float(d_macro.std(ddof=1)),
        "min_delta_macro": float(d_macro.min()),
        "max_delta_macro": float(d_macro.max()),
        "mean_delta_hazardous": float(d_haz.mean()),
        "p_two_sided": p,
        "resolution": {"n_stations": n, "min_p_two_sided": 2.0 ** (1 - n)},
        "verdict": verdict,
        "outlier": _outlier(results, d_macro),
    }


def _outlier(results: list[dict], d_macro: np.ndarray) -> dict | None:
    """Name a station that behaves unlike the others, and say why from the data.

    Reported only when a station is more than two standard deviations from the mean
    difference, so that a merely unlucky station is not promoted into a finding.
    """
    if len(results) < 4 or d_macro.std(ddof=1) == 0:
        return None
    z = (d_macro - d_macro.mean()) / d_macro.std(ddof=1)
    k = int(np.argmax(np.abs(z)))
    if abs(z[k]) < 2.0:
        return None
    r = results[k]
    meds = np.array([x["pm25_median"] for x in results])
    stick = np.array([x["label_unchanged_pct"] for x in results])
    reasons = []
    if abs(r["pm25_median"] - meds.mean()) > 1.5 * meds.std(ddof=1):
        reasons.append(f"its median PM2.5 is {r['pm25_median']:.1f} against a "
                       f"{meds.mean():.1f} average across stations")
    if abs(r["label_unchanged_pct"] - stick.mean()) > 1.5 * stick.std(ddof=1):
        reasons.append(f"its category is unchanged over the horizon "
                       f"{r['label_unchanged_pct']:.1f}% of the time against "
                       f"{stick.mean():.1f}% elsewhere, so its own floor sits "
                       f"unusually {'high' if r['label_unchanged_pct'] > stick.mean() else 'low'}")
    return {"station": r["station"], "z": float(z[k]),
            "delta": float(d_macro[k]),
            "why": ("; ".join(reasons) if reasons else
                    "no distinguishing feature is visible in the data recorded here")}


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def build_report(p: dict) -> str:
    agg, labels = p["aggregate"], p["labels"]
    n = p["n_stations"]
    rows = []
    for r in sorted(p["stations"], key=lambda x: -x["vs_persistence"]["macro_f1"]["observed_diff"]):
        d = r["vs_persistence"]["macro_f1"]
        dh = r["vs_persistence"]["Hazardous"]
        rows.append([
            r["station"], f"{r['n_eval_observed']:,}",
            f"{r['model']['macro_f1']:.4f}", f"{r['persistence']['macro_f1']:.4f}",
            f"{d['observed_diff']:+.4f}" + ("**\\***" if d["significant"] else ""),
            f"{r['model']['f1_Hazardous']:.4f}", f"{r['persistence']['f1_Hazardous']:.4f}",
            f"{dh['observed_diff']:+.4f}" + ("**\\***" if dh["significant"] else ""),
        ])
    profile = _table(
        ["Station", "Median PM2.5", "95th pct", "Category unchanged over h",
         "Hazardous n"],
        [[r["station"], f"{r['pm25_median']:.1f}", f"{r['pm25_p95']:.1f}",
          f"{r['label_unchanged_pct']:.1f}%", f"{r['support']['Hazardous']:,}"]
         for r in sorted(p["stations"], key=lambda x: -x["pm25_median"])])

    out = agg["outlier"]
    outlier_txt = (
        f"**{out['station']} is a clear outlier** ({out['delta']:+.4f} against its own "
        f"floor, {abs(out['z']):.1f} standard deviations from the mean difference). "
        f"Looking for why in the recorded data: {out['why']}."
        if out else
        "**No station is a clear outlier.** No station's difference against its own "
        "floor sits more than two standard deviations from the mean, so the spread "
        "below is ordinary variation rather than one site behaving unlike the others.")

    verdict_txt = {
        "generalisation holds":
            "The model beats the held-out station's own persistence floor at every "
            "station. Generalisation to an unseen site holds.",
        "generalisation largely holds":
            "The model beats the held-out station's own floor at most sites but not "
            "all. Generalisation largely holds, with named exceptions.",
        "generalisation partially holds":
            "The model beats the held-out station's own floor at some sites and loses "
            "at others. Generalisation partially holds: a model trained on eleven "
            "stations is not reliably better than doing nothing at the twelfth.",
        "generalisation fails":
            "The model loses to the held-out station's own persistence floor at most "
            "sites. Generalisation to an unseen station fails.",
    }[agg["verdict"]]

    return f"""# Station-held-out cross-validation — horizon {p['horizon']} h

Rolling-origin cross-validation asks whether a model holds up **over time**. It says
nothing about **space**. A wearable is carried where the training data does not reach,
so the deployment question is whether a model trained on eleven Beijing monitoring
stations works at a twelfth it has never seen.

{n} leave-one-station-out folds. Train on the other eleven, evaluate on the held-out
station, `{p['model']}`, observed labels only. The scaler is fitted on the eleven
training stations alone, because fitting it on all twelve would leak the held-out
station's distribution into its own evaluation.

**Each station is compared against its own persistence floor, not a global one.** A
station whose air is stickier is easier to predict by doing nothing, and scoring it
against another station's baseline would flatter or punish it for geography rather
than for modelling.

---

## 1. Per-station results

{_table(["Held-out station", "Observed n", "Model macro-F1", "Its own floor",
         "Difference", "Model Haz F1", "Floor Haz F1", "Haz difference"], rows)}

`*` marks a difference significant at 0.05 under a paired bootstrap
({p['n_boot']:,} resamples on identical rows).

## 2. Verdict

**{agg['verdict'].capitalize()}.** {verdict_txt}

- Beats its own floor on macro-F1 at **{agg['wins_macro']} of {n}** stations
  (significantly at {agg['wins_macro_significant']}).
- Beats its own floor on Hazardous F1 at **{agg['wins_hazardous']} of {n}**
  (significantly at {agg['wins_hazardous_significant']}).
- Mean difference **{agg['mean_delta_macro']:+.4f}** macro-F1
  (sd {agg['sd_delta_macro']:.4f}, range {agg['min_delta_macro']:+.4f} to
  {agg['max_delta_macro']:+.4f}).
- Wilcoxon signed-rank over the {n} station-level differences:
  two-sided p = **{agg['p_two_sided']:.4f}**. The floor for {n} stations is
  {agg['resolution']['min_p_two_sided']:.4f}, so that value is interpretable rather
  than bounded.

{outlier_txt}

## 3. Station profiles

Whether a station is hard is partly a property of its air, so the profiles are
reported alongside the results rather than left implicit.

{profile}

## 4. Why this is an easier test than rolling-origin CV

**These two results are not in conflict, and the difference between them is
instructive.** Rolling-origin CV found that no Beijing-trained model beats persistence
in a majority of chronological folds. This section may find the opposite. The reason is
that holding out a *station* leaves the *time axis intact*: the model trains on eleven
stations across the whole 2013–2017 record and is evaluated on a twelfth over the same
period. It has therefore already seen every pollution episode, every winter and every
synoptic event in the evaluation window — just measured somewhere else in the same
city.

That is a genuinely easier problem than forecasting a period it has never seen. Beijing
stations are tens of kilometres apart in one airshed, and a haze episode arrives at all
of them; a model that has learned what such an episode looks like at eleven sites is
not being asked to extrapolate when it meets the twelfth.

**So this section bounds spatial transfer within a shared period, not deployment.** A
device carried somewhere new, forecasting a time nobody has seen, faces both problems
at once. The rolling-origin result is the binding one for that case, and nothing here
softens it.

A stricter version of this test would hold out a station *and* the later part of the
record simultaneously. That was not run, and the claim below is limited accordingly.

## 5. What this adds to the thesis

This is a different axis of generalisation from the rolling-origin result, and it is
worth stating which is which. Rolling-origin CV showed that no Beijing-trained model
beats persistence in a majority of chronological folds — a statement about *time*.
This section holds the time axis fixed and varies *place*.

The two together bound the claim a deployed model can make. A model that generalised
across stations but not across time would be a seasonal artefact; one that generalised
across time but not stations would be site-specific. The persistence floor is computed
per station here for the same reason it is computed per dataset elsewhere: a baseline
borrowed from somewhere else is not a baseline.

Regenerate with `python -m src.models.station_holdout_cv`.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    run(load_config(args.config), write=not args.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
