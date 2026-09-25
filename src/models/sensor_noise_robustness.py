"""Phase D — how much does the Hazardous detector degrade on low-cost-sensor input?

Phase 11b trained a Hazardous detector on the US Embassy Dhaka reference monitor, a
research-grade instrument. The device this thesis designs would not carry one. The
question is how much of that result survives the sensor a wearable can actually afford.

**Every noise parameter here is taken from one published validation study**, and the
parameters that study does not report are not modelled rather than being invented:

    Nyarku, Mazaheri, Jayaratne, Dunbabin, Rahman, Uhde, Morawska (2018),
    "Mobile phones as monitors of personal exposure to air pollution: Is this the
    future?", PLOS ONE 13(2), e0193150.

What it publishes, and what this module therefore uses:

    * **Ambient agreement with a reference instrument: R-squared 0.10, 0.23 (Rocklea)
      and 0.28, 0.15 (Woolloongabba).** Two phones at two stations. This is the only
      quantitative agreement figure the paper gives for ambient PM2.5, and it is what
      the noise amplitude is calibrated to reproduce.
    * **A detection floor.** The paper reports the phone's response was "below its
      noise level" at ambient concentrations around 10 ug/m3 for PM2.5, so readings
      below that are treated as uninformative.
    * **Linearity only above the floor**: the response was linear above 5-10 ug/m3 in
      chamber tests, with R-squared 0.86-1.00 at those elevated concentrations.

What it does NOT publish, and what is therefore absent here:

    * no calibration slope or intercept -> **no multiplicative or additive bias is
      applied**, even though real low-cost sensors have one;
    * no quantisation step -> **no rounding is applied**;
    * no numerical bias figure for PM2.5 -> none is assumed.

That makes this an **optimistic** proxy: it degrades the input with noise alone, at a
magnitude the literature supports, and omits the systematic errors the same literature
says exist but does not quantify. A real sensor would do worse. The result is a lower
bound on degradation, not an estimate of field performance, and it is not a substitute
for the calibration work Chapter 6 still lists as a limitation.

    python -m src.models.sensor_noise_robustness

Writes ``reports/sensor_noise_robustness_h6.md`` and the matching JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

# ---------------------------------------------------------------- published values
SOURCE = ("Nyarku, Mazaheri, Jayaratne, Dunbabin, Rahman, Uhde & Morawska (2018), "
          "\"Mobile phones as monitors of personal exposure to air pollution: Is this "
          "the future?\", PLOS ONE 13(2), e0193150")
AMBIENT_R2 = {"Rocklea phone 1": 0.10, "Rocklea phone 2": 0.23,
              "Woolloongabba phone 1": 0.28, "Woolloongabba phone 2": 0.15}
CHAMBER_R2 = {"cigarette smoke M1": 0.90, "cigarette smoke M2": 0.94,
              "concrete dust M1": 0.99, "concrete dust M2": 1.00,
              "petrol exhaust M1": 0.86, "petrol exhaust M2": 0.87}
DETECTION_FLOOR_UGM3 = 10.0      # "below its noise level" at ambient PM2.5 ~10 ug/m3
SEED = 42


def noise_sd_for_r2(signal: np.ndarray, target_r2: float) -> float:
    """Additive-noise standard deviation that reproduces a published R-squared.

    For ``observed = true + e`` with independent ``e``, the squared correlation between
    observed and true is ``var(true) / (var(true) + var(e))``. Solving for the noise
    variance gives ``var(e) = var(true) * (1/R2 - 1)``, so the amplitude follows from
    the published agreement figure and the series' own variance rather than from a
    number chosen to look plausible.
    """
    if not 0 < target_r2 < 1:
        raise ValueError(f"target_r2 must be in (0, 1), got {target_r2}")
    var_signal = float(np.var(signal, ddof=1))
    return float(np.sqrt(var_signal * (1.0 / target_r2 - 1.0)))


def inject(clean: np.ndarray, target_r2: float, *, seed: int = SEED,
           floor: float = DETECTION_FLOOR_UGM3) -> dict:
    """Degrade a reference series to the published agreement level.

    Two effects, both traceable to the source: additive noise scaled to reproduce the
    reported R-squared, and a detection floor below which the instrument is reported
    to be uninformative. Readings below the floor are replaced by a draw from the
    floor region rather than by the floor itself, since a sensor below its noise level
    returns noise, not a constant.
    """
    rng = np.random.default_rng(seed)
    sd = noise_sd_for_r2(clean, target_r2)
    noisy = clean + rng.normal(0.0, sd, size=len(clean))
    below = noisy < floor
    noisy[below] = np.abs(rng.normal(0.0, floor / 2.0, size=int(below.sum())))
    noisy = np.clip(noisy, 0.0, None)
    achieved = float(np.corrcoef(clean, noisy)[0, 1] ** 2)
    return {"values": noisy, "noise_sd": sd, "target_r2": target_r2,
            "achieved_r2": achieved,
            "pct_below_floor": float(below.mean() * 100),
            "mae": float(np.abs(noisy - clean).mean()),
            "bias": float((noisy - clean).mean())}


def noisy_samples(s: pd.DataFrame, cfg, target_r2: float, seed: int = SEED) -> dict:
    """Rebuild the feature columns from a once-noised series.

    Noise is injected into the underlying hourly readings **once**, and the lag
    features are then recomputed from the noised series. Adding independent noise to
    each lag column would be wrong: the same hour appears in several lag columns, and
    treating those as independent draws would understate how correlated a real
    sensor's errors are across a window.

    The **target label is left clean.** The question is whether the model can still
    detect genuinely hazardous air through a cheap sensor, so the truth comes from the
    reference instrument. Persistence is recomputed from the noisy current reading,
    because a device carrying this sensor would have nothing better to persist.
    """
    out = s.copy()
    by_time = s.set_index("datetime")["pm25"]
    inj = inject(by_time.to_numpy(dtype=float), target_r2, seed=seed)
    noised = pd.Series(inj["values"], index=by_time.index)

    out["pm25"] = out["datetime"].map(noised).to_numpy()
    for lag in [c for c in s.columns if c.startswith("pm25_lag")]:
        h = int(lag.replace("pm25_lag", ""))
        shifted = noised.shift(h)
        out[lag] = out["datetime"].map(shifted).to_numpy()
    # bin the noisy current reading to get the persistence prediction a device would make
    bins = [*cfg.breakpoints, np.inf]
    out["y_now"] = pd.cut(out["pm25"].to_numpy(dtype=float), bins=bins,
                          labels=False, right=True)
    out = out.dropna(subset=cfg.features + ["y_now"])
    out["y_now"] = out["y_now"].astype(int)
    return {"samples": out, "injection": {k: v for k, v in inj.items()
                                          if k != "values"}}


def evaluate(cfg, clean: pd.DataFrame, noisy: pd.DataFrame, verbose=True) -> dict:
    """Train on clean data, evaluate on noisy data. No retraining on noise.

    The fold structure is Phase 11b's, unchanged. Within each fold the model is fitted
    on the clean training block exactly as before, and then scored twice: once on the
    clean evaluation block and once on the same rows after injection. Any difference
    is attributable to the input, since nothing about the model changed.
    """
    from sklearn.preprocessing import StandardScaler
    from src.models import dhaka_pm25_model as p11

    say = print if verbose else (lambda *a, **k: None)
    ts = np.sort(clean.datetime.unique())
    edges = np.linspace(int(len(ts) * p11.INITIAL_FRACTION), len(ts),
                        p11.N_FOLDS + 1).astype(int)
    folds = []
    for f in range(p11.N_FOLDS):
        cut = pd.Timestamp(ts[edges[f]])
        end = pd.Timestamp(ts[edges[f + 1] - 1])
        tr = clean[clean.datetime + pd.Timedelta(hours=cfg.horizon) < cut]
        lo = cut + pd.Timedelta(hours=p11.EMBARGO_HOURS)
        ev_c = clean[(clean.datetime >= lo) & (clean.datetime <= end)]
        ev_n = noisy[noisy.datetime.isin(ev_c.datetime)]
        ev_c = ev_c[ev_c.datetime.isin(ev_n.datetime)]
        obs = ~ev_c.is_imputed_pm25.to_numpy()
        y = ev_c.y.to_numpy()[obs]                       # truth stays clean

        sc = StandardScaler().fit(tr[cfg.features].to_numpy(float))
        model = p11.fit_rf(cfg, sc.transform(tr[cfg.features].to_numpy(float))
                           .astype(np.float32), tr.y.to_numpy())

        pred_c = model.predict(sc.transform(
            ev_c[cfg.features].to_numpy(float)).astype(np.float32))[obs]
        pred_n = model.predict(sc.transform(
            ev_n[cfg.features].to_numpy(float)).astype(np.float32))[obs]

        e = {"fold": f + 1,
             "eval_start": str(ev_c.datetime.min()), "eval_end": str(ev_c.datetime.max()),
             "n_train": int(len(tr)), "n_eval": int(obs.sum()),
             "clean": p11.score(y, pred_c, cfg.labels),
             "noisy": p11.score(y, pred_n, cfg.labels),
             "persistence_clean": p11.score(y, ev_c.y_now.to_numpy()[obs], cfg.labels),
             "persistence_noisy": p11.score(y, ev_n.y_now.to_numpy()[obs], cfg.labels)}
        folds.append(e)
        say(f"  F{e['fold']} n={e['n_eval']:>6,} | Haz clean "
            f"{e['clean']['f1_Hazardous']:.4f} -> noisy {e['noisy']['f1_Hazardous']:.4f} "
            f"| floor {e['persistence_clean']['f1_Hazardous']:.4f} -> "
            f"{e['persistence_noisy']['f1_Hazardous']:.4f}")

    def agg(key, metric):
        v = np.array([f[key][metric] for f in folds])
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                "per_fold": [float(x) for x in v]}

    summary = {}
    for metric in ("macro_f1", "f1_Hazardous", "accuracy"):
        c, n = agg("clean", metric), agg("noisy", metric)
        pc, pn = agg("persistence_clean", metric), agg("persistence_noisy", metric)
        summary[metric] = {
            "clean": c, "noisy": n,
            "persistence_clean": pc, "persistence_noisy": pn,
            "absolute_drop": c["mean"] - n["mean"],
            "relative_drop_pct": ((c["mean"] - n["mean"]) / c["mean"] * 100
                                  if c["mean"] else None),
            "noisy_beats_noisy_floor": bool(n["mean"] > pn["mean"]),
            "margin_over_noisy_floor": n["mean"] - pn["mean"],
            "folds_won_noisy": int(sum(a > b for a, b in
                                       zip(n["per_fold"], pn["per_fold"]))),
        }
    return {"folds": folds, "summary": summary, "n_folds": p11.N_FOLDS}


def _tbl(h, rows):
    return "\n".join(["| " + " | ".join(h) + " |",
                      "| " + " | ".join(["---"] * len(h)) + " |"]
                     + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def build_report(p: dict) -> str:
    s, inj, n = p["summary"], p["injection"], p["n_folds"]
    haz, mac = s["f1_Hazardous"], s["macro_f1"]
    main = _tbl(
        ["Metric", "Clean input", "Noise-injected", "Absolute drop", "Relative drop",
         "Noisy floor", "Beats its floor?"],
        [[name,
          f"{b['clean']['mean']:.4f} ± {b['clean']['std']:.4f}",
          f"{b['noisy']['mean']:.4f} ± {b['noisy']['std']:.4f}",
          f"{b['absolute_drop']:+.4f}".replace("+", "−") if b["absolute_drop"] > 0
          else f"{-b['absolute_drop']:+.4f}",
          f"{b['relative_drop_pct']:.1f}%" if b["relative_drop_pct"] is not None else "n/a",
          f"{b['persistence_noisy']['mean']:.4f}",
          f"**{'yes' if b['noisy_beats_noisy_floor'] else 'no'}** "
          f"({b['folds_won_noisy']}/{n} folds)"]
         for name, b in (("Hazardous F1", haz), ("Macro-F1", mac),
                         ("Accuracy", s["accuracy"]))])

    per_fold = _tbl(
        ["Fold", "Evaluation window", "n", "Haz clean", "Haz noisy", "Δ"],
        [[f["fold"], f"{f['eval_start'][:10]} to {f['eval_end'][:10]}",
          f"{f['n_eval']:,}", f"{f['clean']['f1_Hazardous']:.4f}",
          f"{f['noisy']['f1_Hazardous']:.4f}",
          f"{f['noisy']['f1_Hazardous'] - f['clean']['f1_Hazardous']:+.4f}"]
         for f in p["folds"]])

    published = _tbl(["Reported quantity", "Published value", "Used here"],
                     [["Ambient R² vs reference, phone 1 at Rocklea", "0.10", "yes"],
                      ["Ambient R² vs reference, phone 2 at Rocklea", "0.23", "yes"],
                      ["Ambient R² vs reference, phone 1 at Woolloongabba", "0.28", "yes"],
                      ["Ambient R² vs reference, phone 2 at Woolloongabba", "0.15", "yes"],
                      ["Chamber R², elevated concentrations", "0.86 – 1.00",
                       "context only"],
                      ["Response below noise level at ambient PM2.5", "~10 µg/m³",
                       "yes, as a detection floor"],
                      ["Calibration slope / intercept", "**not reported**",
                       "**not modelled**"],
                      ["Quantisation step", "**not reported**", "**not modelled**"],
                      ["Numerical PM2.5 bias", "**not reported**", "**not modelled**"]])

    return f"""# Sensor-noise robustness — Hazardous detector, horizon 6 h

Phase 11b trained a Hazardous detector on the **US Embassy Dhaka reference monitor**, a
research-grade instrument. The device this thesis designs would not carry one. This
section asks how much of that result survives input from a sensor a wearable can
afford.

**This is a synthetic proxy for real sensor noise, not a substitute for field
calibration.** No low-cost sensor was deployed, no co-location study was run, and
nothing here measures how the PulseAir hardware actually behaves. The field-calibration
work remains listed as a limitation.

---

## 1. Where the noise parameters come from

Every parameter is taken from one published validation study, and **the parameters that
study does not report are not modelled rather than invented**:

> {SOURCE}

{published}

The paper evaluated a phone-mounted PM2.5 sensor against reference instruments in a
chamber and at two ambient monitoring stations. Its ambient agreement figures are the
only quantitative PM2.5 agreement it gives, and they are what the noise amplitude here
is calibrated to reproduce.

**Method.** For additive noise, the squared correlation between observed and true is
`var(true) / (var(true) + var(noise))`, so a published R² fixes the noise variance:
`var(noise) = var(true) · (1/R² − 1)`. The amplitude therefore follows from the
published agreement figure and the series' own variance rather than from a number
chosen to look plausible. Readings falling below the reported ~10 µg/m³ detection floor
are replaced by noise, because a sensor below its noise level returns noise rather than
a constant.

**Target R² = {inj['target_r2']:.2f}** (the mid-range of the four published ambient
values). Achieved R² after injection: **{inj['achieved_r2']:.4f}**; noise standard
deviation **{inj['noise_sd']:.1f} µg/m³**; **{inj['pct_below_floor']:.1f}%** of readings
fell below the detection floor; mean absolute error against the reference series
**{inj['mae']:.1f} µg/m³**.

**This is an optimistic proxy.** It degrades the input with noise alone, at a magnitude
the literature supports, and omits the calibration bias and quantisation the same
literature says exist but does not quantify. A real sensor would do worse. Read the
numbers below as a **lower bound on degradation**.

## 2. Result

The model is trained on clean reference data exactly as in Phase 11b and **is not
retrained**. Within each fold it is scored twice on the same rows: once on clean input
and once after injection. The target label stays clean — the question is whether
genuinely hazardous air is still detected through a cheap sensor. Persistence is
recomputed from the noisy reading, since a device carrying this sensor would have
nothing better to persist.

{main}

## 3. Per fold

{per_fold}

## 4. What this establishes

- **Hazardous F1 falls from {haz['clean']['mean']:.4f} to {haz['noisy']['mean']:.4f}**,
  a relative loss of **{haz['relative_drop_pct']:.1f}%**.
- Against the floor a device would actually face — persistence computed from the same
  noisy sensor, {haz['persistence_noisy']['mean']:.4f} — the degraded model
  {'still holds a margin of ' + format(haz['margin_over_noisy_floor'], '+.4f')
   if haz['noisy_beats_noisy_floor'] else
   'no longer leads, by ' + format(haz['margin_over_noisy_floor'], '+.4f')},
  winning **{haz['folds_won_noisy']} of {n}** folds.
- The comparison that matters for a product is the second one. A model that loses
  accuracy on cheap input but still beats what that same cheap input gives you for
  free is doing something; one that does not is not worth shipping.

**The honest bound.** This says what happens under noise of a published magnitude with
no bias and no quantisation. It does not say what happens under a real sensor's
systematic error, because the study cited does not report one and this project has not
measured one. Closing that gap needs a co-location study against a reference monitor,
which is hardware work rather than modelling work.

Regenerate with `python -m src.models.sensor_noise_robustness`.
"""


def run(*, write: bool = True, verbose: bool = True, target_r2: float | None = None,
        seed: int = SEED) -> dict:
    from src.models import dhaka_pm25_model as p11

    say = print if verbose else (lambda *a, **k: None)
    cfg = p11.load_config()
    if target_r2 is None:
        target_r2 = float(np.mean(list(AMBIENT_R2.values())))

    say(f"published ambient R2 values: {AMBIENT_R2}")
    say(f"using target R2 = {target_r2:.4f} (mean of the four)\n")
    clean, _ = p11.build_samples(cfg, verbose=False)
    nz = noisy_samples(clean, cfg, target_r2, seed=seed)
    inj = nz["injection"]
    say(f"injection: noise sd {inj['noise_sd']:.1f} ug/m3 | achieved R2 "
        f"{inj['achieved_r2']:.4f} | {inj['pct_below_floor']:.1f}% below floor | "
        f"MAE {inj['mae']:.1f} ug/m3\n")

    res = evaluate(cfg, clean, nz["samples"], verbose=verbose)
    payload = {"source": SOURCE, "published_ambient_r2": AMBIENT_R2,
               "published_chamber_r2": CHAMBER_R2,
               "detection_floor_ugm3": DETECTION_FLOOR_UGM3,
               "not_modelled": ["calibration slope", "calibration intercept",
                                "quantisation step", "numerical PM2.5 bias"],
               "injection": inj, "seed": seed, **res}
    h = payload["summary"]["f1_Hazardous"]
    say(f"\nHazardous F1: clean {h['clean']['mean']:.4f} -> noisy "
        f"{h['noisy']['mean']:.4f}  ({h['relative_drop_pct']:.1f}% relative loss)")
    say(f"noisy model vs noisy floor ({h['persistence_noisy']['mean']:.4f}): "
        f"{h['margin_over_noisy_floor']:+.4f}, {h['folds_won_noisy']}/"
        f"{payload['n_folds']} folds")

    if write:
        rep = REPO_ROOT / "reports" / "sensor_noise_robustness_h6.md"
        met = REPO_ROOT / "reports" / "metrics" / "sensor_noise_robustness_h6.json"
        met.parent.mkdir(parents=True, exist_ok=True)
        met.write_text(json.dumps(payload, indent=2, default=str))
        rep.write_text(build_report(payload))
        say(f"\nwrote {rep.relative_to(REPO_ROOT)}")
        say(f"wrote {met.relative_to(REPO_ROOT)}")
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--target-r2", type=float, default=None,
                    help="override the published ambient R2 target")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    run(write=not args.no_write, target_r2=args.target_r2, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
