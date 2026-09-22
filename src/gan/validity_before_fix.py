"""Regenerate the validity numbers from the ORIGINAL, BROKEN CTGAN configuration.

**This module exists for one figure.** It deliberately reconstructs the first version
of Phase 4 -- the one that handed CTGAN the model's own nine feature columns directly,
standardized, with the four cyclical encodings modelled as free continuous variables
and the dew-point constraint passed as a plain dict that SDV accepts and ignores. That
configuration was replaced because it produced physically impossible rows. Nothing
here is an attempt to revive it.

What it does:
    Fits the broken configuration, samples the same per-class counts the real run used,
    and measures the same validity checks. Writes reports/metrics/gan_validity_before_fix.json.

What it explicitly does NOT do:
    * It does not write to data/processed/. No training table is touched.
    * It does not train, score, or select any model.
    * It does not change a single reported result. The "after" numbers in
      gan_h6.json, the ablation, and every downstream conclusion are unaffected.
    * Its output feeds exactly one consumer: the before/after panel of figure 04.

The numbers it targets are already in the prose of
``reports/gan_quality_report_h6.md`` section 5, written down when the broken run was
first diagnosed:

    19.5%    of synthetic rows on the unit circle
    82,885   distinct hour_sin values (24 hours exist)
    13.1%    of synthetic rows with DEWP > TEMP (0.18% in the real data)

CTGAN is stochastic and the original run's RNG state was not saved, so an exact match
is not expected and would be slightly suspicious. The JSON records the regenerated
value beside the reported one for each check, with an explicit verdict, so a reader
can see how close the reproduction is rather than taking "it matches" on trust.

    python -m src.gan.validity_before_fix              # ~20 min, 4 synthesizers
    python -m src.gan.validity_before_fix --epochs 5   # smoke test, numbers meaningless
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.gan import augment as ag

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "reports" / "metrics" / "gan_validity_before_fix.json"

# What reports/gan_quality_report_h6.md section 5 already says. Held here as the
# comparison target, never as a substitute: if the run does not reproduce these, the
# JSON says so and the figure carries the regenerated numbers, not these.
REPORTED = {
    "on_unit_circle_pct": 19.5,
    "distinct_hour_sin": 82_885,
    "dewp_above_temp_pct": 13.1,
    "dewp_above_temp_pct_real": 0.18,
}
# How close counts as a reproduction. CTGAN is stochastic; these are generous on
# purpose, because the claim being checked is "the same qualitative failure at the
# same order of magnitude", not "the same random draw".
TOLERANCE = {
    "on_unit_circle_pct": 8.0,       # percentage points
    "distinct_hour_sin": 0.05,       # relative: within 5% of the reported count
    "dewp_above_temp_pct": 5.0,      # percentage points
    "dewp_above_temp_pct_real": 0.05,
}


def broken_metadata(frame: pd.DataFrame):
    """Metadata as the original run had it: every column numerical.

    This is the defect, stated as data. ``hour_sin`` and ``hour_cos`` are a
    deterministic function of one integer with 24 values; typed numerical they become
    two unrelated continuous columns and CTGAN is free to emit any point in the plane.
    """
    from sdv.metadata import Metadata
    columns = {c: {"sdtype": "numerical"} for c in frame.columns}
    return Metadata.load_from_dict({"tables": {"wearable": {"columns": columns}}})


def _add_dict_constraint(synth) -> dict:
    """Attach the dew-point constraint the way the original run did: as a dict.

    SDV accepts this and does nothing with it. That silent no-op is the second half of
    the finding, so it is reproduced rather than skipped -- but the outcome is recorded,
    because if a later SDV version starts rejecting the dict, that is worth knowing and
    the reconstruction is still valid (an ignored constraint and an absent one are the
    same constraint).
    """
    spec = {
        "constraint_class": "Inequality",
        "constraint_parameters": {
            "low_column_name": "DEWP",
            "high_column_name": "TEMP",
            "strict_boundaries": False,
        },
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            synth.add_constraints(constraints=[spec])
        notes = [str(w.message) for w in caught]
        return {"dict_constraint": "accepted_and_ignored" if notes else "accepted",
                "sdv_warnings": notes,
                "detail": ("SDV accepted the dict and warned that it is ignored. The "
                           "original run predated that warning and got silence, but "
                           "the behaviour is the same: the constraint does nothing.")
                          if notes else
                          "SDV accepted the dict form without complaint, as in the "
                          "original run."}
    except Exception as exc:                       # noqa: BLE001 - recorded, not hidden
        return {"dict_constraint": "rejected", "sdv_warnings": [],
                "detail": f"{type(exc).__name__}: {exc}. Proceeding unconstrained, "
                          f"which is what the ignored dict amounted to anyway."}


def _fit_broken(real: pd.DataFrame, cfg: ag.GanConfig, n_samples: int, verbose: bool):
    from sdv.single_table import CTGANSynthesizer

    metadata = broken_metadata(real)
    synth = CTGANSynthesizer(metadata, epochs=cfg.epochs, batch_size=cfg.batch_size,
                             verbose=False, cuda=False)
    constraint_outcome = _add_dict_constraint(synth)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        synth.fit(real)
    fit_s = time.perf_counter() - t0
    sampled = synth.sample(num_rows=n_samples)
    if verbose:
        print(f"    fit {fit_s / 60:.1f} min | sampled {len(sampled):,} | "
              f"constraint {constraint_outcome['dict_constraint']}")
    return sampled, fit_s, constraint_outcome


def zero_tolerance_dewp(frame: pd.DataFrame, cfg: ag.GanConfig) -> dict:
    """DEWP > TEMP with no tolerance, which is how the original check was written.

    ``augment.validity_checks`` uses a 1e-6 degC tolerance to separate a real violation
    from float noise around saturation. The original run had no such tolerance, which
    is why its real-data rate reads 0.18% (every row a hair over, including the ~532
    that are float noise) rather than 0.0014%. Both are computed so the comparison
    against the prose is exact rather than approximate.
    """
    import joblib
    meta = json.loads((cfg.processed_dir / "metadata.json").read_text())
    scaled_cols = list(meta["scaled_columns"])
    scaler = joblib.load(cfg.processed_dir / "scaler.pkl")

    def unscale(part, col):
        i = scaled_cols.index(col)
        return part[col] * scaler.scale_[i] + scaler.mean_[i]

    out = {}
    for name, part in (("real", frame[frame[ag.SOURCE_COL] == ag.REAL]),
                       ("synthetic", frame[frame[ag.SOURCE_COL] == ag.SYNTHETIC])):
        if part.empty:
            continue
        excess = unscale(part, "DEWP") - unscale(part, "TEMP")
        out[name] = {
            "dewp_above_temp_pct": round(float((excess > 0).mean() * 100), 4),
            "dewp_above_temp_n": int((excess > 0).sum()),
            "n": int(len(part)),
        }
    return out


def run(cfg: ag.GanConfig | None = None, *, verbose: bool = True) -> dict:
    import joblib

    cfg = cfg or ag.load_config()
    say = print if verbose else (lambda *a, **k: None)

    bal = ag.class_balance(cfg)
    features = bal["features"]
    train = pd.read_csv(cfg.train_path)
    observed = train[~train["is_imputed_pm25"]]

    say("Reconstructing the ORIGINAL BROKEN CTGAN configuration.")
    say("  * modelled columns: the nine FEATURE columns, standardized")
    say("    " + ", ".join(features))
    say("  * cyclical encodings typed numerical -- the defect being reproduced")
    say("  * DEWP <= TEMP passed as a dict, which SDV ignores")
    say(f"  * {cfg.epochs} epochs, batch {cfg.batch_size}, "
        f"{bal['total_synthetic']:,} synthetic rows across "
        f"{len(bal['minority_classes'])} classes")
    say("  * writes ONE json; no training table, model or result is touched\n")

    frames, timings, outcomes = [], {}, {}
    for i, label in enumerate(bal["minority_classes"], 1):
        need = bal["plan"][label]["n_synthetic"]
        rows = observed.loc[observed["pm25_category"] == label, features]
        say(f"  [{i}/{len(bal['minority_classes'])}] {label}: {len(rows):,} real rows "
            f"-> {need:,} synthetic")
        sampled, fit_s, outcome = _fit_broken(rows.reset_index(drop=True), cfg,
                                              need, verbose)
        sampled = sampled.copy()
        sampled[ag.SOURCE_COL] = ag.SYNTHETIC
        frames.append(sampled)
        timings[label] = round(fit_s / 60, 2)
        outcomes[label] = outcome

    real_part = observed[features].copy()
    real_part[ag.SOURCE_COL] = ag.REAL
    combined = pd.concat([real_part, *frames], ignore_index=True)

    checks = ag.validity_checks(combined, cfg)
    zero_tol = zero_tolerance_dewp(combined, cfg)

    syn = checks["synthetic"]
    regenerated = {
        "on_unit_circle_pct": syn["hour"]["on_unit_circle_pct"],
        "distinct_hour_sin": syn["hour"]["distinct_sin"],
        "dewp_above_temp_pct": zero_tol["synthetic"]["dewp_above_temp_pct"],
        "dewp_above_temp_pct_real": zero_tol["real"]["dewp_above_temp_pct"],
    }

    comparison = {}
    for key, reported in REPORTED.items():
        got, tol = regenerated[key], TOLERANCE[key]
        if key == "distinct_hour_sin":
            delta = abs(got - reported) / reported
            ok, unit = delta <= tol, "relative"
        else:
            delta, unit = abs(got - reported), "absolute"
            ok = delta <= tol
        comparison[key] = {
            "reported_in_gan_quality_report_h6_section_5": reported,
            "regenerated": got,
            "difference": round(float(delta), 4),
            "difference_unit": unit,
            "tolerance": tol,
            "matches": bool(ok),
        }

    payload = {
        "_purpose": (
            "REGENERATED FOR FIGURE PURPOSES ONLY. Reconstructs the original, "
            "deliberately-broken CTGAN configuration (cyclical encodings modelled as "
            "free continuous columns; DEWP <= TEMP passed as a dict SDV ignores) so "
            "figure 04 can show the before column from measured data instead of from "
            "prose. It changes no modelling result and no conclusion; the augmented "
            "training tables, the ablation and every downstream number come from the "
            "FIXED configuration in src/gan/augment.py and are untouched by this file."
        ),
        "_matches_prose_in": "reports/gan_quality_report_h6.md section 5",
        "_not_used_for": ["training", "model selection", "any reported metric"],
        "_caveat": (
            "CTGAN is stochastic and the original run's RNG state was not saved, so "
            "these are a reproduction of the same failure, not the same random draw. "
            "Per-check agreement with the reported numbers is recorded in 'comparison'."
        ),
        "config": {
            "modelled_columns": features,
            "modelled_as": "standardized feature matrix, every column typed numerical",
            "constraint_form": "dict passed to add_constraints (no-op)",
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "horizon": cfg.horizon,
            "classes": bal["minority_classes"],
            "n_synthetic": bal["total_synthetic"],
            "fit_minutes": timings,
            "constraint_outcome": outcomes,
        },
        "validity": checks,
        "validity_zero_tolerance_dewp": zero_tol,
        "regenerated": regenerated,
        "comparison": comparison,
        "all_checks_match": all(c["matches"] for c in comparison.values()),
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epochs; anything below the configured value makes "
                         "the numbers meaningless and is for smoke-testing only")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    cfg = ag.load_config()
    if args.epochs is not None:
        from dataclasses import replace
        cfg = replace(cfg, epochs=args.epochs)
        print(f"!! epochs overridden to {args.epochs}: SMOKE TEST, numbers are not "
              f"comparable to the report\n")

    payload = run(cfg)
    if args.epochs is not None:
        payload["_SMOKE_TEST"] = (
            f"epochs={args.epochs}, not the configured {ag.load_config().epochs}. "
            "These numbers are not a reproduction of anything.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    try:
        shown = args.out.relative_to(REPO_ROOT)
    except ValueError:
        shown = args.out                      # --out may point outside the repo
    print(f"\nwrote {shown}\n")

    print("check                      reported   regenerated   match")
    print("-" * 60)
    for key, c in payload["comparison"].items():
        r, g = c["reported_in_gan_quality_report_h6_section_5"], c["regenerated"]
        mark = "yes" if c["matches"] else "NO"
        print(f"  {key:<24} {r:>8}   {g:>11}   {mark}")
    if payload["all_checks_match"]:
        print("\nAll checks reproduce the numbers already in the report.")
    else:
        off = [k for k, c in payload["comparison"].items() if not c["matches"]]
        print(f"\n** {len(off)} check(s) did NOT reproduce: {', '.join(off)}")
        print("   The figure will carry the regenerated values and say so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
