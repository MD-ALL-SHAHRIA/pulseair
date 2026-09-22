"""Compute the bootstrap comparisons the earlier phases did not run.

Three variants were evaluated but never paired-bootstrapped against persistence,
because each phase only tested the model it selected: XGBoost (lost to RF in Phase 3),
the Transformer (lost to the LSTM on validation in Phase 5), and the compressed forest
(Phase 7 reported macro-F1 only).

The master table needs one row per variant with the same significance column, so the
missing tests are computed here with the same 1,000-resample paired bootstrap used
everywhere else. Results are cached to reports/metrics/gapfill_h6.json.

    python -m src.reporting.fill_gaps
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.gan.ablation import paired_bootstrap
from src.models import baseline as bl

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "reports" / "metrics" / "gapfill_h6.json"
RARE = ("Very unhealthy", "Hazardous")


def score(y, pred, labels):
    return {
        "macro_f1": float(f1_score(y, pred, labels=list(range(len(labels))),
                                   average="macro", zero_division=0)),
        "accuracy": float((y == pred).mean()),
        "per_class": {lab: float(f1_score(y == i, pred == i, zero_division=0))
                      for i, lab in enumerate(labels)},
    }


def main() -> int:
    cfg = bl.load_config()
    labels = cfg.labels
    n_cls = len(labels)

    features, meta = bl.load_feature_columns(cfg)
    train = bl.load_split(cfg, "train", features)
    test = bl.load_split(cfg, "test", features)
    mask = test.observed
    y = test.y[mask]
    print(f"test observed rows: {mask.sum():,}")

    # Persistence, the reference every row is measured against.
    scaler = joblib.load(cfg.processed_dir / "scaler.pkl")
    persist = bl.persistence_baseline(test, cfg, scaler, list(meta["scaled_columns"]))[mask]

    preds: dict[str, np.ndarray] = {}

    # --- XGBoost: not saved in Phase 3 (only the selected RF was), so refit. Same
    # --- config, same seed, so this reproduces the Phase 3 model exactly.
    print("refitting XGBoost (Phase 3 config, seed fixed) ...")
    t0 = time.perf_counter()
    xgb = bl.build_models(cfg)["XGBoost"].fit(train.X, train.y)
    preds["XGBoost"] = xgb.predict(test.X)[mask]
    print(f"  {time.perf_counter() - t0:.0f}s")

    # --- Compressed forest: saved by Phase 7.
    comp_path = cfg.artifacts_dir / "baseline_h6_compressed.pkl"
    print(f"loading {comp_path.name} ...")
    comp_bundle = joblib.load(comp_path)
    comp = comp_bundle["model"]
    comp_name = (f"Compressed RF ({comp.n_estimators} trees x depth "
                 f"{comp.max_depth})")
    preds[comp_name] = comp.predict(test.X)[mask]

    # --- The two-sided re-selection winner: class-weighted AND compressed. This is
    # --- the candidate deployed model, so its test-side floor comparison matters.
    cw_comp_path = cfg.artifacts_dir / "baseline_h6_compressed_cw.pkl"
    cw_comp_name = None
    if cw_comp_path.exists():
        b = joblib.load(cw_comp_path)["model"]
        cw_comp_name = (f"Compressed RF, class-weighted ({b.n_estimators} trees x "
                        f"depth {b.max_depth})")
        preds[cw_comp_name] = b.predict(test.X)[mask]
        print(f"loaded {cw_comp_path.name}: {b.n_estimators} x {b.max_depth}")

    # --- Class-weighted RandomForest: the control for "why CTGAN instead of just
    # --- weighting the loss?". Same data, same hyperparameters, same seed; only
    # --- class_weight changes. Selection convention unchanged (validation), and it
    # --- gets the same bootstrap against persistence as every other variant.
    print("fitting class-weighted RandomForest (class_weight='balanced') ...")
    from sklearn.ensemble import RandomForestClassifier
    import yaml
    rf_cfg = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    rp = dict(rf_cfg["baseline"]["random_forest"])
    t0 = time.perf_counter()
    cw = RandomForestClassifier(
        n_estimators=rp["n_estimators"], max_depth=rp["max_depth"],
        min_samples_leaf=rp["min_samples_leaf"], max_features=rp["max_features"],
        class_weight="balanced", random_state=cfg.seed, n_jobs=-1,
    ).fit(train.X, train.y)
    val = bl.load_split(cfg, "val", features)
    cw_val = f1_score(val.y[val.observed], cw.predict(val.X)[val.observed],
                      labels=list(range(n_cls)), average="macro", zero_division=0)
    preds["RandomForest (class_weight=balanced)"] = cw.predict(test.X)[mask]
    print(f"  {time.perf_counter() - t0:.0f}s | val observed macro-F1 {cw_val:.4f}")

    # --- Transformer: NOT recomputed. Phase 5 saved only the selected model (the LSTM
    # --- won validation), so a paired bootstrap would need a full retrain plus 50 MC
    # --- passes -- roughly 12 minutes for a single table cell, and it hung when run
    # --- detached. Its point estimates are read from dl_h6.json instead and the
    # --- master table marks the significance column "not tested" rather than
    # --- implying a comparison that was never made.
    print("Transformer: reusing Phase 5 point estimates, no paired bootstrap "
          "(see module docstring)")

    # Does class weighting clear the floor on VALIDATION? That is the condition the
    # Phase 7 two-sided rule could not satisfy with any unweighted forest, so it
    # decides whether class weighting reopens the deployment path.
    from src.deployment.compress_export import load_config as dload, persistence_pred
    dcfg = dload()
    val_frame = pd.read_csv(dcfg.processed_dir / "tabular_val.csv")
    v_obs = val.observed
    persist_val = persistence_pred(dcfg, val_frame)[v_obs]
    y_val = val.y[v_obs]
    persist_val_f1 = float(f1_score(y_val, persist_val, labels=list(range(n_cls)),
                                    average="macro", zero_division=0))
    cw_val_bt = paired_bootstrap(y_val, persist_val, cw.predict(val.X)[v_obs],
                                 n_cls, 1000)
    print(f"\nvalidation floor check (the Phase 7 condition (b)):")
    print(f"  persistence val macro-F1      {persist_val_f1:.4f}")
    print(f"  class-weighted RF val macro-F1 {cw_val:.4f}")
    print(f"  difference {cw_val_bt['observed_diff']:+.4f} "
          f"[{cw_val_bt['ci_low']:+.4f}, {cw_val_bt['ci_high']:+.4f}] "
          f"{'SIGNIFICANT' if cw_val_bt['significant'] else 'not significant'}")

    out = {"n_test_observed": int(mask.sum()), "n_boot": 1000, "labels": labels,
           "class_weighted_val_macro_f1": float(cw_val),
           "persistence_val_macro_f1": persist_val_f1,
           "class_weighted_vs_persistence_val": cw_val_bt,
           "compressed_name": comp_name,
           "cw_compressed_name": cw_comp_name,
           "compressed_config": {"n_estimators": int(comp.n_estimators),
                                 "max_depth": int(comp.max_depth)},
           "variants": {}}
    for name, pred in preds.items():
        sc = score(y, pred, labels)
        vs = {"macro_f1": paired_bootstrap(y, persist, pred, n_cls, 1000)}
        for lab in RARE:
            i = labels.index(lab)
            vs[lab] = paired_bootstrap(
                y, persist, pred, n_cls, 1000,
                scorer=lambda yt, yp, i=i: float(f1_score(yt == i, yp == i,
                                                          zero_division=0)))
        out["variants"][name] = {"scores": sc, "vs_persistence": vs}
        t = vs["macro_f1"]
        print(f"  {name:26s} macro-F1 {sc['macro_f1']:.4f} | vs persistence "
              f"{t['observed_diff']:+.4f} [{t['ci_low']:+.4f}, {t['ci_high']:+.4f}] "
              f"{'SIG' if t['significant'] else 'ns'}")

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
