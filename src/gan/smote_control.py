"""SMOTE control — the second cheap alternative to CTGAN.

Phase 4 spent ~24 minutes of generator training, 82,958 synthetic rows, a quality
report, four validity checks and a constraint-aware redesign on CTGAN augmentation,
and the result significantly degraded both advisory classes. Class weighting (one
hyperparameter) beat it. SMOTE is the other control a reviewer will ask for, and it
had not been run.

Same protocol as everything else: same minority classes CTGAN targeted, same
~50%-of-majority ratio, fit on observed training rows only, validation-based
selection, paired bootstrap against persistence on the same test rows.

    python -m src.gan.smote_control
"""

from __future__ import annotations

import argparse
import json
import time
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
OUT = REPO_ROOT / "reports" / "metrics" / "smote_h6.json"
RARE = ("Very unhealthy", "Hazardous")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    from imblearn.over_sampling import SMOTE

    cfg = bl.load_config()
    labels, k = cfg.labels, len(cfg.labels)
    features, meta = bl.load_feature_columns(cfg)
    train = bl.load_split(cfg, "train", features)
    val = bl.load_split(cfg, "val", features)
    test = bl.load_split(cfg, "test", features)
    mask = test.observed
    y = test.y[mask]

    gan = json.loads((REPO_ROOT / "reports" / "metrics" / "gan_h6.json").read_text())
    plan = gan["balance"]["plan"]
    target_n = gan["balance"]["target_n"]
    minority = gan["balance"]["minority_classes"]
    print(f"CTGAN targeted {minority} up to {target_n:,} each "
          f"({gan['balance']['total_synthetic']:,} synthetic rows, "
          f"{sum(gan['fit_minutes'].values()):.0f} min of generator training)")

    # SMOTE interpolates between real minority neighbours, so it can only be fit on
    # rows whose label is trustworthy -- same observed-only rule CTGAN used.
    obs_tr = train.observed
    Xo, yo = train.X[obs_tr], train.y[obs_tr]
    strategy = {labels.index(l): target_n for l in minority}
    print(f"SMOTE sampling_strategy: "
          f"{ {labels[i]: n for i, n in strategy.items()} }")

    t0 = time.perf_counter()
    Xs, ys = SMOTE(sampling_strategy=strategy, random_state=cfg.seed,
                   k_neighbors=5).fit_resample(Xo, yo)
    smote_seconds = time.perf_counter() - t0
    n_syn = len(ys) - len(yo)
    print(f"SMOTE: {len(yo):,} -> {len(ys):,} rows (+{n_syn:,} synthetic) "
          f"in {smote_seconds:.1f}s")

    # Train on the resampled observed rows plus the untouched imputed-label rows, so
    # the real training set is the same one every other variant saw.
    X_aug = np.vstack([Xs, train.X[~obs_tr]])
    y_aug = np.concatenate([ys, train.y[~obs_tr]])

    raw = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    rp = raw["baseline"]["random_forest"]
    t1 = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=rp["n_estimators"], max_depth=rp["max_depth"],
        min_samples_leaf=rp["min_samples_leaf"], max_features=rp["max_features"],
        class_weight=rp["class_weight"], random_state=cfg.seed, n_jobs=-1,
    ).fit(X_aug, y_aug)
    fit_seconds = time.perf_counter() - t1

    val_f1 = float(f1_score(val.y[val.observed], model.predict(val.X)[val.observed],
                            labels=list(range(k)), average="macro", zero_division=0))
    pred = model.predict(test.X)[mask]
    scores = {
        "macro_f1": float(f1_score(y, pred, labels=list(range(k)), average="macro",
                                   zero_division=0)),
        "accuracy": float((y == pred).mean()),
        "per_class": {l: float(f1_score(y == i, pred == i, zero_division=0))
                      for i, l in enumerate(labels)},
    }
    print(f"val macro-F1 {val_f1:.4f} | test macro-F1 {scores['macro_f1']:.4f}")

    # Bootstrap against persistence and against the two incumbent baselines.
    scaler = joblib.load(cfg.processed_dir / "scaler.pkl")
    persist = bl.persistence_baseline(test, cfg, scaler,
                                      list(meta["scaled_columns"]))[mask]
    refs = {"Persistence": persist}
    plain = RandomForestClassifier(
        n_estimators=rp["n_estimators"], max_depth=rp["max_depth"],
        min_samples_leaf=rp["min_samples_leaf"], max_features=rp["max_features"],
        class_weight=rp["class_weight"], random_state=cfg.seed, n_jobs=-1,
    ).fit(train.X, train.y)
    refs["RandomForest (unweighted)"] = plain.predict(test.X)[mask]
    cw = RandomForestClassifier(
        n_estimators=rp["n_estimators"], max_depth=rp["max_depth"],
        min_samples_leaf=rp["min_samples_leaf"], max_features=rp["max_features"],
        class_weight="balanced", random_state=cfg.seed, n_jobs=-1,
    ).fit(train.X, train.y)
    refs["RandomForest (class_weight=balanced)"] = cw.predict(test.X)[mask]

    tests = {}
    for name, ref in refs.items():
        t = {"macro_f1": paired_bootstrap(y, ref, pred, k, 1000)}
        for lab in RARE:
            i = labels.index(lab)
            t[lab] = paired_bootstrap(
                y, ref, pred, k, 1000,
                scorer=lambda yt, yp, i=i: float(f1_score(yt == i, yp == i,
                                                          zero_division=0)))
        tests[name] = t
        m = t["macro_f1"]
        print(f"  vs {name:38s} {m['observed_diff']:+.4f} "
              f"[{m['ci_low']:+.4f}, {m['ci_high']:+.4f}] "
              f"{'SIG' if m['significant'] else 'ns'}")

    payload = {
        "val_macro_f1": val_f1, "scores": scores, "tests": tests,
        "n_synthetic": int(n_syn), "target_n": int(target_n),
        "minority_classes": minority,
        "cost": {"smote_seconds": round(smote_seconds, 1),
                 "rf_fit_seconds": round(fit_seconds, 1),
                 "ctgan_minutes": round(sum(gan["fit_minutes"].values()), 1),
                 "ctgan_synthetic_rows": int(gan["balance"]["total_synthetic"])},
        "n_test_observed": int(mask.sum()),
    }
    if not args.no_write:
        OUT.write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
