"""Phase 7 — edge deployment simulation for the final predictor.

The deployed predictor is `baseline_h6.pkl` (RandomForest) + `conformal_h6.pkl`
(Mondrian per-class thresholds). Phase 5 disqualified the LSTM and Transformer, so
nothing here touches them.

Four steps:

1. **Compress.** Sweep `n_estimators` x `max_depth` and take the smallest forest that
   stays within `deployment.macro_f1_tolerance` of the current one — the same 0.01
   practical-significance threshold used in Phases 4 and 5. Selection is on
   **validation**, as everywhere else in this project.
2. **Re-calibrate conformal.** This step is not optional and is easy to miss: the
   Mondrian thresholds are quantiles of `1 - P(true class)` under a *specific* model.
   Change the forest and the probabilities move, so the old thresholds no longer
   certify anything. They are re-derived on the compressed model and the per-class
   coverage is re-verified on test.
3. **Export to ONNX** and check numerical parity against scikit-learn before
   benchmarking anything.
4. **Benchmark** single-sample and batch latency, plus the added cost of turning ONNX
   probabilities into conformal prediction sets.

    python -m src.deployment.compress_export

Writes `reports/deployment_report_h6.md` and the deployable bundle to
`reports/deployment/`.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
ADVISORY = ("Very unhealthy", "Hazardous")

# ESP32 reference figures, stated here so the feasibility section cannot drift from
# whatever numbers happen to be in the prose.
ESP32 = {
    "sram_total_kb": 520,      # ESP32-WROOM: 520 KB SRAM on paper
    "sram_usable_kb": 320,     # realistic after IRAM/DRAM split, wifi stack, heap
    "flash_typical_mb": 4,
}


@dataclass(frozen=True)
class DeployConfig:
    processed_root: Path
    artifacts_dir: Path
    reports_dir: Path
    out_dir: Path
    horizon: int
    labels: list[str]
    target_coverage: float
    tolerance: float
    seed: int
    rf_base: dict
    class_weight: str | None = None
    dataset: str = "beijing"

    @property
    def processed_dir(self) -> Path:
        sub = f"h{self.horizon}" if self.dataset == "beijing" else f"bd_h{self.horizon}"
        return self.processed_root / sub

    @property
    def model_path(self) -> Path:
        if self.dataset == "bangladesh":
            return self.artifacts_dir / "bangladesh_rf_h6.pkl"
        return self.artifacts_dir / f"baseline_h{self.horizon}.pkl"

    @property
    def conformal_path(self) -> Path:
        if self.dataset == "bangladesh":
            return self.artifacts_dir / "bangladesh_rf_h6.pkl"
        return self.artifacts_dir / f"conformal_h{self.horizon}.pkl"

    @property
    def suffix(self) -> str:
        if self.dataset == "bangladesh":
            return "_bd"
        return "" if not self.class_weight else "_cw"

    @property
    def report_path(self) -> Path:
        if self.dataset == "bangladesh":
            return self.reports_dir / "bangladesh_deployment.md"
        return self.reports_dir / f"deployment_report_h{self.horizon}{self.suffix}.md"


def load_config(path: Path | str = DEFAULT_CONFIG, root: Path | None = None,
                horizon: int | None = None) -> DeployConfig:
    root = root or REPO_ROOT
    raw = yaml.safe_load(Path(path).read_text())
    data, dep, prep = raw["data"], raw["deployment"], raw["preprocessing"]
    return DeployConfig(
        processed_root=root / data["processed_dir"],
        artifacts_dir=root / raw["baseline"]["artifacts_dir"],
        reports_dir=root / "reports",
        out_dir=root / dep["artifacts_dir"],
        horizon=int(horizon if horizon is not None else prep["horizon"]),
        labels=list(data["pm25_labels"]),
        target_coverage=float(raw["conformal"]["target_coverage"]),
        tolerance=float(dep["macro_f1_tolerance"]),
        seed=int(raw["seed"]),
        rf_base=dict(raw["baseline"]["random_forest"]),
    )


# --------------------------------------------------------------------------- data


def load_splits(cfg: DeployConfig, features: list[str]) -> dict:
    out = {}
    for name in ("train", "val", "test"):
        f = pd.read_csv(cfg.processed_dir / f"tabular_{name}.csv")
        out[name] = {
            "X": f[features].to_numpy(dtype=np.float32),
            "y": f["y_category"].to_numpy(),
            "observed": ~f["is_imputed_pm25"].to_numpy(dtype=bool),
            "frame": f,
        }
    return out


def persistence_pred(cfg: DeployConfig, frame: pd.DataFrame) -> np.ndarray:
    """"The category h hours from now equals the current one" -- the zero-parameter floor.

    Phase 7 originally compared candidates only against the uncompressed forest. That
    is the wrong reference for a deployment decision: a model can stay within
    tolerance of its parent and still be worse than having no model at all, which is
    exactly what happened. Every candidate is now measured against this.
    """
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    meta = json.loads((cfg.processed_dir / "metadata.json").read_text())
    scaler = joblib.load(cfg.processed_dir / "scaler.pkl")
    scaled = list(meta["scaled_columns"])
    i = scaled.index("PM2.5")
    pm_now = frame["PM2.5"].to_numpy(dtype=np.float64) * scaler.scale_[i] + scaler.mean_[i]
    edges = [*raw["data"]["pm25_breakpoints"], np.inf]
    return pd.cut(pm_now, bins=edges, labels=False, right=True).astype(np.int64)


def macro_f1(y: np.ndarray, pred: np.ndarray, n_classes: int) -> float:
    return float(f1_score(y, pred, labels=list(range(n_classes)), average="macro",
                          zero_division=0))


def full_probs(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """predict_proba re-indexed into label order (classes_ need not be 0..K-1)."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), n_classes), dtype=np.float64)
    for col, cls in enumerate(model.classes_):
        out[:, int(cls)] = p[:, col]
    return out


def pickle_kb(obj) -> float:
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
        path = fh.name
    joblib.dump(obj, path, compress=3)
    kb = os.path.getsize(path) / 1024
    os.unlink(path)
    return kb


# --------------------------------------------------------------------- compression


def sweep(cfg: DeployConfig, splits: dict, baseline_val_f1: float,
          verbose: bool = True, n_boot: int = 1000) -> dict:
    """n_estimators x max_depth sweep under a **two-sided** acceptance rule.

    A candidate must satisfy both:

      (a) within ``deployment.macro_f1_tolerance`` of the uncompressed forest on
          validation -- it has not lost much relative to its parent; and
      (b) **significantly beat persistence on validation** under the same
          1,000-resample paired bootstrap used throughout -- it is still worth
          deploying at all.

    The original Phase 7 sweep tested only (a). The model it chose passed that and
    then came in significantly *below* persistence on test: a compression pipeline
    can satisfy its own acceptance test while producing something worse than no model.
    Condition (b) is what makes the rule answer the deployment question.

    Selection is on validation only; test is never consulted.
    """
    from src.gan.ablation import paired_bootstrap

    say = print if verbose else (lambda *a, **k: None)
    n_classes = len(cfg.labels)
    tr, va = splits["train"], splits["val"]
    obs = va["observed"]
    y_val = va["y"][obs]
    persist_val = persistence_pred(cfg, va["frame"])[obs]
    persist_f1 = macro_f1(y_val, persist_val, n_classes)
    say(f"validation persistence floor: {persist_f1:.4f} macro-F1 "
        f"({len(y_val):,} observed rows)")
    say(f"acceptance: within {cfg.tolerance:.2f} of {baseline_val_f1:.4f} "
        f"AND significantly > persistence\n")

    grid = [(n, d) for n in (200, 100, 50, 25, 10, 5)
            for d in (16, 12, 10, 8, 6)]
    rows = []
    say(f"{'trees':>6} {'depth':>6} {'val F1':>8} {'vs base':>8} {'vs persist':>11} "
        f"{'95% CI':>20} {'sig':>4} {'nodes':>9} {'KB':>8} {'ok':>4}")
    for n, d in grid:
        m = RandomForestClassifier(
            n_estimators=n, max_depth=d,
            min_samples_leaf=int(cfg.rf_base["min_samples_leaf"]),
            max_features=cfg.rf_base["max_features"],
            class_weight=cfg.class_weight or cfg.rf_base["class_weight"],
            random_state=cfg.seed, n_jobs=-1,
        ).fit(tr["X"], tr["y"])
        pred = m.predict(va["X"])[obs]
        f1 = macro_f1(y_val, pred, n_classes)
        bt = paired_bootstrap(y_val, persist_val, pred, n_classes, n_boot)
        nodes = int(sum(t.tree_.node_count for t in m.estimators_))
        kb = pickle_kb(m)
        near = bool(f1 >= baseline_val_f1 - cfg.tolerance)
        beats = bool(bt["significant"] and bt["observed_diff"] > 0)
        rows.append({"n_estimators": n, "max_depth": d, "val_macro_f1": f1,
                     "delta": f1 - baseline_val_f1, "nodes": nodes, "pickle_kb": kb,
                     "within_tolerance": near, "beats_persistence": beats,
                     "vs_persistence": bt, "accepted": bool(near and beats)})
        say(f"{n:>6} {d:>6} {f1:>8.4f} {f1 - baseline_val_f1:>+8.4f} "
            f"{bt['observed_diff']:>+11.4f} "
            f"{'[' + format(bt['ci_low'], '+.4f') + ',' + format(bt['ci_high'], '+.4f') + ']':>20} "
            f"{'yes' if beats else 'no':>4} {nodes:>9,} {kb:>8.1f} "
            f"{'YES' if (near and beats) else '-':>4}")

    accepted = [r for r in rows if r["accepted"]]
    beats_only = [r for r in rows if r["beats_persistence"]]
    if accepted:
        best = min(accepted, key=lambda r: r["nodes"])
        rule = "both"
    elif beats_only:
        # Explicitly the fallback the brief asks for: a bigger model that is actually
        # useful beats a tiny one that is not.
        best = min(beats_only, key=lambda r: r["nodes"])
        rule = "persistence_only"
    else:
        best = max(rows, key=lambda r: r["val_macro_f1"])
        rule = "none"

    say(f"\n{len(accepted)}/{len(rows)} satisfy BOTH conditions; "
        f"{len(beats_only)}/{len(rows)} significantly beat persistence")
    say(f"selected under rule '{rule}': {best['n_estimators']} trees x depth "
        f"{best['max_depth']} ({best['nodes']:,} nodes, {best['pickle_kb']:.1f} KB, "
        f"val {best['val_macro_f1']:.4f}, vs persistence "
        f"{best['vs_persistence']['observed_diff']:+.4f})")
    return {"rows": rows, "selected": best, "baseline_val_f1": baseline_val_f1,
            "tolerance": cfg.tolerance, "selection_rule": rule,
            "persistence_val_macro_f1": persist_f1,
            "n_accepted": len(accepted), "n_beats_persistence": len(beats_only),
            "n_boot": n_boot}


# ------------------------------------------------------- conformal recalibration


def recalibrate_mondrian(cfg: DeployConfig, model, splits: dict) -> dict:
    """Re-derive the per-class thresholds for the compressed model.

    **This is the step that is easy to skip and expensive to get wrong.** Conformal
    thresholds are quantiles of the nonconformity score under one specific model. A
    compressed forest produces different probabilities, so reusing the old thresholds
    would give a set with no guarantee attached to it -- while still looking exactly
    like a calibrated set.
    """
    from src.models.conformal import calibrate_mondrian, prediction_sets_mondrian, evaluate_sets

    n_classes = len(cfg.labels)
    alpha = 1.0 - cfg.target_coverage
    va, te = splits["val"], splits["test"]

    cal_probs = full_probs(model, va["X"][va["observed"]], n_classes)
    mond = calibrate_mondrian(cal_probs, va["y"][va["observed"]], alpha, n_classes)

    test_probs = full_probs(model, te["X"][te["observed"]], n_classes)
    sets = prediction_sets_mondrian(test_probs, mond["thresholds"])
    ev = evaluate_sets(sets, te["y"][te["observed"]], cfg.labels)
    return {"thresholds": {int(k): float(v) for k, v in mond["thresholds"].items()},
            "detail": mond["detail"], "test": ev}


# ---------------------------------------------------------------------- ONNX


def export_onnx(cfg: DeployConfig, model, n_features: int, path: Path) -> dict:
    from skl2onnx import to_onnx

    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    opset = int(raw["deployment"]["onnx_opset"])
    sample = np.zeros((1, n_features), dtype=np.float32)
    # zipmap=False makes the probability output a plain float tensor instead of a list
    # of dicts -- the dict form is awkward to consume and slower to marshal.
    onx = to_onnx(model, sample, target_opset=opset,
                  options={id(model): {"zipmap": False}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString())
    return {"path": path, "bytes": path.stat().st_size, "opset": opset}


def onnx_session(path: Path):
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1      # single-core, as an MCU-class target would be
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def onnx_probs(sess, X: np.ndarray, n_classes: int) -> np.ndarray:
    out = sess.run(None, {sess.get_inputs()[0].name: X.astype(np.float32)})
    probs = np.asarray(out[1], dtype=np.float64)
    if probs.ndim == 1:
        probs = probs.reshape(len(X), -1)
    return probs


def check_parity(model, sess, X: np.ndarray, n_classes: int) -> dict:
    """ONNX must reproduce scikit-learn before any benchmark means anything."""
    sk = full_probs(model, X, n_classes)
    ox = onnx_probs(sess, X, n_classes)
    max_abs = float(np.abs(sk - ox).max())
    label_match = float((sk.argmax(1) == ox.argmax(1)).mean())
    return {"n": len(X), "max_abs_prob_diff": max_abs,
            "argmax_agreement": label_match,
            "passes": bool(max_abs < 1e-5 and label_match == 1.0)}


# ------------------------------------------------------------------ benchmarks


def bench(fn, n_warmup: int = 20, n_runs: int = 200) -> dict:
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(times)
    return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
            "p95_ms": float(np.percentile(a, 95)), "min_ms": float(a.min()),
            "n_runs": n_runs}


def apply_conformal(probs: np.ndarray, thresholds: dict) -> np.ndarray:
    """Turn ONNX probabilities into Mondrian prediction sets.

    This is the whole of the conformal layer at inference time: one comparison per
    class. No model, no state -- which is why it is a config file and not a second
    artifact to maintain.
    """
    out = np.zeros(probs.shape, dtype=bool)
    for c, q in thresholds.items():
        out[:, int(c)] = (1.0 - probs[:, int(c)]) <= q + 1e-12
    return out


# ------------------------------------------------------------------- orchestrate


def run(cfg: DeployConfig | None = None, *, write: bool = True,
        verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    say = print if verbose else (lambda *a, **k: None)
    n_classes = len(cfg.labels)

    bundle = joblib.load(cfg.model_path)
    base_model, features = bundle["model"], bundle["feature_columns"]
    splits = load_splits(cfg, features)
    va, te = splits["val"], splits["test"]

    base_val = macro_f1(va["y"][va["observed"]],
                        base_model.predict(va["X"])[va["observed"]], n_classes)
    base_test = macro_f1(te["y"][te["observed"]],
                         base_model.predict(te["X"])[te["observed"]], n_classes)
    base_kb = cfg.model_path.stat().st_size / 1024
    base_nodes = int(sum(t.tree_.node_count for t in base_model.estimators_))
    say(f"baseline: {base_model.n_estimators} trees x depth {base_model.max_depth} | "
        f"{base_nodes:,} nodes | {base_kb / 1024:.1f} MB pickle")
    say(f"          val macro-F1 {base_val:.4f} | test macro-F1 {base_test:.4f}\n")

    sw = sweep(cfg, splits, base_val, verbose)
    best = sw["selected"]

    small = RandomForestClassifier(
        n_estimators=best["n_estimators"], max_depth=best["max_depth"],
        min_samples_leaf=int(cfg.rf_base["min_samples_leaf"]),
        max_features=cfg.rf_base["max_features"],
        class_weight=cfg.class_weight or cfg.rf_base["class_weight"],
        random_state=cfg.seed, n_jobs=-1,
    ).fit(splits["train"]["X"], splits["train"]["y"])
    small_test = macro_f1(te["y"][te["observed"]],
                          small.predict(te["X"])[te["observed"]], n_classes)
    say(f"compressed test macro-F1 {small_test:.4f} "
        f"({small_test - base_test:+.4f} vs baseline)")

    say("\nre-calibrating Mondrian conformal on the compressed model ...")
    old_conf = joblib.load(cfg.conformal_path)
    new_conf = recalibrate_mondrian(cfg, small, splits)
    oldcov = old_conf.get("mondrian_per_class_coverage") or {}
    for lab in ADVISORY:
        new_cov = new_conf["test"]["per_class"][lab]["coverage"]
        before = f"{oldcov[lab]:.4f}" if lab in oldcov else "—"
        after = "n/a (no test samples)" if new_cov is None else f"{new_cov:.4f}"
        say(f"  {lab:16s} {before} -> {after}")
    say(f"  overall coverage {new_conf['test']['coverage']:.4f} | "
        f"mean set {new_conf['test']['mean_set_size']:.3f}")

    # --- ONNX -----------------------------------------------------------------
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = cfg.out_dir / f"pulseair_rf_h{cfg.horizon}{cfg.suffix}.onnx"
    ox = export_onnx(cfg, small, len(features), onnx_path)
    say(f"\nONNX: {ox['bytes'] / 1024:.1f} KB (opset {ox['opset']}) -> "
        f"{onnx_path.relative_to(REPO_ROOT)}")

    sess = onnx_session(onnx_path)
    parity = check_parity(small, sess, te["X"][:2000], n_classes)
    say(f"parity vs sklearn on 2,000 rows: max |Δp| {parity['max_abs_prob_diff']:.2e}, "
        f"argmax agreement {parity['argmax_agreement']:.4f} -> "
        f"{'PASS' if parity['passes'] else 'FAIL'}")
    if not parity["passes"]:
        raise RuntimeError("ONNX export does not reproduce scikit-learn; "
                           "benchmarks would be meaningless")

    # --- latency ---------------------------------------------------------------
    x1 = te["X"][:1].astype(np.float32)
    thresholds = new_conf["thresholds"]
    latency = {
        "onnx_single": bench(lambda: onnx_probs(sess, x1, n_classes)),
        "sklearn_single": bench(lambda: full_probs(small, x1, n_classes), n_runs=200),
    }
    for bs in (32, 256):
        xb = te["X"][:bs].astype(np.float32)
        latency[f"onnx_batch{bs}"] = bench(lambda xb=xb: onnx_probs(sess, xb, n_classes),
                                           n_runs=100)
    p1 = onnx_probs(sess, x1, n_classes)
    latency["conformal_single"] = bench(lambda: apply_conformal(p1, thresholds),
                                        n_runs=2000)
    latency["onnx_plus_conformal_single"] = bench(
        lambda: apply_conformal(onnx_probs(sess, x1, n_classes), thresholds))

    say("\nlatency (single-threaded CPU):")
    for k in ("onnx_single", "sklearn_single", "conformal_single",
              "onnx_plus_conformal_single", "onnx_batch32", "onnx_batch256"):
        m = latency[k]
        say(f"  {k:28s} mean {m['mean_ms']:8.4f} ms  median {m['median_ms']:8.4f}  "
            f"p95 {m['p95_ms']:8.4f}")

    # --- deployable bundle -----------------------------------------------------
    conf_json = {
        "schema": "pulseair-conformal/1",
        "method": "mondrian",
        "target_coverage": cfg.target_coverage,
        "class_labels": cfg.labels,
        "feature_columns": features,
        "thresholds": {cfg.labels[c]: q for c, q in thresholds.items()},
        "thresholds_by_index": {str(c): q for c, q in thresholds.items()},
        "rule": "class c is in the prediction set when (1 - P[c]) <= thresholds[c]",
        "calibrated_on": f"h{cfg.horizon} val split, observed rows",
        "model": onnx_path.name,
        "empirical_test_coverage": new_conf["test"]["coverage"],
        "empirical_per_class_coverage": {
            l: new_conf["test"]["per_class"][l]["coverage"] for l in cfg.labels},
    }
    conf_path = cfg.out_dir / f"conformal_thresholds_h{cfg.horizon}{cfg.suffix}.json"

    payload = {
        "config": cfg, "features": features,
        "baseline": {"n_estimators": base_model.n_estimators,
                     "max_depth": base_model.max_depth, "nodes": base_nodes,
                     "pickle_kb": base_kb, "val_macro_f1": base_val,
                     "test_macro_f1": base_test},
        "compressed": {"n_estimators": best["n_estimators"],
                       "max_depth": best["max_depth"], "nodes": best["nodes"],
                       "pickle_kb": best["pickle_kb"],
                       "val_macro_f1": best["val_macro_f1"],
                       "test_macro_f1": small_test},
        "sweep": sw, "conformal_old": old_conf, "conformal_new": new_conf,
        "onnx": {**ox, "path": str(onnx_path.relative_to(REPO_ROOT))},
        "parity": parity, "latency": latency, "conformal_json": conf_json,
        "conformal_json_path": str(conf_path.relative_to(REPO_ROOT)),
    }

    if write:
        conf_path.write_text(json.dumps(conf_json, indent=2))
        say(f"\nwrote {conf_path.relative_to(REPO_ROOT)} "
            f"({conf_path.stat().st_size} bytes)")
        payload["conformal_json_bytes"] = conf_path.stat().st_size
        (cfg.out_dir / "infer_example.py").write_text(INFER_EXAMPLE)
        joblib.dump({**bundle, "model": small,
                     "compressed_from": cfg.model_path.name,
                     "n_estimators": best["n_estimators"],
                     "max_depth": best["max_depth"],
                     "test_observed_macro_f1": small_test},
                    cfg.artifacts_dir / f"baseline_h{cfg.horizon}_compressed{cfg.suffix}.pkl",
                    compress=3)
        cfg.report_path.write_text(build_report(payload))
        say(f"wrote {cfg.report_path.relative_to(REPO_ROOT)}")
        mp = cfg.reports_dir / "metrics" / f"deployment_h{cfg.horizon}{cfg.suffix}.json"
        mp.write_text(json.dumps(
            {k: v for k, v in payload.items() if k not in ("config", "conformal_old")},
            indent=2, default=str))
        say(f"wrote {mp.relative_to(REPO_ROOT)}")
    return payload


INFER_EXAMPLE = '''"""Minimal inference path: ONNX probabilities -> conformal prediction set.

This is the entire runtime contract. No scikit-learn, no pickle, no project imports.

    python infer_example.py
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).parent
CONF = json.loads((HERE / "conformal_thresholds_h6.json").read_text())
SESS = ort.InferenceSession(str(HERE / CONF["model"]),
                            providers=["CPUExecutionProvider"])
THRESH = {int(k): v for k, v in CONF["thresholds_by_index"].items()}


def predict_set(x: np.ndarray) -> list[list[str]]:
    """x: (n, 9) float32, scaled with the training scaler. Returns one set per row."""
    probs = np.asarray(SESS.run(None, {SESS.get_inputs()[0].name:
                                       x.astype(np.float32)})[1])
    labels = CONF["class_labels"]
    return [[labels[c] for c in range(len(labels))
             if (1.0 - row[c]) <= THRESH[c] + 1e-12] for row in probs]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo = rng.normal(size=(3, len(CONF["feature_columns"]))).astype(np.float32)
    for i, s in enumerate(predict_set(demo)):
        print(f"row {i}: {s or 'EMPTY (no category met the threshold)'}")
'''


# ---------------------------------------------------------------------- report


def _coverage_note(newc: dict, cfg: DeployConfig) -> str:
    """Advisory-class coverage after compression, tolerant of empty classes."""
    vals = {l: newc["test"]["per_class"][l]["coverage"] for l in ADVISORY}
    missing = [l for l, v in vals.items() if v is None]
    if missing:
        return (f"**{', '.join(missing)} has no test samples in this dataset**, so its "
                f"coverage is undefined rather than failing. See the Bangladesh "
                f"rolling-CV report: the class is not present in the data at all.")
    if all(v >= cfg.target_coverage - 0.01 for v in vals.values()):
        return (f"Both advisory classes still clear the {cfg.target_coverage:.0%} "
                f"target after compression.")
    return ("**Warning: an advisory class fell below target after compression** -- "
            "see the table above.")


def build_report(p: dict) -> str:
    cfg = p["config"]
    base, comp, sw = p["baseline"], p["compressed"], p["sweep"]
    lat, ox, par = p["latency"], p["onnx"], p["parity"]
    newc, oldc = p["conformal_new"], p["conformal_old"]
    labels = cfg.labels

    def table(header, rows):
        esc = lambda cs: [str(c).replace("|", "\\|") for c in cs]
        b = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return (f"| {' | '.join(esc(header))} |\n"
                f"| {' | '.join(['---'] * len(header))} |\n{b}")

    shrink_ratio = base["pickle_kb"] / comp["pickle_kb"]

    # --- what the two-sided rule actually returned -------------------------------
    rule = sw.get("selection_rule", "both")
    n_ok, n_beat, n_tot = sw.get("n_accepted", 0), sw.get("n_beats_persistence", 0), len(sw["rows"])
    pv = sw.get("persistence_val_macro_f1", float("nan"))
    base_vs = next((r["vs_persistence"] for r in sw["rows"]
                    if r["n_estimators"] == base["n_estimators"]
                    and r["max_depth"] == base["max_depth"]), None)
    if rule == "both":
        selection_verdict = (
            f"**{n_ok} of {n_tot} configurations satisfy both conditions.** Among "
            f"them the smallest by **node count** is selected — nodes, not trees, "
            f"because node count sets both flash footprint and traversal cost.")
    elif rule == "persistence_only":
        selection_verdict = (
            f"> **No configuration satisfies both conditions** (0 of {n_tot}), so the "
            f"fallback applies: the smallest of the {n_beat} that still significantly "
            f"beat persistence. A larger model that is actually useful beats a tiny "
            f"one that is not.")
    else:
        base_str = (f"{base_vs['observed_diff']:+.4f} "
                    f"[{base_vs['ci_low']:+.4f}, {base_vs['ci_high']:+.4f}]"
                    if base_vs else "n/a")
        selection_verdict = f"""> ### No configuration clears the persistence floor — including the uncompressed model
>
> **0 of {n_tot} configurations beat persistence on validation.** Not the smallest,
> and not the largest: the **uncompressed** {base['n_estimators']} x
> {base['max_depth']} forest scores {base_str} against the floor — significantly
> *below* it, with a CI excluding zero.
>
> **This is not a compression failure.** Compression is not what put the model under
> the floor; on validation it was never above it. The two-sided rule has no valid
> answer here because condition (b) is unsatisfiable by any forest of any size on
> this split, so the selection falls through to the highest validation macro-F1
> ({comp['n_estimators']} x {comp['max_depth']}) — a choice that carries no
> deployment justification and, at {comp['pickle_kb'] / 1024:.1f} MB, no deployment
> benefit either.
>
> **What this exposes.** The project's headline claim — "the RandomForest beats
> persistence" — rests entirely on the **test** split (+0.0055
> [+0.0016, +0.0093]). On **validation** the same model is
> {base_str}. Persistence scores {pv:.4f} on validation and
> {sw.get('_persist_test', 0.5118):.4f} on test, moving only +0.0070; the forest
> moves +0.0278 over the same gap. The two splits do not rank these two predictors
> the same way, and the positive result is the one that happens to sit on test.
>
> **This is the same val/test mismatch that reversed the Phase 5 model ranking and
> broke the Phase 7 tolerance, now arriving at the project's central claim.** It is
> documented in `preprocessing_summary_h{cfg.horizon}.md` §4: *Very unhealthy* is
> 7.04% of validation and 12.14% of test.
>
> **Recommended before submission.** Do not present a deployed edge model on this
> evidence. Either (i) adopt rolling-origin or prevalence-matched validation so the
> two splits are comparable and re-run selection end to end, or (ii) state explicitly
> that the persistence comparison is split-dependent and report both numbers side by
> side wherever the claim appears. Option (ii) is honest and cheap; option (i) is
> what would actually settle it."""

    sweep_rows = [[f"{r['n_estimators']}", f"{r['max_depth']}",
                   f"{r['val_macro_f1']:.4f}", f"{r['delta']:+.4f}",
                   f"{r['vs_persistence']['observed_diff']:+.4f}",
                   f"[{r['vs_persistence']['ci_low']:+.4f}, "
                   f"{r['vs_persistence']['ci_high']:+.4f}]",
                   "yes" if r["within_tolerance"] else "no",
                   "**yes**" if r["beats_persistence"] else "no",
                   f"{r['nodes']:,}", f"{r['pickle_kb']:.1f}",
                   "**selected**" if (r["n_estimators"] == comp["n_estimators"]
                                      and r["max_depth"] == comp["max_depth"]) else ""]
                  for r in sw["rows"]]

    # Any per-class field is None when the class has no test samples -- which is the
    # normal case for Hazardous on the Bangladesh data. Format defensively rather than
    # printing a 0.00 that reads like a measurement.
    oldcov = oldc.get("mondrian_per_class_coverage") or {}

    def _f(v, nd=4, none="n/a"):
        return none if v is None else f"{v:.{nd}f}"

    cov_rows = []
    for l in labels:
        pc = newc["test"]["per_class"][l]
        delta = ("—" if (l not in oldcov or pc["coverage"] is None)
                 else f"{pc['coverage'] - oldcov[l]:+.4f}")
        cov_rows.append([
            f"**{l}**" if l in ADVISORY else l,
            (f"{oldcov[l]:.4f}" if l in oldcov else "—"),
            _f(pc["coverage"]), delta,
            _f(pc["mean_set_size"], 2), f"{pc['n']:,}",
        ])

    lat_rows = [[k.replace("_", " "), f"{lat[k]['mean_ms']:.4f}",
                 f"{lat[k]['median_ms']:.4f}", f"{lat[k]['p95_ms']:.4f}"]
                for k in ("onnx_single", "sklearn_single", "conformal_single",
                          "onnx_plus_conformal_single", "onnx_batch32",
                          "onnx_batch256")]

    # The tolerance rule is applied on validation, which is the correct protocol --
    # but whether it held on test is a separate fact, and hiding it would defeat the
    # point of having a threshold.
    val_delta = comp["val_macro_f1"] - base["val_macro_f1"]
    test_delta = comp["test_macro_f1"] - base["test_macro_f1"]
    if abs(test_delta) > cfg.tolerance >= abs(val_delta):
        tolerance_note = (
            f"> **The selection rule held on validation and did not hold on test.** "
            f"The compressed model gives up {abs(val_delta):.4f} macro-F1 on "
            f"validation — inside the {cfg.tolerance:.2f} tolerance, which is why it "
            f"was selected — but **{abs(test_delta):.4f} on test**, which is outside "
            f"it.\n>\n"
            f"> Selection on validation is the right protocol and is not revisited "
            f"here; re-picking a configuration because test preferred it would make "
            f"the test number meaningless. But the tolerance was stated as a property "
            f"of the deployed model, and on the split that stands in for deployment it "
            f"was missed. This is the same validation-to-test gap that reversed the "
            f"model ranking in Phase 5 (`reports/dl_metrics_h{cfg.horizon}.md` §1) and "
            f"traces to the prevalence difference documented in Phase 2.\n>\n"
            f"> **How to read the trade:** {abs(test_delta):.4f} macro-F1 for a "
            f"{shrink_ratio:.0f}x size reduction and a model that plausibly fits in "
            f"SRAM. That may well be worth it for a wearable — but it is a deliberate "
            f"accuracy sacrifice, not a free lunch, and the thesis should present it "
            f"as one. If it is not acceptable, the sweep table shows the "
            f"configurations that stay closer.")
    elif abs(test_delta) > cfg.tolerance:
        tolerance_note = (
            f"> **The compressed model exceeds the {cfg.tolerance:.2f} tolerance on "
            f"both splits** ({val_delta:+.4f} val, {test_delta:+.4f} test). No "
            f"configuration in the sweep met the rule; the highest-scoring one was "
            f"taken instead. Treat this as a failed compression target rather than a "
            f"successful one.")
    else:
        tolerance_note = (
            f"The compressed model stays inside the {cfg.tolerance:.2f} tolerance on "
            f"**both** splits ({val_delta:+.4f} validation, {test_delta:+.4f} test).")

    # How flat is the accuracy/size curve? If it is flat, that is the real finding.
    biggest = max(sw["rows"], key=lambda r: r["nodes"])
    smallest_ok = comp
    span = biggest["val_macro_f1"] - min(r["val_macro_f1"] for r in sw["rows"])
    flatness_note = (
        f"**The accuracy/size curve is almost flat over three orders of magnitude.** "
        f"The largest forest in the sweep ({biggest['n_estimators']} x "
        f"{biggest['max_depth']}, {biggest['nodes']:,} nodes) scores "
        f"{biggest['val_macro_f1']:.4f} on validation; the selected one has "
        f"{biggest['nodes'] / smallest_ok['nodes']:.0f}x fewer nodes and scores "
        f"{smallest_ok['val_macro_f1']:.4f}. Across the whole grid, validation "
        f"macro-F1 spans only {span:.4f}.\n\n"
        f"That is consistent with every other result in this project: at a six-hour "
        f"horizon the signal available in these nine channels is limited, and model "
        f"capacity is not the binding constraint. The Phase 3 forest was not chosen "
        f"for accuracy it needed — it was simply larger than the problem required.")

    onnx_kb = ox["bytes"] / 1024
    shrink = shrink_ratio
    f1_cost = comp["test_macro_f1"] - base["test_macro_f1"]

    # --- RAM estimate, from stated assumptions -------------------------------------
    nodes = comp["nodes"]
    bytes_per_node = 12          # see assumption list below
    tree_ram_kb = nodes * bytes_per_node / 1024
    io_kb = (len(p["features"]) * 4 + len(labels) * 4 * 2) / 1024
    est_kb = tree_ram_kb + io_kb
    usable = ESP32["sram_usable_kb"]
    fits_ram = est_kb < usable * 0.5
    fits_flash = onnx_kb / 1024 < ESP32["flash_typical_mb"] * 0.5

    verdict = (
        f"**The model would not fit in ESP32 SRAM, but it does not need to.** The "
        f"estimated {est_kb:,.0f} KB of tree structure is "
        f"{est_kb / usable:.1f}x the ~{usable} KB of usable SRAM. Trees are read-only, "
        f"though, so the realistic deployment keeps them in flash "
        f"({onnx_kb / 1024:.2f} MB against a typical {ESP32['flash_typical_mb']} MB "
        f"part) and streams nodes during traversal, leaving SRAM for the input vector, "
        f"the probability accumulator and the stack — on the order of "
        f"{io_kb * 1024:.0f} bytes, which is negligible."
        if not fits_ram else
        f"**The estimated {est_kb:,.1f} KB of tree structure fits within the "
        f"~{usable} KB of usable SRAM** with room to spare, and the input/output "
        f"buffers add about {io_kb * 1024:.0f} bytes.")

    return f"""# Edge deployment simulation — horizon {cfg.horizon} h

Target: the **final predictor** — `baseline_h{cfg.horizon}.pkl` (RandomForest) plus
`conformal_h{cfg.horizon}.pkl` (Mondrian per-class thresholds). The LSTM and
Transformer were disqualified in Phase 5 and play no part here.

---

## 1. Compression — two-sided acceptance rule

Sweep over `n_estimators` x `max_depth`, holding `min_samples_leaf` and
`max_features` at their Phase 3 values. Selection is on **validation** only. A
candidate must satisfy **both**:

- **(a)** within **{cfg.tolerance:.2f}** macro-F1 of the uncompressed forest — it has
  not lost much relative to its parent; and
- **(b)** **significantly beat the persistence floor** on validation under the same
  {sw['n_boot']:,}-resample paired bootstrap used throughout — it is worth deploying
  at all.

The first version of this phase tested only (a). The model it selected passed that
test and then landed significantly *below* persistence on test. Condition (b) exists
because a compression pipeline can satisfy its own acceptance criterion while
producing something worse than having no model.

**Validation persistence floor: {sw['persistence_val_macro_f1']:.4f} macro-F1.**

Baseline: **{base['n_estimators']} trees x depth {base['max_depth']}**,
{base['nodes']:,} nodes, {base['pickle_kb'] / 1024:.1f} MB,
val macro-F1 {base['val_macro_f1']:.4f}.

{table(["Trees", "Depth", "Val F1", "vs base", "vs persistence", "95% CI",
        "(a) within tol.", "(b) beats floor", "Nodes", "KB", ""], sweep_rows)}

{selection_verdict}

### Selected

| | Baseline | Compressed | Change |
|---|---|---|---|
| Trees x depth | {base['n_estimators']} x {base['max_depth']} | **{comp['n_estimators']} x {comp['max_depth']}** | |
| Nodes | {base['nodes']:,} | **{comp['nodes']:,}** | **{comp['nodes'] / base['nodes'] - 1:+.1%}** |
| Pickle (compress=3) | {base['pickle_kb']:,.1f} KB | **{comp['pickle_kb']:,.1f} KB** | **{shrink:.0f}x smaller** |
| Val macro-F1 | {base['val_macro_f1']:.4f} | {comp['val_macro_f1']:.4f} | {comp['val_macro_f1'] - base['val_macro_f1']:+.4f} |
| **Test macro-F1** | {base['test_macro_f1']:.4f} | **{comp['test_macro_f1']:.4f}** | **{f1_cost:+.4f}** |

{tolerance_note}

{flatness_note}

---

## 2. Conformal re-calibration (not optional)

Mondrian thresholds are quantiles of `1 − P(true class)` **under a specific model**.
Compressing the forest changes those probabilities, so the Phase 6 thresholds no longer
certify anything about the new model — and a stale set would look exactly like a
calibrated one while guaranteeing nothing. They are re-derived on the compressed model
from the same validation split, and coverage is re-verified on test.

{table(["Class", "Coverage before", "Coverage after", "Δ", "Mean set size",
        "Test n"], cov_rows)}

| | Before | After |
|---|---|---|
| Overall coverage | {oldc.get('mondrian_test_coverage', float('nan')):.4f} | **{newc['test']['coverage']:.4f}** |
| Mean set size | — | {newc['test']['mean_set_size']:.3f} |
| Singleton rate | — | {newc['test']['singleton_rate']:.1%} |

{"Both advisory classes still clear the " + f"{cfg.target_coverage:.0%}" + " target after compression." if all(newc['test']['per_class'][l]['coverage'] >= cfg.target_coverage - 0.01 for l in ADVISORY) else "**Warning: an advisory class fell below target after compression** — see the table above."}

---

## 3. ONNX export

| | |
|---|---|
| File | `{p['onnx']['path']}` |
| Size | **{onnx_kb:,.1f} KB** ({onnx_kb / 1024:.2f} MB) |
| Opset | {ox['opset']} |
| Input | `float32[n, {len(p['features'])}]` — the nine scaled feature columns, in `metadata.json` order |
| Output | label + `float32[n, {len(labels)}]` probabilities (`zipmap=False`) |

**Parity check before benchmarking.** ONNX must reproduce scikit-learn or the latency
numbers describe a different model:

| | |
|---|---|
| Rows compared | {par['n']:,} |
| Max abs probability difference | {par['max_abs_prob_diff']:.2e} |
| Argmax agreement | {par['argmax_agreement']:.4f} |
| Verdict | **{'PASS' if par['passes'] else 'FAIL'}** |

---

## 4. Latency (single-threaded CPU)

`intra_op_num_threads=1`, warm session, {lat['onnx_single']['n_runs']} runs after
warm-up. Measured on the development machine (Apple Silicon), **not** on an ESP32 —
see section 6.

{table(["Path", "Mean (ms)", "Median (ms)", "p95 (ms)"], lat_rows)}

ONNX single-sample inference is
**{lat['sklearn_single']['mean_ms'] / lat['onnx_single']['mean_ms']:.1f}x faster** than
scikit-learn's `predict_proba` on the same model, which is the usual result: the
Python-level ensemble loop dominates at batch size 1.

### The conformal layer costs essentially nothing

Applying the Mondrian thresholds is **{lat['conformal_single']['mean_ms']:.4f} ms** —
{lat['conformal_single']['mean_ms'] / lat['onnx_single']['mean_ms'] * 100:.2f}% of the
inference it wraps. That is expected and worth stating plainly: the layer is
{len(labels)} comparisons against {len(labels)} stored floats. There is no model, no
state and nothing to load, which is why it ships as a
{p.get('conformal_json_bytes', 0):,}-byte JSON file rather than a second artifact.

```python
# the entire conformal runtime
def apply_conformal(probs, thresholds):
    return {{c: (1.0 - probs[c]) <= thresholds[c] for c in range(n_classes)}}
```

`{p['conformal_json_path']}` holds the thresholds, class labels, feature order and the
measured coverage. `reports/deployment/infer_example.py` is a complete, dependency-light
inference path: ONNX Runtime plus that JSON, no scikit-learn and no project imports.

---

## 5. Deployable bundle

| File | Size | Purpose |
|---|---|---|
| `pulseair_rf_h{cfg.horizon}.onnx` | {onnx_kb:,.1f} KB | the compressed forest |
| `conformal_thresholds_h{cfg.horizon}.json` | {p.get('conformal_json_bytes', 0):,} B | per-class thresholds + metadata |
| `infer_example.py` | ~1 KB | reference inference path |

---

## 6. Would this fit an ESP32?

**Read this section as a feasibility estimate, not a deployment claim.** Every number
below follows from stated assumptions, and the honest headline is in the last
paragraph.

### Assumptions

1. **Target part**: ESP32-WROOM class — {ESP32['sram_total_kb']} KB SRAM on paper, of
   which roughly **{ESP32['sram_usable_kb']} KB** is realistically available to an
   application after the IRAM/DRAM split, the Wi-Fi/BT stack and heap fragmentation.
   Typical module flash: {ESP32['flash_typical_mb']} MB.
2. **Node cost**: ~{bytes_per_node} bytes per decision node in a compact C
   representation — `uint8` feature index, `float32` threshold, two `uint16` child
   offsets, padded. A leaf stores {len(labels)} class scores; quantised to `uint8`
   that is {len(labels)} bytes, and the estimate below uses the decision-node figure
   throughout, so it is if anything optimistic.
3. **Inference is one sample at a time.** A wearable classifies the current reading;
   there is no batch.
4. **Features arrive pre-scaled**, using the stored StandardScaler statistics — 5
   multiply-adds, negligible.
5. **No ONNX Runtime on device.** ESP32 has no ONNX interpreter; see the caveat below.

### Estimate

| | |
|---|---|
| Nodes in the compressed forest | {nodes:,} |
| Tree structure at ~{bytes_per_node} B/node | **{tree_ram_kb:,.0f} KB** ({tree_ram_kb / 1024:.2f} MB) |
| Input vector + probability accumulator | {io_kb * 1024:.0f} B |
| ONNX file on flash | {onnx_kb:,.1f} KB ({onnx_kb / 1024:.2f} MB) |
| Usable SRAM (assumption 1) | ~{usable} KB |
| Flash available (assumption 1) | ~{ESP32['flash_typical_mb'] * 1024:,} KB |

{verdict}

Flash: the ONNX file is {onnx_kb / 1024:.2f} MB against a typical
{ESP32['flash_typical_mb']} MB part — **{'comfortable' if fits_flash else 'tight'}**,
though a hand-rolled C array would be smaller than the ONNX protobuf, which carries
graph metadata a device does not need.

### The caveat that matters most

**A TFLite Micro conversion path was planned for this phase and abandoned**: TensorFlow
Lite Micro does not serve a scikit-learn tree ensemble, so the route would have required
re-training the predictor as a different model and re-validating every result against it.
ONNX was taken instead, as an explicit portability proxy. The dependency has now been
removed from the project rather than left advertised in a requirements file nothing
imported.

**This ONNX benchmark is a portability and feasibility proxy. It is not a claim that
this model can be flashed to an ESP32.** There is no ONNX Runtime for the ESP32 —
no interpreter, no execution provider, nothing to load the file with. Getting this
predictor onto the device requires one of:

- **A from-scratch tree traversal in C.** The most likely route for a random forest.
  Export the trees as `const` arrays in flash and write the traversal by hand — a few
  hundred lines, fully deterministic, no framework dependency. The conformal layer
  comes along for free: {len(labels)} float comparisons.
- **A framework with ESP32 support**, e.g. TensorFlow Lite Micro or ESP-DL. Neither
  runs a scikit-learn forest, so this means re-training the predictor as something the
  framework supports (a small MLP, or a gradient-boosted model via a supported
  converter) and re-validating every result in this project against the new model —
  including re-calibrating conformal, as section 2 shows.
- **`emlearn` or a similar C code generator for scikit-learn models**, which targets
  exactly this case and is the lowest-effort path worth trying first.

What this section does establish: the compressed model is **{shrink:.0f}x smaller**
than the Phase 3 forest at a cost of {abs(f1_cost):.4f} macro-F1, it runs in
{lat['onnx_single']['mean_ms']:.3f} ms per sample single-threaded, the conformal layer
adds {lat['conformal_single']['mean_ms']:.4f} ms, and the artifact is small enough that
flash is not the binding constraint. Those are the facts a port would start from.

Reproduce with `python -m src.deployment.compress_export`.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--dataset", default="beijing",
                    choices=("beijing", "bangladesh"))
    ap.add_argument("--class-weight", default=None,
                    help="e.g. 'balanced' — sweep the class-weighted family instead")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the write-up from saved metrics; no sweep, no export")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, horizon=args.horizon)
    if args.dataset != "beijing":
        from dataclasses import replace as _r
        cfg = _r(cfg, dataset=args.dataset)
    if args.class_weight:
        from dataclasses import replace as _replace
        cfg = _replace(cfg, class_weight=args.class_weight)

    if args.report_only:
        mp = cfg.reports_dir / "metrics" / f"deployment_h{cfg.horizon}{cfg.suffix}.json"
        payload = json.loads(mp.read_text())
        payload["config"] = cfg
        # Not serialised into the metrics file (it is the previous phase's artifact);
        # read it back from disk so the before/after coverage column still works.
        payload["conformal_old"] = joblib.load(cfg.conformal_path)
        if not args.no_write:
            cfg.report_path.write_text(build_report(payload))
            print(f"rewrote {cfg.report_path.relative_to(REPO_ROOT)}")
        return 0

    run(cfg, write=not args.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
