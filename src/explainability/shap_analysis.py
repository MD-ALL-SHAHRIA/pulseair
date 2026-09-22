"""SHAP attributions for the deployed RandomForest.

Explains individual predictions from `baseline_h6.pkl` — which of the nine wearable
channels pushed this reading toward the predicted AQI category, and by how much. The
per-case attributions feed `llm_advisory.py`, which turns them into plain language.

`TreeExplainer` is exact for tree ensembles (no sampling), so the values here are the
model's actual Shapley decomposition rather than an approximation.

One thing worth stating up front: SHAP explains **the model**, not the atmosphere. A
large attribution on `PM2.5` means the forest leaned on that channel, not that PM2.5
caused the air quality. For an advisory shown to a wearer that distinction matters, and
`llm_advisory.py` is prompted accordingly.

    python -m src.explainability.shap_analysis

Writes plots to ``reports/shap_examples/`` and per-case attributions to
``reports/metrics/shap_h6.json``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
ADVISORY = ("Very unhealthy", "Hazardous")

# Channels a wearer can act on, versus context they cannot. The advisory layer needs
# the distinction: "CO is high" is actionable, "it is 3am in January" is not.
ACTIONABLE = {"PM2.5", "PM10", "CO"}
CONTEXT = {"TEMP", "DEWP", "hour_sin", "hour_cos", "month_sin", "month_cos"}


@dataclass(frozen=True)
class ShapConfig:
    processed_root: Path
    artifacts_dir: Path
    reports_dir: Path
    horizon: int
    labels: list[str]
    background_size: int
    n_examples: int
    seed: int

    @property
    def processed_dir(self) -> Path:
        return self.processed_root / f"h{self.horizon}"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / f"baseline_h{self.horizon}.pkl"

    @property
    def plot_dir(self) -> Path:
        return self.reports_dir / "shap_examples"


def load_config(path: Path | str = DEFAULT_CONFIG, root: Path | None = None,
                horizon: int | None = None) -> ShapConfig:
    root = root or REPO_ROOT
    raw = yaml.safe_load(Path(path).read_text())
    data, ex, prep = raw["data"], raw["explainability"], raw["preprocessing"]
    return ShapConfig(
        processed_root=root / data["processed_dir"],
        artifacts_dir=root / raw["baseline"]["artifacts_dir"],
        reports_dir=root / "reports",
        horizon=int(horizon if horizon is not None else prep["horizon"]),
        labels=list(data["pm25_labels"]),
        background_size=int(ex["shap_background_size"]),
        n_examples=int(ex["shap_n_examples"]),
        seed=int(raw["seed"]),
    )


# ------------------------------------------------------------------ case picking


def pick_cases(cfg: ShapConfig, frame: pd.DataFrame, sets: np.ndarray,
               observed: np.ndarray) -> pd.DataFrame:
    """A deliberately non-random sample: the cases an advisory has to handle well.

    A uniform sample would be ~35% Unhealthy and would almost never contain a Hazardous
    hour or an ambiguous set, which are precisely the cases worth explaining.
    """
    rng = np.random.default_rng(cfg.seed)
    idx = np.flatnonzero(observed)
    sizes = sets.sum(axis=1)
    y = frame["y_category"].to_numpy()

    wanted, chosen = [], set()

    def take(mask: np.ndarray, n: int, tag: str) -> None:
        pool = [i for i in idx[mask[idx]] if i not in chosen]
        for i in rng.permutation(pool)[:n]:
            chosen.add(int(i))
            wanted.append({"row": int(i), "reason": tag})

    haz = cfg.labels.index("Hazardous")
    vun = cfg.labels.index("Very unhealthy")
    take((y == haz) & (sizes == 1), 2, "Hazardous, confident (singleton set)")
    take((y == haz) & (sizes > 1), 2, "Hazardous, ambiguous (set > 1)")
    take((y == vun) & (sizes > 1), 2, "Very unhealthy, ambiguous")
    take(sizes >= 4, 2, "highly ambiguous (set >= 4)")
    take(sizes == 1, 2, "confident, any class")
    take(np.ones(len(y), dtype=bool), max(cfg.n_examples - len(wanted), 0), "random")

    out = pd.DataFrame(wanted)
    return out.head(cfg.n_examples)


# ------------------------------------------------------------------------- shap


def compute(cfg: ShapConfig, verbose: bool = True) -> dict:
    import shap

    say = print if verbose else (lambda *a, **k: None)
    bundle = joblib.load(cfg.model_path)
    model, features = bundle["model"], bundle["feature_columns"]

    frame = pd.read_csv(cfg.processed_dir / "tabular_test.csv")
    X = frame[features].to_numpy(dtype=np.float32)
    observed = ~frame["is_imputed_pm25"].to_numpy(dtype=bool)

    conformal = joblib.load(cfg.artifacts_dir / f"conformal_h{cfg.horizon}.pkl")
    probs_raw = model.predict_proba(X)
    probs = np.zeros((len(frame), len(cfg.labels)))
    for col, cls in enumerate(model.classes_):
        probs[:, int(cls)] = probs_raw[:, col]
    sets = (1.0 - probs) <= conformal["q"] + 1e-12

    cases = pick_cases(cfg, frame, sets, observed)
    say(f"selected {len(cases)} cases for explanation:")
    for _, c in cases.iterrows():
        say(f"  row {c['row']:>6,}  {c['reason']}")

    say(f"\nTreeExplainer on {model.n_estimators} trees "
        f"(exact for tree ensembles, no background sampling needed)")
    explainer = shap.TreeExplainer(model)
    rows = cases["row"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = explainer.shap_values(X[rows])

    # shap returns (n, n_features, n_classes) for multiclass in recent versions and a
    # list of per-class arrays in older ones. Normalise to (n, n_features, n_classes).
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    values = np.asarray(values)
    say(f"SHAP values shape {values.shape}")

    # Unscale for human-readable feature values in the write-up.
    meta = json.loads((cfg.processed_dir / "metadata.json").read_text())
    scaler = joblib.load(cfg.processed_dir / "scaler.pkl")
    scaled_cols = list(meta["scaled_columns"])

    explained = []
    for k, row in enumerate(rows):
        pred = int(probs[row].argmax())
        contrib = values[k, :, pred]
        order = np.argsort(-np.abs(contrib))
        raw = {}
        for c in scaled_cols:
            i = scaled_cols.index(c)
            raw[c] = float(frame.loc[row, c] * scaler.scale_[i] + scaler.mean_[i])
        stamp = pd.to_datetime(frame.loc[row, "datetime"])
        explained.append({
            "row": int(row),
            "reason": cases.iloc[k]["reason"],
            "station": str(frame.loc[row, "station"]),
            "datetime": str(stamp),
            "hour": int(stamp.hour),
            "true_label": cfg.labels[int(frame.loc[row, "y_category"])],
            "predicted": cfg.labels[pred],
            "probabilities": {l: float(probs[row, i]) for i, l in enumerate(cfg.labels)},
            "conformal_set": [cfg.labels[i] for i in np.flatnonzero(sets[row])],
            "set_size": int(sets[row].sum()),
            "raw_features": raw,
            "shap": [{"feature": features[j], "value": float(contrib[j]),
                      "kind": "actionable" if features[j] in ACTIONABLE else "context"}
                     for j in order],
            "base_value": float(np.ravel(explainer.expected_value)[pred]),
        })
    return {"config": cfg, "cases": explained, "features": features,
            "n_trees": int(model.n_estimators)}


# ------------------------------------------------------------------------ plots


def plot_case(case: dict, cfg: ShapConfig, path: Path) -> None:
    """A horizontal contribution bar for one prediction.

    Deliberately not `shap.force_plot`: a wearable advisory needs something a
    non-specialist reads in two seconds, and the force plot's additive-strip idiom
    needs explaining before it helps.
    """
    top = case["shap"][:6][::-1]
    names = [s["feature"] for s in top]
    vals = [s["value"] for s in top]
    # Colour by direction, hatch the context channels a wearer cannot act on.
    colors = ["#c1443c" if v > 0 else "#3f7fb5" for v in vals]
    hatch = ["" if s["kind"] == "actionable" else "///" for s in top]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.6)
    for bar, h in zip(bars, hatch):
        bar.set_hatch(h)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel(f"SHAP contribution to P({case['predicted']})")
    size = case["set_size"]
    setstr = ", ".join(case["conformal_set"]) if size else "none"
    ax.set_title(
        f"{case['station']} {case['datetime']}\n"
        f"predicted {case['predicted']} (true {case['true_label']}) · "
        f"conformal set [{setstr}]",
        fontsize=9, loc="left")
    ax.tick_params(labelsize=9)
    fig.text(0.99, 0.02, "hatched = context the wearer cannot act on",
             ha="right", fontsize=7, color="#666")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_summary(result: dict, X: np.ndarray, values: np.ndarray, path: Path) -> None:
    """Mean |SHAP| per feature, averaged over classes — which channels the model uses."""
    cfg = result["config"]
    importance = np.abs(values).mean(axis=(0, 2))
    order = np.argsort(importance)
    names = [result["features"][i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, importance[order], color="#3f7fb5")
    ax.set_xlabel("mean |SHAP| (averaged over classes and sampled rows)")
    ax.set_title("Which channels the forest actually uses", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(cfg: ShapConfig | None = None, *, write: bool = True,
        verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    say = print if verbose else (lambda *a, **k: None)
    result = compute(cfg, verbose)

    if not write:
        return result

    cfg.plot_dir.mkdir(parents=True, exist_ok=True)
    for old in cfg.plot_dir.glob("*.png"):
        old.unlink()

    for i, case in enumerate(result["cases"], 1):
        tag = case["reason"].split(",")[0].replace(" ", "_").replace(">=", "ge")
        name = f"{i:02d}_{tag}_row{case['row']}.png"
        plot_case(case, cfg, cfg.plot_dir / name)
        case["plot"] = f"reports/shap_examples/{name}"
    say(f"\nwrote {len(result['cases'])} case plots to "
        f"{cfg.plot_dir.relative_to(REPO_ROOT)}/")

    # Global view over a larger sample, so the summary is not driven by the 12 hand
    # picked cases.
    import shap
    bundle = joblib.load(cfg.model_path)
    frame = pd.read_csv(cfg.processed_dir / "tabular_test.csv")
    rng = np.random.default_rng(cfg.seed)
    sample = rng.choice(len(frame), size=min(cfg.background_size, len(frame)),
                        replace=False)
    Xs = frame[result["features"]].to_numpy(dtype=np.float32)[sample]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gv = shap.TreeExplainer(bundle["model"]).shap_values(Xs)
    gv = np.stack(gv, axis=-1) if isinstance(gv, list) else np.asarray(gv)
    plot_summary(result, Xs, gv, cfg.plot_dir / "00_global_importance.png")
    say(f"wrote global importance over {len(sample)} sampled rows")

    index = cfg.plot_dir / "README.md"
    index.write_text(build_index(result, gv))
    say(f"wrote {index.relative_to(REPO_ROOT)}")

    mpath = cfg.reports_dir / "metrics" / f"shap_h{cfg.horizon}.json"
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(
        {"cases": result["cases"], "features": result["features"],
         "global_mean_abs": {f: float(v) for f, v in
                             zip(result["features"], np.abs(gv).mean(axis=(0, 2)))}},
        indent=2, default=str))
    say(f"wrote {mpath.relative_to(REPO_ROOT)}")
    return result


def build_index(result: dict, gv: np.ndarray) -> str:
    cfg = result["config"]
    feats = result["features"]
    imp = np.abs(gv).mean(axis=(0, 2))
    order = np.argsort(-imp)

    def table(header, rows):
        # `mean |SHAP|` has literal pipes in it; unescaped they split the header cell
        # and leave the table ragged. Escape every cell, header included.
        esc = lambda cs: [str(c).replace("|", "\\|") for c in cs]
        b = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return (f"| {' | '.join(esc(header))} |\n"
                f"| {' | '.join(['---'] * len(header))} |\n{b}")

    gl = [[f"`{feats[i]}`", f"{imp[i]:.4f}",
           "actionable" if feats[i] in ACTIONABLE else "context"] for i in order]

    case_rows = []
    for c in result["cases"]:
        top3 = ", ".join(f"`{s['feature']}` {s['value']:+.3f}" for s in c["shap"][:3])
        case_rows.append([
            f"[{Path(c['plot']).name}]({Path(c['plot']).name})",
            c["reason"], c["true_label"], c["predicted"],
            str(c["set_size"]), top3,
        ])

    return f"""# SHAP examples — horizon {cfg.horizon} h

Attributions for `baseline_h{cfg.horizon}.pkl` ({result['n_trees']} trees) via
`shap.TreeExplainer`, which is exact for tree ensembles — these are the model's actual
Shapley decomposition, not a sampled approximation.

**SHAP explains the model, not the atmosphere.** A large attribution on `PM2.5` means
the forest leaned on that channel for this prediction. It does not mean PM2.5 caused
the air quality, and the advisory copy must not imply it does.

In the plots, red pushes toward the predicted class and blue away from it. Hatched bars
are context channels (time of day, month, temperature, dew point) that a wearer cannot
act on — the distinction matters when the explanation becomes advice.

## Global channel importance

Mean |SHAP| over {gv.shape[0]} randomly sampled test rows, averaged across classes.

{table(["Channel", "mean |SHAP|", "Kind"], gl)}

![global importance](00_global_importance.png)

## Cases

Chosen deliberately, not uniformly: a uniform sample of this test split is ~31%
*Unhealthy* and would rarely contain a Hazardous hour or a genuinely ambiguous set,
which are the cases an advisory has to get right.

{table(["Plot", "Why chosen", "True", "Predicted", "Set size", "Top-3 SHAP"], case_rows)}

Regenerate with `python -m src.explainability.shap_analysis`.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    run(load_config(args.config, horizon=args.horizon), write=not args.no_write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
