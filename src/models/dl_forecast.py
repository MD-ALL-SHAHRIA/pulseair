"""Sequence models for AQI risk classification at the primary horizon.

An LSTM and a Transformer encoder over the 24-hour window, both predicting the same
six-class AQI risk label the tabular baselines predict, both wrapped in MC dropout for
predictive uncertainty. The point of the exercise is a fair comparison, so every
convention from the earlier phases is kept:

* **Unaugmented data only.** Phase 4 found both CTGAN variants significantly degrade
  Very unhealthy and Hazardous (``reports/gan_ablation_h6.md``), so the training set is
  the real one.
* **Fit on all real training rows, select and report on observed rows.** Same as
  ``baseline.py``: imputed-label rows are kept for fitting but excluded from every
  metric, because forward-fill inflates Hazardous prevalence.
* **Selection on validation, never test.**
* **The same paired bootstrap** as the Phase 4 ablation, and the same disqualification
  rule: a model that gains on aggregate while significantly degrading an advisory class
  does not win.

    python -m src.models.dl_forecast

Writes ``reports/dl_metrics_h6.md`` and ``src/models/artifacts/dl_model_h6.pt``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

from src.gan.ablation import paired_bootstrap, CI, N_BOOTSTRAP

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
RARE = ("Very unhealthy", "Hazardous")
N_CALIB_BINS = 10


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class DLConfig:
    processed_root: Path
    artifacts_dir: Path
    reports_dir: Path
    horizon: int
    labels: list[str]
    architectures: list[str]
    num_layers: int
    dropout: float
    lr: float
    batch_size: int
    epochs: int
    patience: int
    lstm: dict
    transformer: dict
    mc_passes: int
    seed: int

    @property
    def processed_dir(self) -> Path:
        return self.processed_root / f"h{self.horizon}"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / f"dl_model_h{self.horizon}.pt"

    @property
    def report_path(self) -> Path:
        return self.reports_dir / f"dl_metrics_h{self.horizon}.md"

    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / "metrics" / f"dl_h{self.horizon}.json"


def load_config(path: Path | str = DEFAULT_CONFIG, root: Path | None = None,
                horizon: int | None = None) -> DLConfig:
    root = root or REPO_ROOT
    raw = yaml.safe_load(Path(path).read_text())
    data, m, prep, unc = raw["data"], raw["model"], raw["preprocessing"], raw["uncertainty"]
    if unc["method"] != "mc_dropout":
        raise ValueError(f"uncertainty.method is {unc['method']!r}; this module "
                         "implements mc_dropout")
    arch = m["architecture"]
    return DLConfig(
        processed_root=root / data["processed_dir"],
        artifacts_dir=root / raw["baseline"]["artifacts_dir"],
        reports_dir=root / "reports",
        horizon=int(horizon if horizon is not None else prep["horizon"]),
        labels=list(data["pm25_labels"]),
        architectures=list(arch) if isinstance(arch, list) else [arch],
        num_layers=int(m["num_layers"]),
        dropout=float(m["dropout"]),
        lr=float(m["lr"]),
        batch_size=int(m["batch_size"]),
        epochs=int(m["epochs"]),
        patience=int(m["patience"]),
        lstm=dict(m["lstm"]),
        transformer=dict(m["transformer"]),
        mc_passes=int(unc["n_samples"]),
        seed=int(raw["seed"]),
    )


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------- data


@dataclass
class SeqSplit:
    name: str
    X: torch.Tensor          # (n, window, features), already scaled by Phase 2
    y: torch.Tensor          # (n,) int64 class index
    observed: np.ndarray     # bool: label came from a measured PM2.5 reading

    def __len__(self) -> int:
        return len(self.y)


def load_split(cfg: DLConfig, name: str) -> SeqSplit:
    with np.load(cfg.processed_dir / f"sequences_{name}.npz", allow_pickle=False) as z:
        X = torch.from_numpy(z["X"].astype(np.float32))
        y = torch.from_numpy(z["y_category"].astype(np.int64))
        observed = ~z["is_imputed_pm25"]
    return SeqSplit(name, X, y, observed)


# --------------------------------------------------------------------------- models


class LSTMClassifier(nn.Module):
    """2-layer LSTM over the window; the final step's hidden state feeds the head."""

    def __init__(self, n_features: int, n_classes: int, hidden: int, layers: int,
                 dropout: float):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(self.drop(out[:, -1]))


class TransformerClassifier(nn.Module):
    """2-layer, 4-head encoder with a learned positional embedding.

    The window is only 24 steps, so a learned embedding over 24 positions costs
    almost nothing and avoids the sinusoidal scheme's arbitrary frequency choice.
    """

    def __init__(self, n_features: int, n_classes: int, window: int, d_model: int,
                 nhead: int, dim_feedforward: int, layers: int, dropout: float):
        super().__init__()
        if d_model % nhead:
            raise ValueError(f"d_model={d_model} is not divisible by nhead={nhead}")
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, window, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        # enable_nested_tensor is a no-op with norm_first and only emits a warning.
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             enable_nested_tensor=False)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.proj(x) + self.pos)
        # Final position, mirroring the LSTM's last-step read-out so the two
        # architectures differ in mechanism rather than in how they pool.
        return self.head(self.drop(h[:, -1]))


def build_model(arch: str, cfg: DLConfig, n_features: int, window: int) -> nn.Module:
    n_classes = len(cfg.labels)
    if arch == "lstm":
        return LSTMClassifier(n_features, n_classes, int(cfg.lstm["hidden_size"]),
                              cfg.num_layers, cfg.dropout)
    if arch == "transformer":
        return TransformerClassifier(
            n_features, n_classes, window, int(cfg.transformer["d_model"]),
            int(cfg.transformer["nhead"]), int(cfg.transformer["dim_feedforward"]),
            cfg.num_layers, cfg.dropout)
    raise ValueError(f"unknown architecture {arch!r}")


# ------------------------------------------------------------------------ training


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    return float(f1_score(y_true, y_pred, labels=list(range(n_classes)),
                          average="macro", zero_division=0))


@torch.no_grad()
def _predict_logits(model: nn.Module, X: torch.Tensor, device, batch: int) -> torch.Tensor:
    out = []
    for i in range(0, len(X), batch):
        out.append(model(X[i:i + batch].to(device)).float().cpu())
    return torch.cat(out)


def train_model(arch: str, cfg: DLConfig, train: SeqSplit, val: SeqSplit,
                device, verbose: bool = True) -> dict:
    """Fit one architecture, early-stopping on validation observed-only macro-F1.

    Fitting uses every real training row; the stopping criterion uses observed rows
    only. That split of duties is the same one ``baseline.py`` uses -- imputed labels
    are usable signal but not a trustworthy yardstick.
    """
    say = print if verbose else (lambda *a, **k: None)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = build_model(arch, cfg, train.X.shape[2], train.X.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()
    n_classes = len(cfg.labels)

    val_y = val.y.numpy()[val.observed]
    best = {"f1": -1.0, "epoch": -1, "state": None}
    history, since_best = [], 0
    generator = torch.Generator().manual_seed(cfg.seed)
    t0 = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = torch.randperm(len(train.X), generator=generator)
        total = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, yb = train.X[idx].to(device), train.y[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.detach().item() * len(idx)

        model.eval()
        pred = _predict_logits(model, val.X, device, 4096).argmax(1).numpy()
        f1 = _macro_f1(val_y, pred[val.observed], n_classes)
        history.append({"epoch": epoch, "train_loss": total / len(perm), "val_macro_f1": f1})

        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            since_best = 0
        else:
            since_best += 1
        if verbose and (epoch <= 3 or epoch % 5 == 0 or since_best == 0):
            say(f"    epoch {epoch:3d}  loss {total / len(perm):.4f}  "
                f"val macro-F1 {f1:.4f}{'  *' if since_best == 0 else ''}")
        if since_best >= cfg.patience:
            say(f"    early stop at epoch {epoch} (no gain for {cfg.patience})")
            break

    model.load_state_dict(best["state"])
    mins = (time.perf_counter() - t0) / 60
    say(f"  {arch}: best val macro-F1 {best['f1']:.4f} @ epoch {best['epoch']} "
        f"| {n_params:,} params | {mins:.1f} min")
    return {"arch": arch, "model": model, "val_macro_f1": best["f1"],
            "best_epoch": best["epoch"], "epochs_run": len(history),
            "n_params": n_params, "history": history, "train_minutes": round(mins, 1)}


# -------------------------------------------------------------------- MC dropout


def _enable_dropout(model: nn.Module) -> None:
    """Put only the stochastic layers back in training mode.

    ``model.train()`` would do the same thing here -- there is no BatchNorm in either
    architecture -- but being explicit means adding a normalisation layer later cannot
    silently change what MC dropout is sampling over. ``nn.LSTM`` carries its
    inter-layer dropout internally, so it has to be switched too.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.LSTM, nn.TransformerEncoderLayer)):
            module.train()


def mc_inference_device(model: nn.Module, X: torch.Tensor, device: torch.device,
                        verbose: bool = True) -> torch.device:
    """The device MC dropout can actually run on.

    MPS has no fused-attention kernel that supports dropout, and the fused path is the
    one taken under ``no_grad``. Training is unaffected (grad enabled falls back to the
    math implementation), so only the MC passes need moving. Probing is better than
    hard-coding a device check: the limitation is a kernel gap that a future torch will
    close, and the probe will notice.
    """
    if device.type != "mps":
        return device
    # Probe with the real input shape: the positional embedding is window-sized.
    probe = X[:2].to(device)
    model.to(device)
    _enable_dropout(model)
    try:
        with torch.no_grad():
            model(probe)
        return device
    except NotImplementedError:
        if verbose:
            print(f"    note: {device.type} cannot run dropout under no_grad for this "
                  f"model (fused attention kernel); MC passes run on cpu")
        return torch.device("cpu")


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, X: torch.Tensor, cfg: DLConfig,
                       device, verbose: bool = True) -> dict:
    """``cfg.mc_passes`` stochastic forward passes; return the predictive distribution.

    Returns the mean class probability, the point prediction, and three uncertainty
    measures. Decomposing entropy is worth the few extra lines: total predictive
    entropy mixes irreducible class overlap with the model's own disagreement across
    passes, and only the second part shrinks with more data.
    """
    device = mc_inference_device(model, X, device, verbose)
    model.to(device)
    model.eval()
    _enable_dropout(model)
    n, n_classes = len(X), model.head.out_features
    mean = torch.zeros(n, n_classes)
    mean_sq_entropy = torch.zeros(n)

    for p in range(cfg.mc_passes):
        probs = torch.softmax(_predict_logits(model, X, device, 4096), dim=1)
        mean += probs
        # Expected entropy of the individual passes -> the aleatoric part.
        mean_sq_entropy += -(probs * torch.log(probs.clamp_min(1e-12))).sum(1)
        if verbose and (p + 1) % 10 == 0:
            print(f"    MC pass {p + 1}/{cfg.mc_passes}")
    mean /= cfg.mc_passes
    mean_sq_entropy /= cfg.mc_passes

    predictive_entropy = -(mean * torch.log(mean.clamp_min(1e-12))).sum(1)
    confidence, pred = mean.max(1)
    return {
        "probs": mean.numpy(),
        "pred": pred.numpy(),
        "confidence": confidence.numpy(),
        "entropy": predictive_entropy.numpy(),
        "aleatoric": mean_sq_entropy.numpy(),
        # Mutual information = total - aleatoric: disagreement between passes, i.e.
        # the part attributable to the model rather than to the data.
        "epistemic": (predictive_entropy - mean_sq_entropy).numpy(),
    }


def calibration(confidence: np.ndarray, correct: np.ndarray,
                n_bins: int = N_CALIB_BINS) -> dict:
    """Equal-width reliability bins plus ECE and MCE.

    A well-calibrated model that says 0.70 is right about 70% of the time. The gap
    column is accuracy minus confidence: negative means overconfident.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins, ece, mce, n = [], 0.0, 0.0, len(confidence)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi) if lo > 0 else \
               (confidence >= lo) & (confidence <= hi)
        count = int(mask.sum())
        if not count:
            bins.append({"low": float(lo), "high": float(hi), "n": 0,
                         "confidence": None, "accuracy": None, "gap": None})
            continue
        conf, acc = float(confidence[mask].mean()), float(correct[mask].mean())
        gap = acc - conf
        ece += count / n * abs(gap)
        mce = max(mce, abs(gap))
        bins.append({"low": float(lo), "high": float(hi), "n": count,
                     "confidence": conf, "accuracy": acc, "gap": gap})
    return {"bins": bins, "ece": float(ece), "mce": float(mce), "n": int(n)}


# ------------------------------------------------------------------- comparators


def reference_predictions(cfg: DLConfig, test: SeqSplit) -> dict:
    """Persistence and the Phase 3 RandomForest, on the very same test samples.

    The tabular and sequence views of a split hold identical samples in identical order
    (Phase 2 asserts it), so the saved RF can be scored directly against the sequence
    labels -- no retraining, and no risk of a differently-seeded forest shifting the
    comparison. The alignment is re-checked here rather than assumed.
    """
    import joblib
    import pandas as pd

    meta = json.loads((cfg.processed_dir / "metadata.json").read_text())
    frame = pd.read_csv(cfg.processed_dir / "tabular_test.csv")
    y_tab = frame["y_category"].to_numpy()
    if not np.array_equal(y_tab, test.y.numpy()):
        raise ValueError("tabular_test.csv and sequences_test.npz are misaligned; "
                         "the RF comparison would be meaningless")

    out = {}
    bundle_path = cfg.artifacts_dir / f"baseline_h{cfg.horizon}.pkl"
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        X = frame[bundle["feature_columns"]].to_numpy(dtype=np.float32)
        out["RandomForest (Phase 3)"] = bundle["model"].predict(X).astype(np.int64)

    scaler = joblib.load(cfg.processed_dir / "scaler.pkl")
    scaled = list(meta["scaled_columns"])
    i = scaled.index("PM2.5")
    pm_now = frame["PM2.5"].to_numpy(dtype=np.float64) * scaler.scale_[i] + scaler.mean_[i]
    edges = [*yaml.safe_load(DEFAULT_CONFIG.read_text())["data"]["pm25_breakpoints"],
             np.inf]
    out["Persistence"] = pd.cut(pm_now, bins=edges, labels=False,
                                right=True).astype(np.int64)
    return out


# --------------------------------------------------------------------- evaluation


def score(y: np.ndarray, pred: np.ndarray, labels: list[str]) -> dict:
    return {
        "macro_f1": _macro_f1(y, pred, len(labels)),
        "accuracy": float((y == pred).mean()),
        "per_class": {lab: float(f1_score(y == i, pred == i, zero_division=0))
                      for i, lab in enumerate(labels)},
    }


def run(cfg: DLConfig | None = None, *, write: bool = True, verbose: bool = True,
        n_boot: int = N_BOOTSTRAP) -> dict:
    cfg = cfg or load_config()
    say = print if verbose else (lambda *a, **k: None)
    device = pick_device()
    say(f"device: {device} | horizon {cfg.horizon}h | seed {cfg.seed}")

    train, val, test = (load_split(cfg, s) for s in ("train", "val", "test"))
    say(f"train {len(train):,} ({int(train.observed.sum()):,} observed) | "
        f"val {len(val):,} ({int(val.observed.sum()):,}) | "
        f"test {len(test):,} ({int(test.observed.sum()):,})")
    say(f"input {tuple(train.X.shape[1:])} -> {len(cfg.labels)} classes")

    trained = {}
    for arch in cfg.architectures:
        say(f"\n[{arch}]")
        trained[arch] = train_model(arch, cfg, train, val, device, verbose)

    # Selection: validation observed-only macro-F1. Test is untouched until now.
    selected = max(trained, key=lambda a: trained[a]["val_macro_f1"])
    say(f"\nselected on VAL observed macro-F1: {selected} "
        f"({trained[selected]['val_macro_f1']:.4f})")

    mask = test.observed
    y_obs = test.y.numpy()[mask]

    say(f"\nMC dropout: {cfg.mc_passes} passes over {len(test):,} test sequences")
    mc = {}
    for arch in cfg.architectures:
        say(f"  [{arch}]")
        mc[arch] = mc_dropout_predict(trained[arch]["model"], test.X, cfg, device, verbose)

    results = {}
    for arch in cfg.architectures:
        m = trained[arch]["model"].to(device).eval()
        det = _predict_logits(m, test.X, device, 4096).argmax(1).numpy()
        results[f"{arch} (MC mean)"] = score(y_obs, mc[arch]["pred"][mask], cfg.labels)
        results[f"{arch} (deterministic)"] = score(y_obs, det[mask], cfg.labels)

    refs = reference_predictions(cfg, test)
    for name, pred in refs.items():
        results[name] = score(y_obs, pred[mask], cfg.labels)

    sel_pred = mc[selected]["pred"][mask]
    calib = {a: calibration(mc[a]["confidence"][mask],
                            (mc[a]["pred"][mask] == y_obs).astype(float))
             for a in cfg.architectures}

    say("\npaired bootstrap vs the references ...")
    tests = {}
    for name, pred in refs.items():
        t = {"macro_f1": paired_bootstrap(y_obs, pred[mask], sel_pred, len(cfg.labels), n_boot)}
        for lab in RARE:
            i = cfg.labels.index(lab)
            t[lab] = paired_bootstrap(
                y_obs, pred[mask], sel_pred, len(cfg.labels), n_boot,
                scorer=lambda yt, yp, i=i: float(f1_score(yt == i, yp == i, zero_division=0)))
        tests[name] = t
        say(f"  vs {name:24s} Δ {t['macro_f1']['observed_diff']:+.4f} "
            f"[{t['macro_f1']['ci_low']:+.4f}, {t['macro_f1']['ci_high']:+.4f}] "
            f"{'SIGNIFICANT' if t['macro_f1']['significant'] else 'not significant'}")

    payload = {
        "config": cfg, "selected": selected, "trained": trained, "results": results,
        "calibration": calib, "tests": tests, "n_boot": n_boot,
        "uncertainty": {a: {"mean_entropy": float(mc[a]["entropy"][mask].mean()),
                            "mean_epistemic": float(mc[a]["epistemic"][mask].mean()),
                            "mean_confidence": float(mc[a]["confidence"][mask].mean())}
                        for a in cfg.architectures},
        "counts": {"train": len(train), "val": len(val), "test": len(test),
                   "test_observed": int(mask.sum())},
    }

    if write:
        cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "architecture": selected,
            "state_dict": trained[selected]["model"].state_dict(),
            "hyperparameters": {
                "num_layers": cfg.num_layers, "dropout": cfg.dropout,
                **(cfg.lstm if selected == "lstm" else cfg.transformer)},
            "n_features": train.X.shape[2], "window": train.X.shape[1],
            "class_labels": cfg.labels, "horizon": cfg.horizon,
            "mc_passes": cfg.mc_passes, "seed": cfg.seed,
            "selected_on": "VALIDATION observed-only macro-F1",
            "val_macro_f1": trained[selected]["val_macro_f1"],
            "test_observed_macro_f1": results[f"{selected} (MC mean)"]["macro_f1"],
            "trained_on": f"data/processed/h{cfg.horizon}/sequences_train.npz (unaugmented)",
        }, cfg.model_path)
        say(f"\nsaved {selected} -> {cfg.model_path.relative_to(REPO_ROOT)} "
            f"({cfg.model_path.stat().st_size / 1e6:.2f} MB)")

        cfg.report_path.write_text(build_report(payload))
        say(f"wrote {cfg.report_path.relative_to(REPO_ROOT)}")
        cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.metrics_path.write_text(json.dumps({
            "selected": selected, "results": results, "tests": tests,
            "calibration": calib, "uncertainty": payload["uncertainty"],
            "counts": payload["counts"], "n_boot": n_boot,
            "training": {a: {k: v for k, v in trained[a].items() if k != "model"}
                         for a in cfg.architectures},
        }, indent=2, default=str))
        say(f"wrote {cfg.metrics_path.relative_to(REPO_ROOT)}")
    return payload


# ------------------------------------------------------------------------- report


def _calibration_comparison(payload: dict) -> str:
    """Report ECE against MCE across architectures, when they disagree.

    ECE is an average over reliability bins and MCE is the worst one. They can rank
    two models in opposite directions, and when they do it is the worst bin that
    matters here: the advisory gates on confidence, so the question is not how well
    calibrated the model is on average but how badly it can be wrong when it is
    confident. Written only when a reversal is present -- there is no point asserting
    a pattern the numbers do not show.
    """
    cal = payload.get("calibration") or {}
    if len(cal) < 2:
        return ""
    rows = sorted(((a, c["ece"], c["mce"]) for a, c in cal.items()), key=lambda r: r[1])
    best_ece, worst_ece = rows[0], rows[-1]
    if best_ece[2] <= worst_ece[2]:          # same ranking on both -> nothing to report
        table = "\n".join(f"| {a} | {e:.4f} | {m:.4f} |" for a, e, m in rows)
        return (f"| Model | ECE | MCE |\n| --- | --- | --- |\n{table}\n\n"
                f"ECE and MCE agree on the ranking here.")
    ratio = best_ece[2] / worst_ece[2] if worst_ece[2] else float("inf")
    table = "\n".join(f"| {a} | {e:.4f} | {m:.4f} |" for a, e, m in rows)
    return f"""### ECE and MCE disagree about which model is better calibrated

| Model | ECE (mean bin gap) | MCE (worst bin gap) |
| --- | --- | --- |
{table}

**{best_ece[0]} has the better ECE ({best_ece[1]:.4f} vs {worst_ece[1]:.4f}) and the worse
MCE ({best_ece[2]:.4f} vs {worst_ece[2]:.4f}) — {ratio:.1f}x worse in its worst bin.** On
average it is the better-calibrated model; where it is most confidently wrong it is far
worse. For a system that suppresses low-confidence warnings, the worst bin is the
operative number, and selecting on ECE alone would have picked the wrong model.

This is the third time in this project an aggregate metric has concealed a tail failure.
The first was the augmentation ablation, where CTGAN and SMOTE both raised macro-F1 while
significantly degrading the two advisory classes — the pattern the disqualification rule
exists to catch. The second was conformal coverage, where the marginal guarantee was met
on average while Hazardous was covered only 84.3% of the time, which Mondrian calibration
fixed. This is the same shape a third time, in a third place. **The recurring lesson is
not about any one metric: it is that an average over a distribution says nothing about
its tail, and in a safety-critical advisory the tail is the product.**


"""


def build_report(payload: dict) -> str:
    cfg, sel = payload["config"], payload["selected"]
    results, tests, calib = payload["results"], payload["tests"], payload["calibration"]
    trained, counts, unc = payload["trained"], payload["counts"], payload["uncertainty"]
    labels = cfg.labels

    def table(header, body):
        esc = lambda cs: [str(c).replace("|", "\\|") for c in cs]
        b = "\n".join("| " + " | ".join(esc(r)) + " |" for r in body)
        return f"| {' | '.join(header)} |\n| {' | '.join(['---'] * len(header))} |\n{b}"

    order = [f"{sel} (MC mean)", f"{sel} (deterministic)"]
    order += [f"{a} (MC mean)" for a in cfg.architectures if a != sel]
    order += [n for n in ("RandomForest (Phase 3)", "Persistence") if n in results]

    main = [[f"**{n}**" if n == order[0] else n,
             f"{results[n]['accuracy']:.4f}", f"**{results[n]['macro_f1']:.4f}**",
             f"{results[n]['per_class']['Very unhealthy']:.4f}",
             f"{results[n]['per_class']['Hazardous']:.4f}"] for n in order]

    per_cls = [[f"**{l}**" if l in RARE else l,
                *[f"{results[n]['per_class'][l]:.4f}" for n in order]] for l in labels]

    train_rows = [[a, f"{trained[a]['n_params']:,}", f"{trained[a]['val_macro_f1']:.4f}",
                   f"{trained[a]['best_epoch']}/{trained[a]['epochs_run']}",
                   f"{trained[a]['train_minutes']:.1f}"] for a in cfg.architectures]

    cal = calib[sel]
    cal_rows = [[f"{b['low']:.1f}–{b['high']:.1f}", f"{b['n']:,}",
                 "—" if b["confidence"] is None else f"{b['confidence']:.4f}",
                 "—" if b["accuracy"] is None else f"{b['accuracy']:.4f}",
                 "—" if b["gap"] is None else f"{b['gap']:+.4f}"]
                for b in cal["bins"] if b["n"]]

    boot_rows = []
    for ref in tests:
        for metric in ("macro_f1", *RARE):
            t = tests[ref][metric]
            boot_rows.append([
                ref, "Macro-F1" if metric == "macro_f1" else f"{metric} F1",
                f"{t['observed_diff']:+.4f}",
                f"[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]",
                f"{t['p_two_sided']:.3f}",
                "**yes**" if t["significant"] else "no",
            ])

    # Learning-rate robustness, if the sweep has been run. Worth printing: without it
    # a reader cannot tell an honest negative result from an undertrained one.
    sweep_block = ""
    sweep_path = cfg.reports_dir / "metrics" / f"dl_lr_sweep_h{cfg.horizon}.json"
    if sweep_path.exists():
        sw = json.loads(sweep_path.read_text())
        grid = sw["grid"]
        rows = [[a, *[f"**{sw['results'][a][g]['val_macro_f1']:.4f}**"
                      if sw["results"][a][g]["val_macro_f1"] ==
                      max(sw["results"][a][x]["val_macro_f1"] for x in grid)
                      else f"{sw['results'][a][g]['val_macro_f1']:.4f}" for g in grid]]
                for a in cfg.architectures if a in sw["results"]]
        picked = {a: max(grid, key=lambda g: sw["results"][a][g]["val_macro_f1"])
                  for a in sw["results"]}
        sweep_block = f"""
### Learning-rate robustness

The first run of this phase used `lr: 1e-3` and both architectures peaked at epoch 2–4
and then declined while training loss kept falling. That pattern is ambiguous — it can
mean the task is hard, or that the step size is too large — so the rate was swept before
any conclusion was drawn. A negative result from an undertrained model is not a result.

Validation macro-F1 (observed rows), {sw['epochs']} epoch ceiling, patience
{sw['patience']}:

{table(["Architecture", *[f"lr={g}" for g in grid]], rows)}

Chosen: {', '.join(f'`{a}` at {picked[a]}' for a in picked)}. `configs/default.yaml`
now sets `model.lr: {cfg.lr:g}`. The sweep reads validation only, so the test split is
still untouched by any of these decisions. Reproduce with
`python -m src.models.dl_forecast --lr-sweep`.
"""

    # Validation ranked the models one way and test ranked them the other. That is
    # worth its own paragraph, not a footnote: it is the concrete cost of the
    # prevalence gap Phase 2 warned about.
    transfer_block = ""
    rf_metrics = cfg.reports_dir / "metrics" / f"baseline_h{cfg.horizon}.json"
    rf_key = "RandomForest (Phase 3)"
    if rf_metrics.exists() and rf_key in results:
        rf = json.loads(rf_metrics.read_text())
        rf_val = rf["results"]["RandomForest"]["val"]["observed"]["macro_f1"]
        rf_test = results[rf_key]["macro_f1"]
        dl_val = trained[sel]["val_macro_f1"]
        dl_test = results[f"{sel} (MC mean)"]["macro_f1"]
        if (dl_val > rf_val) != (dl_test > rf_test):
            transfer_block = f"""
### Validation preferred the sequence model; test did not

{table(["Model", "Validation macro-F1", "Test macro-F1", "Change"],
       [[f"{sel} (selected)", f"{dl_val:.4f}", f"{dl_test:.4f}", f"{dl_test - dl_val:+.4f}"],
        ["RandomForest (Phase 3)", f"{rf_val:.4f}", f"{rf_test:.4f}",
         f"{rf_test - rf_val:+.4f}"]])}

On validation the {sel} leads the forest by {dl_val - rf_val:+.4f}. On test it trails by
{dl_test - rf_test:+.4f}. **The ranking reverses**, and the protocol was followed
correctly — selection never touched test.

This is the validation caveat from `reports/preprocessing_summary_h{cfg.horizon}.md`
arriving in practice. The chronological split leaves Very unhealthy at 7.04% of
validation and 12.14% of test, so the two splits are not interchangeable yardsticks for
a macro-averaged metric: the forest *gains* {rf_test - rf_val:+.4f} moving to test while
the {sel} *loses* {dl_test - dl_val:+.4f}. Selecting on validation remains the right
protocol — the alternative is worse — but a validation win at this margin is not
evidence of a test win, and the write-up should not present it as one.
"""

    # Aleatoric vs epistemic: which kind of uncertainty dominates decides whether more
    # data or a bigger model is even the right lever.
    u = unc[sel]
    epi_share = u["mean_epistemic"] / u["mean_entropy"] if u["mean_entropy"] else 0.0
    uncertainty_note = (
        f"Of the {u['mean_entropy']:.4f} nats of mean predictive entropy, only "
        f"{u['mean_epistemic']:.4f} ({epi_share:.1%}) is epistemic — the part that comes "
        f"from the model disagreeing with itself across passes. "
        + ("The remaining ~{:.0%} is aleatoric: irreducible overlap between classes "
           "given these nine channels and this horizon. **That is a statement about the "
           "task, not the model.** More capacity, more epochs, or more data move the "
           "epistemic sliver and leave the rest alone, which is consistent with every "
           "result in this phase and the last two. If the six-hour-ahead category is to "
           "be predicted better, the input has to change — more channels, a longer "
           "window, or spatial context from neighbouring stations — not the "
           "architecture.".format(1 - epi_share)
           if epi_share < 0.1 else
           "A substantial epistemic share means the model is genuinely uncertain rather "
           "than the task being ambiguous, so more data or capacity may still help.")
    )

    verdict = _verdict(payload)
    over = [b for b in cal["bins"] if b["gap"] is not None and b["gap"] < -0.05]
    cal_note = (
        f"ECE **{cal['ece']:.4f}**, MCE **{cal['mce']:.4f}**. "
        + (f"{len(over)} bin(s) are overconfident by more than 5 points — the model "
           f"claims more certainty than it earns there, which matters for an advisory "
           f"that suppresses low-confidence warnings."
           if over else
           "No bin is overconfident by more than 5 points: where the model says 0.7 it "
           "is right about 70% of the time, which is what the advisory layer needs if "
           "it is going to gate on confidence.")
    )
    cal_compare = _calibration_comparison(payload)

    return f"""# Sequence models — horizon {cfg.horizon} h

LSTM and Transformer encoder over the {payload['counts'].get('window', 24)}-hour window,
predicting the same six-class AQI risk label as every earlier phase. MC dropout with
**{cfg.mc_passes} stochastic passes** supplies the predictive distribution.

Trained on **unaugmented** `data/processed/h{cfg.horizon}/sequences_train.npz`
({counts['train']:,} sequences). Phase 4 found both CTGAN variants significantly degrade
the advisory classes, so augmentation is off. Fitting uses every real training row;
selection and every number below use **observed rows only** ({counts['test_observed']:,}
of {counts['test']:,} test sequences), excluding imputed labels. Architecture was chosen
on **validation** macro-F1; test was not consulted until the table below.

---

## 1. Training and selection

{table(["Architecture", "Parameters", "Val macro-F1 (observed)", "Best/run epochs",
        "Minutes"], train_rows)}
{transfer_block}

Selected: **{sel}** on validation macro-F1
{trained[sel]['val_macro_f1']:.4f}. No class weighting, matching the tabular baselines —
the comparison is only meaningful if both sides face the imbalance on the same terms.
{sweep_block}

---

## 2. Test results (observed labels only)

{table(["Model", "Accuracy", "Macro-F1", "V.unhealthy F1", "Hazardous F1"], main)}

### Per-class F1

{table(["Class", *order], per_cls)}

---

## 3. Uncertainty and calibration

MC dropout keeps dropout active at inference and averages {cfg.mc_passes} passes. The
confidence below is the maximum of that averaged class distribution. Total predictive
entropy is split into the part inherent to the data and the part that is the model
disagreeing with itself across passes — only the second shrinks with more data.

{table(["Architecture", "Mean confidence", "Mean predictive entropy",
        "Mean epistemic (MI)"],
       [[a, f"{unc[a]['mean_confidence']:.4f}", f"{unc[a]['mean_entropy']:.4f}",
         f"{unc[a]['mean_epistemic']:.4f}"] for a in cfg.architectures])}

### Reliability — {sel}, {cal['n']:,} observed test predictions

A calibrated model's accuracy inside a confidence bin matches the bin's mean confidence.
Gap is accuracy minus confidence; negative is overconfident.

{table(["Confidence bin", "n", "Mean confidence", "Accuracy", "Gap"], cal_rows)}

{cal_note}

{cal_compare}

{uncertainty_note}

---

## 4. Paired bootstrap ({payload['n_boot']:,} resamples, {CI}% CI)

Same procedure as the Phase 4 ablation: both models scored on the same resampled rows
each iteration, so the comparison stays paired.

{table(["Baseline", "Metric", "Δ (DL − baseline)", f"{CI}% CI", "p (two-sided)",
        "Significant"], boot_rows)}

---

## 5. Does the complexity earn its place?

{verdict}

Reproduce with `python -m src.models.dl_forecast`.
"""


def _verdict(payload: dict) -> str:
    """Phase 4's rule, unchanged: aggregate gains do not excuse advisory-class losses."""
    MATERIAL = 0.01
    cfg, sel, tests = payload["config"], payload["selected"], payload["tests"]
    results = payload["results"]
    rf_key = "RandomForest (Phase 3)"
    dl_name = f"{sel} (MC mean)"

    lines = []
    if rf_key not in tests:
        return (f"`{rf_key}` was not on disk, so no comparison was made. Run "
                f"`python -m src.models.baseline` first.")

    rf = tests[rf_key]
    harmed = [m for m in RARE if rf[m]["significant"] and rf[m]["observed_diff"] < 0]
    macro = rf["macro_f1"]
    dl_f1 = results[dl_name]["macro_f1"]
    rf_f1 = results[rf_key]["macro_f1"]
    pers_f1 = results.get("Persistence", {}).get("macro_f1")

    if harmed:
        detail = "; ".join(
            f"{m} {rf[m]['observed_diff']:+.4f} "
            f"[{rf[m]['ci_low']:+.4f}, {rf[m]['ci_high']:+.4f}]" for m in harmed)
        lines.append(
            f"**No — and it is disqualified on the same rule Phase 4 used.** The "
            f"selected {sel} significantly degrades {detail} against the RandomForest. "
            f"Whatever it does to macro-F1, a model that is worse at the classes a "
            f"wearable advisory exists to raise is not the model to ship.")
    elif macro["significant"] and macro["observed_diff"] >= MATERIAL:
        lines.append(
            f"**Yes.** The {sel} beats the RandomForest by "
            f"{macro['observed_diff']:+.4f} macro-F1, {CI}% CI "
            f"[{macro['ci_low']:+.4f}, {macro['ci_high']:+.4f}] — significant and above "
            f"the {MATERIAL} practical threshold — with no significant regression on "
            f"either advisory class.")
    elif macro["significant"]:
        lines.append(
            f"**Not really.** The {sel} beats the RandomForest by "
            f"{macro['observed_diff']:+.4f} macro-F1 with a CI of "
            f"[{macro['ci_low']:+.4f}, {macro['ci_high']:+.4f}] that excludes zero, so "
            f"the gain is real — but it is below the {MATERIAL} practical threshold. "
            f"Across {payload['counts']['test_observed']:,} test rows a bootstrap "
            f"resolves differences far smaller than anything that changes a decision.")
    else:
        lines.append(
            f"**No.** The {sel} moves macro-F1 by {macro['observed_diff']:+.4f} against "
            f"the RandomForest, {CI}% CI [{macro['ci_low']:+.4f}, {macro['ci_high']:+.4f}] "
            f"— the interval contains zero. On this data a 2-layer sequence model with "
            f"MC dropout does not measurably outperform a random forest reading a single "
            f"timestep.")

    if pers_f1 is not None:
        p = tests.get("Persistence", {}).get("macro_f1")
        if p:
            lines.append(
                f"Against the zero-parameter persistence rule the {sel} is "
                f"{p['observed_diff']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}]"
                + (", which does clear the floor."
                   if p["significant"] and p["observed_diff"] > 0
                   else ", which does **not** clear the floor — the headline number "
                        "should be read with that in mind."))

    lines.append(
        f"For the record: {sel} {dl_f1:.4f}, RandomForest {rf_f1:.4f}"
        + (f", persistence {pers_f1:.4f}." if pers_f1 is not None else ".")
        + f" The forest trains in ~15 s on CPU and needs no MC passes; the {sel} took "
          f"{payload['trained'][sel]['train_minutes']:.0f} min on GPU plus "
          f"{cfg.mc_passes} inference passes per prediction. On an ESP32 that gap is the "
          f"whole argument, and it has to be paid for with accuracy that is not here.")

    lines.append(
        "**What MC dropout does buy**, independent of the accuracy question, is a "
        "calibrated confidence per prediction — which is what Phase 6's advisory layer "
        "needs to say \"I am not sure\" instead of guessing. A random forest's vote "
        "share is not a substitute. If the sequence model is kept, keep it for the "
        "uncertainty, and say so rather than implying an accuracy win that the "
        "bootstrap does not support.")
    return "\n\n".join(lines)


# ------------------------------------------------------------------- lr sweep


LR_GRID = (3e-3, 1e-3, 3e-4, 1e-4)


def lr_sweep(cfg: DLConfig, grid=LR_GRID, epochs: int = 25, patience: int = 6,
             verbose: bool = True) -> dict:
    """Validation macro-F1 across learning rates, for both architectures.

    This exists because the first full run peaked at epoch 2-4 and then declined while
    training loss kept falling. That pattern can mean the task is genuinely hard, or it
    can mean the step size is too large -- and a negative result published from an
    under-trained model is not a negative result. Selection stays on validation, so
    choosing the rate this way costs nothing from the test split.
    """
    say = print if verbose else (lambda *a, **k: None)
    device = pick_device()
    train, val = load_split(cfg, "train"), load_split(cfg, "val")
    from dataclasses import replace

    out = {}
    say(f"{'arch':12s} {'lr':>8s} {'val macro-F1':>13s} {'@epoch':>7s} {'min':>6s}")
    for arch in cfg.architectures:
        out[arch] = {}
        for lr in grid:
            r = train_model(arch, replace(cfg, lr=lr, epochs=epochs, patience=patience),
                            train, val, device, verbose=False)
            out[arch][f"{lr:g}"] = {"val_macro_f1": r["val_macro_f1"],
                                    "best_epoch": r["best_epoch"],
                                    "epochs_run": r["epochs_run"],
                                    "minutes": r["train_minutes"]}
            say(f"{arch:12s} {lr:8.0e} {r['val_macro_f1']:13.4f} {r['best_epoch']:7d} "
                f"{r['train_minutes']:6.1f}")
    return {"grid": [f"{lr:g}" for lr in grid], "epochs": epochs,
            "patience": patience, "results": out}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the write-up from saved metrics, no retraining")
    ap.add_argument("--lr-sweep", action="store_true",
                    help="validation macro-F1 across learning rates, then stop")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, horizon=args.horizon)

    if args.report_only:
        # Rebuild the write-up from the saved metrics. Everything build_report reads is
        # in there except the fitted models, which it never touches.
        m = json.loads(cfg.metrics_path.read_text())
        payload = {
            "config": cfg, "selected": m["selected"], "results": m["results"],
            "tests": m["tests"], "calibration": m["calibration"],
            "uncertainty": m["uncertainty"], "counts": m["counts"],
            "n_boot": m["n_boot"], "trained": m["training"],
        }
        if not args.no_write:
            cfg.report_path.write_text(build_report(payload))
            print(f"rewrote {cfg.report_path.relative_to(REPO_ROOT)}")
        return 0

    if args.epochs is not None:
        from dataclasses import replace
        cfg = replace(cfg, epochs=args.epochs)

    if args.lr_sweep:
        sweep = lr_sweep(cfg)
        if not args.no_write:
            path = cfg.reports_dir / "metrics" / f"dl_lr_sweep_h{cfg.horizon}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(sweep, indent=2))
            print(f"wrote {path.relative_to(REPO_ROOT)}")
        return 0

    run(cfg, write=not args.no_write, n_boot=args.n_boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
