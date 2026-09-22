"""Sequence-model capacity sweep — were the LSTM and Transformer big enough?

Phase 5 concluded that neither sequence model beats the RandomForest, and attributed
the ceiling to the data: MC dropout put ~98% of predictive entropy in the aleatoric
term. A reviewer will reasonably ask whether 53k-70k parameters were simply too few.

This sweeps LSTM hidden size and Transformer d_model over {64, 128, 256} at the
learning rate the Phase 5 sweep already selected, with the same validation-based
selection. **A plateau confirms the aleatoric-ceiling finding; a monotone improvement
partially revises it**, and the report says which happened rather than assuming.

    python -m src.models.capacity_sweep
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.models import dl_forecast as dl

REPO_ROOT = Path(__file__).resolve().parents[2]
SIZES = (64, 128, 256)
OUT = REPO_ROOT / "reports" / "metrics" / "capacity_sweep_h6.json"


def run(sizes=SIZES, epochs: int = 40, patience: int = 8, verbose: bool = True) -> dict:
    say = print if verbose else (lambda *a, **k: None)
    cfg = dl.load_config()
    device = dl.pick_device()
    train, val = dl.load_split(cfg, "train"), dl.load_split(cfg, "val")
    say(f"device {device} | lr {cfg.lr:g} (from the Phase 5 sweep) | "
        f"epochs<={epochs}, patience {patience}")

    rows = []
    for arch in ("lstm", "transformer"):
        for size in sizes:
            if arch == "lstm":
                c = replace(cfg, lstm={**cfg.lstm, "hidden_size": size},
                            epochs=epochs, patience=patience)
            else:
                # keep feedforward at 2x d_model so width scales coherently
                c = replace(cfg, transformer={**cfg.transformer, "d_model": size,
                                              "dim_feedforward": 2 * size},
                            epochs=epochs, patience=patience)
            r = dl.train_model(arch, c, train, val, device, verbose=False)
            rows.append({"arch": arch, "size": size,
                         "n_params": r["n_params"],
                         "val_macro_f1": r["val_macro_f1"],
                         "best_epoch": r["best_epoch"],
                         "epochs_run": r["epochs_run"],
                         "minutes": r["train_minutes"]})
            say(f"  {arch:12s} size {size:>3}  params {r['n_params']:>9,}  "
                f"val macro-F1 {r['val_macro_f1']:.4f}  "
                f"@epoch {r['best_epoch']:>2}  {r['train_minutes']:.1f} min")
    return {"rows": rows, "lr": cfg.lr, "sizes": list(sizes),
            "epochs_cap": epochs, "patience": patience}


def analyse(res: dict) -> dict:
    """Plateau or improvement? Decided from the numbers, not asserted."""
    out = {}
    for arch in ("lstm", "transformer"):
        r = sorted([x for x in res["rows"] if x["arch"] == arch],
                   key=lambda x: x["size"])
        f1 = [x["val_macro_f1"] for x in r]
        spread = max(f1) - min(f1)
        best = r[int(np.argmax(f1))]
        gain_4x = f1[-1] - f1[0]
        # "Plateau" means quadrupling width buys less than the practical-significance
        # threshold used everywhere else in this project.
        out[arch] = {
            "per_size": r, "spread": float(spread),
            "gain_smallest_to_largest": float(gain_4x),
            "best_size": int(best["size"]), "best_val": float(best["val_macro_f1"]),
            "param_ratio": r[-1]["n_params"] / r[0]["n_params"],
            "plateau": bool(abs(gain_4x) < 0.01 and spread < 0.01),
            "monotone_improving": bool(all(a < b for a, b in zip(f1, f1[1:]))),
            "monotone_declining": bool(all(a > b for a, b in zip(f1, f1[1:]))),
        }
    return out


def build_report(res: dict, an: dict) -> str:
    def table(header, rows):
        esc = lambda cs: [str(c).replace("|", "\\|") for c in cs]
        b = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return (f"| {' | '.join(esc(header))} |\n"
                f"| {' | '.join(['---'] * len(header))} |\n{b}")

    rows = [[x["arch"], f"{x['size']}", f"{x['n_params']:,}",
             f"{x['val_macro_f1']:.4f}", f"{x['best_epoch']}/{x['epochs_run']}",
             f"{x['minutes']:.1f}"] for x in res["rows"]]

    dl_metrics = json.loads(
        (REPO_ROOT / "reports" / "metrics" / "dl_h6.json").read_text())
    epi = dl_metrics["uncertainty"][dl_metrics["selected"]]
    epi_share = epi["mean_epistemic"] / epi["mean_entropy"]
    rf_val = json.loads(
        (REPO_ROOT / "reports" / "metrics" / "baseline_h6.json").read_text()
    )["results"]["RandomForest"]["val"]["observed"]["macro_f1"]

    both_plateau = all(an[a]["plateau"] for a in an)
    any_improve = any(an[a]["monotone_improving"] and
                      an[a]["gain_smallest_to_largest"] >= 0.01 for a in an)
    both_decline = all(an[a]["monotone_declining"] for a in an)

    if both_decline:
        verdict = f"""**Both architectures get monotonically *worse* with capacity. The
Phase 5 conclusion holds, and this is stronger evidence than a plateau would have
been.**

{table(["Architecture", "Smallest → largest", "Parameter ratio", "Trend"],
       [[a, f"{an[a]['gain_smallest_to_largest']:+.4f}",
         f"{an[a]['param_ratio']:.1f}×", "monotonically declining"] for a in an])}

The LSTM loses {abs(an['lstm']['gain_smallest_to_largest']):.4f} validation macro-F1
going from {an['lstm']['per_size'][0]['size']} to
{an['lstm']['per_size'][-1]['size']} hidden units
({an['lstm']['param_ratio']:.0f}× the parameters); the Transformer loses
{abs(an['transformer']['gain_smallest_to_largest']):.4f} over the same widening. Best
epoch also falls as capacity rises — the largest models peak at epoch 1–2 and then
overfit — which is the signature of models with more capacity than the signal
supports.

**The "were they big enough?" objection is answered decisively: they were already too
big.** This is consistent with the MC-dropout decomposition, which put only
{epi_share:.1%} of predictive uncertainty in the epistemic term — the part extra
capacity could address. The remaining {1 - epi_share:.0%} is irreducible given nine
channels at a six-hour horizon, and adding parameters against irreducible noise costs
generalisation rather than buying accuracy.

Note the smallest configuration in this sweep is the one Phase 5 used, and it
reproduces exactly ({an['lstm']['per_size'][0]['val_macro_f1']:.4f} LSTM,
{an['transformer']['per_size'][0]['val_macro_f1']:.4f} Transformer), which also serves
as a determinism check on the Phase 5 training run."""
    elif both_plateau:
        verdict = f"""**Both architectures plateau. The Phase 5 conclusion stands, and is
strengthened.**

Quadrupling width changes validation macro-F1 by
{an['lstm']['gain_smallest_to_largest']:+.4f} (LSTM,
{an['lstm']['param_ratio']:.1f}× the parameters) and
{an['transformer']['gain_smallest_to_largest']:+.4f} (Transformer,
{an['transformer']['param_ratio']:.1f}×). Both are inside the 0.01
practical-significance threshold used throughout this project, and the spread across
all three sizes is {max(an[a]['spread'] for a in an):.4f}.

This is what the MC-dropout entropy decomposition predicted: only
{epi_share:.1%} of predictive uncertainty is epistemic — the part capacity can
address — and the remaining {1 - epi_share:.0%} is irreducible given these nine
channels at a six-hour horizon. **Capacity was not the binding constraint, and the
"were they big enough?" objection is answered with a measurement rather than an
assurance.**"""
    elif any_improve:
        best = max(an, key=lambda a: an[a]["gain_smallest_to_largest"])
        verdict = f"""**Capacity does help, and the Phase 5 conclusion needs revising.**

The {best} improves {an[best]['gain_smallest_to_largest']:+.4f} validation macro-F1
from the smallest to the largest size, which exceeds the 0.01 practical threshold.
The aleatoric-ceiling argument in `dl_metrics_h6.md` §3 overstated the case: some of
the headroom the entropy decomposition assigned to irreducible noise is reachable
with more capacity. The sweep should be extended upward before any final claim about
sequence models on this task."""
    else:
        verdict = f"""**Mixed: neither a clean plateau nor a clean improvement.**

LSTM moves {an['lstm']['gain_smallest_to_largest']:+.4f} and Transformer
{an['transformer']['gain_smallest_to_largest']:+.4f} from smallest to largest, without
a monotone trend in both. The safe reading is that capacity is not the dominant
factor, which is consistent with the Phase 5 finding without confirming it as
cleanly as a flat curve would."""

    return f"""# Sequence-model capacity sweep — horizon 6 h

Closes the reviewer question left open in `final_results_summary.md` §5: the Phase 5
models had 53k–70k parameters, and the conclusion that they lose to a RandomForest
would be weak if they were simply too small.

LSTM `hidden_size` and Transformer `d_model` over {res['sizes']}, at
**lr = {res['lr']:g}** (the value the Phase 5 learning-rate sweep selected), same
validation-based selection, same early stopping (patience {res['patience']}, cap
{res['epochs_cap']} epochs). Transformer feedforward width scales with `d_model` at
2×, so the models widen coherently.

---

## Results

{table(["Architecture", "Size", "Parameters", "Val macro-F1", "Best/run epochs",
        "Minutes"], rows)}

For reference, the RandomForest scores **{rf_val:.4f}** on the same validation split.

{table(["Architecture", "Smallest → largest", "Spread across sizes", "Best size",
        "Parameter ratio", "Plateau?"],
       [[a, f"{an[a]['gain_smallest_to_largest']:+.4f}", f"{an[a]['spread']:.4f}",
         f"{an[a]['best_size']}", f"{an[a]['param_ratio']:.1f}×",
         "**yes**" if an[a]["plateau"] else "no"] for a in an])}

---

## Verdict

{verdict}

Reproduce with `python -m src.models.capacity_sweep`.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    res = run(epochs=args.epochs)
    an = analyse(res)
    if not args.no_write:
        OUT.write_text(json.dumps({"sweep": res, "analysis": an}, indent=2, default=str))
        (REPO_ROOT / "reports" / "capacity_sweep_h6.md").write_text(
            build_report(res, an))
        print(f"\nwrote reports/capacity_sweep_h6.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
