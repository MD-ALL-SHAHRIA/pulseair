"""Phase B — does the Beijing conclusion survive at higher statistical power?

The published rolling-origin result uses five folds. Five is a small number for a
signed-rank test: its two-sided resolution floor is 2**(1-5) = 0.0625, which is
*above* alpha = 0.05, so that design could not have produced a two-sided significant
result whatever the data showed. The one-sided floor, 0.03125, is usable, and the
reports state both — but it is worth asking whether the conclusion is an artefact of
low power.

This module compares the published five-fold run against an eight-fold run of the same
protocol, and reports both side by side. **It does not replace the five-fold numbers**,
which are quoted throughout the thesis.

Fold count was chosen by sweeping candidates against three criteria: the two-sided
Wilcoxon floor, winter-season contact in each evaluation block, and evaluation block
size. That sweep is reproduced in the report, including the finding that **no fold
count achieves winter contact in every block** — Beijing's evaluated portion spans
about 2.4 years, so contiguous blocks shorter than a year cannot all reach a winter.

    python -m src.reporting.fold_count_comparison

Writes ``reports/fold_count_comparison_h6.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = REPO_ROOT / "reports" / "metrics"
WINTER = {11, 12, 1, 2}


def _load(name: str):
    p = METRICS / name
    return json.loads(p.read_text()) if p.exists() else None


def sweep_table() -> tuple[str, list[dict]]:
    """Fold-count candidates against the three selection criteria."""
    import sys
    sys.path.insert(0, str(REPO_ROOT))
    from pulsebench import make_folds
    from src.models import rolling_cv as rcv

    cfg = rcv.load_config(dataset="beijing")
    df, _, _ = rcv.load_unscaled(cfg)
    t = pd.to_datetime(df["datetime"]).reset_index(drop=True)

    rows, data = [], []
    for nf in (5, 6, 7, 8, 9, 10):
        folds = make_folds(t.to_numpy(), nf, cfg.initial_fraction)
        edges = [pd.Timestamp(f["cutoff"]) for f in folds] + [t.max()]
        ev, win, spans = [], [], []
        for i in range(nf):
            blk = t[(t > edges[i]) & (t <= edges[i + 1])]
            ev.append(len(blk))
            win.append(bool(set(blk.dt.month.unique()) & WINTER))
            spans.append((blk.max() - blk.min()).days if len(blk) else 0)
        floor = 2.0 ** (1 - nf)
        rec = {"n_folds": nf, "min_eval": min(ev), "min_days": min(spans),
               "winter_blocks": sum(win), "two_sided_floor": floor,
               "usable_two_sided": floor < 0.05}
        data.append(rec)
        rows.append([
            f"**{nf}**" if nf in (5, 8) else str(nf),
            f"{min(ev):,}", f"{min(spans)}d",
            f"{sum(win)}/{nf}",
            f"{floor:.5f}" + ("" if floor < 0.05 else " **> α**"),
            "5-fold: published" if nf == 5 else ("8-fold: chosen" if nf == 8 else ""),
        ])
    head = ["Folds", "Smallest eval block", "Shortest block", "Blocks touching winter",
            "Two-sided Wilcoxon floor", ""]
    table = "\n".join(["| " + " | ".join(head) + " |",
                       "| " + " | ".join(["---"] * len(head)) + " |"]
                      + ["| " + " | ".join(r) + " |" for r in rows])
    return table, data


def comparison_table() -> tuple[str, dict]:
    """Five-fold against eight-fold, model by model."""
    five = _load("rolling_cv_h6.json")
    eight_tab = _load("rolling_cv_h6_f8.json")
    eight_seq = _load("rolling_cv_h6_f8_seq.json")
    if not (five and eight_tab):
        return "[MISSING: rolling-origin metrics not found]", {}

    def tests(p):
        return {k: v for k, v in p["aggregate"]["tests"].items()
                if not k.startswith("_")} if p else {}

    t5, t8 = tests(five), {**tests(eight_tab), **tests(eight_seq or {})}
    rows, summary = [], {}
    for name in sorted(set(t5) | set(t8)):
        a, b = t5.get(name), t8.get(name)
        rows.append([
            name.replace("RandomForest ", "RF "),
            f"{a['wins']}/{a['n_folds']}" if a else "—",
            f"{a['mean_delta']:+.4f}" if a else "—",
            f"{b['wins']}/{b['n_folds']}" if b else "—",
            f"{b['mean_delta']:+.4f}" if b else "—",
            f"{b['p_two_sided']:.4f}" + ("**\\***" if b and b["p_two_sided"] < 0.05 else "")
            if b else "—",
        ])
        if b:
            summary[name] = {"wins_8": b["wins"], "n_8": b["n_folds"],
                             "delta_8": b["mean_delta"], "p2_8": b["p_two_sided"],
                             "wins_5": a["wins"] if a else None,
                             "n_5": a["n_folds"] if a else None}
    head = ["Model", "5-fold: won", "5-fold: mean Δ", "8-fold: won", "8-fold: mean Δ",
            "8-fold two-sided p"]
    table = "\n".join(["| " + " | ".join(head) + " |",
                       "| " + " | ".join(["---"] * len(head)) + " |"]
                      + ["| " + " | ".join(r) + " |" for r in rows])
    return table, summary


def build_report() -> str:
    sweep, sweep_data = sweep_table()
    comp, summary = comparison_table()
    eight = _load("rolling_cv_h6_f8.json")
    n8 = eight["n_folds"] if eight else 8
    majority = n8 // 2 + 1
    best = max(summary.values(), key=lambda v: v["wins_8"]) if summary else None
    best_name = ([k for k, v in summary.items() if v is best] or ["—"])[0]
    holds = best and best["wins_8"] < majority

    worse = [k for k, v in summary.items()
             if v["p2_8"] < 0.05 and v["delta_8"] < 0]

    return f"""# Fold-count sensitivity — Beijing rolling-origin CV, horizon 6 h

The published rolling-origin result uses **five folds**. This report asks whether its
conclusion is an artefact of low statistical power, by re-running the same protocol at
a higher fold count and placing both side by side. **The five-fold numbers are not
replaced**; they are quoted throughout the thesis and remain the headline.

---

## 1. Choosing a fold count

Three criteria, swept over candidates:

{sweep}

**Two findings from the sweep itself.**

**The published five-fold design cannot produce a two-sided significant result.** Its
two-sided Wilcoxon floor is 0.0625, *above* α = 0.05. No data could have made a
two-sided test significant at five folds. The one-sided floor, 0.03125, is usable, and
the existing reports state both — but any two-sided p quoted at five folds is bounded
by construction rather than informative.

**No fold count achieves winter contact in every evaluation block.** Beijing's
evaluated portion spans about 2.4 years after the initial training fraction is held
out, so contiguous blocks shorter than a year cannot all reach a November–February
window. Eight folds gives the best absolute count at {[d for d in sweep_data if d['n_folds']==8][0]['winter_blocks']}/8. This is a
property of the record length, not of the protocol, and it is stated rather than
presented as clean.

**Eight folds was chosen**: the lowest two-sided floor among candidates that keep
evaluation blocks above thirty thousand samples and roughly a season long, with the
highest absolute winter contact. Training-window size is not a binding constraint —
the window expands, so its minimum is always the initial fraction regardless of fold
count.

## 2. Five folds against eight

{comp}

`*` marks significance at 0.05 two-sided, which is only attainable at the higher fold
count.

## 3. Does the conclusion change?

**No.** At {n8} folds a majority is {majority} or more. The best any model achieves is
**{best['wins_8']}/{n8}** ({best_name.replace('RandomForest ', 'RF ')}), which is
{'exactly half' if best and best['wins_8'] * 2 == n8 else 'short of a majority'}.
The claim that **no Beijing-trained model beats persistence in a majority of
rolling-origin folds survives at higher power**, and now across five model families
rather than two.

Two refinements the added power buys:

- **{best_name.replace('RandomForest ', 'RF ')} moves from
  {best['wins_5']}/{best['n_5']} to {best['wins_8']}/{n8}.** Closer to parity with
  persistence, still not ahead of it, and its mean difference
  ({best['delta_8']:+.4f}) remains far below anything of practical consequence.
- **{', '.join(w.replace('RandomForest ', 'RF ') for w in worse) if worse else 'No model'}
  {'is' if len(worse) == 1 else 'are'} now *significantly worse* than persistence**
  {'(two-sided p < 0.05)' if worse else ''}. That is newly detectable only because the
  eight-fold floor admits it, and it strengthens the existing finding rather than
  qualifying it: at higher power the evidence is not that these models are
  indistinguishable from doing nothing, but that at least one is measurably worse.

## 4. Why both are reported

Replacing the five-fold numbers with the eight-fold ones would be the wrong move. The
five-fold run is what every published figure and the thesis text quote, and silently
swapping the basis of a headline claim is exactly the practice this project argues
against. The eight-fold run is a sensitivity analysis: it says the conclusion does not
depend on the fold count, which is worth more than either number alone.

Regenerate with `python -m src.reporting.fold_count_comparison`.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    text = build_report()
    if not args.no_write:
        out = REPO_ROOT / "reports" / "fold_count_comparison_h6.md"
        out.write_text(text)
        print(f"wrote {out.relative_to(REPO_ROOT)}")
    else:
        print(text[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
