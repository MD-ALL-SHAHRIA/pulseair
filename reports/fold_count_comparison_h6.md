# Fold-count sensitivity — Beijing rolling-origin CV, horizon 6 h

The published rolling-origin result uses **five folds**. This report asks whether its
conclusion is an artefact of low statistical power, by re-running the same protocol at
a higher fold count and placing both side by side. **The five-fold numbers are not
replaced**; they are quoted throughout the thesis and remain the headline.

---

## 1. Choosing a fold count

Three criteria, swept over candidates:

| Folds | Smallest eval block | Shortest block | Blocks touching winter | Two-sided Wilcoxon floor |  |
| --- | --- | --- | --- | --- | --- |
| **5** | 50,364 | 174d | 3/5 | 0.06250 **> α** | 5-fold: published |
| 6 | 41,964 | 145d | 5/6 | 0.03125 |  |
| 7 | 35,976 | 124d | 5/7 | 0.01562 |  |
| **8** | 31,476 | 109d | 6/8 | 0.00781 | 8-fold: chosen |
| 9 | 27,972 | 97d | 6/9 | 0.00391 |  |
| 10 | 25,176 | 87d | 6/10 | 0.00195 |  |

**Two findings from the sweep itself.**

**The published five-fold design cannot produce a two-sided significant result.** Its
two-sided Wilcoxon floor is 0.0625, *above* α = 0.05. No data could have made a
two-sided test significant at five folds. The one-sided floor, 0.03125, is usable, and
the existing reports state both — but any two-sided p quoted at five folds is bounded
by construction rather than informative.

**No fold count achieves winter contact in every evaluation block.** Beijing's
evaluated portion spans about 2.4 years after the initial training fraction is held
out, so contiguous blocks shorter than a year cannot all reach a November–February
window. Eight folds gives the best absolute count at 6/8. This is a
property of the record length, not of the protocol, and it is stated rather than
presented as clean.

**Eight folds was chosen**: the lowest two-sided floor among candidates that keep
evaluation blocks above thirty thousand samples and roughly a season long, with the
highest absolute winter contact. Training-window size is not a binding constraint —
the window expands, so its minimum is always the initial fraction regardless of fold
count.

## 2. Five folds against eight

| Model | 5-fold: won | 5-fold: mean Δ | 8-fold: won | 8-fold: mean Δ | 8-fold two-sided p |
| --- | --- | --- | --- | --- | --- |
| LSTM | — | — | 2/8 | -0.0236 | 0.1094 |
| RF (class_weight=balanced) | 2/5 | -0.0006 | 4/8 | +0.0019 | 0.8438 |
| RF (unweighted) | 2/5 | -0.0147 | 2/8 | -0.0137 | 0.2500 |
| Transformer | — | — | 2/8 | -0.0235 | 0.0547 |
| XGBoost | — | — | 0/8 | -0.0290 | 0.0078**\*** |

`*` marks significance at 0.05 two-sided, which is only attainable at the higher fold
count.

## 3. Does the conclusion change?

**No.** At 8 folds a majority is 5 or more. The best any model achieves is
**4/8** (RF (class_weight=balanced)), which is
exactly half.
The claim that **no Beijing-trained model beats persistence in a majority of
rolling-origin folds survives at higher power**, and now across five model families
rather than two.

Two refinements the added power buys:

- **RF (class_weight=balanced) moves from
  2/5 to 4/8.** Closer to parity with
  persistence, still not ahead of it, and its mean difference
  (+0.0019) remains far below anything of practical consequence.
- **XGBoost
  is now *significantly worse* than persistence**
  (two-sided p < 0.05). That is newly detectable only because the
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
