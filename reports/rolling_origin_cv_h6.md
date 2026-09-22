# Rolling-origin cross-validation — horizon 6 h

5 expanding-window folds over the full 4-year span, built
to answer one question the single chronological split could not: **how often does a
learned model actually beat persistence?**

Every earlier phase evaluated on one split, and validation and test disagreed — the
unweighted forest wins on test and loses on validation, the class-weighted forest does
the reverse. That disagreement is unresolvable with one split, because the answer
depends on which season lands in the evaluation block.

---

## 1. Protocol

- **Expanding window.** Fold *k* trains on everything before its cutoff and evaluates
  on the contiguous block after it. Training data grows with *k*.
- **Cutoffs** are spread across the timeline: the first
  40% is training-only, the remainder split into 5
  equal contiguous evaluation blocks. Each block therefore covers a different part of
  the year, which is the point.
- **Embargo of 24 hours** between train and eval. Training uses
  samples whose *target* precedes the cutoff; evaluation uses samples whose
  24-hour input window begins at or after it. Without the gap, an evaluation
  sample's input window would contain hours the model was trained on as labels.
- **Scaler refit inside every fold**, on that fold's training rows only. The published
  splits are standardised with global-train statistics, which would leak future
  information into every fold; features are inverted to physical units and
  re-standardised per fold.
- **Station grouping** is unchanged: all 12 stations share one hourly axis, so a
  cutoff divides every station at the same moment, and no sample's window spans two
  stations (guaranteed at construction in Phase 2).
- **Observed labels only** in evaluation (`is_imputed_pm25 == False`).

| Fold | Cutoff | Evaluation block | Months covered | Train n | Eval n (observed) |
| --- | --- | --- | --- | --- | --- |
| 1 | 2014-10-05 | 2014-10-06 .. 2015-03-29 | Jan, Feb, Mar, Oct, Nov, Dec | 167,808 | 48,874 |
| 2 | 2015-03-29 | 2015-03-30 .. 2015-09-20 | Mar, Apr, May, Jun, Jul, Aug, Sep | 218,172 | 48,948 |
| 3 | 2015-09-20 | 2015-09-21 .. 2016-03-14 | Jan, Feb, Mar, Sep, Oct, Nov, Dec | 268,536 | 49,105 |
| 4 | 2016-03-14 | 2016-03-15 .. 2016-09-06 | Mar, Apr, May, Jun, Jul, Aug, Sep | 318,912 | 48,822 |
| 5 | 2016-09-06 | 2016-09-07 .. 2017-02-28 | Jan, Feb, Sep, Oct, Nov, Dec | 369,276 | 49,253 |

---

## 2. Per-fold macro-F1

| Fold | Persistence | RandomForest (unweighted) | RandomForest (class_weight=balanced) | Δ RandomForest | Δ RandomForest (cw) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.4773 | 0.4796 | 0.4703 | **+0.0023** | -0.0070 |
| 2 | 0.4353 | 0.3855 | 0.4363 | -0.0498 | **+0.0011** |
| 3 | 0.5155 | 0.5077 | 0.5080 | -0.0078 | -0.0075 |
| 4 | 0.5108 | 0.4838 | 0.5260 | -0.0270 | **+0.0152** |
| 5 | 0.5051 | 0.5138 | 0.5002 | **+0.0086** | -0.0049 |
| **mean** | 0.4888 | 0.4741 | 0.4882 | -0.0147 | -0.0006 |
| **std** | 0.0334 | 0.0517 | 0.0352 | 0.0238 | 0.0095 |

---

## 3. Fold-level comparison against persistence

The unit of analysis is the **fold**, not the sample. A percentile bootstrap is
inappropriate here: it assumes exchangeable draws within one comparison, and with
5 folds it can only ever resample the same 5 numbers. A **Wilcoxon signed-rank
test** on the 5 paired deltas is the right instrument — it asks whether the fold-level
differences are consistently one-signed without assuming normality.

| Model | Mean Δ | Std Δ | Folds won | Wilcoxon p (2-sided) | p (1-sided) | p₁ < 0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| RandomForest (unweighted) | -0.0147 | 0.0238 | **2/5** | 0.4375 | 0.8438 | no |
| RandomForest (class_weight=balanced) | -0.0006 | 0.0095 | **2/5** | 0.8125 | 0.6875 | no |

> **A 5-pair signed-rank test has a hard resolution floor.** The smallest two-sided
> p it can return is **0.0625** and the smallest one-sided p is
> **0.03125** — reached only when all 5 deltas share a sign. So
> **no two-sided result here can ever reach α = 0.05, regardless of the data.** That
> is a property of the sample size, not evidence of absence. The one-sided test can
> reach significance, and the **folds-won count is the more informative statistic** at
> this n.

### Class support per evaluation block

An F1 of 0.0 on a class with no samples is not a model result. Support is reported
first so the next table can be read correctly.

| Class | F1 | F2 | F3 | F4 | F5 | Total |
| --- | --- | --- | --- | --- | --- | --- |
| Good | 9,994 | 7,055 | 10,384 | 6,857 | 8,826 | 43,116 |
| Moderate | 9,106 | 13,798 | 11,257 | 12,302 | 9,988 | 56,451 |
| Unhealthy (sensitive) | 4,286 | 7,962 | 5,368 | 8,623 | 4,530 | 30,769 |
| Unhealthy | 14,797 | 17,590 | 12,280 | 18,241 | 14,834 | 77,742 |
| **Very unhealthy** | 6,901 | 2,427 | 5,545 | 2,180 | 7,390 | 24,443 |
| **Hazardous** | 3,790 | 116 | 4,271 | 619 | 3,685 | 12,481 |

Every advisory class appears in at least 5 of 5 evaluation blocks, so the rare-class comparison below is interpretable.

### Rare-class F1 across folds

Means are taken over **folds with non-zero support only**; `n/a` marks a block where
the class does not occur.

| Class | Model | Mean (usable folds) | Usable folds | Per fold |
| --- | --- | --- | --- | --- |
| Very unhealthy | Persistence | 0.4552 | 5/5 | 0.468  0.390  0.485  0.416  0.517 |
| Very unhealthy | RandomForest (unweighted) | 0.4532 | 5/5 | 0.507  0.368  0.516  0.345  0.530 |
| Very unhealthy | RandomForest (class_weight=balanced) | 0.4571 | 5/5 | 0.488  0.428  0.464  0.427  0.479 |
| Hazardous | Persistence | 0.5192 | 5/5 | 0.593  0.169  0.638  0.606  0.590 |
| Hazardous | RandomForest (unweighted) | 0.5136 | 5/5 | 0.608  0.074  0.665  0.606  0.614 |
| Hazardous | RandomForest (class_weight=balanced) | 0.5247 | 5/5 | 0.597  0.125  0.668  0.648  0.585 |

---

## 4. Verdict

**RandomForest (unweighted) beats persistence in only 2/5 folds** (mean -0.0147, std 0.0238). The sign of the effect changes with the evaluation block, which is precisely the instability the single-split analysis suspected.

**RandomForest (class_weight=balanced) beats persistence in only 2/5 folds** (mean -0.0006, std 0.0095). The sign of the effect changes with the evaluation block, which is precisely the instability the single-split analysis suspected.

**The instability is confirmed, and it is not a single-split artifact.** No model clears the persistence floor in a majority of folds (best: `RandomForest (unweighted)` at 2/5). Five evaluation blocks spanning different seasons give the same answer the single split hinted at: at a 6-hour horizon on these nine channels, a learned model is not reliably better than assuming the next six hours look like now. This upgrades the finding from a possible artifact to a robust multi-fold negative result, which is the publishable version of it.

Reproduce with `python -m src.models.rolling_cv`.
