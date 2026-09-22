# Rolling-origin cross-validation — Bangladesh, horizon 6 h

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
  30% is training-only, the remainder split into 5
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
| 1 | 2023-08-02 | 2023-08-03 .. 2024-01-18 | Jan, Aug, Sep, Oct, Nov, Dec | 34,700 | 16,108 |
| 2 | 2024-01-18 | 2024-01-19 .. 2024-07-05 | Jan, Feb, Mar, Apr, May, Jun, Jul | 50,904 | 16,112 |
| 3 | 2024-07-05 | 2024-07-06 .. 2024-12-21 | Jul, Aug, Sep, Oct, Nov, Dec | 67,112 | 16,108 |
| 4 | 2024-12-21 | 2024-12-22 .. 2025-06-07 | Jan, Feb, Mar, Apr, May, Jun, Dec | 83,316 | 16,112 |
| 5 | 2025-06-07 | 2025-06-08 .. 2025-11-23 | Jun, Jul, Aug, Sep, Oct, Nov | 99,524 | 16,112 |

---

## 2. Per-fold macro-F1

| Fold | Persistence | RandomForest (unweighted) | RandomForest (class_weight=balanced) | Δ RandomForest | Δ RandomForest (cw) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.4214 | 0.4622 | 0.4582 | **+0.0408** | **+0.0368** |
| 2 | 0.4458 | 0.4603 | 0.4517 | **+0.0145** | **+0.0060** |
| 3 | 0.4841 | 0.4745 | 0.5120 | -0.0096 | **+0.0279** |
| 4 | 0.4566 | 0.4505 | 0.5233 | -0.0061 | **+0.0667** |
| 5 | 0.3806 | 0.4312 | 0.4203 | **+0.0506** | **+0.0398** |
| **mean** | 0.4377 | 0.4557 | 0.4731 | +0.0181 | +0.0354 |
| **std** | 0.0391 | 0.0162 | 0.0433 | 0.0271 | 0.0219 |

---

## 3. Fold-level comparison against persistence

The unit of analysis is the **fold**, not the sample. A percentile bootstrap is
inappropriate here: it assumes exchangeable draws within one comparison, and with
5 folds it can only ever resample the same 5 numbers. A **Wilcoxon signed-rank
test** on the 5 paired deltas is the right instrument — it asks whether the fold-level
differences are consistently one-signed without assuming normality.

| Model | Mean Δ | Std Δ | Folds won | Wilcoxon p (2-sided) | p (1-sided) | p₁ < 0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| RandomForest (unweighted) | +0.0181 | 0.0271 | **3/5** | 0.3125 | 0.1562 | no |
| RandomForest (class_weight=balanced) | +0.0354 | 0.0219 | **5/5** | 0.0625 | 0.0312 | yes |

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
| Good | 2,877 | 1,509 | 5,398 | 1,076 | 5,762 | 16,622 |
| Moderate | 6,120 | 6,996 | 5,485 | 5,124 | 7,341 | 31,066 |
| Unhealthy (sensitive) | 3,449 | 3,522 | 1,942 | 3,894 | 1,550 | 14,357 |
| Unhealthy | 3,662 | 4,076 | 3,210 | 5,622 | 1,453 | 18,023 |
| **Very unhealthy** | 0 | 9 | 73 | 394 | 6 | 482 |
| **Hazardous** | 0 | 0 | 0 | 2 | 0 | 2 |

> **Hazardous cannot be evaluated on this dataset.** Hazardous has 2 samples across all 5 evaluation blocks combined (4 blocks contain none). No number of folds fixes this — the class is not in the data. Any F1 or coverage figure for these classes below is reported for completeness and should not be cited as a result. **Phase 11b validates these classes on reference-grade ground-truth data instead** (`reports/dhaka_ground_truth_model_h6.md`).

### Rare-class F1 across folds

Means are taken over **folds with non-zero support only**; `n/a` marks a block where
the class does not occur.

| Class | Model | Mean (usable folds) | Usable folds | Per fold |
| --- | --- | --- | --- | --- |
| Very unhealthy | Persistence | 0.2229 | 4/5 | n/a  0.111  0.329  0.452  0.000 |
| Very unhealthy | RandomForest (unweighted) | 0.0415 | 4/5 | n/a  0.000  0.000  0.166  0.000 |
| Very unhealthy | RandomForest (class_weight=balanced) | 0.2196 | 4/5 | n/a  0.023  0.316  0.539  0.000 |
| Hazardous | Persistence | 0.0000 | 1/5 | n/a  n/a  n/a  0.000  n/a |
| Hazardous | RandomForest (unweighted) | 0.0000 | 1/5 | n/a  n/a  n/a  0.000  n/a |
| Hazardous | RandomForest (class_weight=balanced) | 0.0000 | 1/5 | n/a  n/a  n/a  0.000  n/a |

---

## 4. Verdict

**RandomForest (unweighted) beats persistence in only 3/5 folds** (mean +0.0181, std 0.0271). The sign of the effect changes with the evaluation block, which is precisely the instability the single-split analysis suspected.

**RandomForest (class_weight=balanced) beats persistence in all 5/5 folds** (mean +0.0354, one-sided Wilcoxon p = 0.03125). Every evaluation block, every season. This is the consistency a single split could not demonstrate.

**The open question from the single-split analysis is resolved.** `RandomForest (class_weight=balanced)` clears the persistence floor in every fold, so the earlier validation/test disagreement was an artifact of one arbitrary chronological cut rather than a property of the model. The thesis can state that a learned model beats persistence at h=6, with 5-fold rolling-origin evidence behind it — and should report the per-fold spread rather than a single number.

Reproduce with `python -m src.models.rolling_cv`.
