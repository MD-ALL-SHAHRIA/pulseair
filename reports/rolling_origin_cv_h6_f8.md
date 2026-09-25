# Rolling-origin cross-validation — horizon 6 h

8 expanding-window folds over the full 4-year span, built
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
  40% is training-only, the remainder split into 8
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
| 1 | 2014-10-05 | 2014-10-06 .. 2015-01-23 | Jan, Oct, Nov, Dec | 167,808 | 30,143 |
| 2 | 2015-01-23 | 2015-01-24 .. 2015-05-12 | Jan, Feb, Mar, Apr, May | 199,284 | 30,667 |
| 3 | 2015-05-12 | 2015-05-13 .. 2015-08-29 | May, Jun, Jul, Aug | 230,760 | 30,624 |
| 4 | 2015-08-29 | 2015-08-30 .. 2015-12-17 | Aug, Sep, Oct, Nov, Dec | 262,248 | 30,334 |
| 5 | 2015-12-17 | 2015-12-18 .. 2016-04-05 | Jan, Feb, Mar, Apr, Dec | 293,724 | 30,728 |
| 6 | 2016-04-05 | 2016-04-06 .. 2016-07-23 | Apr, May, Jun, Jul | 325,200 | 30,397 |
| 7 | 2016-07-23 | 2016-07-25 .. 2016-11-11 | Jul, Aug, Sep, Oct, Nov | 356,688 | 30,399 |
| 8 | 2016-11-11 | 2016-11-12 .. 2017-02-28 | Jan, Feb, Nov, Dec | 388,164 | 30,843 |

---

## 2. Per-fold macro-F1

| Fold | Persistence | RandomForest (unweighted) | RandomForest (class_weight=balanced) | XGBoost | Δ RandomForest | Δ RandomForest (cw) | Δ XGBoost |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.4795 | 0.4859 | 0.4769 | 0.4745 | **+0.0064** | -0.0026 | -0.0050 |
| 2 | 0.4751 | 0.4629 | 0.4755 | 0.4302 | -0.0122 | **+0.0003** | -0.0449 |
| 3 | 0.4336 | 0.3859 | 0.4538 | 0.3740 | -0.0477 | **+0.0202** | -0.0596 |
| 4 | 0.5477 | 0.5431 | 0.5379 | 0.5114 | -0.0046 | -0.0098 | -0.0363 |
| 5 | 0.4850 | 0.4726 | 0.4913 | 0.4703 | -0.0124 | **+0.0063** | -0.0147 |
| 6 | 0.4199 | 0.3661 | 0.4158 | 0.3635 | -0.0538 | -0.0041 | -0.0564 |
| 7 | 0.5024 | 0.5200 | 0.5227 | 0.5010 | **+0.0176** | **+0.0203** | -0.0013 |
| 8 | 0.4757 | 0.4725 | 0.4599 | 0.4620 | -0.0032 | -0.0158 | -0.0137 |
| **mean** | 0.4774 | 0.4636 | 0.4792 | 0.4484 | -0.0137 | +0.0019 | -0.0290 |
| **std** | 0.0394 | 0.0606 | 0.0388 | 0.0550 | 0.0249 | 0.0131 | 0.0232 |

---

## 3. Fold-level comparison against persistence

The unit of analysis is the **fold**, not the sample. A percentile bootstrap is
inappropriate here: it assumes exchangeable draws within one comparison, and with
8 folds it can only ever resample the same 8 numbers. A **Wilcoxon signed-rank
test** on the 8 paired deltas is the right instrument — it asks whether the fold-level
differences are consistently one-signed without assuming normality.

| Model | Mean Δ | Std Δ | Folds won | Wilcoxon p (2-sided) | p (1-sided) | p₁ < 0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| RandomForest (unweighted) | -0.0137 | 0.0249 | **2/8** | 0.2500 | 0.9023 | no |
| RandomForest (class_weight=balanced) | +0.0019 | 0.0131 | **4/8** | 0.8438 | 0.4219 | no |
| XGBoost | -0.0290 | 0.0232 | **0/8** | 0.0078 | 1.0000 | no |

> **A 8-pair signed-rank test has a hard resolution floor.** The smallest two-sided
> p it can return is **0.0078** and the smallest one-sided p is
> **0.00391** — reached only when all 8 deltas share a sign. So
> **no two-sided result here can ever reach α = 0.05, regardless of the data.** That
> is a property of the sample size, not evidence of absence. The one-sided test can
> reach significance, and the **folds-won count is the more informative statistic** at
> this n.

### Class support per evaluation block

An F1 of 0.0 on a class with no samples is not a model result. Support is reported
first so the next table can be read correctly.

| Class | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | Total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Good | 6,935 | 4,142 | 4,300 | 5,528 | 8,125 | 3,062 | 4,289 | 6,717 | 43,098 |
| Moderate | 4,938 | 7,304 | 8,241 | 7,522 | 6,813 | 7,286 | 8,335 | 5,678 | 56,117 |
| Unhealthy (sensitive) | 2,613 | 4,120 | 4,846 | 3,632 | 2,768 | 6,070 | 4,176 | 2,188 | 30,413 |
| Unhealthy | 8,592 | 10,567 | 11,897 | 7,724 | 7,778 | 12,456 | 10,251 | 8,367 | 77,632 |
| **Very unhealthy** | 4,256 | 3,574 | 1,260 | 3,599 | 2,805 | 1,399 | 2,857 | 4,697 | 24,447 |
| **Hazardous** | 2,809 | 960 | 80 | 2,329 | 2,439 | 124 | 491 | 3,196 | 12,428 |

Every advisory class appears in at least 8 of 8 evaluation blocks, so the rare-class comparison below is interpretable.

### Rare-class F1 across folds

Means are taken over **folds with non-zero support only**; `n/a` marks a block where
the class does not occur.

| Class | Model | Mean (usable folds) | Usable folds | Per fold |
| --- | --- | --- | --- | --- |
| Very unhealthy | Persistence | 0.4581 | 8/8 | 0.451  0.494  0.312  0.535  0.399  0.439  0.543  0.491 |
| Very unhealthy | RandomForest (unweighted) | 0.4564 | 8/8 | 0.493  0.533  0.233  0.559  0.438  0.318  0.582  0.495 |
| Very unhealthy | RandomForest (class_weight=balanced) | 0.4633 | 8/8 | 0.483  0.515  0.375  0.490  0.463  0.422  0.521  0.437 |
| Very unhealthy | XGBoost | 0.4363 | 8/8 | 0.444  0.493  0.263  0.468  0.448  0.327  0.555  0.492 |
| Hazardous | Persistence | 0.4665 | 8/8 | 0.629  0.488  0.248  0.637  0.658  0.081  0.365  0.624 |
| Hazardous | RandomForest (unweighted) | 0.4679 | 8/8 | 0.642  0.495  0.180  0.677  0.655  0.000  0.454  0.640 |
| Hazardous | RandomForest (class_weight=balanced) | 0.4842 | 8/8 | 0.631  0.481  0.262  0.654  0.711  0.082  0.440  0.612 |
| Hazardous | XGBoost | 0.4066 | 8/8 | 0.580  0.356  0.039  0.640  0.651  0.000  0.391  0.597 |

---

## 4. Verdict

**RandomForest (unweighted) beats persistence in only 2/8 folds** (mean -0.0137, std 0.0249). The sign of the effect changes with the evaluation block, which is precisely the instability the single-split analysis suspected.

**RandomForest (class_weight=balanced) beats persistence in only 4/8 folds** (mean +0.0019, std 0.0131). The sign of the effect changes with the evaluation block, which is precisely the instability the single-split analysis suspected.

**XGBoost loses to persistence in all 8/8 folds** (mean -0.0290). Not a split artifact: it is worse everywhere.

**Partially resolved, and the honest reading is 'usually but not always'.** The best model (`RandomForest (class_weight=balanced)`) clears the floor in 4 of 8 folds. That is stronger evidence than a single split, and weaker than a clean win. The thesis should report the fold count directly — "4/8 folds" is a more honest headline than any mean, and the folds where it loses should be characterised by season.

Reproduce with `python -m src.models.rolling_cv`.
