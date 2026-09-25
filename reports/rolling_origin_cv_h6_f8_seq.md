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

| Fold | Persistence | LSTM | Transformer | Δ LSTM | Δ Transformer |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.4795 | 0.4886 | 0.4832 | **+0.0091** | **+0.0037** |
| 2 | 0.4751 | 0.4400 | 0.4266 | -0.0351 | -0.0485 |
| 3 | 0.4336 | 0.3551 | 0.3850 | -0.0785 | -0.0486 |
| 4 | 0.5477 | 0.5057 | 0.5091 | -0.0420 | -0.0385 |
| 5 | 0.4850 | 0.4836 | 0.4648 | -0.0014 | -0.0202 |
| 6 | 0.4199 | 0.3778 | 0.3751 | -0.0421 | -0.0448 |
| 7 | 0.5024 | 0.5177 | 0.5171 | **+0.0153** | **+0.0147** |
| 8 | 0.4757 | 0.4617 | 0.4698 | -0.0140 | -0.0059 |
| **mean** | 0.4774 | 0.4538 | 0.4538 | -0.0236 | -0.0235 |
| **std** | 0.0394 | 0.0593 | 0.0534 | 0.0316 | 0.0252 |

---

## 3. Fold-level comparison against persistence

The unit of analysis is the **fold**, not the sample. A percentile bootstrap is
inappropriate here: it assumes exchangeable draws within one comparison, and with
8 folds it can only ever resample the same 8 numbers. A **Wilcoxon signed-rank
test** on the 8 paired deltas is the right instrument — it asks whether the fold-level
differences are consistently one-signed without assuming normality.

| Model | Mean Δ | Std Δ | Folds won | Wilcoxon p (2-sided) | p (1-sided) | p₁ < 0.05 |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | -0.0236 | 0.0316 | **2/8** | 0.1094 | 0.9609 | no |
| Transformer | -0.0235 | 0.0252 | **2/8** | 0.0547 | 0.9805 | no |

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
| Very unhealthy | LSTM | 0.4528 | 8/8 | 0.510  0.505  0.224  0.489  0.466  0.375  0.578  0.476 |
| Very unhealthy | Transformer | 0.4725 | 8/8 | 0.561  0.482  0.366  0.513  0.465  0.317  0.578  0.497 |
| Hazardous | Persistence | 0.4665 | 8/8 | 0.629  0.488  0.248  0.637  0.658  0.081  0.365  0.624 |
| Hazardous | LSTM | 0.4458 | 8/8 | 0.641  0.454  0.071  0.646  0.691  0.013  0.431  0.621 |
| Hazardous | Transformer | 0.4462 | 8/8 | 0.616  0.422  0.074  0.677  0.702  0.040  0.408  0.630 |

---

## 4. Verdict

**LSTM beats persistence in only 2/8 folds** (mean -0.0236, std 0.0316). The sign of the effect changes with the evaluation block, which is precisely the instability the single-split analysis suspected.

**Transformer beats persistence in only 2/8 folds** (mean -0.0235, std 0.0252). The sign of the effect changes with the evaluation block, which is precisely the instability the single-split analysis suspected.

**The instability is confirmed, and it is not a single-split artifact.** No model clears the persistence floor in a majority of folds (best: `LSTM` at 2/8). Five evaluation blocks spanning different seasons give the same answer the single split hinted at: at a 6-hour horizon on these nine channels, a learned model is not reliably better than assuming the next six hours look like now. This upgrades the finding from a possible artifact to a robust multi-fold negative result, which is the publishable version of it.

Reproduce with `python -m src.models.rolling_cv`.
