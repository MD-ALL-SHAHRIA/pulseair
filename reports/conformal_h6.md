# Conformal prediction — horizon 6 h

Split conformal prediction wrapped around **RandomForest**
(`baseline_h6.pkl`), the Phase 3 baseline, used exactly as saved — no retraining.

> **Scope note (added after Phase 10).** This report documents the conformal method as
> developed on **Beijing** data. It is **not** the deployed system. Rolling-origin CV
> later showed no Beijing-trained model beats persistence in more than 2 of 5 folds
> (`reports/rolling_origin_cv_h6.md`), so nothing here ships. The deployed
> candidate is the Bangladesh-native model in `reports/bangladesh_deployment.md`, which
> re-derives these thresholds on its own data. Read this as the calibration
> methodology, which transferred; not as a predictor, which did not.

Phase 5 found the LSTM and Transformer significantly worse than this forest, and worse
on Hazardous specifically (`reports/dl_metrics_h6.md`). So the uncertainty
machinery is attached to the model that won rather than keeping a sequence model around
for MC dropout; no deep model is carried forward.

---

## 1. Calibration

| | |
|---|---|
| Calibration set | `val` split, observed rows only |
| Calibration points | 61,466 |
| Nonconformity score | `1 − P(true class)` |
| Target coverage | **90%** (α = 0.10) |
| Quantile rank | 55,321 of 61,466 — `⌈(n+1)(1−α)⌉` |
| **Threshold q** | **0.8911** |
| Set rule | include every class with `P ≥ 0.1089` |

Calibration uses **observed rows only** (`is_imputed_pm25 == False`), consistent with
every metric in this project: a threshold tuned against forward-filled labels would
inherit the Hazardous inflation documented in Phase 2.

The `(n+1)` in the rank is what makes this a finite-sample guarantee rather than an
asymptotic one. Measured back on the calibration set, coverage is
0.9000 — at target by construction, which checks the
quantile was taken correctly and says nothing about generalisation.

---

## 2. Empirical coverage on test

61,530 observed test rows of 62,772.

| | Target | Empirical | Difference |
|---|---|---|---|
| **Marginal coverage** | 0.9000 | **0.8908** | -0.0092 |

### Per-class conditional coverage

| Class | n | Coverage | vs target | Status | Mean set size | Median |
| --- | --- | --- | --- | --- | --- | --- |
| Good | 11,006 | 0.8331 | -0.0669 | **UNDER** | 2.36 | 2 |
| Moderate | 14,020 | 0.8912 | -0.0088 | on target | 2.70 | 3 |
| Unhealthy (sensitive) | 6,439 | 0.7776 | -0.1224 | **UNDER** | 2.84 | 3 |
| Unhealthy | 18,824 | 0.9668 | +0.0668 | over | 2.37 | 2 |
| **Very unhealthy** | 7,554 | 0.9047 | +0.0047 | on target | 2.41 | 2 |
| **Hazardous** | 3,687 | 0.8432 | -0.0568 | **UNDER** | 2.22 | 2 |

> **The advisory classes are under-covered. State this in the thesis.**

**Hazardous** fall below the 90% target. The worst is
**Hazardous** at **0.8432** — 0.0568 short, across
3,687 test cases. In plain terms: when the true category is Hazardous, the prediction
set misses it about 16% of the time, not 10%.

**This is not a calibration bug — it is exactly what the guarantee does and does not
promise.** Split conformal delivers *marginal* coverage: averaged over all test points
the set contains the truth 89.1% of the time, and it does. Nothing in the
procedure equalises coverage across classes, and when a class is both rare and hard, the
slack lands there. For a device whose purpose is warning about hazardous air, the
marginal number flatters and the per-class number is the one that matters.

Two honest options, neither free:

- **Mondrian (class-conditional) conformal** — a separate threshold per class. Buys
  per-class coverage at the cost of larger sets and a thinner effective calibration set
  for the rare classes (3,687 points for the worst one here, workable but not
  generous).
- **Keep marginal coverage and state the limitation**, ensuring the advisory copy never
  implies a per-category guarantee that does not exist.

Until one is chosen: **the advisory layer must not claim "90% confident" for a
Hazardous prediction.** The 90% is a fleet-wide average, not a promise about that
reading.

---

## 3. Prediction set size

The actionable number. A set of one is a confident call; anything larger is genuine
ambiguity the advisory layer has to communicate rather than hide.

| | |
|---|---|
| Mean set size | **2.487** |
| Median set size | **2** |
| Singletons (size 1) | **7.2%** |
| Empty (size 0) | 0.00% |

| Set size | n | Share | Meaning |
| --- | --- | --- | --- |
| 1 | 4,410 | 7.17% | confident: a single category |
| 2 | 29,132 | 47.35% | two plausible categories |
| 3 | 21,625 | 35.15% | three categories |
| 4 | 6,334 | 10.29% | four |
| 5 | 29 | 0.05% | five |


At 90% coverage this model is confident — a single category — only
7.2% of the time. The median reading yields
2 categories. **That is the honest operating point**, and it
follows directly from the Phase 3–5 finding that this task is hard six hours out: a
model that cannot separate adjacent AQI bands cannot produce small sets, and forcing
smaller ones would only move the error from "visibly ambiguous" to "quietly wrong".

### By true class

| Class | n | Mean set size | Median |
| --- | --- | --- | --- |
| Good | 11,006 | 2.36 | 2 |
| Moderate | 14,020 | 2.70 | 3 |
| Unhealthy (sensitive) | 6,439 | 2.84 | 3 |
| Unhealthy | 18,824 | 2.37 | 2 |
| **Very unhealthy** | 7,554 | 2.41 | 2 |
| **Hazardous** | 3,687 | 2.22 | 2 |

---

## 4. Mondrian (class-conditional) conformal

The marginal threshold above is a single number pooled over every calibration point,
so a class that is both rare and hard absorbs the slack — which is what put Hazardous
at 0.8432. Mondrian calibrates a separate
quantile *within each class*, converting the marginal guarantee into a per-class one.

| Class | Calib n | q (per class) | Coverage: marginal | Coverage: Mondrian | Δ | Set size: marginal | Set size: Mondrian |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Good | 11,180 | 0.9289 | 0.8331 | 0.8673 | +0.0343 | 2.36 | 2.45 |
| Moderate | 14,070 | 0.8885 | 0.8912 | 0.8877 | -0.0034 | 2.70 | 2.76 |
| Unhealthy (sensitive) | 8,952 | 0.9256 | 0.7776 | 0.8635 | +0.0859 | 2.84 | 2.92 |
| Unhealthy | 20,419 | 0.7542 | 0.9668 | 0.8808 | -0.0860 | 2.37 | 2.69 |
| **Very unhealthy** | 4,282 | 0.9288 | 0.9047 | 0.9357 | +0.0310 | 2.41 | 2.37 |
| **Hazardous** | 2,563 | 0.9481 | 0.8432 | 0.9083 | +0.0651 | 2.22 | 2.07 |

| | Marginal | Mondrian |
|---|---|---|
| Overall coverage | 0.8908 | **0.8865** |
| Mean set size | 2.487 | **2.610** |
| Median set size | 2 | **3** |
| Singletons | 7.2% | **2.1%** |
| Empty sets | 0.00% | **0.00%** |

**Mondrian fixes the advisory classes.** Very unhealthy 0.9047 → **0.9357**, Hazardous 0.8432 → **0.9083**. The cost is set size: mean 2.487 → **2.610**, singletons 7.2% → **2.1%**. That is the trade in plain terms — a genuine per-class guarantee is paid for with wider, less decisive sets. Still below target after Mondrian: Good, Moderate, Unhealthy (sensitive), Unhealthy.

**Mondrian is what the advisory layer uses** (`default_method: "mondrian"` in
`conformal_h6.pkl`). The reason is specific: an advisory that says "this could
be Hazardous" needs the Hazardous guarantee to hold for Hazardous readings, not on
average across a fleet where the common classes carry the average. Larger sets are the
honest price, and the advisory copy names the ambiguity rather than hiding it.

---

## 5. What this replaces

| Before | After this phase |
|---|---|
| Point prediction, no uncertainty | Prediction set with 90% marginal coverage |
| MC dropout on a sequence model that lost to the forest | Conformal wrapper on the forest that won |
| Confidence as a softmax number with no operational meaning | Set size, directly interpretable |

`conformal_h6.pkl` holds the threshold, probability floor, feature order and
class labels — everything needed to build sets at inference. It is ~1 KB and pairs with
the existing `baseline_h6.pkl`.

**No LSTM or Transformer is carried forward as a deployed artifact.**
`dl_model_h6.pt` stays in `src/models/artifacts/` as the record of the
Phase 5 experiment and should not be loaded by the advisory layer.

Reproduce with `python -m src.models.conformal`.
