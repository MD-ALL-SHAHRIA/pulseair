# Baseline metrics -- horizon 6 h

Random Forest vs XGBoost, predicting the AQI risk category **6 hours ahead**.

> **Headline: the learned baselines barely beat doing nothing.**
>
> RandomForest reaches **0.5173** observed-only macro-F1. A zero-parameter
> persistence rule -- "next hour's category equals this hour's" -- reaches
> **0.5118**. The margin is **+0.0055**. On Hazardous specifically,
> persistence scores 0.5897 against RandomForest's 0.6098.
>
> This is a property of the task, not a bug in the models: **the AQI category is
> unchanged from t to t+6 in 53.87% of test samples**, so echoing
> the current class is right 53.9% of the time for free. RandomForest's accuracy is
> 57.99% -- +4.12
> points on top of that. PM2.5 at time t alone carries most of the signal. **Treat 0.5118, not 0.5173, as the bar the
> sequence model has to clear.** If Phase 3 reports 0.81 macro-F1 as an improvement over
> "the baseline", it is claiming credit for roughly nothing.
>
> A longer horizon would make this a real forecasting problem; see
> `reports/horizon_comparison.md` for how persistence degrades as the horizon grows.

---

Task: predict the PM2.5 AQI risk category **6 hours ahead** from the nine
wearable feature channels. Trained on
`data/processed/h6/tabular_train.csv`
(294,192 samples), evaluated on `tabular_test.csv` (62,772).

Inputs are `metadata.json -> feature_columns` only:
`PM2.5`, `PM10`, `CO`, `TEMP`, `DEWP`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`.
The `is_imputed*` columns are **evaluation slicers, not inputs** -- the loader raises if
one ever appears in the feature list.

Hyperparameters come from the `baseline:` block in `configs/default.yaml`
(seed 42). No tuning, no resampling, no class weighting: these are the floor the
GAN-augmented sequence model has to clear, so they are deliberately plain.

---

## 1. All test rows (62,772 samples)

| Model | n | Accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- | --- |
| RandomForest | 62,772 | 0.5799 | **0.5176** | 0.5565 |
| XGBoost | 62,772 | 0.5599 | **0.4995** | 0.5396 |
| Persistence | 62,772 | 0.5387 | **0.5153** | 0.5387 |

### Per-class F1

| Class | RandomForest | XGBoost | Persistence | Support |
| --- | --- | --- | --- | --- |
| Good | 0.6000 | 0.5727 | 0.5937 | 11,297 |
| Moderate | 0.5438 | 0.5267 | 0.4980 | 14,250 |
| Unhealthy (sensitive) | 0.1478 | 0.1585 | 0.2625 | 6,584 |
| Unhealthy | 0.6835 | 0.6676 | 0.6313 | 19,216 |
| **Very unhealthy** | 0.5237 | 0.5102 | 0.5095 | 7,629 |
| **Hazardous** | 0.6067 | 0.5612 | 0.5967 | 3,796 |

---

## 2. Observed labels only (61,530 samples, `is_imputed_pm25 == False`)

The 1,242 excluded samples
(2.0% of test) carry a label derived from a
forward-filled PM2.5 reading rather than a measured one. **This is the set that matters
for the thesis** -- it is the only one not contaminated by the forward-fill artifact.

| Model | n | Accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- | --- |
| RandomForest | 61,530 | 0.5793 | **0.5173** | 0.5558 |
| XGBoost | 61,530 | 0.5599 | **0.4991** | 0.5394 |
| Persistence | 61,530 | 0.5356 | **0.5118** | 0.5356 |

### Per-class F1

| Class | RandomForest | XGBoost | Persistence | Support |
| --- | --- | --- | --- | --- |
| Good | 0.5981 | 0.5730 | 0.5888 | 11,006 |
| Moderate | 0.5440 | 0.5286 | 0.4963 | 14,020 |
| Unhealthy (sensitive) | 0.1432 | 0.1536 | 0.2571 | 6,439 |
| Unhealthy | 0.6820 | 0.6665 | 0.6286 | 18,824 |
| **Very unhealthy** | 0.5265 | 0.5122 | 0.5104 | 7,554 |
| **Hazardous** | 0.6098 | 0.5607 | 0.5897 | 3,687 |

---

## 3. All rows vs observed-only

| Model | Macro-F1 all | Macro-F1 obs | Δ | V.unhealthy all | V.unhealthy obs | Δ | Hazardous all | Hazardous obs | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForest | 0.5176 | 0.5173 | -0.0003 | 0.5237 | 0.5265 | +0.0028 | 0.6067 | 0.6098 | +0.0031 |
| XGBoost | 0.4995 | 0.4991 | -0.0004 | 0.5102 | 0.5122 | +0.0019 | 0.5612 | 0.5607 | -0.0005 |
| Persistence | 0.5153 | 0.5118 | -0.0035 | 0.5095 | 0.5104 | +0.0009 | 0.5967 | 0.5897 | -0.0070 |

### What the comparison says

Best model by observed-only macro-F1: **RandomForest**.

- Macro-F1 moves by -0.0003, which is not a meaningful shift -- this figure is not materially contaminated.
- Hazardous F1 moves by +0.0031, which is not a meaningful shift -- this figure is not materially contaminated.
- Very unhealthy F1 moves by +0.0028, which is not a meaningful shift -- this figure is not materially contaminated.

So the forward-fill artifact does **not** materially move this baseline's scores. That is a different question from the one `reports/preprocessing_summary.md` raises, and both answers stand: imputation inflates Hazardous *prevalence* (8.38% of imputed readings vs 4.47% of observed ones), but only 1,242 test samples (2.0%) carry an imputed label, and only 109 of the 3,796 Hazardous ones do. Too few rows to shift an F1. Keep reporting both numbers -- the check is cheap and the answer could change once CTGAN starts generating from this distribution.

Where RandomForest sends true Hazardous hours on observed-only rows
(3,687 samples):

| Predicted as | n | share |
| --- | --- | --- |
| Good | 14 | 0.4% |
| Moderate | 20 | 0.5% |
| Unhealthy | 310 | 8.4% |
| Very unhealthy | 971 | 26.3% |
| Hazardous | 2,372 | 64.3% |

---

## 4. Persistence reference

`Persistence` predicts that the category 6 hours from now equals the current
hour's. It has no parameters and needs no training. See
`reports/horizon_comparison.md` for how it degrades across h=1/6/12/24.

It scores **0.5118** observed-only macro-F1 against RandomForest's **0.5173** -- the learned model is +0.0055 ahead. **That is not a meaningful margin.** At one-hour horizon the class rarely changes, so a learned model that barely clears persistence has not demonstrated much. Treat persistence, not the RF/XGBoost numbers, as the bar the sequence model must beat.

---

## 5. Saved artifact

`src/models/artifacts/baseline_h6.pkl` holds a joblib bundle
(44.4 MB, `compress=3`): the fitted **RandomForest**, its
`feature_columns` and `class_labels`, and the selection metric. Load it with
`joblib.load(...)` and index `["model"]`.

The forest is depth- and leaf-capped in `configs/default.yaml` for a reason worth
recording: an unbounded forest on these 294k rows pickles to **2.4 GB** and scores
*lower* (0.7970 vs 0.5173 observed macro-F1) -- it memorises
rather than generalises. Neither variant is remotely deployable to an ESP32; that is
what the TFLite Micro path in `src/deployment/` is for, and a tree ensemble is not the
model that will make that trip.

Selection used **validation observed-only macro-F1**
(0.4895) -- observed-only rather than overall,
because the overall figure is inflated by the forward-fill artifact, and validation
rather than test, because test is reserved for final reporting. Test would have picked the same model, but that is an observation made after the fact, not part of the decision. The selection rule does not consult it.

Reproduce with `python -m src.models.baseline`.
