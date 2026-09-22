# Baseline metrics -- horizon 1 h

Random Forest vs XGBoost, predicting the AQI risk category **1 hours ahead**.

> **Headline: the learned baselines barely beat doing nothing.**
>
> RandomForest reaches **0.7994** observed-only macro-F1. A zero-parameter
> persistence rule -- "next hour's category equals this hour's" -- reaches
> **0.7933**. The margin is **+0.0061**. On Hazardous specifically,
> persistence scores 0.8823 against RandomForest's 0.8832.
>
> This is a property of the task, not a bug in the models: **the AQI category is
> unchanged from t to t+1 in 80.50% of test samples**, so echoing
> the current class is right 80.5% of the time for free. RandomForest's accuracy is
> 81.08% -- +0.58
> points on top of that. PM2.5 at time t alone carries most of the signal. **Treat 0.7933, not 0.7994, as the bar the
> sequence model has to clear.** If Phase 3 reports 0.81 macro-F1 as an improvement over
> "the baseline", it is claiming credit for roughly nothing.
>
> A longer horizon would make this a real forecasting problem; see
> `reports/horizon_comparison.md` for how persistence degrades as the horizon grows.

---

Task: predict the PM2.5 AQI risk category **1 hours ahead** from the nine
wearable feature channels. Trained on
`data/processed/h1/tabular_train.csv`
(294,252 samples), evaluated on `tabular_test.csv` (62,832).

Inputs are `metadata.json -> feature_columns` only:
`PM2.5`, `PM10`, `CO`, `TEMP`, `DEWP`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`.
The `is_imputed*` columns are **evaluation slicers, not inputs** -- the loader raises if
one ever appears in the feature list.

Hyperparameters come from the `baseline:` block in `configs/default.yaml`
(seed 42). No tuning, no resampling, no class weighting: these are the floor the
GAN-augmented sequence model has to clear, so they are deliberately plain.

---

## 1. All test rows (62,832 samples)

| Model | n | Accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- | --- |
| RandomForest | 62,832 | 0.8108 | **0.8004** | 0.8100 |
| XGBoost | 62,832 | 0.8058 | **0.7965** | 0.8052 |
| Persistence | 62,832 | 0.8050 | **0.7975** | 0.8050 |

### Per-class F1

| Class | RandomForest | XGBoost | Persistence | Support |
| --- | --- | --- | --- | --- |
| Good | 0.8101 | 0.8008 | 0.8005 | 11,315 |
| Moderate | 0.7623 | 0.7537 | 0.7532 | 14,274 |
| Unhealthy (sensitive) | 0.6413 | 0.6321 | 0.6379 | 6,602 |
| Unhealthy | 0.8821 | 0.8801 | 0.8778 | 19,216 |
| **Very unhealthy** | 0.8304 | 0.8313 | 0.8297 | 7,629 |
| **Hazardous** | 0.8761 | 0.8812 | 0.8857 | 3,796 |

---

## 2. Observed labels only (61,585 samples, `is_imputed_pm25 == False`)

The 1,247 excluded samples
(2.0% of test) carry a label derived from a
forward-filled PM2.5 reading rather than a measured one. **This is the set that matters
for the thesis** -- it is the only one not contaminated by the forward-fill artifact.

| Model | n | Accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- | --- |
| RandomForest | 61,585 | 0.8089 | **0.7994** | 0.8080 |
| XGBoost | 61,585 | 0.8042 | **0.7947** | 0.8035 |
| Persistence | 61,585 | 0.8011 | **0.7933** | 0.8011 |

### Per-class F1

| Class | RandomForest | XGBoost | Persistence | Support |
| --- | --- | --- | --- | --- |
| Good | 0.8057 | 0.7985 | 0.7953 | 11,024 |
| Moderate | 0.7594 | 0.7530 | 0.7492 | 14,044 |
| Unhealthy (sensitive) | 0.6348 | 0.6268 | 0.6295 | 6,452 |
| Unhealthy | 0.8805 | 0.8786 | 0.8753 | 18,824 |
| **Very unhealthy** | 0.8324 | 0.8307 | 0.8280 | 7,554 |
| **Hazardous** | 0.8832 | 0.8806 | 0.8823 | 3,687 |

---

## 3. All rows vs observed-only

| Model | Macro-F1 all | Macro-F1 obs | Δ | V.unhealthy all | V.unhealthy obs | Δ | Hazardous all | Hazardous obs | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RandomForest | 0.8004 | 0.7994 | -0.0010 | 0.8304 | 0.8324 | +0.0020 | 0.8761 | 0.8832 | +0.0071 |
| XGBoost | 0.7965 | 0.7947 | -0.0018 | 0.8313 | 0.8307 | -0.0006 | 0.8812 | 0.8806 | -0.0005 |
| Persistence | 0.7975 | 0.7933 | -0.0042 | 0.8297 | 0.8280 | -0.0017 | 0.8857 | 0.8823 | -0.0034 |

### What the comparison says

Best model by observed-only macro-F1: **RandomForest**.

- Macro-F1 moves by -0.0010, which is not a meaningful shift -- this figure is not materially contaminated.
- Hazardous F1 moves by +0.0071, which is not a meaningful shift -- this figure is not materially contaminated.
- Very unhealthy F1 moves by +0.0020, which is not a meaningful shift -- this figure is not materially contaminated.

So the forward-fill artifact does **not** materially move this baseline's scores. That is a different question from the one `reports/preprocessing_summary.md` raises, and both answers stand: imputation inflates Hazardous *prevalence* (8.38% of imputed readings vs 4.47% of observed ones), but only 1,247 test samples (2.0%) carry an imputed label, and only 109 of the 3,796 Hazardous ones do. Too few rows to shift an F1. Keep reporting both numbers -- the check is cheap and the answer could change once CTGAN starts generating from this distribution.

Where RandomForest sends true Hazardous hours on observed-only rows
(3,687 samples):

| Predicted as | n | share |
| --- | --- | --- |
| Good | 1 | 0.0% |
| Moderate | 1 | 0.0% |
| Unhealthy | 14 | 0.4% |
| Very unhealthy | 419 | 11.4% |
| Hazardous | 3,252 | 88.2% |

---

## 4. Persistence reference

`Persistence` predicts that the category 1 hours from now equals the current
hour's. It has no parameters and needs no training. See
`reports/horizon_comparison.md` for how it degrades across h=1/6/12/24.

It scores **0.7933** observed-only macro-F1 against RandomForest's **0.7994** -- the learned model is +0.0061 ahead. **That is not a meaningful margin.** At one-hour horizon the class rarely changes, so a learned model that barely clears persistence has not demonstrated much. Treat persistence, not the RF/XGBoost numbers, as the bar the sequence model must beat.

---

## 5. Saved artifact

`src/models/artifacts/baseline_h1.pkl` holds a joblib bundle
(35.4 MB, `compress=3`): the fitted **RandomForest**, its
`feature_columns` and `class_labels`, and the selection metric. Load it with
`joblib.load(...)` and index `["model"]`.

The forest is depth- and leaf-capped in `configs/default.yaml` for a reason worth
recording: an unbounded forest on these 294k rows pickles to **2.4 GB** and scores
*lower* (0.7970 vs 0.7994 observed macro-F1) -- it memorises
rather than generalises. Neither variant is remotely deployable to an ESP32; that is
what the TFLite Micro path in `src/deployment/` is for, and a tree ensemble is not the
model that will make that trip.

Selection used **validation observed-only macro-F1**
(0.7850) -- observed-only rather than overall,
because the overall figure is inflated by the forward-fill artifact, and validation
rather than test, because test is reserved for final reporting. Test would have picked the same model, but that is an observation made after the fact, not part of the decision. The selection rule does not consult it.

Reproduce with `python -m src.models.baseline`.
