# Phase 10 — external validation on Bangladesh

The device's target population. Source: Mendeley Data
[`9j447cynb9` v2](https://data.mendeley.com/datasets/9j447cynb9/2),
`AQI Bangladesh.csv` (98.3 MB, SHA-256 verified on download).

---

## 1. Data audit — read this first

The dataset is published as **103 cities, 2000-2025,
1,048,551 hourly records**. Inspection does not support that description,
and most of the file is not usable.

| Claim | As published | As measured |
| --- | --- | --- |
| Cities | 103 | **30** |
| Rows | 1,048,551 | 1,048,551 (matches) |
| Date range | 2000-2025 | 2000-01-01 .. 2025-11-23 |
| Temperature channel | — | **absent** |
| Dew point channel | — | **absent** |
| CO₂ coverage | available | **32.3%** of the usable window |

`cities.csv` lists 102 city records, but the measurement file contains only
**30** distinct cities.

### The pre-2022 portion does not survive inspection

Four independent signals put the boundary at **2022-08-05**:

| Signal | Before 2022-08-05 | After | Reading |
| --- | --- | --- | --- |
| Dhaka yearly median PM2.5 | linear in year, R² = **0.9920** (slope +5.63 µg/m³/yr) | no trend | a real city's yearly median does not lie on a straight line for 22 years |
| PM2.5 ceiling | **4.9%** of 2018+ hours sit at exactly 250.0 | not clipped | hard clip, not a physical limit |
| CO median | 1.56 | 424 | a unit change mid-file (≈250×), i.e. two sources spliced |
| PM2.5 lag-1 autocorrelation | 0.905 | 0.983 | the later block behaves like real hourly air quality |

Only **Dhaka** carries the pre-2022 history; every other city begins at
2022-08-05. The most likely explanation is that Dhaka was
back-filled synthetically to give the dataset a longer nominal span.

**Everything below uses the 2022-08-05 onward window only**:
849,831 rows (81% of the file),
30 cities, 2022-08-05 to 2025-11-23
(3.30 years).

> **For the thesis.** This audit is worth a paragraph in its own right. The dataset is
> the top Google result for Bangladeshi air-quality data and is published on a
> reputable repository with a DOI; 81% of it is not usable, and nothing in the record
> says so. Any external-validation claim built on the advertised 2000–2025 span would
> be built on generated numbers.

### A correction to the CO₂ premise

The project brief noted that this dataset carries **CO₂ directly**, unlike Beijing,
and that this would improve sensor alignment for the wearable. **It does not hold.**
CO₂ is present in only **32.3%** of the usable window — absent entirely before
2024 and complete only from 2025. It cannot serve as a modelling channel here.

The schema (`pm10`, `pm2_5`, `carbon_monoxide`, `carbon_dioxide`,
`nitrogen_dioxide`, `sulphur_dioxide`, `ozone`) matches the **Open-Meteo Air Quality
API** exactly, which suggests the usable portion is CAMS reanalysis rather than
ground-station measurement. That is adequate for a transfer study and should be
stated: it is modelled ambient air quality, not sensor data.

---

## 2. Preprocessing — same conventions as Phase 2

Cities: **Barisāl, Chittagong, Comilla, Dhaka** — four major metros, each with complete
hourly coverage over the usable window.

Everything carried over unchanged: per-city forward fill with an `is_imputed`
provenance flag, identical cyclical hour/month encodings, the same EPA PM2.5
breakpoints, a chronological 70%/15%/15%
split on the shared hourly axis, scaler fitted on training rows only, window
24 h / horizon 6 h, and metrics on observed labels only.

| | |
|---|---|
| Rows after filtering | 115,872 |
| Rows dropped by forward fill | 0 |
| Samples (train / val / test) | 81,020 / 17,380 / 17,356 |
| Test samples with observed labels | 17,356 |

**Feature set: 7 channels, not nine.** Temperature and dew point do
not exist in this dataset, so the Beijing feature set cannot be reproduced. The shared
subset is `PM2.5`, `PM10`, `CO`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`.

---

## 3. The persistence floor on Bangladesh

Recomputed, not assumed.

| | Beijing h6 | **Bangladesh h6** |
|---|---|---|
| Label unchanged over the horizon | 53.87% | **63.72%** |
| Persistence macro-F1 | 0.5118 | **0.3807** |
| Persistence Hazardous F1 | 0.5897 | **0.0000** |

---

## 4. Results

| Model | Macro-F1 | Accuracy | Hazardous F1 | V.unhealthy F1 | vs persistence (1,000-resample bootstrap) |
| --- | --- | --- | --- | --- | --- |
| Persistence | 0.3807 | 0.6372 | 0.0000 | 0.0000 | — (reference) |
| Beijing model (transfer) | 0.3258 | 0.5922 | 0.0000 | 0.0000 | -0.0550 [-0.0606, -0.0492] **significant** |
| Bangladesh-native RandomForest (class_weight=balanced) | 0.4075 | 0.6703 | 0.0000 | 0.0000 | +0.0268 [+0.0208, +0.0326] **significant** |

### Per-class F1

| Class | Persistence | Beijing model (transfer) | Bangladesh-native RandomForest (class_weight=balanced) |
| --- | --- | --- | --- |
| Good | 0.7112 | 0.6478 | 0.7641 |
| Moderate | 0.6566 | 0.6266 | 0.6467 |
| Unhealthy (sensitive) | 0.3541 | 0.1217 | 0.4098 |
| Unhealthy | 0.5625 | 0.5585 | 0.6246 |
| **Very unhealthy** | 0.0000 | 0.0000 | 0.0000 |
| **Hazardous** | 0.0000 | 0.0000 | 0.0000 |

> ### The advisory classes are effectively absent from this test split
>
> | Class | Train | Val | Test |
> | --- | --- | --- | --- |
> | Good | 16,046 | 755 | **6,176** |
> | Moderate | 29,388 | 5,063 | **7,927** |
> | Unhealthy (sensitive) | 16,337 | 4,201 | **1,757** |
> | Unhealthy | 18,836 | 6,924 | **1,490** |
> | **Very unhealthy** | 387 | 435 | **6** |
> | **Hazardous** | 26 | 2 | **0** |
>
> **Very unhealthy, Hazardous have 6 and 0 test
> samples respectively.** Their F1 is therefore 0 or meaningless for every model, and
> the macro-F1 comparison above is effectively a four-class comparison.
>
> The cause is the chronological split: the last 15% of the usable
> window runs from 2025-05-26 to 2025-11-23, which is Bangladesh's monsoon and post-monsoon
> period. Dhaka's severe pollution is a November–February phenomenon, and this split
> puts almost none of it in test.
>
> **This is the Phase 9 seasonal-split problem arriving in the external validation**,
> and it means the headline comparison here cannot speak to the classes the device
> exists to warn about. Rolling-origin CV over this dataset — the machinery is already
> in `src/models/rolling_cv.py` — would rotate a winter into the evaluation block and
> is required before any advisory-class claim is made on Bangladesh data.

### Native model selection

Both candidates were fitted on Bangladesh training data and chosen on **validation**,
as in every other phase:

| Candidate | Validation macro-F1 |
| --- | --- |
| RandomForest | 0.4192 |
| RandomForest (class_weight=balanced) | 0.4983 |

Selected: **RandomForest (class_weight=balanced)**.

---

## 5. Conformal coverage on Bangladesh

Mondrian thresholds re-calibrated on the **Bangladesh validation split** — thresholds
are quantiles under a specific model and distribution, so the Beijing ones certify
nothing here.

| Model | Coverage | Hazardous | V.unhealthy | Mean set size | Singletons |
| --- | --- | --- | --- | --- | --- |
| Beijing model (transfer) | 0.9474 | n/a | 0.0000 | 3.34 | 0.0% |
| Bangladesh-native RandomForest (class_weight=balanced) | 0.9169 | n/a | 0.0000 | 2.77 | 0.0% |

---

## 6. Rolling-origin cross-validation

5 expanding-window folds over the usable window, 30% initial
training. All 5 evaluation blocks touch Bangladesh's Nov–Feb high-pollution season;
the fold count was chosen for that reason and the coverage is verified in
`reports/bangladesh_rolling_cv.md`.

| Model | Macro-F1 (mean ± std) | Mean Δ vs persistence | Folds won | Wilcoxon p (1-sided) |
| --- | --- | --- | --- | --- |
| Persistence | 0.4377 ± 0.0391 | — | — | — |
| RandomForest (unweighted) | 0.4557 ± 0.0162 | +0.0181 | **3/5** | 0.1562 |
| RandomForest (class_weight=balanced) | 0.4731 ± 0.0433 | +0.0354 | **5/5** | 0.0312 |

**`RandomForest (class_weight=balanced)` beats persistence in all 5/5 folds** (mean +0.0354, one-sided Wilcoxon p = 0.0312). Every evaluation block, every season in the usable window. **This is the strongest deployment claim in the project** — it is the only model, on either dataset, that clears its persistence floor in every rolling-origin fold.

> **The advisory classes still cannot be evaluated, and more folds will not fix it.**
> Across all 5 evaluation blocks combined there are **2 Hazardous** and
> **482 Very unhealthy** samples. Hazardous is absent from
> 4 of 5 blocks.
>
> The cause is the data, not the split — and **Phase 11 has now confirmed it against a
> reference-grade instrument** (`reports/dhaka_ground_truth_validation.md`). Over
> 23,010 jointly observed hours the US Embassy Dhaka monitor records **1,602 Hazardous
> hours where this dataset records 17**, and above 150 µg/m³ the reanalysis runs
> ~138 µg/m³ low. The reanalysis understates South Asian peak PM2.5, and the error is
> concentrated exactly in the advisory range.
>
> **Advisory-class validation for Bangladesh requires ground-station measurements** —
> the US Embassy Dhaka reference monitor or Department of Environment CAMS stations,
> not this product. **Phase 11b does exactly that**
> (`reports/dhaka_ground_truth_model_h6.md`): on the reference series a PM2.5-only
> model beats persistence on Hazardous F1 in 7/7 rolling-origin folds (0.4451 vs
> 0.3167, p = 0.0156). That validates the *task*; it does not validate the 7-channel
> model evaluated in this report, which cannot run on a single-pollutant source. Until then, no Hazardous or Very-unhealthy claim should be made on
> Bangladeshi data, and the 6-hour macro-F1 result above should be
> cited as covering the four common classes.

---

## 7. Recommendation

**The native model clears the Bangladesh persistence floor; the transferred Beijing model does not.** That is the expected and the desirable result: it says the methodology transfers and the *weights* do not, which is exactly why a target-population model is the deployment candidate.

**Deploy the Bangladesh-native model.** `src/models/artifacts/bangladesh_rf_h6.pkl`
holds the selected forest, its feature order, class labels and re-calibrated Mondrian
thresholds. Dhaka's pollution profile, sources and seasonal cycle differ substantially
from Beijing's, and the transfer numbers above show the difference is not
cosmetic.

**Cite the Beijing pipeline as the methodology, not as the model.** What transfers is
the discipline, and it is what makes the Bangladesh number trustworthy rather than
merely reported:

- the **persistence floor** reported alongside every result (Phase 3);
- the **horizon selection** that rejected h=1 as autocorrelation rather than forecasting
  (Phase 3);
- the **GAN ablation** with a disqualification rule on the advisory classes (Phase 4);
- **rolling-origin cross-validation**, which showed single-split wins on Beijing sit
  inside the fold-to-fold variance of the baseline (Phase 9);
- **Mondrian conformal calibration**, after marginal coverage was found to under-cover
  Hazardous (Phase 6);
- **imputation provenance**, so every metric here is computed on observed labels only
  (Phase 2).

**Carry the Phase 9 caveat across.** The Bangladesh result above is a single
chronological split. The Beijing work showed a single split can produce a win whose
sign flips under rolling-origin evaluation, and nothing about Bangladesh makes that
less likely — its usable window is 3.3 years
against Beijing's 4. **Rolling-origin CV on this dataset is the next step before any
deployment claim**, and the machinery already exists in `src/models/rolling_cv.py`.

Reproduce with `python -m src.preprocessing.bangladesh` then
`python -m src.models.bangladesh_validation`.
