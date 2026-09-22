# CTGAN ablation — horizon 6 h

Three training sets, one shared test split, 1,000-resample paired bootstrap.

All scores are on **real, observed-label test rows only** (61,530 of
62,772): synthetic rows never enter validation or test, and imputed-label
rows are excluded because forward-fill inflates Hazardous prevalence. Model selection
within each variant is on **validation** observed-only macro-F1; test is reporting only.

## 1. Synthetic data validity (re-verified)

The first generation run produced physically impossible rows that SDV's quality score
did not catch. The generator was redesigned — integer hour/month as categoricals with
the cyclical pair computed after sampling, and an `sdv.cag.Inequality` constraint on
DEWP ≤ TEMP enforced during training. All four checks now pass:

| Check | Synthetic | Real | Required |
| --- | --- | --- | --- |
| `hour_sin² + hour_cos² = 1` | 100.00% | 100% | 100% |
| distinct `hour_sin` values | 21 | 21 | ~21–24 |
| `month_sin² + month_cos² = 1` | 100.00% | 100% | 100% |
| DEWP > TEMP | 0.00% | 0.00% | 0% |

Full detail and the methodological note are in
`reports/gan_quality_report_h6.md`. **No conclusion below comes from the
earlier, buggy run.**

---

## 2. Results

| Variant | Train rows | of which synthetic | Selected | Accuracy | Macro-F1 | V.unhealthy F1 | Hazardous F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Persistence** | — | — | — | 0.5356 | **0.5118** | 0.5104 | 0.5897 |
| **unaugmented** | 294,192 | — | RandomForest | 0.5793 | **0.5173** | 0.5265 | 0.6098 |
| **broad-4** | 377,150 | 82,958 | RandomForest | 0.5687 | **0.5212** | 0.4973 | 0.5908 |
| **targeted-2** | 352,391 | 58,199 | RandomForest | 0.5678 | **0.5043** | 0.4903 | 0.5843 |

### Per-class F1

| Class | Persistence | unaugmented | broad-4 | targeted-2 |
| --- | --- | --- | --- | --- |
| Good | 0.5888 | 0.5981 | 0.6095 | 0.5990 |
| Moderate | 0.4963 | 0.5440 | 0.5147 | 0.5428 |
| Unhealthy (sensitive) | 0.2571 | 0.1432 | 0.2376 | 0.1381 |
| Unhealthy | 0.6286 | 0.6820 | 0.6772 | 0.6715 |
| **Very unhealthy** | 0.5104 | 0.5265 | 0.4973 | 0.4903 |
| **Hazardous** | 0.5897 | 0.6098 | 0.5908 | 0.5843 |

---

## 3. Paired bootstrap vs the unaugmented baseline

1,000 resamples of the test rows, both variants scored on the *same* rows each
time. The interval is the 95% percentile CI of the difference; "significant" means it
excludes zero.

| Variant | Metric | Δ vs unaugmented | 95% CI | p (two-sided) | Significant |
| --- | --- | --- | --- | --- | --- |
| broad-4 | Macro-F1 | +0.0039 | [+0.0011, +0.0066] | 0.008 | **yes** |
| broad-4 | Very unhealthy F1 | -0.0292 | [-0.0360, -0.0227] | 0.000 | **yes** |
| broad-4 | Hazardous F1 | -0.0190 | [-0.0258, -0.0118] | 0.000 | **yes** |
| targeted-2 | Macro-F1 | -0.0129 | [-0.0154, -0.0105] | 0.000 | **yes** |
| targeted-2 | Very unhealthy F1 | -0.0362 | [-0.0434, -0.0285] | 0.000 | **yes** |
| targeted-2 | Hazardous F1 | -0.0255 | [-0.0325, -0.0181] | 0.000 | **yes** |

**On the two advisory classes, every significant move is a regression:**

- **broad-4 / Very unhealthy**: -0.0292 [-0.0360, -0.0227]
- **broad-4 / Hazardous**: -0.0190 [-0.0258, -0.0118]
- **targeted-2 / Very unhealthy**: -0.0362 [-0.0434, -0.0285]
- **targeted-2 / Hazardous**: -0.0255 [-0.0325, -0.0181]

This is the finding that decides the recommendation. Very unhealthy and Hazardous are the classes the augmentation was built to help and the only two a wearable advisory really turns on — and adding synthetic rows to them made both *worse*, in every variant, with intervals that exclude zero. More synthetic Hazardous rows produced a model that is worse at Hazardous.

### And against the persistence floor

The comparison that decides whether any of this is a forecasting model at all:

| Variant | Δ vs persistence | 95% CI | Significant |
| --- | --- | --- | --- |
| unaugmented | +0.0055 | [+0.0016, +0.0093] | **yes** |
| broad-4 | +0.0094 | [+0.0055, +0.0136] | **yes** |
| targeted-2 | -0.0075 | [-0.0115, -0.0033] | **yes** |

---

## 4. Which variant carries forward

**Carry `unaugmented` forward to Phase 5.**

Both augmented variants are disqualified on the classes that matter most:

- `broad-4` significantly **degrades** Very unhealthy -0.0292 [-0.0360, -0.0227]; Hazardous -0.0190 [-0.0258, -0.0118]
- `targeted-2` significantly **degrades** Very unhealthy -0.0362 [-0.0434, -0.0285]; Hazardous -0.0255 [-0.0325, -0.0181]

`broad-4`'s macro-F1 does rise +0.0039 [+0.0011, +0.0066], and that interval does exclude zero — **but look at where the gain comes from.** `Unhealthy (sensitive)` jumps from 0.1432 to 0.2376 (+0.0944), while Very unhealthy and Hazardous both fall. Macro-F1 weights all six classes equally, so one large gain on a mid-range class more than covers losses on the two the device exists to warn about. **For a wearable air-quality advisory that is a bad trade, not a result** — a missed Hazardous hour and a missed moderately-unhealthy hour are not equally costly, and macro-F1 does not know that.

Keep the magnitude in view too: +0.0039 macro-F1 is statistically detectable across 61,530 test rows and practically negligible. Significance at this sample size is cheap; a generator in the pipeline is not.

`broad-4` holds the highest macro-F1 point estimate. It is still not the right thing to carry forward.

All of unaugmented, broad-4 clear the persistence floor with a CI excluding zero, so the learned models are doing real work beyond echoing their input. That was not true at h=1.

Regenerate with `python -m src.gan.ablation`.
