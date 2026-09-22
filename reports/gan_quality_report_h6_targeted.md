# CTGAN quality report — horizon 6 h

One `CTGANSynthesizer` per minority class, 300 epochs, batch size
500, seed 42. Total fit time 8 min.

Fit on `data/processed/h6/tabular_train.csv` restricted to
**`is_imputed_pm25 == False`** — 288,032 of 294,192
training rows (97.9%). Forward-fill
roughly doubles Hazardous prevalence, so fitting on unfiltered rows would have taught
the generator an imputation artifact and then amplified it.

Modelled columns (7): `PM2.5`, `PM10`, `CO`, `TEMP`, `DEWP`, `hour`, `month` — the five
sensor channels **in raw units** plus the **integer** hour and month as categoricals.
The cyclical `sin`/`cos` pairs are *not* generated; they are computed from the sampled
integer afterwards by the pipeline's own `add_cyclical`, so a synthetic row's encoding
is identical to a real one's. `DEWP ≤ TEMP` is enforced by an `sdv.cag.Inequality`
constraint during fitting and sampling. Raw units matter for that: after StandardScaler
the two channels have different means and scales, so the inequality is not a column
comparison at all. See section 5 for why the first design failed.

The class identity is carried by *which* synthesizer produced a row, so the label never
becomes a modelled column and cannot leak.

---

## 1. The h6 observed-only training distribution

These counts are recomputed here, not carried over. Both the horizon change and the
observed-only filter move them, so the h1 and all-rows percentages reported in earlier
phases do not apply.

Majority class: **Unhealthy** at 102,722 observed rows.
A class counts as minority below **50%** of that, and is topped
up to **50%** of it (51,361 rows).

| Class | All train rows | Observed rows | Observed % | vs majority | Synthetic needed |
| --- | --- | --- | --- | --- | --- |
| Good | 41,104 | 40,266 | 13.98% | 0.392 | — |
| Moderate | 63,989 | 62,824 | 21.81% | 0.612 | — |
| Unhealthy (sensitive) | 38,480 | 37,697 | 13.09% | 0.367 | — |
| Unhealthy | 104,898 | 102,722 | 35.66% | 1.000 | — |
| Very unhealthy **(minority)** | 33,032 | 32,338 | 11.23% | 0.315 | 19,023 |
| Hazardous **(minority)** | 12,689 | 12,185 | 4.23% | 0.119 | 39,176 |

2 of 6 classes fall below the threshold: Very unhealthy, Hazardous.

---

## 2. Synthetic data quality

SDV's quality report compares each class's synthetic rows against the real rows that
class was fit on. 1.00 is a perfect match; the floor configured for this project is
**0.90**.

| Class | Real rows | Synthetic | Overall | Column shapes | Pair trends | Fit (min) |
| --- | --- | --- | --- | --- | --- | --- |
| Very unhealthy | 32,338 | 19,023 | 0.8734 | 0.8966 | 0.8503 | 5.9 |
| Hazardous | 12,185 | 39,176 | 0.7930 | 0.9128 | 0.6733 | 2.2 |

### Per-column shape scores

| Column | Very unhealthy | Hazardous |
| --- | --- | --- |
| `PM2.5` | 0.9495 | 0.9422 |
| `PM10` | 0.9345 | 0.9154 |
| `CO` | 0.8625 | 0.9258 |
| `TEMP` | 0.9155 | 0.9360 |
| `DEWP` | 0.9132 | 0.9202 |
| `hour` | 0.8319 | 0.8594 |
| `month` | 0.8689 | 0.8905 |

### Flagged

**5 column(s) and 8 pair(s) scored below the 0.90 floor set in `configs/default.yaml`.** They are listed here rather than passed over:

| Class | Column | Score | Shortfall |
| --- | --- | --- | --- |
| Very unhealthy | `hour` | 0.8319 | 0.0681 below floor |
| Very unhealthy | `CO` | 0.8625 | 0.0375 below floor |
| Very unhealthy | `month` | 0.8689 | 0.0311 below floor |
| Hazardous | `hour` | 0.8594 | 0.0406 below floor |
| Hazardous | `month` | 0.8905 | 0.0095 below floor |

Worst correlation pairs:

| Class | Pair | Score |
| --- | --- | --- |
| Very unhealthy | `DEWP` x `month` | 0.5322 |
| Very unhealthy | `TEMP` x `month` | 0.6569 |
| Very unhealthy | `PM2.5` x `PM10` | 0.8921 |
| Hazardous | `DEWP` x `month` | 0.4985 |
| Hazardous | `TEMP` x `month` | 0.5164 |
| Hazardous | `PM2.5` x `PM10` | 0.5676 |
| Hazardous | `PM10` x `CO` | 0.6892 |
| Hazardous | `PM2.5` x `CO` | 0.7038 |

A low **column shape** score means the synthetic marginal distribution for that channel does not match the real one — the generator has the wrong histogram. A low **pair trend** score means the marginals may be fine but the joint structure is not: the synthetic rows break a correlation the real sensor data has. For a wearable that is the more damaging failure, because PM2.5 and PM10 moving together is most of what distinguishes a real reading from noise.

---

## 2b. Constraint validity — what the quality score does not measure

| Check | Real | Synthetic | Expected |
| --- | --- | --- | --- |
| `hour_sin` / `hour_cos` on the unit circle | 100.0% | 100.0% | must be 100% |
| distinct `hour_sin` values | 21 | 21 | ~24 possible |
| `month_sin` / `month_cos` on the unit circle | 100.0% | 100.0% | must be 100% |
| distinct `month_sin` values | 11 | 11 | ~12 possible |
| DEWP > TEMP by more than 1e-06 °C | 0.0014% (4 rows) | 0.0000% (0 rows) | 0% |
| largest DEWP − TEMP excess | 0.5 °C | 5.33e-15 °C | ≤ float noise |
| negative `PM2.5` after unscaling | 0.00% | 0.00% | must be 0% |
| negative `PM10` after unscaling | 0.00% | 0.00% | must be 0% |
| negative `CO` after unscaling | 0.00% | 0.00% | must be 0% |

All constraint checks are within the range set by the real data.

---

## 3. Resulting training distribution

| Class | Real | Synthetic | Total | vs majority |
| --- | --- | --- | --- | --- |
| Good | 41,104 | 0 | 41,104 | 0.392 |
| Moderate | 63,989 | 0 | 63,989 | 0.610 |
| Unhealthy (sensitive) | 38,480 | 0 | 38,480 | 0.367 |
| Unhealthy | 104,898 | 0 | 104,898 | 1.000 |
| Very unhealthy | 33,032 | 19,023 | 52,055 | 0.496 |
| Hazardous | 12,689 | 39,176 | 51,865 | 0.494 |

Written to `data/processed/h6/tabular_train_augmented_targeted.csv`
(352,391 rows), with a **`source`** column holding `real` or
`synthetic`.

**Later phases must filter on `source`, not on `is_imputed_pm25`.** Synthetic rows carry
`is_imputed_pm25 == False` because they are not imputed, so an observed-only filter
inherited from Phase 2 will happily include them. Any reporting-only metric needs
`source == "real"` as well.

Note the real rows include the 6,160
imputed-label rows: CTGAN did not learn from them, but dropping them from the training
table would discard real data for a reason that only applies to the generator.

---

## 4. Read this before using the output

- The rebalance is **partial by design** (50% of majority, not parity).
  A neckband does not meet hazardous air half the time, and a classifier trained to
  expect that will over-warn in the field. The base rate matters to the advisory layer.
- Quality scores measure whether synthetic rows *look like* real ones. They do not
  measure whether adding them helps a downstream model, and as section 2b shows they do
  not measure whether a row is physically possible either. The downstream question is
  answered in `reports/baseline_metrics_h6_augmented.md`.
- The weakest single column across all classes is `hour` for *Very unhealthy* at 0.8319, which is **below** the 0.90 floor.

---

## 5. Methodological note — synthesizer quality metrics are not validity checks

**Keep this in the thesis as a caveat.** It is a finding about evaluating generative
models, not an incident report.

The first version of this step handed CTGAN the model's own nine feature columns
directly, including the four cyclical encodings, in standardized units. SDV's quality
report scored that output **0.89–0.93 overall**, with Column Shapes 0.88–0.91 and Column
Pair Trends 0.88–0.95. On those numbers the synthetic data looked usable.

It was not. Two defects, both invisible to the score:

1. **Invalid cyclical encodings.** `hour_sin` and `hour_cos` are a deterministic
   function of one integer with 24 possible values, and every real row satisfies
   `hour_sin² + hour_cos² = 1`. CTGAN modelled them as two unrelated continuous
   columns. Only **19.5%** of synthetic rows landed on the unit circle, and the
   generator emitted **82,885 distinct `hour_sin` values** where 24 hours exist. Four
   fifths of the synthetic rows encoded a time of day that cannot occur. Each column's
   *marginal* was reproduced acceptably, which is all Column Shapes measures.

2. **A violated physical constraint.** Dew point cannot exceed air temperature.
   **13.1%** of synthetic rows had `DEWP > TEMP`, against 0.18% in the real data. TEMP
   and DEWP each scored above the quality floor individually, and their linear
   correlation was reproduced well enough to pass Column Pair Trends — but a
   correlation coefficient does not encode an inequality.

The general lesson: **Column Shapes compares one marginal at a time, Column Pair Trends
compares linear association, and neither asks whether a row is possible.** A synthetic
record can match every marginal and every pairwise correlation in the training data and
still be a reading no instrument could produce. A high SDV score is evidence that the
synthetic data resembles the real data distributionally. It is not evidence that the
data is valid, and for engineered features or physically constrained channels the two
come apart completely.

What this implies for anyone using a tabular synthesizer:

- **Do not hand a synthesizer engineered features.** Model the underlying variable and
  recompute the derived ones. Here: model the integer hour, then compute sin/cos.
- **Encode hard constraints as constraints, not as hope.** SDV's `Inequality` enforces
  `DEWP ≤ TEMP` during fitting and sampling. Checking afterwards only tells you how
  much of the output to throw away.
- **Write domain validity checks and run them every time.** Section 2b of this report
  is four assertions. They caught what a 0.93 quality score missed, and they cost
  nothing to run.
- One SDV-specific trap worth recording: passing the constraint as a plain dict to
  `add_constraints` is accepted **silently and has no effect** — measured at 51%
  violations with the dict versus 0% with the `sdv.cag.Inequality` object. An API that
  accepts a no-op without complaint is exactly the kind of thing a validity check
  catches and a quality score does not.

A further wrinkle worth recording: 196 real training rows
(0.44%) have `DEWP` marginally above `TEMP` — instrument tolerance, since
the two are measured to 0.1 °C by separate sensors. SDV correctly refuses to fit an
`Inequality` on data that already violates it, so those rows are clipped to
`DEWP == TEMP` (saturation) before fitting. The refusal is the right behaviour: a
constraint the training data contradicts is a modelling error, not a detail.

Regenerate with `python -m src.gan.augment --variant targeted`.
