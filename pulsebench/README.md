# pulsebench

An evaluation protocol for **seasonal, imbalanced, categorical time-series
classification** — the shape of problem where the usual reporting conventions quietly
mislead.

Extracted from a thesis on wearable air-quality forecasting. Nothing in the package is
air-quality specific: it takes any dataframe with a datetime axis and a categorical
target.

## Why it exists

The thesis set out to build a better forecaster and ended up with a methodological
result instead. Four things went wrong in ways that were only visible because the
evaluation was built to catch them:

1. **A model beat the baseline by +0.006 macro-F1 and the baseline was a one-line
   rule.** At a 1-hour horizon the target was 80.5% autocorrelated; "the next hour
   looks like this hour" scored 0.7933 against the model's 0.7994. Most of the
   reported accuracy was the autocorrelation of the target, not skill.
2. **A single chronological split gave the wrong answer.** The same model won on the
   test block and lost on the validation block, because which winter lands where
   changes the result more than the model does. Across five rotating folds, the
   baseline's own fold-to-fold spread (0.0802) was an order of magnitude larger than
   any model-versus-model difference measured anywhere in the project.
3. **Two augmentation methods improved the aggregate metric by damaging the classes
   that mattered.** CTGAN and SMOTE both produced significant macro-F1 gains while
   significantly degrading both safety-critical classes. Macro-F1 weights six classes
   equally; a hazardous-air warning and a moderate-air warning are not equally costly.
4. **A five-fold signed-rank test cannot reach α = 0.05.** Its smallest possible
   two-sided p is 0.0625. Several "no significant difference" conclusions in that
   range were untestable by construction, not negative.

Each of the four functions here exists to make one of those failures hard to commit.

## Install

Nothing to install — the package sits in this repository and depends only on
`numpy`, `pandas`, `scipy` and `scikit-learn`.

```python
from pulsebench import (persistence_floor, rolling_origin_cv,
                        advisory_disqualification, bonferroni_report)
```

## The four pieces

### Seasonal baseline: `seasonal_naive_floor`

```python
from pulsebench import seasonal_naive_floor
seasonal = seasonal_naive_floor(df, "risk", horizon=6, season_length=24,
                               group_col="station")
```

Predict the target from the latest matching seasonal timestamp available at the
forecast origin. For horizons longer than a season, repeat the last available
season (no future observations). Origin, target and seasonal source must all exist;
time joins preserve gaps and group boundaries. The returned metrics have the same
shape as `persistence_floor`, but pair counts can differ because seasonal history
may be missing. `label_unchanged_pct` compares the seasonal source and target here.
When `season_length=horizon`, the result agrees with persistence. Step counts must
be positive integers; `freq` defaults to hours.

### `persistence_floor(df, target_col, horizon)`

The zero-parameter rule: the class at *t+h* equals the class at *t*. Report it beside
every headline number, or the number does not say whether the model is working.

```python
floor = persistence_floor(df, target_col="risk", horizon=6, group_col="station")
print(floor["macro_f1"], floor["label_unchanged_pct"])
```

Pairs are formed by a time join, so a gap in the series never produces a pair that
silently spans it. `group_col` keeps pairs inside a station, city or sensor.

### `rolling_origin_cv(df, model_fn, n_folds, embargo_hours, min_class_support=None)`

Expanding-window cross-validation that rotates the evaluation block through the
series, so each fold lands on a different part of the year.

```python
out = rolling_origin_cv(
    df, model_fn=lambda: RandomForestClassifier(class_weight="balanced"),
    n_folds=7, embargo_hours=48, min_class_support=50,
    feature_cols=FEATURES, target_col="y", baseline_col="persistence",
    scale_cols=SENSOR_CHANNELS, observed_col="is_observed")

print(out["aggregate"]["macro_f1"]["wins"], "/", out["n_folds"])
```

Three things it handles that are easy to get wrong:

- **Embargo.** Training uses rows whose *target* precedes the cutoff; evaluation uses
  rows whose feature window begins `embargo_hours` after it. Without the gap, an
  evaluation row's lagged features contain hours the model trained on as labels.
- **Per-fold scaling.** Any scaler is refit inside the fold. Scaling once over the
  whole frame leaks future distribution information into every fold.
- **Sparse classes.** With `min_class_support`, a class with too few examples in a
  block is marked unevaluable rather than scored 0.0. An F1 of zero on an absent class
  reads like a model failure and is not one.

**Report `wins`, not just the mean.** "Beats the baseline in 7 of 7 folds" is a
different and more honest claim than a mean difference, especially at small fold
counts where the significance test is weak.

### `advisory_disqualification(results_by_class, protected_classes, practical_threshold=0.01)`

Reject an intervention that helps on average by hurting a class you named as
safety-critical.

```python
verdict = advisory_disqualification(
    per_class_deltas, protected_classes=["Hazardous", "Very unhealthy"],
    aggregate_delta=0.0039, aggregate_significant=True)

if verdict["disqualified"]:
    print(verdict["reason"])
```

Precedence is deliberate: a significant regression on a protected class disqualifies
regardless of the aggregate. Only then is the aggregate gain checked against
`practical_threshold`, because a large test set lets a bootstrap resolve differences
far below anything that changes a decision.

### `bonferroni_report(comparisons, alpha=0.05, n_resamples=None, method="bonferroni")`

Family-wise correction over a set of baseline comparisons, with the resampling
resolution floor made explicit.

```python
rep = bonferroni_report(
    {"RandomForest": {"p": 0.002, "delta": +0.0055}, ...}, n_resamples=1000)
print(rep["summary"])
print(format_markdown(rep))
```

`n_resamples` matters: a bootstrap with 1,000 resamples cannot resolve a two-sided p
below 0.002, so values at that floor are shown as `<0.0020` rather than as an exact
figure the procedure could not have produced. The report separates claims that survive
correction from those that were significant at `alpha` and are not — the second group
is what a reader needs flagged.

`method` selects the correction. The default, `"bonferroni"`, compares every p against
`alpha / k`. `"holm"` applies the step-down procedure — sort ascending, compare the
i-th against `alpha / (k - i)`, stop at the first failure — which controls the same
family-wise error rate and is uniformly at least as powerful, so it never rejects
fewer hypotheses on the same input. Bonferroni remains the default because it is the
more conservative and the less arguable of the two.

The resolution floor is independent of the choice: Holm does not let a signed-rank
test over five folds produce a p below `2 ** (1 - 5)`. `format_markdown` names the
method it rendered and, for Holm, shows the per-rank threshold, since that differs by
position rather than being one number.

### `dataset_audit(df, time_col, value_col, group_col=None, period="YS")`

Structural checks for a fabricated or spliced segment in a published series. Four
tests, none conclusive alone:

| Check | What it looks for |
| --- | --- |
| `trend_linearity` | the longest leading segment a straight line explains too well |
| `hard_clip` | one exact value carrying far more mass than its neighbours |
| `scale_discontinuity` | a step in level between windows, as a unit change produces |
| `autocorrelation_break` | a regime change in lag-1 persistence |

```python
from pulsebench import dataset_audit
out = dataset_audit(df, time_col="datetime", value_col="pm2_5", group_col="city")
out["verdict"]              # 'fabrication suspected' | 'inconclusive' | 'no structural anomaly'
out["suspected_boundary"]   # where the series changes character, or None
out["boundary_candidates"]  # what each locating check proposed
out["boundary_agreement"]   # False when they disagree, rather than averaging them
```

Two design points earn their place. The trend test fits **period medians, not raw
observations** — hourly noise swamps a linear signal, and on this project's real
dataset annual medians give a prefix R² of 0.96 where the raw hours give 0.14. And
disagreeing boundary estimates are reported as disagreeing: an earlier version took
their median and turned a correct 2023 estimate plus a meaningless 2004 one into 2013.

It has a stated resolution limit: a fabricated segment shorter than `MIN_PERIODS`
aggregation units cannot be found, because a line through fewer points than that says
nothing. The aggregation rung is therefore chosen to yield at least `2 * MIN_PERIODS`
periods — a rung giving exactly `MIN_PERIODS` passes a length check and is still
useless, because the shortest prefix it can fit is the whole series, so a fabricated
prefix can never be separated from the genuine remainder. Pass a finer `period` when
the suspected segment is short.

**Validated against a real case.** Run on the unfiltered 1,048,551-row Mendeley
Bangladesh file with no boundary supplied, it returns *fabrication suspected* on 3 of
4 checks and dates the boundary to 2023-01-01, 149 days from the 2022-08-05 date that
manual inspection had found. It identifies Dhaka as the affected city unprompted.

## Tests

```bash
pytest pulsebench/tests/ -q
```

The suite runs on **synthetic data only**, which is the point: a package claiming to be
dataset-agnostic should not need the study's own data to demonstrate that. The one
exception is `test_retrofit_phase9.py`, which pins a frozen fixture of the original
study's output and checks that extracting this package into it changed no number — a
refactor that moved a fold boundary or a p-value would invalidate every result the
thesis reports.

## Limitations

- **Hourly-oriented.** `embargo_hours` is named for the granularity the package was
  built at. Other frequencies work via `freq`, but the embargo argument stays in hours.
- **Classification only.** The metrics are F1-based; regression targets are out of
  scope.
- **The signed-rank test is weak at small fold counts.** That is a property of the
  test, not of the implementation, and `aggregate["resolution"]` reports the floor so a
  null is never mistaken for evidence of absence.
- **No parallelism.** Folds are fit sequentially; the underlying estimator's own
  `n_jobs` still applies.

## Citing

See `CITATION.cff`.
