# Phase 11b — PM2.5-only model on Dhaka ground truth

The experiment the project has been building toward. Every Bangladesh-side result
until now covered the four common AQI classes only, because the Mendeley reanalysis
contains 2 Hazardous hours in 3.3 years. Phase 11 showed the US Embassy reference
monitor records 985 µg/m³ peaks and thousands of Hazardous hours.
This trains and validates a model on it.

> **Two models, two jobs. This one does not replace the deployed model.**
>
> | | 7-channel Bangladesh-native (Phase 10) | **PM2.5-only (this phase)** |
> |---|---|---|
> | Role | **deployment candidate** | **advisory-class validation evidence** |
> | Data | 4 cities, reanalysis, 3.3 yr | 1 station, reference-grade, 9.1 yr |
> | Channels | PM2.5, PM10, CO + time | PM2.5 + lags + time |
> | Validated on | four common classes, 5/5 folds | all six classes, 7 folds |
> | Hazardous support | 2 hours total | 2,820 hours across folds |
>
> Neither subsumes the other: this one has the rare classes but one pollutant and one
> site; that one has the channels and the spatial spread but no rare classes.

---

## 1. Features and gap handling

Without other pollutants, recent history is the only signal. Features
(10): PM2.5 at *t*, cyclical hour/month, and lagged PM2.5 at
t−1, t−3, t−6, t−12, t−24. Target: the EPA class at t+6 h, same
breakpoints as everywhere else.

|  | Hours |
| --- | --- |
| Complete hourly grid | 79,457 |
| Observed readings | 75,344 |
| Missing (instrument downtime) | 4,113 |
| Short gaps (≤3 h) forward-filled + flagged | 535 |
| Long gaps left unfilled | **3,578** |
| **Windows dropped** | **4,980** (6.3%) |
| **Usable samples** | **74,477** |
| Samples with an imputed label | 499 |

Gaps longer than 3 hours are **not** filled, and any
sample whose lags or target reach across one is dropped. Carrying a value forward
across a multi-day outage would manufacture exactly the autocorrelation the
persistence floor measures, which would flatter every model in this report.

---

## 2. Single chronological split

70%/15%/15%, RandomForest with
`class_weight='balanced'` — the only imbalance intervention that did not damage the
advisory classes in Phases 4, 2.10 and 2.11. Validation macro-F1
0.4695; 11,144 observed test samples.

| Class | Model F1 | Persistence F1 | Δ | Test n |
| --- | --- | --- | --- | --- |
| Good | 0.3733 | 0.4361 | -0.0628 | 199 |
| Moderate | 0.5186 | 0.5506 | -0.0320 | 1,619 |
| Unhealthy (sensitive) | 0.4333 | 0.3563 | +0.0770 | 1,742 |
| Unhealthy | 0.6251 | 0.5785 | +0.0466 | 4,427 |
| **Very unhealthy** | 0.4900 | 0.4794 | +0.0107 | 2,264 |
| **Hazardous** | 0.4526 | 0.3237 | +0.1289 | 893 |
| **Macro-F1** | **0.4822** | 0.4541 | **+0.0281** |  |

Paired bootstrap (1,000 resamples) against persistence:
macro-F1 **+0.0281**
[+0.0150, +0.0410];
Hazardous **+0.1289**
[+0.1004, +0.1540];
Very unhealthy **+0.0107**
[-0.0119, +0.0338].

---

## 3. Rolling-origin cross-validation (7 folds)

Nine years of data supports more folds than the 3.3-year Bangladesh set. **7 folds**
at 30% initial training: every evaluation block contains
Nov–Feb, every block carries real Hazardous support, and — the reason for going above
five — **the two-sided Wilcoxon floor falls to 0.0156**, so
unlike the 5-fold runs elsewhere in this project the test can actually reach α = 0.05.
Embargo 48 h (the widest feature reach, t−24 plus the horizon).

| Fold | Eval block | Winter | Train n | Eval n | Hazardous n | Model macro-F1 | Persistence | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2019-01..2019-12 | yes | 22,337 | 7,367 | 226 | 0.4421 | 0.3657 | +0.0765 |
| 2 | 2019-12..2020-11 | yes | 29,784 | 7,382 | 266 | 0.4451 | 0.4011 | +0.0440 |
| 3 | 2020-11..2021-09 | yes | 37,232 | 7,171 | 494 | 0.4911 | 0.4246 | +0.0665 |
| 4 | 2021-09..2022-08 | yes | 44,680 | 7,386 | 233 | 0.4385 | 0.3935 | +0.0450 |
| 5 | 2022-09..2023-07 | yes | 52,127 | 7,390 | 639 | 0.4677 | 0.4325 | +0.0353 |
| 6 | 2023-07..2024-05 | yes | 59,575 | 7,389 | 448 | 0.4561 | 0.4228 | +0.0333 |
| 7 | 2024-05..2025-03 | yes | 67,023 | 7,381 | 514 | 0.4817 | 0.4530 | +0.0287 |

### The advisory classes, per fold

| Fold | Haz n | Haz F1 model | Haz F1 persist | Δ | VU F1 model | VU F1 persist | Δ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 226 | 0.3966 | 0.2545 | +0.1421 | 0.4443 | 0.2545 | +0.1898 |
| 2 | 266 | 0.4468 | 0.3483 | +0.0985 | 0.5159 | 0.4281 | +0.0878 |
| 3 | 494 | 0.4848 | 0.3125 | +0.1723 | 0.5403 | 0.4608 | +0.0795 |
| 4 | 233 | 0.3086 | 0.2398 | +0.0688 | 0.5591 | 0.4400 | +0.1191 |
| 5 | 639 | 0.5629 | 0.4088 | +0.1542 | 0.4810 | 0.4095 | +0.0715 |
| 6 | 448 | 0.4545 | 0.3144 | +0.1401 | 0.4836 | 0.4750 | +0.0086 |
| 7 | 514 | 0.4615 | 0.3385 | +0.1230 | 0.4783 | 0.4479 | +0.0304 |

### Aggregate

| Metric | Model (mean ± std) | Persistence | Mean Δ | Folds won | Wilcoxon p (2-sided) | Significant |
| --- | --- | --- | --- | --- | --- | --- |
| Macro-F1 | 0.4603 ± 0.0205 | 0.4133 | +0.0470 | **7/7** | 0.0156 | **yes** |
| Good | 0.2890 ± 0.1271 | 0.3221 | -0.0332 | **2/7** | 0.2969 | no |
| Moderate | 0.5269 ± 0.0321 | 0.5535 | -0.0267 | **1/7** | 0.0781 | no |
| Unhealthy (sensitive) | 0.4088 ± 0.0541 | 0.3172 | +0.0916 | **7/7** | 0.0156 | **yes** |
| Unhealthy | 0.5920 ± 0.0411 | 0.5538 | +0.0382 | **6/7** | 0.0312 | **yes** |
| **Very unhealthy** | 0.5003 ± 0.0399 | 0.4165 | +0.0838 | **7/7** | 0.0156 | **yes** |
| **Hazardous** | 0.4451 ± 0.0783 | 0.3167 | +0.1284 | **7/7** | 0.0156 | **yes** |

---

## 4. Is the Hazardous-class claim validated?

**Yes — the Hazardous-class claim is validated, at the level stated below.**

The model beats persistence on **Hazardous F1 in 7/7 rolling-origin
folds**, mean 0.4451 against persistence's
0.3167 (Δ +0.1284, two-sided Wilcoxon
p = 0.0156). Every evaluation block contains winter and carries real
Hazardous support (226–639
hours). Macro-F1 is 0.4603 ± 0.0205, beating persistence
in 7/7 folds.

Unlike the 5-fold runs elsewhere in this project, **7 folds let the two-sided test
actually reach significance** — its floor here is 0.0156 rather
than 0.0625.

### What this does and does not license

- **Does:** the advisory-class evidence gap that blocked Phases 10 and 10b is closed.
  Hazardous is measurable on this data (226–639
  hours per evaluation block against 2 in the entire Mendeley record), and the
  question has an answer.
- **Does not:** license a deployment claim for the neckband. This model reads **one
  pollutant at one station**. The device carries PM2.5, PM10 and CO sensors and is
  meant to work across Bangladesh. This is validation evidence about the *task*, not
  a shippable predictor.
- **Next step, stated concretely:** join this reference PM2.5 series to co-located
  meteorology and the other pollutant channels — or deploy reference-grade monitors
  at the other three cities — so the 7-channel model can be validated on data that
  actually contains the classes it is meant to warn about.

Reproduce with `python -m src.models.dhaka_pm25_model`.
