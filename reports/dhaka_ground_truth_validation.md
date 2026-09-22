# Phase 11 — ground-truth validation, US Embassy Dhaka reference monitor

Phases 10 and its rolling-origin follow-up both stopped at the same wall: the Mendeley
reanalysis product contains **2 Hazardous hours in 3.3 years**, so the
advisory-class claim could not be evaluated. Section 4.9 hypothesised that the
reanalysis understates South Asian peak PM2.5. This phase tests that against a
reference-grade instrument.

Source: AirNow Embassy historical files (`files.airnowtech.org`), Dhaka,
2016–2025, one station, PM2.5 only, hourly, QC-flagged.

---

## 1. Download and verification

|  | Value |
| --- | --- |
| Yearly files downloaded | 10 |
| Rows concatenated | 77,716 |
| QC flags | Valid 75,374, Missing 2,138, Invalid 197, Suspect 7 |
| QC-Valid rows | **75,374** |
| Independently reported | 75,374 |
| Match | **exact** |
| Sentinel (−999) rows dropped | 24 |
| Duplicate timestamps dropped | 6 |
| Final usable rows | **75,344** |
| Date range | 2016-03-01 .. 2025-03-24 (9.06 years) |
| Hourly coverage | 94.8% of 79,457 slots |

The QC-Valid count matches the independently reported figure **exactly**, and the date
range matches to the day. Unlike the Mendeley product, this source survives inspection:
the ~5% of missing hours are real instrument downtime, flagged as such rather than
back-filled.

PM2.5 (µg/m³): min 0, median 65,
99th percentile **361**, max **985**.

---

## 2. Class distribution — the headline comparison

Same EPA breakpoints as everywhere else in this project
(12/35.4/55.4/150.4/250.4 µg/m³).

| Class | Hours | Share |
| --- | --- | --- |
| Good | 2,536 | 3.37% |
| Moderate | 16,702 | 22.17% |
| Unhealthy (sensitive) | 13,346 | 17.71% |
| Unhealthy | 27,688 | 36.75% |
| **Very unhealthy** | 11,284 | 14.98% |
| **Hazardous** | 3,788 | 5.03% |

|  | Mendeley reanalysis (3.3 yr, 4 cities) | Embassy reference (9.1 yr, 1 station) |
| --- | --- | --- |
| **Hazardous** hours | **2** | **3,788** |
| **Very unhealthy** hours | 482 | 11,284 |
| Hazardous as % of record | ~0.0002% | 5.03% |

**The reference instrument records 1,894× more Hazardous
hours than the reanalysis product**, from a single station over a longer record. This
is the finding that motivated the phase, and it is unambiguous.

---

## 3. Persistence floor on real ground-truth data

Univariate: does the class at *t* predict the class at *t+6*? No model, no features
beyond PM2.5 itself.

|  | Value |
| --- | --- |
| Sample pairs | 74,535 |
| Label unchanged over 6 h | 47.78% |
| **Persistence macro-F1 (h=6)** | **0.4225** |
| Accuracy | 0.4778 |

| Class | Support | Persistence F1 |
| --- | --- | --- |
| Good | 2,479 | 0.3384 |
| Moderate | 16,456 | 0.5586 |
| Unhealthy (sensitive) | 13,186 | 0.3353 |
| Unhealthy | 27,459 | 0.5515 |
| **Very unhealthy** | 11,200 | 0.4291 |
| **Hazardous** | 3,755 | 0.3221 |

### How this compares

| Dataset | Persistence macro-F1 (h=6) | Label unchanged |
| --- | --- | --- |
| Beijing (reference-grade, 12 stations) | 0.5118 | 53.9% |
| Bangladesh Mendeley (reanalysis, 4 cities) | 0.3807 | 63.7% |
| **Dhaka Embassy (reference-grade, 1 station)** | **0.4225** | **47.8%** |

The difficulty of the task on real ground-truth Dhaka data sits
between the two,
which is a useful sanity check on the reanalysis-based results: the reanalysis was not
producing an artificially easy or artificially hard problem in aggregate — its
distortion is concentrated at the top of the range, as section 4 shows.

---

## 4. Direct comparison: reanalysis vs reference instrument

The two sources align best at an offset of **+6 h**, consistent with the Embassy files being local time (UTC+6) and the reanalysis archive being UTC. The offset was found by scanning correlation over ±12 h rather than assumed.

| PM2.5 statistic (µg/m³) | Reference instrument | Reanalysis | Difference |
| --- | --- | --- | --- |
| Mean | 105.3 | 52.6 | -52.7 |
| 95th percentile | 270.0 | 124.3 | -145.7 |
| 99th percentile | 397.0 | 177.4 | -219.6 |
| Maximum | 970.0 | 290.6 | -679.4 |
| Hazardous hours | 1,602 | 17 | -1,585 |
| Very unhealthy hours | 3,921 | 436 | -3,485 |

Correlation over the overlap: Pearson **0.7293**,
Spearman 0.8088, MAE
**55.0 µg/m³**, class agreement
30.8%.

### Does this confirm the Section 4.9 hypothesis?

**Confirmed, and the size of the effect is larger than the hypothesis
assumed.** Over 23,010 jointly observed hours the two sources agree
reasonably in the middle of the range (Pearson r = 0.729, MAE
55.0 µg/m³) and diverge sharply at the top. The reanalysis 99th percentile is
220 µg/m³ below the instrument's, its
maximum is 679 µg/m³ below, and over
the same period it records **17 Hazardous hours against the
instrument's 1,602**.

Restricted to hours the instrument puts at or above 150.4 µg/m³
(5,523 of them), the reanalysis runs
**138 µg/m³ low on average** (MAE 138). The bias is not
uniform noise — it is concentrated exactly where an air-quality advisory has to be
right.

---

## 5. Verdict

**Hazardous-class advisory claim: partially validated — the blocker is removed.**

### Can the Hazardous-class advisory claim now be validated?

**Partially — the blocker is removed, but the claim is not yet made.**

What changed: the reference monitor records **3,788 Hazardous hours**
(5.03% of 75,344) over 9.06 years, against
**2 hours in 3.3 years** in the reanalysis product. The class that
could not be evaluated at all in Phase 10 is abundant here. Very unhealthy likewise:
11,284 hours against 482.

What has **not** changed, and must not be overstated:

- **This dataset is PM2.5-only.** No PM10, CO, temperature or dew point. The deployed
  Bangladesh model takes seven channels and **cannot be run on it**. Nothing in this
  phase evaluates that model.
- **It is one station.** The Bangladesh model is trained across four cities; a single
  embassy rooftop in Dhaka is not a substitute for that spatial spread.
- **No model has been trained or scored here.** The only quantity computed is the
  persistence floor, which needs no model.

**The honest status of the advisory-class claim is: still open, but now testable.**
Phase 10 could not test it because the data contained 2 positive examples. This phase
shows the data exists. Closing it requires either (a) joining the reference PM2.5
series to co-located meteorology and pollutant channels so the multi-channel model can
run, or (b) training a PM2.5-only model on this series and validating it under the same
rolling-origin protocol. Either is a real piece of work and neither is done.

### What this phase does establish

1. **The Section 4.9 hypothesis is confirmed.**
   The reanalysis product systematically understates peak PM2.5 in Dhaka, and the error is concentrated exactly in the advisory range.
2. **The Phase 10 sparsity was an artifact of the data source, not of Dhaka's air.**
   Dhaka has abundant Hazardous hours; the reanalysis product does not represent them.
3. **Any future deployment claim for the advisory classes must be built on
   ground-station data.** This phase identifies the source, verifies it against an
   independent count, and shows it contains the signal — which is the prerequisite the
   earlier phases were missing.

Reproduce with `python -m src.models.dhaka_ground_truth`.
