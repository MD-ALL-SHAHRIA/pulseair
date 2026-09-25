# Sensor-noise robustness — Hazardous detector, horizon 6 h

Phase 11b trained a Hazardous detector on the **US Embassy Dhaka reference monitor**, a
research-grade instrument. The device this thesis designs would not carry one. This
section asks how much of that result survives input from a sensor a wearable can
afford.

**This is a synthetic proxy for real sensor noise, not a substitute for field
calibration.** No low-cost sensor was deployed, no co-location study was run, and
nothing here measures how the PulseAir hardware actually behaves. The field-calibration
work remains listed as a limitation.

---

## 1. Where the noise parameters come from

Every parameter is taken from one published validation study, and **the parameters that
study does not report are not modelled rather than invented**:

> Nyarku, Mazaheri, Jayaratne, Dunbabin, Rahman, Uhde & Morawska (2018), "Mobile phones as monitors of personal exposure to air pollution: Is this the future?", PLOS ONE 13(2), e0193150

| Reported quantity | Published value | Used here |
| --- | --- | --- |
| Ambient R² vs reference, phone 1 at Rocklea | 0.10 | yes |
| Ambient R² vs reference, phone 2 at Rocklea | 0.23 | yes |
| Ambient R² vs reference, phone 1 at Woolloongabba | 0.28 | yes |
| Ambient R² vs reference, phone 2 at Woolloongabba | 0.15 | yes |
| Chamber R², elevated concentrations | 0.86 – 1.00 | context only |
| Response below noise level at ambient PM2.5 | ~10 µg/m³ | yes, as a detection floor |
| Calibration slope / intercept | **not reported** | **not modelled** |
| Quantisation step | **not reported** | **not modelled** |
| Numerical PM2.5 bias | **not reported** | **not modelled** |

The paper evaluated a phone-mounted PM2.5 sensor against reference instruments in a
chamber and at two ambient monitoring stations. Its ambient agreement figures are the
only quantitative PM2.5 agreement it gives, and they are what the noise amplitude here
is calibrated to reproduce.

**Method.** For additive noise, the squared correlation between observed and true is
`var(true) / (var(true) + var(noise))`, so a published R² fixes the noise variance:
`var(noise) = var(true) · (1/R² − 1)`. The amplitude therefore follows from the
published agreement figure and the series' own variance rather than from a number
chosen to look plausible. Readings falling below the reported ~10 µg/m³ detection floor
are replaced by noise, because a sensor below its noise level returns noise rather than
a constant.

**Target R² = 0.19** (the mid-range of the four published ambient
values). Achieved R² after injection: **0.2041**; noise standard
deviation **167.0 µg/m³**; **33.1%** of readings
fell below the detection floor; mean absolute error against the reference series
**96.4 µg/m³**.

**This is an optimistic proxy.** It degrades the input with noise alone, at a magnitude
the literature supports, and omits the calibration bias and quantisation the same
literature says exist but does not quantify. A real sensor would do worse. Read the
numbers below as a **lower bound on degradation**.

## 2. Result

The model is trained on clean reference data exactly as in Phase 11b and **is not
retrained**. Within each fold it is scored twice on the same rows: once on clean input
and once after injection. The target label stays clean — the question is whether
genuinely hazardous air is still detected through a cheap sensor. Persistence is
recomputed from the noisy reading, since a device carrying this sensor would have
nothing better to persist.

| Metric | Clean input | Noise-injected | Absolute drop | Relative drop | Noisy floor | Beats its floor? |
| --- | --- | --- | --- | --- | --- | --- |
| Hazardous F1 | 0.4451 ± 0.0783 | 0.3193 ± 0.0798 | −0.1259 | 28.3% | 0.1790 | **yes** (7/7 folds) |
| Macro-F1 | 0.4603 ± 0.0205 | 0.2640 ± 0.0194 | −0.1963 | 42.7% | 0.1498 | **yes** (7/7 folds) |
| Accuracy | 0.5082 ± 0.0337 | 0.2891 ± 0.0157 | −0.2191 | 43.1% | 0.1729 | **yes** (7/7 folds) |

## 3. Per fold

| Fold | Evaluation window | n | Haz clean | Haz noisy | Δ |
| --- | --- | --- | --- | --- | --- |
| 1 | 2019-01-28 to 2019-12-13 | 7,367 | 0.3966 | 0.2654 | -0.1312 |
| 2 | 2019-12-15 to 2020-11-01 | 7,382 | 0.4468 | 0.3002 | -0.1466 |
| 3 | 2020-11-03 to 2021-09-17 | 7,171 | 0.4848 | 0.3556 | -0.1293 |
| 4 | 2021-09-19 to 2022-08-30 | 7,386 | 0.3086 | 0.1801 | -0.1285 |
| 5 | 2022-09-01 to 2023-07-10 | 7,390 | 0.5629 | 0.4241 | -0.1389 |
| 6 | 2023-07-12 to 2024-05-17 | 7,389 | 0.4545 | 0.3361 | -0.1184 |
| 7 | 2024-05-19 to 2025-03-24 | 7,381 | 0.4615 | 0.3735 | -0.0880 |

## 4. What this establishes

- **Hazardous F1 falls from 0.4451 to 0.3193**,
  a relative loss of **28.3%**.
- Against the floor a device would actually face — persistence computed from the same
  noisy sensor, 0.1790 — the degraded model
  still holds a margin of +0.1403,
  winning **7 of 7** folds.
- The comparison that matters for a product is the second one. A model that loses
  accuracy on cheap input but still beats what that same cheap input gives you for
  free is doing something; one that does not is not worth shipping.

**The honest bound.** This says what happens under noise of a published magnitude with
no bias and no quantisation. It does not say what happens under a real sensor's
systematic error, because the study cited does not report one and this project has not
measured one. Closing that gap needs a co-location study against a reference monitor,
which is hardware work rather than modelling work.

Regenerate with `python -m src.models.sensor_noise_robustness`.
