# HJ 633-2012 sensitivity analysis

Every class label in this project comes from **US EPA** PM2.5 breakpoints applied to
Chinese monitoring data. China's own standard, **HJ 633-2012** (built on
GB 3095-2012), uses higher cut-points. This was listed as an open limitation; it is
closed here.

The breakpoints are not a neutral relabelling — they define the class boundaries and
therefore the class balance, the rare-class prevalence that motivated the CTGAN work,
and the conformal thresholds. The question is whether the headline result survives
the change.

**Scope:** only the class distribution and the Phase 3 baseline are re-run. The GAN
ablation, sequence models, conformal calibration and deployment work are not.

---

## 1. The two schemes

| Scheme | PM2.5 cut-points (µg/m³) | Classes |
| --- | --- | --- |
| US EPA AQI | 12 / 35.4 / 55.4 / 150.4 / 250.4 | Good, Moderate, Unhealthy (sensitive), Unhealthy, Very unhealthy, Hazardous |
| HJ 633-2012 | 35 / 75 / 115 / 150 / 250 | Excellent, Good, Lightly polluted, Moderately polluted, Heavily polluted, Severely polluted |

The Chinese scale starts its second band at 35 µg/m³ where EPA starts at 12, so a
large block of hours that EPA calls *Moderate* the Chinese scale calls *Excellent*.

---

## 2. How much the class balance moves

### Under EPA

| Class | Train % | Test % | Train n |
| --- | --- | --- | --- |
| Good | 13.98% | 17.89% | 40,266 |
| Moderate | 21.81% | 22.79% | 62,824 |
| Unhealthy (sensitive) | 13.09% | 10.46% | 37,697 |
| Unhealthy | 35.66% | 30.59% | 102,722 |
| **Very unhealthy** | 11.23% | 12.28% | 32,338 |
| **Hazardous** | 4.23% | 5.99% | 12,185 |

### Under HJ 633-2012

| Class | Train % | Test % | Train n |
| --- | --- | --- | --- |
| Excellent | 35.79% | 40.67% | 103,090 |
| Good | 23.94% | 19.63% | 68,961 |
| Lightly polluted | 16.03% | 13.54% | 46,174 |
| Moderately polluted | 8.78% | 7.89% | 25,284 |
| **Heavily polluted** | 11.23% | 12.28% | 32,338 |
| **Severely polluted** | 4.23% | 5.99% | 12,185 |

**The rarest advisory class goes from 4.23% (EPA) to 4.23%
(HJ 633-2012)** of the training split. The Chinese scheme is
more skewed at the top end, which
strengthens the case for the rare-class
augmentation work in Phase 4 — that work was sized against the EPA prevalence, and
the thesis should say which scale its imbalance argument depends on.

---

## 3. Does the RandomForest still beat persistence?

Same protocol: chronological split, observed labels only, 1,000-resample paired
bootstrap on the same 61,530 test rows.

| Labelling | Persistence macro-F1 | RF macro-F1 | Δ | 95% CI | RF beats persistence? |
| --- | --- | --- | --- | --- | --- |
| EPA (used throughout this project) | 0.5118 | 0.5173 | +0.0055 | [+0.0016, +0.0093] | **yes** |
| HJ 633-2012 (Chinese national standard) | 0.4905 | 0.4919 | +0.0015 | [-0.0026, +0.0055] | no |

Label persistence over the 6-hour horizon also differs:
**53.6%** of samples keep their EPA class against
**57.4%** under HJ 633-2012 — a coarser scale at the low
end means fewer boundary crossings, which is why the persistence floor itself moves.

---

## 4. Verdict

**The conclusion changes under the Chinese labelling.** Under EPA the RandomForest is +0.0055 against persistence; under HJ 633-2012 it is +0.0015. The Phase 3 result is sensitive to the breakpoint scheme, and the thesis must say so.

**What remains conditional on the EPA scale.** The Phase 4 augmentation targets, the
Phase 6 per-class conformal thresholds, and every per-class F1 in the project were
computed against EPA bands. This analysis shows the *headline* persistence comparison
is robust to the choice; it does not show the rare-class results are. A full re-run
under HJ 633-2012 remains future work, and is a one-line config change
(`data.pm25_breakpoints`) plus the pipeline.

**For a Bangladeshi deployment neither scale is obviously correct** — Bangladesh's
Department of Environment publishes its own AQI, closer to the US EPA scheme than to
China's. The labelling should follow the jurisdiction the device is used in, and this
analysis shows how to check whether that choice matters.

Reproduce with `python -m src.models.hj633_sensitivity`.
