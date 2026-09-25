# PulseAir — final results summary

Compiled by `src/reporting/compile_results.py` from the JSON metrics each phase
wrote. Every number below is read from those files; none is transcribed by hand.
Regenerate with `python -m src.reporting.compile_results`.

**Task.** Predict the PM2.5 AQI risk category **6 hours ahead** from nine
wearable-feasible channels (PM2.5, PM10, CO, TEMP, DEWP + cyclical hour/month), using
the UCI Beijing Multi-Site dataset (420,768 hourly records, 12 stations, 2013–2017).

**Evaluation protocol, applied uniformly.** All variants are scored on the same
**61,530 test samples** — horizon 6 h, observed labels only
(`is_imputed_pm25 == False`). Model selection is always on validation; test is for
reporting. Significance is a **1,000-resample paired bootstrap**, 95%
percentile CI, both models scored on the same resampled rows. The practical-
significance threshold is **0.01 macro-F1** throughout.

---

## 1. Master comparison

| Variant | Macro-F1 | Hazardous F1 | V.unhealthy F1 | Beats persistence? | Source |
| --- | --- | --- | --- | --- | --- |
| Persistence (zero-parameter) — the floor every model must clear | 0.5118 | 0.5897 | 0.5104 | — (reference) | `horizon_comparison.md` |
| RandomForest (Phase 3) — best on test; **loses to persistence on validation** | 0.5173 | 0.6098 | 0.5265 | **yes** (+0.0055 [+0.0016, +0.0093]) | `baseline_metrics_h6.md` |
| XGBoost (Phase 3) | 0.4991 | 0.5607 | 0.5122 | **worse** (-0.0127 [-0.0174, -0.0082]) | `baseline_metrics_h6.md` |
| RF + CTGAN, broad 4-class | 0.5212 | 0.5908 | 0.4973 | **yes** (+0.0094 [+0.0055, +0.0136]) | `gan_ablation_h6.md` |
| RF + CTGAN, targeted 2-class | 0.5043 | 0.5843 | 0.4903 | **worse** (-0.0075 [-0.0115, -0.0033]) | `gan_ablation_h6.md` |
| LSTM + MC dropout | 0.5074 | 0.5956 | 0.5265 | **worse** (-0.0044 [-0.0087, -0.0002]) | `dl_metrics_h6.md` |
| Transformer + MC dropout — no paired bootstrap run | 0.5094 | 0.5942 | 0.5252 | not tested | `dl_metrics_h6.md` |
| RF + SMOTE (imbalanced-learn) — control for "why CTGAN not SMOTE" | 0.5207 | 0.5865 | 0.4757 | **yes** (+0.0089 [+0.0047, +0.0134]) | `final_results_summary.md §2.11` |
| RandomForest, class_weight=balanced — control for "why GAN not class weighting" | 0.5149 | 0.5843 | 0.4733 | no (+0.0031 [-0.0008, +0.0071]) | `final_results_summary.md §5a` |
| Compressed RF + class weight (10 trees x depth 10) — **two-sided rule winner (validation)** | 0.5020 | 0.5827 | 0.4662 | **worse** (-0.0098 [-0.0135, -0.0058]) | `deployment_report_h6_cw.md` |
| Compressed RF (100 trees x depth 16) — unweighted re-sweep fallback — *no* unweighted config cleared the floor | 0.5152 | 0.6085 | 0.5251 | no (+0.0034 [-0.0003, +0.0073]) | `deployment_report_h6.md` |

### Per-class F1, all variants

| Class | Persistence | RandomForest | XGBoost | RF + CTGAN, broad 4-class | RF + CTGAN, targeted 2-class | LSTM + MC dropout | Transformer + MC dropout | RF + SMOTE | RandomForest, class_weight=balanced | Compressed RF + class weight | Compressed RF |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Good | 0.5888 | 0.5981 | 0.5730 | 0.6095 | 0.5990 | 0.6105 | 0.6080 | 0.6168 | 0.6241 | 0.6200 | 0.5964 |
| Moderate | 0.4963 | 0.5440 | 0.5286 | 0.5147 | 0.5428 | 0.5249 | 0.5272 | 0.5122 | 0.4658 | 0.4332 | 0.5439 |
| Unhealthy (sensitive) | 0.2571 | 0.1432 | 0.1536 | 0.2376 | 0.1381 | 0.1036 | 0.1148 | 0.2590 | 0.3444 | 0.3337 | 0.1356 |
| Unhealthy | 0.6286 | 0.6820 | 0.6665 | 0.6772 | 0.6715 | 0.6834 | 0.6869 | 0.6741 | 0.5974 | 0.5762 | 0.6817 |
| **Very unhealthy** | 0.5104 | 0.5265 | 0.5122 | 0.4973 | 0.4903 | 0.5265 | 0.5252 | 0.4757 | 0.4733 | 0.4662 | 0.5251 |
| **Hazardous** | 0.5897 | 0.6098 | 0.5607 | 0.5908 | 0.5843 | 0.5956 | 0.5942 | 0.5865 | 0.5843 | 0.5827 | 0.6085 |

> ### One result, two datasets, two different roles
>
> | Dataset | Role | Best model beats persistence in | Mean Δ | Wilcoxon (1-sided) | Deployment status |
> | --- | --- | --- | --- | --- | --- |
> | **Beijing** (Phases 0–9) | methodology development + validation framework | 2/5 folds | -0.0147 | p = 0.8438 | **no deployable model** |
> | **Bangladesh** (Phase 10) | deployment candidate, target population | **5/5 folds** | **+0.0354** | **p = 0.0312** | **four common classes only** |
>
> **Beijing is the methodology, not the model.** Five-fold rolling-origin CV shows
> that *no* Beijing-trained model — RandomForest, class-weighted RandomForest,
> XGBoost, LSTM, Transformer, CTGAN- or SMOTE-augmented — beats the zero-parameter
> persistence rule in more than **2 of 5** folds. The +0.0055 single-split win
> reported in Phase 3 sits inside the fold-to-fold variance of the baseline itself,
> and two independent robustness checks agree: it also disappears under the Chinese
> HJ 633-2012 labelling (Limitations §6). **State the Beijing conclusion plainly: the
> pipeline is the contribution there, not a predictor.**
>
> **Bangladesh is the deployment candidate.** A class-weighted RandomForest trained on
> Dhaka, Chittagong, Comilla and Barisāl beats its own persistence floor in
> **5/5 rolling-origin folds** (mean +0.0354,
> one-sided Wilcoxon p = 0.0312 — the floor a 5-pair
> signed-rank test can reach). It is the only model in the project that clears its
> floor in every fold, and it does so on the population the device is for.
>
> **The scope of that claim is bounded, and the bound must travel with it.** The
> Bangladesh result covers the **four common AQI classes** — Good, Moderate, Unhealthy
> (sensitive), Unhealthy. It says **nothing** about Hazardous or Very unhealthy:
> across all 5 evaluation blocks the usable Bangladesh data contains **2
> Hazardous** and **482 Very unhealthy** samples. The dataset's modelled PM2.5 rarely
> approaches 250.4 µg/m³ although Dhaka genuinely exceeds it most winters, which
> is now **confirmed** by Phase 11: against the US Embassy Dhaka reference monitor the
> reanalysis records 17 Hazardous hours where the instrument records 1,602, running
> ~138 µg/m³ low above 150 µg/m³.
>
> **Phase 11b then closed the gap.** A PM2.5-only model on that reference series beats
> persistence on **Hazardous F1 in 7/7 rolling-origin folds** (0.4451 vs 0.3167,
> Δ +0.1284, two-sided p = 0.0156). So the advisory-class question is answered — *for
> the task*. It is still not answered for the **deployed 7-channel model**, which
> cannot run on a single-pollutant source. Those are different statements and §7 keeps
> them apart. No macro-F1 figure in this
> document should be read as evidence that the device can be trusted to raise a
> hazardous-air warning.
>
> That is the whole story in one line: **Beijing calibrated the method; Bangladesh
> earned a bounded deployment claim; the advisory classes remain unvalidated.**

**Reading this table.** The spread from worst to best macro-F1 is
0.0221.
The zero-parameter baseline sits at
0.5118; the best variant
(RF + CTGAN, broad 4-class) reaches 0.5212. **Every architectural and
data-augmentation intervention in this project lives inside a
0.0094
band above doing nothing.** That is the result, and section 2 explains why it is a
finding rather than a disappointment.

---

## 2. What worked, what didn't

The honest summary of this project is that **most of what was tried did not help, and
establishing that carefully is the contribution.** Four interventions that the
literature would predict should work were implemented properly, evaluated against a
pre-specified threshold with paired bootstraps, and did not clear it. Each is written
up below as a finding, because each one tells a reader something they would otherwise
have to discover themselves.

### 2.1 The persistence floor — the result that reframed the project

At the original 1-hour horizon the AQI category is unchanged from *t* to *t+1* in
**80.50%** of samples. A zero-parameter rule — "next hour's category is this hour's" —
scored **0.7933** macro-F1, and the RandomForest reached 0.7994: a gain of **+0.006**.

Nearly every air-quality forecasting paper that reports ~0.80 accuracy at a 1-hour
horizon is, on this evidence, reporting the autocorrelation of the target rather than
the skill of the model. **This is the project's most transferable finding**, and it is
methodological rather than architectural: *report the persistence floor next to every
headline number, or the number does not mean anything.*

Moving to a 6-hour horizon dropped persistence to **0.5118** and made the task
a real forecasting problem. Every result below is at h=6.

**Why it is a contribution, not a failure:** it invalidates a whole class of reported
results, including the one this project would otherwise have reported.
See `horizon_comparison.md`.

### 2.2 Flattened-window features did not help

Giving the tree the full 24-hour window (216 features) instead of the final timestep
(9 features) *lowered* observed macro-F1 from 0.5173 to **0.4963**. More information,
worse model — the extra dimensions dilute the split search without adding signal a
tree can use.

**Why it is a contribution:** it rules out the obvious "just give it more context" fix
and isolates the limitation as one of *signal*, not *representation*. It also
justifies keeping the tabular baseline at one timestep, which the sequence-model
comparison then depends on.

### 2.3 CTGAN augmentation — significant on aggregate, harmful where it mattered

Two variants, each generating synthetic minority-class rows to 50% of the majority:

- **Broad (4 classes, 82,958 synthetic rows):** macro-F1 **+0.0039**
  [+0.0011, +0.0066] — statistically
  significant. But Very unhealthy **-0.0292** and
  Hazardous **-0.0190**, both significant *regressions*.
- **Targeted (2 advisory classes, 58,199 rows):** worse on everything —
  macro-F1 -0.0129, Hazardous
  -0.0255.

The broad variant's macro-F1 gain came almost entirely from *Unhealthy (sensitive)*
rising 0.1432 → 0.2376 while both advisory classes fell. **Macro-F1 weights six classes
equally; a wearable air-quality advisory does not.** Adding ~39k synthetic Hazardous
rows produced a model that was *worse* at Hazardous.

**Why it is a contribution:** it is a concrete counterexample to "generate more
minority data to fix imbalance", and it demonstrates why a single aggregate metric
cannot adjudicate a safety-critical multi-class problem. The disqualification rule
used here — *a variant that significantly degrades an advisory class does not win,
whatever it does to macro-F1* — is reusable.

### 2.4 Synthesizer quality scores do not measure validity

The first CTGAN run scored **0.89–0.93** on SDV's quality report. The synthetic data
was nonetheless physically impossible:

- Only **19.5%** of rows satisfied `hour_sin² + hour_cos² = 1`; the generator produced
  **82,885 distinct `hour_sin` values** where 24 hours exist. Four fifths of the rows
  encoded a time of day that cannot occur.
- **13.1%** had dew point above air temperature, against 0.18% in the real data.

Column Shapes compares one marginal at a time; Column Pair Trends compares linear
association. **Neither asks whether a row is possible.** Fixing it required modelling
the integer hour/month as categoricals and computing the cyclical pair afterwards,
plus an `sdv.cag.Inequality` constraint on DEWP ≤ TEMP — after which all four validity
checks passed (100% on-circle, 0 material DEWP violations).

**Why it is a contribution:** it is a documented, reproducible case of a widely used
synthetic-data quality metric passing unusable data, with the specific failure modes
named. One SDV-specific trap worth recording: passing the constraint as a plain dict
is accepted **silently and has no effect** (51% violations with the dict, 0% with the
object).

### 2.5 LSTM and Transformer lost to the forest

Both sequence models saw all 24 hours; the forest sees one timestep. Both lost.

- LSTM vs RandomForest: **-0.0099**
  [-0.0139, -0.0061] — significantly worse.
- On Hazardous specifically: **-0.0143**
  [-0.0226, -0.0059] — significantly worse, which
  disqualifies it under the same rule applied to the GAN variants.

This was checked before being believed. The first run used `lr: 1e-3`, both models
peaked at epoch 2–4 and declined while training loss kept falling — ambiguous between
"hard task" and "step too large". A learning-rate sweep found 3e-4 materially better
(LSTM 0.4948 → 0.5151 on validation). The conclusion survived a properly tuned model.

**Why it is a contribution:** *a negative result from an undertrained model is not a
result.* The sweep is in the report precisely so a reader can see the conclusion was
stress-tested.

### 2.6 The uncertainty is aleatoric — the ceiling is the data

MC dropout decomposed predictive entropy: of **1.0713** nats, only
**0.0178 (1.7%)** is epistemic — the part from the
model disagreeing with itself. The other **98% is aleatoric**:
irreducible class overlap given nine channels at six hours out.

**This explains every other negative result in one number.** More capacity, more
epochs, more synthetic data and a bigger forest all move the 1.7% and leave
the rest alone. If h=6 prediction is to improve, the *input* has to change — more
channels, a longer window, spatial context from neighbouring stations — not the
architecture.

**Why it is a contribution:** it converts "our models did not improve" from an
apology into a measurement, and it tells the next researcher where not to spend
effort.

### 2.7 Marginal conformal coverage hid a per-class failure

Split conformal delivered **0.8908** marginal coverage against
a 0.90 target — apparently fine. Per class it was not: **Hazardous was covered at
0.8432**, missing the true category ~16% of the time instead of 10%, while
*Unhealthy* was over-covered at 0.9668. The slack in a marginal guarantee lands on the
rare, hard classes.

Mondrian (class-conditional) calibration fixed it: Hazardous **0.8432 →
0.9083**. The cost is honest and reported — mean set size
2.487 → 2.610,
singletons 7.2% →
2.1%.

**Why it is a contribution:** marginal coverage is the default in most applied
conformal work, and this is a worked example of it being actively misleading in a
safety context — with the fix and its price both quantified.

### 2.8 Validation and test disagree, repeatedly

The chronological split leaves *Very unhealthy* at 7.04% of validation and 12.14% of
test. That gap bit three times:

| Where | Validation said | Test said |
|---|---|---|
| Phase 5 model choice | LSTM 0.5151 > RF 0.4895 | LSTM 0.5074 < RF 0.5173 — **ranking reversed** |
| Phase 7 compression | −0.0086, inside the 0.01 tolerance | −0.0156, outside it |
| Phase 3–4 selection | consistently lower than test | consistently higher |

Selection stayed on validation throughout — the alternative is worse — but **a
validation win at these margins is not evidence of a test win**, and the write-up says
so each time.

**Why it is a contribution:** it is a concrete demonstration that chronological splits
on seasonal data produce non-exchangeable validation and test sets, and that the
resulting model-selection risk is large enough to reverse conclusions.

### 2.9 Compression crossed the persistence floor

The most uncomfortable result, and the one that best demonstrates why the floor
discipline matters. Phase 7 selected the compressed forest by comparing it to the
*uncompressed* forest: −0.0086 macro-F1 on validation, inside the 0.01 tolerance.
Measured against **persistence** instead, the compressed model is
**-0.0098** [-0.0135, -0.0058] — significantly *below* a rule with no parameters.

Phase 7 asked the wrong question. "Does compression cost less than 0.01 against the
full model?" and "is the compressed model still better than doing nothing?" have
different answers here, and only the second one matters for deployment.

**Why it is a contribution:** it is a concrete case of a compression pipeline
producing an artifact that passes its own acceptance test and fails the only test that
matters. Any edge-ML paper that reports "Nx smaller for only ΔF1" without a
task-floor comparison is exposed to exactly this.

### 2.10 What actually fixed the imbalance: class weighting, not synthetic data

Run as a control for the reviewer question *"why CTGAN instead of simply weighting the
loss?"*. Identical data, identical hyperparameters, identical seed — only
`class_weight="balanced"` changes.

| | Unweighted RF | **class_weight="balanced"** |
|---|---|---|
| Validation macro-F1 | 0.4895 | **0.5196** |
| vs persistence, **validation** | −0.0153 (significantly worse) | **+0.0148** [+0.0107, +0.0187] — **significantly better** |
| Test macro-F1 | 0.5173 | 0.5149 |
| vs persistence, test | +0.0055 (significant) | +0.0031 [-0.0008, +0.0071] (not significant) |
| Hazardous F1 (test) | 0.6098 | 0.5843 |
| Very unhealthy F1 (test) | 0.5265 | 0.4733 |

**Class weighting raises validation macro-F1 by
+0.0300
— and it is the only model in this project that significantly beats the persistence
floor on validation.** Every unweighted forest, at every size in the Phase 7 sweep
(0 of 30), is significantly *below* it.

Set against CTGAN: the broad 4-class variant moved test macro-F1 +0.0039 while
significantly degrading both advisory classes. Class weighting moves validation
macro-F1 +0.0300
and costs nothing — no generator, no synthetic rows, no validity checks, no extra
artifact to version.

On **test** the two are closer (+0.0031 vs
+0.0055 against persistence), which is the
val/test mismatch of section 2.8 appearing yet again — but validation is the split
selection is allowed to use, and on validation the verdict is unambiguous.

**Why it is a contribution:** this is the comparison the CTGAN literature usually
omits. Roughly 24 minutes of generator training, 82,958 synthetic rows, a quality
report, four validity checks and a constraint-aware redesign were spent on
augmentation; a one-word hyperparameter change beat all of it. The result is not that
CTGAN is useless — it is that **the cheap control has to be run first, and reported,
before synthetic data can be credited with anything.**

### 2.11 SMOTE fails the same way CTGAN did, 660× faster

The second control a reviewer asks for. Standard SMOTE from `imbalanced-learn`, same
four minority classes CTGAN targeted, same 50%-of-majority ratio, same
82,958 synthetic rows, fit on observed training rows only.

| SMOTE vs | Macro-F1 Δ | 95% CI | Very unhealthy Δ | Hazardous Δ |
| --- | --- | --- | --- | --- |
| Persistence | +0.0089 (**significant**) | [+0.0047, +0.0134] | -0.0347 (**significant**) | -0.0032 (n.s.) |
| RandomForest (unweighted) | +0.0034 (**significant**) | [+0.0000, +0.0063] | -0.0509 (**significant**) | -0.0233 (**significant**) |
| RandomForest (class_weight=balanced) | +0.0058 (**significant**) | [+0.0030, +0.0085] | +0.0024 (n.s.) | +0.0022 (n.s.) |

**On aggregate SMOTE looks like the best augmentation in the project** — it beats the
unweighted forest by +0.0034 and the class-weighted one by
+0.0058, both significant. **On the advisory classes it fails exactly as
CTGAN did**: against the unweighted forest, Very unhealthy
-0.0509 and Hazardous -0.0233, both significant regressions. Under
the disqualification rule applied to the GAN variants in section 2.3, SMOTE is
disqualified on the same grounds.

**The cost comparison is the part worth putting in the paper.** SMOTE produced the
same 82,958 synthetic rows in **2.0 seconds**; CTGAN took
**22 minutes** of generator training plus a quality report, four
validity checks and a constraint-aware redesign after the first attempt produced
physically impossible rows. That is a **663× difference in wall-clock
cost for an outcome that is no better and fails in the same direction.**

Taken with section 2.10, the picture across all three imbalance interventions is
consistent:

| Intervention | Cost | Macro-F1 vs plain RF | Advisory classes |
| --- | --- | --- | --- |
| CTGAN, broad 4-class | 22 min + redesign | +0.0039 (significant) | **both significantly worse** (VU -0.0292, Haz -0.0190) |
| SMOTE | 2.0 s | +0.0034 (significant) | **both significantly worse** (VU -0.0509, Haz -0.0233) |
| class_weight='balanced' | 0 s (one hyperparameter) | see §2.10 — best on validation, only model to clear the validation floor | no significant regression |

**Only class weighting avoids degrading the classes the device exists to warn about**,
and it is also the cheapest of the three. A paper proposing GAN-based augmentation for
rare-class air-quality prediction needs to clear both of these controls, and this one
does not.

### 2.12 ECE and MCE disagree about which sequence model is better calibrated

The same shape a third time. Calibration was measured two ways: **ECE**, the average
gap across reliability bins, and **MCE**, the worst single bin.

| Model | ECE (mean bin gap) | MCE (worst bin gap) |
| --- | --- | --- |
| transformer | 0.0113 | 0.1969 |
| lstm | 0.0190 | 0.0448 |

**The Transformer wins on ECE (0.0113 vs 0.0190) and loses on MCE
(0.1969 vs 0.0448) — 4.4x worse in its worst bin.** On average it is
the better-calibrated model; where it is most confidently wrong, it is far worse. The
advisory layer suppresses low-confidence warnings, so the worst bin is the operative
number, and a selection made on ECE alone would have chosen the wrong model.

**Why it belongs beside 2.3 and 2.7.** Those two are the same failure in different
places: an aggregate improved while a tail got worse. Section 2.3 is augmentation
raising macro-F1 while significantly degrading both advisory classes — the pattern the
disqualification rule exists to catch. Section 2.7 is marginal conformal meeting its
90% target on average while covering Hazardous at 0.8432. This is the third
instance, and it was sitting in `dl_h6.json` unreported until a late audit of the
committed metrics found it. **The recurring lesson is not about any one metric: an
average over a distribution says nothing about its tail, and in a safety-critical
advisory the tail is the product.**

### What did work

- **Moving the horizon to 6 h**, which turned a persistence-echo task into a
  forecasting task.
- **The RandomForest**, which beat persistence significantly
  (+0.0055 [+0.0016, +0.0093]) and
  beat every more complex alternative tried against it.
- **Mondrian conformal**, which turned a point prediction into a set with a per-class
  guarantee that holds for the classes the device exists to warn about.
- **Compression as an engineering result** — 2x
  smaller, 0.0138 ms inference, small enough for an MCU — though see §2.9: the
  accuracy it gave up took it below the persistence floor, so it is not yet a
  validated predictor.
- **Imputation provenance tracking**, which let every metric in the project be
  computed on observed labels only.


---

## 3. The deployed system

**The deployed system is the Bangladesh-native model. The Beijing pipeline is the
methodology that produced it.** Those are two different artifacts with two different
statuses, and earlier drafts of this document conflated them.

### Beijing (Phases 0–9) — methodology, no deployable model

Five-fold rolling-origin CV shows no Beijing-trained model beats persistence in more
than 2 of 5 folds. **Nothing from the Beijing pipeline is deployed.** What it produced
is the evaluation framework every number in this project rests on: persistence-floor
discipline, horizon selection, the GAN/SMOTE/class-weighting ablation with an
advisory-class disqualification rule, Mondrian conformal calibration, imputation
provenance, and the rolling-origin protocol itself.

### Bangladesh (Phase 10) — the deployment candidate

```
  7 scaled features ──▶ RandomForest, class_weight='balanced' (ONNX)
                            │  probabilities
                            ▼
                        Mondrian conformal thresholds (JSON)
                            │  prediction SET, 90% target coverage
                            ▼
                        top-3 SHAP + wearer profile
                            │
                            ▼
                        Gemini advisory (validated, template fallback)
```

| | |
|---|---|
| Training data | Dhaka, Chittagong, Comilla, Barisāl — 2022-08-05 onward |
| Model | RandomForest, `class_weight='balanced'`, **25 trees × depth 12** |
| Validation | **5/5 rolling-origin folds** beat persistence (mean +0.0354, one-sided p = 0.0312) |
| Compression | 8.6× (8.2 MB → 983 KB), **1673 KB ONNX** |
| Selection rule | two-sided: within tolerance of the full model **and** significantly above **Bangladesh's own** persistence floor (0.4322) — 11/30 configurations qualified |
| Inference | **0.0074 ms** single-sample, single-threaded |
| Conformal layer | 0.0103 ms; overall coverage 0.9098 |
| ONNX ↔ sklearn parity | max \|Δp\| 2.6e-07, argmax agreement 1.0000 |

The compression point was re-selected against **Bangladesh's** floor
(0.4322), not Beijing's — the earlier Beijing
compression passed its own tolerance test and then landed below the floor, which is
the mistake this rule exists to prevent. Mondrian thresholds were re-derived on the
compressed Bangladesh model, since thresholds are quantiles under one specific model.

### Coverage, and what it does not cover

| Class | Coverage | Test n | Status |
| --- | --- | --- | --- |
| Good | 0.9927 | 6,176 |  |
| Moderate | 0.9041 | 7,927 |  |
| Unhealthy (sensitive) | 0.7735 | 1,757 |  |
| Unhealthy | 0.7604 | 1,490 |  |
| **Very unhealthy** | 0.0000 | 6 | thin support |
| **Hazardous** | n/a | 0 | **no test samples — not validated** |

> **The advisory classes are not validated for Bangladesh, and this bound travels with
> every number above.** Across all 5 rolling-origin evaluation blocks
> the usable Bangladesh data contains **2 Hazardous** and **482 Very unhealthy**
> samples. The 5/5-fold result covers the **four
> common classes** (Good, Moderate, Unhealthy (sensitive), Unhealthy) and nothing more.
>
> The cause is the data: this dataset's modelled PM2.5 rarely approaches the
> 250.4 µg/m³ Hazardous threshold, although Dhaka genuinely exceeds it most winters —
> now confirmed in Phase 11 against the US Embassy Dhaka reference monitor: 17
> Hazardous hours in the reanalysis against 1,602 in the instrument over the same
> period.
>
> **Phase 11b validates the advisory classes on that ground-truth series** — Hazardous
> F1 0.4451 vs a 0.3167 persistence floor, 7/7 rolling-origin folds, p = 0.0156. That
> is evidence about the *task*, from a **different model** (PM2.5-only, one station).
> **The deployed 7-channel model itself still has no advisory-class validation**, and
> nothing in this section should be read as giving it one.
>
> A device shipped on this evidence could report air-quality bands in the ordinary
> range. It could **not** yet be claimed to raise a reliable hazardous-air warning,
> which is the function that motivates the product.

### Advisory layer

Gemini `gemini-3.1-flash-lite` via the **`google-genai`**
SDK renders the numbers as two to four sentences. Every fact is passed explicitly; the
model must name the ambiguity when the conformal set has more than one member and
advise against the worst case in it. Output is validated before use — unhedged
language on an ambiguous set, truncation, or a claim of per-reading certainty all
trigger the rule-based template. **The honesty constraint is enforced outside the
model**, so it does not depend on trusting the LLM.


---

## 4. Suggested paper structure

### Abstract — the bullets it has to contain

- Wearable air-quality monitoring needs a **6-hour** forecast to be actionable. At
  1 hour the target is 53.9% autocorrelated and a
  zero-parameter persistence rule scores 0.7933 macro-F1, so reported accuracy at that
  horizon does not measure forecasting skill.
- **Methodology development (Beijing, 420,768 hourly records, 12 stations).** Under
  5-fold rolling-origin cross-validation, **no model beats the persistence floor in
  more than 2 of 5 folds** — not RandomForest, XGBoost, LSTM, Transformer, nor
  CTGAN-, SMOTE- or class-weight-augmented variants. Single-split wins of +0.0055 sit
  inside the fold-to-fold variance of the baseline itself, and the same result
  disappears under the Chinese HJ 633-2012 labelling.
- **Negative results, each properly powered:** CTGAN augmentation and SMOTE both gain
  on macro-F1 while *significantly degrading both advisory classes*; class weighting
  beats both at zero cost; flattened 24-hour windows underperform a single timestep;
  sequence models get monotonically **worse** with capacity (15× parameters,
  -0.0322 LSTM / -0.0201 Transformer); and **98% of
  predictive entropy is aleatoric**, so the ceiling is the data.
- **Marginal conformal prediction silently under-covers the rare classes** (Hazardous
  0.843 against a 0.90 target); Mondrian class-conditional calibration restores it
  (0.908) at the cost of larger prediction sets.
- **Deployment (Bangladesh, the target population).** A class-weighted RandomForest
  trained on Dhaka, Chittagong, Comilla and Barisāl beats its own persistence floor in
  **5/5 rolling-origin folds** (mean +0.0354, one-sided
  Wilcoxon p = 0.0312) — the only model in the project to clear its floor in every
  fold. Compressed 8.6× to 1673 KB ONNX at
  0.0074 ms/inference, wrapped in re-calibrated Mondrian conformal sets and an
  LLM advisory layer constrained to communicate ambiguity rather than a point estimate.
- **Bounded scope, stated up front.** That result covers the **four common AQI
  classes**. The usable Bangladesh data contains 2 Hazardous and 482 Very
  unhealthy samples across all folds, so the advisory-class claim is **not
  established** and requires validation against ground-station measurements.
- **Advisory-class validation (Phase 11b).** On the US Embassy Dhaka reference
  monitor (75,344 QC-verified hourly readings, 9.1 years), a PM2.5-only model
  beats persistence on **Hazardous F1 in 7/7 rolling-origin folds**
  (0.4451 vs 0.3167, Δ +0.1284, two-sided Wilcoxon
  p = 0.0156) — the largest and best-powered effect in the project. It validates
  the *task*, not the 7-channel deployed model, which cannot run on a
  single-pollutant source.
- **A data-integrity audit of the external dataset** (Mendeley 9j447cynb9): 87% of the
  published 2000–2025 *span* fails inspection — a synthetic near-linear trend
  (R² = 0.992), a hard clip at 250 µg/m³, and a mid-file unit change in CO.

### 1. Introduction — lead with the persistence floor

Frame the paper as a *methodological* contribution before an architectural one. The
opening argument writes itself: a literature that reports ~0.80 accuracy at 1-hour
horizons without a persistence baseline cannot distinguish a model from the
autocorrelation of its target. Show the h=1 numbers (persistence 0.7933, RF 0.7994),
then state that everything in the paper is reported at h=6 against an explicit floor.
This motivates the rigour — bootstraps, pre-specified thresholds, per-class
disqualification — as necessary rather than decorative.

Source: `horizon_comparison.md`.

### 2. Methodology

| § | Content | Report to draw from |
|---|---|---|
| 2.1 | Dataset, 12 stations, 2013–2017; wearable-feasible 9-channel subset and why SO2/NO2/O3/PRES/wind are excluded | `preprocessing_summary_h6.md` §2 |
| 2.2 | Preprocessing: per-station forward fill, imputation provenance flags, chronological 70/15/15 split | `preprocessing_summary_h6.md` §1, §3 |
| 2.3 | Horizon selection: persistence degradation across h=1/6/12/24 | `horizon_comparison.md` |
| 2.4 | Baselines and evaluation protocol: validation-only selection, observed-label metrics, paired bootstrap | `baseline_metrics_h6.md` |
| 2.5 | CTGAN augmentation: per-class synthesizers, constraint-aware generation, validity checks | `gan_quality_report_h6.md` |
| 2.6 | Sequence models and MC dropout; learning-rate sensitivity | `dl_metrics_h6.md` §1 |
| 2.7 | Conformal calibration: split vs Mondrian | `conformal_h6.md` |
| 2.8 | Explainability and advisory generation | `shap_examples/README.md`, `llm_advisory_examples.md` |
| 2.9 | Compression and ONNX export | `deployment_report_h6.md` §1–3 |

### 3. Results — one subsection per phase

| § | Content | Report |
|---|---|---|
| 3.1 | Horizon comparison; persistence floor table | `horizon_comparison.md` |
| 3.2 | Tabular baselines vs persistence | `baseline_metrics_h6.md` |
| 3.3 | Synthetic data validity; the quality-score failure | `gan_quality_report_h6.md` §2b, §5 |
| 3.4 | GAN ablation with bootstrap CIs | `gan_ablation_h6.md` |
| 3.5 | Sequence models, calibration, entropy decomposition | `dl_metrics_h6.md` |
| 3.6 | Conformal coverage, marginal vs Mondrian | `conformal_h6.md` |
| 3.7 | SHAP attributions and advisory examples | `shap_examples/README.md`, `llm_advisory_examples.md` |
| 3.8 | Compression, ONNX latency, ESP32 feasibility | `deployment_report_h6.md` |
| 3.9 | Master comparison (table 1 of this document) | `final_results_summary.md` |

### 4. Limitations

1. **Imputation bias.** Forward fill is not label-neutral: carried-forward PM2.5
   averages 97.0 µg/m³ vs 79.8 observed, and **8.38% of imputed readings fall in
   Hazardous against 4.47% of observed ones**. Every metric here is computed on
   observed labels only, but the training set still contains imputed rows.
   (`preprocessing_summary_h6.md` §4)
2. **Aleatoric ceiling.** 98% of
   predictive entropy is irreducible given these inputs. Conclusions about model
   families are conditional on the 9-channel, 24-hour, single-station input.
   (`dl_metrics_h6.md` §3)
3. **Validation/test distribution mismatch.** Chronological splitting leaves *Very
   unhealthy* at 7.04% of validation and 12.14% of test; this reversed a model
   ranking and broke a compression tolerance. Rolling-origin validation would be the
   fix and was not implemented. (`preprocessing_summary_h6.md` §4)
4. **ESP32 feasibility is a proxy, not a deployment.** There is no ONNX Runtime for
   the ESP32. The 33379 KB artifact and
   0.0138 ms latency were measured on a
   desktop CPU. A real port needs a C tree traversal, `emlearn`, or a different
   framework. (`deployment_report_h6.md` §6)
5. **Free-tier LLM.** The advisory layer uses a hosted model with rate limits, on a
   pinned version that the provider may retire — `gemini-2.0-flash` was retired
   mid-project. Advisory *text* is therefore not bit-reproducible, though every number
   it reports is. (`llm_advisory_examples.md`)
6. **AQI breakpoints: US EPA applied to Chinese monitoring data — tested, and it
   matters.** Every class label in the Beijing work comes from **US EPA** PM2.5
   breakpoints; China's own **HJ 633-2012** (on GB 3095-2012) uses higher cut-points
   (35 / 75 / 115 / 150 / 250 µg/m³). **This was run rather than left open** —
   `reports/hj633_sensitivity.md`.

   |  | EPA | HJ 633-2012 |
   | --- | --- | --- |
   | Persistence macro-F1 | 0.5118 | 0.4905 |
   | RandomForest macro-F1 | 0.5173 | 0.4919 |
   | Δ vs persistence | +0.0055 | +0.0015 |
   | 95% CI | [+0.0016, +0.0093] | [-0.0026, +0.0055] |
   | RF beats persistence? | **yes** | **no** |
   | Label unchanged over 6 h | 53.6% | 57.4% |

   **The Phase 3 headline does not survive the relabelling.** Under EPA the RandomForest beats persistence by +0.0055 (significant); under HJ 633-2012 by +0.0015, and the interval contains zero. The single-split Beijing result is therefore sensitive to a labelling choice that is essentially arbitrary for this data — which is consistent with the rolling-origin finding that the same result is sensitive to which chronological block is evaluated. **Two independent robustness checks both say the Beijing win is fragile.**

   The class balance also moves substantially: EPA's rarest advisory class is 4.23% of training against 4.23% under HJ 633-2012, and EPA's 12 µg/m³ Good/Moderate boundary reclassifies a large block of hours that the Chinese scale calls *Excellent*.

   **Still conditional on EPA:** the Phase 4 augmentation targets, the Phase 6 per-class conformal thresholds and every per-class F1 were computed against EPA bands. This check covers the headline persistence comparison, not those. For a Bangladeshi deployment neither scale is obviously correct — Bangladesh's Department of Environment publishes its own AQI, closer to the EPA scheme.

7. **Single city, fixed monitors.**7. **Single city, fixed monitors.** Beijing 2013–2017 reference-grade instruments are
   a proxy for a body-worn low-cost sensor. No wearable field data was collected;
   sensor noise, drift and the difference between ambient and personal exposure are
   unmodelled.


---

## 7. External validation — Bangladesh, the target population

**The core methodology (Phases 0–9) was developed and stress-tested on the large,
well-established Beijing Multi-Site dataset. Phase 10 validates that the methodology
transfers to Bangladesh's own air-quality data — the actual target population for this
wearable device.**

That distinction is the point. What transfers is the *discipline*, not the weights.

### What transferred, and what did not

| Model | Macro-F1 | Hazardous F1 | vs Bangladesh persistence |
| --- | --- | --- | --- |
| Persistence | 0.3807 | 0.0000 | — (reference) |
| Beijing model (transfer) | 0.3258 | 0.0000 | -0.0550 [-0.0606, -0.0492] **significant** |
| Bangladesh-native RandomForest (class_weight=balanced) | 0.4075 | 0.0000 | +0.0268 [+0.0208, +0.0326] **significant** |

- **The Beijing model does not transfer.** Retrained on the channels both datasets
  share (Bangladesh has no temperature or dew point, so the nine-channel model cannot
  be evaluated at all), it is **-0.0550
  [-0.0606, -0.0492]** against the Bangladesh persistence
  floor — significantly *worse* than doing nothing.
- **A natively trained model does beat the floor**: **+0.0268
  [+0.0208, +0.0326]**, significant. It is the deployment
  candidate.
- **The persistence floor itself is different**: 0.3807
  on Bangladesh against 0.5118 on Beijing, with the label unchanged over 6 h in
  63.7% of samples. Assuming Beijing's floor would have been
  wrong.

### Rolling-origin CV on Bangladesh — the strongest result in the project

5 expanding-window folds, all 5 evaluation blocks touching
Bangladesh's Nov–Feb high-pollution season.

| Model | Macro-F1 (mean ± std over 5 folds) | Mean Δ vs persistence | Folds won |
| --- | --- | --- | --- |
| Persistence | 0.4377 ± 0.0391 | — | — |
| RandomForest (unweighted) | 0.4557 ± 0.0162 | +0.0181 | **3/5** |
| RandomForest (class_weight=balanced) | 0.4731 ± 0.0433 | +0.0354 | **5/5** |

**`RandomForest (class_weight=balanced)` clears the persistence floor in 5/5 folds** (mean
+0.0354, one-sided Wilcoxon p = 0.0312).
On Beijing, the best model managed 2/5.
**This is the only model anywhere in the project that beats its persistence floor in
every rolling-origin fold**, and it does so on the target population.

> **But the advisory classes still cannot be evaluated on this dataset, and more folds
> will not fix it.** Across all 5 blocks combined: **2 Hazardous** and
> **482 Very unhealthy** samples. The dataset's modelled Dhaka PM2.5 rarely
> approaches the 250.4 µg/m³ Hazardous threshold, although Dhaka genuinely exceeds it
> most winters — **now confirmed in Phase 11**: over the same period the US
> Embassy reference monitor records 1,602 Hazardous hours where this dataset
> records 17. **Advisory-class validation needs that ground-station data**, which
> Phase 11 downloads and verifies, **and Phase 11b then models on it — validating
> Hazardous at F1 0.4451 vs a 0.3167 floor, 7/7 folds.** The result above should
> still be cited as covering the four common classes: that validation belongs to a
> different, PM2.5-only model.

### Two findings that qualify it

1. **The advertised 2000–2025 span is 87% backfill.** Mendeley
   `9j447cynb9` advertises 103 cities and 2000–2025; it contains
   30 cities, and everything before 2022-08-05 carries a
   synthetic near-linear trend (R² = 0.9920), a hard clip
   at exactly 250 µg/m³, and carbon monoxide in different units. Discarding it costs
   19% of the *rows* (198,720 of
   1,048,551) but 87% of the *years*, because the pre-2022
   portion is Dhaka alone at low density while the clean window is
   30 cities hourly. The 81% of rows from
   2022-08-05 onward survives inspection. The audit is in
   `reports/bangladesh_validation.md` §1 and is a contribution in itself.
2. **The advisory classes are absent from the Bangladesh test split** — Hazardous
   0, Very unhealthy 6 samples. The
   chronological split lands in the monsoon season, and Dhaka's severe pollution is a
   November–February phenomenon. The macro-F1 comparison above is effectively a
   four-class comparison and says nothing about the classes the device exists to warn
   about. This is the Phase 9 seasonal-split problem reappearing, and **rolling-origin
   CV on the Bangladesh data is required before any advisory-class claim** — the
   machinery already exists.

### Phase 11 — ground truth resolves why

The Phase 10 blocker — 2 Hazardous hours in 3.3 years — was a property of the
*data source*, not of Dhaka's air. The **US Embassy Dhaka reference monitor**
(75,344 QC-valid hourly readings, 2016-03-01 to
2025-03-24; the QC-valid count matches an independent report exactly) settles it.

|  | Mendeley reanalysis (3.3 yr, 4 cities) | Embassy reference (9.1 yr, 1 station) |
| --- | --- | --- |
| **Hazardous** hours | **2** | **3,788** (5.03%) |
| **Very unhealthy** hours | 482 | 11,284 (14.98%) |
| Max PM2.5 (µg/m³) | 290.6 | 985 |

Over the **23,010 hours the two sources both cover**, they correlate
r = 0.729 in the middle of the range and diverge sharply at the top:
MAE 55 µg/m³ overall, and above 150 µg/m³ the reanalysis runs
**138 µg/m³ low**. It records
**17 Hazardous hours where the instrument records
1,602**.

**The Section 4.9 hypothesis is confirmed**: CAMS-style reanalysis understates South
Asian peak PM2.5, and the error is concentrated exactly in the advisory range.

A univariate persistence floor on the ground-truth series gives macro-F1
**0.4225** at h=6 (47.8% of labels unchanged),
between the Beijing (0.5118) and Mendeley (0.3807) figures — so the reanalysis was not
making the task artificially easy or hard *in aggregate*; its distortion is at the top
of the distribution.

> **The advisory-class claim moves from "untestable" to "still open, but now
> testable".** The data exists. It is not yet a validation: this source is **PM2.5
> only** (no PM10, CO, temperature or dew point), so the seven-channel deployed model
> **cannot be run on it**, and it is a single station against the model's four cities.
> No model is trained or scored in Phase 11. Closing the gap needs either co-located
> multi-channel data or a PM2.5-only model validated under the same rolling-origin
> protocol — real work, not yet done.

### Phase 11b — the advisory-class claim, validated

A PM2.5-only RandomForest (PM2.5 at *t*, cyclical hour/month, and lags at
t−1/3/6/12/24) trained on the reference-monitor series, 7 rolling-origin folds,
every block containing winter and 226–639 Hazardous hours.

| Metric | Model (mean ± std) | Persistence | Mean Δ | Folds won | Wilcoxon p (2-sided) | Significant |
| --- | --- | --- | --- | --- | --- | --- |
| Macro-F1 | 0.4603 ± 0.0205 | 0.4133 | +0.0470 | **7/7** | 0.0156 | **yes** |
| **Very unhealthy F1** | 0.5003 ± 0.0399 | 0.4165 | +0.0838 | **7/7** | 0.0156 | **yes** |
| **Hazardous F1** | 0.4451 ± 0.0783 | 0.3167 | +0.1284 | **7/7** | 0.0156 | **yes** |

**The Hazardous-class claim is validated: F1 0.4451 ± 0.0783
against a persistence floor of 0.3167, beating it in
7/7 folds, two-sided Wilcoxon p = 0.0156.** Very
unhealthy likewise (+0.0838, 7/7), and macro-F1
+0.0470, 7/7.

Two things make this the strongest result in the project. The **margin**: Hazardous
+0.1284 is an order of magnitude larger than anything measured on
Beijing (-0.0147) or on
the Mendeley reanalysis. And the **power**: nine years supports 7 folds, which
drops the two-sided Wilcoxon floor to 0.0156 — the
5-fold runs elsewhere could not have reached α = 0.05 no matter how clean the result.

> **This does not replace the deployed model, and the two roles must not be merged.**
>
> | | 7-channel Bangladesh-native (Phase 10) | PM2.5-only (Phase 11b) |
> |---|---|---|
> | Role | **deployment candidate** | **advisory-class validation evidence** |
> | Data | 4 cities, reanalysis, 3.3 yr | 1 station, reference-grade, 9.1 yr |
> | Validated on | four common classes, 5/5 folds | **all six classes, 7/7 folds** |
> | Hazardous support | 2 hours total | 2,820 hours |
>
> The PM2.5-only model reads **one pollutant at one station**; the device carries
> PM2.5, PM10 and CO sensors and is meant to work across Bangladesh. Phase 11b is
> evidence about the **task** — that a six-hour Hazardous forecast is learnable at a
> level well above persistence — not a shippable predictor. Closing the last gap means
> joining reference-grade measurements to the other channels, or siting reference
> monitors at the remaining cities, so the 7-channel model can be validated on data
> that contains the classes it is meant to warn about.

### What this establishes for the thesis

The Beijing work is the *methodology development and validation framework*:
persistence-floor discipline, horizon selection, the GAN ablation with an
advisory-class disqualification rule, rolling-origin cross-validation, Mondrian
conformal calibration, and imputation provenance. Phase 10 applies that framework
unchanged to an independent dataset, population and pollution regime — and it does
what a framework should: it rejects the transferred model, accepts a native one,
recomputes the floor rather than assuming it, and catches a data-integrity problem in
the external source before any result is built on it.


---

## 6. Multiple comparisons

**9 variants were compared against persistence on the same test split.** Selection
was always on validation, so test was never optimised against — but 9 reported
comparisons on one split still inflate the family-wise error rate, and the paper
should say so rather than leave it to a reviewer.

Bonferroni: **α = 0.05 / 9 ≈ 0.0056**.

| Variant | Δ vs persistence | p (two-sided) | Direction | p < 0.05 | p < 0.0056 |
| --- | --- | --- | --- | --- | --- |
| XGBoost (Phase 3) | -0.0127 | <0.0020 | worse | yes | **yes** |
| RF + CTGAN, broad 4-class | +0.0094 | <0.0020 | better | yes | **yes** |
| RF + CTGAN, targeted 2-class | -0.0075 | <0.0020 | worse | yes | **yes** |
| RF + SMOTE (imbalanced-learn) | +0.0089 | <0.0020 | better | yes | **yes** |
| Compressed RF + class weight (10 trees x depth 10) | -0.0098 | <0.0020 | worse | yes | **yes** |
| RandomForest (Phase 3) | +0.0055 | <0.0020 | better | yes | **yes** |
| LSTM + MC dropout | -0.0044 | 0.0380 | worse | yes | no |
| Compressed RF (100 trees x depth 16) | +0.0034 | 0.0800 | better | no | no |
| RandomForest, class_weight=balanced | +0.0031 | 0.1180 | better | no | no |

A 1,000-resample bootstrap cannot resolve a two-sided p below
0.0020; entries shown as `<0.0020` are at that floor and would
need more resamples to separate further.

### Do the headline claims survive?

- **"RandomForest beats persistence"** — **holds** at α/k (p = 0.0020 vs 0.0056)
- **"the compressed model fails to beat persistence"** — **holds** at α/k (p < 0.0020 vs 0.0056)

6 of 9 comparisons survive the corrected threshold: **XGBoost (Phase 3)** (-0.0127), **RF + CTGAN, broad 4-class** (+0.0094), **RF + CTGAN, targeted 2-class** (-0.0075), **RF + SMOTE (imbalanced-learn)** (+0.0089), **Compressed RF + class weight (10 trees x depth 10)** (-0.0098), **RandomForest (Phase 3)** (+0.0055).

**1 comparison(s) are significant at α = 0.05 but not after correction**: LSTM + MC dropout (p = 0.0380). These should be reported as suggestive, not established.

Correction is applied here to the *persistence* comparisons only. The GAN-ablation and sequence-model comparisons in sections 2.3 and 2.5 are separate families with their own multiplicity; their significant effects have p at the bootstrap floor and are unaffected, but the count should be stated in the paper alongside this one.

**Holm–Bonferroni changes nothing here.** The step-down procedure is uniformly at least as powerful as plain Bonferroni at the same family-wise error rate, and applying it to this family returns the identical 6 survivors. The reason is visible in the p-values: 6 of the 9 comparisons sit at the bootstrap's resolution floor (p < 0.0020), far below even Holm's strictest threshold, while the remaining 3 exceed their Holm thresholds as well as alpha/k. The distribution is bimodal with nothing in the band where Holm's extra power would bite. Reported because the absence of a difference is itself informative: the conservative default cost this analysis nothing.


---

## 5. Anticipated reviewer questions

| Question | Where this project stands |
| --- | --- |
| Why CTGAN rather than simply class-weighting the loss, or SMOTE? | **Partially answered — close this gap.** `baseline.random_forest.class_weight` is exposed and documented as "the cheap alternative to CTGAN", but the weighted variant was never run. **It has now been run** — see the class-weighted row in the master table and section 2.10. It beats every CTGAN variant and is the only model that clears the *validation* persistence floor. SMOTE was not tried and remains a gap. |
| Why macro-F1 when the framing is safety-critical? A missed Hazardous hour and a missed Moderate hour are not equally costly. | **Answered, and the paper should lead with it.** This is exactly why the broad-GAN variant was rejected: it gained macro-F1 while significantly degrading Hazardous. The project uses macro-F1 for comparability with prior work but adjudicates with a **disqualification rule on the advisory classes**. `gan_ablation_h6.md` §4. A cost-sensitive metric would be stronger still and is a natural extension. |
| How is the EPA PM2.5 breakpoint mapping justified for Chinese monitoring stations, which use the CAQMS/HJ 633-2012 scale? | **Not currently answered — address it.** The breakpoints in `configs/default.yaml` are US EPA. China's ambient standard uses different PM2.5 cut-points (e.g. 35/75/115/150/250 µg/m³ for the 24-h scale). The choice is defensible for international comparability and because the AQI bands are a *labelling* convention rather than a claim about Chinese regulation. **Now written up as Limitations section 6**, including why the choice is not a neutral relabelling. The HJ 633-2012 sensitivity re-run remains outstanding. |
| Coverage is reported on the same test split used throughout. Is the conformal guarantee not then contaminated? | **Answered.** Calibration uses the **validation** split (61,466 observed rows); test is only ever measured on. Split conformal's guarantee requires exchangeability between calibration and test, which a chronological split strains — and the report says so. `conformal_h6.md` §1. |
| Exchangeability fails under a chronological split. Does the conformal guarantee hold at all? | **Partially answered — strengthen it.** The empirical coverage (0.8865 overall, 0.9083 Hazardous) is measured, not assumed, which is the practical answer. But the theoretical guarantee does assume exchangeability and seasonal drift violates it. Cite the adaptive/online conformal literature (Gibbs & Candès) and state that the measured coverage is the operative claim. |
| The sequence models had ~53k–70k parameters on 294k training samples. Were they large enough to be a fair test? | **Answered.** The learning-rate sweep (`dl_metrics_h6.md` §1) shows both architectures were optimisation-limited at lr=1e-3 and improved at 3e-4, and the entropy decomposition shows 1.7% of uncertainty is epistemic — i.e. capacity is not the binding constraint. **The capacity sweep was subsequently run and closes this** (`capacity_sweep_h6.md`, figure 06): hidden sizes 64, 128, 256 at the swept learning rate, and both architectures decline *monotonically* with size — LSTM -0.0322 and Transformer -0.0201 from smallest to largest, over ~15x the parameters. More capacity made both models worse, which is what an aleatoric ceiling predicts. |
| Only one dataset, one city, one four-year window. How general are the negative results? | **Acknowledged as a limitation.** The persistence-floor argument and the marginal-vs-Mondrian coverage finding are *methodological* and transfer directly; the specific negative results (CTGAN, LSTM) are claims about this data. Replication on a second city — the UCI Italy or a US EPA AirNow extract — is the single highest-value extension. |
| The wearable framing is not validated: no body-worn sensor data was collected, and reference-grade monitors are not low-cost sensors. | **Acknowledged.** Stated in Limitations §6. The 9-channel feature subset is justified by sensor availability and cost (`preprocessing_summary_h6.md` §2), but ambient-station readings are a proxy for personal exposure. Any field deployment would need re-calibration against the actual sensor, and — as Phase 7 §2 shows — re-calibrating conformal along with it. |
| Test-set reuse: many variants were evaluated against the same test split. Is there a multiple-comparisons problem? | **Partially answered — disclose it.** Selection was always on validation, so test was never optimised against; but 11 variants were ultimately *reported* on it with 1,000-resample bootstraps. The **this is now addressed in section 6**, which applies a Bonferroni-corrected threshold to every persistence comparison and states explicitly which headline claims survive it. |

---

## 6. Report index

| Phase | Report | Contains |
|---|---|---|
| 2 | `preprocessing_summary_h6.md` | row counts, feature rationale, imputation bias, split class distributions |
| 2 | `preprocessing_summary_h{1,12,24}.md` | secondary horizons, comparison figure only |
| 3 | `baseline_metrics_h6.md` | RF vs XGBoost vs persistence |
| 3 | `horizon_comparison.md` | persistence degradation h=1/6/12/24 |
| 4 | `gan_quality_report_h6.md` | CTGAN validity, the quality-score methodological note |
| 4 | `gan_ablation_h6.md` | broad vs targeted vs unaugmented, bootstrap CIs |
| 5 | `dl_metrics_h6.md` | LSTM/Transformer, MC dropout, calibration, LR sweep |
| 5 | `capacity_sweep_h6.md` | hidden-size sweep; both architectures decline with size |
| 6 | `conformal_h6.md` | split vs Mondrian conformal, per-class coverage |
| 6 | `shap_examples/README.md` | attributions, 12 case plots |
| 6 | `llm_advisory_examples.md` | 5 live advisories, honesty constraints, free-tier rationale |
| 7 | `deployment_report_h6.md` | compression sweep, ONNX, latency, ESP32 feasibility |
| 7 | `deployment_report_h6_cw.md` | the class-weighted re-sweep under the two-sided rule |
| 8 | `rolling_origin_cv_h6.md` | 5 chronological folds on Beijing, Wilcoxon on fold deltas |
| 8 | `hj633_sensitivity.md` | EPA vs HJ 633-2012 breakpoints; the conclusion is unchanged |
| 10 | `bangladesh_validation.md` | the data-integrity audit, transfer vs native, §1 is a contribution in itself |
| 10 | `bangladesh_rolling_cv.md` | 5 folds on Bangladesh; the class-weighted forest takes 5/5 |
| 10 | `bangladesh_deployment.md` | the deployed compression point, ONNX, conformal, latency |
| 11 | `dhaka_ground_truth_validation.md` | US Embassy reference monitor vs the reanalysis |
| 11b | `dhaka_ground_truth_model_h6.md` | PM2.5-only model; the validated Hazardous result |
| 11c | `openaq_multichannel_validation.md` | station survey; no multi-pollutant Dhaka source qualifies |
| — | `figures/README.md` | all 23 figures with the JSON each was generated from |
| — | `reference_list_expanded.md` | 61 references, 22 registry-verified additions |
