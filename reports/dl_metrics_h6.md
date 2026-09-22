# Sequence models — horizon 6 h

LSTM and Transformer encoder over the 24-hour window,
predicting the same six-class AQI risk label as every earlier phase. MC dropout with
**50 stochastic passes** supplies the predictive distribution.

Trained on **unaugmented** `data/processed/h6/sequences_train.npz`
(294,192 sequences). Phase 4 found both CTGAN variants significantly degrade
the advisory classes, so augmentation is off. Fitting uses every real training row;
selection and every number below use **observed rows only** (61,530
of 62,772 test sequences), excluding imputed labels. Architecture was chosen
on **validation** macro-F1; test was not consulted until the table below.

---

## 1. Training and selection

| Architecture | Parameters | Val macro-F1 (observed) | Best/run epochs | Minutes |
| --- | --- | --- | --- | --- |
| lstm | 52,870 | 0.5151 | 6/14 | 2.4 |
| transformer | 69,510 | 0.5148 | 6/14 | 4.8 |

### Validation preferred the sequence model; test did not

| Model | Validation macro-F1 | Test macro-F1 | Change |
| --- | --- | --- | --- |
| lstm (selected) | 0.5151 | 0.5074 | -0.0077 |
| RandomForest (Phase 3) | 0.4895 | 0.5173 | +0.0277 |

On validation the lstm leads the forest by +0.0255. On test it trails by
-0.0099. **The ranking reverses**, and the protocol was followed
correctly — selection never touched test.

This is the validation caveat from `reports/preprocessing_summary_h6.md`
arriving in practice. The chronological split leaves Very unhealthy at 7.04% of
validation and 12.14% of test, so the two splits are not interchangeable yardsticks for
a macro-averaged metric: the forest *gains* +0.0277 moving to test while
the lstm *loses* -0.0077. Selecting on validation remains the right
protocol — the alternative is worse — but a validation win at this margin is not
evidence of a test win, and the write-up should not present it as one.


Selected: **lstm** on validation macro-F1
0.5151. No class weighting, matching the tabular baselines —
the comparison is only meaningful if both sides face the imbalance on the same terms.

### Learning-rate robustness

The first run of this phase used `lr: 1e-3` and both architectures peaked at epoch 2–4
and then declined while training loss kept falling. That pattern is ambiguous — it can
mean the task is hard, or that the step size is too large — so the rate was swept before
any conclusion was drawn. A negative result from an undertrained model is not a result.

Validation macro-F1 (observed rows), 25 epoch ceiling, patience
6:

| Architecture | lr=0.003 | lr=0.001 | lr=0.0003 | lr=0.0001 |
| --- | --- | --- | --- | --- |
| lstm | 0.4962 | 0.4948 | **0.5151** | 0.5024 |
| transformer | 0.4986 | 0.5026 | **0.5148** | 0.5127 |

Chosen: `lstm` at 0.0003, `transformer` at 0.0003. `configs/default.yaml`
now sets `model.lr: 0.0003`. The sweep reads validation only, so the test split is
still untouched by any of these decisions. Reproduce with
`python -m src.models.dl_forecast --lr-sweep`.


---

## 2. Test results (observed labels only)

| Model | Accuracy | Macro-F1 | V.unhealthy F1 | Hazardous F1 |
| --- | --- | --- | --- | --- |
| **lstm (MC mean)** | 0.5734 | **0.5074** | 0.5265 | 0.5956 |
| lstm (deterministic) | 0.5724 | **0.5082** | 0.5270 | 0.5953 |
| transformer (MC mean) | 0.5752 | **0.5094** | 0.5252 | 0.5942 |
| RandomForest (Phase 3) | 0.5793 | **0.5173** | 0.5265 | 0.6098 |
| Persistence | 0.5356 | **0.5118** | 0.5104 | 0.5897 |

### Per-class F1

| Class | lstm (MC mean) | lstm (deterministic) | transformer (MC mean) | RandomForest (Phase 3) | Persistence |
| --- | --- | --- | --- | --- | --- |
| Good | 0.6105 | 0.6103 | 0.6080 | 0.5981 | 0.5888 |
| Moderate | 0.5249 | 0.5229 | 0.5272 | 0.5440 | 0.4963 |
| Unhealthy (sensitive) | 0.1036 | 0.1107 | 0.1148 | 0.1432 | 0.2571 |
| Unhealthy | 0.6834 | 0.6828 | 0.6869 | 0.6820 | 0.6286 |
| **Very unhealthy** | 0.5265 | 0.5270 | 0.5252 | 0.5265 | 0.5104 |
| **Hazardous** | 0.5956 | 0.5953 | 0.5942 | 0.6098 | 0.5897 |

---

## 3. Uncertainty and calibration

MC dropout keeps dropout active at inference and averages 50 passes. The
confidence below is the maximum of that averaged class distribution. Total predictive
entropy is split into the part inherent to the data and the part that is the model
disagreeing with itself across passes — only the second shrinks with more data.

| Architecture | Mean confidence | Mean predictive entropy | Mean epistemic (MI) |
| --- | --- | --- | --- |
| lstm | 0.5706 | 1.0713 | 0.0178 |
| transformer | 0.5802 | 1.0462 | 0.0340 |

### Reliability — lstm, 61,530 observed test predictions

A calibrated model's accuracy inside a confidence bin matches the bin's mean confidence.
Gap is accuracy minus confidence; negative is overconfident.

| Confidence bin | n | Mean confidence | Accuracy | Gap |
| --- | --- | --- | --- | --- |
| 0.2–0.3 | 462 | 0.2761 | 0.2597 | -0.0164 |
| 0.3–0.4 | 6,971 | 0.3593 | 0.3966 | +0.0373 |
| 0.4–0.5 | 13,021 | 0.4556 | 0.4679 | +0.0123 |
| 0.5–0.6 | 15,559 | 0.5494 | 0.5655 | +0.0161 |
| 0.6–0.7 | 13,758 | 0.6465 | 0.6381 | -0.0084 |
| 0.7–0.8 | 8,185 | 0.7458 | 0.7126 | -0.0331 |
| 0.8–0.9 | 3,345 | 0.8344 | 0.8054 | -0.0291 |
| 0.9–1.0 | 229 | 0.9138 | 0.8690 | -0.0448 |

ECE **0.0190**, MCE **0.0448**. No bin is overconfident by more than 5 points: where the model says 0.7 it is right about 70% of the time, which is what the advisory layer needs if it is going to gate on confidence.

Of the 1.0713 nats of mean predictive entropy, only 0.0178 (1.7%) is epistemic — the part that comes from the model disagreeing with itself across passes. The remaining ~98% is aleatoric: irreducible overlap between classes given these nine channels and this horizon. **That is a statement about the task, not the model.** More capacity, more epochs, or more data move the epistemic sliver and leave the rest alone, which is consistent with every result in this phase and the last two. If the six-hour-ahead category is to be predicted better, the input has to change — more channels, a longer window, or spatial context from neighbouring stations — not the architecture.

---

## 4. Paired bootstrap (1,000 resamples, 95% CI)

Same procedure as the Phase 4 ablation: both models scored on the same resampled rows
each iteration, so the comparison stays paired.

| Baseline | Metric | Δ (DL − baseline) | 95% CI | p (two-sided) | Significant |
| --- | --- | --- | --- | --- | --- |
| RandomForest (Phase 3) | Macro-F1 | -0.0099 | [-0.0139, -0.0061] | 0.000 | **yes** |
| RandomForest (Phase 3) | Very unhealthy F1 | -0.0001 | [-0.0074, +0.0073] | 0.988 | no |
| RandomForest (Phase 3) | Hazardous F1 | -0.0143 | [-0.0226, -0.0059] | 0.004 | **yes** |
| Persistence | Macro-F1 | -0.0044 | [-0.0087, -0.0002] | 0.038 | **yes** |
| Persistence | Very unhealthy F1 | +0.0161 | [+0.0064, +0.0256] | 0.002 | **yes** |
| Persistence | Hazardous F1 | +0.0059 | [-0.0052, +0.0177] | 0.286 | no |

---

## 5. Does the complexity earn its place?

**No — and it is disqualified on the same rule Phase 4 used.** The selected lstm significantly degrades Hazardous -0.0143 [-0.0226, -0.0059] against the RandomForest. Whatever it does to macro-F1, a model that is worse at the classes a wearable advisory exists to raise is not the model to ship.

Against the zero-parameter persistence rule the lstm is -0.0044 [-0.0087, -0.0002], which does **not** clear the floor — the headline number should be read with that in mind.

For the record: lstm 0.5074, RandomForest 0.5173, persistence 0.5118. The forest trains in ~15 s on CPU and needs no MC passes; the lstm took 2 min on GPU plus 50 inference passes per prediction. On an ESP32 that gap is the whole argument, and it has to be paid for with accuracy that is not here.

**What MC dropout does buy**, independent of the accuracy question, is a calibrated confidence per prediction — which is what Phase 6's advisory layer needs to say "I am not sure" instead of guessing. A random forest's vote share is not a substitute. If the sequence model is kept, keep it for the uncertainty, and say so rather than implying an accuracy win that the bootstrap does not support.

Reproduce with `python -m src.models.dl_forecast`.
