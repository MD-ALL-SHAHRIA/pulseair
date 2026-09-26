# pulseair-ml

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22979304.svg)](https://doi.org/10.5281/zenodo.22979304)
[![tests](https://github.com/MD-ALL-SHAHRIA/pulseair/actions/workflows/tests.yml/badge.svg)](https://github.com/MD-ALL-SHAHRIA/pulseair/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code of Conduct](https://img.shields.io/badge/code%20of%20conduct-contributor%20covenant-ff69b4.svg)](CODE_OF_CONDUCT.md)

Research code for an undergraduate thesis on **wearable air-quality forecasting**: a
GAN-augmented deep-learning pipeline with uncertainty quantification and an LLM-based
explainable risk advisory, built for **PulseAir**, an ESP32 air-quality neckband. Everything
here happens off-device. The project is **rigor-first**: every headline number is reported
next to a zero-parameter persistence floor, every intervention is tested with a paired
bootstrap and rolling-origin cross-validation, and interventions that improve an aggregate
metric while degrading the safety-critical classes are disqualified rather than headlined.
That discipline is what the project mainly produced. The **Beijing** multi-site dataset was
the methodology bench — and on it, no model beats persistence in more than 2 of 5
rolling-origin folds, so **nothing from Beijing is deployed**. The **Bangladesh** data, the
wearable's actual target population, produced the deployment candidate, and a ground-truth
model against the US Embassy Dhaka reference monitor supplies the one validated
advisory-class result in the project.

---

## Headline results

All Beijing numbers are 6-hour-ahead PM2.5 AQI-category macro-F1 on observed (non-imputed)
test rows. Confidence intervals are 95% paired bootstrap, 1,000 resamples.

**The floor.** A zero-parameter rule — "the category in 6 hours is the category now" —
scores **0.5118** macro-F1. Quote it next to every number below; without it a macro-F1 does
not say whether the model is working.

| Variant | Macro-F1 | Hazardous F1 | V. unhealthy F1 | Beats persistence? |
|---|---|---|---|---|
| **Persistence (zero-parameter)** | **0.5118** | 0.5897 | 0.5104 | — *(reference)* |
| RandomForest (Phase 3) | 0.5173 | 0.6098 | 0.5265 | yes on test (+0.0055 [+0.0016, +0.0093]) — **loses on validation** |
| XGBoost (Phase 3) | 0.4991 | 0.5607 | 0.5122 | worse (−0.0127 [−0.0174, −0.0082]) |
| RF + CTGAN, broad 4-class | 0.5212 | 0.5908 | 0.4973 | yes (+0.0094) — **disqualified**, degrades V. unhealthy |
| RF + CTGAN, targeted 2-class | 0.5043 | 0.5843 | 0.4903 | worse (−0.0075) |
| RF + SMOTE | 0.5207 | 0.5865 | 0.4757 | yes (+0.0089) — **disqualified**, same failure, 660× faster |
| RF, `class_weight='balanced'` | 0.5149 | 0.5843 | 0.4733 | no (+0.0031 [−0.0008, +0.0071]) |
| LSTM + MC dropout | 0.5074 | 0.5956 | 0.5265 | worse (−0.0044 [−0.0087, −0.0002]) |
| Transformer + MC dropout | 0.5094 | 0.5942 | 0.5252 | not tested |

**Rolling-origin CV settles it.** Five chronological expanding-window folds with embargo,
per-fold scaler refit, Wilcoxon signed-rank on the fold deltas: **no Beijing model wins more
than 2 of 5 folds.** Persistence's own fold-to-fold spread (0.0802) is larger than any
model–persistence difference. The single-split "RandomForest beats persistence" result does
not survive.

**Bangladesh — the deployment candidate.** 7 shared channels (PM2.5, PM10, CO + 4 cyclical),
four cities, 2022-08-05 onward:

| | |
|---|---|
| Model | RandomForest, `class_weight='balanced'`, 25 trees × depth 12 |
| Rolling-origin CV | **5 of 5 folds beat persistence**, mean Δ **+0.0354**, one-sided *p* = 0.0312 |
| Compression | 8.6× (8.2 MB → 983 KB); 1,673 KB ONNX |
| Inference | **0.0074 ms** single sample, single-threaded; conformal layer 0.0103 ms |
| Conformal coverage | 0.9098 overall (Mondrian, 90% target) |
| **Scope of the claim** | the **four common classes only** — the eval blocks contain 2 Hazardous and 482 Very-unhealthy samples in total |

**Phase 11b — the advisory classes, validated.** A separate PM2.5-only model trained on the
**US Embassy Dhaka reference monitor** (75,344 QC-passed hours, 3,788 genuinely Hazardous):

| | Model | Persistence floor | Folds won | *p* (two-sided) |
|---|---|---|---|---|
| **Hazardous F1** | **0.4451** | 0.3167 | **7 / 7** | **0.0156** |
| Macro-F1 | 0.4603 | 0.4133 | 7 / 7 | 0.0156 |

This is evidence about the **task**, from a **different model** than the deployed one. The
deployed 7-channel Bangladesh model still carries **no advisory-class validation**, and
nothing above should be read as giving it one.

Full derivations, per-class tables and negative results: **[`reports/final_results_summary.md`](reports/final_results_summary.md)**.

---

## Key findings

Written for a reader who will never open the thesis.

- **The persistence floor reframed the project.** At a 1-hour horizon the AQI category is
  unchanged 80.5% of the time, so a zero-parameter rule scores 0.7933 and RandomForest
  clears it by 0.006 — the task was nearly trivial and the metric hid it. Moving to 6 hours
  drops the floor to 0.5118 and exposes how little any model adds. **A macro-F1 without its
  persistence floor is uninterpretable**, and most of this project's negative results are
  only visible once the floor is on the same page.

- **CTGAN's quality score did not measure validity.** The first CTGAN run scored 0.90+ on
  SDV's column-shape and pair-trend report while **51% of synthetic rows had dew point above
  temperature** — physically impossible air. The dict-form constraint SDV accepted was
  silently ignored; only the `sdv.cag.Inequality` object enforced it, taking violations to
  0%. A distributional quality score is not a validity check, and the two can disagree
  completely.

- **Every augmentation intervention failed the same way.** CTGAN (22.1 min) and SMOTE (2.0 s)
  both raised aggregate macro-F1 *and* significantly degraded both advisory classes. That is
  the exact pattern the disqualification rule exists to catch: a headline improvement bought
  by getting worse at the rare, dangerous cases the device exists to warn about. The only
  intervention that did not degrade them was plain class weighting.

- **Capacity was never the binding constraint.** Both sequence architectures got *monotonically
  worse* with more parameters; compressing the forest 315× cost 0.0156 macro-F1. MC-dropout
  decomposition explains it: **98.3% of predictive entropy is aleatoric** — irreducible noise
  in the data, not model error. No amount of model is going to fix that.

- **Single splits disagree with themselves.** Validation and test picked different winners
  repeatedly. Rolling-origin CV over 5 chronological folds resolved it: no Beijing model
  wins more than 2 of 5. Select on validation, report on test, and trust neither until
  multiple folds agree.

- **Marginal conformal coverage hid a per-class failure.** Split conformal hit its 90%
  marginal target while covering Hazardous only **84.3%** of the time — the slack lands
  exactly on the rare classes. Mondrian (class-conditional) calibration fixed it (0.908).

- **A published dataset's advertised 25-year history is 87% backfill.** The Mendeley
  Bangladesh AQI dataset (`9j447cynb9`) sells 2000–2025; everything before 2022-08-05
  carries a near-linear synthetic trend (R² = 0.992), a hard clip at exactly 250.0 µg/m³,
  and CO in inconsistent units. Dropping it costs **19% of the rows but 87% of the
  years** — the discarded portion is Dhaka alone at low density, while the usable window
  is 30 cities hourly. Checking the survivors against the US Embassy reference monitor
  over the overlap: **17 Hazardous hours in the reanalysis against 1,602 in the
  instrument.** The audit is recomputed from the raw file by `audit()`, not asserted —
  see [`reports/bangladesh_validation.md`](reports/bangladesh_validation.md).

- **ONNX on the ESP32 is a portability proxy, not a flashability claim.** There is no ONNX
  Runtime for the ESP32. A real port needs a C tree traversal, `emlearn`, or a different
  framework. The benchmark measures what it measures.

---

## Repository structure

```
pulseair-ml/
├── src/
│   ├── preprocessing/   # Beijing pipeline, Bangladesh adapter, OpenAQ survey, downloads
│   ├── models/          # baselines, LSTM/Transformer, conformal, rolling CV, Dhaka models
│   ├── gan/             # CTGAN augmentation, SMOTE control, paired-bootstrap ablation
│   ├── explainability/  # SHAP attribution + Gemini risk advisory
│   ├── deployment/      # compression sweep, ONNX export, latency benchmarks
│   └── reporting/       # results compilation into reports/final_results_summary.md
├── pulsebench/          # standalone evaluation toolkit (see below) — no src/ imports
├── reports/             # 26 generated markdown reports + metrics/*.json + figures/
├── data/                # raw / interim / processed / external — all gitignored
├── configs/default.yaml # every hyperparameter and threshold in the project
├── notebooks/           # exploratory analysis
└── tests/
```

**Data flows one way:** `data/raw → src/preprocessing → data/processed → src/models →
reports/`. Everything under `data/` and all model binaries (`*.pkl`, `*.pt`, `*.onnx`) are
gitignored. The markdown reports, the metrics JSON and the thesis figures **are** tracked,
so every claim is checkable and every figure is rebuildable without re-running a phase.

---

## Setup

Requires **Python 3.11**.

```bash
git clone https://github.com/MD-ALL-SHAHRIA/pulseair.git
cd pulseair
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` holds intended version ranges; `requirements.lock.txt` is the exact
resolved set (200 pins) used to produce the committed results.

Verify the stack:

```bash
python -c "import torch, sdv, shap, onnxruntime, xgboost; print('ok')"
```

### API keys

Copy `.env.example` to `.env` and fill in the keys you need. **`.env` is gitignored and
must stay that way.**

```bash
cp .env.example .env
```

| Variable | Needed for | Required? |
|---|---|---|
| `GEMINI_API_KEY` | `src.explainability.llm_advisory` — live advisory generation | optional; falls back to a rule-based template |
| `OPENAQ_API_KEY` | `src.preprocessing.openaq_survey` — Phase 11c station survey | optional; only for re-running the survey |
| `ANTHROPIC_API_KEY` | legacy, pre-Gemini advisory path | not used by current code |

Without `GEMINI_API_KEY` the advisory layer degrades to the rule-based template and labels
its own output as such — it never fails silently.

### Environment notes

- **numpy is pinned `<2.0`** — parts of the SDV stack are still built against numpy 1.x.
- **`onnxscript` is required** by `torch.onnx.export` from torch 2.9 onward; not optional.
- **`xgboost` needs OpenMP on macOS.** `pip install xgboost` succeeds but importing it fails
  with `Library not loaded: @rpath/libomp.dylib` until `brew install libomp`.
- Torch **MPS** works for training, but MPS cannot run fused attention with dropout under
  `no_grad` — MC-dropout inference probes for this and falls back to CPU automatically.

---

## Data

Nothing under `data/` is committed. Three sources:

**1. UCI Beijing Multi-Site Air Quality** ([id=501](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)) — 420,768 hourly rows, 12 stations, 2013-03-01 .. 2017-02-28. Fetched automatically:

```bash
python -m src.preprocessing.download     # -> data/raw/beijing_multisite.csv
```

> `fetch_ucirepo(id=501)` does not work for this dataset — UCI reports it as *"exists in the
> repository, but is not available for import"* because it ships as a nested zip of 12
> per-station CSVs. `download.py` pulls the archive directly and asserts the 420,768-row /
> 12-station shape before saving.

**2. Bangladesh AQI (Mendeley `9j447cynb9` v2)** — manual download to
`data/raw/bangladesh_aqi.csv`. **Read [`reports/bangladesh_validation.md`](reports/bangladesh_validation.md)
before using it**; only the portion from 2022-08-05 onward survives the integrity audit.

**3. US Embassy Dhaka reference monitor** — AirNow Embassy historical files from
`files.airnowtech.org`, yearly CSVs placed in `data/raw/dhaka_embassy/{2016..2025}.csv`.

---

## Reproducing each phase

Run from the repo root with the venv active. Phase ordering matters — later phases read
earlier phases' artifacts.

```bash
# Phase 2 — preprocessing (all four horizons; h6 is primary)
python -m src.preprocessing.pipeline --all-horizons     # -> data/processed/h{1,6,12,24}/
python -m pytest src/preprocessing/test_pipeline.py -q  # 33 tests: shapes, leakage, scaler fit

# Phase 3 — tabular baselines + horizon sweep
python -m src.models.baseline                           # -> src/models/artifacts/baseline_h6.pkl
python -m src.models.baseline --horizon-sweep           # -> reports/horizon_comparison.md

# Phase 4 — CTGAN augmentation and its ablation
python -m src.gan.augment --plan-only                   # class balance + plan, no training
python -m src.gan.augment                               # ~2.5 h: 4 synthesizers x 300 epochs
python -m src.models.baseline --augmented               # retrain on the augmented table
python -m src.gan.ablation                              # -> reports/gan_ablation_h6.md
python -m src.gan.smote_control                         # SMOTE control (2.0 s)
python -m src.gan.validity_before_fix                   # figure-only: re-measures the
                                                        # original broken config (~20 min)

# Phase 5 — sequence models with MC dropout
python -m src.models.dl_forecast                        # -> reports/dl_metrics_h6.md
python -m src.models.capacity_sweep                     # -> reports/capacity_sweep_h6.md

# Phase 6 — conformal prediction, SHAP, LLM advisory
python -m src.models.conformal                          # -> reports/conformal_h6.md
python -m src.explainability.shap_analysis              # -> reports/shap_examples/
python -m src.explainability.llm_advisory               # -> reports/llm_advisory_examples.md

# Phase 7 — compression, ONNX export, latency
python -m src.deployment.compress_export                # -> reports/deployment_report_h6.md
python -m src.deployment.compress_export --class-weight # class-weighted variant

# Phase 8/9 — rolling-origin CV, sensitivity, summary
python -m src.models.rolling_cv                         # -> reports/rolling_origin_cv_h6.md
python -m src.models.hj633_sensitivity                  # EPA vs HJ 633-2012 breakpoints
python -m src.reporting.compile_results                 # -> reports/final_results_summary.md
python -m src.reporting.generate_figures                # -> reports/figures/ (23 thesis figures)
python -m src.reporting.build_thesis                    # -> docs/PulseAir_Thesis.docx

# Phase 10 — Bangladesh external validation + deployment
python -m src.preprocessing.bangladesh                  # -> data/processed/bd_h6/
python -m src.models.bangladesh_validation              # -> reports/bangladesh_validation.md
python -m src.models.rolling_cv --dataset bangladesh    # -> reports/bangladesh_rolling_cv.md
python -m src.deployment.compress_export --dataset bangladesh

# Phase 11 — Dhaka ground truth and the PM2.5-only advisory-class model
python -m src.models.dhaka_ground_truth                 # -> reports/dhaka_ground_truth_validation.md
python -m src.models.dhaka_pm25_model                   # -> reports/dhaka_ground_truth_model_h6.md
python -m src.preprocessing.openaq_survey               # -> reports/openaq_multichannel_validation.md
```

**Two standing rules in this codebase.** The primary horizon is **6 hours**
(`preprocessing.horizon` in `configs/default.yaml`); h1/h12/h24 exist for the comparison
figure only. And model selection is on the **validation** split, with test reserved for
final reporting — even when both splits would pick the same model, because knowing they
agree requires looking, which is the thing being avoided.

---

## `pulsebench` — the evaluation protocol, extracted

The methodology outlasts the model it produced, so it is packaged separately in
**[`pulsebench/`](pulsebench/README.md)**. It depends on nothing in `src/`, needs only the
numeric stack, and works on any dataframe with a datetime axis and a categorical target.
**You can install it on its own, without the rest of this repository:**

```bash
pip install "git+https://github.com/MD-ALL-SHAHRIA/pulseair.git"
```

```python
from pulsebench import (
    persistence_floor,          # the zero-parameter baseline, time-joined not row-shifted
    rolling_origin_cv,          # expanding-window folds, embargo, per-fold scaling
    advisory_disqualification,  # reject gains bought by degrading a protected class
    bonferroni_report,          # family-wise correction with the resampling floor stated
)

floor = persistence_floor(df, target_col="aqi_class", horizon=6, time_col="datetime")
result = rolling_origin_cv(df, model_fn, n_folds=5, embargo_hours=24, ...)
```

```bash
python -m pytest pulsebench/tests/ -q   # 25 synthetic tests + a Phase 9 regression pin
python -m doctest pulsebench/*.py -v | tail -1         # 28 doctests
```

The retrofit is load-bearing, not decorative: Phase 9's Beijing rolling-origin CV calls
`pulsebench` directly, and `test_retrofit_phase9.py` pins the pre-extraction numbers so the
refactor is provably behaviour-preserving.

---

## Reports

All 26 generated reports live in [`reports/`](reports/) and are committed, so every result
is readable without running anything. Start with:

| Report | What's in it |
|---|---|
| [`final_results_summary.md`](reports/final_results_summary.md) | master comparison, all negative results, deployed system, paper structure, reviewer questions |
| [`horizon_comparison.md`](reports/horizon_comparison.md) | the persistence floor at h=1/6/12/24 |
| [`gan_ablation_h6.md`](reports/gan_ablation_h6.md) | CTGAN ablation with paired bootstrap |
| [`rolling_origin_cv_h6.md`](reports/rolling_origin_cv_h6.md) | the 5-fold Beijing result |
| [`bangladesh_validation.md`](reports/bangladesh_validation.md) | the dataset integrity audit |
| [`dhaka_ground_truth_model_h6.md`](reports/dhaka_ground_truth_model_h6.md) | the validated advisory-class result |
| [`deployment_report_h6.md`](reports/deployment_report_h6.md) | compression sweep, ONNX, latency |
| [`figures/`](reports/figures/README.md) | 23 publication figures, each with the JSON it was built from |

---

## Citation

Releases from v1.0.1 onward are archived on Zenodo. Cite the **concept DOI**
[`10.5281/zenodo.22979304`](https://doi.org/10.5281/zenodo.22979304), which always
resolves to the latest archived version:

> Shahria, M. A., Mithila, S. D., Rudro, A. S., & Payel, I. I. PulseAir: rigor-first
> evaluation of wearable air-quality forecasting. Zenodo. 10.5281/zenodo.22979304

To cite the exact version you ran, use that release's own DOI from the
[Zenodo record](https://doi.org/10.5281/zenodo.22979304) instead — the concept DOI
moves, a version DOI does not.

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff) (project) or
[`pulsebench/CITATION.cff`](pulsebench/CITATION.cff) (the toolkit alone); GitHub renders
a ready-made citation from these under *"Cite this repository"*. Note that the
`CITATION.cff` files shipped in **v1.0.0 through v1.0.2 fail CFF schema validation** — a
required field was missing from every `references` entry. Use v1.0.3 or later if a
reference manager needs to parse them.

---

## Contributing

This repository has two halves with different expectations. **`src/` is thesis code** —
it reproduces a fixed set of published results, so changes that alter those numbers need
discussion before they need a patch. **`pulsebench/` is built for reuse, and that's where
contributions are most welcome.**

Good places to start:

- **[Open issues tagged `good first issue`](https://github.com/MD-ALL-SHAHRIA/pulseair/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)** — scoped, self-contained, and each one says what "done" looks like.
- **Run `pulsebench` on your own data and tell us what broke.** It has been exercised on
  two air-quality datasets. Any seasonal, imbalanced, categorical time series is fair
  game, and the failure modes we haven't seen yet are the most useful thing you can bring.
- **Challenge a result.** Every claim in `reports/` is checkable from the committed
  metrics JSON, and several are negative findings that would be easy to get wrong. There's
  a [dedicated issue template](.github/ISSUE_TEMPLATE/methodology.yml) for exactly this.
- **Questions and open-ended ideas** belong in
  [Discussions](https://github.com/MD-ALL-SHAHRIA/pulseair/discussions).

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first — particularly the evaluation conventions,
which are the house rules the whole codebase rests on. Also:
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), [`SECURITY.md`](SECURITY.md).

---

## License

MIT — see [`LICENSE`](LICENSE). © 2026 Md All Shahria, Sanjeda Dewan Mithila,
Anik Sarker Rudro, Irfanul Islam Payel.
