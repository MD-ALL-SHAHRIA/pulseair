# Claim ledger

An audit of every major claim in `docs/PulseAir_Thesis.docx`, chapter by chapter,
against the committed evidence. Built for review before any thesis edit. The last column
records the fix each flagged claim needed; the one overclaim found (claim 17) has since
been applied to the thesis, as noted there.

Status key: **SUPPORTED** — the evidence backs the claim as stated;
**PARTIALLY SUPPORTED** — true but with a material caveat the thesis already states or
should; **OVERCLAIMED** — the wording says more than the evidence shows and needs
softening.

Evidence sources are the committed `reports/metrics/*.json` and `reports/*.md` the
thesis builds from; every number is recomputed at build time, not transcribed.

| # | Claim | Evidence source | Statistical support | Limitation | Location | Status | Fix |
|---|-------|-----------------|---------------------|------------|----------|--------|--------------------------------|
| 1 | At h=1 the AQI category is unchanged ~80% of the time and persistence scores ~0.79 macro-F1; at h=6 the floor drops to ~0.51 | `horizon_comparison.json` | Descriptive (label-unchanged %, macro-F1) | None material | §1.2, §4.1, Fig 1 | SUPPORTED | — |
| 2 | Persistence must be reported beside every headline number (the central methodological claim) | Whole evaluation design | Framing, not a test | It is a discipline, not a new technique (see novelty audit) | §1.2, §4.1 | SUPPORTED | — |
| 3 | A RandomForest reaches ~0.53 macro-F1 on the Beijing test split | `ablation_h6.json` | Point estimate on fixed split | Single split; §4.2 is the honest re-test | §4.3, Fig 2 | SUPPORTED | — |
| 4 | No Beijing-trained model beats persistence in a majority of rolling-origin folds | `rolling_cv_h6.json`, `rolling_cv_h6_f8.json` | 5-fold best ≤2/5; 8-fold best 4/8, XGBoost significantly worse (p=0.0078) | Wilcoxon assumes fold independence (now caveated §3.10); folds-won is the primary statistic | §4.2, §4.15.2, Figs 13,24 | SUPPORTED | — (the negative result is *strengthened* by the fold-independence caveat) |
| 5 | CTGAN and SMOTE raise macro-F1 while significantly degrading both advisory classes, and are disqualified | `ablation_h6.json`, `gan_h6.json`, `smote_h6.json` | Paired bootstrap 95% CI on per-class deltas | Disqualification threshold is a stated design choice | §4.3, Fig 3 | SUPPORTED | — |
| 6 | An SDV quality score >0.89 co-exists with physically impossible synthetic records | `gan_h6.json`, `gan_validity_before_fix.json` | Concrete counterexamples (cyclical + temp/dew-point) | One synthesiser, one config | §4.3, contributions | SUPPORTED | — |
| 7 | Added capacity makes both sequence architectures monotonically worse | `capacity_sweep_h6.json` | Monotone trend across hidden sizes | Two architectures; one dataset | §4.4, Fig 6 | SUPPORTED | — |
| 8 | ~98% of predictive entropy is aleatoric, not epistemic | `dl_h6.json` (MC dropout) | Entropy decomposition | **MC dropout is an approximation**; a deep ensemble was not run | §4.4, §5.2, Fig 7 | PARTIALLY SUPPORTED | None needed — §6.4 already states dropout is an approximation and the conclusion is "well-supported rather than settled" |
| 9 | Split conformal under-covers the rare classes; Mondrian restores per-class coverage | `conformal_h6.json` | Measured empirical coverage | Exchangeability strained by chronological split (stated §6.4) | §4.5, Figs 8,9 | SUPPORTED | — |
| 10 | A compressed forest exports to ONNX with measured latency feasible for the target class of hardware | `deployment_h6*.json` | Measured sizes/latency | Not measured on ESP32 silicon (stated §6.5, §8.1) | §4.6, §4.13, Fig 10 | SUPPORTED | — |
| 11 | The conclusion is unchanged under HJ 633-2012 vs EPA breakpoints | `hj633_h6.json` | Re-run under both schemes | — | §4.7, Fig 11 | SUPPORTED | — |
| 12 | Holm-Bonferroni changes no verdict vs Bonferroni | computed from `rolling_cv_h6.json` p-values | Step-down vs single-step, bimodal p-values | Qualitative (no separate JSON) | §4.8, §4.15.1, Fig 12 | SUPPORTED | — |
| 13 | The published Bangladesh dataset is substantially fabricated over its advertised span | `bangladesh_h6.json`, `integrity_audit_bangladesh.json`, `dhaka_monthly_crosscheck.json` | 5 structural signatures + independent DoE and Embassy cross-checks | Audit is of one dataset | §4.9, §4.15.6 | SUPPORTED | — |
| 14 | Discarding the fabricated portion costs ~19% of rows but ~87% of the advertised span | `bangladesh_h6.json:audit` | Recomputed from raw file | — | §4.9 | SUPPORTED | — |
| 15 | On Bangladesh a class-weighted forest beats persistence in 5 of 5 folds (p=0.0312) | `rolling_cv_h6_bangladesh.json` | Folds-won 5/5; Wilcoxon one-sided p=0.0312 | **p-value anti-conservative** (fold independence); folds-won is primary; evaluation blocks too sparse to cover advisory classes | §4.9, Figs 14,15 | SUPPORTED | — (fold-independence caveat now added §3.10, §4.9, §6.4) |
| 16 | The reanalysis records far fewer Hazardous hours than the reference monitor | `dhaka_ground_truth.json` | Direct count comparison | Single station | §4.10, Fig 18 | SUPPORTED | — |
| 17 | A PM2.5-only model "closes the gap", detecting Hazardous above its floor in 7 of 7 folds (p=0.0156) | `dhaka_pm25_model_h6.json` | Folds-won 7/7; Wilcoxon two-sided p=0.0156 (at resolution floor) | **Different model** (PM2.5-only, single-station) than the deployed multi-channel one; p anti-conservative | §4.11, Fig 19 | SUPPORTED (fix applied) | **APPLIED:** "closes the gap" → "narrows the gap" in the Abstract and §1.3. (The unrelated "closes that gap" in §4.6, about the compression criterion, and the forward-looking "closing the gap" in §8.1 are correct and were left as-is.) |
| 18 | No OpenAQ station near Dhaka meets multi-pollutant coverage; DoE operates 31 stations but none are on open platforms | `openaq_survey.json`, NAQMP 2024-2030 | Descriptive survey + primary-source counts | Access, not existence, is the barrier (stated) | §4.12, Fig 20 | SUPPORTED | — |
| 19 | The deployed multi-channel model's advisory classes remain unvalidated on ground truth | design + §4.9–4.12 | Stated as a limitation, not a positive claim | This is itself the honest limitation | §4.13, §6.1 | SUPPORTED | — |
| 20 | Leave-one-station-out: the forest beats each station's own floor 12/12 | `station_holdout_h6.json` | 12/12, mean +0.043, p=0.0005 | **Folds overlap in time** — easier than forecasting the future (stated §4.15.3, §6.3) | §4.15.3, Fig 25 | PARTIALLY SUPPORTED | None needed — the temporal-overlap caveat is already stated at both the result and in Limitations |
| 21 | The Hazardous detector survives low-cost-sensor noise, losing ~28% F1 but still beating its noisy floor 7/7 | `sensor_noise_robustness_h6.json` | 7/7 vs noisy floor; noise sized from published R² | **Simulation of noise, not a device error model** (stated §6.4) | §4.15.4, Fig 26 | SUPPORTED | — |
| 22 | Selective prediction raises accuracy on the confident subset but not macro-F1 | `selective_prediction_h6.json` | Full vs confident comparison | Abstains on the safety-critical tail; report-only (stated §6.4) | §4.15.5, Fig 27 | SUPPORTED | — |
| 23 | PulseBench packages the protocol as a reusable, dataset-agnostic library | `pulsebench/` + tests | Software artifact; synthetic-data tests | Components are standard practice; packaging is the contribution (see novelty audit) | §3.12, contributions | SUPPORTED | — |
| 24 | The LLM advisory layer's honesty constraints are enforced outside the model | `llm_advisory_examples.md`, `shap_h6.json` | Worked examples | 5 live advisories; free-tier model | §4.13, §4.14 | SUPPORTED | — |

## Summary

- **23 of 24** major claims are SUPPORTED or PARTIALLY SUPPORTED with a caveat the thesis
  already states.
- **One OVERCLAIMED item was found (wording only) and has since been fixed:** claim 17's
  phrase **"closes the gap"** (used in the Abstract and in the §1.3 contributions list)
  overstates what a PM2.5-only, single-station model does for the *deployed*
  multi-channel model. §4.11 already says in the body that it "is not evidence about the
  deployed model", so the abstract/contributions wording contradicts the thesis's own
  careful framing. **Fix applied:** "closes the/that gap" → "narrows the gap" in both
  places (Abstract and §1.3); no number changes. The unrelated "closes that gap" in §4.6
  (compression criterion) and "closing the gap" in §8.1 (future work) are correct uses
  and were left unchanged.
- The two rolling-origin significance claims (15, 17) and the station-holdout claim (20)
  are the places where a p-value or a generalisation could be read as stronger than it
  is; all three now carry the correct caveat in the thesis (fold independence for 15/17;
  temporal overlap for 20), so their status is SUPPORTED / PARTIALLY SUPPORTED rather
  than OVERCLAIMED.
- **External-validation scope is not overclaimed:** the thesis states plainly that the
  deployed model's advisory classes are unvalidated (claim 19) and that multi-pollutant
  validation in Dhaka is currently impossible (claim 18). That honest limitation stands
  and is not inflated anywhere.
