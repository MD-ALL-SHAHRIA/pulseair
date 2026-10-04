# Defense slides: number -> source

Every number on docs/PulseAir_Defense.pptx is read from these files by `src/reporting/build_defense.py`; figures come from `reports/figures/`.

## Slide 3 Why this matters

| Value | Number | Source |
|---|---|---|
| `doe_stations` | 31 | docs/reference/NAQMP_2024-2030.pdf §3.3.2 (quoted in thesis §4.12) |
| `doe_cams` | 16 | docs/reference/NAQMP_2024-2030.pdf §3.3.2 (quoted in thesis §4.12) |
| `doe_cms` | 15 | docs/reference/NAQMP_2024-2030.pdf §3.3.2 (quoted in thesis §4.12) |
| `doe_cities` | 13 | docs/reference/NAQMP_2024-2030.pdf §3.3.2 (quoted in thesis §4.12) |
| `oq_n` | 22 | reports/metrics/openaq_survey.json |
| `oq_radius` | 25 | reports/metrics/openaq_survey.json |
| `oq_pass` | 0 | reports/metrics/openaq_survey.json |

## Slide 4 The problem

| Value | Number | Source |
|---|---|---|
| `none_macro` | 0.5173 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `ctgan_macro` | 0.5212 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `none_haz` | 0.6098 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `ctgan_haz` | 0.5908 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |

## Slide 7 Datasets

| Value | Number | Source |
|---|---|---|
| `clean_rows` | 849,831 | reports/metrics/bangladesh_h6.json |
| `actual_cities` | 30 | reports/metrics/bangladesh_h6.json |
| `gt_qc_hours` | 75,344 | reports/metrics/dhaka_ground_truth.json |
| `oq_n` | 22 | reports/metrics/openaq_survey.json |
| `beijing_rows` | 420,768 | configs/default.yaml (data.expected_rows) |
| `beijing_stations` | 12 | reports/metrics/station_holdout_h6.json |

## Slide 8 Persistence floor

| Value | Number | Source |
|---|---|---|
| `unch_h1` | 80.5 | reports/metrics/horizon_comparison.json |
| `floor_h1` | 0.7933 | reports/metrics/horizon_comparison.json |
| `floor_h6` | 0.5118 | reports/metrics/horizon_comparison.json |

## Slide 11 h = 1 result

| Value | Number | Source |
|---|---|---|
| `rf_h1` | 0.7994 | reports/baseline_metrics_h1.md |
| `floor_h1` | 0.7933 | reports/metrics/horizon_comparison.json |
| `unch_h1` | 80.5 | reports/metrics/horizon_comparison.json |

## Slide 12 Eight folds

| Value | Number | Source |
|---|---|---|
| `f8_cw_wins` | 4 | reports/metrics/rolling_cv_h6_f8.json |
| `f8_n` | 8 | reports/metrics/rolling_cv_h6_f8.json |
| `f8_rf_wins` | 2 | reports/metrics/rolling_cv_h6_f8.json |
| `f8_xgb_wins` | 0 | reports/metrics/rolling_cv_h6_f8.json |
| `f8_xgb_delta` | -0.0290 | reports/metrics/rolling_cv_h6_f8.json |
| `f8_xgb_p` | 0.0078 | reports/metrics/rolling_cv_h6_f8.json |

## Slide 13 Augmentation

| Value | Number | Source |
|---|---|---|
| `none_macro` | 0.5173 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `none_vu` | 0.5265 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `none_haz` | 0.6098 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `ctgan_macro` | 0.5212 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `ctgan_vu` | 0.4973 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `ctgan_haz` | 0.5908 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `smote_macro` | 0.5207 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `smote_vu` | 0.4757 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `smote_haz` | 0.5865 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `cw_macro` | 0.5149 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `cw_vu` | 0.4733 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `cw_haz` | 0.5843 | reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json |
| `smote_s` | 2.0 | reports/metrics/smote_h6.json |
| `ctgan_min` | 22 | reports/metrics/smote_h6.json |

## Slide 14 Capacity

| Value | Number | Source |
|---|---|---|
| `cap_lstm` | -0.0322 | reports/metrics/capacity_sweep_h6.json |
| `cap_tr` | -0.0201 | reports/metrics/capacity_sweep_h6.json |
| `cap_ratio` | 15 | reports/metrics/capacity_sweep_h6.json |
| `aleatoric` | 98.3% | reports/metrics/dl_h6.json (uncertainty block) |

## Slide 15 Uncertainty

| Value | Number | Source |
|---|---|---|
| `cov_marg` | 0.8908 | reports/metrics/conformal_h6.json |
| `haz_cov_marg` | 0.8432 | reports/metrics/conformal_h6.json |
| `haz_cov_mond` | 0.9083 | reports/metrics/conformal_h6.json |
| `ece_lstm` | 0.0190 | reports/metrics/dl_h6.json |
| `mce_lstm` | 0.0448 | reports/metrics/dl_h6.json |
| `ece_transformer` | 0.0113 | reports/metrics/dl_h6.json |
| `mce_transformer` | 0.1969 | reports/metrics/dl_h6.json |

## Slide 16 Data integrity

| Value | Number | Source |
|---|---|---|
| `span_pct` | 87 | reports/metrics/bangladesh_h6.json (audit block, recomputed) |
| `rows_pct` | 19 | reports/metrics/bangladesh_h6.json (audit block, recomputed) |
| `stated_cities` | 103 | reports/metrics/bangladesh_h6.json |
| `actual_cities` | 30 | reports/metrics/bangladesh_h6.json |
| `trend_r2` | 0.9920 | reports/metrics/bangladesh_h6.json |
| `clip_pct` | 4.9 | reports/metrics/bangladesh_h6.json |

## Slide 17 Monsoon washout

| Value | Number | Source |
|---|---|---|
| `cc_months` | 53 | reports/metrics/dhaka_monthly_crosscheck.json |
| `cc_bias` | +43 | reports/metrics/dhaka_monthly_crosscheck.json |
| `cc_mon_doe` | 35 | reports/metrics/dhaka_monthly_crosscheck.json |
| `cc_mon_men` | 112 | reports/metrics/dhaka_monthly_crosscheck.json |
| `cc_r2_doe` | 0.13 | reports/metrics/dhaka_monthly_crosscheck.json |
| `cc_r2_men` | 1.00 | reports/metrics/dhaka_monthly_crosscheck.json |

## Slide 18 Clean window: Bangladesh

| Value | Number | Source |
|---|---|---|
| `bd_wins` | 5 | reports/metrics/rolling_cv_h6_bangladesh.json |
| `bd_n` | 5 | reports/metrics/rolling_cv_h6_bangladesh.json |
| `bd_delta` | +0.0354 | reports/metrics/rolling_cv_h6_bangladesh.json |
| `bd_p` | 0.0312 | reports/metrics/rolling_cv_h6_bangladesh.json |

## Slide 19 Ground truth

| Value | Number | Source |
|---|---|---|
| `gt_ref_h` | 1,602 | reports/metrics/dhaka_ground_truth.json |
| `gt_rea_h` | 17 | reports/metrics/dhaka_ground_truth.json |
| `gt_hours` | 23,010 | reports/metrics/dhaka_ground_truth.json |
| `gt_r` | 0.729 | reports/metrics/dhaka_ground_truth.json |
| `gt_bias` | -52.7 | reports/metrics/dhaka_ground_truth.json |
| `gt_agree` | 30.8% | reports/metrics/dhaka_ground_truth.json |

## Slide 20 Hazardous detector

| Value | Number | Source |
|---|---|---|
| `hz_f1` | 0.4451 | reports/metrics/dhaka_pm25_model_h6.json |
| `hz_floor` | 0.3167 | reports/metrics/dhaka_pm25_model_h6.json |
| `hz_wins` | 7 | reports/metrics/dhaka_pm25_model_h6.json |
| `hz_n` | 7 | reports/metrics/dhaka_pm25_model_h6.json |
| `hz_p` | 0.0156 | reports/metrics/dhaka_pm25_model_h6.json |
| `sn_drop` | 28 | reports/metrics/sensor_noise_robustness_h6.json |
| `sn_noisy` | 0.3193 | reports/metrics/sensor_noise_robustness_h6.json |
| `sn_won` | 7 | reports/metrics/sensor_noise_robustness_h6.json |
| `sn_r2` | 0.20 | reports/metrics/sensor_noise_robustness_h6.json |

## Slide 21 Extended validation

| Value | Number | Source |
|---|---|---|
| `sh_wins` | 12 | reports/metrics/station_holdout_h6.json |
| `sh_n` | 12 | reports/metrics/station_holdout_h6.json |
| `sh_sig` | 11 | reports/metrics/station_holdout_h6.json |
| `sh_delta` | +0.0431 | reports/metrics/station_holdout_h6.json |
| `sp_frac` | 47% | reports/metrics/selective_prediction_h6.json |
| `sp_acc_full` | 0.5793 | reports/metrics/selective_prediction_h6.json |
| `sp_acc_conf` | 0.6289 | reports/metrics/selective_prediction_h6.json |
| `sp_f1_full` | 0.5173 | reports/metrics/selective_prediction_h6.json |
| `sp_f1_conf` | 0.5160 | reports/metrics/selective_prediction_h6.json |

## Slide 22 Statistical honesty

| Value | Number | Source |
|---|---|---|
| `b5_n` | 5 | reports/metrics/rolling_cv_h6.json |
| `b5_cw_wins` | 2 | reports/metrics/rolling_cv_h6.json |
| `f8_n` | 8 | reports/metrics/rolling_cv_h6_f8.json |
| `pers_sd5` | 0.0334 | reports/metrics/rolling_cv_h6.json |

## Slide 23 National context

| Value | Number | Source |
|---|---|---|
| `doe_stations` | 31 | docs/reference/NAQMP_2024-2030.pdf §3.3.2 (quoted in thesis §4.12) |
| `doe_cams` | 16 | docs/reference/NAQMP_2024-2030.pdf §3.3.2 (quoted in thesis §4.12) |
| `doe_cms` | 15 | docs/reference/NAQMP_2024-2030.pdf §3.3.2 (quoted in thesis §4.12) |
| `doe_cities` | 13 | docs/reference/NAQMP_2024-2030.pdf §3.3.2 (quoted in thesis §4.12) |
| `oq_n` | 22 | reports/metrics/openaq_survey.json |
| `oq_comp` | 1 | reports/metrics/openaq_survey.json |
| `oq_pass` | 0 | reports/metrics/openaq_survey.json |

## Slide 24 Limitations

| Value | Number | Source |
|---|---|---|
| `bd_test_haz` | 0 | reports/metrics/bangladesh_h6.json (conformal block) |
| `bd_test_vu` | 6 | reports/metrics/bangladesh_h6.json (conformal block) |

## Slide 25 Conclusion

| Value | Number | Source |
|---|---|---|
| `f8_cw_wins` | 4 | reports/metrics/rolling_cv_h6_f8.json |
| `f8_n` | 8 | reports/metrics/rolling_cv_h6_f8.json |
| `span_pct` | 87 | reports/metrics/bangladesh_h6.json (audit block, recomputed) |
| `hz_wins` | 7 | reports/metrics/dhaka_pm25_model_h6.json |
| `hz_n` | 7 | reports/metrics/dhaka_pm25_model_h6.json |

## Slide B Persistence floor by horizon

| Value | Number | Source |
|---|---|---|
| `floor_h1` | 0.7933 | reports/metrics/horizon_comparison.json |
| `floor_h6` | 0.5118 | reports/metrics/horizon_comparison.json |
| `floor_h12` | 0.3994 | reports/metrics/horizon_comparison.json |
| `floor_h24` | 0.2862 | reports/metrics/horizon_comparison.json |
| `unch_h1` | 80.5 | reports/metrics/horizon_comparison.json |
| `unch_h6` | 53.9 | reports/metrics/horizon_comparison.json |
| `unch_h12` | 42.4 | reports/metrics/horizon_comparison.json |
| `unch_h24` | 30.4 | reports/metrics/horizon_comparison.json |

## Slide B Per-class coverage

| Value | Number | Source |
|---|---|---|
| `pc_good_marg` | 0.8331 | reports/metrics/conformal_h6.json |
| `pc_good_mond` | 0.8673 | reports/metrics/conformal_h6.json |
| `pc_good_n` | 11,006 | reports/metrics/conformal_h6.json |
| `pc_moderate_marg` | 0.8912 | reports/metrics/conformal_h6.json |
| `pc_moderate_mond` | 0.8877 | reports/metrics/conformal_h6.json |
| `pc_moderate_n` | 14,020 | reports/metrics/conformal_h6.json |
| `pc_unhealthy_s_marg` | 0.7776 | reports/metrics/conformal_h6.json |
| `pc_unhealthy_s_mond` | 0.8635 | reports/metrics/conformal_h6.json |
| `pc_unhealthy_s_n` | 6,439 | reports/metrics/conformal_h6.json |
| `pc_unhealthy_marg` | 0.9668 | reports/metrics/conformal_h6.json |
| `pc_unhealthy_mond` | 0.8808 | reports/metrics/conformal_h6.json |
| `pc_unhealthy_n` | 18,824 | reports/metrics/conformal_h6.json |
| `pc_very_v_marg` | 0.9047 | reports/metrics/conformal_h6.json |
| `pc_very_v_mond` | 0.9357 | reports/metrics/conformal_h6.json |
| `pc_very_v_n` | 7,554 | reports/metrics/conformal_h6.json |
| `pc_hazardous_marg` | 0.8432 | reports/metrics/conformal_h6.json |
| `pc_hazardous_mond` | 0.9083 | reports/metrics/conformal_h6.json |
| `pc_hazardous_n` | 3,687 | reports/metrics/conformal_h6.json |

## Slide B Fold detail

| Value | Number | Source |
|---|---|---|
| `fold1_win` | 2014-10-06 to 2015-03-29 | reports/metrics/rolling_cv_h6.json |
| `fold1_pers` | 0.4773 | reports/metrics/rolling_cv_h6.json |
| `fold1_cw` | 0.4703 | reports/metrics/rolling_cv_h6.json |
| `fold2_win` | 2015-03-30 to 2015-09-20 | reports/metrics/rolling_cv_h6.json |
| `fold2_pers` | 0.4353 | reports/metrics/rolling_cv_h6.json |
| `fold2_cw` | 0.4363 | reports/metrics/rolling_cv_h6.json |
| `fold3_win` | 2015-09-21 to 2016-03-14 | reports/metrics/rolling_cv_h6.json |
| `fold3_pers` | 0.5155 | reports/metrics/rolling_cv_h6.json |
| `fold3_cw` | 0.5080 | reports/metrics/rolling_cv_h6.json |
| `fold4_win` | 2016-03-15 to 2016-09-06 | reports/metrics/rolling_cv_h6.json |
| `fold4_pers` | 0.5108 | reports/metrics/rolling_cv_h6.json |
| `fold4_cw` | 0.5260 | reports/metrics/rolling_cv_h6.json |
| `fold5_win` | 2016-09-07 to 2017-02-28 | reports/metrics/rolling_cv_h6.json |
| `fold5_pers` | 0.5051 | reports/metrics/rolling_cv_h6.json |
| `fold5_cw` | 0.5002 | reports/metrics/rolling_cv_h6.json |

## Slide B dataset_audit

| Value | Number | Source |
|---|---|---|
| `ia_verdict` | fabrication suspected | reports/metrics/integrity_audit_bangladesh.json |
| `ia_flagged` | 3 | reports/metrics/integrity_audit_bangladesh.json |
| `ia_boundary` | 2023-01-01 | reports/metrics/integrity_audit_bangladesh.json |
| `ia_err` | 149 | reports/metrics/integrity_audit_bangladesh.json |
| `ia_city` | Dhaka | reports/metrics/integrity_audit_bangladesh.json |
| `clean_start` | 2022-08-05 | reports/metrics/bangladesh_h6.json |

## Slide B Deployed predictor

| Value | Number | Source |
|---|---|---|
| `dep_trees` | 25 | reports/metrics/deployment_h6_bd.json |
| `dep_depth` | 12 | reports/metrics/deployment_h6_bd.json |
| `dep_kb` | 983 | reports/metrics/deployment_h6_bd.json |
| `dep_lat` | 0.0074 | reports/metrics/deployment_h6_bd.json |
| `dep_cov` | 0.9098 | reports/metrics/deployment_h6_bd.json |
| `dep_parity` | 1.0000 | reports/metrics/deployment_h6_bd.json |

## Slide B Reproducibility

| Value | Number | Source |
|---|---|---|
| `tests` | 162 | pytest --collect-only over pulsebench/tests, src, tests (run 2026-10-04) |
| `ci` | GitHub Actions: tests.yml, citation.yml | .github/workflows/ |
| `doi` | 10.5281/zenodo.22979304 | README.md (Zenodo DOI badge) |

## Slide 9 Rolling-origin CV

| Value | Number | Source |
|---|---|---|
| `embargo_h` | 24 | reports/metrics/rolling_cv_h6.json |
| `init_frac` | 40% | reports/metrics/rolling_cv_h6.json |
| `b5_n` | 5 | reports/metrics/rolling_cv_h6.json |
| `f8_n` | 8 | reports/metrics/rolling_cv_h6_f8.json |
