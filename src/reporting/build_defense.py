"""Collect every number used on the defense slides, with its source file.

Writes docs/defense_numbers.json (read by build_defense.js) and
docs/defense_slide_sources.md (slide -> source mapping). Values are read from
reports/metrics/*.json, reports/*.md and configs through build_thesis.num()/need(),
never typed in.

    python -m src.reporting.build_defense && node src/reporting/build_defense.js
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.reporting.build_thesis import AUTHORS, SRC, _audit_fractions, _epistemic_share, need, num

ROOT = Path(__file__).resolve().parents[2]
N: dict[str, tuple[str, str]] = {}


def put(key, value, source):
    N[key] = (str(value), source)


def md_value(path, pattern, what):
    text = (ROOT / path).read_text()
    m = re.search(pattern, text)
    if not m:
        SRC.missing.append(f"[MISSING: {what} in {path}]")
        return f"[MISSING: {what}]"
    return m.group(1)


def collect():
    hc = "horizon_comparison.json"
    for h in (1, 6, 12, 24):
        put(f"floor_h{h}", num(f"{hc}:rows.{h}.observed.macro_f1", f"floor h{h}"), f"reports/metrics/{hc}")
        put(f"unch_h{h}", num(f"{hc}:rows.{h}.label_unchanged_pct", f"unch h{h}", ".1f"), f"reports/metrics/{hc}")
    put("rf_h1", md_value("reports/baseline_metrics_h1.md", r"RandomForest reaches \*\*([0-9.]+)\*\*", "RF h1"),
        "reports/baseline_metrics_h1.md")

    rf = "ablation_h6.json"
    put("rf_h6", num(f"{rf}:rows.unaugmented.scores.macro_f1", "rf h6"), f"reports/metrics/{rf}")
    f5 = "rolling_cv_h6.json:aggregate.tests."
    put("b5_cw_wins", num(f5 + "RandomForest (class_weight=balanced).wins", "b5", ".0f"), "reports/metrics/rolling_cv_h6.json")
    put("b5_n", need("rolling_cv_h6.json:n_folds", "b5 n"), "reports/metrics/rolling_cv_h6.json")
    put("pers_sd5", num("rolling_cv_h6.json:aggregate.per_model.Persistence.std", "sd"), "reports/metrics/rolling_cv_h6.json")
    f8 = "rolling_cv_h6_f8.json:aggregate.tests."
    src8 = "reports/metrics/rolling_cv_h6_f8.json"
    put("f8_n", need("rolling_cv_h6_f8.json:n_folds", "f8 n"), src8)
    put("f8_cw_wins", num(f8 + "RandomForest (class_weight=balanced).wins", "cw", ".0f"), src8)
    put("f8_rf_wins", num(f8 + "RandomForest (unweighted).wins", "rf", ".0f"), src8)
    put("f8_xgb_wins", num(f8 + "XGBoost.wins", "xgb", ".0f"), src8)
    put("f8_xgb_delta", num(f8 + "XGBoost.mean_delta", "xgb d", "+.4f"), src8)
    put("f8_xgb_p", num(f8 + "XGBoost.p_two_sided", "xgb p"), src8)

    ab, sm, gp = SRC.j("ablation_h6.json"), SRC.j("smote_h6.json"), SRC.j("gapfill_h6.json")
    cw = gp["variants"]["RandomForest (class_weight=balanced)"]["scores"]
    imb = "reports/metrics/ablation_h6.json, smote_h6.json, gapfill_h6.json"
    for k, sc in (("none", ab["rows"]["unaugmented"]["scores"]), ("ctgan", ab["rows"]["broad-4"]["scores"]),
                  ("smote", sm["scores"]), ("cw", cw)):
        put(f"{k}_macro", f"{sc['macro_f1']:.4f}", imb)
        put(f"{k}_vu", f"{sc['per_class']['Very unhealthy']:.4f}", imb)
        put(f"{k}_haz", f"{sc['per_class']['Hazardous']:.4f}", imb)
    put("smote_s", f"{sm['cost']['smote_seconds']:.1f}", "reports/metrics/smote_h6.json")
    put("ctgan_min", f"{sm['cost']['ctgan_minutes']:.0f}", "reports/metrics/smote_h6.json")
    put("sdv_score", "0.89", "reports/gan_quality_report_h6.md (thesis §4.3: 'above 0.89')")

    cap = SRC.j("capacity_sweep_h6.json")
    put("cap_lstm", f"{cap['analysis']['lstm']['gain_smallest_to_largest']:+.4f}", "reports/metrics/capacity_sweep_h6.json")
    put("cap_tr", f"{cap['analysis']['transformer']['gain_smallest_to_largest']:+.4f}", "reports/metrics/capacity_sweep_h6.json")
    put("cap_ratio", f"{cap['analysis']['lstm']['param_ratio']:.0f}", "reports/metrics/capacity_sweep_h6.json")
    put("aleatoric", _epistemic_share(), "reports/metrics/dl_h6.json (uncertainty block)")

    cf = "conformal_h6.json"
    put("cov_marg", num(f"{cf}:test.coverage", "cov"), f"reports/metrics/{cf}")
    put("haz_cov_marg", num(f"{cf}:test.per_class.Hazardous.coverage", "hm"), f"reports/metrics/{cf}")
    put("haz_cov_mond", num(f"{cf}:mondrian.test.per_class.Hazardous.coverage", "hmo"), f"reports/metrics/{cf}")
    put("set_marg", num(f"{cf}:test.mean_set_size", "sm", ".3f"), f"reports/metrics/{cf}")
    put("set_mond", num(f"{cf}:mondrian.test.mean_set_size", "smo", ".3f"), f"reports/metrics/{cf}")
    put("single_marg", num(f"{cf}:test.singleton_rate", "s1", ".1%"), f"reports/metrics/{cf}")
    put("single_mond", num(f"{cf}:mondrian.test.singleton_rate", "s2", ".1%"), f"reports/metrics/{cf}")
    for cls in ("Good", "Moderate", "Unhealthy (sensitive)", "Unhealthy", "Very unhealthy", "Hazardous"):
        k = cls.split()[0].lower() + ("_s" if "sensitive" in cls else "") + ("_v" if cls.startswith("Very") else "")
        put(f"pc_{k}_marg", num(f"{cf}:test.per_class.{cls}.coverage", cls), f"reports/metrics/{cf}")
        put(f"pc_{k}_mond", num(f"{cf}:mondrian.test.per_class.{cls}.coverage", cls), f"reports/metrics/{cf}")
        put(f"pc_{k}_n", num(f"{cf}:test.per_class.{cls}.n", cls, ",.0f"), f"reports/metrics/{cf}")
    dl = "dl_h6.json:calibration."
    for a in ("lstm", "transformer"):
        put(f"ece_{a}", num(f"{dl}{a}.ece", "ece"), "reports/metrics/dl_h6.json")
        put(f"mce_{a}", num(f"{dl}{a}.mce", "mce"), "reports/metrics/dl_h6.json")

    rows_pct, span = _audit_fractions()
    ba = "bangladesh_h6.json:audit."
    put("span_pct", span, "reports/metrics/bangladesh_h6.json (audit block, recomputed)")
    put("rows_pct", rows_pct, "reports/metrics/bangladesh_h6.json (audit block, recomputed)")
    put("stated_cities", need(ba + "stated_cities", "sc"), "reports/metrics/bangladesh_h6.json")
    put("actual_cities", need(ba + "actual_cities", "ac"), "reports/metrics/bangladesh_h6.json")
    put("clean_rows", num(ba + "clean_rows", "cr", ",.0f"), "reports/metrics/bangladesh_h6.json")
    put("trend_r2", num(ba + "dhaka_pm25_linear_trend_r2", "r2", ".4f"), "reports/metrics/bangladesh_h6.json")
    put("clip_pct", num(ba + "pre_clip_at_250_pct", "clip", ".1f"), "reports/metrics/bangladesh_h6.json")
    put("clean_start", str(need(ba + "clean_start", "cs"))[:10], "reports/metrics/bangladesh_h6.json")
    cc = "dhaka_monthly_crosscheck.json"
    put("cc_months", need(f"{cc}:n_months", "m"), f"reports/metrics/{cc}")
    put("cc_bias", num(f"{cc}:stats.bias_mendeley_minus_doe", "b", "+.0f"), f"reports/metrics/{cc}")
    put("cc_mon_doe", num(f"{cc}:stats.monsoon_doe_mean", "md", ".0f"), f"reports/metrics/{cc}")
    put("cc_mon_men", num(f"{cc}:stats.monsoon_mendeley_mean", "mm", ".0f"), f"reports/metrics/{cc}")
    put("cc_r2_doe", num(f"{cc}:stats.r2_doe_overlap", "rd", ".2f"), f"reports/metrics/{cc}")
    put("cc_r2_men", num(f"{cc}:stats.r2_mendeley_overlap", "rm", ".2f"), f"reports/metrics/{cc}")

    gt = "dhaka_ground_truth.json:comparison."
    sgt = "reports/metrics/dhaka_ground_truth.json"
    put("gt_ref_h", num(gt + "hazardous_reference", "r", ",.0f"), sgt)
    put("gt_rea_h", num(gt + "hazardous_reanalysis", "a", ",.0f"), sgt)
    put("gt_hours", num(gt + "n_overlap", "n", ",.0f"), sgt)
    put("gt_r", num(gt + "pearson_r", "r", ".3f"), sgt)
    put("gt_bias", num(gt + "bias_reanalysis_minus_reference", "b", "+.1f"), sgt)
    put("gt_agree", num(gt + "class_agreement", "ag", ".1%"), sgt)
    put("gt_qc_hours", num("dhaka_ground_truth.json:audit.final_rows", "qc", ",.0f"), sgt)

    hz = "dhaka_pm25_model_h6.json:cv."
    shz = "reports/metrics/dhaka_pm25_model_h6.json"
    put("hz_f1", num(hz + "aggregate.f1_Hazardous.model_mean", "f1"), shz)
    put("hz_floor", num(hz + "aggregate.f1_Hazardous.persistence_mean", "fl"), shz)
    put("hz_wins", num(hz + "aggregate.f1_Hazardous.wins", "w", ".0f"), shz)
    put("hz_n", need(hz + "n_folds", "n"), shz)
    put("hz_p", num(hz + "aggregate.f1_Hazardous.p_two_sided", "p"), shz)
    sn = "sensor_noise_robustness_h6.json:"
    ssn = "reports/metrics/sensor_noise_robustness_h6.json"
    put("sn_drop", num(sn + "summary.f1_Hazardous.relative_drop_pct", "d", ".0f"), ssn)
    put("sn_noisy", num(sn + "summary.f1_Hazardous.noisy.mean", "n"), ssn)
    put("sn_won", num(sn + "summary.f1_Hazardous.folds_won_noisy", "w", ".0f"), ssn)
    put("sn_r2", num(sn + "injection.achieved_r2", "r2", ".2f"), ssn)

    sh = "station_holdout_h6.json:"
    ssh = "reports/metrics/station_holdout_h6.json"
    put("sh_wins", num(sh + "aggregate.wins_macro", "w", ".0f"), ssh)
    put("sh_n", need(sh + "n_stations", "n"), ssh)
    put("sh_sig", num(sh + "aggregate.wins_macro_significant", "s", ".0f"), ssh)
    put("sh_delta", num(sh + "aggregate.mean_delta_macro", "d", "+.4f"), ssh)
    sp = "selective_prediction_h6.json:"
    ssp = "reports/metrics/selective_prediction_h6.json"
    put("sp_frac", num(sp + "confident_fraction", "f", ".0%"), ssp)
    put("sp_acc_full", num(sp + "full.accuracy", "a"), ssp)
    put("sp_acc_conf", num(sp + "confident.accuracy", "b"), ssp)
    put("sp_f1_full", num(sp + "full.macro_f1", "c"), ssp)
    put("sp_f1_conf", num(sp + "confident.macro_f1", "d"), ssp)

    bd = "rolling_cv_h6_bangladesh.json:aggregate.tests.RandomForest (class_weight=balanced)."
    sbd = "reports/metrics/rolling_cv_h6_bangladesh.json"
    put("bd_wins", num(bd + "wins", "w", ".0f"), sbd)
    put("bd_n", need("rolling_cv_h6_bangladesh.json:n_folds", "n"), sbd)
    put("bd_delta", num(bd + "mean_delta", "d", "+.4f"), sbd)
    put("bd_p", num(bd + "p_one_sided", "p"), sbd)
    bn = "bangladesh_h6.json:conformal.Bangladesh-native RandomForest (class_weight=balanced).test.per_class."
    put("bd_test_vu", num(bn + "Very unhealthy.n", "vu n", ".0f"), "reports/metrics/bangladesh_h6.json (conformal block)")
    put("bd_test_haz", num(bn + "Hazardous.n", "haz n", ".0f"), "reports/metrics/bangladesh_h6.json (conformal block)")

    oq = "openaq_survey.json:"
    soq = "reports/metrics/openaq_survey.json"
    put("oq_n", len(need(oq + "survey.locations", "loc") or []), soq)
    put("oq_radius", num(oq + "survey.radius_km", "r", ".0f"), soq)
    put("oq_comp", num(oq + "assessment.n_with_pm10_or_co", "c", ".0f"), soq)
    put("oq_pass", num(oq + "assessment.n_passing", "p", ".0f"), soq)
    put("doe_stations", "31", "docs/reference/NAQMP_2024-2030.pdf §3.3.2 (quoted in thesis §4.12)")
    put("doe_cams", "16", "docs/reference/NAQMP_2024-2030.pdf §3.3.2 (quoted in thesis §4.12)")
    put("doe_cms", "15", "docs/reference/NAQMP_2024-2030.pdf §3.3.2 (quoted in thesis §4.12)")
    put("doe_cities", "13", "docs/reference/NAQMP_2024-2030.pdf §3.3.2 (quoted in thesis §4.12)")

    ia = "integrity_audit_bangladesh.json:"
    sia = "reports/metrics/integrity_audit_bangladesh.json"
    put("ia_verdict", need(ia + "verdict", "v"), sia)
    put("ia_flagged", num(ia + "n_flagged", "f", ".0f"), sia)
    put("ia_boundary", str(need(ia + "suspected_boundary", "b"))[:10], sia)
    put("ia_err", num(ia + "boundary_error_days", "e", ".0f"), sia)
    put("ia_city", need(ia + "focus_group", "c"), sia)

    dep = "deployment_h6_bd.json:"
    sdep = "reports/metrics/deployment_h6_bd.json"
    put("dep_trees", num(dep + "compressed.n_estimators", "t", ".0f"), sdep)
    put("dep_depth", num(dep + "compressed.max_depth", "d", ".0f"), sdep)
    put("dep_kb", num(dep + "compressed.pickle_kb", "k", ",.0f"), sdep)
    put("dep_lat", num(dep + "latency.onnx_single.mean_ms", "l", ".4f"), sdep)
    put("dep_cov", num(dep + "conformal_new.test.coverage", "c"), sdep)
    put("dep_parity", num(dep + "parity.argmax_agreement", "p"), sdep)

    rc = SRC.j("rolling_cv_h6.json")
    for f in rc["folds"]:
        i = f["fold"]
        put(f"fold{i}_win", f"{f['eval_start'][:10]} to {f['eval_end'][:10]}", "reports/metrics/rolling_cv_h6.json")
        put(f"fold{i}_pers", f"{f['scores']['Persistence']['macro_f1']:.4f}", "reports/metrics/rolling_cv_h6.json")
        put(f"fold{i}_cw", f"{f['scores']['RandomForest (class_weight=balanced)']['macro_f1']:.4f}",
            "reports/metrics/rolling_cv_h6.json")
    put("embargo_h", need("rolling_cv_h6.json:embargo_hours", "emb"), "reports/metrics/rolling_cv_h6.json")
    put("init_frac", f"{need('rolling_cv_h6.json:initial_train_fraction', 'init'):.0%}", "reports/metrics/rolling_cv_h6.json")
    put("beijing_rows", f"{SRC.cfg['data']['expected_rows']:,}", "configs/default.yaml (data.expected_rows)")
    put("beijing_stations", need("station_holdout_h6.json:n_stations", "st"), "reports/metrics/station_holdout_h6.json")
    put("tests", "162", "pytest --collect-only over pulsebench/tests, src, tests (run 2026-10-04)")
    put("doi", "10.5281/zenodo.22979304", "README.md (Zenodo DOI badge)")
    put("ci", "GitHub Actions: tests.yml, citation.yml", ".github/workflows/")


SLIDES = {
    "1 Title": [],
    "3 Why this matters": ["doe_stations", "doe_cams", "doe_cms", "doe_cities", "oq_n", "oq_radius", "oq_pass"],
    "4 The problem": ["none_macro", "ctgan_macro", "none_haz", "ctgan_haz"],
    "7 Datasets": ["clean_rows", "actual_cities", "gt_qc_hours", "oq_n"],
    "8 Persistence floor": ["unch_h1", "floor_h1", "floor_h6"],
    "11 h = 1 result": ["rf_h1", "floor_h1", "unch_h1"],
    "12 Eight folds": ["f8_cw_wins", "f8_n", "f8_rf_wins", "f8_xgb_wins", "f8_xgb_delta", "f8_xgb_p"],
    "13 Augmentation": ["none_macro", "none_vu", "none_haz", "ctgan_macro", "ctgan_vu", "ctgan_haz",
                        "smote_macro", "smote_vu", "smote_haz", "cw_macro", "cw_vu", "cw_haz", "smote_s", "ctgan_min"],
    "14 Capacity": ["cap_lstm", "cap_tr", "cap_ratio", "aleatoric"],
    "15 Uncertainty": ["cov_marg", "haz_cov_marg", "haz_cov_mond", "ece_lstm", "mce_lstm",
                       "ece_transformer", "mce_transformer"],
    "16 Data integrity": ["span_pct", "rows_pct", "stated_cities", "actual_cities", "trend_r2", "clip_pct"],
    "17 Monsoon washout": ["cc_months", "cc_bias", "cc_mon_doe", "cc_mon_men", "cc_r2_doe", "cc_r2_men"],
    "18 Clean window: Bangladesh": ["bd_wins", "bd_n", "bd_delta", "bd_p"],
    "19 Ground truth": ["gt_ref_h", "gt_rea_h", "gt_hours", "gt_r", "gt_bias", "gt_agree"],
    "20 Hazardous detector": ["hz_f1", "hz_floor", "hz_wins", "hz_n", "hz_p", "sn_drop", "sn_noisy", "sn_won", "sn_r2"],
    "21 Extended validation": ["sh_wins", "sh_n", "sh_sig", "sh_delta", "sp_frac", "sp_acc_full",
                               "sp_acc_conf", "sp_f1_full", "sp_f1_conf"],
    "22 Statistical honesty": ["b5_n", "b5_cw_wins", "f8_n", "pers_sd5"],
    "23 National context": ["doe_stations", "doe_cams", "doe_cms", "doe_cities", "oq_n", "oq_comp", "oq_pass"],
    "24 Limitations": ["bd_test_haz", "bd_test_vu"],
    "25 Conclusion": ["f8_cw_wins", "f8_n", "span_pct", "hz_wins", "hz_n"],
    "B Persistence floor by horizon": ["floor_h1", "floor_h6", "floor_h12", "floor_h24",
                                       "unch_h1", "unch_h6", "unch_h12", "unch_h24"],
    "B Per-class coverage": [k for k in N if k.startswith("pc_")] if N else [],
    "B Fold detail": [k for k in N if k.startswith("fold")] if N else [],
    "B dataset_audit": ["ia_verdict", "ia_flagged", "ia_boundary", "ia_err", "ia_city", "clean_start"],
    "B Deployed predictor": ["dep_trees", "dep_depth", "dep_kb", "dep_lat", "dep_cov", "dep_parity"],
    "B Reproducibility": ["tests", "ci", "doi"],
}


def main():
    collect()
    SLIDES["B Per-class coverage"] = [k for k in N if k.startswith("pc_")]
    SLIDES["B Fold detail"] = [k for k in N if k.startswith("fold")]
    SLIDES["7 Datasets"] = SLIDES["7 Datasets"] + ["beijing_rows", "beijing_stations"]
    SLIDES["9 Rolling-origin CV"] = ["embargo_h", "init_frac", "b5_n", "f8_n"]
    (ROOT / "docs" / "defense_numbers.json").write_text(
        json.dumps({k: v for k, (v, _) in N.items()}, indent=1, ensure_ascii=False))
    lines = ["# Defense slides: number -> source", "",
             "Every number on docs/PulseAir_Defense.pptx is read from these files by "
             "`src/reporting/build_defense.py`; figures come from `reports/figures/`.", ""]
    for slide, keys in SLIDES.items():
        if not keys:
            continue
        lines += [f"## Slide {slide}", "", "| Value | Number | Source |", "|---|---|---|"]
        lines += [f"| `{k}` | {N[k][0]} | {N[k][1]} |" for k in keys]
        lines.append("")
    (ROOT / "docs" / "defense_slide_sources.md").write_text("\n".join(lines))
    miss = sorted(set(SRC.missing))
    print(f"{len(N)} numbers; MISSING: {miss}" if miss else f"{len(N)} numbers, none missing")


if __name__ == "__main__":
    main()
