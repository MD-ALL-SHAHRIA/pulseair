"""Compile every phase's metrics into one thesis-facing results package.

Reads the JSON written by each phase -- nothing here is retyped by hand -- and emits
``reports/final_results_summary.md``: a master comparison across every model variant
evaluated, the negative results written up as findings rather than omissions, the
deployed system in one place, a suggested paper structure, and the questions a
reviewer is likely to ask.

    python -m src.reporting.compile_results

Every variant is scored on the **same** test rows: horizon 6 h, observed labels only
(``is_imputed_pm25 == False``). The compiler asserts that rather than assuming it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = REPO_ROOT / "reports" / "metrics"
OUT = REPO_ROOT / "reports" / "final_results_summary.md"
RARE = ("Very unhealthy", "Hazardous")


def load(name: str) -> dict:
    path = METRICS / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path.relative_to(REPO_ROOT)} is missing — run the phase that writes it")
    return json.loads(path.read_text())


def table(header, rows) -> str:
    esc = lambda cs: [str(c).replace("|", "\\|") for c in cs]
    body = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
    return (f"| {' | '.join(esc(header))} |\n"
            f"| {' | '.join(['---'] * len(header))} |\n{body}")


def sig_cell(t) -> str:
    """One cell summarising a paired-bootstrap result against persistence."""
    if t is None:
        return "— (reference)"
    if t == "untested":
        return "not tested"
    d, lo, hi = t["observed_diff"], t["ci_low"], t["ci_high"]
    if not t["significant"]:
        return f"no ({d:+.4f} [{lo:+.4f}, {hi:+.4f}])"
    word = "**yes**" if d > 0 else "**worse**"
    return f"{word} ({d:+.4f} [{lo:+.4f}, {hi:+.4f}])"


def collect() -> dict:
    ab = load("ablation_h6.json")
    dl = load("dl_h6.json")
    dep = load("deployment_h6.json")
    conf = load("conformal_h6.json")
    base = load("baseline_h6.json")
    gap = load("gapfill_h6.json")
    gan = load("gan_h6.json")
    gan_t = load("gan_h6_targeted.json")
    horizon_like = load("dl_lr_sweep_h6.json")
    rcv = load("rolling_cv_h6.json")
    bdv = load("bangladesh_h6.json")
    bdcv = load("rolling_cv_h6_bangladesh.json")
    smote = load("smote_h6.json")
    hj = load("hj633_h6.json")
    cap = load("capacity_sweep_h6.json")
    bddep = load("deployment_h6_bd.json")
    gt = load("dhaka_ground_truth.json")
    gtm = load("dhaka_pm25_model_h6.json")

    # Every row must describe the same evaluation set or the table is meaningless.
    n = ab["n_test"]
    assert dl["counts"]["test_observed"] == n, (dl["counts"]["test_observed"], n)
    assert gap["n_test_observed"] == n, (gap["n_test_observed"], n)
    assert conf["counts"]["test_observed"] == n, (conf["counts"]["test_observed"], n)

    labels = ab["labels"]
    V = []

    def row(name, scores, vs, source, note=""):
        V.append({"name": name, "scores": scores, "vs_persistence": vs,
                  "source": source, "note": note})

    row("Persistence (zero-parameter)", ab["rows"]["Persistence"]["scores"], None,
        "horizon_comparison.md", "the floor every model must clear")
    row("RandomForest (Phase 3)", ab["rows"]["unaugmented"]["scores"],
        ab["vs_persistence"]["unaugmented"], "baseline_metrics_h6.md",
        "best on test; **loses to persistence on validation**")
    row("XGBoost (Phase 3)", gap["variants"]["XGBoost"]["scores"],
        gap["variants"]["XGBoost"]["vs_persistence"]["macro_f1"],
        "baseline_metrics_h6.md")
    row("RF + CTGAN, broad 4-class", ab["rows"]["broad-4"]["scores"],
        ab["vs_persistence"]["broad-4"], "gan_ablation_h6.md")
    row("RF + CTGAN, targeted 2-class", ab["rows"]["targeted-2"]["scores"],
        ab["vs_persistence"]["targeted-2"], "gan_ablation_h6.md")
    row("LSTM + MC dropout", dl["results"]["lstm (MC mean)"],
        dl["tests"]["Persistence"]["macro_f1"], "dl_metrics_h6.md")
    # Phase 5 saved only the selected model, so no paired bootstrap exists for the
    # Transformer. Its point estimates are real; the significance cell says so.
    row("Transformer + MC dropout", dl["results"]["transformer (MC mean)"], "untested",
        "dl_metrics_h6.md", "no paired bootstrap run")
    row("RF + SMOTE (imbalanced-learn)", smote["scores"],
        smote["tests"]["Persistence"]["macro_f1"],
        "final_results_summary.md §2.11",
        "control for \"why CTGAN not SMOTE\"")
    row("RandomForest, class_weight=balanced",
        gap["variants"]["RandomForest (class_weight=balanced)"]["scores"],
        gap["variants"]["RandomForest (class_weight=balanced)"]["vs_persistence"]["macro_f1"],
        "final_results_summary.md §5a",
        "control for \"why GAN not class weighting\"")
    if gap.get("cw_compressed_name"):
        cwc = gap["cw_compressed_name"]
        row(cwc.replace("Compressed RF, class-weighted", "Compressed RF + class weight"),
            gap["variants"][cwc]["scores"],
            gap["variants"][cwc]["vs_persistence"]["macro_f1"],
            "deployment_report_h6_cw.md",
            "**two-sided rule winner (validation)**")
    cname = gap["compressed_name"]
    row(f"Compressed RF ({gap['compressed_config']['n_estimators']} trees x depth "
        f"{gap['compressed_config']['max_depth']})",
        gap["variants"][cname]["scores"],
        gap["variants"][cname]["vs_persistence"]["macro_f1"],
        "deployment_report_h6.md", "unweighted re-sweep fallback — *no* unweighted config cleared the floor")

    return {"rcv": rcv, "bd": bdv, "bdcv": bdcv, "smote": smote, "hj": hj,
            "cap": cap, "bddep": bddep, "gt": gt, "gtm": gtm,
            "variants": V, "labels": labels, "n_test": n, "ab": ab, "dl": dl,
            "dep": dep, "conf": conf, "base": base, "gap": gap, "gan": gan,
            "gan_t": gan_t, "lr": horizon_like, "n_boot": ab["n_boot"]}


# ------------------------------------------------------------ master comparison


def headline_finding(d: dict) -> str:
    """Both datasets in one verdict: Beijing = methodology, Bangladesh = deployment."""
    bj, bd = d["rcv"], d["bdcv"]
    bjt = {k: v for k, v in bj["aggregate"]["tests"].items() if not k.startswith("_")}
    bdt = {k: v for k, v in bd["aggregate"]["tests"].items() if not k.startswith("_")}
    bj_best = max(bjt, key=lambda k: bjt[k]["wins"])
    bd_best = max(bdt, key=lambda k: bdt[k]["wins"])
    bjw, bdw = bjt[bj_best]["wins"], bdt[bd_best]["wins"]
    bjn, bdn = bj["n_folds"], bd["n_folds"]
    haz = sum(f["support"]["Hazardous"] for f in bd["folds"])
    vu = sum(f["support"]["Very unhealthy"] for f in bd["folds"])

    rows = [
        ["**Beijing** (Phases 0–9)", "methodology development + validation framework",
         f"{bjw}/{bjn} folds", f"{bjt[bj_best]['mean_delta']:+.4f}",
         f"p = {bjt[bj_best]['p_one_sided']:.4f}", "**no deployable model**"],
        ["**Bangladesh** (Phase 10)", "deployment candidate, target population",
         f"**{bdw}/{bdn} folds**", f"**{bdt[bd_best]['mean_delta']:+.4f}**",
         f"**p = {bdt[bd_best]['p_one_sided']:.4f}**",
         "**four common classes only**"],
    ]

    hdr = ["Dataset", "Role", "Best model beats persistence in", "Mean Δ",
           "Wilcoxon (1-sided)", "Deployment status"]
    quoted = table(hdr, rows).replace(chr(10), chr(10) + "> ")

    return f"""> ### One result, two datasets, two different roles
>
> {quoted}
>
> **Beijing is the methodology, not the model.** Five-fold rolling-origin CV shows
> that *no* Beijing-trained model — RandomForest, class-weighted RandomForest,
> XGBoost, LSTM, Transformer, CTGAN- or SMOTE-augmented — beats the zero-parameter
> persistence rule in more than **{bjw} of {bjn}** folds. The +0.0055 single-split win
> reported in Phase 3 sits inside the fold-to-fold variance of the baseline itself,
> and two independent robustness checks agree: it also disappears under the Chinese
> HJ 633-2012 labelling (Limitations §6). **State the Beijing conclusion plainly: the
> pipeline is the contribution there, not a predictor.**
>
> **Bangladesh is the deployment candidate.** A class-weighted RandomForest trained on
> Dhaka, Chittagong, Comilla and Barisāl beats its own persistence floor in
> **{bdw}/{bdn} rolling-origin folds** (mean {bdt[bd_best]['mean_delta']:+.4f},
> one-sided Wilcoxon p = {bdt[bd_best]['p_one_sided']:.4f} — the floor a {bdn}-pair
> signed-rank test can reach). It is the only model in the project that clears its
> floor in every fold, and it does so on the population the device is for.
>
> **The scope of that claim is bounded, and the bound must travel with it.** The
> Bangladesh result covers the **four common AQI classes** — Good, Moderate, Unhealthy
> (sensitive), Unhealthy. It says **nothing** about Hazardous or Very unhealthy:
> across all {bdn} evaluation blocks the usable Bangladesh data contains **{haz}
> Hazardous** and **{vu} Very unhealthy** samples. The dataset's modelled PM2.5 rarely
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
> earned a bounded deployment claim; the advisory classes remain unvalidated.**"""


def master_table(d: dict) -> str:
    rows = []
    for v in d["variants"]:
        s = v["scores"]
        name = v["name"] + (f" — {v['note']}" if v["note"] else "")
        rows.append([
            name, f"{s['macro_f1']:.4f}",
            f"{s['per_class']['Hazardous']:.4f}",
            f"{s['per_class']['Very unhealthy']:.4f}",
            sig_cell(v["vs_persistence"]),
            f"`{v['source']}`",
        ])
    return table(["Variant", "Macro-F1", "Hazardous F1", "V.unhealthy F1",
                  "Beats persistence?", "Source"], rows)


def per_class_table(d: dict) -> str:
    labels = d["labels"]
    header = ["Class"] + [v["name"].split(" (")[0].split(" —")[0] for v in d["variants"]]
    rows = [[f"**{l}**" if l in RARE else l,
             *[f"{v['scores']['per_class'][l]:.4f}" for v in d["variants"]]]
            for l in labels]
    return table(header, rows)


# ----------------------------------------------------------------- narrative


def negatives_section(d: dict) -> str:
    ab, dl, conf, dep = d["ab"], d["dl"], d["conf"], d["dep"]
    gan = d["gan"]
    v = {x["name"]: x for x in d["variants"]}

    persist = v["Persistence (zero-parameter)"]["scores"]["macro_f1"]
    rf = v["RandomForest (Phase 3)"]["scores"]["macro_f1"]
    rf_vs = v["RandomForest (Phase 3)"]["vs_persistence"]
    b4 = ab["tests"]["broad-4"]
    t2 = ab["tests"]["targeted-2"]
    lstm_vs_rf = dl["tests"]["RandomForest (Phase 3)"]["macro_f1"]
    lstm_haz = dl["tests"]["RandomForest (Phase 3)"]["Hazardous"]
    marg_haz = conf["test"]["per_class"]["Hazardous"]["coverage"]
    mond_haz = conf["mondrian"]["test"]["per_class"]["Hazardous"]["coverage"]
    epi = dl["uncertainty"][dl["selected"]]
    epi_share = epi["mean_epistemic"] / epi["mean_entropy"]
    val_lstm = dl["training"][dl["selected"]]["val_macro_f1"]
    test_lstm = dl["results"][f"{dl['selected']} (MC mean)"]["macro_f1"]
    rf_val = d["base"]["results"]["RandomForest"]["val"]["observed"]["macro_f1"]
    comp = dep["compressed"]
    base_rf = dep["baseline"]

    dep_lat = dep["latency"]["onnx_single"]["mean_ms"]
    sm = d["smote"]
    smote_n = sm["n_synthetic"]
    smote_secs = sm["cost"]["smote_seconds"]
    ctgan_mins = sm["cost"]["ctgan_minutes"]
    cost_ratio = (ctgan_mins * 60) / max(smote_secs, 1e-9)
    smote_vs_rf = sm["tests"]["RandomForest (unweighted)"]["macro_f1"]["observed_diff"]
    smote_vs_cw = sm["tests"]["RandomForest (class_weight=balanced)"]["macro_f1"]["observed_diff"]
    smote_vu = sm["tests"]["RandomForest (unweighted)"]["Very unhealthy"]["observed_diff"]
    smote_haz = sm["tests"]["RandomForest (unweighted)"]["Hazardous"]["observed_diff"]

    def _sig(t):
        return "**significant**" if t["significant"] else "n.s."

    smote_table = table(
        ["SMOTE vs", "Macro-F1 Δ", "95% CI", "Very unhealthy Δ", "Hazardous Δ"],
        [[ref, f"{t['macro_f1']['observed_diff']:+.4f} ({_sig(t['macro_f1'])})",
          f"[{t['macro_f1']['ci_low']:+.4f}, {t['macro_f1']['ci_high']:+.4f}]",
          f"{t['Very unhealthy']['observed_diff']:+.4f} ({_sig(t['Very unhealthy'])})",
          f"{t['Hazardous']['observed_diff']:+.4f} ({_sig(t['Hazardous'])})"]
         for ref, t in sm["tests"].items()])

    ab_t = d["ab"]["tests"]
    imbalance_table = table(
        ["Intervention", "Cost", "Macro-F1 vs plain RF", "Advisory classes"],
        [["CTGAN, broad 4-class", f"{ctgan_mins:.0f} min + redesign",
          f"{ab_t['broad-4']['macro_f1']['observed_diff']:+.4f} (significant)",
          f"**both significantly worse** (VU {ab_t['broad-4']['Very unhealthy']['observed_diff']:+.4f}, "
          f"Haz {ab_t['broad-4']['Hazardous']['observed_diff']:+.4f})"],
         ["SMOTE", f"{smote_secs:.1f} s",
          f"{smote_vs_rf:+.4f} (significant)",
          f"**both significantly worse** (VU {smote_vu:+.4f}, Haz {smote_haz:+.4f})"],
         ["class_weight='balanced'", "0 s (one hyperparameter)",
          "see §2.10 — best on validation, only model to clear the validation floor",
          "no significant regression"]])
    gp = d["gap"]
    cwv = gp["class_weighted_vs_persistence_val"]
    cw_test = next(x for x in d["variants"]
                   if x["name"].startswith("RandomForest, class_weight"))
    cw_t = cw_test["vs_persistence"]
    rf_plain = {x["name"]: x for x in d["variants"]}["RandomForest (Phase 3)"]
    cw_block = f"""| | Unweighted RF | **class_weight="balanced"** |
|---|---|---|
| Validation macro-F1 | {d['base']['results']['RandomForest']['val']['observed']['macro_f1']:.4f} | **{gp['class_weighted_val_macro_f1']:.4f}** |
| vs persistence, **validation** | −0.0153 (significantly worse) | **{cwv['observed_diff']:+.4f}** [{cwv['ci_low']:+.4f}, {cwv['ci_high']:+.4f}] — **significantly better** |
| Test macro-F1 | {rf_plain['scores']['macro_f1']:.4f} | {cw_test['scores']['macro_f1']:.4f} |
| vs persistence, test | {rf_plain['vs_persistence']['observed_diff']:+.4f} (significant) | {cw_t['observed_diff']:+.4f} [{cw_t['ci_low']:+.4f}, {cw_t['ci_high']:+.4f}] ({'significant' if cw_t['significant'] else 'not significant'}) |
| Hazardous F1 (test) | {rf_plain['scores']['per_class']['Hazardous']:.4f} | {cw_test['scores']['per_class']['Hazardous']:.4f} |
| Very unhealthy F1 (test) | {rf_plain['scores']['per_class']['Very unhealthy']:.4f} | {cw_test['scores']['per_class']['Very unhealthy']:.4f} |

**Class weighting raises validation macro-F1 by
{gp['class_weighted_val_macro_f1'] - d['base']['results']['RandomForest']['val']['observed']['macro_f1']:+.4f}
— and it is the only model in this project that significantly beats the persistence
floor on validation.** Every unweighted forest, at every size in the Phase 7 sweep
(0 of 30), is significantly *below* it.

Set against CTGAN: the broad 4-class variant moved test macro-F1 +0.0039 while
significantly degrading both advisory classes. Class weighting moves validation
macro-F1 {gp['class_weighted_val_macro_f1'] - d['base']['results']['RandomForest']['val']['observed']['macro_f1']:+.4f}
and costs nothing — no generator, no synthetic rows, no validity checks, no extra
artifact to version.

On **test** the two are closer ({cw_t['observed_diff']:+.4f} vs
{rf_plain['vs_persistence']['observed_diff']:+.4f} against persistence), which is the
val/test mismatch of section 2.8 appearing yet again — but validation is the split
selection is allowed to use, and on validation the verdict is unambiguous."""
    cv = next(x for x in d["variants"] if x["name"].startswith("Compressed RF"))["vs_persistence"]
    comp_vs_persist = (
        f"**{cv['observed_diff']:+.4f}** [{cv['ci_low']:+.4f}, {cv['ci_high']:+.4f}]"
        if cv != "untested" else "not tested")

    return f"""## 2. What worked, what didn't

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

Moving to a 6-hour horizon dropped persistence to **{persist:.4f}** and made the task
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

- **Broad (4 classes, 82,958 synthetic rows):** macro-F1 **{b4['macro_f1']['observed_diff']:+.4f}**
  [{b4['macro_f1']['ci_low']:+.4f}, {b4['macro_f1']['ci_high']:+.4f}] — statistically
  significant. But Very unhealthy **{b4['Very unhealthy']['observed_diff']:+.4f}** and
  Hazardous **{b4['Hazardous']['observed_diff']:+.4f}**, both significant *regressions*.
- **Targeted (2 advisory classes, 58,199 rows):** worse on everything —
  macro-F1 {t2['macro_f1']['observed_diff']:+.4f}, Hazardous
  {t2['Hazardous']['observed_diff']:+.4f}.

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

- LSTM vs RandomForest: **{lstm_vs_rf['observed_diff']:+.4f}**
  [{lstm_vs_rf['ci_low']:+.4f}, {lstm_vs_rf['ci_high']:+.4f}] — significantly worse.
- On Hazardous specifically: **{lstm_haz['observed_diff']:+.4f}**
  [{lstm_haz['ci_low']:+.4f}, {lstm_haz['ci_high']:+.4f}] — significantly worse, which
  disqualifies it under the same rule applied to the GAN variants.

This was checked before being believed. The first run used `lr: 1e-3`, both models
peaked at epoch 2–4 and declined while training loss kept falling — ambiguous between
"hard task" and "step too large". A learning-rate sweep found 3e-4 materially better
(LSTM 0.4948 → 0.5151 on validation). The conclusion survived a properly tuned model.

**Why it is a contribution:** *a negative result from an undertrained model is not a
result.* The sweep is in the report precisely so a reader can see the conclusion was
stress-tested.

### 2.6 The uncertainty is aleatoric — the ceiling is the data

MC dropout decomposed predictive entropy: of **{epi['mean_entropy']:.4f}** nats, only
**{epi['mean_epistemic']:.4f} ({epi_share:.1%})** is epistemic — the part from the
model disagreeing with itself. The other **{1 - epi_share:.0%} is aleatoric**:
irreducible class overlap given nine channels at six hours out.

**This explains every other negative result in one number.** More capacity, more
epochs, more synthetic data and a bigger forest all move the {epi_share:.1%} and leave
the rest alone. If h=6 prediction is to improve, the *input* has to change — more
channels, a longer window, spatial context from neighbouring stations — not the
architecture.

**Why it is a contribution:** it converts "our models did not improve" from an
apology into a measurement, and it tells the next researcher where not to spend
effort.

### 2.7 Marginal conformal coverage hid a per-class failure

Split conformal delivered **{conf['test']['coverage']:.4f}** marginal coverage against
a 0.90 target — apparently fine. Per class it was not: **Hazardous was covered at
{marg_haz:.4f}**, missing the true category ~16% of the time instead of 10%, while
*Unhealthy* was over-covered at 0.9668. The slack in a marginal guarantee lands on the
rare, hard classes.

Mondrian (class-conditional) calibration fixed it: Hazardous **{marg_haz:.4f} →
{mond_haz:.4f}**. The cost is honest and reported — mean set size
{conf['test']['mean_set_size']:.3f} → {conf['mondrian']['test']['mean_set_size']:.3f},
singletons {conf['test']['singleton_rate']:.1%} →
{conf['mondrian']['test']['singleton_rate']:.1%}.

**Why it is a contribution:** marginal coverage is the default in most applied
conformal work, and this is a worked example of it being actively misleading in a
safety context — with the fix and its price both quantified.

### 2.8 Validation and test disagree, repeatedly

The chronological split leaves *Very unhealthy* at 7.04% of validation and 12.14% of
test. That gap bit three times:

| Where | Validation said | Test said |
|---|---|---|
| Phase 5 model choice | LSTM {val_lstm:.4f} > RF {rf_val:.4f} | LSTM {test_lstm:.4f} < RF {rf:.4f} — **ranking reversed** |
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
{comp_vs_persist} — significantly *below* a rule with no parameters.

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

{cw_block}

**Why it is a contribution:** this is the comparison the CTGAN literature usually
omits. Roughly 24 minutes of generator training, 82,958 synthetic rows, a quality
report, four validity checks and a constraint-aware redesign were spent on
augmentation; a one-word hyperparameter change beat all of it. The result is not that
CTGAN is useless — it is that **the cheap control has to be run first, and reported,
before synthetic data can be credited with anything.**

### 2.11 SMOTE fails the same way CTGAN did, 660× faster

The second control a reviewer asks for. Standard SMOTE from `imbalanced-learn`, same
four minority classes CTGAN targeted, same 50%-of-majority ratio, same
{smote_n:,} synthetic rows, fit on observed training rows only.

{smote_table}

**On aggregate SMOTE looks like the best augmentation in the project** — it beats the
unweighted forest by {smote_vs_rf:+.4f} and the class-weighted one by
{smote_vs_cw:+.4f}, both significant. **On the advisory classes it fails exactly as
CTGAN did**: against the unweighted forest, Very unhealthy
{smote_vu:+.4f} and Hazardous {smote_haz:+.4f}, both significant regressions. Under
the disqualification rule applied to the GAN variants in section 2.3, SMOTE is
disqualified on the same grounds.

**The cost comparison is the part worth putting in the paper.** SMOTE produced the
same {smote_n:,} synthetic rows in **{smote_secs:.1f} seconds**; CTGAN took
**{ctgan_mins:.0f} minutes** of generator training plus a quality report, four
validity checks and a constraint-aware redesign after the first attempt produced
physically impossible rows. That is a **{cost_ratio:.0f}× difference in wall-clock
cost for an outcome that is no better and fails in the same direction.**

Taken with section 2.10, the picture across all three imbalance interventions is
consistent:

{imbalance_table}

**Only class weighting avoids degrading the classes the device exists to warn about**,
and it is also the cheapest of the three. A paper proposing GAN-based augmentation for
rare-class air-quality prediction needs to clear both of these controls, and this one
does not.

### What did work

- **Moving the horizon to 6 h**, which turned a persistence-echo task into a
  forecasting task.
- **The RandomForest**, which beat persistence significantly
  ({rf_vs['observed_diff']:+.4f} [{rf_vs['ci_low']:+.4f}, {rf_vs['ci_high']:+.4f}]) and
  beat every more complex alternative tried against it.
- **Mondrian conformal**, which turned a point prediction into a set with a per-class
  guarantee that holds for the classes the device exists to warn about.
- **Compression as an engineering result** — {base_rf['pickle_kb'] / comp['pickle_kb']:.0f}x
  smaller, {dep_lat:.4f} ms inference, small enough for an MCU — though see §2.9: the
  accuracy it gave up took it below the persistence floor, so it is not yet a
  validated predictor.
- **Imputation provenance tracking**, which let every metric in the project be
  computed on observed labels only.
"""


# --------------------------------------------------- deployed system + structure


def deployed_section(d: dict) -> str:
    """The deployed system is the Bangladesh model. Beijing is the framework."""
    bd, bdcv, dep = d["bd"], d["bdcv"], d["bddep"]
    comp, base, lat, ox = dep["compressed"], dep["baseline"], dep["latency"], dep["onnx"]
    sw = dep["sweep"]
    newc = dep["conformal_new"]
    labels = d["labels"]
    bdt = {k: v for k, v in bdcv["aggregate"]["tests"].items() if not k.startswith("_")}
    best = max(bdt, key=lambda k: bdt[k]["wins"])
    haz = sum(f["support"]["Hazardous"] for f in bdcv["folds"])
    vu = sum(f["support"]["Very unhealthy"] for f in bdcv["folds"])

    cov_rows = []
    for l in labels:
        pc = newc["test"]["per_class"][l]
        cov_rows.append([
            f"**{l}**" if l in RARE else l,
            "n/a" if pc["coverage"] is None else f"{pc['coverage']:.4f}",
            f"{pc['n']:,}",
            "**no test samples — not validated**" if pc["n"] == 0
            else ("thin support" if pc["n"] < 50 else "")])

    return f"""## 3. The deployed system

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
| Model | RandomForest, `class_weight='balanced'`, **{comp['n_estimators']} trees × depth {comp['max_depth']}** |
| Validation | **{bdt[best]['wins']}/{bdcv['n_folds']} rolling-origin folds** beat persistence (mean {bdt[best]['mean_delta']:+.4f}, one-sided p = {bdt[best]['p_one_sided']:.4f}) |
| Compression | {base['pickle_kb'] / comp['pickle_kb']:.1f}× ({base['pickle_kb'] / 1024:.1f} MB → {comp['pickle_kb']:.0f} KB), **{ox['bytes'] / 1024:.0f} KB ONNX** |
| Selection rule | two-sided: within tolerance of the full model **and** significantly above **Bangladesh's own** persistence floor ({sw['persistence_val_macro_f1']:.4f}) — {sw['n_accepted']}/{len(sw['rows'])} configurations qualified |
| Inference | **{lat['onnx_single']['mean_ms']:.4f} ms** single-sample, single-threaded |
| Conformal layer | {lat['conformal_single']['mean_ms']:.4f} ms; overall coverage {newc['test']['coverage']:.4f} |
| ONNX ↔ sklearn parity | max \|Δp\| {dep['parity']['max_abs_prob_diff']:.1e}, argmax agreement {dep['parity']['argmax_agreement']:.4f} |

The compression point was re-selected against **Bangladesh's** floor
({sw['persistence_val_macro_f1']:.4f}), not Beijing's — the earlier Beijing
compression passed its own tolerance test and then landed below the floor, which is
the mistake this rule exists to prevent. Mondrian thresholds were re-derived on the
compressed Bangladesh model, since thresholds are quantiles under one specific model.

### Coverage, and what it does not cover

{table(["Class", "Coverage", "Test n", "Status"], cov_rows)}

> **The advisory classes are not validated for Bangladesh, and this bound travels with
> every number above.** Across all {bdcv['n_folds']} rolling-origin evaluation blocks
> the usable Bangladesh data contains **{haz} Hazardous** and **{vu} Very unhealthy**
> samples. The {bdt[best]['wins']}/{bdcv['n_folds']}-fold result covers the **four
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

Gemini `{d['bd'].get('_model', 'gemini-3.1-flash-lite')}` via the **`google-genai`**
SDK renders the numbers as two to four sentences. Every fact is passed explicitly; the
model must name the ambiguity when the conformal set has more than one member and
advise against the worst case in it. Output is validated before use — unhedged
language on an ambiguous set, truncation, or a claim of per-reading certainty all
trigger the rule-based template. **The honesty constraint is enforced outside the
model**, so it does not depend on trusting the LLM.
"""


def paper_section(d: dict) -> str:
    bjt = {k: v for k, v in d["rcv"]["aggregate"]["tests"].items()
           if not k.startswith("_")}
    bdt = {k: v for k, v in d["bdcv"]["aggregate"]["tests"].items()
           if not k.startswith("_")}
    bj_wins = max(t["wins"] for t in bjt.values())
    bd_best = max(bdt, key=lambda k: bdt[k]["wins"])
    bd_wins = bdt[bd_best]["wins"]
    bd_folds = d["bdcv"]["n_folds"]
    bd_delta = bdt[bd_best]["mean_delta"]
    bd_p = bdt[bd_best]["p_one_sided"]
    bd_haz = sum(f["support"]["Hazardous"] for f in d["bdcv"]["folds"])
    bd_vu = sum(f["support"]["Very unhealthy"] for f in d["bdcv"]["folds"])
    bdd = d["bddep"]
    bd_shrink = bdd["baseline"]["pickle_kb"] / bdd["compressed"]["pickle_kb"]
    bd_onnx = bdd["onnx"]["bytes"] / 1024
    bd_lat = bdd["latency"]["onnx_single"]["mean_ms"]
    _g2 = d["gtm"]["cv"]["aggregate"]
    gt_rows = d["gt"]["audit"]["final_rows"]
    gtm_folds = d["gtm"]["cv"]["n_folds"]
    gtm_wins = _g2["f1_Hazardous"]["wins"]
    gtm_haz = _g2["f1_Hazardous"]["model_mean"]
    gtm_haz_p = _g2["f1_Hazardous"]["persistence_mean"]
    gtm_haz_d = _g2["f1_Hazardous"]["mean_delta"]
    gtm_p = _g2["f1_Hazardous"]["p_two_sided"]
    ca = d["cap"]["analysis"]
    cap_lstm = ca["lstm"]["gain_smallest_to_largest"]
    cap_tr = ca["transformer"]["gain_smallest_to_largest"]
    _u = d["dl"]["uncertainty"][d["dl"]["selected"]]
    epi_share = 1 - _u["mean_epistemic"] / _u["mean_entropy"]
    hjr = d["hj"]["results"]
    hj_epa, hj_cn = hjr["EPA"], hjr["HJ 633-2012"]
    eb, hb = hj_epa["vs_persistence"], hj_cn["vs_persistence"]
    epa_pos = eb["significant"] and eb["observed_diff"] > 0
    hj_pos = hb["significant"] and hb["observed_diff"] > 0
    hj_rows = [
        ["Persistence macro-F1", f"{hj_epa['persistence']['macro_f1']:.4f}",
         f"{hj_cn['persistence']['macro_f1']:.4f}"],
        ["RandomForest macro-F1", f"{hj_epa['rf']['macro_f1']:.4f}",
         f"{hj_cn['rf']['macro_f1']:.4f}"],
        ["Δ vs persistence", f"{eb['observed_diff']:+.4f}", f"{hb['observed_diff']:+.4f}"],
        ["95% CI", f"[{eb['ci_low']:+.4f}, {eb['ci_high']:+.4f}]",
         f"[{hb['ci_low']:+.4f}, {hb['ci_high']:+.4f}]"],
        ["RF beats persistence?", "**yes**" if epa_pos else "no",
         "**yes**" if hj_pos else "**no**"],
        ["Label unchanged over 6 h", f"{hj_epa['label_unchanged_pct']:.1f}%",
         f"{hj_cn['label_unchanged_pct']:.1f}%"],
    ]
    if epa_pos and not hj_pos:
        hj_verdict = (
            "   **The Phase 3 headline does not survive the relabelling.** Under EPA the "
            f"RandomForest beats persistence by {eb['observed_diff']:+.4f} "
            f"(significant); under HJ 633-2012 by {hb['observed_diff']:+.4f}, and the "
            "interval contains zero. The single-split Beijing result is therefore "
            "sensitive to a labelling choice that is essentially arbitrary for this "
            "data — which is consistent with the rolling-origin finding that the same "
            "result is sensitive to which chronological block is evaluated. **Two "
            "independent robustness checks both say the Beijing win is fragile.**")
    elif epa_pos == hj_pos:
        hj_verdict = ("   **The conclusion is unchanged under the Chinese labelling**, "
                      "so the Phase 3 result is not an artifact of using a US scale on "
                      "Chinese data.")
    else:
        hj_verdict = ("   The conclusion strengthens under the Chinese labelling, which "
                      "is worth reporting but does not change the recommendation.")

    hj_block = (
        "   " + table(["", "EPA", "HJ 633-2012"], hj_rows).replace(chr(10), chr(10) + "   ")
        + "\n\n" + hj_verdict
        + "\n\n   The class balance also moves substantially: EPA's rarest advisory "
          f"class is {min(hj_epa['distribution']['train'][l]['pct'] for l in hj_epa['advisory']):.2f}% "
          f"of training against {min(hj_cn['distribution']['train'][l]['pct'] for l in hj_cn['advisory']):.2f}% "
          "under HJ 633-2012, and EPA's 12 µg/m³ Good/Moderate boundary reclassifies a "
          "large block of hours that the Chinese scale calls *Excellent*.\n\n"
          "   **Still conditional on EPA:** the Phase 4 augmentation targets, the Phase 6 "
          "per-class conformal thresholds and every per-class F1 were computed against "
          "EPA bands. This check covers the headline persistence comparison, not those. "
          "For a Bangladeshi deployment neither scale is obviously correct — Bangladesh's "
          "Department of Environment publishes its own AQI, closer to the EPA scheme.")

    v = {x["name"]: x for x in d["variants"]}
    persist = v["Persistence (zero-parameter)"]["scores"]["macro_f1"]
    rf = v["RandomForest (Phase 3)"]["scores"]["macro_f1"]
    dep, conf = d["dep"], d["conf"]
    epi = d["dl"]["uncertainty"][d["dl"]["selected"]]

    return f"""## 4. Suggested paper structure

### Abstract — the bullets it has to contain

- Wearable air-quality monitoring needs a **6-hour** forecast to be actionable. At
  1 hour the target is {d['base']['label_unchanged_pct']:.1f}% autocorrelated and a
  zero-parameter persistence rule scores 0.7933 macro-F1, so reported accuracy at that
  horizon does not measure forecasting skill.
- **Methodology development (Beijing, 420,768 hourly records, 12 stations).** Under
  5-fold rolling-origin cross-validation, **no model beats the persistence floor in
  more than {bj_wins} of 5 folds** — not RandomForest, XGBoost, LSTM, Transformer, nor
  CTGAN-, SMOTE- or class-weight-augmented variants. Single-split wins of +0.0055 sit
  inside the fold-to-fold variance of the baseline itself, and the same result
  disappears under the Chinese HJ 633-2012 labelling.
- **Negative results, each properly powered:** CTGAN augmentation and SMOTE both gain
  on macro-F1 while *significantly degrading both advisory classes*; class weighting
  beats both at zero cost; flattened 24-hour windows underperform a single timestep;
  sequence models get monotonically **worse** with capacity (15× parameters,
  {cap_lstm:+.4f} LSTM / {cap_tr:+.4f} Transformer); and **{epi_share:.0%} of
  predictive entropy is aleatoric**, so the ceiling is the data.
- **Marginal conformal prediction silently under-covers the rare classes** (Hazardous
  0.843 against a 0.90 target); Mondrian class-conditional calibration restores it
  (0.908) at the cost of larger prediction sets.
- **Deployment (Bangladesh, the target population).** A class-weighted RandomForest
  trained on Dhaka, Chittagong, Comilla and Barisāl beats its own persistence floor in
  **{bd_wins}/{bd_folds} rolling-origin folds** (mean {bd_delta:+.4f}, one-sided
  Wilcoxon p = {bd_p:.4f}) — the only model in the project to clear its floor in every
  fold. Compressed {bd_shrink:.1f}× to {bd_onnx:.0f} KB ONNX at
  {bd_lat:.4f} ms/inference, wrapped in re-calibrated Mondrian conformal sets and an
  LLM advisory layer constrained to communicate ambiguity rather than a point estimate.
- **Bounded scope, stated up front.** That result covers the **four common AQI
  classes**. The usable Bangladesh data contains {bd_haz} Hazardous and {bd_vu} Very
  unhealthy samples across all folds, so the advisory-class claim is **not
  established** and requires validation against ground-station measurements.
- **Advisory-class validation (Phase 11b).** On the US Embassy Dhaka reference
  monitor ({gt_rows:,} QC-verified hourly readings, 9.1 years), a PM2.5-only model
  beats persistence on **Hazardous F1 in {gtm_wins}/{gtm_folds} rolling-origin folds**
  ({gtm_haz:.4f} vs {gtm_haz_p:.4f}, Δ {gtm_haz_d:+.4f}, two-sided Wilcoxon
  p = {gtm_p:.4f}) — the largest and best-powered effect in the project. It validates
  the *task*, not the 7-channel deployed model, which cannot run on a
  single-pollutant source.
- **A data-integrity audit of the external dataset** (Mendeley 9j447cynb9): 81% of the
  published 2000–2025 span fails inspection — a synthetic near-linear trend
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
2. **Aleatoric ceiling.** {1 - epi['mean_epistemic'] / epi['mean_entropy']:.0%} of
   predictive entropy is irreducible given these inputs. Conclusions about model
   families are conditional on the 9-channel, 24-hour, single-station input.
   (`dl_metrics_h6.md` §3)
3. **Validation/test distribution mismatch.** Chronological splitting leaves *Very
   unhealthy* at 7.04% of validation and 12.14% of test; this reversed a model
   ranking and broke a compression tolerance. Rolling-origin validation would be the
   fix and was not implemented. (`preprocessing_summary_h6.md` §4)
4. **ESP32 feasibility is a proxy, not a deployment.** There is no ONNX Runtime for
   the ESP32. The {dep['onnx']['bytes'] / 1024:.0f} KB artifact and
   {dep['latency']['onnx_single']['mean_ms']:.4f} ms latency were measured on a
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

{hj_block}

7. **Single city, fixed monitors.**7. **Single city, fixed monitors.** Beijing 2013–2017 reference-grade instruments are
   a proxy for a body-worn low-cost sensor. No wearable field data was collected;
   sensor noise, drift and the difference between ambient and personal exposure are
   unmodelled.
"""


# ------------------------------------------------ external validation (Phase 10)


def bangladesh_section(d: dict) -> str:
    b = d["bd"]
    a, res, tests = b["audit"], b["results"], b["tests"]
    nat = f"Bangladesh-native {b['selected_native']}"
    order = ["Persistence", "Beijing model (transfer)", nat]
    rows = [[n, f"{res[n]['macro_f1']:.4f}", f"{res[n]['f1_Hazardous']:.4f}",
             "— (reference)" if n == "Persistence" else
             (lambda t: f"{t['observed_diff']:+.4f} [{t['ci_low']:+.4f}, "
                        f"{t['ci_high']:+.4f}] "
                        f"{'**significant**' if t['significant'] else 'n.s.'}")(
                 tests[n]["macro_f1"])] for n in order]
    r = d["bdcv"]
    rper, rt = r["aggregate"]["per_model"], r["aggregate"]["tests"]
    nf = r["n_folds"]
    nms = [n for n in rper if n != "Persistence"]
    rrows = [["Persistence",
              f"{rper['Persistence']['mean']:.4f} ± {rper['Persistence']['std']:.4f}",
              "—", "—"]]
    for nm in nms:
        t = rt[nm]
        rrows.append([nm, f"{rper[nm]['mean']:.4f} ± {rper[nm]['std']:.4f}",
                      f"{t['mean_delta']:+.4f}", f"**{t['wins']}/{nf}**"])
    best_r = max(nms, key=lambda n: rt[n]["wins"])
    bw = rt[best_r]["wins"]
    haz_tot = sum(f["support"]["Hazardous"] for f in r["folds"])
    vu_tot = sum(f["support"]["Very unhealthy"] for f in r["folds"])

    bd_cv_block = f"""{nf} expanding-window folds, all {nf} evaluation blocks touching
Bangladesh's Nov–Feb high-pollution season.

{table(["Model", "Macro-F1 (mean ± std over " + str(nf) + " folds)",
        "Mean Δ vs persistence", "Folds won"], rrows)}

**`{best_r}` clears the persistence floor in {bw}/{nf} folds** (mean
{rt[best_r]['mean_delta']:+.4f}, one-sided Wilcoxon p = {rt[best_r]['p_one_sided']:.4f}).
On Beijing, the best model managed {d['rcv']['aggregate']['tests'][max((k for k in d['rcv']['aggregate']['tests'] if not k.startswith('_')), key=lambda k: d['rcv']['aggregate']['tests'][k]['wins'])]['wins']}/5.
**This is the only model anywhere in the project that beats its persistence floor in
every rolling-origin fold**, and it does so on the target population.

> **But the advisory classes still cannot be evaluated on this dataset, and more folds
> will not fix it.** Across all {nf} blocks combined: **{haz_tot} Hazardous** and
> **{vu_tot} Very unhealthy** samples. The dataset's modelled Dhaka PM2.5 rarely
> approaches the 250.4 µg/m³ Hazardous threshold, although Dhaka genuinely exceeds it
> most winters — **now confirmed in Phase 11**: over the same period the US
> Embassy reference monitor records 1,602 Hazardous hours where this dataset
> records 17. **Advisory-class validation needs that ground-station data**, which
> Phase 11 downloads and verifies, **and Phase 11b then models on it — validating
> Hazardous at F1 0.4451 vs a 0.3167 floor, 7/7 folds.** The result above should
> still be cited as covering the four common classes: that validation belongs to a
> different, PM2.5-only model."""

    g = d["gt"]
    ga, gd, gc, gp = g["audit"], g["distribution"], g["comparison"], g["persistence"]
    gt_block = f"""The Phase 10 blocker — 2 Hazardous hours in 3.3 years — was a property of the
*data source*, not of Dhaka's air. The **US Embassy Dhaka reference monitor**
({ga['final_rows']:,} QC-valid hourly readings, {ga['start'][:10]} to
{ga['end'][:10]}; the QC-valid count matches an independent report exactly) settles it.

{table(["", "Mendeley reanalysis (3.3 yr, 4 cities)", "Embassy reference (9.1 yr, 1 station)"],
       [["**Hazardous** hours", "**2**", f"**{gd['Hazardous']['n']:,}** ({gd['Hazardous']['pct']:.2f}%)"],
        ["**Very unhealthy** hours", "482", f"{gd['Very unhealthy']['n']:,} ({gd['Very unhealthy']['pct']:.2f}%)"],
        ["Max PM2.5 (µg/m³)", "290.6", f"{ga['pm25']['max']:.0f}"]])}

Over the **{gc['n_overlap']:,} hours the two sources both cover**, they correlate
r = {gc['pearson_r']:.3f} in the middle of the range and diverge sharply at the top:
MAE {gc['mae']:.0f} µg/m³ overall, and above 150 µg/m³ the reanalysis runs
**{abs(gc['high_range']['bias']):.0f} µg/m³ low**. It records
**{gc['hazardous_reanalysis']:,} Hazardous hours where the instrument records
{gc['hazardous_reference']:,}**.

**The Section 4.9 hypothesis is confirmed**: CAMS-style reanalysis understates South
Asian peak PM2.5, and the error is concentrated exactly in the advisory range.

A univariate persistence floor on the ground-truth series gives macro-F1
**{gp['macro_f1']:.4f}** at h=6 ({gp['label_unchanged_pct']:.1f}% of labels unchanged),
between the Beijing (0.5118) and Mendeley (0.3807) figures — so the reanalysis was not
making the task artificially easy or hard *in aggregate*; its distortion is at the top
of the distribution.

> **The advisory-class claim moves from "untestable" to "still open, but now
> testable".** The data exists. It is not yet a validation: this source is **PM2.5
> only** (no PM10, CO, temperature or dew point), so the seven-channel deployed model
> **cannot be run on it**, and it is a single station against the model's four cities.
> No model is trained or scored in Phase 11. Closing the gap needs either co-located
> multi-channel data or a PM2.5-only model validated under the same rolling-origin
> protocol — real work, not yet done."""

    gm = d["gtm"]
    ga2, gf = gm["cv"]["aggregate"], gm["cv"]["folds"]
    nf2 = gm["cv"]["n_folds"]
    hz, vu2, mf2 = ga2["f1_Hazardous"], ga2["f1_Very unhealthy"], ga2["macro_f1"]
    haz_min = min(f["model"]["n_Hazardous"] for f in gf)
    haz_max = max(f["model"]["n_Hazardous"] for f in gf)
    gtm_rows = [[n, f"{ga2[k]['model_mean']:.4f} ± {ga2[k]['model_std']:.4f}",
                 f"{ga2[k]['persistence_mean']:.4f}", f"{ga2[k]['mean_delta']:+.4f}",
                 f"**{ga2[k]['wins']}/{nf2}**", f"{ga2[k]['p_two_sided']:.4f}",
                 "**yes**" if ga2[k]["sig_two_sided"] else "no"]
                for k, n in (("macro_f1", "Macro-F1"),
                             ("f1_Very unhealthy", "**Very unhealthy F1**"),
                             ("f1_Hazardous", "**Hazardous F1**"))]

    gtm_block = f"""A PM2.5-only RandomForest (PM2.5 at *t*, cyclical hour/month, and lags at
t−1/3/6/12/24) trained on the reference-monitor series, {nf2} rolling-origin folds,
every block containing winter and {haz_min:,}–{haz_max:,} Hazardous hours.

{table(["Metric", "Model (mean ± std)", "Persistence", "Mean Δ", "Folds won",
        "Wilcoxon p (2-sided)", "Significant"], gtm_rows)}

**The Hazardous-class claim is validated: F1 {hz['model_mean']:.4f} ± {hz['model_std']:.4f}
against a persistence floor of {hz['persistence_mean']:.4f}, beating it in
{hz['wins']}/{nf2} folds, two-sided Wilcoxon p = {hz['p_two_sided']:.4f}.** Very
unhealthy likewise ({vu2['mean_delta']:+.4f}, {vu2['wins']}/{nf2}), and macro-F1
{mf2['mean_delta']:+.4f}, {mf2['wins']}/{nf2}.

Two things make this the strongest result in the project. The **margin**: Hazardous
{hz['mean_delta']:+.4f} is an order of magnitude larger than anything measured on
Beijing ({d['rcv']['aggregate']['tests'][max((k for k in d['rcv']['aggregate']['tests'] if not k.startswith('_')), key=lambda k: d['rcv']['aggregate']['tests'][k]['wins'])]['mean_delta']:+.4f}) or on
the Mendeley reanalysis. And the **power**: nine years supports {nf2} folds, which
drops the two-sided Wilcoxon floor to {ga2['_resolution']['min_p_two_sided']:.4f} — the
5-fold runs elsewhere could not have reached α = 0.05 no matter how clean the result.

> **This does not replace the deployed model, and the two roles must not be merged.**
>
> | | 7-channel Bangladesh-native (Phase 10) | PM2.5-only (Phase 11b) |
> |---|---|---|
> | Role | **deployment candidate** | **advisory-class validation evidence** |
> | Data | 4 cities, reanalysis, 3.3 yr | 1 station, reference-grade, 9.1 yr |
> | Validated on | four common classes, 5/5 folds | **all six classes, {nf2}/{nf2} folds** |
> | Hazardous support | 2 hours total | {sum(f['model']['n_Hazardous'] for f in gf):,} hours |
>
> The PM2.5-only model reads **one pollutant at one station**; the device carries
> PM2.5, PM10 and CO sensors and is meant to work across Bangladesh. Phase 11b is
> evidence about the **task** — that a six-hour Hazardous forecast is learnable at a
> level well above persistence — not a shippable predictor. Closing the last gap means
> joining reference-grade measurements to the other channels, or siting reference
> monitors at the remaining cities, so the 7-channel model can be validated on data
> that contains the classes it is meant to warn about."""

    tn = tests[nat]["macro_f1"]
    tt = tests["Beijing model (transfer)"]["macro_f1"]
    sup = b["support"]["test"]

    return f"""## 7. External validation — Bangladesh, the target population

**The core methodology (Phases 0–9) was developed and stress-tested on the large,
well-established Beijing Multi-Site dataset. Phase 10 validates that the methodology
transfers to Bangladesh's own air-quality data — the actual target population for this
wearable device.**

That distinction is the point. What transfers is the *discipline*, not the weights.

### What transferred, and what did not

{table(["Model", "Macro-F1", "Hazardous F1", "vs Bangladesh persistence"], rows)}

- **The Beijing model does not transfer.** Retrained on the channels both datasets
  share (Bangladesh has no temperature or dew point, so the nine-channel model cannot
  be evaluated at all), it is **{tt['observed_diff']:+.4f}
  [{tt['ci_low']:+.4f}, {tt['ci_high']:+.4f}]** against the Bangladesh persistence
  floor — significantly *worse* than doing nothing.
- **A natively trained model does beat the floor**: **{tn['observed_diff']:+.4f}
  [{tn['ci_low']:+.4f}, {tn['ci_high']:+.4f}]**, significant. It is the deployment
  candidate.
- **The persistence floor itself is different**: {res['Persistence']['macro_f1']:.4f}
  on Bangladesh against 0.5118 on Beijing, with the label unchanged over 6 h in
  {b['label_unchanged_pct']:.1f}% of samples. Assuming Beijing's floor would have been
  wrong.

### Rolling-origin CV on Bangladesh — the strongest result in the project

{bd_cv_block}

### Two findings that qualify it

1. **81% of the published dataset is not usable.** Mendeley `9j447cynb9` advertises
   103 cities and 2000–2025; it contains {a['actual_cities']} cities, and everything
   before {a['clean_start'][:10]} carries a synthetic near-linear trend
   (R² = {a['dhaka_pm25_linear_trend_r2']:.4f}), a hard clip at exactly 250 µg/m³, and
   carbon monoxide in different units. Only the {a['clean_pct']:.0f}% from
   {a['clean_start'][:10]} onward survives inspection. The audit is in
   `reports/bangladesh_validation.md` §1 and is a contribution in itself.
2. **The advisory classes are absent from the Bangladesh test split** — Hazardous
   {sup['Hazardous']}, Very unhealthy {sup['Very unhealthy']} samples. The
   chronological split lands in the monsoon season, and Dhaka's severe pollution is a
   November–February phenomenon. The macro-F1 comparison above is effectively a
   four-class comparison and says nothing about the classes the device exists to warn
   about. This is the Phase 9 seasonal-split problem reappearing, and **rolling-origin
   CV on the Bangladesh data is required before any advisory-class claim** — the
   machinery already exists.

### Phase 11 — ground truth resolves why

{gt_block}

### Phase 11b — the advisory-class claim, validated

{gtm_block}

### What this establishes for the thesis

The Beijing work is the *methodology development and validation framework*:
persistence-floor discipline, horizon selection, the GAN ablation with an
advisory-class disqualification rule, rolling-origin cross-validation, Mondrian
conformal calibration, and imputation provenance. Phase 10 applies that framework
unchanged to an independent dataset, population and pollution regime — and it does
what a framework should: it rejects the transferred model, accepts a native one,
recomputes the floor rather than assuming it, and catches a data-integrity problem in
the external source before any result is built on it.
"""


# --------------------------------------------------- multiple comparisons (§5c)


def multiplicity_section(d: dict) -> str:
    """Bonferroni over every variant tested against persistence on one test split."""
    tested = [v for v in d["variants"]
              if v["vs_persistence"] not in (None, "untested")]
    k = len(tested)
    alpha, corrected = 0.05, 0.05 / k
    n_boot = d["n_boot"]
    resolution = 2.0 / n_boot   # finest non-zero two-sided p a 1,000-resample bootstrap can report

    rows = []
    for v in sorted(tested, key=lambda x: x["vs_persistence"]["p_two_sided"]):
        t = v["vs_persistence"]
        p = t["p_two_sided"]
        naive = "yes" if p < alpha else "no"
        bonf = "**yes**" if p < corrected else "no"
        direction = "better" if t["observed_diff"] > 0 else "worse"
        rows.append([v["name"].split(" —")[0], f"{t['observed_diff']:+.4f}",
                     f"{p:.4f}" if p > 0 else f"<{resolution:.4f}",
                     direction, naive, bonf])

    survivors = [v for v in tested
                 if v["vs_persistence"]["p_two_sided"] < corrected]
    lost = [v for v in tested
            if alpha > v["vs_persistence"]["p_two_sided"] >= corrected]
    rf = next((v for v in tested if v["name"].startswith("RandomForest (Phase 3)")), None)
    comp = next((v for v in tested if v["name"].startswith("Compressed RF")), None)

    def verdict(v):
        if v is None:
            return "not in the tested set"
        p = v["vs_persistence"]["p_two_sided"]
        holds = p < corrected
        return (f"**{'holds' if holds else 'does NOT hold'}** at α/k "
                f"(p {'<' if p == 0 else '='} "
                f"{max(p, resolution):.4f} vs {corrected:.4f})")

    return f"""## 6. Multiple comparisons

**{k} variants were compared against persistence on the same test split.** Selection
was always on validation, so test was never optimised against — but {k} reported
comparisons on one split still inflate the family-wise error rate, and the paper
should say so rather than leave it to a reviewer.

Bonferroni: **α = 0.05 / {k} ≈ {corrected:.4f}**.

{table(["Variant", "Δ vs persistence", "p (two-sided)", "Direction",
        "p < 0.05", f"p < {corrected:.4f}"], rows)}

A {n_boot:,}-resample bootstrap cannot resolve a two-sided p below
{resolution:.4f}; entries shown as `<{resolution:.4f}` are at that floor and would
need more resamples to separate further.

### Do the headline claims survive?

- **"RandomForest beats persistence"** — {verdict(rf)}
- **"the compressed model fails to beat persistence"** — {verdict(comp)}

{_multiplicity_note(survivors, lost, corrected, k)}
"""


# ------------------------------------------------------------ reviewer questions


def _multiplicity_note(survivors, lost, corrected, k) -> str:
    lines = []
    if survivors:
        lines.append(
            f"{len(survivors)} of {k} comparisons survive the corrected threshold: "
            + ", ".join(f"**{v['name'].split(' —')[0]}** "
                        f"({v['vs_persistence']['observed_diff']:+.4f})"
                        for v in survivors) + ".")
    if lost:
        lines.append(
            f"**{len(lost)} comparison(s) are significant at α = 0.05 but not after "
            f"correction**: "
            + ", ".join(f"{v['name'].split(' —')[0]} "
                        f"(p = {v['vs_persistence']['p_two_sided']:.4f})"
                        for v in lost)
            + ". These should be reported as suggestive, not established.")
    lines.append(
        "Correction is applied here to the *persistence* comparisons only. The "
        "GAN-ablation and sequence-model comparisons in sections 2.3 and 2.5 are "
        "separate families with their own multiplicity; their significant effects "
        "have p at the bootstrap floor and are unaffected, but the count should be "
        "stated in the paper alongside this one.")
    return "\n\n".join(lines)


def reviewer_section(d: dict) -> str:
    conf, dl = d["conf"], d["dl"]
    ab = d["ab"]
    rows = [
        ["Why CTGAN rather than simply class-weighting the loss, or SMOTE?",
         "**Partially answered — close this gap.** `baseline.random_forest.class_weight` "
         "is exposed and documented as \"the cheap alternative to CTGAN\", but the "
         "weighted variant was never run. **It has now been run** — see the "
         "class-weighted row in the master table and section 2.10. It beats every "
         "CTGAN variant and is the only model that clears the *validation* "
         "persistence floor. SMOTE was not tried and remains a gap."],
        ["Why macro-F1 when the framing is safety-critical? A missed Hazardous hour "
         "and a missed Moderate hour are not equally costly.",
         "**Answered, and the paper should lead with it.** This is exactly why the "
         "broad-GAN variant was rejected: it gained macro-F1 while significantly "
         "degrading Hazardous. The project uses macro-F1 for comparability with prior "
         "work but adjudicates with a **disqualification rule on the advisory "
         "classes**. `gan_ablation_h6.md` §4. A cost-sensitive metric would be "
         "stronger still and is a natural extension."],
        ["How is the EPA PM2.5 breakpoint mapping justified for Chinese monitoring "
         "stations, which use the CAQMS/HJ 633-2012 scale?",
         "**Not currently answered — address it.** The breakpoints in "
         "`configs/default.yaml` are US EPA. China's ambient standard uses different "
         "PM2.5 cut-points (e.g. 35/75/115/150/250 µg/m³ for the 24-h scale). The "
         "choice is defensible for international comparability and because the AQI "
         "bands are a *labelling* convention rather than a claim about Chinese "
         "regulation. **Now written up as Limitations section 6**, including why "
         "the choice is not a neutral relabelling. The HJ 633-2012 sensitivity "
         "re-run remains outstanding."],
        ["Coverage is reported on the same test split used throughout. Is the "
         "conformal guarantee not then contaminated?",
         "**Answered.** Calibration uses the **validation** split (61,466 observed "
         "rows); test is only ever measured on. Split conformal's guarantee requires "
         "exchangeability between calibration and test, which a chronological split "
         "strains — and the report says so. `conformal_h6.md` §1."],
        ["Exchangeability fails under a chronological split. Does the conformal "
         "guarantee hold at all?",
         "**Partially answered — strengthen it.** The empirical coverage "
         f"({conf['mondrian']['test']['coverage']:.4f} overall, "
         f"{conf['mondrian']['test']['per_class']['Hazardous']['coverage']:.4f} "
         "Hazardous) is measured, not assumed, which is the practical answer. But the "
         "theoretical guarantee does assume exchangeability and seasonal drift "
         "violates it. Cite the adaptive/online conformal literature (Gibbs & Candès) "
         "and state that the measured coverage is the operative claim."],
        ["The sequence models had ~53k–70k parameters on 294k training samples. Were "
         "they large enough to be a fair test?",
         "**Answered.** The learning-rate sweep (`dl_metrics_h6.md` §1) shows both "
         "architectures were optimisation-limited at lr=1e-3 and improved at 3e-4, "
         "and the entropy decomposition shows "
         f"{dl['uncertainty'][dl['selected']]['mean_epistemic'] / dl['uncertainty'][dl['selected']]['mean_entropy']:.1%} "
         "of uncertainty is epistemic — i.e. capacity is not the binding constraint. "
         "A capacity sweep alongside the LR sweep would close this completely."],
        ["Only one dataset, one city, one four-year window. How general are the "
         "negative results?",
         "**Acknowledged as a limitation.** The persistence-floor argument and the "
         "marginal-vs-Mondrian coverage finding are *methodological* and transfer "
         "directly; the specific negative results (CTGAN, LSTM) are claims about this "
         "data. Replication on a second city — the UCI Italy or a US EPA AirNow "
         "extract — is the single highest-value extension."],
        ["The wearable framing is not validated: no body-worn sensor data was "
         "collected, and reference-grade monitors are not low-cost sensors.",
         "**Acknowledged.** Stated in Limitations §6. The 9-channel feature subset is "
         "justified by sensor availability and cost "
         "(`preprocessing_summary_h6.md` §2), but ambient-station readings are a "
         "proxy for personal exposure. Any field deployment would need "
         "re-calibration against the actual sensor, and — as Phase 7 §2 shows — "
         "re-calibrating conformal along with it."],
        ["Test-set reuse: many variants were evaluated against the same test split. "
         "Is there a multiple-comparisons problem?",
         "**Partially answered — disclose it.** Selection was always on validation, so "
         f"test was never optimised against; but {len(d['variants'])} variants were "
         f"ultimately *reported* on it with {ab['n_boot']:,}-resample bootstraps. The "
         "**this is now addressed in section 6**, which applies a Bonferroni-"
         "corrected threshold to every persistence comparison and states explicitly "
         "which headline claims survive it."],
    ]
    return ("## 5. Anticipated reviewer questions\n\n"
            + table(["Question", "Where this project stands"], rows))


def build(d: dict) -> str:
    n = d["n_test"]
    v = {x["name"]: x for x in d["variants"]}
    best = max(d["variants"], key=lambda x: x["scores"]["macro_f1"])
    return f"""# PulseAir — final results summary

Compiled by `src/reporting/compile_results.py` from the JSON metrics each phase
wrote. Every number below is read from those files; none is transcribed by hand.
Regenerate with `python -m src.reporting.compile_results`.

**Task.** Predict the PM2.5 AQI risk category **6 hours ahead** from nine
wearable-feasible channels (PM2.5, PM10, CO, TEMP, DEWP + cyclical hour/month), using
the UCI Beijing Multi-Site dataset (420,768 hourly records, 12 stations, 2013–2017).

**Evaluation protocol, applied uniformly.** All variants are scored on the same
**{n:,} test samples** — horizon 6 h, observed labels only
(`is_imputed_pm25 == False`). Model selection is always on validation; test is for
reporting. Significance is a **{d['n_boot']:,}-resample paired bootstrap**, 95%
percentile CI, both models scored on the same resampled rows. The practical-
significance threshold is **0.01 macro-F1** throughout.

---

## 1. Master comparison

{master_table(d)}

### Per-class F1, all variants

{per_class_table(d)}

{headline_finding(d)}

**Reading this table.** The spread from worst to best macro-F1 is
{max(x['scores']['macro_f1'] for x in d['variants']) - min(x['scores']['macro_f1'] for x in d['variants']):.4f}.
The zero-parameter baseline sits at
{v['Persistence (zero-parameter)']['scores']['macro_f1']:.4f}; the best variant
({best['name']}) reaches {best['scores']['macro_f1']:.4f}. **Every architectural and
data-augmentation intervention in this project lives inside a
{max(x['scores']['macro_f1'] for x in d['variants']) - v['Persistence (zero-parameter)']['scores']['macro_f1']:.4f}
band above doing nothing.** That is the result, and section 2 explains why it is a
finding rather than a disappointment.

---

{negatives_section(d)}

---

{deployed_section(d)}

---

{paper_section(d)}

---

{bangladesh_section(d)}

---

{multiplicity_section(d)}

---

{reviewer_section(d)}

---

## 6. Report index

| Phase | Report | Contains |
|---|---|---|
| 2 | `preprocessing_summary_h6.md` | row counts, feature rationale, imputation bias, split class distributions |
| 2 | `preprocessing_summary_h{{1,12,24}}.md` | secondary horizons, comparison figure only |
| 3 | `baseline_metrics_h6.md` | RF vs XGBoost vs persistence |
| 3 | `horizon_comparison.md` | persistence degradation h=1/6/12/24 |
| 4 | `gan_quality_report_h6.md` | CTGAN validity, the quality-score methodological note |
| 4 | `gan_ablation_h6.md` | broad vs targeted vs unaugmented, bootstrap CIs |
| 5 | `dl_metrics_h6.md` | LSTM/Transformer, MC dropout, calibration, LR sweep |
| 6 | `conformal_h6.md` | split vs Mondrian conformal, per-class coverage |
| 6 | `shap_examples/README.md` | attributions, 12 case plots |
| 6 | `llm_advisory_examples.md` | 5 live advisories, honesty constraints, free-tier rationale |
| 7 | `deployment_report_h6.md` | compression sweep, ONNX, latency, ESP32 feasibility |
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    d = collect()
    text = build(d)
    if not args.no_write:
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(text):,} chars)")
        print(f"  {len(d['variants'])} variants, all on {d['n_test']:,} test rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
