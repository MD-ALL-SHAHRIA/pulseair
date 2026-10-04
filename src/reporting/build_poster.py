"""Build the PulseAir research poster from the FST poster template.

The template (docs/template/Revised_Poster Presentation Template_FST 5.pptx, one
30 x 48 in slide) is copied and filled in: every text box keeps its own fonts, sizes,
colours and paragraph formats, which are cloned from the template's own paragraphs.
Numbers are read from reports/metrics/*.json through build_thesis.num(), so the poster
cannot disagree with the thesis.

The template has five visual slots; the poster uses three (the workflow schematic in
the Methods picture slot, one Data Analysis figure, one Results table). The template's
bar chart and pyramid diagram are removed, and the middle column's text boxes move up
into the space they leave.

    python -m src.reporting.build_poster          # -> docs/PulseAir_Poster.pptx
"""
from __future__ import annotations

import copy
import re
import shutil
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image

from src.reporting.build_thesis import AUTHORS, SRC, _audit_fractions, need, num

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "docs" / "template" / "Revised_Poster Presentation Template_FST 5.pptx"
OUT = ROOT / "docs" / "PulseAir_Poster.pptx"
FIGS = ROOT / "reports" / "figures"
EMU = 914400

NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
A = lambda t: f"{{{NS['a']}}}{t}"   # noqa: E731
P = lambda t: f"{{{NS['p']}}}{t}"   # noqa: E731

SUPERVISOR, CO_SUPERVISOR = "Mirza Asif Mahmud", "Dipta Justin Gomes"


# ------------------------------------------------------------------ content
def content() -> dict:
    f8 = "rolling_cv_h6_f8.json:aggregate.tests."
    bd = "rolling_cv_h6_bangladesh.json:aggregate.tests.RandomForest (class_weight=balanced)."
    hz = "dhaka_pm25_model_h6.json:cv.aggregate.f1_Hazardous."
    cc = "dhaka_monthly_crosscheck.json:stats."
    gt = "dhaka_ground_truth.json:comparison."
    rows_pct, span = _audit_fractions()
    c = {}
    c["title"] = "PulseAir: Wearable Air-Quality Risk Forecasting"
    c["subtitle"] = "Honest baselines, conformal uncertainty and external validation for Bangladesh"
    c["names"] = ", ".join(f"{a} (xx-xxxxx-x)" for a in AUTHORS)
    c["intro"] = (
        f"A wearable air-quality forecaster is only useful if it beats doing nothing. "
        f"One hour ahead the AQI category is unchanged in {num('horizon_comparison.json:rows.1.label_unchanged_pct', 'h1', '.1f')}% "
        f"of hours, so a no-model persistence rule scores "
        f"{num('horizon_comparison.json:rows.1.observed.macro_f1', 'h1 floor')} macro-F1. "
        f"PulseAir forecasts six hours ahead, where that floor falls to {num('horizon_comparison.json:rows.6.observed.macro_f1', 'h6 floor')}.")
    c["methods_lead"] = (
        "UCI Beijing Multi-Site data (12 stations, 2013–2017) for development; the "
        "Mendeley Bangladesh AQI data and the US Embassy Dhaka monitor for validation.")
    c["methods_h1"] = "EVALUATION DISCIPLINE"
    c["methods_b1"] = [
        "Every score is reported beside the persistence floor on the same rows.",
        "Rolling-origin CV with a 24-hour embargo, released as the PulseBench package.",
    ]
    c["methods_h2"] = "FIVE-SIGNATURE DATA AUDIT"
    c["methods_b2"] = [
        "Linear trend, hard clip at 250 µg/m³, CO unit change, one low-density city.",
        "Fifth check against DoE monthly reports; only the clean window is used.",
    ]
    c["workflow_caption"] = "Figure B: Study workflow, from data collection to the validated advisory."
    c["da_lead"] = (
        "The decisive audit check compares the dataset's discarded pre-2022 Dhaka series "
        "with the Department of Environment's own published monthly averages ")
    c["da_ref"] = "[See Figure A]."
    c["da_items"] = [
        f"Over a {need('dhaka_monthly_crosscheck.json:n_months', 'months')}-month overlap the "
        f"dataset runs {num(cc + 'bias_mendeley_minus_doe', 'bias', '+.0f')} µg/m³ high.",
        f"Real PM2.5 falls to {num(cc + 'monsoon_doe_mean', 'monsoon doe', '.0f')} µg/m³ "
        f"each monsoon; the fabricated series stays at "
        f"{num(cc + 'monsoon_mendeley_mean', 'monsoon men', '.0f')}.",
        f"Real yearly medians show no trend (R² {num(cc + 'r2_doe_overlap', 'r2 doe', '.2f')}); "
        f"fabricated ones form a line (R² {num(cc + 'r2_mendeley_overlap', 'r2 men', '.2f')}).",
    ]
    c["fig_caption"] = ("Figure A: Discarded pre-2022 Dhaka series (fabricated) against the DoE's "
                        "published monthly CAMS average (real).")
    c["table_title"] = "Table 1: Headline results vs the persistence floor"
    c["table_head"] = ["Evaluation", "Model", "Folds won", "Key number"]
    c["table_rows"] = [
        ["Beijing, 8 folds", "RF, class-weighted",
         f"{num(f8 + 'RandomForest (class_weight=balanced).wins', 'f8 wins', '.0f')} / "
         f"{need('rolling_cv_h6_f8.json:n_folds', 'f8 n')}",
         f"XGBoost worse (p = {num(f8 + 'XGBoost.p_two_sided', 'xgb p')})"],
        ["Bangladesh, 5 folds", "RF, class-weighted",
         f"{num(bd + 'wins', 'bd wins', '.0f')} / {need('rolling_cv_h6_bangladesh.json:n_folds', 'bd n')}",
         f"{num(bd + 'mean_delta', 'bd delta', '+.4f')} macro-F1"],
        ["Dhaka reference, 7 folds", "PM2.5-only detector",
         f"{num(hz + 'wins', 'hz wins', '.0f')} / {need('dhaka_pm25_model_h6.json:cv.n_folds', 'hz n')}",
         f"Hazardous F1 {num(hz + 'model_mean', 'hz f1')} vs {num(hz + 'persistence_mean', 'hz floor')}"],
        ["Bangladesh data audit", "5-signature audit", "—",
         f"{span}% of span fabricated"],
    ]
    c["table_note"] = ("*Folds won against persistence is the primary statistic. Data: "
                       "reports/metrics/ in the PulseAir repository.")
    c["results"] = [
        "The same protocol gives opposite verdicts on the two datasets. On Beijing no model "
        "beats persistence in a majority of eight rolling-origin folds, and both CTGAN and "
        "SMOTE augmentation were rejected for degrading the hazardous classes.",
        f"On the clean Bangladesh window a class-weighted Random Forest wins every fold, and "
        f"on the reference monitor a PM2.5-only detector finds hazardous air above its "
        f"floor in all {need('dhaka_pm25_model_h6.json:cv.n_folds', 'hz n')} folds (Table 1).",
    ]
    c["sub_head"] = "Ground Truth Moves the Limitation"
    c["sub_body"] = (
        f"Against the US Embassy monitor the dataset records "
        f"{num(gt + 'hazardous_reanalysis', 'rea h', ',.0f')} Hazardous hours where the "
        f"instrument records {num(gt + 'hazardous_reference', 'ref h', ',.0f')}: the missing "
        f"hazardous air was a data problem, not Dhaka's (Figure D).")
    c["concl_lead"] = "Rigorous baselines changed this project's conclusions more than any model did."
    c["concl"] = [
        ("Baselines decide claims: ",
         "a forecaster's accuracy means nothing without the persistence floor beside it; on "
         "Beijing no model clears it consistently, and GAN augmentation hurt the classes "
         "that matter most."),
        ("Data must be audited: ",
         f"{span}% of a widely indexed Bangladesh dataset's span is fabricated; on clean and "
         f"reference data, models beat persistence in every fold."),
    ]
    c["refs"] = [
        "S. Chen, “Beijing Multi-Site Air Quality,” UCI Machine Learning Repository, 2017, doi: 10.24432/C5RK5G.",
        "K. Hasan et al., “Forecasting particulate matter in Dhaka megacity,” Environ. Pollut. Manag., vol. 1, pp. 235–247, 2024.",
        "L. J. Tashman, “Out-of-sample tests of forecasting accuracy,” Int. J. Forecast., vol. 16, no. 4, pp. 437–450, 2000.",
        "A. N. Angelopoulos and S. Bates, “Conformal prediction: A gentle introduction,” Found. Trends Mach. Learn., vol. 16, no. 4, pp. 494–591, 2023.",
        "Government of Bangladesh, DoE, Bangladesh National Air Quality Management Plan 2024–2030, 2024.",
    ]
    ab, sm, gp = SRC.j("ablation_h6.json"), SRC.j("smote_h6.json"), SRC.j("gapfill_h6.json")
    cw = gp["variants"]["RandomForest (class_weight=balanced)"]["scores"]
    def f1s(sc):
        return f"{sc['per_class']['Very unhealthy']:.4f} / {sc['per_class']['Hazardous']:.4f}"
    c["abl_head"] = "Augmentation Ablation"
    c["abl_body"] = (
        f"CTGAN and SMOTE raise macro-F1 but degrade both advisory classes, so the "
        f"protected-class rule rejects them. SMOTE matched CTGAN in "
        f"{sm['cost']['smote_seconds']:.1f} s instead of {sm['cost']['ctgan_minutes']:.0f} min.")
    c["abl_title"] = "Table 2: Class-imbalance interventions (Beijing)"
    c["abl_rows"] = [
        ["Intervention", "Macro-F1", "V. unhealthy / Hazardous F1", "Verdict"],
        ["None (Random Forest)", f"{ab['rows']['unaugmented']['scores']['macro_f1']:.4f}",
         f1s(ab["rows"]["unaugmented"]["scores"]), "Baseline"],
        ["CTGAN (4-class)", f"{ab['rows']['broad-4']['scores']['macro_f1']:.4f}",
         f1s(ab["rows"]["broad-4"]["scores"]), "Disqualified"],
        ["SMOTE", f"{sm['scores']['macro_f1']:.4f}", f1s(sm["scores"]), "Disqualified"],
        ["Class weighting", f"{cw['macro_f1']:.4f}", f1s(cw), "Not disqualified"],
    ]
    c["kn_head"] = "Key Numbers"
    c["cards"] = [
        (f"{num('horizon_comparison.json:rows.1.label_unchanged_pct', 'h1', '.1f')}%",
         "AQI unchanged one hour ahead"),
        (f"{num(f8 + 'RandomForest (class_weight=balanced).wins', 'f8 wins', '.0f')} / "
         f"{need('rolling_cv_h6_f8.json:n_folds', 'f8 n')}", "Beijing folds won (best model)"),
        (f"{num(bd + 'wins', 'bd wins', '.0f')} / {need('rolling_cv_h6_bangladesh.json:n_folds', 'bd n')}",
         "Bangladesh folds won"),
        (f"{num(hz + 'wins', 'hz wins', '.0f')} / {need('dhaka_pm25_model_h6.json:cv.n_folds', 'hz n')}",
         "Hazardous detector folds won"),
        (f"{span}%", "Advertised span fabricated"),
        (f"{num('deployment_h6_bd.json:compressed.pickle_kb', 'kb', ',.0f')} KB",
         "Compressed model, ONNX export"),
    ]
    c["figc_caption"] = ("Figure C: Folds won against persistence under the same protocol: "
                         "Beijing against Bangladesh.")
    c["figd_caption"] = ("Figure D: The Bangladesh dataset against the US Embassy reference monitor: "
                         "advisory-class hours and the PM2.5 distribution.")
    c["ack"] = (
        f"We thank our supervisor, {SUPERVISOR}, and co-supervisor, {CO_SUPERVISOR}, for "
        f"their guidance, and the Department of Computer Science, AIUB, for its support.")
    c["fige_caption"] = ("Figure E: Persistence floor (macro-F1) at 1, 6, 12 and 24 hours; the "
                         "highlighted bar is the six-hour horizon used.")
    c["figf_caption"] = ("Figure F: Hazardous-class F1 of the PM2.5-only detector against the "
                         "persistence floor in each rolling-origin fold.")
    c["figg_caption"] = (f"Figure G: Per-class conformal coverage; Mondrian calibration raises "
                         f"Hazardous coverage from "
                         f"{num('conformal_h6.json:test.per_class.Hazardous.coverage', 'marg haz', '.3f')} to "
                         f"{num('conformal_h6.json:mondrian.test.per_class.Hazardous.coverage', 'mond haz', '.3f')}.")
    c["figh_caption"] = ("Figure H: What each imbalance intervention does to macro-F1 and to the two "
                         "advisory classes.")
    dep = "deployment_h6_bd.json:"
    c["t3_title"] = "Table 3: The deployed Bangladesh predictor"
    c["t3_rows"] = [
        ["Component", "Configuration", "Measured"],
        ["Model", f"RF, class-weighted, {num(dep + 'compressed.n_estimators', 'trees', '.0f')} trees "
                  f"x depth {num(dep + 'compressed.max_depth', 'depth', '.0f')}",
         f"{num(dep + 'compressed.pickle_kb', 'kb', ',.0f')} KB; ONNX "
         f"{num(dep + 'onnx.bytes', 'onnx', ',.0f')} bytes"],
        ["Inference", "ONNX Runtime, one sample (workstation)",
         f"{num(dep + 'latency.onnx_single.mean_ms', 'lat', '.4f')} ms"],
        ["Uncertainty", "Mondrian conformal, 90% target",
         f"Coverage {num(dep + 'conformal_new.test.coverage', 'cov')}, mean set "
         f"{num(dep + 'conformal_new.test.mean_set_size', 'set', '.2f')}"],
        ["Export check", "ONNX against scikit-learn",
         f"Argmax agreement {num(dep + 'parity.argmax_agreement', 'parity')}"],
    ]
    return c


# ------------------------------------------------------------------ xml helpers
def shape(tree, name):
    hits = tree.xpath(f'//p:cNvPr[@name="{name}"]', namespaces=NS)
    assert hits, name
    e = hits[0]
    while e.tag not in (P("sp"), P("graphicFrame"), P("pic"), P("cxnSp")):
        e = e.getparent()
    return e


def paras(sp):
    return sp.findall(f".//{A('p')}")


def make_par(proto, runs):
    """Clone a template paragraph (keeps pPr) with new runs. runs: [(text, bold|None)]."""
    p = copy.deepcopy(proto)
    rproto = next((r for r in p.findall(A("r"))), None)
    end = p.find(A("endParaRPr"))
    for x in list(p):
        if x.tag in (A("r"), A("br"), A("fld"), A("endParaRPr")):
            p.remove(x)
    for text, bold in runs:
        r = copy.deepcopy(rproto)
        rpr = r.find(A("rPr"))
        rpr.attrib.pop("err", None)
        if bold is True:
            rpr.set("b", "1")
        elif bold is False:
            rpr.attrib.pop("b", None)
        r.find(A("t")).text = text
        p.append(r)
    if end is not None:
        p.append(end)
    return p


def fill(sp, plist):
    body = sp.find(f".//{P('txBody')}")
    for p in body.findall(A("p")):
        body.remove(p)
    for p in plist:
        body.append(p)


def move(sp, y=None, x=None, cx=None, cy=None):
    xfrm = sp.find(f".//{A('xfrm')}")
    if xfrm is None:
        xfrm = sp.find(f".//{P('xfrm')}")
    off, ext = xfrm.find(A("off")), xfrm.find(A("ext"))
    for el, key, v in ((off, "y", y), (off, "x", x), (ext, "cx", cx), (ext, "cy", cy)):
        if v is not None:
            el.set(key, str(int(v * EMU)))


def add_picture(tree, logo, rid, name, sid, x, y, w, h):
    pic = copy.deepcopy(logo)
    c = pic.find(f".//{P('cNvPr')}")
    c.set("id", str(sid)); c.set("name", name); c.set("descr", name)
    for ext in c.findall(A("extLst")):
        c.remove(ext)
    pic.find(f".//{A('blip')}").set(f"{{{NS['r']}}}embed", rid)
    off = pic.find(f".//{A('off')}"); ext = pic.find(f".//{A('xfrm')}/{A('ext')}")
    off.set("x", str(int(x * EMU))); off.set("y", str(int(y * EMU)))
    ext.set("cx", str(int(w * EMU))); ext.set("cy", str(int(h * EMU)))
    sppr = pic.find(P("spPr"))
    for x_ in list(sppr):
        if x_.tag not in (A("xfrm"), A("prstGeom")):
            sppr.remove(x_)
    logo.getparent().append(pic)


def fit(png, max_w, max_h):
    with Image.open(png) as im:
        ar = im.height / im.width
    w = min(max_w, max_h / ar)
    return w, w * ar


# ------------------------------------------------------------------ build
LAYOUT = {
    # left column (x 0.98, w 8.96)
    "wf_y": 20.45, "wf_h": 4.5, "da_y": 26.05, "figa_y": 33.05, "fige_y": 38.35,
    # middle column (x 10.6)
    "table_cy": 4.6, "note_y": 9.55, "mid_line_y": 10.58, "results_y": 11.15,
    "figc_y": 18.45, "figg_y": 23.4, "sub_y": 28.5, "concl_y": 31.75, "figf_y": 38.75,
    # right column (x 20.4)
    "arch_y": 4.51, "arch_h": 5.75, "right_line_y": 11.2, "ack_y": 11.45, "right_line2_y": 14.4,
    "kn_y": 14.65, "cards_y": 15.65, "card_h": 1.6, "abl_y": 21.4, "abl_table_y": 24.1,
    "abl_table_cy": 4.0, "figh_y": 28.45, "figd_y": 33.0, "t3_y": 37.6, "t3_cy": 4.0,
    "bottom_y": 42.85, "refs_y": 42.95,
}
COLS = {"L": (0.98, 8.96), "M": (10.6, 8.96), "R": (20.4, 8.87)}


def build(layout: dict | None = None) -> Path:
    L = dict(LAYOUT, **(layout or {}))
    c = content()
    work = OUT.with_suffix(".work")
    shutil.rmtree(work, ignore_errors=True)
    with zipfile.ZipFile(TEMPLATE) as z:
        z.extractall(work)
    slide = work / "ppt" / "slides" / "slide1.xml"
    tree = etree.parse(str(slide))
    media = work / "ppt" / "media"
    ids = iter(range(3000, 3400))
    new_rels = []

    pristine_table = copy.deepcopy(shape(tree, "Table 1026"))
    pristine_ack = copy.deepcopy(shape(tree, "TextBox 1038"))
    pristine_line = copy.deepcopy(shape(tree, "Straight Connector 1040"))
    pristine_cap = copy.deepcopy(shape(tree, "TextBox 53"))
    band = copy.deepcopy(shape(tree, "Rectangle 14").find(f"{P('spPr')}/{A('solidFill')}"))
    logo = shape(tree, "Picture 2")
    spt = logo.getparent()

    # ---- helpers ---------------------------------------------------------------------
    def caption(text, x, y, w):
        e = copy.deepcopy(pristine_cap)
        cn = e.find(f".//{P('cNvPr')}"); cn.set("id", str(next(ids))); cn.set("name", "Caption " + text[:8])
        fill(e, [make_par(paras(pristine_cap)[0], [(text, None)])])
        move(e, x=x, y=y, cx=w)
        spt.append(e)

    def figure(png, col, y, max_h, text):
        x, w0 = COLS[col]
        rid = f"rIdP{len(new_rels) + 1}"
        name = f"poster_{Path(png).stem}.png"
        shutil.copy(FIGS / png, media / name)
        new_rels.append((rid, name))
        w, h = fit(media / name, w0, max_h)
        add_picture(tree, logo, rid, text.split(":")[0], next(ids), x + (w0 - w) / 2, y, w, h)
        caption(text, x, y + h + 0.08, w0)
        return y + h

    def section_box(head, body, col, y):
        e = copy.deepcopy(pristine_ack)
        cn = e.find(f".//{P('cNvPr')}"); cn.set("id", str(next(ids))); cn.set("name", "Section " + head)
        ps_ = paras(e)
        out_ = [make_par(ps_[0], [(head, None)])] + ([make_par(ps_[1], [(body, None)])] if body else [])
        fill(e, out_)
        move(e, x=COLS[col][0], y=y)
        spt.append(e)

    def line(col, y):
        e = copy.deepcopy(pristine_line)
        e.find(f".//{P('cNvPr')}").set("id", str(next(ids)))
        move(e, x=COLS[col][0] + 0.1, y=y)
        spt.append(e)

    def table(tf, title, rows_, widths, x, y, cy):
        """Fill a template table: title row + header row + data rows; fewer columns than
        the template's four are handled by dropping grid columns and merged cells."""
        n = len(widths)
        grid = tf.find(f".//{A('tblGrid')}")
        for gc in grid.findall(A("gridCol"))[n:]:
            grid.remove(gc)
        for gc, w_ in zip(grid.findall(A("gridCol")), widths):
            gc.set("w", str(int(w_ * EMU)))
        trs = tf.findall(f".//{A('tr')}")
        for tr in trs:
            for tc in tr.findall(A("tc"))[n:]:
                tr.remove(tc)
        first = trs[0].find(A("tc"))
        first.set("gridSpan", str(n))
        tp = first.find(f".//{A('p')}")
        tp.getparent().replace(tp, make_par(tp, [(title, True)]))
        for tr, vals in zip(trs[1:], rows_):
            for tc, v in zip(tr.findall(A("tc")), vals):
                p_ = tc.find(f".//{A('p')}")
                p_.getparent().replace(p_, make_par(p_, [(v, True if tr is trs[1] else None)]))
        for tr in trs[1 + len(rows_):]:
            tr.getparent().remove(tr)
        move(tf, x=x, y=y, cy=cy)

    # ---- header ------------------------------------------------------------------------
    t = shape(tree, "TextBox 7"); fill(t, [make_par(paras(t)[0], [(c["title"], None)])])
    for nm, key in (("TextBox 11", "subtitle"), ("TextBox 13", "names")):
        s_ = shape(tree, nm)
        fill(s_, [make_par(paras(s_)[0], [(c[key], None)])])
        move(s_, cx=24.0)

    # ---- left column -------------------------------------------------------------------
    s_ = shape(tree, "TextBox 3"); ps = paras(s_)
    fill(s_, [ps[0], make_par(ps[1], [(c["intro"], None)])])
    s_ = shape(tree, "TextBox 46"); ps = paras(s_)
    head, lead, sub, bul = ps[0], ps[1], ps[2], ps[4]
    out = [head, make_par(lead, [(c["methods_lead"], None)]), make_par(sub, [(c["methods_h1"], True)])]
    out += [make_par(bul, [(b, None)]) for b in c["methods_b1"]]
    out += [make_par(sub, [(c["methods_h2"], True)])] + [make_par(bul, [(b, None)]) for b in c["methods_b2"]]
    fill(s_, out)
    s_ = shape(tree, "TextBox 54"); ps = paras(s_)
    head, lead = ps[0], ps[1]
    items = [p_ for p_ in ps if p_.find(f".//{A('buAutoNum')}") is not None]
    lp = make_par(lead, [(c["da_lead"], None)])
    ref_run = next((r for r in lead.findall(A("r")) if "See Figure" in (r.findtext(A("t")) or "")), None)
    if ref_run is not None:
        rr = copy.deepcopy(ref_run); rr.find(A("t")).text = c["da_ref"]; lp.append(rr)
    fill(s_, [head, lp] + [make_par(items[0], [(t_, None)]) for t_ in c["da_items"]])
    move(s_, y=L["da_y"])

    # ---- middle column -----------------------------------------------------------------
    tf = shape(tree, "Table 1026")
    table(tf, c["table_title"], [c["table_head"]] + c["table_rows"], (2.55, 2.25, 1.35, 2.87),
          9760271 / EMU, 4134024 / EMU, L["table_cy"])   # the template's own table position
    s_ = shape(tree, "TextBox 1027"); fill(s_, [make_par(paras(s_)[0], [(c["table_note"], None)])])
    move(s_, y=L["note_y"])
    move(shape(tree, "Straight Connector 1029"), y=L["mid_line_y"])
    s_ = shape(tree, "TextBox 1030"); ps = paras(s_)
    fill(s_, [ps[0]] + [make_par(ps[1], [(t_, None)]) for t_ in c["results"]])
    move(s_, y=L["results_y"])
    s_ = shape(tree, "TextBox 1034"); ps = paras(s_)
    fill(s_, [make_par(ps[0], [(c["sub_head"], None)]), make_par(ps[1], [(c["sub_body"], None)])])
    move(s_, y=L["sub_y"])
    s_ = shape(tree, "TextBox 1035"); ps = paras(s_)
    fill(s_, [ps[0]] + [make_par(ps[2], [(lab, True), (txt, False)]) for lab, txt in c["concl"]])
    move(s_, y=L["concl_y"])

    # ---- right column ------------------------------------------------------------------
    s_ = shape(tree, "TextBox 1037"); ps = paras(s_)
    fill(s_, [ps[0]] + [make_par(ps[2], [(r_, None)]) for r_ in c["refs"]])
    move(s_, x=0.98, y=L["refs_y"], cx=28.3)          # references close the poster, full width
    sp_pr = s_.find(P("spPr"))
    for f_ in sp_pr.findall(A("solidFill")):
        sp_pr.replace(f_, etree.Element(A("noFill")))
    for r_ in s_.findall(f".//{A('p')}")[1:]:
        for rp in r_.iter(A("rPr")):
            rp.set("sz", "2000")
    move(shape(tree, "Straight Connector 1040"), y=L["right_line_y"])
    s_ = shape(tree, "TextBox 1038"); ps = paras(s_)
    fill(s_, [ps[0], make_par(ps[1], [(c["ack"], None)])])
    move(s_, y=L["ack_y"])

    # ---- drop the template's extra chart, diagram and placeholder ------------------------
    gone = []
    for nm in ("Chart 4", "Chart Placeholder 14", "Picture Placeholder 11", "Arc 1033",
               "TextBox 1032", "Picture Placeholder 7", "TextBox 53"):
        e = shape(tree, nm)
        gone += e.xpath(".//@r:id | .//@r:dm | .//@r:lo | .//@r:qs | .//@r:cs", namespaces=NS)
        e.getparent().remove(e)

    # ---- figures -----------------------------------------------------------------------
    figure("29_methodology_workflow.png", "L", L["wf_y"], L["wf_h"], c["workflow_caption"])
    figure("28_dhaka_monthly_fabrication.png", "L", L["figa_y"], 4.0, c["fig_caption"])
    figure("01_persistence_floor_vs_horizon.png", "L", L["fige_y"], 3.45, c["fige_caption"])
    figure("15_folds_won_summary.png", "M", L["figc_y"], 4.2, c["figc_caption"])
    figure("08_conformal_coverage.png", "M", L["figg_y"], 4.2, c["figg_caption"])
    figure("19_phase11b_hazardous.png", "M", L["figf_y"], 3.15, c["figf_caption"])
    figure("03_advisory_class_tradeoff.png", "R", L["figh_y"], 3.6, c["figh_caption"])
    figure("18_mendeley_vs_embassy.png", "R", L["figd_y"], 3.6, c["figd_caption"])
    figure("21_pipeline_overview.png", "R", L["arch_y"], L["arch_h"],
           "Figure I: Pipeline overview: the Beijing methodology track, the Bangladesh "
           "deployment track and the deployed advisory.")

    # ---- right column panels: key numbers, Table 2, Table 3 -------------------------------
    line("R", L["right_line2_y"])
    section_box(c["kn_head"], "", "R", L["kn_y"])
    cw_, ch_, gx, gy = 4.3, L["card_h"], 0.27, 0.25
    for k, (big, small) in enumerate(c["cards"]):
        r_, col = divmod(k, 2)
        x0, y0 = 20.5 + col * (cw_ + gx), L["cards_y"] + r_ * (ch_ + gy)
        card = etree.SubElement(spt, P("sp"))
        card.append(etree.fromstring(
            f'<p:nvSpPr xmlns:p="{NS["p"]}"><p:cNvPr id="{next(ids)}" name="Stat card {k + 1}"/>'
            f'<p:cNvSpPr/><p:nvPr/></p:nvSpPr>'))
        sppr = etree.fromstring(
            f'<p:spPr xmlns:p="{NS["p"]}" xmlns:a="{NS["a"]}"><a:xfrm><a:off x="{int(x0 * EMU)}" y="{int(y0 * EMU)}"/>'
            f'<a:ext cx="{int(cw_ * EMU)}" cy="{int(ch_ * EMU)}"/></a:xfrm>'
            f'<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val 10000"/></a:avLst></a:prstGeom></p:spPr>')
        sppr.append(copy.deepcopy(band))
        sppr.append(etree.fromstring(f'<a:ln xmlns:a="{NS["a"]}"><a:noFill/></a:ln>'))
        card.append(sppr)
        card.append(etree.fromstring(
            f'<p:txBody xmlns:p="{NS["p"]}" xmlns:a="{NS["a"]}"><a:bodyPr wrap="square" lIns="91440" rIns="91440" tIns="0" bIns="0" anchor="ctr"><a:normAutofit/></a:bodyPr><a:lstStyle/>'
            f'<a:p><a:pPr algn="ctr"/><a:r><a:rPr lang="en-US" sz="5400" b="1"><a:solidFill><a:schemeClr val="bg1"/></a:solidFill>'
            f'<a:latin typeface="Cambria"/><a:ea typeface="Cambria"/></a:rPr><a:t>{big}</a:t></a:r></a:p>'
            f'<a:p><a:pPr algn="ctr"/><a:r><a:rPr lang="en-US" sz="2000"><a:solidFill><a:schemeClr val="bg1"/></a:solidFill>'
            f'<a:latin typeface="Arial"/><a:ea typeface="Arial"/><a:cs typeface="Arial"/></a:rPr><a:t>{small}</a:t></a:r></a:p></p:txBody>'))
    section_box(c["abl_head"], c["abl_body"], "R", L["abl_y"])
    t2 = copy.deepcopy(pristine_table)
    t2.find(f".//{P('cNvPr')}").set("id", str(next(ids))); t2.find(f".//{P('cNvPr')}").set("name", "Table 2")
    table(t2, c["abl_title"], c["abl_rows"], (2.75, 1.5, 2.65, 1.97), 20.4, L["abl_table_y"], L["abl_table_cy"])
    spt.append(t2)
    t3 = copy.deepcopy(pristine_table)
    t3.find(f".//{P('cNvPr')}").set("id", str(next(ids))); t3.find(f".//{P('cNvPr')}").set("name", "Table 3")
    table(t3, c["t3_title"], c["t3_rows"], (1.9, 3.6, 3.37), 20.4, L["t3_y"], L["t3_cy"])
    spt.append(t3)

    # ---- bottom separators, one per column, just above the footer -----------------------
    for nm in ("Straight Connector 1023", "Straight Connector 1036"):
        move(shape(tree, nm), y=L["bottom_y"])
    b = copy.deepcopy(shape(tree, "Straight Connector 1036"))
    b.find(f".//{P('cNvPr')}").set("id", str(next(ids)))
    move(b, x=20.5)
    spt.append(b)
    tree.write(str(slide), xml_declaration=True, encoding="UTF-8", standalone=True)

    rels = work / "ppt" / "slides" / "_rels" / "slide1.xml.rels"
    rx = rels.read_text()
    for rid in set(gone):
        rx = re.sub(rf'<Relationship Id="{rid}"[^>]*/>', "", rx)
    rx = re.sub(r'<Relationship Id="rId\d+" Type="[^"]*diagram[^"]*"[^>]*/>', "", rx)
    img = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
    rx = rx.replace("</Relationships>", "".join(
        f'<Relationship Id="{rid}" Type="{img}" Target="../media/{name}"/>' for rid, name in new_rels)
        + "</Relationships>")
    rels.write_text(rx)
    return work


def pack(work: Path) -> Path:
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        ct = work / "[Content_Types].xml"
        z.write(ct, "[Content_Types].xml")
        for f in sorted(work.rglob("*")):
            if f.is_file() and f != ct:
                z.write(f, f.relative_to(work).as_posix())
    shutil.rmtree(work)
    return OUT


if __name__ == "__main__":
    import sys
    out = pack(build())
    missing = sorted(set(SRC.missing))
    print("wrote", out.relative_to(ROOT))
    print("MISSING:", missing) if missing else print("no MISSING values")
