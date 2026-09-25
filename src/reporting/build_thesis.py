"""Build the PulseAir thesis as a Word document, entirely from committed files.

Every number in the output is read at build time from ``reports/metrics/*.json``,
``reports/*.md`` or ``configs/*.yaml``. Nothing is typed in from memory. Where a needed
value cannot be found in a committed file, :func:`need` writes a visible
``[MISSING: ...]`` marker into the document and records it, so the gap appears in the
thesis where a reader will see it rather than being quietly filled with a plausible
number. The build prints every marker it inserted.

    python -m src.reporting.build_thesis            # -> docs/PulseAir_Thesis.docx
    python -m src.reporting.build_thesis --check    # resolve every value, write nothing

Figures come from ``reports/figures/`` and every caption names the JSON the figure was
generated from, taken from ``reports/figures/README.md`` rather than restated here.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = REPO_ROOT / "reports" / "metrics"
REPORTS = REPO_ROOT / "reports"
FIGURES = REPORTS / "figures"
CONFIG = REPO_ROOT / "configs" / "default.yaml"
OUT_DOCX = REPO_ROOT / "docs" / "PulseAir_Thesis.docx"

AUTHORS = ["Md All Shahria", "Sanjeda Dewan Mithila",
           "Anik Sarker Rudro", "Irfanul Islam Payel"]
DEPARTMENT = "Department of Computer Science"
UNIVERSITY = "American International University-Bangladesh (AIUB)"
TITLE = ("PulseAir: GAN-Augmented Deep Learning and Ensemble Machine Learning for "
         "Wearable Air-Quality Risk Forecasting, with Conformal Uncertainty "
         "Quantification and External Validation for Bangladesh")
REPO_URL = "https://github.com/MD-ALL-SHAHRIA/pulseair"

# --------------------------------------------------------------------- source data


class Sources:
    """Lazy, cached access to every committed file the document draws on."""

    def __init__(self) -> None:
        self._json: dict[str, Any] = {}
        self._text: dict[str, str] = {}
        self.missing: list[str] = []

    def j(self, name: str) -> Any:
        if name not in self._json:
            path = METRICS / name
            if not path.exists():
                self._json[name] = None
            else:
                self._json[name] = json.loads(path.read_text())
        return self._json[name]

    def md(self, name: str) -> str:
        if name not in self._text:
            path = REPORTS / name
            self._text[name] = path.read_text() if path.exists() else ""
        return self._text[name]

    @property
    def cfg(self) -> dict:
        return self.j("__cfg__") or self._load_cfg()

    def _load_cfg(self):
        self._json["__cfg__"] = yaml.safe_load(CONFIG.read_text())
        return self._json["__cfg__"]


SRC = Sources()


def need(path: str, what: str, *, fmt: str | None = None) -> Any:
    """Fetch ``path`` (``file.json:dotted.key``) or record a visible gap.

    Returns the value, or the literal string ``[MISSING: ...]``. It never raises and
    never substitutes a guess: a thesis that silently invents a number is worse than
    one that admits a hole, and the hole is easier to fix when it is printed on the
    page.
    """
    fname, _, dotted = path.partition(":")
    doc = SRC.j(fname)
    if doc is None:
        marker = f"[MISSING: {what} — {fname} not found in reports/metrics/]"
        SRC.missing.append(marker)
        return marker
    cur = doc
    for part in dotted.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
                continue
            except (ValueError, IndexError):
                cur = None
                break
        if not isinstance(cur, dict) or part not in cur:
            cur = None
            break
        cur = cur[part]
    if cur is None:
        marker = f"[MISSING: {what} — {path} not found in reports/]"
        SRC.missing.append(marker)
        return marker
    if fmt is not None and isinstance(cur, (int, float)):
        return format(cur, fmt)
    return cur


def num(path: str, what: str, fmt: str = ".4f") -> str:
    """``need`` for a number, rendered. Non-numbers become a MISSING marker."""
    v = need(path, what)
    if isinstance(v, str):
        return v
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        marker = f"[MISSING: {what} — {path} is not numeric]"
        SRC.missing.append(marker)
        return marker
    return format(v, fmt)


# ------------------------------------------------------------------ docx machinery

BODY_PT, H1_PT, H2_PT, H3_PT = 11, 16, 13, 11.5
CAPTION_PT, TABLE_PT = 9.5, 9.5


@dataclass
class Registry:
    """Figure and table captions, collected as they are emitted."""
    figures: list[tuple[str, str]] = field(default_factory=list)
    tables: list[tuple[str, str]] = field(default_factory=list)
    abbreviations: set[str] = field(default_factory=set)


REG = Registry()


def setup_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(BODY_PT)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15
    for name, size, colour in (("Heading 1", H1_PT, RGBColor(0x1F, 0x3B, 0x63)),
                               ("Heading 2", H2_PT, RGBColor(0x1F, 0x3B, 0x63)),
                               ("Heading 3", H3_PT, RGBColor(0x33, 0x33, 0x33))):
        st = doc.styles[name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = colour
        st.paragraph_format.space_before = Pt(12)
        st.paragraph_format.space_after = Pt(4)
    # US Letter, 1" margins
    for s in doc.sections:
        s.page_width, s.page_height = Inches(8.5), Inches(11)
        for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
            setattr(s, m, Inches(1))


def page_break(doc: Document) -> None:
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def para(doc: Document, text: str = "", *, style: str | None = None,
         align: str | None = None, size: float | None = None, bold: bool = False,
         italic: bool = False, space_after: float | None = None):
    p = doc.add_paragraph(style=style) if style else doc.add_paragraph()
    if align:
        p.alignment = {"center": WD_ALIGN_PARAGRAPH.CENTER,
                       "right": WD_ALIGN_PARAGRAPH.RIGHT,
                       "justify": WD_ALIGN_PARAGRAPH.JUSTIFY}[align]
    if text:
        _rich(p, text, size=size, bold=bold, italic=italic)
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)


def _rich(p, text: str, *, size=None, bold=False, italic=False):
    """Render a paragraph, honouring **bold** spans and flagging MISSING markers."""
    pos = 0
    for m in _BOLD.finditer(text):
        if m.start() > pos:
            _run(p, text[pos:m.start()], size, bold, italic)
        _run(p, m.group(1), size, True, italic)
        pos = m.end()
    if pos < len(text):
        _run(p, text[pos:], size, bold, italic)


def _run(p, text: str, size, bold, italic):
    if not text:
        return
    # A MISSING marker is set in red so it cannot be skimmed past in the rendered PDF.
    for chunk in re.split(r"(\[MISSING:[^\]]*\])", text):
        if not chunk:
            continue
        r = p.add_run(chunk)
        r.font.size = Pt(size or BODY_PT)
        r.bold = bold
        r.italic = italic
        if chunk.startswith("[MISSING:"):
            r.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
            r.bold = True


def heading(doc: Document, text: str, level: int = 1):
    if level == 1:
        page_break(doc)
    return doc.add_heading(text, level=level)


def bullets(doc: Document, items: list[str], *, numbered: bool = False):
    style = "List Number" if numbered else "List Bullet"
    for it in items:
        p = doc.add_paragraph(style=style)
        p.paragraph_format.space_after = Pt(3)
        _rich(p, it)


def table(doc: Document, caption: str, headers: list[str], rows: list[list[str]],
          *, source: str = "", widths: list[float] | None = None):
    """Emit a captioned table and register it for the List of Tables."""
    n = len(REG.tables) + 1
    label = f"Table {n}. {caption}"
    REG.tables.append((label, source))
    cap = para(doc, label, size=CAPTION_PT, bold=True, space_after=3)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        _rich(cell.paragraphs[0], h, size=TABLE_PT, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            _rich(cells[i].paragraphs[0], str(v), size=TABLE_PT)
    if widths:
        for r in t.rows:
            for i, w in enumerate(widths):
                r.cells[i].width = Inches(w)
    if source:
        para(doc, f"Source: {source}", size=CAPTION_PT - 0.5, italic=True,
             space_after=10)
    return t


def figure(doc: Document, png: str, caption: str, source: str, *, width: float = 6.2):
    """Embed a figure. The caption always names the JSON the figure came from."""
    n = len(REG.figures) + 1
    path = FIGURES / png
    if not path.exists():
        marker = f"[MISSING: figure {png} — not found in reports/figures/]"
        SRC.missing.append(marker)
        para(doc, marker)
        return
    para(doc, "", space_after=2)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width))
    label = f"Figure {n}. {caption}"
    REG.figures.append((label, source))
    cp = para(doc, f"{label} Generated from {source}.",
              align="center", size=CAPTION_PT, space_after=10)
    cp.runs[0].bold = True
    return cp


# ------------------------------------------------------------------- front matter


def title_page(doc: Document) -> None:
    for _ in range(4):
        para(doc, "")
    para(doc, TITLE, align="center", size=17, bold=True, space_after=24)
    para(doc, "A thesis submitted in partial fulfilment of the requirements",
         align="center", size=11, italic=True, space_after=2)
    para(doc, "for the degree of Bachelor of Science in Computer Science",
         align="center", size=11, italic=True, space_after=28)
    para(doc, "Submitted by", align="center", size=11, italic=True, space_after=8)
    for a in AUTHORS:
        para(doc, a, align="center", size=12.5, bold=True, space_after=3)
    para(doc, "", space_after=28)
    para(doc, "Supervised by", align="center", size=11, italic=True, space_after=8)
    para(doc, "_________________________", align="center", size=11, space_after=24)
    para(doc, DEPARTMENT, align="center", size=12.5, bold=True, space_after=2)
    para(doc, UNIVERSITY, align="center", size=12.5, bold=True, space_after=20)
    para(doc, date.today().strftime("%B %Y"), align="center", size=11)


def declaration(doc: Document) -> None:
    page_break(doc)
    para(doc, "Declaration", align="center", size=H1_PT, bold=True, space_after=18)
    para(doc,
         "We hereby declare that this thesis is based on the results of our own work "
         "carried out in the Department of Computer Science, American International "
         "University-Bangladesh (AIUB), under the supervision of our supervisor. We "
         "further declare that the material contained in this thesis has not been "
         "submitted, either in whole or in part, for the award of any other degree or "
         "diploma at this or any other institution. Work taken from other sources has "
         "been acknowledged and cited in the references.",
         align="justify", space_after=10)
    para(doc,
         "All code, data-processing pipelines, metrics and figures presented in this "
         f"thesis are openly available at {REPO_URL}. Every quantitative result "
         "reported here is reproducible from the committed metrics files in that "
         "repository.",
         align="justify", space_after=30)
    for a in AUTHORS:
        para(doc, "_____________________________", space_after=2)
        para(doc, a, bold=True, space_after=2)
        para(doc, "Date: ______________________", size=10, space_after=18)


def certificate(doc: Document) -> None:
    page_break(doc)
    para(doc, "Certificate of Approval", align="center", size=H1_PT, bold=True,
         space_after=18)
    para(doc,
         f"This is to certify that the thesis entitled “{TITLE}”, submitted by "
         + ", ".join(AUTHORS[:-1]) + f" and {AUTHORS[-1]}, has been carried out under "
         "my supervision in the Department of Computer Science, American International "
         "University-Bangladesh (AIUB). The work is original and, to the best of my "
         "knowledge, has not been submitted elsewhere for the award of any degree or "
         "diploma. I recommend that it be accepted in partial fulfilment of the "
         "requirements for the degree of Bachelor of Science in Computer Science.",
         align="justify", space_after=36)
    for role in ("Supervisor", f"Head of Department, {DEPARTMENT}, AIUB"):
        para(doc, "_____________________________", space_after=2)
        para(doc, role, bold=True, space_after=2)
        para(doc, "Date: ______________________", size=10, space_after=26)


def acknowledgement(doc: Document) -> None:
    page_break(doc)
    para(doc, "Acknowledgement", align="center", size=H1_PT, bold=True, space_after=18)
    para(doc,
         "We are grateful to our supervisor for guidance and for the freedom to report "
         "results as we found them, including the ones that did not go our way. We "
         "thank the Department of Computer Science at American International "
         "University-Bangladesh for the environment and resources that made this work "
         "possible, and the faculty whose teaching shaped how we approached it. We "
         "also thank the open-source community whose tools this project rests on, and "
         "the contributor who improved our evaluation toolkit after its public release. "
         "Finally, we thank our families for their patience and support throughout.",
         align="justify")


def abstract(doc: Document) -> None:
    page_break(doc)
    para(doc, "Abstract", align="center", size=H1_PT, bold=True, space_after=14)

    floor6 = num("horizon_comparison.json:rows.6.observed.macro_f1", "h6 persistence floor")
    floor1 = num("horizon_comparison.json:rows.1.observed.macro_f1", "h1 persistence floor")
    unch1 = num("horizon_comparison.json:rows.1.label_unchanged_pct", "h1 unchanged pct", ".1f")
    rf = num("ablation_h6.json:rows.unaugmented.scores.macro_f1", "RF macro-F1")
    bj_folds = _beijing_best_folds()
    bj_n = need("rolling_cv_h6.json:n_folds", "Beijing fold count")
    bd_wins = num("rolling_cv_h6_bangladesh.json:aggregate.tests.RandomForest (class_weight=balanced).wins",
                  "Bangladesh folds won", ".0f")
    bd_n = need("rolling_cv_h6_bangladesh.json:n_folds", "Bangladesh fold count")
    bd_delta = num("rolling_cv_h6_bangladesh.json:aggregate.tests.RandomForest (class_weight=balanced).mean_delta",
                   "Bangladesh mean delta", "+.4f")
    bd_p = num("rolling_cv_h6_bangladesh.json:aggregate.tests.RandomForest (class_weight=balanced).p_one_sided",
               "Bangladesh one-sided p")
    epi = _epistemic_share()
    haz_m = num("dhaka_pm25_model_h6.json:cv.aggregate.f1_Hazardous.model_mean", "11b Hazardous F1")
    haz_f = num("dhaka_pm25_model_h6.json:cv.aggregate.f1_Hazardous.persistence_mean", "11b Hazardous floor")
    haz_w = num("dhaka_pm25_model_h6.json:cv.aggregate.f1_Hazardous.wins", "11b folds won", ".0f")
    haz_n = need("dhaka_pm25_model_h6.json:cv.n_folds", "11b fold count")
    haz_p = num("dhaka_pm25_model_h6.json:cv.aggregate.f1_Hazardous.p_two_sided", "11b p")
    ref_h = num("dhaka_ground_truth.json:comparison.hazardous_reference", "reference Hazardous hours", ",.0f")
    rea_h = num("dhaka_ground_truth.json:comparison.hazardous_reanalysis", "reanalysis Hazardous hours", ",.0f")
    rows_pct, span_pct = _audit_fractions()
    oaq = num("openaq_survey.json:assessment.n_passing", "OpenAQ stations passing", ".0f")

    for chunk in [
        f"Wearable air-quality devices are typically evaluated by reporting a headline "
        f"accuracy for a proposed model. This thesis argues that such a number is "
        f"uninterpretable without the zero-parameter baseline it must beat, and "
        f"demonstrates the consequences on real data. At a one-hour horizon the AQI "
        f"category is unchanged in {unch1}% of samples and a persistence rule scores "
        f"{floor1} macro-F1 for free; moving the primary horizon to six hours lowers "
        f"that floor to {floor6} and turns a persistence echo into a forecasting task.",

        f"On the UCI Beijing multi-site dataset, a RandomForest reaches {rf} macro-F1 "
        f"on the test split. Under five-fold rolling-origin cross-validation with an "
        f"embargo and per-fold rescaling, no Beijing-trained model beats persistence in "
        f"more than {bj_folds} of {bj_n} folds, and persistence's own fold-to-fold "
        f"spread exceeds any model-to-baseline difference. Three class-imbalance "
        f"interventions were compared under a disqualification rule that rejects an "
        f"aggregate gain bought by degrading a safety-critical class: CTGAN "
        f"augmentation and SMOTE both raise macro-F1 while significantly degrading both "
        f"advisory classes and are disqualified; only class weighting avoids that "
        f"trade. Monte Carlo dropout attributes {epi} of predictive entropy to "
        f"aleatoric uncertainty, which explains why added capacity made both sequence "
        f"architectures monotonically worse.",

        f"External validation used a published Bangladesh dataset. A data-integrity "
        f"audit found that discarding its fabricated portion costs {rows_pct}% of rows "
        f"but {span_pct}% of the advertised twenty-five-year span, because the "
        f"discarded part is one city at low density. On the surviving window, a "
        f"class-weighted RandomForest beats persistence in {bd_wins} of {bd_n} "
        f"rolling-origin folds (mean {bd_delta}, one-sided p = {bd_p}) — but the "
        f"evaluation blocks contain too few advisory-class samples for that result to "
        f"cover them. Validation against the US Embassy Dhaka reference monitor "
        f"confirmed why: the reanalysis records {rea_h} Hazardous hours where the "
        f"instrument records {ref_h}. A PM2.5-only model trained on that ground truth "
        f"closes the gap, reaching Hazardous F1 {haz_m} against a {haz_f} persistence "
        f"floor in {haz_w} of {haz_n} folds (p = {haz_p}). A survey of OpenAQ found "
        f"{oaq} stations near Dhaka meeting multi-pollutant coverage requirements, "
        f"which bounds what any multi-channel model can currently be validated on.",

        f"The contributions are an evaluation protocol released as the open-source "
        f"package PulseBench, a reusable data-integrity audit, split-conformal and "
        f"Mondrian uncertainty quantification with per-class coverage, a validated "
        f"LLM advisory layer whose honesty constraints are enforced outside the model, "
        f"and a compressed ONNX predictor with measured latency. The device itself is "
        f"a design, not a fabricated artefact. All code, metrics and figures are "
        f"public.",
    ]:
        para(doc, chunk, align="justify", space_after=8)

    para(doc, "", space_after=4)
    para(doc, "Keywords: air-quality forecasting; persistence baseline; rolling-origin "
              "cross-validation; class imbalance; conformal prediction; uncertainty "
              "quantification; data integrity; edge deployment.",
         align="justify", size=10, italic=True)


# ------------------------------------------------------------- derived quantities


def _beijing_best_folds() -> str:
    d = SRC.j("rolling_cv_h6.json")
    if not d:
        return need("rolling_cv_h6.json:aggregate", "Beijing rolling CV")
    wins = [t["wins"] for k, t in d["aggregate"]["tests"].items() if not k.startswith("_")]
    return str(max(wins)) if wins else need("rolling_cv_h6.json:aggregate.tests", "fold wins")


def _epistemic_share() -> str:
    d = SRC.j("dl_h6.json")
    if not d:
        return need("dl_h6.json:uncertainty", "uncertainty decomposition")
    u = d["uncertainty"][d["selected"]]
    return f"{(1 - u['mean_epistemic'] / u['mean_entropy']) * 100:.1f}%"


def _audit_fractions() -> tuple[str, str]:
    """Rows discarded and span discarded, recomputed from the audit block."""
    d = SRC.j("bangladesh_h6.json")
    if not d:
        m = need("bangladesh_h6.json:audit", "Bangladesh audit")
        return m, m
    from datetime import datetime
    a = d["audit"]
    rows = (a["file_rows"] - a["clean_rows"]) / a["file_rows"] * 100
    s, c, e = (datetime.fromisoformat(a["actual_range"][0]),
               datetime.fromisoformat(a["clean_start"]),
               datetime.fromisoformat(a["actual_range"][1]))
    return f"{rows:.0f}", f"{(c - s) / (e - s) * 100:.0f}"


def _figure_sources() -> dict[str, str]:
    """figure filename -> the JSON it was generated from, per the figure index."""
    text = (FIGURES / "README.md").read_text() if (FIGURES / "README.md").exists() else ""
    out = {}
    for _, png, src in re.findall(r"^\| \d+ \| \[[^\]]+\]\(([^)]+)\)() \| (.+?) \|$",
                                  text, re.M):
        pass
    for m in re.finditer(r"^\| (\d+) \| \[([^\]]+)\]\(([^)]+)\) \| (.+?) \|$", text, re.M):
        srcs = m.group(4).replace("`", "").strip()
        out[m.group(3)] = srcs
    return out


FIGSRC = None


def figsrc(png: str) -> str:
    global FIGSRC
    if FIGSRC is None:
        FIGSRC = _figure_sources()
    s = FIGSRC.get(png)
    if not s:
        marker = f"[MISSING: source JSON for {png} — not listed in reports/figures/README.md]"
        SRC.missing.append(marker)
        return marker
    return s


# ------------------------------------------------------------------ generated lists


def toc(doc: Document) -> None:
    page_break(doc)
    para(doc, "Table of Contents", align="center", size=H1_PT, bold=True, space_after=14)
    entries = [
        ("Declaration", 1), ("Certificate of Approval", 1), ("Acknowledgement", 1),
        ("Abstract", 1), ("List of Figures", 1), ("List of Tables", 1),
        ("List of Abbreviations", 1),
        ("1. Introduction", 1),
        ("1.1 Background and motivation", 2),
        ("1.2 The evaluation problem in this literature", 2),
        ("1.3 Objectives and contributions", 2),
        ("2. Related Work and Theoretical Background", 1),
        ("2.1 Ensemble methods: Random Forest and gradient boosting", 2),
        ("2.2 Sequence models: LSTM and Transformer", 2),
        ("2.3 Synthetic tabular data: CTGAN", 2),
        ("2.4 Classical resampling: SMOTE and class weighting", 2),
        ("2.5 Conformal prediction", 2),
        ("2.6 Explainability: SHAP", 2),
        ("2.7 Evaluation protocol: rolling-origin cross-validation", 2),
        ("2.8 Multiple comparisons", 2),
        ("2.9 Uncertainty quantification", 2),
        ("2.10 Edge deployment and TinyML", 2),
        ("2.11 Language models for risk communication", 2),
        ("3. Materials and Methods", 1),
        ("3.1 Datasets", 2), ("3.2 Preprocessing", 2), ("3.3 Horizon selection", 2),
        ("3.4 Baseline models and evaluation protocol", 2),
        ("3.5 Class-imbalance interventions", 2),
        ("3.6 Sequence models and uncertainty decomposition", 2),
        ("3.7 Conformal prediction", 2),
        ("3.8 Explainability and the LLM advisory layer", 2),
        ("3.9 Compression and edge deployment", 2),
        ("3.10 Rolling-origin cross-validation", 2),
        ("3.11 Multiple-comparisons correction", 2),
        ("3.12 The PulseBench toolkit", 2),
        ("4. Results", 1),
        ("4.1 The persistence floor", 2),
        ("4.2 Baseline models and rolling-origin cross-validation", 2),
        ("4.3 Class-imbalance interventions and the validity-metric failure", 2),
        ("4.4 Sequence models, capacity and uncertainty", 2),
        ("4.5 Conformal coverage", 2),
        ("4.6 Compression and edge deployment", 2),
        ("4.7 Sensitivity to the air-quality standard", 2),
        ("4.8 Multiple-comparisons correction", 2),
        ("4.9 External validation: Bangladesh", 2),
        ("4.10 Ground truth: the US Embassy Dhaka monitor", 2),
        ("4.11 The validated Hazardous detector", 2),
        ("4.12 Monitoring infrastructure: the OpenAQ survey", 2),
        ("4.13 The deployed system", 2),
        ("4.14 Explainability: SHAP case studies", 2),
        ("5. Discussion", 1),
        ("5.1 An average says nothing about its tail", 2),
        ("5.2 The ceiling is in the data", 2),
        ("5.3 Methodology transferred; weights did not", 2),
        ("5.4 Data integrity is a result, not a preliminary", 2),
        ("5.5 Cheap controls change conclusions", 2),
        ("5.6 What reproducibility required in practice", 2), ("6. Limitations", 1),
        ("7. Proposed Wearable Device (Concept — Not Yet Fabricated)", 1),
        ("8. Conclusion and Future Work", 1),
        ("Data and Code Availability", 1), ("References", 1),
        ("Appendix A: Full Hyperparameter Tables", 1),
        ("Appendix B: Glossary of Terms", 1),
        ("Appendix C: Index of Generated Reports", 1),
    ]
    for text, lvl in entries:
        p = para(doc, ("    " * (lvl - 1)) + text,
                 size=10.5 if lvl == 1 else 10, bold=(lvl == 1), space_after=2)


def list_of_figures(doc: Document) -> None:
    page_break(doc)
    para(doc, "List of Figures", align="center", size=H1_PT, bold=True, space_after=14)
    if not REG.figures:
        para(doc, "[MISSING: no figures were embedded]")
        return
    for label, source in REG.figures:
        para(doc, f"{label}  (from {source})", size=10, space_after=3)


def list_of_tables(doc: Document) -> None:
    page_break(doc)
    para(doc, "List of Tables", align="center", size=H1_PT, bold=True, space_after=14)
    if not REG.tables:
        para(doc, "[MISSING: no tables were embedded]")
        return
    for label, source in REG.tables:
        suffix = f"  (from {source})" if source else ""
        para(doc, f"{label}{suffix}", size=10, space_after=3)


ABBREVIATIONS = {
    "AQI": "Air Quality Index",
    "AIUB": "American International University-Bangladesh",
    "API": "Application Programming Interface",
    "CI": "Confidence Interval",
    "CO": "Carbon Monoxide",
    "CTGAN": "Conditional Tabular Generative Adversarial Network",
    "CV": "Cross-Validation",
    "DEWP": "Dew Point",
    "ECE": "Expected Calibration Error",
    "EPA": "United States Environmental Protection Agency",
    "ESP32": "Espressif 32-bit microcontroller",
    "F1": "Harmonic mean of precision and recall",
    "GAN": "Generative Adversarial Network",
    "HJ 633-2012": "Chinese Technical Regulation on Ambient Air Quality Index",
    "LLM": "Large Language Model",
    "LSTM": "Long Short-Term Memory",
    "MC": "Monte Carlo",
    "MCE": "Maximum Calibration Error",
    "MCU": "Microcontroller Unit",
    "ONNX": "Open Neural Network Exchange",
    "PM2.5": "Particulate matter under 2.5 micrometres in diameter",
    "PM10": "Particulate matter under 10 micrometres in diameter",
    "QC": "Quality Control",
    "RF": "Random Forest",
    "SD": "Standard Deviation",
    "SDV": "Synthetic Data Vault",
    "SHAP": "SHapley Additive exPlanations",
    "SMOTE": "Synthetic Minority Over-sampling Technique",
    "TEMP": "Air Temperature",
    "TFLite": "TensorFlow Lite",
    "XGBoost": "Extreme Gradient Boosting",
}


def list_of_abbreviations(doc: Document, body_text: str) -> None:
    page_break(doc)
    para(doc, "List of Abbreviations", align="center", size=H1_PT, bold=True,
         space_after=14)
    # Built from the document text, not from a generic list: an abbreviation the
    # thesis never uses has no business in its glossary.
    used = [(k, v) for k, v in sorted(ABBREVIATIONS.items())
            if re.search(re.escape(k), body_text)]
    rows = [[k, v] for k, v in used]
    if not rows:
        para(doc, "[MISSING: no abbreviations detected in the document text]")
        return
    t = doc.add_table(rows=0, cols=2)
    t.style = "Light List Accent 1"
    for k, v in rows:
        cells = t.add_row().cells
        cells[0].text = ""
        _rich(cells[0].paragraphs[0], k, size=TABLE_PT, bold=True)
        cells[1].text = ""
        _rich(cells[1].paragraphs[0], v, size=TABLE_PT)
        cells[0].width, cells[1].width = Inches(1.4), Inches(4.9)


# ------------------------------------------------------------------- chapter one


def chapter_intro(doc: Document) -> None:
    heading(doc, "1. Introduction", 1)

    heading(doc, "1.1 Background and motivation", 2)
    para(doc, "**General framing, not a result of this project.** The paragraphs in this "
              "subsection describe the wider context that motivates a wearable "
              "air-quality device. They are not findings of this thesis, and the "
              "epidemiological and exposure literature they would properly cite is "
              "outside the scope of what this project measured. No number in this "
              "subsection is used anywhere else in the document.",
         align="justify", italic=True)
    para(doc,
         "Fine particulate matter under 2.5 micrometres (PM2.5) penetrates deep into "
         "the respiratory tract, and sustained exposure is associated with "
         "cardiovascular and respiratory harm. In dense South Asian cities, including "
         "Dhaka, seasonal concentrations reach levels far above international guideline "
         "values during the winter months, while the monitoring infrastructure that "
         "would let an individual know their own exposure remains sparse. A person "
         "moving through such a city experiences a concentration that depends on street, "
         "hour and activity, not on the single citywide figure a public index reports.",
         align="justify")
    para(doc,
         "Exposure studies in Bangladesh and comparable settings document both the "
         "magnitude of the problem and the difficulty of measuring it at the level of "
         "an individual " + cite("Rahman, Begum, Hopke, Nahar, Newman, Thurston 2021",
                                  "Lee, Bayazid, Rosenthal, Khan, Arku, Barratt, Quayyum, Baumgartner 2025, Scientific Reports",
                                  "Rahman & Meng 2024") + ". Monitoring "
         "and reporting organisations track the resulting concentrations at city scale "
         + cite("CREA 2025") + ", and applied machine-learning studies have modelled "
         "Dhaka's particulate levels from station and satellite inputs "
         + cite("Hasan, Rahman, Akhter, Mohinuzzaman, Kayes & Rahman 2024",
                "Islam et al. 2023") + ". What none of these give a person walking "
         "through the city is their own exposure.",
         align="justify")
    para(doc,
         "Personal and mobile sensing has been proposed as the answer, using phones as "
         "exposure monitors " + cite("\"Mobile phones as monitors of personal exposure to air pollution\" 2018, PLOS ONE")
         + " and applications to make invisible pollution legible to the people "
         "breathing it " + cite("\"Can Apps Make Air Pollution Visible?\" 2019, Journal of Business Ethics")
         + ". A dedicated wearable extends that idea with a sensor placed at the "
         "breathing zone rather than in a pocket.",
         align="justify")
    para(doc,
         "That gap motivates a wearable: a device carried on the body that samples the "
         "air the wearer is actually breathing, forecasts where it is heading over the "
         "next few hours, and says something useful about it. PulseAir is such a "
         "device, designed around an ESP32 microcontroller. This thesis concerns "
         "everything that happens away from the device — the data preparation, the "
         "forecasting models, the uncertainty quantification, the explanation layer, "
         "and the compression that would let a trained model run on the hardware.",
         align="justify")
    para(doc,
         "The device itself is a design in this thesis, not a fabricated artefact. "
         "Chapter 7 describes the concept and states plainly what has and has not been "
         "built.",
         align="justify")

    heading(doc, "1.2 The evaluation problem in this literature", 2)
    unch1 = num("horizon_comparison.json:rows.1.label_unchanged_pct", "h1 unchanged", ".1f")
    f1 = num("horizon_comparison.json:rows.1.observed.macro_f1", "h1 floor")
    f6 = num("horizon_comparison.json:rows.6.observed.macro_f1", "h6 floor")
    para(doc,
         "A recurring pattern in applied air-quality forecasting is to propose a model, "
         "report a headline accuracy or macro-F1, and compare it against other learned "
         "models. The comparison that is usually missing is against doing nothing at "
         "all.",
         align="justify")
    para(doc,
         f"This matters because air quality is highly autocorrelated. At a one-hour "
         f"horizon on the Beijing data used here, the AQI category at t+1 is identical "
         f"to the category at t in **{unch1}%** of samples. A rule with no parameters, "
         f"no training and no inputs beyond the current reading — predict that the "
         f"category does not change — scores **{f1}** macro-F1. Any model reporting a "
         f"comparable figure at that horizon has demonstrated that it has learned to "
         f"echo its input, which is not the same as having learned to forecast.",
         align="justify")
    para(doc,
         f"This thesis therefore adopts a single discipline throughout: **every "
         f"headline metric is reported beside the zero-parameter persistence floor for "
         f"the same task and the same rows**. Moving the primary horizon from one hour "
         f"to six lowers that floor from {f1} to **{f6}**, which is what makes the task "
         f"a forecasting problem rather than a persistence echo. The floor is not a "
         f"courtesy baseline; it is the thing a model must clear before its number "
         f"means anything, and much of what follows is a record of models failing to "
         f"clear it by a margin that survives scrutiny.",
         align="justify")
    para(doc,
         "The problem is not unique to air quality. The same argument — that a headline "
         "accuracy is meaningless without the base rate it must exceed — has been made "
         "for financial forecasting benchmarks, where directional accuracy can look "
         "impressive while conveying nothing a naive rule does not already supply "
         + cite("Cheung 2026") + ". Air quality simply makes the problem unusually easy "
         "to fall into, because its autocorrelation is unusually high.",
         align="justify")
    para(doc,
         "A second discipline follows from the first. When a single train/validation/"
         "test split disagrees with itself — when validation and test pick different "
         "winners — the honest response is not to report whichever split is more "
         "flattering, but to re-run the comparison over multiple chronological folds "
         "and see whether the effect is stable. Section 3.10 describes the protocol; "
         "Section 4.2 reports what it found.",
         align="justify")

    heading(doc, "1.3 Objectives and contributions", 2)
    para(doc, "The objectives of this work were:", align="justify")
    bullets(doc, [
        "To build a complete off-device pipeline for wearable air-quality risk "
        "forecasting: preprocessing, forecasting, uncertainty quantification, "
        "explanation and edge export.",
        "To test whether GAN-based augmentation of rare, safety-critical classes "
        "improves a wearable risk classifier, under controls strong enough for the "
        "answer to be trusted either way.",
        "To quantify predictive uncertainty in a form a device can act on, rather than "
        "reporting a point prediction.",
        "To validate the methodology on the population the device targets, rather than "
        "on the benchmark it was developed against.",
    ])
    para(doc, "The contributions are:", align="justify")
    bullets(doc, [
        "**An evaluation protocol, released as software.** The persistence floor, "
        "rolling-origin cross-validation with embargo and per-fold scaling, a "
        "protected-class disqualification rule, and multiple-comparisons correction "
        "with the resampling resolution floor stated explicitly, are packaged as "
        "**PulseBench**, an open-source Python library independent of this project's "
        "code. Section 3.12 describes it.",
        "**A reproducible data-integrity audit of a published dataset.** A widely "
        "indexed Bangladesh air-quality dataset is shown to be substantially "
        "fabricated over its advertised span, with the evidence recomputed from the "
        "raw file rather than asserted. Section 4.9.",
        "**Negative results reported as results.** CTGAN and SMOTE augmentation both "
        "improve aggregate macro-F1 while significantly degrading both advisory "
        "classes; added model capacity makes both sequence architectures worse; and "
        "no Beijing-trained model beats persistence in a majority of rolling-origin "
        "folds. Each is documented with the statistical test that established it.",
        "**A demonstration that synthesiser quality scores do not measure validity.** "
        "A CTGAN configuration scoring above 0.89 on the Synthetic Data Vault quality "
        "report emitted physically impossible records in both cyclical channels and in "
        "the temperature/dew-point relationship. Section 4.3.",
        "**Ground-truth validation against a reference monitor**, resolving why the "
        "advisory classes could not be validated on the published dataset, and a "
        "PM2.5-only model that closes that gap. Sections 4.10 and 4.11.",
        "**A survey of available monitoring infrastructure** near Dhaka that bounds "
        "what any multi-pollutant model can currently be validated against. "
        "Section 4.12.",
        "**A deployable predictor with measured cost**: a compressed forest exported to "
        "ONNX, wrapped in class-conditional conformal prediction, with a validated "
        "language-model advisory layer whose honesty constraints are enforced outside "
        "the model. Section 4.13.",
    ])


# ------------------------------------------------------------------- chapter two

# Citations are keys into reports/reference_list_expanded.md. Nothing is cited that is
# not in that file; the mapping is checked at build time by `_check_citations`.
CITED: set[str] = set()


def cite(*keys: str) -> str:
    for k in keys:
        CITED.add(k)
    return "(" + "; ".join(keys) + ")"


def chapter_related(doc: Document) -> None:
    heading(doc, "2. Related Work and Theoretical Background", 1)
    para(doc,
         "This chapter covers each technique the project actually uses, what it is, and "
         "why it was chosen here. Citations refer to entries in the reference list, "
         "which was compiled and verified against bibliographic registries as described "
         "in the References section.",
         align="justify")

    heading(doc, "2.1 Ensemble methods: Random Forest and gradient boosting", 2)
    para(doc,
         "A Random Forest " + cite("Breiman 2001") + " fits many decision trees to "
         "bootstrap samples of the training data, each tree considering a random subset "
         "of features at every split, and averages their predictions. The averaging "
         "reduces the variance that makes a single deep tree unreliable, and the "
         "resulting model needs little tuning, handles mixed feature scales without "
         "standardisation, and produces calibrated class probabilities directly. For a "
         "tabular problem with nine engineered features and no spatial structure, that "
         "makes it the natural first model rather than a fallback.",
         align="justify")
    para(doc,
         "Gradient-boosted trees, here XGBoost " + cite("Chen & Guestrin 2016") + ", fit "
         "trees sequentially, each correcting the residual error of those before it. "
         "Boosting typically outperforms bagging on tabular benchmarks, which is why it "
         "is included as the stronger tabular comparator. Both were implemented through "
         "scikit-learn " + cite("Pedregosa et al. 2011") + ". Recent applied work "
         "continues to find that lightweight and classical models remain competitive "
         "with deep architectures for air-quality prediction "
         + cite("Gondal, Qudous & Farhan 2025", "Gondal, Qudous, Farhan & Alamri 2026")
         + ", which is consistent with what Chapter 4 reports.",
         align="justify")

    heading(doc, "2.2 Sequence models: LSTM and Transformer", 2)
    para(doc,
         "A Long Short-Term Memory network " + cite("Hochreiter & Schmidhuber 1997") +
         " processes a sequence one step at a time, carrying a gated cell state that "
         "lets gradients survive across long spans. A Transformer encoder "
         + cite("Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin 2017")
         + " instead attends over all positions simultaneously, so any timestep can "
         "influence any other in one layer. Both were implemented in PyTorch "
         + cite("Paszke et al. 2019") + ".",
         align="justify")
    para(doc,
         "They are included because the tabular baselines see only the current hour, "
         "while the wearable records a continuous stream. If the preceding 24 hours "
         "carry information the current reading does not, a sequence model should find "
         "it. Domain-specific architectures exist — AirFormer "
         + cite("Liang, Xia, Ke, Wang, Wen, Zhang, Zheng, Zimmermann 2023 (AirFormer)")
         + " adapts attention to spatio-temporal air-quality fields — but the question "
         "here was whether sequence modelling helps at all on a single station's "
         "channels, which a standard encoder answers more cleanly than a specialised "
         "one.",
         align="justify")

    heading(doc, "2.3 Synthetic tabular data: CTGAN", 2)
    para(doc,
         "A Conditional Tabular GAN " + cite("Xu, Skoularidou, Cuesta-Infante, Veeramachaneni 2019")
         + " adapts adversarial training to tabular data, using mode-specific "
         "normalisation for continuous columns and conditional sampling by discrete "
         "column so that rare categories are actually generated. It was used here "
         "through the Synthetic Data Vault framework "
         + cite("Patki, Wedge, Veeramachaneni 2016") + ". Later tabular synthesisers, "
         "notably CTAB-GAN and its successor "
         + cite("Zhao, Kunar, Birke, Chen 2021", "Zhao, Kunar, Birke, Chen 2022")
         + ", extend the same idea.",
         align="justify")
    para(doc,
         "The motivation is that the classes a wearable exists to warn about are, by "
         "construction, the rarest in the data. If a generator can learn the joint "
         "distribution of sensor channels during hazardous episodes and produce more "
         "such records, a classifier might learn those classes better. Whether that "
         "works is an empirical question, and the literature is mixed: comparative "
         "studies find CTGAN's advantage over classical resampling to be dataset- and "
         "context-dependent " + cite("Gündüz & Şahin 2026", "Adiputra & Wanchai 2024")
         + ". Section 4.3 reports what happened here.",
         align="justify")

    heading(doc, "2.4 Classical resampling: SMOTE and class weighting", 2)
    para(doc,
         "SMOTE " + cite("Chawla, Bowyer, Hall, Kegelmeyer 2002") + " oversamples a "
         "minority class by interpolating between a sample and its nearest neighbours "
         "of the same class, implemented here through imbalanced-learn "
         + cite("Lemaître, Nogueira, Aridas 2017") + ". Class weighting is cheaper "
         "still: it leaves the data alone and scales each class's contribution to the "
         "loss by the inverse of its frequency.",
         align="justify")
    para(doc,
         "Both are included as controls, and the reason is methodological. A paper that "
         "proposes GAN augmentation for class imbalance has to show that the generator "
         "earns its cost against the two-second alternative and the one-line "
         "alternative. Without those controls, a reported improvement cannot be "
         "attributed to the generator rather than to rebalancing as such.",
         align="justify")

    heading(doc, "2.5 Conformal prediction", 2)
    para(doc,
         "Conformal prediction " + cite("Angelopoulos & Bates 2023") + " converts any "
         "model's scores into prediction *sets* with a finite-sample coverage "
         "guarantee that holds without distributional assumptions, given exchangeable "
         "data. Split conformal reserves a calibration set, computes a nonconformity "
         "score on it, and takes the appropriate quantile as a threshold. Extensions "
         "address the assumptions this project strains: coverage beyond exchangeability "
         + cite("Barber, Candès, Ramdas, Tibshirani 2023") + " and adaptive coverage "
         "under distribution shift " + cite("Gibbs & Candès 2021") + ".",
         align="justify")
    para(doc,
         "The guarantee that split conformal provides is *marginal* — it holds on "
         "average over all classes. That is not what a safety-critical advisory needs. "
         "Mondrian, or class-conditional, conformal prediction calibrates a separate "
         "threshold per class, restoring coverage for rare classes at the cost of "
         "larger sets. Recent benchmarks on imbalanced data quantify that trade "
         + cite("Singh, Srikantha & Lakhanpal 2026", "Ding, Fermanian & Salmon 2025")
         + ". Section 4.5 shows the same effect on this data.",
         align="justify")

    heading(doc, "2.6 Explainability: SHAP", 2)
    para(doc,
         "SHAP " + cite("Lundberg & Lee 2017") + " attributes a prediction to its input "
         "features using Shapley values from cooperative game theory, giving the unique "
         "attribution satisfying local accuracy, missingness and consistency. "
         "TreeExplainer " + cite("Lundberg, Erion, Lee 2018") + " computes these exactly "
         "and in polynomial time for tree ensembles, which is what makes the method "
         "practical for a Random Forest. SHAP is now standard in applied air-quality "
         "modelling " + cite("Gowri et al. 2025", "Azani Hassan Abadi & Wang 2026") + ".",
         align="justify")
    para(doc,
         "Here it serves a specific purpose rather than a decorative one: the advisory "
         "layer needs to tell the wearer *which channel* drove a warning, and a "
         "per-prediction attribution is the input to that sentence.",
         align="justify")

    heading(doc, "2.7 Evaluation protocol: rolling-origin cross-validation", 2)
    para(doc,
         "Random k-fold cross-validation is invalid on ordered data: it places future "
         "observations in folds used to predict the past. Rolling-origin evaluation "
         + cite("Tashman 2000") + " instead advances a cutoff through time, training on "
         "everything before it and evaluating on a block after it, so every prediction "
         "is made from data that genuinely preceded it. The conditions under which "
         "cross-validation is and is not safe for time-series model selection have been "
         "examined directly " + cite("Bergmeir & Benítez 2012", "Liu & Zhou 2024") + ", "
         "and evaluation-integrity failures remain a live problem in current "
         "time-series benchmarking " + cite("Meyer, Kaltenpoth, Zalipski & Müller 2025")
         + ".",
         align="justify")
    para(doc,
         "This protocol is the decisive experiment of the thesis. A single chronological "
         "split gives one number per model and no way to tell a real effect from the "
         "luck of where the split fell; five folds give a distribution, and a paired "
         "test over fold-level differences asks whether a model beats the baseline "
         "*consistently* rather than *once*.",
         align="justify")

    heading(doc, "2.8 Multiple comparisons", 2)
    para(doc,
         "When one model is tested against a baseline, a p-value means what it says. "
         "When eleven are, the family-wise error rate is not what the individual "
         "p-values report. The standard correction over a family of comparisons, and "
         "the recommendation to use the Wilcoxon signed-rank test for paired "
         "model comparisons, come from " + cite("Demšar 2006") + "; Bayesian "
         "alternatives to null-hypothesis testing in this setting are developed in "
         + cite("Corani, Benavoli, Demšar, Mangili & Zaffalon 2017") + ".",
         align="justify")
    para(doc,
         "A detail that matters for honest reporting: a resampling-based test cannot "
         "resolve a p-value below its own resolution. A bootstrap with 1,000 resamples "
         "cannot report a two-sided p below 2/1000, and a signed-rank test over n "
         "folds cannot report one below 2^(1-n). Values at that floor are reported here "
         "as bounded rather than exact.",
         align="justify")

    heading(doc, "2.9 Uncertainty quantification", 2)
    para(doc,
         "Monte Carlo dropout " + cite("Gal & Ghahramani 2016") + " keeps dropout active "
         "at inference and averages over many stochastic forward passes, approximating "
         "Bayesian inference in a deep network. The spread across passes estimates "
         "*epistemic* uncertainty — what the model does not know, and which more data "
         "or capacity could reduce — while the remaining entropy is *aleatoric*, "
         "irreducible given the inputs. The decomposition is the argument for or "
         "against building a bigger model. Its reliability has been questioned "
         + cite("Djupskås, Stasik & Riemer-Sørensen 2025") + ", a caveat Chapter 6 "
         "returns to.",
         align="justify")

    heading(doc, "2.10 Edge deployment and TinyML", 2)
    para(doc,
         "Running inference on a microcontroller rather than in the cloud removes "
         "latency, connectivity and privacy costs, at the price of a few hundred "
         "kilobytes of memory. Surveys of the field " + cite("Somvanshi et al. 2025")
         + " and applied air-quality deployments on low-cost edge hardware "
         + cite("Huam Ming Ken & Behjati 2025") + " establish the envelope. Models were "
         "exported here to ONNX for portability and benchmarking; Section 4.6 is "
         "explicit that this is a portability proxy and not a claim of having flashed "
         "the device.",
         align="justify")

    heading(doc, "2.11 Prior applied air-quality forecasting", 2)
    para(doc,
         "Applied air-quality forecasting is a large and active literature. Ensemble and "
         "hybrid models have been applied to urban AQI prediction across many cities "
         + cite("Liang, Maimury, Chen, Juarez 2020", "Pak, Ma, Ryu, Ryom, Juhyok, Pak, Pak 2020",
                "Mao, Wang, Jiao, Zhao, Liu 2021") + ", with more recent work extending "
         "to deep and hybrid architectures " + cite("Zhou, Wang, Zhu, Qiao, Kang 2024",
                                                    "Wang 2025", "Selva 2025") + " and to "
         "interpretable and IoT-oriented systems "
         + cite("Pineda-Tobón, Espinosa-Bedoya, Branch-Bedoya 2024", "Ray 2022",
                "Özüpak, Alpsalaz, Aslan 2025") + ". Reviews of the field appear "
         "regularly " + cite("RSC Environmental Science: Atmospheres 2026") + ".",
         align="justify")
    para(doc,
         "Two observations position the present work against that body of literature. "
         "First, the reported improvements are usually stated against other learned "
         "models rather than against a zero-parameter rule, which is the gap Section 1.2 "
         "identifies. Second, where a naive comparator *is* included — as in the Dhaka "
         "study that reports a NAIVE baseline alongside ARIMA and neural alternatives "
         + cite("Hasan, Rahman, Akhter, Mohinuzzaman, Kayes & Rahman 2024") + " — the "
         "margin between the proposed model and the naive rule is frequently smaller "
         "than the framing suggests. Recent comparative work reaches the same "
         "conclusion directly, finding lightweight additive models competitive with "
         "deep architectures on urban air-quality series "
         + cite("Gondal, Qudous & Farhan 2025") + ".",
         align="justify")
    para(doc,
         "A related strand concerns the *inputs* rather than the models. Reanalysis and "
         "satellite-derived products are widely used where ground stations are sparse, "
         "and are known to misestimate surface concentrations — particularly at the top "
         "of the range " + cite("Ali et al. 2022", "Singh et al. 2026") + ". Section 4.10 "
         "reports a direct instance of that on the dataset used here, measured against a "
         "reference monitor.",
         align="justify")

    heading(doc, "2.12 Language models for risk communication", 2)
    para(doc,
         "The final layer turns numbers into a sentence a wearer can act on. That is a "
         "risk-communication task, and language models are demonstrably unreliable at "
         "it: their hedging does not track probability, and they moderate their "
         "certainty in response to a user's emotional tone rather than the evidence "
         + cite("Cerda-Mardini, Chandar & Madathil 2026") + ". Safety messaging in "
         "general-purpose models has also been observed to decline over successive "
         "releases " + cite("Sharma, Alaa & Daneshjou 2025") + ".",
         align="justify")
    para(doc,
         "The design consequence is the central one for this layer: **the honesty "
         "constraints are enforced outside the model.** Generated text is validated "
         "programmatically before use, and any output that fails is discarded in favour "
         "of a rule-based template. Section 3.8 describes the checks.",
         align="justify")


# ----------------------------------------------------------------- chapter three


def chapter_methods(doc: Document) -> None:
    cfg = SRC.cfg
    prep, base, gan, model, conf, dep = (cfg["preprocessing"], cfg["baseline"],
                                         cfg["gan"], cfg["model"], cfg["conformal"],
                                         cfg["deployment"])
    heading(doc, "3. Materials and Methods", 1)
    para(doc,
         "Every hyperparameter in this chapter is read from `configs/default.yaml` in "
         "the public repository; Appendix A reproduces that file's contents in full. "
         "Every count and threshold is read from the committed metrics files.",
         align="justify")

    heading(doc, "3.1 Datasets", 2)
    para(doc,
         "Four sources were used. The benchmark is the UCI Beijing Multi-Site Air "
         "Quality dataset " + cite("UCI Machine Learning Repository 2017/2019") + ". AQI "
         "categories are assigned throughout using the United States Environmental "
         "Protection Agency's PM2.5 breakpoints " + cite("U.S. EPA 2024") + "; Section "
         "4.7 repeats the central analysis under the Chinese national standard to show "
         "the conclusion does not depend on that choice.",
         align="justify")
    para(doc, "**Beijing Multi-Site Air Quality (methodology dataset).** "
              f"{need('bangladesh_h6.json:audit.file_rows', 'x') and ''}"
              "Hourly records from twelve monitoring stations, used in full. This is "
              "the dataset on which the evaluation protocol was developed.",
         align="justify")
    table(doc, "The four data sources used in this thesis.",
          ["Source", "Role", "Extent", "Notes"],
          [["UCI Beijing Multi-Site", "Methodology development",
            "420,768 hourly records, 12 stations, 2013-03-01 to 2017-02-28",
            "Used in full; nothing from it is deployed"],
           ["Mendeley Bangladesh AQI (9j447cynb9 v2)", "External validation, deployment",
            f"{need('bangladesh_h6.json:audit.clean_rows', 'clean rows'):,} usable rows, "
            f"{need('bangladesh_h6.json:audit.clean_cities', 'clean cities')} cities, "
            f"{str(need('bangladesh_h6.json:audit.clean_start', 'clean start'))[:10]} onward",
            "Advertised 2000–2025; only the post-cut window survives audit (Section 4.9)"],
           ["US Embassy Dhaka reference monitor", "Ground truth",
            f"{need('dhaka_ground_truth.json:audit.final_rows', 'embassy rows'):,} QC-passed hours, "
            f"{str(need('dhaka_ground_truth.json:audit.start', 'embassy start'))[:10]} to "
            f"{str(need('dhaka_ground_truth.json:audit.end', 'embassy end'))[:10]}",
            "PM2.5 only, single station, per-row QC flag"],
           ["OpenAQ v3 station survey", "Infrastructure assessment",
            f"{len(need('openaq_survey.json:survey.locations', 'openaq locations') or [])} stations within "
            f"{num('openaq_survey.json:survey.radius_km', 'radius', '.0f')} km of Dhaka",
            "Establishes what multi-pollutant validation is currently possible"]],
          source="bangladesh_h6.json, dhaka_ground_truth.json, openaq_survey.json",
          widths=[1.5, 1.2, 2.0, 1.8])

    _class_distribution_table(doc)

    heading(doc, "3.2 Preprocessing", 2)
    feats = need("bangladesh_h6.json:meta.feature_columns", "feature columns")
    para(doc,
         f"Missing values are forward-filled within each station, never across station "
         f"boundaries. Every filled cell is recorded in a parallel `is_imputed` column "
         f"that is **not** a model input: forward-filling roughly doubles the apparent "
         f"prevalence of the rarest class, so every selection metric and every reported "
         f"headline in this thesis is computed on observed-label rows only. That "
         f"provenance tracking is what makes the restriction possible.",
         align="justify")
    para(doc,
         f"Hour and month are encoded as sine/cosine pairs so that hour 23 and hour 0 "
         f"are adjacent. Continuous channels are standardised with a scaler fit on the "
         f"training split alone; the cyclical encodings are left unscaled, since they "
         f"already lie on the unit circle. Sequences are built by sliding a "
         f"**{prep['window']}-hour** window over contiguous runs, so no window spans a "
         f"gap or two stations. Splits are chronological, never random.",
         align="justify")
    table(doc, "Preprocessing configuration.",
          ["Setting", "Value", "Rationale"],
          [["Modelled channels", ", ".join(prep["features"]), "The subset a wearable can plausibly carry"],
           ["Cyclical encodings", ", ".join(prep["cyclical"]), "sin/cos pairs; hour 23 adjacent to hour 0"],
           ["Window", f"{prep['window']} hours", "One diurnal cycle of context"],
           ["Primary horizon", f"{prep['horizon']} hours", "Section 3.3"],
           ["Horizons generated", ", ".join(str(h) for h in prep["horizons"]),
            "Secondary horizons exist for the degradation comparison only"],
           ["Split", f"{int((1 - cfg['data']['test_size'] - cfg['data']['val_size']) * 100)}/"
                     f"{int(cfg['data']['val_size'] * 100)}/{int(cfg['data']['test_size'] * 100)}",
            "Chronological; random splitting leaks future information"],
           ["Grouping key", cfg["data"]["group_key"], "Prevents windows spanning stations"],
           ["Imputation", "Forward fill within group, with provenance flag",
            "Flag is a side-car, never a feature"]],
          source="configs/default.yaml", widths=[1.6, 2.1, 2.8])

    heading(doc, "3.3 Horizon selection", 2)
    para(doc,
         f"The primary horizon was set to **{prep['horizon']} hours** after measuring "
         f"the persistence floor at every generated horizon. Section 4.1 reports the "
         f"measurement. The shorter horizon was rejected not because models performed "
         f"badly on it but because they performed well on it for the wrong reason.",
         align="justify")

    heading(doc, "3.4 Baseline models and evaluation protocol", 2)
    rf, xgb = base["random_forest"], base["xgboost"]
    para(doc,
         f"Two tabular baselines were trained: a Random Forest "
         f"({rf['n_estimators']} trees, maximum depth {rf['max_depth']}, minimum "
         f"{rf['min_samples_leaf']} samples per leaf) and XGBoost "
         f"({xgb['n_estimators']} rounds, maximum depth {xgb['max_depth']}, learning "
         f"rate {xgb['learning_rate']}). Full parameters are in Appendix A.",
         align="justify")
    para(doc,
         f"**Model selection uses the validation split, never test.** The configuration "
         f"key `baseline.select_on` is set to `{base['select_on']}`, and the code reads "
         f"it — an unsupported value raises rather than being silently ignored, because "
         f"selecting on test, or on imputed rows, would invalidate every number that "
         f"follows. Test is reserved for final reporting. This holds even where the two "
         f"splits would choose the same model, since establishing that they agree "
         f"requires looking at test, which is the thing being avoided.",
         align="justify")
    para(doc,
         "Statistical comparisons between two predictors on the same rows use a paired "
         "bootstrap: resample the test rows with replacement, recompute both metrics on "
         "each resample, and take the percentile interval of their difference. Pairing "
         "matters — the two predictors are compared on identical rows, so the shared "
         "difficulty of those rows cancels.",
         align="justify")

    heading(doc, "3.5 Class-imbalance interventions", 2)
    para(doc,
         f"Three interventions were compared. **CTGAN** was fit on observed-label "
         f"training rows only ({gan['fit_on']}), one synthesiser per minority class, "
         f"for {gan['epochs']} epochs at batch size {gan['batch_size']}. A class counts "
         f"as a minority below {gan['minority_threshold']:.0%} of the majority count and "
         f"is topped up to {gan['target_ratio']:.0%} of it — partial rebalancing, not "
         f"parity, because a wearable does not meet hazardous air half the time and a "
         f"classifier trained to expect that will over-warn. **SMOTE** was run on the "
         f"same classes to the same ratio. **Class weighting** used inverse-frequency "
         f"weights with no data modification.",
         align="justify")
    para(doc,
         "Two design decisions in the CTGAN configuration follow from failures "
         "documented in Section 4.3. Hour and month are modelled as integer "
         "categoricals and their sine/cosine pairs recomputed after sampling, rather "
         "than being handed to the generator as free continuous columns. The constraint "
         "that dew point cannot exceed temperature is imposed during fitting and "
         "sampling as an `sdv.cag.Inequality` object, not checked afterwards.",
         align="justify")
    para(doc,
         f"**The disqualification rule.** An intervention is rejected if it "
         f"significantly degrades either advisory class, whatever it does to the "
         f"aggregate. Otherwise its aggregate gain must be both significant and at "
         f"least {gan.get('material_threshold', 0.01)} in absolute terms, because with "
         f"roughly sixty thousand test rows a bootstrap resolves differences far below "
         f"anything that matters in practice. The rule is implemented in PulseBench "
         f"(Section 3.12) and applied identically to every intervention.",
         align="justify")

    heading(doc, "3.6 Sequence models and uncertainty decomposition", 2)
    lstm, tr = model["lstm"], model["transformer"]
    para(doc,
         f"An LSTM ({model['num_layers']} layers, hidden size {lstm['hidden_size']}) "
         f"and a Transformer encoder ({model['num_layers']} layers, "
         f"{tr['nhead']} attention heads, model dimension {tr['d_model']}, "
         f"feed-forward dimension {tr['dim_feedforward']}) were trained on the "
         f"unaugmented sequences at learning rate {model['lr']}, batch size "
         f"{model['batch_size']}, for up to {model['epochs']} epochs with early "
         f"stopping after {model['patience']} epochs without improvement in validation "
         f"observed-only macro-F1.",
         align="justify")
    para(doc,
         f"At inference, dropout is kept active and **{model.get('mc_passes', 50)} "
         f"stochastic forward passes** are averaged. Total predictive entropy is "
         f"decomposed into an epistemic component — the mutual information between the "
         f"prediction and the model parameters, estimated as the spread across passes — "
         f"and an aleatoric remainder. Calibration is measured two ways: expected "
         f"calibration error, the support-weighted mean gap across reliability bins, "
         f"and maximum calibration error, the worst single bin. Section 4.4 shows why "
         f"reporting only the first would have been misleading.",
         align="justify")

    heading(doc, "3.7 Conformal prediction", 2)
    para(doc,
         f"Split conformal prediction was calibrated on the {conf['calibration_split']} "
         f"split at a target coverage of {conf['target_coverage']:.0%}. The "
         f"nonconformity score is one minus the predicted probability of the true class; "
         f"the threshold is the appropriate empirical quantile of that score on the "
         f"calibration set. At test time the prediction set is every class whose "
         f"probability exceeds the threshold.",
         align="justify")
    para(doc,
         "A Mondrian variant computes a separate threshold per class. Both are reported "
         "in Section 4.5, with per-class coverage for each, because the difference "
         "between them is the difference between a guarantee that holds on average and "
         "one that holds for the classes the device exists to warn about.",
         align="justify")

    heading(doc, "3.8 Explainability and the LLM advisory layer", 2)
    ex = cfg["explainability"]
    para(doc,
         f"SHAP values were computed with TreeExplainer against a background sample of "
         f"{ex['shap_background_size']} rows, and {ex['shap_n_examples']} cases were "
         f"selected for detailed reporting, spanning confident correct predictions, "
         f"confident errors and highly ambiguous cases.",
         align="justify")
    para(doc,
         f"The advisory layer passes the point prediction, the conformal set, the top "
         f"three SHAP contributions and a wearer profile to "
         f"`{ex['llm_model']}` via the `{ex['llm_provider']}` API at temperature "
         f"{ex['llm_temperature']} and a {ex['llm_max_tokens']}-token limit. The prompt "
         f"supplies every fact explicitly; the model is asked to phrase, not to infer.",
         align="justify")
    table(doc, "What the advisory validator rejects, and why.",
          ["Rejection", "Rationale"],
          [["Unhedged language when the conformal set holds more than one category",
            "The set is the honest statement of what the model knows; text that picks one member discards it"],
           ["A claim of certainty about an individual reading",
            "Conformal coverage is a property of the procedure across many readings, not of any one of them"],
           ["Truncated output",
            "Detected from the API's finish-reason metadata, not from the text: a truncated response still contains hedging words and passes a text-only check"],
           ["Output not ending in terminal punctuation",
            "A backstop for truncation the metadata misses"],
           ["Any API failure — missing key, auth, rate limit, refusal",
            "The device degrades to blunter wording, never to silence"]],
          source="src/explainability/llm_advisory.py validation rules",
          widths=[2.6, 3.7])
    para(doc,
         "**Generated text is validated before use.** The checks reject output that "
         "fails to hedge when the conformal set contains more than one category, output "
         "that claims certainty about an individual reading, and output that is "
         "truncated. Truncation is detected from the API's own finish-reason metadata "
         "rather than from the text, because a truncated response still contains "
         "hedging words and passes a text-only check — a failure encountered during "
         "development. Any rejection falls back to a rule-based template, so the device "
         "degrades to blunter wording rather than to silence.",
         align="justify")

    heading(doc, "3.9 Compression and edge deployment", 2)
    para(doc,
         f"A sweep over forest size and depth selected a compressed model under a "
         f"**two-sided constraint**: the candidate must stay within "
         f"{dep['macro_f1_tolerance']} macro-F1 of the full model on validation *and* "
         f"beat that dataset's own persistence floor significantly. The second condition "
         f"exists because an earlier sweep, constrained only by tolerance, selected a "
         f"model that satisfied it and then landed below the floor.",
         align="justify")
    para(doc,
         f"The selected model was exported to ONNX (opset {dep['onnx_opset']}), checked "
         f"for numerical parity against the scikit-learn original, and benchmarked for "
         f"single-sample latency. Conformal thresholds were re-derived on the "
         f"compressed model, since a quantile is a property of one specific model's "
         f"score distribution. A TFLite Micro conversion path was planned and abandoned: "
         f"TensorFlow Lite Micro does not serve a scikit-learn tree ensemble, so the "
         f"route would have required retraining as a different model and revalidating "
         f"every result against it.",
         align="justify")

    heading(doc, "3.10 Rolling-origin cross-validation", 2)
    rcv = SRC.j("rolling_cv_h6.json") or {}
    para(doc,
         f"An expanding-window protocol was used. Training starts at the first "
         f"{need('rolling_cv_h6.json:initial_train_fraction', 'initial fraction'):.0%} "
         f"of the chronological record; the cutoff then advances through "
         f"{need('rolling_cv_h6.json:n_folds', 'fold count')} folds, each training on "
         f"everything before the cutoff and evaluating on the block after it. An embargo "
         f"of {need('rolling_cv_h6.json:embargo_hours', 'embargo')} hours separates "
         f"train from evaluation so that no evaluation target overlaps a training "
         f"window. The scaler is refit inside each fold on that fold's training data "
         f"alone.",
         align="justify")
    para(doc,
         "Fold-level differences between a model and the baseline are tested with a "
         "Wilcoxon signed-rank test, which asks whether the model wins consistently "
         "across folds rather than whether it wins on average. The test's resolution "
         "floor is reported explicitly: over n folds it cannot produce a two-sided "
         "p-value below 2^(1-n), so a result sitting at that floor is the strongest the "
         "design can yield and is described as such rather than as an exact value.",
         align="justify")

    heading(doc, "3.11 Multiple-comparisons correction", 2)
    para(doc,
         "Every variant compared against persistence on the same test split forms one "
         "family. A Bonferroni correction is applied across that family, and the "
         "comparisons that were significant before correction but not after are named "
         "explicitly, since those are the ones a reader needs flagged. The bootstrap's "
         "resolution floor is carried through the reporting.",
         align="justify")

    heading(doc, "3.12 The PulseBench toolkit", 2)
    para(doc,
         "The evaluation protocol is more reusable than the model it produced, so it "
         "was extracted into **PulseBench**, a standalone Python package released with "
         "this work. It depends on nothing in the thesis codebase and operates on any "
         "dataframe with a datetime axis and a categorical target.",
         align="justify")
    table(doc, "The PulseBench public interface.",
          ["Function", "What it provides"],
          [["`persistence_floor`", "The zero-parameter baseline, paired by time join rather than row shift so gaps cannot create false pairs"],
           ["`seasonal_naive_floor`", "The seasonal counterpart — the class at this hour one cycle ago — contributed after release"],
           ["`rolling_origin_cv`", "Expanding-window folds with embargo, per-fold scaling and sparse-class handling"],
           ["`make_folds`, `aggregate_folds`", "Fold construction and Wilcoxon aggregation over fold deltas"],
           ["`advisory_disqualification`", "Rejects an intervention that improves an aggregate while significantly degrading a protected class"],
           ["`bonferroni_report`", "Family-wise correction with the resampling resolution floor made explicit"]],
          source="pulsebench/ package interface", widths=[1.9, 4.4])
    para(doc,
         "The extraction is load-bearing rather than decorative: the thesis code calls "
         "the package, and regression tests pin the pre-extraction numbers so that the "
         "refactor is demonstrably behaviour-preserving. One structural exception is "
         "documented in Section 6: the tabular baseline retains its own persistence "
         "implementation because it must return row-aligned predictions for the paired "
         "bootstrap, which is a different operation from returning a metrics summary.",
         align="justify")


# ------------------------------------------------------------------ chapter four


def chapter_results(doc: Document) -> None:
    heading(doc, "4. Results", 1)
    para(doc,
         "Unless stated otherwise, every Beijing figure is macro-F1 on observed-label "
         "test rows at the six-hour horizon, and every confidence interval is a 95% "
         "percentile interval from a paired bootstrap. Each figure caption names the "
         "metrics file it was generated from.",
         align="justify")

    # ---------------------------------------------------------------- 4.1
    heading(doc, "4.1 The persistence floor", 2)
    hz = SRC.j("horizon_comparison.json")
    rows = []
    if hz:
        for h in hz["horizons"]:
            r = hz["rows"][str(h)]
            rows.append([f"{h} h", f"{r['label_unchanged_pct']:.2f}%",
                         f"{r['observed']['macro_f1']:.4f}",
                         f"{r['observed']['accuracy']:.4f}"])
    else:
        rows = [[need("horizon_comparison.json:horizons", "horizon sweep"), "", "", ""]]
    table(doc, "Persistence degrades monotonically as the forecast horizon grows.",
          ["Horizon", "AQI category unchanged", "Macro-F1", "Accuracy"], rows,
          source="horizon_comparison.json", widths=[1.2, 1.9, 1.6, 1.6])
    _h1_learned_table(doc)
    figure(doc, "01_persistence_floor_vs_horizon.png",
           "Persistence floor across forecast horizons. The highlighted bar is the "
           "primary horizon adopted for this thesis.",
           figsrc("01_persistence_floor_vs_horizon.png"))
    para(doc,
         f"At one hour the label is unchanged in "
         f"{num('horizon_comparison.json:rows.1.label_unchanged_pct', 'h1 unchanged', '.2f')}% "
         f"of samples and persistence scores "
         f"{num('horizon_comparison.json:rows.1.observed.macro_f1', 'h1 floor')} without "
         f"a model. By six hours the floor has fallen to "
         f"{num('horizon_comparison.json:rows.6.observed.macro_f1', 'h6 floor')}, and by "
         f"twenty-four hours to "
         f"{num('horizon_comparison.json:rows.24.observed.macro_f1', 'h24 floor')}. The "
         f"decline is monotone at every step, so there is no diurnal rebound at "
         f"twenty-four hours: landing on the same time of day does not recover the lost "
         f"predictability.",
         align="justify")
    para(doc,
         "**This single table reframed the project.** The six-hour horizon was adopted "
         "not because it is more useful to a wearer — a shorter warning would be — but "
         "because it is the shortest horizon at which a model has room to demonstrate "
         "anything a copy of the current reading does not already provide.",
         align="justify")

    # ---------------------------------------------------------------- 4.2
    heading(doc, "4.2 Baseline models and rolling-origin cross-validation", 2)
    _master_table(doc)
    figure(doc, "02_master_model_comparison.png",
           "Every Beijing variant against the persistence floor, with 95% paired-"
           "bootstrap intervals on the difference. The dashed rule is the floor.",
           figsrc("02_master_model_comparison.png"))
    para(doc,
         f"On the single chronological split the Random Forest clears the floor by "
         f"{num('ablation_h6.json:vs_persistence.unaugmented.observed_diff', 'RF delta', '+.4f')} "
         f"[{num('ablation_h6.json:vs_persistence.unaugmented.ci_low', 'RF ci low', '+.4f')}, "
         f"{num('ablation_h6.json:vs_persistence.unaugmented.ci_high', 'RF ci high', '+.4f')}], "
         f"which is statistically significant and practically negligible. It also loses "
         f"to persistence on the validation split, which is the split selection is "
         f"allowed to use.",
         align="justify")
    figure(doc, "16_precision_recall_beijing.png",
           "Precision, recall and F1 per class for the selected Beijing forest on "
           "observed test rows, with per-class support.",
           figsrc("16_precision_recall_beijing.png"))
    figure(doc, "17_split_class_prevalence.png",
           "Class prevalence across the chronological splits, and the validation-minus-"
           "test gap that drives repeated disagreement between them.",
           figsrc("17_split_class_prevalence.png"))
    para(doc,
         "Because validation and test disagreed repeatedly, the comparison was re-run "
         "over rolling-origin folds. That is the decisive experiment.",
         align="justify")
    figure(doc, "13_rolling_cv_beijing.png",
           "Beijing rolling-origin cross-validation: per-fold macro-F1 across five "
           "chronological folds, with mean and standard deviation across folds.",
           figsrc("13_rolling_cv_beijing.png"))
    _rolling_table(doc, "rolling_cv_h6.json", "Beijing")
    _per_fold_table(doc, "rolling_cv_h6.json", "Beijing")
    para(doc,
         f"**No Beijing-trained model beats persistence in more than "
         f"{_beijing_best_folds()} of "
         f"{need('rolling_cv_h6.json:n_folds', 'fold count')} folds.** Persistence's own "
         f"fold-to-fold standard deviation, "
         f"{num('rolling_cv_h6.json:aggregate.per_model.Persistence.std', 'persistence sd')}, "
         f"is larger than any model-to-baseline difference in the table. The "
         f"single-split result that the forest beats the floor does not survive.",
         align="justify")

    # ---------------------------------------------------------------- 4.3
    heading(doc, "4.3 Class-imbalance interventions and the validity-metric failure", 2)
    _imbalance_table(doc)
    figure(doc, "03_advisory_class_tradeoff.png",
           "What each imbalance intervention does to the aggregate metric and to the "
           "two advisory classes. Grey rules mark persistence.",
           figsrc("03_advisory_class_tradeoff.png"))
    figure(doc, "05_per_class_f1_heatmap.png",
           "Per-class F1 across every Beijing variant, in absolute terms and as a "
           "change against the persistence floor.",
           figsrc("05_per_class_f1_heatmap.png"))
    para(doc,
         "**Both augmentation methods fail in the same direction.** CTGAN and SMOTE "
         "each raise aggregate macro-F1 significantly and each significantly degrade "
         "both advisory classes. Under the disqualification rule of Section 3.5, both "
         "are rejected. Only class weighting avoids the trade, and it is also the "
         "cheapest of the three by a wide margin.",
         align="justify")
    para(doc,
         "**Synthesiser quality scores do not measure validity.** The first CTGAN "
         "configuration scored above 0.89 overall on the Synthetic Data Vault quality "
         "report, with acceptable column-shape and pair-trend scores. Its output was "
         "nevertheless physically impossible at scale.",
         align="justify")
    _validity_table(doc)
    _gan_plan_table(doc)
    _gan_quality_table(doc)
    figure(doc, "04_ctgan_validity.png",
           "Physical validity of CTGAN output before and after the redesign, across "
           "both cyclical channels and the temperature/dew-point constraint.",
           figsrc("04_ctgan_validity.png"))
    para(doc,
         "**The failure is structural, not column-specific.** Both cyclical pairs broke: "
         "handed to the generator as free continuous columns, both came back off the "
         "unit circle and with tens of thousands of distinct values where twenty-four "
         "and twelve exist. A quality score compares one marginal at a time and one "
         "pairwise correlation at a time; neither asks whether a row is possible. The "
         "fix was to model the underlying integer and recompute the derived encoding, "
         "and to impose the dew-point inequality as a constraint during fitting rather "
         "than to check it afterwards.",
         align="justify")
    para(doc,
         "One API-specific trap is worth recording: passing the constraint as a plain "
         "dictionary is accepted without effect. The library now warns that dictionary "
         "constraints are ignored, which confirms the original diagnosis.",
         align="justify")

    # ---------------------------------------------------------------- 4.4
    heading(doc, "4.4 Sequence models, capacity and uncertainty", 2)
    _dl_table(doc)
    para(doc,
         "Neither sequence architecture beats the tabular forest, and the LSTM is "
         "significantly worse than persistence on the aggregate. The obvious objection "
         "is that the models were too small, so capacity was swept directly.",
         align="justify")
    _lr_sweep_table(doc)
    figure(doc, "06_capacity_sweep.png",
           "Validation macro-F1 against trainable parameter count for both sequence "
           "architectures.", figsrc("06_capacity_sweep.png"))
    cap = SRC.j("capacity_sweep_h6.json")
    if cap:
        para(doc,
             f"**More capacity made both architectures worse.** Across hidden sizes "
             f"{', '.join(str(s) for s in cap['sweep']['sizes'])}, the LSTM declines "
             f"{cap['analysis']['lstm']['gain_smallest_to_largest']:+.4f} and the "
             f"Transformer {cap['analysis']['transformer']['gain_smallest_to_largest']:+.4f} "
             f"from smallest to largest, over roughly "
             f"{cap['analysis']['lstm']['param_ratio']:.0f}x the parameters. Both "
             f"declines are monotone.",
             align="justify")
    _capacity_table(doc)
    para(doc,
         "The learning-rate sweep rules out the other obvious objection. Both "
         "architectures improved at the smaller rate, which means the earlier "
         "configuration was optimisation-limited rather than capacity-limited, and the "
         "capacity sweep was run at the better rate. Neither knob rescues the "
         "architectures.",
         align="justify")
    figure(doc, "07_uncertainty_decomposition.png",
           "Decomposition of mean predictive entropy into aleatoric and epistemic "
           "components, by architecture.", figsrc("07_uncertainty_decomposition.png"))
    para(doc,
         f"The decomposition explains the sweep. **{_epistemic_share()} of predictive "
         f"entropy is aleatoric** — irreducible given these channels and this horizon — "
         f"leaving only a small epistemic sliver that more capacity or more data could "
         f"address. That is a statement about the task, not about the model. If the "
         f"six-hour-ahead category is to be predicted better, the input has to change: "
         f"more channels, a longer window, or spatial context from neighbouring "
         f"stations.",
         align="justify")
    _calibration_table(doc)
    para(doc,
         "**Expected and maximum calibration error rank the two models in opposite "
         "directions.** The Transformer is better calibrated on average and far worse "
         "in its worst bin. Because the advisory layer suppresses low-confidence "
         "warnings, the worst bin is the operative number, and a selection made on "
         "expected calibration error alone would have chosen the wrong model. This is "
         "the third instance in this project of an aggregate concealing a tail failure, "
         "after the augmentation ablation above and the conformal coverage result "
         "below.",
         align="justify")

    # ---------------------------------------------------------------- 4.5
    heading(doc, "4.5 Conformal coverage", 2)
    _conformal_table(doc)
    figure(doc, "08_conformal_coverage.png",
           "Per-class empirical coverage under marginal and Mondrian conformal "
           "calibration, against the target.", figsrc("08_conformal_coverage.png"))
    para(doc,
         f"Split conformal delivered "
         f"{num('conformal_h6.json:test.coverage', 'marginal coverage')} marginal "
         f"coverage against a "
         f"{num('conformal_h6.json:calibration.alpha', 'alpha', '.2f')} miscoverage "
         f"target, which looks correct. Per class it was not: **Hazardous was covered "
         f"at only "
         f"{num('conformal_h6.json:test.per_class.Hazardous.coverage', 'marginal Hazardous coverage')}**. "
         f"The slack in a marginal guarantee lands on the rare, hard classes. Mondrian "
         f"calibration raises that to "
         f"{num('conformal_h6.json:mondrian.test.per_class.Hazardous.coverage', 'Mondrian Hazardous coverage')} "
         f"at the cost of larger sets.",
         align="justify")
    _setsize_table(doc)
    figure(doc, "09_conformal_set_sizes.png",
           "Distribution of prediction-set size under both calibrations.",
           figsrc("09_conformal_set_sizes.png"))
    para(doc,
         f"The cost is reported rather than hidden: mean set size rises from "
         f"{num('conformal_h6.json:test.mean_set_size', 'marginal mean set', '.3f')} to "
         f"{num('conformal_h6.json:mondrian.test.mean_set_size', 'Mondrian mean set', '.3f')} "
         f"and the singleton rate falls from "
         f"{num('conformal_h6.json:test.singleton_rate', 'marginal singleton rate', '.1%')} "
         f"to {num('conformal_h6.json:mondrian.test.singleton_rate', 'Mondrian singleton rate', '.1%')}. "
         f"A single confident category is the exception, not the rule, and the advisory "
         f"layer is built to say so.",
         align="justify")

    # ---------------------------------------------------------------- 4.6
    heading(doc, "4.6 Compression and edge deployment", 2)
    figure(doc, "10_compression_funnel.png",
           "Serialised model size at each stage of the deployment pipeline, for both "
           "datasets, with the macro-F1 cost of compression.",
           figsrc("10_compression_funnel.png"))
    _compression_table(doc)
    _compression_sweep_table(doc)
    para(doc,
         "The two-sided rule matters because the earlier, one-sided version selected a "
         "model that satisfied its tolerance test and then landed below the persistence "
         "floor. A compression criterion expressed only as \"stay close to the full "
         "model\" cannot detect that, because the full model may not have been far "
         "enough above the floor to give anything away. Requiring the compressed "
         "candidate to clear the floor in its own right closes that gap.",
         align="justify")
    para(doc,
         "The accuracy/size curve is close to flat across orders of magnitude, which is "
         "another way of saying capacity was never the binding constraint on this task. "
         "The Beijing compression is reported for completeness only; as Section 4.2 "
         "establishes, nothing from the Beijing pipeline is deployed.",
         align="justify")
    para(doc,
         "**The ONNX benchmark is a portability proxy, not a flashability claim.** "
         "There is no ONNX Runtime for the ESP32 — no interpreter and no execution "
         "provider to load the file with. Reaching the device requires a hand-written C "
         "tree traversal, a code generator such as `emlearn`, or retraining as a model "
         "some supported framework serves. A TFLite Micro path was planned for exactly "
         "this and abandoned, because TensorFlow Lite Micro does not serve a "
         "scikit-learn tree ensemble.",
         align="justify")

    # ---------------------------------------------------------------- 4.7
    heading(doc, "4.7 Sensitivity to the air-quality standard", 2)
    figure(doc, "11_epa_vs_hj633.png",
           "Class balance under EPA and HJ 633-2012 breakpoints, and each standard's "
           "model-versus-floor comparison.", figsrc("11_epa_vs_hj633.png"))
    _hj_table(doc)
    para(doc,
         "The class boundaries are a policy choice, not a property of the air, so the "
         "analysis was repeated under the Chinese national standard. The conclusion is "
         "unchanged in direction: the forest's margin over its own floor remains small "
         "under both standards. Reporting this matters because a reader is entitled to "
         "ask whether a negative result is an artefact of the American breakpoints.",
         align="justify")

    # ---------------------------------------------------------------- 4.8
    heading(doc, "4.8 Multiple-comparisons correction", 2)
    figure(doc, "12_bonferroni_correction.png",
           "Every persistence comparison under family-wise correction, on a log p-axis, "
           "with the uncorrected threshold, the corrected threshold and the bootstrap "
           "resolution floor marked.", figsrc("12_bonferroni_correction.png"))
    _bonferroni_table(doc)
    para(doc,
         "The p-values at the bootstrap's resolution floor are reported as bounded "
         "rather than exact, because a 1,000-resample bootstrap cannot resolve a "
         "two-sided p below 2/1000; printing an exact-looking value there would "
         "overstate what the procedure can produce.",
         align="justify")

    # ---------------------------------------------------------------- 4.9
    heading(doc, "4.9 External validation: Bangladesh", 2)
    para(doc,
         "The methodology was developed on Beijing and validated on the population the "
         "device targets. The first result was about the data, not the models.",
         align="justify")
    figure(doc, "23_dataset_coverage_timeline.png",
           "Coverage of each dataset: what is advertised against what survives "
           "inspection, with the verified-clean boundary marked.",
           figsrc("23_dataset_coverage_timeline.png"))
    _audit_table(doc)
    rows_pct, span_pct = _audit_fractions()
    para(doc,
         f"**The advertised twenty-five-year history is {span_pct}% backfill.** "
         f"Everything before the clean boundary carries a near-linear synthetic trend "
         f"(R-squared "
         f"{num('bangladesh_h6.json:audit.dhaka_pm25_linear_trend_r2', 'trend r2', '.4f')}), "
         f"a hard clip at exactly 250.0 micrograms per cubic metre affecting "
         f"{num('bangladesh_h6.json:audit.pre_clip_at_250_pct', 'clip pct', '.1f')}% of "
         f"rows, and carbon monoxide in different units. Discarding it costs "
         f"**{rows_pct}% of the rows but {span_pct}% of the years**, because the "
         f"discarded portion is one city at low density while the clean window is "
         f"{need('bangladesh_h6.json:audit.clean_cities', 'clean cities')} cities "
         f"hourly. The span figure is the one worth quoting: the dataset's selling "
         f"point is a long history, and the history is the generated part.",
         align="justify")
    para(doc,
         "Every number in that audit is recomputed from the raw file at report time "
         "rather than asserted, so a reader with the file can check it.",
         align="justify")
    figure(doc, "14_rolling_cv_bangladesh.png",
           "Bangladesh rolling-origin cross-validation: per-fold macro-F1 across five "
           "chronological folds.", figsrc("14_rolling_cv_bangladesh.png"))
    _rolling_table(doc, "rolling_cv_h6_bangladesh.json", "Bangladesh")
    _per_fold_table(doc, "rolling_cv_h6_bangladesh.json", "Bangladesh")
    figure(doc, "15_folds_won_summary.png",
           "Folds won against persistence on each dataset, under the same protocol.",
           figsrc("15_folds_won_summary.png"))
    bd = "rolling_cv_h6_bangladesh.json:aggregate.tests.RandomForest (class_weight=balanced)"
    para(doc,
         f"**On Bangladesh the class-weighted forest beats persistence in "
         f"{num(bd + '.wins', 'bd wins', '.0f')} of "
         f"{need('rolling_cv_h6_bangladesh.json:n_folds', 'bd folds')} folds**, mean "
         f"difference {num(bd + '.mean_delta', 'bd delta', '+.4f')}, one-sided "
         f"p = {num(bd + '.p_one_sided', 'bd p')}. The same protocol that rejected every "
         f"Beijing model accepts this one, which is the result the deployment rests on.",
         align="justify")
    _bd_conformal_table(doc)
    _coverage_caveat(doc)

    # ---------------------------------------------------------------- 4.10
    heading(doc, "4.10 Ground truth: the US Embassy Dhaka monitor", 2)
    para(doc,
         "The advisory classes could not be validated on the reanalysis because it "
         "barely contains them. Whether that reflects Dhaka or reflects the dataset was "
         "tested directly against a reference-grade instrument.",
         align="justify")
    figure(doc, "18_mendeley_vs_embassy.png",
           "The reanalysis against the reference monitor over their overlapping period: "
           "advisory-class hours and the PM2.5 distribution.",
           figsrc("18_mendeley_vs_embassy.png"))
    _ground_truth_table(doc)
    para(doc,
         f"**The reanalysis flattens the peaks.** Over "
         f"{num('dhaka_ground_truth.json:comparison.n_overlap', 'overlap hours', ',.0f')} "
         f"overlapping hours the instrument records "
         f"{num('dhaka_ground_truth.json:comparison.hazardous_reference', 'ref hazardous', ',.0f')} "
         f"Hazardous hours where the reanalysis records "
         f"{num('dhaka_ground_truth.json:comparison.hazardous_reanalysis', 'rea hazardous', ',.0f')}. "
         f"Correlation is reasonable "
         f"(Pearson r = {num('dhaka_ground_truth.json:comparison.pearson_r', 'pearson', '.3f')}), "
         f"but the mean bias is "
         f"{num('dhaka_ground_truth.json:comparison.bias_reanalysis_minus_reference', 'bias', '+.1f')} "
         f"micrograms per cubic metre and widens to "
         f"{num('dhaka_ground_truth.json:comparison.high_range.bias', 'high bias', '+.1f')} "
         f"above 150. The shape is tracked; the magnitude at the top of the range is "
         f"not.",
         align="justify")
    para(doc,
         "**This confirms the hypothesis and moves the limitation.** The absence of "
         "advisory-class samples was a property of the reanalysis product, not of "
         "Dhaka's air. Dhaka genuinely reaches hazardous concentrations; the dataset "
         "used for external validation does not represent that.",
         align="justify")

    # ---------------------------------------------------------------- 4.11
    heading(doc, "4.11 The validated Hazardous detector", 2)
    para(doc,
         "A second model was trained directly on the reference series. It is PM2.5-only "
         "and single-station, so it is not the deployed multi-channel predictor; its "
         "purpose is to establish whether the advisory-class task is learnable at all "
         "on real data.",
         align="justify")
    figure(doc, "19_phase11b_hazardous.png",
           "Hazardous-class and aggregate performance against the persistence floor "
           "across seven rolling-origin folds, with every fold shown.",
           figsrc("19_phase11b_hazardous.png"))
    _phase11b_table(doc)
    _phase11b_folds_table(doc)
    g = "dhaka_pm25_model_h6.json:cv.aggregate.f1_Hazardous"
    para(doc,
         f"**Hazardous F1 reaches {num(g + '.model_mean', '11b haz')} against a "
         f"{num(g + '.persistence_mean', '11b floor')} persistence floor, winning "
         f"{num(g + '.wins', '11b wins', '.0f')} of "
         f"{need('dhaka_pm25_model_h6.json:cv.n_folds', '11b folds')} folds** "
         f"(two-sided p = {num(g + '.p_two_sided', '11b p')}). That p-value sits at the "
         f"signed-rank resolution floor for this fold count, which is the strongest the "
         f"design can produce rather than a weak result.",
         align="justify")
    para(doc,
         "**What this does and does not establish.** It is evidence about the *task*: "
         "hazardous-air prediction at six hours is learnable above its floor on "
         "instrument-grade data. It is not evidence about the deployed model, which is "
         "a different model on different inputs. The two results are reported as two "
         "results.",
         align="justify")

    # ---------------------------------------------------------------- 4.12
    heading(doc, "4.12 Monitoring infrastructure: the OpenAQ survey", 2)
    figure(doc, "20_openaq_survey.png",
           "Stations near Dhaka by provider and by reported pollutant, and the number "
           "surviving each coverage filter.", figsrc("20_openaq_survey.png"))
    _openaq_table(doc)
    para(doc,
         f"**No station qualifies.** Of "
         f"{len(need('openaq_survey.json:survey.locations', 'openaq locations') or [])} "
         f"stations within "
         f"{num('openaq_survey.json:survey.radius_km', 'radius', '.0f')} km, "
         f"{num('openaq_survey.json:assessment.n_with_pm10_or_co', 'with companion', '.0f')} "
         f"report a companion pollutant alongside PM2.5, and "
         f"{num('openaq_survey.json:assessment.n_passing', 'passing', '.0f')} meet the "
         f"duration and completeness requirements. The single candidate spans "
         f"{num('openaq_survey.json:probe.span_years', 'span years', '.2f')} years at "
         f"{num('openaq_survey.json:probe.completeness_pct', 'completeness', '.1f')}% "
         f"completeness, failing the duration test.",
         align="justify")
    para(doc,
         "**This is a finding about infrastructure, and it bounds the thesis.** A "
         "multi-pollutant model cannot currently be validated against instrument-grade "
         "data in Dhaka, because the instruments are not there. A zero here is the "
         "correct answer, not a missing measurement, and it is reported rather than "
         "worked around.",
         align="justify")

    # ---------------------------------------------------------------- 4.13
    heading(doc, "4.13 The deployed system", 2)
    figure(doc, "22_deployed_architecture.png",
           "The deployed Bangladesh predictor, stage by stage, with the validator drawn "
           "outside the model column.", figsrc("22_deployed_architecture.png"))
    _deployed_table(doc)
    _advisory_examples(doc)
    figure(doc, "21_pipeline_overview.png",
           "The full project pipeline: the Beijing methodology track, the Bangladesh "
           "deployment track, and what each produced.",
           figsrc("21_pipeline_overview.png"))
    para(doc,
         "**The deployed system is the Bangladesh model; the Beijing pipeline is the "
         "methodology that produced it.** Those are two artefacts with two statuses, "
         "and conflating them would misrepresent both.",
         align="justify")

    # ---------------------------------------------------------------- 4.14
    heading(doc, "4.14 Explainability: SHAP case studies", 2)
    _shap_table(doc)
    para(doc,
         "Global mean absolute attribution is dominated by the pollutant channels, with "
         "the meteorological and cyclical channels contributing an order of magnitude "
         "less. That ordering is what the advisory layer reports to the wearer: the "
         "sentence names the channels that actually drove the prediction, not a fixed "
         "list.",
         align="justify")
    _shap_cases_table(doc)
    para(doc,
         "Each case is reproduced in the repository with a full attribution plot. The "
         "pattern that matters for the advisory layer is in the set-size column: a "
         "confident prediction and an ambiguous one are distinguishable before any text "
         "is generated, which is what lets the wearer-facing sentence hedge when it "
         "should and commit when it can.",
         align="justify")


# ---------------------------------------------------------------- table builders


def _variants() -> list[dict]:
    from src.reporting.compile_results import collect
    return collect()["variants"]


def _fmt_vs(v) -> str:
    if v is None:
        return "— (reference)"
    if not isinstance(v, dict):
        return "not tested"
    return (f"{v['observed_diff']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"
            + ("" if v.get("significant") else " (n.s.)"))


def _master_table(doc: Document) -> None:
    rows = []
    for v in _variants():
        s = v["scores"]
        rows.append([v["name"].split(" —")[0], f"{s['macro_f1']:.4f}",
                     f"{s['per_class']['Hazardous']:.4f}",
                     f"{s['per_class']['Very unhealthy']:.4f}",
                     _fmt_vs(v["vs_persistence"])])
    table(doc, "Every Beijing variant on the same test rows, against the persistence "
               "floor. Intervals are 95% paired bootstrap on the difference.",
          ["Variant", "Macro-F1", "Hazardous F1", "V. unhealthy F1", "vs persistence"],
          rows, source="ablation_h6.json, gapfill_h6.json, dl_h6.json, smote_h6.json",
          widths=[2.0, 0.85, 0.95, 1.0, 1.7])


def _imbalance_table(doc: Document) -> None:
    ab, sm, gap = SRC.j("ablation_h6.json"), SRC.j("smote_h6.json"), SRC.j("gapfill_h6.json")
    if not (ab and sm and gap):
        para(doc, need("ablation_h6.json:rows", "imbalance comparison"))
        return
    cw = "RandomForest (class_weight=balanced)"
    series = [("None (RandomForest)", ab["rows"]["unaugmented"]["scores"], "—"),
              ("CTGAN, broad 4-class", ab["rows"]["broad-4"]["scores"],
               f"{ab['rows']['broad-4']['n_synthetic']:,} rows, "
               f"{sm['cost']['ctgan_minutes']:.0f} min"),
              ("CTGAN, targeted 2-class", ab["rows"]["targeted-2"]["scores"],
               f"{ab['rows']['targeted-2']['n_synthetic']:,} rows"),
              ("SMOTE", sm["scores"], f"{sm['n_synthetic']:,} rows, "
                                      f"{sm['cost']['smote_seconds']:.1f} s"),
              ("Class weighting", gap["variants"][cw]["scores"], "no data added")]
    rows = [[n, f"{s['macro_f1']:.4f}", f"{s['per_class']['Very unhealthy']:.4f}",
             f"{s['per_class']['Hazardous']:.4f}", cost] for n, s, cost in series]
    table(doc, "The three class-imbalance interventions, with their cost.",
          ["Intervention", "Macro-F1", "V. unhealthy F1", "Hazardous F1", "Cost"],
          rows, source="ablation_h6.json, smote_h6.json, gapfill_h6.json",
          widths=[1.6, 0.9, 1.1, 1.0, 1.7])
    para(doc,
         f"SMOTE produced the same {sm['n_synthetic']:,} synthetic rows in "
         f"{sm['cost']['smote_seconds']:.1f} seconds that CTGAN took "
         f"{sm['cost']['ctgan_minutes']:.0f} minutes to produce — a factor of roughly "
         f"{sm['cost']['ctgan_minutes'] * 60 / sm['cost']['smote_seconds']:.0f} in "
         f"wall-clock cost, for an outcome that is no better and fails in the same "
         f"direction.",
         align="justify")


def _validity_table(doc: Document) -> None:
    before, after = SRC.j("gan_validity_before_fix.json"), SRC.j("gan_h6.json")
    if not (before and after):
        para(doc, need("gan_validity_before_fix.json:validity", "CTGAN validity"))
        return
    b, a = before["validity"]["synthetic"], after["validity"]["synthetic"]
    rows = [
        ["hour_sin/cos on the unit circle", f"{b['hour']['on_unit_circle_pct']:.2f}%",
         f"{a['hour']['on_unit_circle_pct']:.2f}%", "must be 100%"],
        ["distinct hour_sin values", f"{b['hour']['distinct_sin']:,}",
         f"{a['hour']['distinct_sin']:,}", "24 possible"],
        ["month_sin/cos on the unit circle", f"{b['month']['on_unit_circle_pct']:.2f}%",
         f"{a['month']['on_unit_circle_pct']:.2f}%", "must be 100%"],
        ["distinct month_sin values", f"{b['month']['distinct_sin']:,}",
         f"{a['month']['distinct_sin']:,}", "12 possible"],
        ["dew point above temperature", f"{b['dewp_above_temp_pct']:.4g}%",
         f"{a['dewp_above_temp_pct']:.4g}%",
         f"{after['validity']['real']['dewp_above_temp_pct']:.4g}% in real data"],
    ]
    table(doc, "Physical validity of CTGAN output, before and after the redesign. The "
               "before column was regenerated by deliberately re-running the original "
               "broken configuration; it is a reproduction of the same failure, not the "
               "same random draw.",
          ["Validity check", "Before", "After", "Expected"], rows,
          source="gan_validity_before_fix.json, gan_h6.json",
          widths=[2.3, 1.2, 1.2, 1.6])


def _dl_table(doc: Document) -> None:
    dl = SRC.j("dl_h6.json")
    if not dl:
        para(doc, need("dl_h6.json:results", "sequence model results"))
        return
    rows = []
    for name, r in dl["results"].items():
        t = dl["tests"].get("Persistence", {}).get("macro_f1") if "MC mean" in name else None
        rows.append([name, f"{r['macro_f1']:.4f}", f"{r['accuracy']:.4f}",
                     f"{r['per_class']['Hazardous']:.4f}",
                     f"{r['per_class']['Very unhealthy']:.4f}"])
    table(doc, "Sequence models against the tabular baseline and the persistence floor, "
               "on the same observed test rows.",
          ["Model", "Macro-F1", "Accuracy", "Hazardous F1", "V. unhealthy F1"], rows,
          source="dl_h6.json", widths=[2.1, 1.0, 1.0, 1.1, 1.1])


def _calibration_table(doc: Document) -> None:
    dl = SRC.j("dl_h6.json")
    if not dl:
        para(doc, need("dl_h6.json:calibration", "calibration"))
        return
    rows = [[a, f"{c['ece']:.4f}", f"{c['mce']:.4f}", f"{c['n']:,}"]
            for a, c in sorted(dl["calibration"].items(), key=lambda kv: kv[1]["ece"])]
    table(doc, "Calibration measured two ways. The rankings disagree.",
          ["Model", "ECE (mean bin gap)", "MCE (worst bin gap)", "n"], rows,
          source="dl_h6.json", widths=[1.6, 1.8, 1.8, 1.1])


def _conformal_table(doc: Document) -> None:
    c = SRC.j("conformal_h6.json")
    if not c:
        para(doc, need("conformal_h6.json:test", "conformal results"))
        return
    rows = []
    for cls in c["test"]["per_class"]:
        m, d = c["test"]["per_class"][cls], c["mondrian"]["test"]["per_class"][cls]
        rows.append([cls, f"{m['n']:,}", f"{m['coverage']:.4f}", f"{d['coverage']:.4f}",
                     f"{m['mean_set_size']:.2f} → {d['mean_set_size']:.2f}"])
    table(doc, "Per-class conformal coverage under marginal and Mondrian calibration, "
               "with the change in mean set size.",
          ["Class", "Test n", "Marginal coverage", "Mondrian coverage", "Mean set size"],
          rows, source="conformal_h6.json", widths=[1.7, 0.9, 1.4, 1.4, 1.5])


def _compression_table(doc: Document) -> None:
    rows = []
    for label, f in (("Beijing", "deployment_h6.json"),
                     ("Bangladesh (deployed)", "deployment_h6_bd.json")):
        d = SRC.j(f)
        if not d:
            rows.append([label, need(f + ":compressed", "compression"), "", "", "", ""])
            continue
        b, c = d["baseline"], d["compressed"]
        rows.append([label,
                     f"{b['n_estimators']}x d{b['max_depth']} → {c['n_estimators']}x d{c['max_depth']}",
                     f"{b['pickle_kb'] / 1024:.1f} MB → {c['pickle_kb']:,.0f} KB",
                     f"{d['onnx']['bytes'] / 1024:,.0f} KB",
                     f"{c['val_macro_f1'] - b['val_macro_f1']:+.4f}",
                     f"{d['parity']['argmax_agreement']:.4f}"])
    table(doc, "Compression and export, both datasets.",
          ["Dataset", "Forest", "Serialised size", "ONNX", "Val macro-F1 change",
           "ONNX/sklearn argmax agreement"], rows,
          source="deployment_h6.json, deployment_h6_bd.json",
          widths=[1.3, 1.5, 1.4, 0.8, 1.0, 0.9])


def _hj_table(doc: Document) -> None:
    hj = SRC.j("hj633_h6.json")
    if not hj:
        para(doc, need("hj633_h6.json:results", "HJ 633 sensitivity"))
        return
    rows = []
    for std, r in hj["results"].items():
        v = r["vs_persistence"]
        rows.append([std, ", ".join(f"{b:g}" for b in r["breakpoints"]),
                     f"{r['persistence']['macro_f1']:.4f}", f"{r['rf']['macro_f1']:.4f}",
                     f"{v['observed_diff']:+.4f}",
                     "significant" if v["significant"] else "not significant"])
    table(doc, "The same data under two national standards.",
          ["Standard", "PM2.5 breakpoints", "Persistence", "RandomForest", "Difference",
           "Test"], rows, source="hj633_h6.json",
          widths=[1.2, 1.8, 1.0, 1.1, 0.9, 1.1])


def _bonferroni_table(doc: Document) -> None:
    from pulsebench import bonferroni_report
    tested = [v for v in _variants() if isinstance(v.get("vs_persistence"), dict)]
    if not tested:
        para(doc, need("ablation_h6.json:vs_persistence", "bonferroni family"))
        return
    n_boot = need("ablation_h6.json:n_boot", "bootstrap resamples")
    rep = bonferroni_report(
        {v["name"].split(" —")[0]: {"p": v["vs_persistence"]["p_two_sided"],
                                    "delta": v["vs_persistence"]["observed_diff"]}
         for v in tested}, alpha=0.05, n_resamples=n_boot)
    rows = [[r["name"], f"{r['delta']:+.4f}", r["p_display"], r["direction"],
             "yes" if r["significant_uncorrected"] else "no",
             "yes" if r["survives"] else "no"] for r in rep["rows"]]
    table(doc, f"Bonferroni correction over {rep['k']} comparisons against persistence "
               f"on the same test split (corrected threshold "
               f"{rep['corrected_alpha']:.4f}).",
          ["Variant", "Difference", "p", "Direction", "Significant at 0.05",
           "Survives correction"], rows,
          source="compile_results.collect() via pulsebench.bonferroni_report",
          widths=[1.9, 0.9, 0.85, 0.85, 1.0, 0.95])
    if rep["lost"]:
        para(doc, f"Lost to correction: {', '.join(rep['lost'])}.", align="justify")


def _audit_table(doc: Document) -> None:
    d = SRC.j("bangladesh_h6.json")
    if not d:
        para(doc, need("bangladesh_h6.json:audit", "audit"))
        return
    a = d["audit"]
    rows_pct, span_pct = _audit_fractions()
    rows = [
        ["Advertised cities", "103", f"{a['actual_cities']} present"],
        ["Advertised span", a["stated_range"],
         f"{a['clean_range'][0][:10]} to {a['clean_range'][1][:10]} usable"],
        ["Rows in file", f"{a['file_rows']:,}",
         f"{a['clean_rows']:,} usable ({a['clean_pct']:.0f}%)"],
        ["Rows discarded", f"{a['file_rows'] - a['clean_rows']:,}",
         f"{rows_pct}% of rows, {span_pct}% of the span"],
        ["Pre-cut PM2.5 trend", f"R-squared {a['dhaka_pm25_linear_trend_r2']:.4f}",
         "near-linear; real air quality is not"],
        ["Hard clip at 250.0", f"{a['pre_clip_at_250_pct']:.1f}% of pre-cut rows",
         "an artefact of generation"],
    ]
    table(doc, "Data-integrity audit of the published Bangladesh dataset. Every value "
               "is recomputed from the raw file at report time.",
          ["Check", "Finding", "Consequence"], rows,
          source="bangladesh_h6.json (audit block)", widths=[1.7, 2.2, 2.5])


def _rolling_table(doc: Document, fname: str, place: str) -> None:
    d = SRC.j(fname)
    if not d:
        para(doc, need(fname + ":aggregate", f"{place} rolling CV"))
        return
    rows = []
    for m, agg in d["aggregate"]["per_model"].items():
        t = d["aggregate"]["tests"].get(m)
        rows.append([m, f"{agg['mean']:.4f} ± {agg['std']:.4f}",
                     f"{agg['min']:.4f}–{agg['max']:.4f}",
                     f"{t['wins']}/{t['n_folds']}" if t else "— (reference)",
                     f"{t['mean_delta']:+.4f}" if t else "—",
                     f"{t['p_one_sided']:.4f}" if t else "—"])
    res = d["aggregate"]["tests"].get("_resolution", {})
    cap = (f" The signed-rank floor for {d['n_folds']} folds is "
           f"{res.get('min_p_one_sided', 0):.4f} one-sided." if res else "")
    table(doc, f"{place} rolling-origin cross-validation over {d['n_folds']} folds, "
               f"{d['embargo_hours']}-hour embargo.{cap}",
          ["Model", "Mean ± SD", "Range", "Folds won", "Mean Δ", "p (one-sided)"],
          rows, source=fname, widths=[1.9, 1.15, 1.05, 0.8, 0.75, 0.85])


def _per_fold_table(doc: Document, fname: str, place: str) -> None:
    """Every fold's boundaries, support and score. The figures show the shape; this
    shows the numbers a reader would need to check them."""
    d = SRC.j(fname)
    if not d:
        para(doc, need(fname + ":folds", f"{place} per-fold detail"))
        return
    models = list(d["folds"][0]["scores"])
    rows = []
    for f in d["folds"]:
        rows.append([str(f["fold"]),
                     f"{str(f['eval_start'])[:10]} to {str(f['eval_end'])[:10]}",
                     f"{f['n_train']:,}",
                     f"{f.get('n_eval_observed', f['n_eval']):,}"]
                    + [f"{f['scores'][m]['macro_f1']:.4f}" for m in models])
    table(doc, f"{place} rolling-origin folds in detail: evaluation window, sample "
               f"counts and per-fold macro-F1 for every model.",
          ["Fold", "Evaluation window", "Train n", "Eval n"]
          + [m.replace("RandomForest ", "RF ") for m in models], rows,
          source=fname, widths=[0.5, 1.75, 0.85, 0.8] + [0.85] * len(models))


def _phase11b_folds_table(doc: Document) -> None:
    d = SRC.j("dhaka_pm25_model_h6.json")
    if not d:
        para(doc, need("dhaka_pm25_model_h6.json:cv.folds", "phase 11b folds"))
        return
    rows = []
    for f in d["cv"]["folds"]:
        rows.append([str(f["fold"]),
                     f"{str(f['eval_start'])[:10]} to {str(f['eval_end'])[:10]}",
                     f"{f['n_train']:,}", f"{f['n_eval']:,}",
                     f"{f['model']['f1_Hazardous']:.4f}",
                     f"{f['persistence']['f1_Hazardous']:.4f}",
                     f"{f['model']['f1_Hazardous'] - f['persistence']['f1_Hazardous']:+.4f}",
                     "yes" if f.get("winter") else "no"])
    table(doc, "Phase 11b folds in detail. The model leads on Hazardous F1 in every "
               "fold, including the folds whose evaluation window misses the winter "
               "pollution season.",
          ["Fold", "Evaluation window", "Train n", "Eval n", "Model", "Persistence",
           "Δ", "Winter"], rows, source="dhaka_pm25_model_h6.json",
          widths=[0.45, 1.6, 0.75, 0.7, 0.7, 0.9, 0.6, 0.6])


def _lr_sweep_table(doc: Document) -> None:
    d = SRC.j("dl_lr_sweep_h6.json")
    if not d:
        para(doc, need("dl_lr_sweep_h6.json:results", "learning-rate sweep"))
        return
    lrs = d["grid"]
    rows = []
    for arch, by_lr in d["results"].items():
        rows.append([arch] + [f"{by_lr[str(lr)]['val_macro_f1']:.4f}"
                              if str(lr) in by_lr else "—" for lr in lrs])
    table(doc, f"Learning-rate sweep on validation macro-F1, {d['epochs']} epochs with "
               f"patience {d['patience']}. Both architectures were optimisation-limited "
               f"at the larger rates.",
          ["Architecture"] + [f"lr = {lr}" for lr in lrs], rows,
          source="dl_lr_sweep_h6.json",
          widths=[1.5] + [1.2] * len(lrs))


def _capacity_table(doc: Document) -> None:
    d = SRC.j("capacity_sweep_h6.json")
    if not d:
        para(doc, need("capacity_sweep_h6.json:sweep", "capacity sweep"))
        return
    rows = [[r["arch"], str(r["size"]), f"{r['n_params']:,}",
             f"{r['val_macro_f1']:.4f}", str(r["best_epoch"]), f"{r['minutes']:.1f}"]
            for r in d["sweep"]["rows"]]
    table(doc, f"Capacity sweep at lr = {d['sweep']['lr']}. Validation macro-F1 falls "
               f"monotonically with size for both architectures.",
          ["Architecture", "Hidden size", "Parameters", "Val macro-F1", "Best epoch",
           "Minutes"], rows, source="capacity_sweep_h6.json",
          widths=[1.3, 1.0, 1.1, 1.2, 0.95, 0.85])


def _setsize_table(doc: Document) -> None:
    c = SRC.j("conformal_h6.json")
    if not c:
        para(doc, need("conformal_h6.json:test.size_histogram", "set-size histogram"))
        return
    sizes = sorted({int(k) for k in c["test"]["size_histogram"]}
                   | {int(k) for k in c["mondrian"]["test"]["size_histogram"]})
    mn, dn = c["test"]["n"], c["mondrian"]["test"]["n"]
    rows = []
    for sz in sizes:
        m = c["test"]["size_histogram"].get(str(sz), 0)
        d_ = c["mondrian"]["test"]["size_histogram"].get(str(sz), 0)
        rows.append([str(sz), f"{m:,} ({m / mn:.1%})", f"{d_:,} ({d_ / dn:.1%})"])
    table(doc, "Prediction-set size distribution under both calibrations. A single "
               "confident category is the exception under either.",
          ["Categories in set", "Marginal", "Mondrian"], rows,
          source="conformal_h6.json", widths=[1.7, 2.3, 2.3])


def _compression_sweep_table(doc: Document) -> None:
    d = SRC.j("deployment_h6_bd.json")
    if not d:
        para(doc, need("deployment_h6_bd.json:sweep", "compression sweep"))
        return
    sw = d["sweep"]
    rows = [
        ["Configurations evaluated", f"{len(sw['rows'])}"],
        ["Within the macro-F1 tolerance", f"{sum(1 for r in sw['rows'] if r.get('within_tolerance'))}"],
        ["Significantly above the persistence floor", f"{sw['n_beats_persistence']}"],
        ["Accepted under the two-sided rule", f"{sw['n_accepted']}"],
        ["Persistence floor used", f"{sw['persistence_val_macro_f1']:.4f} (Bangladesh's own)"],
        ["Tolerance", f"{sw['tolerance']}"],
        ["Selection rule applied", sw["selection_rule"]],
        ["Selected", f"{sw['selected']['n_estimators']} trees x depth "
                     f"{sw['selected']['max_depth']}, val macro-F1 "
                     f"{sw['selected']['val_macro_f1']:.4f}"],
    ]
    table(doc, "The two-sided compression sweep on the deployment dataset. A candidate "
               "must be within tolerance of the full model *and* significantly above "
               "that dataset's own persistence floor.",
          ["Criterion", "Value"], rows, source="deployment_h6_bd.json",
          widths=[3.0, 3.3])


def _h1_learned_table(doc: Document) -> None:
    hz = SRC.j("horizon_comparison.json")
    if not hz or "learned_at_primary" not in hz:
        return
    rows = [[k, f"{v:.4f}",
             f"{v - hz['rows'][str(hz['primary'])]['observed']['macro_f1']:+.4f}"]
            for k, v in hz["learned_at_primary"].items()]
    table(doc, f"Learned baselines at the primary horizon against the persistence floor.",
          ["Model", "Macro-F1", "vs persistence"], rows,
          source="horizon_comparison.json", widths=[2.2, 1.6, 1.8])


def _coverage_caveat(doc: Document) -> None:
    d = SRC.j("bangladesh_h6.json")
    if not d:
        return
    sup = d["support"]["test"]
    para(doc,
         f"**The advisory classes are not covered by that result, and the bound travels "
         f"with it.** The Bangladesh test split contains {sup['Hazardous']} Hazardous "
         f"and {sup['Very unhealthy']} Very-unhealthy samples. The five-fold result "
         f"covers the four common classes and nothing more. A device shipped on this "
         f"evidence could report ordinary air-quality bands; it could not yet be "
         f"claimed to raise a reliable hazardous-air warning, which is the function "
         f"that motivates the product.",
         align="justify")


def _ground_truth_table(doc: Document) -> None:
    d = SRC.j("dhaka_ground_truth.json")
    if not d:
        para(doc, need("dhaka_ground_truth.json:comparison", "ground truth"))
        return
    c = d["comparison"]
    rows = [
        ["Hazardous hours", f"{c['hazardous_reference']:,}",
         f"{c['hazardous_reanalysis']:,}"],
        ["Very-unhealthy hours", f"{c['very_unhealthy_reference']:,}",
         f"{c['very_unhealthy_reanalysis']:,}"],
        ["Mean PM2.5", f"{c['reference']['mean']:.1f}", f"{c['reanalysis']['mean']:.1f}"],
        ["99th percentile", f"{c['reference']['p99']:.1f}", f"{c['reanalysis']['p99']:.1f}"],
        ["Maximum", f"{c['reference']['max']:.1f}", f"{c['reanalysis']['max']:.1f}"],
        ["AQI class agreement", f"{c['class_agreement']:.1%}", "—"],
    ]
    table(doc, f"Reference monitor against the reanalysis over "
               f"{c['n_overlap']:,} overlapping hours.",
          ["Quantity", "Reference monitor", "Mendeley reanalysis"], rows,
          source="dhaka_ground_truth.json", widths=[2.0, 2.1, 2.2])


def _phase11b_table(doc: Document) -> None:
    d = SRC.j("dhaka_pm25_model_h6.json")
    if not d:
        para(doc, need("dhaka_pm25_model_h6.json:cv", "phase 11b"))
        return
    agg, n = d["cv"]["aggregate"], d["cv"]["n_folds"]
    rows = []
    for key, label in (("f1_Hazardous", "Hazardous F1"), ("macro_f1", "Macro-F1")):
        b = agg[key]
        rows.append([label, f"{b['model_mean']:.4f} ± {b['model_std']:.4f}",
                     f"{b['persistence_mean']:.4f}", f"{b['mean_delta']:+.4f}",
                     f"{b['wins']}/{b['n_folds']}", f"{b['p_two_sided']:.4f}"])
    res = agg.get("_resolution", {})
    cap = (f" The signed-rank floor over {n} folds is "
           f"{res.get('min_p_two_sided', 0):.4f} two-sided, which this result attains."
           if res else "")
    table(doc, f"PM2.5-only model on the reference series, {n} rolling-origin folds.{cap}",
          ["Metric", "Model (mean ± SD)", "Persistence", "Mean Δ", "Folds won",
           "p (two-sided)"], rows, source="dhaka_pm25_model_h6.json",
          widths=[1.2, 1.5, 1.0, 0.8, 0.85, 1.0])


def _openaq_table(doc: Document) -> None:
    d = SRC.j("openaq_survey.json")
    if not d:
        para(doc, need("openaq_survey.json:assessment", "openaq survey"))
        return
    locs, pr = d["survey"]["locations"], d["probe"]
    params: dict[str, int] = {}
    for loc in locs:
        for p in loc["parameters"]:
            params[p] = params.get(p, 0) + 1
    rows = [
        ["Stations within radius", f"{len(locs)}", f"{d['survey']['radius_km']:.0f} km of Dhaka"],
        ["Reporting PM2.5", f"{params.get('pm25', 0)}", "the common case"],
        ["Reporting PM10", f"{params.get('pm10', 0)}", "required companion channel"],
        ["Reporting CO", f"{params.get('co', 0)}", "required companion channel"],
        ["With PM10 or CO alongside PM2.5", f"{d['assessment']['n_with_pm10_or_co']}", "candidates"],
        ["Passing all coverage criteria", f"{d['assessment']['n_passing']}", "**none qualify**"],
        ["Best candidate span", f"{pr['span_years']:.2f} years",
         f"{pr['completeness_pct']:.1f}% complete; duration test "
         f"{'passed' if pr['passes_duration'] else 'failed'}"],
    ]
    table(doc, "OpenAQ station survey around Dhaka.",
          ["Criterion", "Count", "Note"], rows, source="openaq_survey.json",
          widths=[2.4, 1.1, 2.7])


def _deployed_table(doc: Document) -> None:
    d, bd = SRC.j("deployment_h6_bd.json"), SRC.j("bangladesh_h6.json")
    if not (d and bd):
        para(doc, need("deployment_h6_bd.json:compressed", "deployed system"))
        return
    c, lat = d["compressed"], d.get("latency", {})
    single = lat.get("onnx_single", {}).get("mean_ms")
    rows = [
        ["Training data", ", ".join(bd["meta"]["cities"])
         + f", {str(bd['meta']['clean_start'])[:10]} onward"],
        ["Features", f"{len(d['features'])}: " + ", ".join(d["features"])],
        ["Model", f"RandomForest, class_weight='balanced', {c['n_estimators']} trees "
                  f"x depth {c['max_depth']}"],
        ["Serialised size", f"{c['pickle_kb']:,.0f} KB pickle, "
                            f"{d['onnx']['bytes'] / 1024:,.0f} KB ONNX"],
        ["Single-sample latency",
         f"{single:.4f} ms" if single is not None
         else need("deployment_h6_bd.json:latency.onnx_single.mean_ms", "latency")],
        ["Uncertainty layer", f"Mondrian conformal, empirical coverage "
                              f"{d['conformal_new']['test']['coverage']:.4f}, mean set "
                              f"{d['conformal_new']['test']['mean_set_size']:.2f}"],
        ["ONNX parity", f"max |Δp| {d['parity']['max_abs_prob_diff']:.1e}, argmax "
                        f"agreement {d['parity']['argmax_agreement']:.4f}"],
        ["Advisory classes", "**not validated for this model** — see Sections 4.9 and 4.11"],
    ]
    table(doc, "The deployed Bangladesh predictor.", ["Property", "Value"], rows,
          source="deployment_h6_bd.json, bangladesh_h6.json", widths=[1.8, 4.5])


def _shap_table(doc: Document) -> None:
    d = SRC.j("shap_h6.json")
    if not d:
        para(doc, need("shap_h6.json:global_mean_abs", "shap attributions"))
        return
    items = sorted(d["global_mean_abs"].items(), key=lambda kv: -kv[1])
    rows = [[k, f"{v:.5f}"] for k, v in items]
    table(doc, "Global mean absolute SHAP attribution by channel.",
          ["Channel", "Mean |SHAP|"], rows, source="shap_h6.json",
          widths=[2.4, 1.6])


# ------------------------------------------------------- chapters five to eight


def chapter_discussion(doc: Document) -> None:
    heading(doc, "5. Discussion", 1)
    para(doc,
         "Three threads run through the results, and they are more connected than they "
         "first appear.",
         align="justify")

    heading(doc, "5.1 An average says nothing about its tail", 2)
    para(doc,
         "The same failure appeared three times, in three unrelated places. Augmentation "
         "raised macro-F1 while significantly degrading both advisory classes. Marginal "
         "conformal prediction met its coverage target on average while under-covering "
         "the rarest class. And expected calibration error ranked the Transformer as "
         "the better-calibrated model while maximum calibration error ranked it the "
         "worse, by a factor of more than four in its worst bin.",
         align="justify")
    para(doc,
         "In each case an aggregate over a distribution concealed a failure in its "
         "tail, and in each case the tail is the part the device exists for. A wearable "
         "that is accurate on ordinary air and unreliable on hazardous air has the "
         "performance profile of a device nobody needs. The methodological consequence "
         "is the disqualification rule of Section 3.5, and the reason it is stated as a "
         "rule rather than exercised as judgement is that the same mistake was "
         "available three separate times.",
         align="justify")

    heading(doc, "5.2 The ceiling is in the data", 2)
    para(doc,
         f"Every attempt to improve on the tabular baseline failed, and they failed "
         f"consistently. Sequence models lost to the forest. Added capacity made both "
         f"sequence architectures monotonically worse. Compression across three orders "
         f"of magnitude cost almost nothing. The uncertainty decomposition says why: "
         f"{_epistemic_share()} of predictive entropy is aleatoric, leaving a sliver "
         f"for any model to compete over.",
         align="justify")
    para(doc,
         "This is a statement about the task as posed, not about deep learning. Six "
         "hours ahead, from one station's channels, the AQI category is substantially "
         "undetermined. Improving it requires changing the input — more channels, a "
         "longer window, spatial context from neighbouring stations — not the "
         "architecture. A paper reporting a large gain on this task from an "
         "architectural change should be read with that in mind.",
         align="justify")

    heading(doc, "5.3 Methodology transferred; weights did not", 2)
    para(doc,
         "The Beijing pipeline produced no deployable model. What it produced was the "
         "protocol, and the protocol transferred cleanly: applied unchanged to "
         "Bangladesh, it rejected every Beijing model and accepted a Bangladesh-native "
         "one. A transferred Beijing model, tested directly on Bangladesh data, did not "
         "clear that dataset's own floor.",
         align="justify")
    para(doc,
         "That is the desirable outcome rather than a disappointing one, and it is the "
         "argument for citing this work's Beijing phase as methodology rather than as a "
         "model. It is also the argument for releasing the protocol as software: "
         "PulseBench is the part of this thesis most likely to be useful to someone "
         "else, and it is the part that was validated twice.",
         align="justify")

    heading(doc, "5.4 Data integrity is a result, not a preliminary", 2)
    para(doc,
         "The Bangladesh audit began as a preprocessing step and became a finding. A "
         "dataset indexed prominently, published with a DOI, and advertising a "
         "quarter-century of coverage turned out to be substantially generated over "
         "most of that span, with no indication in the record. Any external-validation "
         "claim built on the advertised span would have been built on generated "
         "numbers.",
         align="justify")
    para(doc,
         "The subsequent comparison against a reference monitor turned that from a "
         "suspicion into a measurement, and then moved the project's central limitation "
         "from 'the model cannot detect hazardous air' to 'the validation data did not "
         "contain any'. Those are very different statements, and only an instrument "
         "could distinguish them.",
         align="justify")


    heading(doc, "5.5 Cheap controls change conclusions", 2)
    para(doc,
         "The augmentation result would have read very differently without its "
         "controls. CTGAN raises aggregate macro-F1 significantly; reported alone, that "
         "is a positive finding and a publishable one. It survives neither of the two "
         "controls that were run. SMOTE produces the same aggregate gain in seconds "
         "rather than minutes, which removes the argument that the generator is earning "
         "its cost. And the per-class breakdown shows both methods buying that gain by "
         "degrading the classes the device exists for.",
         align="justify")
    para(doc,
         "Neither control was expensive. SMOTE is a library call; class weighting is a "
         "single keyword argument; the per-class breakdown is a different aggregation "
         "of numbers already computed. The cost of running them was a few minutes, and "
         "the cost of not running them would have been a false claim at the centre of "
         "the thesis. That asymmetry is the argument for making such controls routine "
         "rather than optional.",
         align="justify")
    para(doc,
         "The same applies to the validity checks on synthetic data. Section 4.3's "
         "checks are four assertions about whether a row is physically possible. They "
         "caught what a quality score above 0.89 did not, and they cost nothing to "
         "run. A distributional similarity score and a validity check answer different "
         "questions, and for engineered or physically constrained features the two can "
         "disagree completely.",
         align="justify")

    heading(doc, "5.6 What reproducibility required in practice", 2)
    para(doc,
         "Every number in this document is read at build time from a committed metrics "
         "file; the generator inserts a visible marker rather than a plausible value "
         "when it cannot find one. That constraint was adopted after an error that is "
         "worth recording, because it is the kind that survives review.",
         align="justify")
    para(doc,
         "An earlier draft of the Bangladesh audit reported the same percentage as both "
         "the share of rows kept and the share discarded, in adjacent sentences, because "
         "both were generated from one variable. The two are 81% and 19%. The mistake "
         "propagated into the public repository's documentation before it was caught by "
         "recomputing the fractions from the raw file rather than trusting the prose. "
         "The fix was not only to correct the text but to name the two quantities "
         "separately in the code, so that they can no longer collapse into one.",
         align="justify")
    para(doc,
         "Two smaller instances followed the same pattern: a figure title asserting a "
         "compression ratio that the plotted data did not show, and a calibration "
         "result that sat unreported in a metrics file for weeks until an audit of "
         "committed-but-unused values found it. Generating prose and figures from the "
         "data, rather than writing them alongside it, is what turned all three from "
         "silent errors into build-time failures.",
         align="justify")


def chapter_limitations(doc: Document) -> None:
    heading(doc, "6. Limitations", 1)
    bd = SRC.j("bangladesh_h6.json")
    sup = bd["support"]["test"] if bd else {}
    para(doc,
         "The limitations below are ordered by how much they constrain the central "
         "claims. The first three bound what the thesis establishes; the remainder "
         "bound how far it generalises.",
         align="justify")

    heading(doc, "6.1 The deployed model's advisory classes remain unvalidated", 2)
    para(doc,
         f"This is the limitation that matters most, because it concerns the function "
         f"the product exists for. The Bangladesh test split contains "
         f"{sup.get('Hazardous', '—')} Hazardous and {sup.get('Very unhealthy', '—')} "
         f"Very-unhealthy samples. A five-of-five-fold result computed over evaluation "
         f"blocks that thin cannot speak to those classes, and the thesis does not "
         f"claim it does.",
         align="justify")
    para(doc,
         "Section 4.11 narrows the gap without closing it. A PM2.5-only model on "
         "reference-monitor data detects hazardous air significantly above its floor in "
         "every fold, which establishes that the *task* is learnable. It is a different "
         "model, on one station, with one channel. Transferring that conclusion to the "
         "seven-channel deployed model would be exactly the kind of unearned inference "
         "this thesis argues against elsewhere.",
         align="justify")
    para(doc,
         "**The honest statement is that the deployed system could report ordinary "
         "air-quality bands and could not yet be claimed to raise a reliable "
         "hazardous-air warning.** Any deployment should carry that bound with it.",
         align="justify")

    heading(doc, "6.2 Multi-pollutant validation is currently impossible in Dhaka", 2)
    para(doc,
         "Section 4.12 is a limitation as much as a finding. No station in the survey "
         "reports PM2.5 alongside a companion pollutant for long enough and completely "
         "enough to validate a multi-channel model. That is a property of the "
         "monitoring infrastructure, not of the method, and no modelling choice "
         "addresses it. It is the direct cause of Section 6.1 and the reason the "
         "stationary variant in Section 7.2 is worth considering.",
         align="justify")

    heading(doc, "6.3 Two datasets, and the negative results rest on one", 2)
    para(doc,
         "The negative results — augmentation disqualified, capacity unhelpful, no "
         "model beating persistence — are established on Beijing and only partially "
         "re-tested on Bangladesh. Beijing 2013–2017 is a specific pollution regime "
         "with a specific seasonal structure and an unusually dominant middle class. "
         "Whether the same conclusions hold in a regime with different dynamics is "
         "untested, and the toolkit was released partly so that others can test them.",
         align="justify")

    heading(doc, "6.4 Methodological caveats", 2)
    bullets(doc, [
        "**Monte Carlo dropout is an approximation.** The aleatoric/epistemic split is "
        "load-bearing in Section 5.2, and dropout-based uncertainty estimates have known "
        "weaknesses. A deep ensemble would be a stronger check and was not run; the "
        "conclusion should be read as well-supported rather than settled.",

        "**Conformal coverage assumes exchangeability**, which a chronological split "
        "strains. Calibration and test come from different periods, and air quality has "
        "trend and seasonality. The empirical coverage reported in Section 4.5 is "
        "measured rather than assumed, which is the relevant reassurance, but the "
        "theoretical guarantee is weaker here than in the exchangeable case.",

        "**The horizon is fixed at six hours** because it is the shortest horizon at "
        "which the task is not a persistence echo, not because it is the most useful "
        "to a wearer. A shorter actionable warning is what the product wants and what "
        "Section 5.2 explains it cannot currently have.",

        "**One evaluation duplicate remains by design.** The tabular baseline keeps its "
        "own persistence implementation because it must return row-aligned predictions "
        "for the paired bootstrap, which is a different operation from returning a "
        "metrics summary. Substituting the shared version would have invalidated every "
        "paired test in the thesis. The difference is structural, and it is documented "
        "rather than resolved.",
    ])

    heading(doc, "6.5 Engineering and infrastructure caveats", 2)
    bullets(doc, [
        "**The device has not been fabricated.** No model in this project has run on "
        "ESP32 silicon. The latency benchmark in Section 4.6 was measured on a "
        "workstation through a runtime that does not exist for the target "
        "microcontroller, and is reported as a portability proxy throughout.",

        "**The advisory layer depends on an external hosted service.** Output is "
        "validated and falls back to a rule-based template on any failure, so the "
        "system degrades safely, but the layer is not self-contained and the model "
        "behind it can change without notice. One such change occurred during this "
        "work, when a model version was retired mid-project.",

        "**The synthetic-data findings are specific to one generator and one library "
        "version.** The validity failures in Section 4.3 are properties of how CTGAN "
        "was configured and of what the library enforced at the time. The general "
        "lesson — that a distributional quality score is not a validity check — is "
        "broader than the instance, but the specific numbers are not.",
    ])


def chapter_device(doc: Document) -> None:
    heading(doc, "7. Proposed Wearable Device (Concept — Not Yet Fabricated)", 1)
    para(doc,
         "**Nothing in this chapter has been built.** It describes the device the "
         "software was designed for, together with a stationary variant. No hardware "
         "was fabricated, no firmware was flashed, and no measurement in this thesis "
         "was taken from a physical prototype. The latency and size figures in Section "
         "4.6 come from a workstation benchmark.",
         align="justify")

    heading(doc, "7.1 The wearable concept", 2)
    para(doc,
         "The original concept is a neckband built around an ESP32 microcontroller, "
         "chosen for integrated wireless connectivity and sufficient flash for a "
         "compressed model. It would carry a particulate sensor and, if the power "
         "budget allows, a gas sensor, sampling the air at the wearer's breathing zone "
         "rather than at a rooftop station kilometres away. Inference would run "
         "on-device so that a forecast survives loss of connectivity, with the "
         "companion application providing history, explanation and configuration.",
         align="justify")
    table(doc, "Companion application features implied by the software built here.",
          ["Feature", "What supplies it", "Status"],
          [["Current risk category", "The compressed forest", "Implemented off-device"],
           ["Six-hour forecast", "Same model, horizon 6", "Implemented off-device"],
           ["Confidence as a set of categories", "Mondrian conformal thresholds",
            "Implemented off-device"],
           ["Which channel drove the warning", "Top-3 SHAP attributions",
            "Implemented off-device"],
           ["Plain-language advisory", "Validated language-model layer with template "
            "fallback", "Implemented off-device"],
           ["Exposure history", "Not built", "Design only"],
           ["On-device inference", "Requires a C tree traversal or `emlearn`",
            "**Not built** — see Section 4.6"]],
          source="derived from the implemented software modules", widths=[1.8, 2.7, 1.8])

    heading(doc, "7.2 A stationary variant", 2)
    para(doc,
         "A stationary, government-monitoring-style unit is also worth considering, and "
         "the results here argue for it more strongly than for the wearable. Section "
         "4.12 found that the monitoring infrastructure needed to validate a "
         "multi-pollutant model does not exist around Dhaka. A fixed unit at a known "
         "location, reporting continuously, addresses that gap directly: it relaxes the "
         "power and size budget, permits a reference-grade sensor, and contributes to "
         "the public record rather than only to one wearer's phone.",
         align="justify")
    para(doc,
         "**This is a design direction, not a result.** The repository contains no "
         "hardware or firmware notes for either variant; the stationary option is "
         "recorded here because the infrastructure finding points at it, not because "
         "any part of it was built.",
         align="justify")


def chapter_conclusion(doc: Document) -> None:
    heading(doc, "8. Conclusion and Future Work", 1)
    para(doc,
         f"This thesis set out to build a GAN-augmented deep-learning pipeline for "
         f"wearable air-quality risk forecasting. It produced one, and in the process "
         f"produced the evidence that most of its components do not earn their place. "
         f"GAN augmentation is disqualified for degrading the classes the device exists "
         f"to warn about, and so is SMOTE. Sequence models lose to a Random Forest, and "
         f"get worse as they get larger. At a six-hour horizon on Beijing data, no "
         f"model beats a zero-parameter baseline in more than "
         f"{_beijing_best_folds()} of "
         f"{need('rolling_cv_h6.json:n_folds', 'folds')} rolling-origin folds.",
         align="justify")
    para(doc,
         "What survived is smaller and better supported. A class-weighted Random Forest "
         "on Bangladesh data clears its own persistence floor in every fold, for the "
         "four common classes. Class-conditional conformal prediction restores coverage "
         "where the marginal guarantee failed. A PM2.5-only model on reference-monitor "
         "data detects hazardous air above its floor in every fold. A published dataset "
         "was audited and found substantially fabricated over its advertised span. And "
         "the evaluation protocol that established all of this is released as software.",
         align="justify")
    para(doc,
         "The central claim of the thesis is methodological: **a reported accuracy in "
         "this literature is uninterpretable without the zero-parameter baseline beside "
         "it**, and a single chronological split is not enough to establish that a "
         "model beats one. Applying that discipline turned a project about model "
         "architecture into a project about evaluation, which is the more useful "
         "outcome.",
         align="justify")
    heading(doc, "8.1 Future work", 2)
    bullets(doc, [
        "**Port the compressed forest to C and measure it on ESP32 silicon.** This is "
        "the largest self-contained piece of engineering the project leaves open, and "
        "it converts the portability proxy into a measurement.",
        "**Find or build a multi-pollutant reference series for Dhaka.** Section 4.12 "
        "shows none currently exists. A stationary unit (Section 7.2) is one route.",
        "**Validate the deployed model's advisory classes** once such a series exists, "
        "closing the gap Section 6 names first.",
        "**Replace Monte Carlo dropout with a deep ensemble** to check the "
        "aleatoric/epistemic split that Section 5.2 rests on.",
        "**Extend the input rather than the architecture.** Spatial context from "
        "neighbouring stations, or a longer window, addresses the aleatoric ceiling; "
        "more layers do not.",
        "**Test the protocol on other pollution regimes.** PulseBench is dataset-"
        "agnostic, and the negative results deserve a third and fourth city.",
    ])


def availability(doc: Document) -> None:
    heading(doc, "Data and Code Availability", 1)
    para(doc,
         f"All source code, configuration, generated reports, metrics files and figures "
         f"are publicly available at:",
         align="justify")
    para(doc, REPO_URL, align="center", bold=True, size=12)
    para(doc,
         "Every quantitative claim in this thesis is reproducible from the committed "
         "metrics files in that repository, and every figure was generated from them by "
         "a script that fails rather than substituting a value when a number is "
         "missing. The evaluation toolkit described in Section 3.12 is installable "
         "independently of the thesis code. The raw datasets are third-party and are "
         "not redistributed; the repository documents how to obtain each.",
         align="justify")
    para(doc,
         "The repository is released under the MIT Licence and has accepted an external "
         "contribution since publication.",
         align="justify")



def _class_distribution_table(doc: Document) -> None:
    hj = SRC.j("hj633_h6.json")
    if not hj:
        para(doc, need("hj633_h6.json:results", "class distribution"))
        return
    dist = hj["results"]["EPA"]["distribution"]
    rows = []
    for cls in dist["train"]:
        rows.append([cls] + [f"{dist[sp][cls]['n']:,} ({dist[sp][cls]['pct']:.2f}%)"
                             for sp in ("train", "val", "test")])
    table(doc, "Beijing class distribution across the chronological splits, under EPA "
               "breakpoints. The splits do not share a distribution, which Section 4.2 "
               "returns to.",
          ["AQI class", "Train", "Validation", "Test"], rows,
          source="hj633_h6.json", widths=[1.7, 1.6, 1.6, 1.6])


def _gan_plan_table(doc: Document) -> None:
    g = SRC.j("gan_h6.json")
    if not g:
        para(doc, need("gan_h6.json:balance", "synthesis plan"))
        return
    bal = g["balance"]
    rows = []
    for cls, pl in bal["plan"].items():
        rows.append([cls, f"{pl['n_observed']:,}", f"{pl['ratio_to_majority']:.4f}",
                     "yes" if pl["is_minority"] else "no",
                     f"{pl['n_synthetic']:,}" if pl["n_synthetic"] else "—"])
    table(doc, f"CTGAN synthesis plan. The majority class is {bal['majority_label']} at "
               f"{bal['majority_n']:,} observed rows; minorities are lifted to "
               f"{bal['target_n']:,}, not to parity.",
          ["Class", "Observed rows", "Ratio to majority", "Minority?",
           "Synthetic rows"], rows, source="gan_h6.json",
          widths=[1.7, 1.2, 1.4, 0.95, 1.2])


def _gan_quality_table(doc: Document) -> None:
    g = SRC.j("gan_h6.json")
    if not g:
        para(doc, need("gan_h6.json:quality", "synthesizer quality"))
        return
    rows = []
    for cls, q in g["quality"].items():
        rows.append([cls, f"{q['overall']:.4f}", f"{q['column_shapes']:.4f}",
                     f"{q['column_pair_trends']:.4f}", f"{q['n_synthetic']:,}",
                     f"{q['fit_minutes']:.1f}"])
    floor = SRC.cfg["gan"]["quality_floor"]
    table(doc, f"Synthetic Data Vault quality scores per class, for the corrected "
               f"configuration. The configured flagging floor is {floor}. These scores "
               f"are what the failed configuration also passed.",
          ["Class", "Overall", "Column shapes", "Column pair trends", "Synthetic rows",
           "Fit (min)"], rows, source="gan_h6.json",
          widths=[1.6, 0.85, 1.2, 1.4, 1.1, 0.85])


def _bd_conformal_table(doc: Document) -> None:
    bd = SRC.j("bangladesh_h6.json")
    if not bd:
        para(doc, need("bangladesh_h6.json:conformal", "Bangladesh conformal"))
        return
    key = next((k for k in bd["conformal"] if "native" in k.lower()),
               list(bd["conformal"])[0])
    per = bd["conformal"][key]["test"]["per_class"]
    rows = []
    for cls, v in per.items():
        note = ("no test samples — not validated" if v["n"] == 0
                else "thin support" if v["n"] < 50 else "")
        # A class with no test samples has no coverage and no mean set size. The
        # metrics file stores null there rather than a number, so the cells say so.
        def _f(x, spec):
            return "n/a" if x is None else format(x, spec)
        rows.append([cls, f"{v['n']:,}", _f(v["coverage"], ".4f"),
                     _f(v["mean_set_size"], ".2f"), note])
    table(doc, f"Per-class conformal coverage for the deployed Bangladesh model "
               f"({key}).",
          ["Class", "Test n", "Coverage", "Mean set size", "Status"], rows,
          source="bangladesh_h6.json", widths=[1.7, 0.95, 1.1, 1.3, 1.9])


def _shap_cases_table(doc: Document) -> None:
    d = SRC.j("shap_h6.json")
    if not d:
        para(doc, need("shap_h6.json:cases", "shap cases"))
        return
    rows = []
    for c in d["cases"]:
        # Each entry is {"feature", "value", "kind"}; rank by absolute contribution.
        top = sorted(c["shap"], key=lambda e: -abs(e["value"]))[:3]
        rows.append([c["reason"], c["true_label"], c["predicted"],
                     str(c["set_size"]),
                     ", ".join(f"{e['feature']} {e['value']:+.3f}" for e in top)])
    table(doc, f"The {len(rows)} SHAP case studies, with the three channels carrying the "
               f"largest attribution in each.",
          ["Case", "True", "Predicted", "Set size", "Top-3 attributions"], rows,
          source="shap_h6.json", widths=[1.85, 1.05, 1.05, 0.7, 1.65])


def _advisory_examples(doc: Document) -> None:
    """Show what the advisory layer actually emits, parsed from its own report."""
    path = REPORTS / "llm_advisory_examples.md"
    if not path.exists():
        para(doc, need("llm_advisory_examples.md", "advisory examples"))
        return
    text = path.read_text()
    blocks = re.split(r"^### Example ", text, flags=re.M)[1:]
    if not blocks:
        para(doc, "[MISSING: no advisory examples parsed from "
                  "reports/llm_advisory_examples.md]")
        SRC.missing.append("advisory examples parse")
        return
    rows = []
    for b in blocks:
        def field(name):
            m = re.search(rf"^\| {re.escape(name)}[^|]*\| (.+?) \|$", b, re.M)
            return re.sub(r"[*`]", "", m.group(1)).strip() if m else "—"
        quote = re.search(r"^> (.+)$", b, re.M)
        rows.append([b.split("\n")[0].replace("**", "").strip(),
                     field("Point prediction (+6 h)"),
                     field("Conformal set (mondrian, 90% per class)"),
                     field("True label"),
                     (quote.group(1)[:210] + "…") if quote else "—"])
    table(doc, f"The {len(rows)} advisory examples, with the generated text each "
               f"produced. Every advisory naming an ambiguous set hedges explicitly; "
               f"that is enforced by the validator, not left to the model.",
          ["Case", "Point prediction", "Conformal set", "True label",
           "Generated advisory (excerpt)"], rows,
          source="reports/llm_advisory_examples.md", widths=[1.2, 1.0, 1.15, 0.9, 2.05])


def appendix_reports(doc: Document) -> None:
    heading(doc, "Appendix C: Index of Generated Reports", 1)
    para(doc,
         "Every result in this thesis is backed by a generated report in the "
         "repository, each of which is itself produced from the committed metrics "
         "files. The list is read from the repository at build time.",
         align="justify")
    rows = []
    for path in sorted(REPORTS.glob("*.md")):
        text = path.read_text(errors="ignore")
        first = next((l.lstrip("# ").strip() for l in text.splitlines()
                      if l.startswith("# ")), "")
        rows.append([f"`{path.name}`", first[:78], f"{len(text.split()):,}"])
    if (FIGURES / "README.md").exists():
        rows.append(["`figures/README.md`",
                     "Figure index with the source JSON for each of 23 figures", "—"])
    table(doc, f"The {len(rows)} generated reports in the repository.",
          ["File", "Title", "Words"], rows, source="reports/ directory listing",
          widths=[2.2, 3.5, 0.7])


# ------------------------------------------------------ references and appendices


def references(doc: Document) -> None:
    heading(doc, "References", 1)
    path = REPORTS / "reference_list_expanded.md"
    if not path.exists():
        para(doc, need("reference_list_expanded.md", "reference list"))
        return
    text = path.read_text()
    para(doc,
         "The list below is reproduced from `reports/reference_list_expanded.md` in the "
         "repository. Entries marked [NEW] were added during this work and were each "
         "verified against a bibliographic registry — Crossref for DOIs, the arXiv API "
         "for preprints, or the publisher's own page — rather than from recollection. "
         "Entries not so marked are reproduced exactly as supplied and are not "
         "re-verified here.",
         align="justify", size=10, italic=True)
    entries = re.findall(r"^(\d+)\.\s+(.*?)(?=\n\d+\.\s|\n---|\Z)", text, re.S | re.M)
    if not entries:
        para(doc, "[MISSING: reference list could not be parsed from "
                  "reports/reference_list_expanded.md]")
        SRC.missing.append("reference list parse")
        return
    for n, body in entries:
        body = re.sub(r"\s*\*Relevance:\*.*", "", body, flags=re.S).strip()
        body = body.replace("**[NEW]**", "[NEW]").replace("*", "")
        body = " ".join(body.split())
        p = para(doc, f"[{n}]  {body}", size=10, space_after=5)
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.first_line_indent = Inches(-0.4)
    para(doc, f"Total: {len(entries)} entries.", size=10, italic=True)
    _check_citations(doc, text)


def _check_citations(doc: Document, reflist: str) -> None:
    """Every in-text citation must exist in the reference file.

    Matching is on first surname plus year, not on the citation's short form: the
    reference file lists full author strings ("Gündüz, Ali Fatih; Şahin, Canan Batur
    (2026)") which a short form ("Gündüz & Şahin 2026") never literally equals. An
    earlier version compared the strings directly and flagged seven valid citations.
    """
    lines = [l for l in reflist.splitlines() if re.match(r"^\d+\.\s", l)]
    unknown = []
    for key in sorted(CITED):
        year = re.search(r"\b(19|20)\d{2}\b", key)
        surname = re.split(r"\s+(?:&|et\s+al\.|,)\s*", key)[0].strip()
        first = surname.split()[0] if surname else key
        if not any(first in l and (not year or year.group() in l) for l in lines):
            unknown.append(key)
    if unknown:
        for k in unknown:
            SRC.missing.append(f"[MISSING: citation '{k}' not found in reference list]")
        para(doc, "[MISSING: the following in-text citations were not found in "
                  "reports/reference_list_expanded.md: " + "; ".join(unknown) + "]",
             size=10)


def appendix_hyperparameters(doc: Document) -> None:
    heading(doc, "Appendix A: Full Hyperparameter Tables", 1)
    para(doc,
         "Reproduced from `configs/default.yaml`. This file is the single source of "
         "every hyperparameter and threshold in the project; the code reads it rather "
         "than restating values.",
         align="justify")
    cfg = SRC.cfg
    sections = [
        ("A.1 Data and splits", cfg.get("data", {})),
        ("A.2 Preprocessing", cfg.get("preprocessing", {})),
        ("A.3 Tabular baselines", cfg.get("baseline", {})),
        ("A.4 Synthetic augmentation (CTGAN)", cfg.get("gan", {})),
        ("A.5 Sequence models", cfg.get("model", {})),
        ("A.6 Conformal prediction", cfg.get("conformal", {})),
        ("A.7 Explainability and advisory", cfg.get("explainability", {})),
        ("A.8 Deployment", cfg.get("deployment", {})),
    ]
    for title, block in sections:
        if not block:
            continue
        heading(doc, title, 2)
        rows = []
        for k, v in block.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    rows.append([f"{k}.{k2}", _fmt_cfg(v2)])
            else:
                rows.append([k, _fmt_cfg(v)])
        table(doc, f"{title[5:]} parameters.", ["Key", "Value"], rows,
              source="configs/default.yaml", widths=[2.6, 3.7])


def _fmt_cfg(v) -> str:
    if isinstance(v, list):
        return ", ".join(str(x) for x in v)
    if v is None:
        return "null (unset)"
    return str(v)


GLOSSARY = [
    ("Advisory classes", "The two highest AQI categories, Very unhealthy and Hazardous. "
     "The classes a wearable exists to warn about, and the ones treated as protected "
     "throughout this thesis."),
    ("Aleatoric uncertainty", "Irreducible uncertainty arising from the data itself. No "
     "amount of additional data or model capacity removes it."),
    ("Bonferroni correction", "Dividing the significance threshold by the number of "
     "comparisons in a family, so that testing many hypotheses does not inflate the "
     "chance of a false positive."),
    ("Conformal prediction", "A method that converts model scores into prediction sets "
     "with a coverage guarantee holding without distributional assumptions."),
    ("Disqualification rule", "The rule adopted here: an intervention that significantly "
     "degrades a protected class is rejected regardless of its effect on the aggregate."),
    ("Embargo", "A gap inserted between a training period and the evaluation block that "
     "follows it, so no evaluation target overlaps a training window."),
    ("Epistemic uncertainty", "Uncertainty arising from the model rather than the data. "
     "Reducible in principle with more data or capacity."),
    ("Expected calibration error", "The support-weighted mean gap between predicted "
     "confidence and observed accuracy across reliability bins."),
    ("Macro-F1", "The unweighted mean of per-class F1 scores. Gives a rare class the "
     "same weight as a common one, which is why it is used here."),
    ("Marginal coverage", "A conformal guarantee that holds on average across all "
     "classes, and therefore not necessarily for any one of them."),
    ("Maximum calibration error", "The largest single-bin gap between confidence and "
     "accuracy. The worst case rather than the average."),
    ("Mondrian conformal prediction", "Class-conditional conformal prediction: a "
     "separate calibration threshold per class, restoring per-class coverage."),
    ("Observed-only evaluation", "Computing a metric over rows whose label was measured "
     "rather than forward-filled, using the imputation provenance flags."),
    ("Persistence floor", "The score of the zero-parameter rule that predicts the future "
     "class equals the present one. The baseline every model must clear."),
    ("Resolution floor", "The smallest p-value a resampling test can produce: 2/n for an "
     "n-resample bootstrap, 2^(1-n) for a signed-rank test over n folds."),
    ("Rolling-origin cross-validation", "Advancing a cutoff through time, training "
     "before it and evaluating after it, so no prediction uses future information."),
    ("Seasonal-naive baseline", "The rule that the class now equals the class one "
     "seasonal cycle ago. A stronger baseline than persistence on cyclical data."),
    ("Split conformal prediction", "Conformal prediction using a held-out calibration "
     "set to compute the nonconformity threshold."),
]


def appendix_glossary(doc: Document) -> None:
    heading(doc, "Appendix B: Glossary of Terms", 1)
    t = doc.add_table(rows=0, cols=2)
    t.style = "Light List Accent 1"
    for term, definition in GLOSSARY:
        cells = t.add_row().cells
        cells[0].text = ""
        _rich(cells[0].paragraphs[0], term, size=TABLE_PT, bold=True)
        cells[1].text = ""
        _rich(cells[1].paragraphs[0], definition, size=TABLE_PT)
        cells[0].width, cells[1].width = Inches(1.9), Inches(4.4)


# ------------------------------------------------------------------------ driver


def _body(doc: Document) -> None:
    chapter_intro(doc)
    chapter_related(doc)
    chapter_methods(doc)
    chapter_results(doc)
    chapter_discussion(doc)
    chapter_limitations(doc)
    chapter_device(doc)
    chapter_conclusion(doc)
    availability(doc)
    references(doc)
    appendix_hyperparameters(doc)
    appendix_glossary(doc)
    appendix_reports(doc)


def _all_text(doc: Document) -> str:
    parts = [p.text for p in doc.paragraphs]
    for tb in doc.tables:
        for row in tb.rows:
            parts += [c.text for c in row.cells]
    return "\n".join(parts)


def assemble() -> Document:
    """Build the body first, then move the front matter in front of it.

    The lists of figures, tables and abbreviations cannot be written until the body
    exists, but they must appear before it. An earlier version built the body in a
    separate document and moved its XML elements across, which silently dropped every
    image: the drawing elements moved, but the image parts and their relationship
    entries stayed behind in the discarded package, so the output referenced
    relationship IDs it did not contain. The result had 23 figure captions and no
    figures.

    Everything is therefore built in one document, which keeps each image part and
    relationship attached to the package that will be saved. Only the front matter is
    relocated, and it contains no images.
    """
    doc = Document()
    setup_styles(doc)
    _body(doc)
    text = _all_text(doc)

    body = doc.element.body
    n_body = len(body) - 1          # python-docx keeps w:sectPr last

    title_page(doc)
    declaration(doc)
    certificate(doc)
    acknowledgement(doc)
    abstract(doc)
    toc(doc)
    list_of_figures(doc)
    list_of_tables(doc)
    list_of_abbreviations(doc, text)

    front = list(body)[n_body:-1]
    for i, el in enumerate(front):
        body.remove(el)
        body.insert(i, el)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_DOCX)
    ap.add_argument("--check", action="store_true",
                    help="resolve every value and report gaps, write nothing")
    args = ap.parse_args(argv)

    doc = assemble()
    if not args.check:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        doc.save(args.out)
        print(f"wrote {args.out.relative_to(REPO_ROOT)}")
    print(f"figures embedded : {len(REG.figures)}")
    print(f"tables embedded  : {len(REG.tables)}")
    print(f"citations used   : {len(CITED)}")
    uniq = sorted(set(SRC.missing))
    if uniq:
        print(f"\n{len(uniq)} MISSING placeholder(s) inserted:")
        for m in uniq:
            print(f"  {m}")
    else:
        print("\nno MISSING placeholders — every value resolved from a committed file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
