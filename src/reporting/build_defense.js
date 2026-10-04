// PulseAir thesis defense deck. Every number comes from docs/defense_numbers.json
// (written by src/reporting/build_defense.py from committed metrics); every figure
// from reports/figures/. Run:  python -m src.reporting.build_defense && node src/reporting/build_defense.js
const fs = require("fs");
const path = require("path");
const pptxgen = require(process.env.PPTXGENJS || "pptxgenjs");

const ROOT = path.resolve(__dirname, "..", "..");
const N = JSON.parse(fs.readFileSync(path.join(ROOT, "docs", "defense_numbers.json"), "utf8"));
const FIG = (f) => path.join(ROOT, "reports", "figures", f);

// ---- identity -----------------------------------------------------------------------
const C = { bg: "0B1D33", panel: "132A4A", panel2: "0F2340", amber: "F5A524", text: "F4F1EA",
            muted: "9FB0C7", card: "F7F5F0", navyText: "0B1D33", red: "E8735A" };
const HEAD = "Cambria", BODY = "Calibri";
const W = 13.333, H = 7.5, MX = 0.6, FLOOR_Y = 6.72;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "PulseAir thesis defense";
pres.author = "Md All Shahria, Sanjeda Dewan Mithila, Anik Sarker Rudro, Irfanul Islam Payel";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

pres.defineSlideMaster({
  title: "PULSE",
  background: { color: C.bg },
  objects: [
    // the persistence floor: a thin baseline that recurs on every slide
    { line: { x: MX, y: FLOOR_Y, w: W - 2 * MX, h: 0, line: { color: C.amber, width: 1.25 } } },
    { text: { text: "persistence floor", options: { x: W - MX - 2.2, y: FLOOR_Y - 0.27, w: 2.2, h: 0.25,
      fontFace: BODY, fontSize: 10, color: C.amber, align: "right", italic: true, margin: 0 } } },
    { text: { text: "PulseAir  |  AIUB", options: { x: MX, y: 6.95, w: 4, h: 0.3, fontFace: BODY,
      fontSize: 11, color: C.muted, margin: 0 } } },
  ],
  slideNumber: { x: W - MX - 0.6, y: 6.95, w: 0.6, h: 0.3, fontFace: BODY, fontSize: 11, color: C.muted, align: "right" },
});

let section = null;
function slide(sectionTitle) {
  if (sectionTitle && sectionTitle !== section) { pres.addSection({ title: sectionTitle }); section = sectionTitle; }
  return pres.addSlide({ masterName: "PULSE", sectionTitle: section });
}

function title(s, kicker, text) {
  if (kicker) s.addText(kicker.toUpperCase(), { x: MX, y: 0.38, w: 9, h: 0.3, fontFace: BODY, fontSize: 13,
    bold: true, color: C.amber, charSpacing: 3, margin: 0, isTextBox: true });
  s.addText(text, { x: MX, y: 0.68, w: W - 2 * MX, h: 0.8, fontFace: HEAD, fontSize: 30, bold: true,
    color: C.text, margin: 0, valign: "top", isTextBox: true });
}

function body(s, text, x, y, w, h, opts = {}) {
  s.addText(text, Object.assign({ x, y, w, h, fontFace: BODY, fontSize: 18, color: C.text, margin: 0,
    valign: "top", isTextBox: true, paraSpaceAfter: 6 }, opts));
}

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

// figure on an off-white card, fitted without stretching or cropping
function figure(s, file, x, y, w, h, caption) {
  const capH = caption ? 0.32 : 0;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: h - capH, fill: { color: C.card },
    line: { color: C.card }, rectRadius: 0.08 });
  const pad = 0.12, aw = w - 2 * pad, ah = h - capH - 2 * pad;
  const sz = pngSize(FIG(file)), ar = sz.h / sz.w;
  let iw = aw, ih = aw * ar;
  if (ih > ah) { ih = ah; iw = ah / ar; }
  s.addImage({ path: FIG(file), x: x + pad + (aw - iw) / 2, y: y + pad + (ah - ih) / 2, w: iw, h: ih,
    altText: caption || file });
  if (caption) s.addText(caption, { x, y: y + h - capH + 0.04, w, h: capH, fontFace: BODY, fontSize: 12,
    italic: true, color: C.muted, margin: 0, isTextBox: true });
}

function big(s, value, caption, x, y, w, opts = {}) {
  const size = opts.size || 88;
  const vh = size / 72 * 1.05;
  s.addText(value, { x, y, w, h: vh, fontFace: HEAD, fontSize: size, bold: true, color: opts.color || C.amber,
    margin: 0, align: opts.align || "left", valign: "bottom", isTextBox: true, fit: "shrink" });
  if (caption) s.addText(caption, { x, y: y + vh + 0.05, w, h: opts.capH || 0.9, fontFace: BODY, fontSize: opts.capSize || 18,
    color: C.text, margin: 0, align: opts.align || "left", valign: "top", isTextBox: true });
}

function riser(s, x, top) {   // a result drawn relative to the floor: a tick rising from the baseline
  s.addShape(pres.shapes.LINE, { x, y: top, w: 0, h: FLOOR_Y - top, line: { color: C.amber, width: 2 } });
}

function table(s, rows, x, y, w, colW, opts = {}) {
  const fs_ = opts.fontSize || 16;
  const data = rows.map((r, i) => r.map((v) => ({ text: String(v), options: i === 0
    ? { bold: true, color: C.navyText, fill: { color: C.amber }, fontFace: BODY, fontSize: fs_ }
    : { color: C.text, fill: { color: i % 2 ? C.panel : C.panel2 }, fontFace: BODY, fontSize: fs_ } })));
  s.addTable(data, { x, y, w, colW, border: { type: "solid", pt: 0.5, color: C.bg }, margin: 0.06,
    valign: "middle", rowH: opts.rowH || 0.42, autoPage: false });
}

function card(s, x, y, w, h, head, text, opts = {}) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.panel }, line: { color: opts.edge || C.panel },
    rectRadius: 0.08 });
  s.addText(head, { x: x + 0.18, y: y + 0.12, w: w - 0.36, h: 0.42, fontFace: HEAD, fontSize: opts.headSize || 20,
    bold: true, color: C.amber, margin: 0, isTextBox: true, valign: "top" });
  if (text) s.addText(text, { x: x + 0.18, y: y + 0.58, w: w - 0.36, h: h - 0.68, fontFace: BODY,
    fontSize: opts.size || 18, color: C.text, margin: 0, isTextBox: true, valign: "top" });
}

function divider(act, name, line) {
  const s = slide(name);
  s.addText(act, { x: MX + 0.3, y: 2.1, w: 6, h: 0.5, fontFace: BODY, fontSize: 18, bold: true, color: C.amber,
    charSpacing: 4, margin: 0, isTextBox: true });
  s.addText(name, { x: MX + 0.3, y: 2.6, w: W - 2 * MX - 0.3, h: 1.2, fontFace: HEAD, fontSize: 48, bold: true,
    color: C.text, margin: 0, isTextBox: true });
  s.addText(line, { x: MX + 0.3, y: 3.9, w: 10, h: 0.6, fontFace: BODY, fontSize: 20, color: C.muted, margin: 0,
    isTextBox: true });
  riser(s, MX - 0.2, 2.1);
  s.addNotes(`${act}: ${name}. ${line} This section is short; each following slide carries one message. Hand over here if the next part is presented by another team member.`);
  return s;
}

// =====================================================================================
// 1 Title
let s = slide("Opening");
s.addText("BSc THESIS DEFENSE  ·  CSC 4298", { x: MX, y: 0.55, w: 8, h: 0.35, fontFace: BODY, fontSize: 14,
  bold: true, color: C.amber, charSpacing: 3, margin: 0, isTextBox: true });
s.addText("PulseAir", { x: MX, y: 1.0, w: 8, h: 1.2, fontFace: HEAD, fontSize: 72, bold: true, color: C.text, margin: 0, isTextBox: true });
s.addText("Wearable air-quality risk forecasting with honest baselines, conformal uncertainty and external validation for Bangladesh",
  { x: MX, y: 2.2, w: 8.2, h: 1.0, fontFace: BODY, fontSize: 20, color: C.text, margin: 0, isTextBox: true });
s.addText([
  { text: "Md All Shahria (xx-xxxxx-x)", options: { breakLine: true } },
  { text: "Sanjeda Dewan Mithila (xx-xxxxx-x)", options: { breakLine: true } },
  { text: "Anik Sarker Rudro (xx-xxxxx-x)", options: { breakLine: true } },
  { text: "Irfanul Islam Payel (xx-xxxxx-x)" },
], { x: MX, y: 3.45, w: 5.5, h: 1.5, fontFace: BODY, fontSize: 18, color: C.text, margin: 0, isTextBox: true, paraSpaceAfter: 4 });
s.addText([
  { text: "Supervisor", options: { color: C.amber, bold: true, breakLine: true } },
  { text: "Mirza Asif Mahmud", options: { breakLine: true } },
  { text: "Co-Supervisor", options: { color: C.amber, bold: true, breakLine: true } },
  { text: "Dipta Justin Gomes" },
], { x: 6.6, y: 3.45, w: 3.4, h: 1.5, fontFace: BODY, fontSize: 18, color: C.text, margin: 0, isTextBox: true });
s.addText("Department of Computer Science  ·  Faculty of Science and Technology  ·  American International University-Bangladesh (AIUB)",
  { x: MX, y: 5.55, w: 11, h: 0.4, fontFace: BODY, fontSize: 15, color: C.muted, margin: 0, isTextBox: true });
figure(s, "21_pipeline_overview.png", 9.0, 0.55, 3.75, 2.6);
s.addNotes("Good morning. We are presenting PulseAir, a wearable air-quality risk forecaster for Bangladesh. The thesis built the full software pipeline behind such a device and then tested every part of it against the simplest possible baseline. The headline is methodological: an accuracy number means little without the baseline beside it. The wearable itself is a design; no hardware was fabricated.");

// 2 Outline
s = slide("Opening");
title(s, "Outline", "Five acts, one discipline: beat the floor first");
const acts = [["I", "Why it matters"], ["II", "Data & method"], ["III", "Results"], ["IV", "Integrity & ground truth"], ["V", "Honesty & close"]];
acts.forEach(([n, t], i) => {
  const x = MX + 0.4 + i * 2.45;
  s.addShape(pres.shapes.OVAL, { x: x, y: FLOOR_Y - 0.17, w: 0.34, h: 0.34, fill: { color: C.amber }, line: { color: C.amber } });
  s.addShape(pres.shapes.LINE, { x: x + 0.17, y: 3.1, w: 0, h: FLOOR_Y - 3.27, line: { color: C.muted, width: 1, dashType: "dash" } });
  s.addText(n, { x: x - 0.4, y: 2.0, w: 1.2, h: 0.8, fontFace: HEAD, fontSize: 44, bold: true, color: C.amber, margin: 0, isTextBox: true });
  s.addText(t, { x: x - 0.4, y: 2.75, w: 2.3, h: 0.5, fontFace: BODY, fontSize: 19, color: C.text, margin: 0, isTextBox: true });
});
s.addNotes("The talk follows five acts. First why the problem matters, then the data and the evaluation method. Act three gives the modelling results, act four the data-integrity and ground-truth findings, and act five the limitations and conclusions. The dots sit on the persistence floor line, which you will see on every slide: results are always read against it.");

// ---- Act I
divider("ACT I", "Why it matters", "A personal warning needs forecasts that are better than doing nothing.");

// 3 Why this matters
s = slide("Why it matters");
title(s, "Why this matters", "The stations exist. Their open data does not.");
big(s, N.doe_stations, `DoE monitoring stations (${N.doe_cams} continuous + ${N.doe_cms} compact) in ${N.doe_cities} cities — NAQMP 2024–2030`, MX, 1.9, 5.4);
big(s, N.oq_pass, `of ${N.oq_n} OpenAQ stations within ${N.oq_radius} km of Dhaka meet multi-pollutant coverage`, 6.9, 1.9, 5.4, { color: C.red });
riser(s, MX - 0.2, 3.5); riser(s, 6.7, 3.5);
s.addNotes(`Bangladesh's National Air Quality Management Plan records ${N.doe_stations} government stations in ${N.doe_cities} cities. But when we searched the open OpenAQ platform, ${N.oq_pass} of ${N.oq_n} stations near Dhaka had the multi-pollutant coverage we needed. The obstacle is access, not existence. That gap is why a personal, forecasting device is attractive, and why it must be validated carefully.`);

// 4 The problem
s = slide("Why it matters");
title(s, "The problem", "An average can rise while the dangerous classes fall");
figure(s, "03_advisory_class_tradeoff.png", MX, 1.65, 7.6, 4.85, "Figure: effect of each imbalance intervention on macro-F1 and the two advisory classes");
big(s, `${N.ctgan_macro}`, `CTGAN macro-F1, above the plain forest's ${N.none_macro}`, 8.6, 1.7, 4.1, { size: 60, capH: 0.8 });
big(s, `${N.ctgan_haz}`, `CTGAN Hazardous F1, below the plain forest's ${N.none_haz}`, 8.6, 3.95, 4.1, { size: 60, color: C.red, capH: 0.8 });
s.addNotes(`This is the failure mode the thesis is about. GAN augmentation raised the aggregate macro-F1 from ${N.none_macro} to ${N.ctgan_macro}. On the same rows, Hazardous F1 fell from ${N.none_haz} to ${N.ctgan_haz}. A wearable exists for exactly those hours, so an aggregate score alone would have rewarded the wrong model.`);

// 5 Research questions
s = slide("Why it matters");
title(s, "Research questions", "Four questions, each answerable with a negative result");
[["RQ1", "Can an off-device pipeline forecast six-hour AQI risk better than persistence?"],
 ["RQ2", "Does GAN augmentation of rare, safety-critical classes survive strong controls?"],
 ["RQ3", "Can uncertainty be expressed in a form a device can act on?"],
 ["RQ4", "Does the method hold on Bangladesh, the population it targets?"]].forEach(([h, t], i) => {
  card(s, MX + (i % 2) * 6.15, 1.75 + Math.floor(i / 2) * 2.35, 5.95, 2.1, h, t, { size: 19 });
});
s.addNotes("These map one-to-one to the thesis objectives. The first asks whether forecasting beats doing nothing at six hours. The second tests GAN augmentation against cheap controls. The third asks for uncertainty a device can use, and the fourth for validation on Bangladeshi data rather than the Beijing benchmark.");

// 6 Contributions
s = slide("Why it matters");
title(s, "Contributions", "What is new, and what is careful practice");
table(s, [
  ["Type", "Contribution", "Claim level"],
  ["Methodological", "Protected-class disqualification rule", "Novel against cited work"],
  ["Empirical", "Fabrication audit of a published Bangladesh dataset", "Novel for that dataset"],
  ["Empirical", "Negative results: augmentation, capacity, persistence", "Reported as results"],
  ["Empirical", "Reference-monitor validation and Hazardous detector", "Narrows the gap"],
  ["Engineering", "PulseBench evaluation package", "Packaging, not new parts"],
  ["Not claimed", "Persistence floor; fold-independence caveat", "Rigour, not invention"],
], MX, 1.7, W - 2 * MX, [2.4, 6.4, 3.33], { rowH: 0.62, fontSize: 17 });
s.addNotes("We separate what is new from what is careful practice, as the thesis does. The clearest methodological contribution is the disqualification rule; the clearest empirical one is the dataset audit. PulseBench packages standard components, so its value is reuse. The persistence floor and the fold-independence caveat are rigour, and we do not claim them as inventions.");

// ---- Act II
divider("ACT II", "Data & method", "Four sources, one protocol, a floor every model must clear.");

// 7 Datasets
s = slide("Data & method");
title(s, "Datasets", "Four sources, each with a defined role");
table(s, [
  ["Source", "Role", "Extent"],
  ["UCI Beijing Multi-Site", "Method development", `${N.beijing_rows} hourly rows, ${N.beijing_stations} stations`],
  ["Mendeley Bangladesh AQI", "External validation", `${N.clean_rows} clean rows, ${N.actual_cities} cities`],
  ["US Embassy Dhaka", "Ground truth (PM2.5)", `${N.gt_qc_hours} QC-passed hours`],
  ["OpenAQ + DoE reports", "Infrastructure, cross-check", `${N.oq_n} stations surveyed; monthly CAMS`],
], MX, 1.7, 6.9, [2.3, 2.1, 2.5], { rowH: 0.7, fontSize: 15 });
figure(s, "23_dataset_coverage_timeline.png", 7.75, 1.7, 4.98, 4.8, "Figure: advertised span against the verified-clean window");
s.addNotes(`Beijing, with ${N.beijing_rows} hourly records from ${N.beijing_stations} stations, is where the method was developed. Bangladesh is the validation target; only ${N.clean_rows} rows in ${N.actual_cities} cities survive our audit. The US Embassy monitor gives instrument-grade PM2.5, and OpenAQ and the DoE reports test infrastructure and integrity. The timeline on the right shows how much of the advertised Bangladesh span survives.`);

// 8 Persistence floor
s = slide("Data & method");
title(s, "Method 1 · The persistence floor", "Beat zero parameters first");
big(s, N.floor_h1, `macro-F1 for "the category will not change", one hour ahead`, MX, 1.75, 5.3);
body(s, `At six hours the floor falls to ${N.floor_h6}; that is where forecasting starts.`, MX, 4.3, 5.0, 1.0);
riser(s, MX - 0.2, 1.8);
figure(s, "01_persistence_floor_vs_horizon.png", 6.2, 1.65, 6.53, 4.85, "Figure: persistence floor across horizons");
s.addNotes(`Air quality changes slowly. One hour ahead the category is unchanged ${N.unch_h1}% of the time, so the rule 'predict no change' scores ${N.floor_h1} macro-F1 with no training at all. Every model in the thesis is reported beside this floor on the same rows. We forecast six hours ahead because the floor drops to ${N.floor_h6}, giving models room to show real skill.`);

// 9 Rolling-origin CV
s = slide("Data & method");
title(s, "Method 2 · Rolling-origin cross-validation", "Train on the past, skip a day, test the next block");
for (let i = 0; i < 5; i++) {
  const y = 1.85 + i * 0.72, trainW = 3.2 + i * 1.3, x0 = MX + 0.2;
  s.addShape(pres.shapes.RECTANGLE, { x: x0, y, w: trainW, h: 0.42, fill: { color: C.panel }, line: { color: C.muted, width: 0.75 } });
  s.addShape(pres.shapes.RECTANGLE, { x: x0 + trainW, y, w: 0.28, h: 0.42, fill: { color: C.bg }, line: { color: C.muted, width: 0.75, dashType: "dash" } });
  s.addShape(pres.shapes.RECTANGLE, { x: x0 + trainW + 0.28, y, w: 1.25, h: 0.42, fill: { color: C.amber }, line: { color: C.amber } });
  s.addText(`Fold ${i + 1}`, { x: x0 + 0.1, y, w: 1.2, h: 0.42, fontFace: BODY, fontSize: 14, color: C.text, margin: 0, valign: "middle", isTextBox: true });
}
[["train: everything before the cutoff", C.panel], [`${N.embargo_h}-hour embargo`, C.bg], ["evaluate the next block", C.amber]].forEach(([t, col], i) => {
  s.addShape(pres.shapes.RECTANGLE, { x: MX + 0.2 + i * 3.9, y: 5.6, w: 0.35, h: 0.3, fill: { color: col }, line: { color: C.muted, width: 0.75 } });
  s.addText(t, { x: MX + 0.65 + i * 3.9, y: 5.55, w: 3.4, h: 0.4, fontFace: BODY, fontSize: 16, color: C.text, margin: 0, isTextBox: true });
});
body(s, `Scaler refit inside every fold. ${N.b5_n} folds, re-run at ${N.f8_n}.`, 9.2, 1.9, 3.5, 1.4);
s.addNotes(`A random split would let the future leak into training. Instead the cutoff rolls forward: train on everything before it, skip a ${N.embargo_h}-hour embargo so no target overlaps training, then evaluate the next block. Training starts at the first ${N.init_frac} of the record and the scaler is refit inside each fold. We ran ${N.b5_n} folds and then ${N.f8_n}, for a reason act five explains.`);

// 10 Disqualification rule + PulseBench
s = slide("Data & method");
title(s, "Method 3 · The disqualification rule", "No aggregate gain may be bought from a protected class");
const flow = [["Intervention", C.panel], ["Significantly degrades Very unhealthy or Hazardous?", C.panel], ["Rejected", C.red]];
card(s, MX, 1.8, 2.6, 1.1, "Intervention", "", { headSize: 20 });
card(s, MX + 3.0, 1.8, 3.4, 1.9, "Protected class hurt?", "Significant drop in Very unhealthy or Hazardous F1", { size: 16 });
card(s, MX + 3.0, 4.15, 3.4, 1.9, "Aggregate gain real?", "Significant and at least 0.01 macro-F1", { size: 16 });
card(s, MX, 4.15, 2.6, 1.1, "Accepted", "", { edge: C.amber });
card(s, MX + 6.8, 1.8, 2.0, 1.1, "Rejected", "", { edge: C.red });
s.addShape(pres.shapes.LINE, { x: MX + 2.6, y: 2.35, w: 0.4, h: 0, line: { color: C.amber, width: 2, endArrowType: "triangle" } });
s.addShape(pres.shapes.LINE, { x: MX + 6.4, y: 2.35, w: 0.4, h: 0, line: { color: C.red, width: 2, endArrowType: "triangle" } });
s.addText("yes", { x: MX + 6.35, y: 1.95, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 14, color: C.red, margin: 0, isTextBox: true });
s.addShape(pres.shapes.LINE, { x: MX + 4.7, y: 3.7, w: 0, h: 0.45, line: { color: C.amber, width: 2, endArrowType: "triangle" } });
s.addText("no", { x: MX + 4.8, y: 3.75, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 14, color: C.amber, margin: 0, isTextBox: true });
s.addShape(pres.shapes.LINE, { x: MX + 3.0, y: 4.7, w: -0.4, h: 0, line: { color: C.amber, width: 2, endArrowType: "triangle" } });
table(s, [["PulseBench function", "Provides"],
  ["persistence_floor", "Zero-parameter baseline"],
  ["rolling_origin_cv", "Folds, embargo, per-fold scaling"],
  ["advisory_disqualification", "The rule on the left"],
  ["bonferroni_report", "Family-wise correction"],
  ["dataset_audit", "Fabrication checks"]], 9.15, 1.8, 3.58, [1.9, 1.68], { fontSize: 13, rowH: 0.62 });
s.addNotes("This rule is the clearest methodological contribution. An intervention that significantly degrades either advisory class is rejected, whatever it does to the average. Otherwise its gain must be both significant and at least 0.01 macro-F1, because with tens of thousands of rows a bootstrap resolves tiny differences. The rule and the rest of the protocol are packaged as PulseBench, an open-source library listed on the right.");

// ---- Act III
divider("ACT III", "Results", "Every model, read against the floor.");

// 11 h = 1
s = slide("Results");
title(s, "Result 1 · One hour ahead", "Persistence nearly ties the trained model");
const top = 1.9, scale = 4.5;  // bar height = value * scale, drawn up from the floor
[["Persistence", N.floor_h1, C.muted], ["Random Forest", N.rf_h1, C.amber]].forEach(([lab, v, col], i) => {
  const hgt = parseFloat(v) * scale, x = MX + 1.0 + i * 2.6;
  s.addShape(pres.shapes.RECTANGLE, { x, y: FLOOR_Y - hgt, w: 1.7, h: hgt, fill: { color: col }, line: { color: col } });
  s.addText(v, { x, y: FLOOR_Y - hgt - 0.6, w: 1.7, h: 0.55, fontFace: HEAD, fontSize: 30, bold: true, color: C.text, align: "center", margin: 0, isTextBox: true });
  s.addText(lab, { x: x - 0.3, y: FLOOR_Y - 0.55, w: 2.3, h: 0.45, fontFace: BODY, fontSize: 16, bold: true, color: C.navyText, align: "center", margin: 0, isTextBox: true });
});
big(s, `${N.unch_h1}%`, "of hours keep the same AQI category one hour later", 7.3, 1.9, 5.4);
body(s, "A one-hour score mostly measures echoing the input.", 7.3, 4.45, 5.4, 0.8);
s.addNotes(`At one hour the trained Random Forest scores ${N.rf_h1} macro-F1 and persistence scores ${N.floor_h1}. The gap is tiny because ${N.unch_h1}% of hours keep their category. A model reporting a high one-hour score has mostly learned to copy its input. This is why we moved the primary horizon to six hours.`);

// 12 Eight folds
s = slide("Results");
title(s, "Result 2 · Beijing, eight rolling-origin folds", "No model beats persistence in a majority of folds");
figure(s, "24_rolling_cv_8fold.png", MX, 1.65, 7.7, 4.85, "Figure: per-fold macro-F1; persistence dashed");
big(s, `${N.f8_cw_wins} / ${N.f8_n}`, "folds won by the best model (class-weighted forest)", 8.7, 1.75, 4.0, { size: 80 });
body(s, `XGBoost: ${N.f8_xgb_wins} of ${N.f8_n} folds, Δ ${N.f8_xgb_delta}, p = ${N.f8_xgb_p}.`, 8.7, 4.4, 4.0, 1.2);
s.addNotes(`This is the central Beijing result. Across ${N.f8_n} chronological folds the best model wins exactly half, ${N.f8_cw_wins}. XGBoost wins none and is significantly worse than doing nothing, with a mean difference of ${N.f8_xgb_delta} and p equal to ${N.f8_xgb_p}. The lines interleave with the dashed persistence line: on this data, a model beating the floor once on one split is not evidence it forecasts better.`);

// 13 Augmentation
s = slide("Results");
title(s, "Result 3 · Class-imbalance interventions", "Augmentation lifts the average, hurts the advisory classes");
table(s, [["Intervention", "Macro-F1", "V. unhealthy F1", "Hazardous F1", "Verdict"],
  ["None (forest)", N.none_macro, N.none_vu, N.none_haz, "Baseline"],
  ["CTGAN (GAN)", N.ctgan_macro, N.ctgan_vu, N.ctgan_haz, "Disqualified"],
  ["SMOTE", N.smote_macro, N.smote_vu, N.smote_haz, "Disqualified"],
  ["Class weighting", N.cw_macro, N.cw_vu, N.cw_haz, "Not disqualified"]], MX, 1.75, 6.9, [1.8, 1.15, 1.35, 1.25, 1.35], { rowH: 0.62, fontSize: 15 });
body(s, `SMOTE matched CTGAN's rows in ${N.smote_s} s; CTGAN took ${N.ctgan_min} min.`, MX, 5.2, 6.8, 0.8);
figure(s, "05_per_class_f1_heatmap.png", 7.75, 1.65, 4.98, 4.85, "Figure: per-class F1 across variants");
s.addNotes(`Both augmentation methods raise macro-F1 and both significantly lower Very unhealthy and Hazardous F1, so the rule disqualifies them. Only class weighting avoids that trade, and it costs nothing. SMOTE produced the same synthetic rows in ${N.smote_s} seconds that CTGAN needed ${N.ctgan_min} minutes for. Without these cheap controls, the GAN result would have looked like a success.`);

// 14 Capacity
s = slide("Results");
title(s, "Result 4 · Sequence models", "Bigger LSTM and Transformer models get worse");
figure(s, "06_capacity_sweep.png", MX, 1.65, 6.0, 4.85, "Figure: validation macro-F1 against parameters");
figure(s, "07_uncertainty_decomposition.png", 6.75, 1.65, 3.6, 4.85, "Figure: entropy decomposition");
big(s, N.aleatoric, "of predictive entropy is aleatoric: irreducible from these inputs", 10.55, 1.75, 2.2, { size: 46, capSize: 16, capH: 2.2 });
s.addNotes(`Over about ${N.cap_ratio} times more parameters, validation macro-F1 fell by ${N.cap_lstm} for the LSTM and ${N.cap_tr} for the Transformer, monotonically. Monte Carlo dropout explains it: ${N.aleatoric} of predictive entropy is aleatoric, so there is little left for capacity to win. To forecast better, the input must change, not the architecture.`);

// 15 Uncertainty
s = slide("Results");
title(s, "Result 5 · Uncertainty", "Coverage on average is not coverage on Hazardous");
figure(s, "08_conformal_coverage.png", MX, 1.65, 7.0, 4.85, "Figure: per-class coverage, marginal vs Mondrian");
big(s, `${N.haz_cov_marg} → ${N.haz_cov_mond}`, "Hazardous coverage, marginal → class-conditional (Mondrian)", 7.95, 1.75, 4.8, { size: 40, capH: 0.8 });
table(s, [["Model", "ECE (mean)", "MCE (worst bin)"],
  ["Transformer", N.ece_transformer, N.mce_transformer],
  ["LSTM", N.ece_lstm, N.mce_lstm]], 7.95, 3.85, 4.78, [1.7, 1.5, 1.58], { rowH: 0.55, fontSize: 16 });
body(s, "The two calibration metrics rank the models in opposite order.", 7.95, 5.6, 4.78, 0.7, { fontSize: 18 });
s.addNotes(`Split conformal met its ${N.cov_marg} overall coverage, yet Hazardous was covered only ${N.haz_cov_marg}. Class-conditional calibration raised it to ${N.haz_cov_mond}, at the cost of larger sets. Calibration showed the same pattern: the Transformer has the lower average error but a much worse worst bin. Three times in this thesis, an average hid a tail failure.`);

// ---- Act IV
divider("ACT IV", "Integrity & ground truth", "The validation data was the first result.");

// 16 Data integrity
s = slide("Integrity & ground truth");
title(s, "Data integrity", "Most of the advertised history is generated");
big(s, `${N.span_pct}%`, `of the advertised 25-year span is backfill; ${N.rows_pct}% of rows`, MX, 1.75, 4.6, { size: 110 });
riser(s, MX - 0.2, 1.8);
[["1 Near-linear trend", `R² ${N.trend_r2}`], ["2 Hard clip at 250 µg/m³", `${N.clip_pct}% of rows`],
 ["3 CO unit change", "mid-file"], ["4 One low-density city", "before the boundary"],
 ["5 No monsoon washout", "against DoE reports"]].forEach(([h, t], i) => {
  card(s, 5.6 + (i % 2) * 3.6, 1.7 + Math.floor(i / 2) * 1.6, 3.4, 1.4, h, t, { headSize: 18, size: 18 });
});
s.addNotes(`The Bangladesh file advertises ${N.stated_cities} cities and 25 years; ${N.actual_cities} cities are present. Five independent signatures mark everything before the clean boundary as generated, including a near-linear trend with R-squared ${N.trend_r2} and a hard clip affecting ${N.clip_pct}% of rows. Discarding it costs ${N.rows_pct}% of rows but ${N.span_pct}% of the span. Every number is recomputed from the raw file.`);

// 17 Monsoon washout
s = slide("Integrity & ground truth");
title(s, "The fifth signature", "Real air washes out each monsoon; fabricated data does not");
figure(s, "28_dhaka_monthly_fabrication.png", MX, 1.65, 8.6, 4.85, "Figure: discarded pre-2022 Dhaka series against DoE monthly CAMS averages");
big(s, `${N.cc_mon_doe} vs ${N.cc_mon_men}`, "monsoon PM2.5 (µg/m³): DoE real vs fabricated", 9.5, 1.75, 3.25, { size: 50, capH: 0.9 });
body(s, `Yearly trend R²: ${N.cc_r2_doe} real, ${N.cc_r2_men} fabricated.`, 9.5, 3.9, 3.25, 1.0);
s.addNotes(`As an external check, the discarded Dhaka series was compared month by month with the Department of Environment's own published averages over ${N.cc_months} months. It runs ${N.cc_bias} micrograms high, and in the monsoon the real series falls to ${N.cc_mon_doe} while the fabricated one stays at ${N.cc_mon_men}. The real yearly medians have no trend; the fabricated ones are almost a straight line. This confirms the exclusion; it does not reopen the data.`);

// 18 Clean Bangladesh window
s = slide("Integrity & ground truth");
title(s, "On the clean window", "The same protocol accepts a Bangladesh-native model");
big(s, `${N.bd_wins} / ${N.bd_n}`, `folds won by a class-weighted forest; mean Δ ${N.bd_delta}`, MX, 1.75, 4.6, { size: 100 });
riser(s, MX - 0.2, 1.8);
body(s, `p = ${N.bd_p} is supplementary; folds won is the primary statistic.`, MX, 4.6, 4.5, 1.0, { fontSize: 18, color: C.muted });
figure(s, "15_folds_won_summary.png", 5.6, 1.65, 7.13, 4.85, "Figure: folds won, Beijing against Bangladesh");
s.addNotes(`On the ${N.actual_cities}-city clean window, the class-weighted forest beats persistence in all ${N.bd_n} folds, mean difference ${N.bd_delta}. The protocol that rejected every Beijing model accepts this one. The p-value of ${N.bd_p} is supporting evidence only, because folds are not independent. The deployment rests on this result.`);

// 19 Ground truth
s = slide("Integrity & ground truth");
title(s, "Ground truth", "The instrument sees the hazardous air the dataset misses");
big(s, `${N.gt_ref_h} vs ${N.gt_rea_h}`, `Hazardous hours: US Embassy monitor vs dataset, over ${N.gt_hours} shared hours`, MX, 1.75, 5.0, { size: 72 });
riser(s, MX - 0.2, 1.8);
body(s, `Correlation r = ${N.gt_r}, but bias ${N.gt_bias} µg/m³.`, MX, 4.4, 4.8, 0.8);
figure(s, "18_mendeley_vs_embassy.png", 5.85, 1.65, 6.88, 4.85, "Figure: advisory-class hours and PM2.5 distribution");
s.addNotes(`The advisory classes were missing from the Bangladesh test data. Against the US Embassy reference monitor, over ${N.gt_hours} overlapping hours, the instrument records ${N.gt_ref_h} Hazardous hours where the dataset records ${N.gt_rea_h}. The shapes correlate, r equals ${N.gt_r}, but the dataset flattens the peaks. So the limitation moved: Dhaka has hazardous air; the dataset does not show it.`);

// 20 Hazardous detector
s = slide("Integrity & ground truth");
title(s, "Six-hour Hazardous detector", "Learnable on instrument data, alive under sensor noise");
figure(s, "19_phase11b_hazardous.png", MX, 1.65, 6.0, 4.85, "Figure: Hazardous F1 against persistence, 7 folds");
figure(s, "26_sensor_noise_robustness.png", 6.75, 1.65, 3.85, 4.85, "Figure: clean vs low-cost-sensor noise");
big(s, `${N.hz_wins} / ${N.hz_n}`, `folds; F1 ${N.hz_f1} vs ${N.hz_floor}`, 10.75, 1.75, 2.0, { size: 48, capSize: 16, capH: 0.9 });
big(s, `−${N.sn_drop}%`, `under sensor noise; still ${N.sn_won}/${N.hz_n} vs noisy floor`, 10.75, 3.75, 2.0, { size: 44, color: C.red, capSize: 16, capH: 1.3 });
s.addNotes(`A PM2.5-only model trained on the Embassy series reaches Hazardous F1 ${N.hz_f1} against a ${N.hz_floor} floor, winning all ${N.hz_n} folds. With Gaussian noise sized to a published low-cost sensor R-squared of ${N.sn_r2}, Hazardous F1 drops ${N.sn_drop}% to ${N.sn_noisy}, yet still beats a floor computed on the same noisy input in ${N.sn_won} of ${N.hz_n} folds. This is evidence about the task on one station, not about the deployed multi-channel model.`);

// 21 Extended validation
s = slide("Integrity & ground truth");
title(s, "Extended validation", "Two more tests, each with its catch");
figure(s, "25_station_holdout.png", MX, 1.65, 5.95, 4.1, "Figure: leave-one-station-out");
figure(s, "27_selective_prediction.png", 6.78, 1.65, 5.95, 4.1, "Figure: selective prediction");
body(s, `${N.sh_wins}/${N.sh_n} held-out stations beat their floor. Catch: folds overlap in time.`, MX, 5.85, 5.95, 0.8, { fontSize: 18 });
body(s, `Accuracy ${N.sp_acc_full} → ${N.sp_acc_conf}; macro-F1 ${N.sp_f1_full} → ${N.sp_f1_conf}.`, 6.78, 5.85, 5.95, 0.8, { fontSize: 16 });
s.addNotes(`Holding out each Beijing station in turn, the forest beats that station's floor at ${N.sh_wins} of ${N.sh_n}, mean gain ${N.sh_delta}; but the folds overlap in time, so this tests place, not the future. Abstaining where the conformal set is large keeps ${N.sp_frac} of cases and raises accuracy from ${N.sp_acc_full} to ${N.sp_acc_conf}. Macro-F1 does not rise, because the abstained cases are mostly the safety-critical tail.`);

// ---- Act V
divider("ACT V", "Honesty & close", "What the evidence supports, and where it stops.");

// 22 Statistical honesty
s = slide("Honesty & close");
title(s, "Statistical honesty", "Folds share data, so p-values overstate");
big(s, "0.0625", `smallest two-sided p a ${N.b5_n}-fold signed-rank test can reach: above α = 0.05`, MX, 1.75, 5.4, { size: 90 });
riser(s, MX - 0.2, 1.8);
for (let i = 0; i < 4; i++) {
  s.addShape(pres.shapes.RECTANGLE, { x: 6.6, y: 1.9 + i * 0.62, w: 2.2 + i * 1.0, h: 0.4, fill: { color: C.panel }, line: { color: C.muted, width: 0.75 } });
  s.addShape(pres.shapes.RECTANGLE, { x: 6.6 + 2.2 + i * 1.0, y: 1.9 + i * 0.62, w: 0.9, h: 0.4, fill: { color: C.amber }, line: { color: C.amber } });
}
body(s, "Expanding windows overlap. Folds won is the primary statistic; p-values support it.", 6.6, 4.6, 6.1, 1.2);
s.addNotes(`A five-fold signed-rank test cannot reach significance: its smallest two-sided p is 0.0625. So we re-ran Beijing at ${N.f8_n} folds. More importantly, expanding-window folds share training data, so fold results are correlated and nominal p-values are anti-conservative. We therefore treat folds won as the primary, assumption-light statistic, and that makes the Beijing null result safer, not weaker.`);

// 23 National context
s = slide("Honesty & close");
title(s, "National context", "A data-driven complement to the national roadmap");
big(s, N.doe_stations, `DoE stations: ${N.doe_cams} continuous, ${N.doe_cms} compact, ${N.doe_cities} cities`, MX, 1.75, 4.3, { size: 80 });
body(s, "NAQMP 2024–2030 foresees forecast-triggered action, via chemistry-transport models (WRF-Chem, CAMx).", MX, 4.35, 4.4, 1.6, { fontSize: 17 });
figure(s, "20_openaq_survey.png", 5.4, 1.65, 7.33, 4.85, `Figure: OpenAQ survey, ${N.oq_n} stations, ${N.oq_pass} qualifying`);
s.addNotes(`The National Air Quality Management Plan records ${N.doe_stations} stations and anticipates forecast-triggered management of high-pollution days, through chemistry-transport models. Our work is the complementary data-driven route, with evidence on where it works. The survey on the right shows the access problem: of ${N.oq_n} OpenAQ stations near Dhaka, ${N.oq_comp} reports a companion pollutant and ${N.oq_pass} qualify.`);

// 24 Limitations
s = slide("Honesty & close");
title(s, "Limitations", "What we did not do");
[["Hardware not built", "No model has run on ESP32 silicon"],
 ["Advisory classes unvalidated", `Deployed test split: ${N.bd_test_haz} Hazardous, ${N.bd_test_vu} Very unhealthy`],
 ["PM2.5-only ground truth", "One Embassy station, one channel"],
 ["Beijing-only negatives", "One city, one regime, 2013–2017"],
 ["Approximate uncertainty", "MC dropout; conformal assumes exchangeability"],
 ["Hosted language model", "Validator and template fallback, not self-contained"]].forEach(([h, t], i) => {
  card(s, MX + (i % 3) * 4.1, 1.75 + Math.floor(i / 3) * 2.4, 3.9, 2.15, h, t, { headSize: 19, size: 18 });
});
s.addNotes(`We state the limits the thesis states. No hardware was built, and no model has run on the microcontroller. The deployed model's advisory classes are unvalidated because its test split holds ${N.bd_test_haz} Hazardous and ${N.bd_test_vu} Very-unhealthy samples. The Hazardous result is PM2.5-only on one station, the negative results come from one city, uncertainty methods are approximate, and the advisory depends on a hosted service.`);

// 25 Conclusion
s = slide("Honesty & close");
title(s, "Conclusion", "Three takeaways");
[["1", "Report the floor", `Best Beijing model: ${N.f8_cw_wins} of ${N.f8_n} folds`],
 ["2", "Audit the data", `${N.span_pct}% of a published span was generated`],
 ["3", "Validate on ground truth", `Hazardous detector: ${N.hz_wins} of ${N.hz_n} folds`]].forEach(([n, h, t], i) => {
  const x = MX + i * 4.1;
  s.addText(n, { x, y: 1.7, w: 1.0, h: 1.2, fontFace: HEAD, fontSize: 80, bold: true, color: C.amber, margin: 0, isTextBox: true });
  s.addText(h, { x, y: 3.0, w: 3.8, h: 0.6, fontFace: HEAD, fontSize: 26, bold: true, color: C.text, margin: 0, isTextBox: true });
  s.addText(t, { x, y: 3.65, w: 3.8, h: 1.0, fontFace: BODY, fontSize: 19, color: C.text, margin: 0, isTextBox: true });
  riser(s, x + 0.05, 4.8);
});
s.addNotes(`Three takeaways. First, an accuracy number is uninterpretable without the persistence floor beside it; on Beijing the best model won ${N.f8_cw_wins} of ${N.f8_n} folds. Second, validation data must be audited; ${N.span_pct}% of a published span was generated. Third, ground truth changes conclusions: on the reference monitor, hazardous air is learnable six hours ahead in ${N.hz_wins} of ${N.hz_n} folds.`);

// 26 Future work + thanks
s = slide("Honesty & close");
title(s, "Future work", "Thank you — questions?");
body(s, [
  { text: "Port the forest to C and measure it on an ESP32", options: { bullet: true, breakLine: true } },
  { text: "Find or build a multi-pollutant Dhaka reference series", options: { bullet: true, breakLine: true } },
  { text: "Validate the deployed advisory classes", options: { bullet: true, breakLine: true } },
  { text: "Deep ensembles to check the uncertainty split", options: { bullet: true, breakLine: true } },
  { text: "Change the input: spatial context, longer windows", options: { bullet: true } },
], MX, 1.8, 7.0, 4.2, { fontSize: 20, paraSpaceAfter: 10 });
s.addText("Thank you", { x: 8.0, y: 2.1, w: 4.7, h: 1.2, fontFace: HEAD, fontSize: 54, bold: true, color: C.amber, margin: 0, isTextBox: true });
s.addText("Questions?", { x: 8.0, y: 3.3, w: 4.7, h: 0.9, fontFace: HEAD, fontSize: 36, color: C.text, margin: 0, isTextBox: true });
s.addText(`Code, data pipeline and figures: github.com/MD-ALL-SHAHRIA/pulseair · DOI ${N.doi}`, { x: 8.0, y: 4.4, w: 4.7, h: 0.9, fontFace: BODY, fontSize: 15, color: C.muted, margin: 0, isTextBox: true });
s.addNotes("The largest open engineering step is running the compressed forest on the microcontroller itself. Validating the advisory classes needs a multi-pollutant reference series for Dhaka, which the survey showed does not yet exist openly. Methodologically, a deep ensemble would test the uncertainty split, and better inputs, not bigger models, are the route to better forecasts. Thank you; we welcome your questions.");

// ===================================================================================== BACKUP
divider("BACKUP", "Backup slides", "Details for questions.");

s = slide("Backup");
title(s, "Backup · Fold detail", "Beijing five-fold windows and scores");
table(s, [["Fold", "Evaluation window", "Persistence", "Forest (class-weighted)"],
  ...[1, 2, 3, 4, 5].map((i) => [String(i), N[`fold${i}_win`], N[`fold${i}_pers`], N[`fold${i}_cw`]])],
  MX, 1.75, 8.0, [0.9, 3.5, 1.6, 2.0], { rowH: 0.58, fontSize: 16 });
figure(s, "13_rolling_cv_beijing.png", 8.85, 1.65, 3.88, 4.85, "Figure: five-fold scores");
body(s, `Persistence's own fold-to-fold SD (${N.pers_sd5}) exceeds any model-to-floor difference.`, MX, 5.5, 8.0, 0.8, { fontSize: 18 });
s.addNotes(`Each row is one chronological evaluation block. The forest wins ${N.b5_cw_wins} of ${N.b5_n}. Persistence itself varies by a standard deviation of ${N.pers_sd5} between blocks, larger than any model-to-floor gap, which is why a single split cannot settle the question.`);

s = slide("Backup");
title(s, "Backup · Why persistence is the right floor", "Air quality is strongly autocorrelated");
table(s, [["Horizon", "Category unchanged", "Persistence macro-F1"],
  ...[1, 6, 12, 24].map((h) => [`${h} h`, `${N[`unch_h${h}`]}%`, N[`floor_h${h}`]])], MX, 1.75, 6.0, [1.5, 2.25, 2.25], { rowH: 0.62, fontSize: 18 });
figure(s, "17_split_class_prevalence.png", 6.95, 1.65, 5.78, 4.85, "Figure: class prevalence across splits");
body(s, "Zero parameters, no training: any learned model must clear it on the same rows.", MX, 5.0, 6.0, 0.9);
s.addNotes("Persistence uses only the current reading, so any model must beat it to show it learned dynamics. The floor falls monotonically with horizon, with no diurnal rebound at 24 hours. The right panel shows why splits disagree: class mix shifts between periods.");

s = slide("Backup");
title(s, "Backup · Per-class coverage", "Conformal coverage by class");
table(s, [["Class", "Test n", "Marginal", "Mondrian"],
  ["Good", N.pc_good_n, N.pc_good_marg, N.pc_good_mond],
  ["Moderate", N.pc_moderate_n, N.pc_moderate_marg, N.pc_moderate_mond],
  ["Unhealthy (sensitive)", N.pc_unhealthy_s_n, N.pc_unhealthy_s_marg, N.pc_unhealthy_s_mond],
  ["Unhealthy", N.pc_unhealthy_n, N.pc_unhealthy_marg, N.pc_unhealthy_mond],
  ["Very unhealthy", N.pc_very_v_n, N.pc_very_v_marg, N.pc_very_v_mond],
  ["Hazardous", N.pc_hazardous_n, N.pc_hazardous_marg, N.pc_hazardous_mond]], MX, 1.75, 6.4, [2.5, 1.3, 1.3, 1.3], { rowH: 0.55, fontSize: 16 });
figure(s, "16_precision_recall_beijing.png", 7.3, 1.65, 5.43, 4.85, "Figure: precision, recall, F1 per class");
s.addNotes(`Marginal calibration over-covers the largest class and under-covers the rare ones. Mondrian calibration evens this out, raising Hazardous from ${N.haz_cov_marg} to ${N.haz_cov_mond}. Mean set size rises from ${N.set_marg} to ${N.set_mond}, and singletons fall from ${N.single_marg} to ${N.single_mond}.`);

s = slide("Backup");
title(s, "Backup · dataset_audit", "The audit, re-run blind as a reusable function");
table(s, [["Output", "Value"], ["Verdict", N.ia_verdict], ["Checks flagged", `${N.ia_flagged} of 4`],
  ["Boundary found", N.ia_boundary], ["Manual boundary", N.clean_start], ["Difference", `${N.ia_err} days`],
  ["Affected city", N.ia_city]], MX, 1.75, 6.0, [2.6, 3.4], { rowH: 0.58, fontSize: 18 });
figure(s, "04_ctgan_validity.png", 6.95, 1.65, 5.78, 4.85, "Figure: a related check — CTGAN physical validity");
s.addNotes(`Given only the raw Mendeley file and no boundary, PulseBench's dataset_audit returns '${N.ia_verdict}' on ${N.ia_flagged} of four checks, places the boundary ${N.ia_err} days from the manual one, and names ${N.ia_city} unprompted. The right panel shows a related validity check from the augmentation work: quality scores did not catch physically impossible synthetic rows.`);

s = slide("Backup");
title(s, "Backup · Deployed predictor", "Measured on a workstation; not flashed");
table(s, [["Component", "Value"], ["Model", `Forest, ${N.dep_trees} trees × depth ${N.dep_depth}`], ["Size", `${N.dep_kb} KB`],
  ["ONNX latency", `${N.dep_lat} ms per sample`], ["Conformal coverage", N.dep_cov], ["ONNX parity", `argmax agreement ${N.dep_parity}`]],
  MX, 1.75, 6.0, [2.4, 3.6], { rowH: 0.6, fontSize: 18 });
figure(s, "22_deployed_architecture.png", 6.95, 1.65, 5.78, 4.85, "Figure: deployed architecture");
s.addNotes("The deployed Bangladesh predictor is small and fast on a workstation. ONNX is a portability proxy: there is no ONNX runtime for the ESP32, so the device step remains future work. The validator sits outside the language model so honesty rules are enforced by code.");

s = slide("Backup");
title(s, "Backup · Multiple comparisons", "Family-wise correction, resolution floor marked");
figure(s, "12_bonferroni_correction.png", MX, 1.65, 8.2, 4.85, "Figure: every persistence comparison under correction");
body(s, "A 1,000-resample bootstrap cannot resolve p below 2/1000; such values are reported as bounds.", 9.1, 1.9, 3.6, 2.4);
s.addNotes("All variants compared with persistence on the same split form one family, corrected with Bonferroni; Holm reproduced the same survivors. Values at the bootstrap's resolution floor are reported as bounded rather than exact.");

s = slide("Backup");
title(s, "Backup · Reproducibility", "Every number regenerates from committed files");
big(s, N.tests, "automated tests (thesis code + PulseBench)", MX, 1.75, 4.3, { size: 90 });
body(s, [{ text: `CI: ${N.ci}`, options: { breakLine: true } }, { text: `DOI: ${N.doi}` }], MX, 4.4, 4.5, 1.2, { fontSize: 18 });
figure(s, "29_methodology_workflow.png", 5.3, 1.65, 7.43, 4.85, "Figure: study workflow");
s.addNotes(`The repository carries ${N.tests} automated tests run in continuous integration, and the release is archived with DOI ${N.doi}. The thesis builder reads every number from committed metrics and inserts a visible marker rather than a plausible value when one is missing. The same is true of these slides.`);

pres.writeFile({ fileName: path.join(ROOT, "docs", "PulseAir_Defense.pptx") }).then((f) => console.log("wrote", f));
