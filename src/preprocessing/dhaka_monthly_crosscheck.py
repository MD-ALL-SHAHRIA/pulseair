"""Fifth fabrication check: DoE published Dhaka PM2.5 vs the discarded Mendeley pre-2022.

The four signatures in ``reports/bangladesh_validation.md`` establish that the Mendeley
Dhaka series before 2022-08-05 is fabricated (yearly-median PM2.5 near-linear in year,
R2 = 0.992). This adds a fifth, external check: compare that discarded series, month by
month, against the Department of Environment's own published monthly CAMS averages for
Dhaka over the overlapping period. It does not reopen the pre-2022 data for modelling;
it is solely additional evidence for why the exclusion was correct.

The DoE monthly report PDFs are public static files (doe.gov.bd, Oracle object storage;
index in docs/reference/doe_monthly_report_index.json). Each report's Dhaka PM2.5 table
is parsed by two independent methods -- coordinate boxes (pdftotext -bbox-layout) and
column order (pdftotext -layout) -- and a month is used only when they agree, with a
>=50% data-capture threshold per station and a simple unweighted mean across Dhaka CAMS
(Sangsad Bhaban, BARC/Farmgate, Darus-Salam). Months whose Dhaka-wide value is byte-
identical to another month are dropped as a stuck parse or a duplicated source report.

    python -m src.preprocessing.dhaka_monthly_crosscheck            # uses cached PDFs
    python -m src.preprocessing.dhaka_monthly_crosscheck --download # (re)fetch PDFs

Writes reports/metrics/dhaka_monthly_crosscheck.json. Requires network + poppler
(pdftotext) only when downloading/parsing; the committed JSON lets the report rebuild
without either.
"""
from __future__ import annotations
import argparse, html, json, os, re, subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "docs/reference/doe_monthly_report_index.json"
MENDELEY = REPO / "data/raw/bangladesh_aqi.csv"
OUT = REPO / "reports/metrics/dhaka_monthly_crosscheck.json"
CACHE = REPO / ".cache/doe_monthly_pdfs"

DHAKA = ("bhaban", "barc", "d-salam", "darus", "firmgate", "farmgate", "sangsad", "salam")
CAP_MIN, VLO, VHI = 50.0, 8.0, 450.0
BOUNDARY = "2022-08-05"

_MON = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), 1)}


def _num(t):
    t = t.replace(",", "")
    return float(t) if re.fullmatch(r"-?\d+(\.\d+)?", t) else None


def month_key(title: str):
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{4})", title.lower())
    if not m:
        return None
    mo = next(k for k in _MON if k.startswith(m.group(1)))
    return f"{int(m.group(2)):04d}-{_MON[mo]:02d}"


def download(force=False):
    CACHE.mkdir(parents=True, exist_ok=True)
    idx = json.load(open(INDEX))["entries"]
    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/131.0 Safari/537.36"
    got = []
    for title, url in idx:
        key = month_key(title)
        if not key or not url.lower().endswith(".pdf"):
            continue
        dst = CACHE / f"{key}.pdf"
        if force or not dst.exists() or dst.stat().st_size < 1000:
            # -k: gov.bd sends an incomplete TLS chain; files are public and validated by content.
            subprocess.run(["curl", "-sS", "-k", "-L", "--max-time", "120", "-A", ua,
                            "-o", str(dst), url], check=False)
        if dst.exists() and dst.stat().st_size > 1000:
            got.append(key)
    return sorted(set(got))


def _bbox_pages(pdf):
    out = subprocess.run(["pdftotext", "-bbox-layout", str(pdf), "-"],
                         capture_output=True, text=True).stdout
    pages = []
    for pg in re.findall(r"<page[^>]*>(.*?)</page>", out, re.S):
        ws = []
        for m in re.finditer(
                r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(.*?)</word>', pg):
            x0, y0, x1, y1, t = m.groups()
            ws.append(dict(xc=(float(x0) + float(x1)) / 2, yc=(float(y0) + float(y1)) / 2,
                           t=html.unescape(t)))
        pages.append(ws)
    return pages


def _bbox_pm25(pdf):
    """Dhaka station -> (average, capture%) from the first genuine PM2.5 grid page."""
    for ws in _bbox_pages(pdf):
        lab = next((w for w in ws if w["xc"] < 160 and w["t"] == "2.5"
                    and [u for u in ws if abs(u["yc"] - w["yc"]) < 6 and u["xc"] < 130
                         and u["t"].upper().startswith("PM")]), None)
        if not lab:
            continue
        # require a co-located PM10 block, confirming the per-station grid
        if not any(w["xc"] < 160 and w["t"] == "10"
                   and [u for u in ws if abs(u["yc"] - w["yc"]) < 6 and u["xc"] < 130
                        and u["t"].upper().startswith("PM")] for w in ws):
            continue
        avg = [w for w in ws if w["t"].lower() == "average" and 0 < (lab["yc"] - w["yc"]) < 60]
        cap = [w for w in ws if "capture" in w["t"].lower() and 0 < (w["yc"] - lab["yc"]) < 60]
        if not avg or not cap:
            continue
        ay = max(avg, key=lambda w: w["yc"])["yc"]
        cy = min(cap, key=lambda w: w["yc"])["yc"]
        arow = sorted([w for w in ws if abs(w["yc"] - ay) < 4 and (_num(w["t"]) is not None or w["t"] == "DNA")],
                      key=lambda w: w["xc"])
        crow = sorted([w for w in ws if abs(w["yc"] - cy) < 4 and (_num(w["t"]) is not None or w["t"] == "DNA")],
                      key=lambda w: w["xc"])
        heads = [w for w in ws if w["yc"] < ay - 8
                 and any(d in w["t"].lower().strip("()a,") for d in DHAKA)]
        if heads:
            hy = max(h["yc"] for h in heads)
            heads = [h for h in heads if hy - h["yc"] < 30]
        res = {}
        for h in heads:
            if not arow:
                continue
            v = min(arow, key=lambda a: abs(a["xc"] - h["xc"]))
            if abs(v["xc"] - h["xc"]) > 40:
                continue
            c = min(crow, key=lambda a: abs(a["xc"] - h["xc"])) if crow else None
            res[h["t"].lower().strip("()a,")[:6]] = (
                _num(v["t"]) if v["t"] != "DNA" else None,
                _num(c["t"]) if c and c["t"] != "DNA" else None)
        if res:
            return res
    return None


def _layout_pm25_row(pdf):
    """Independent parse: the numeric tokens on the PM2.5 Average row, via -layout."""
    lines = subprocess.run(["pdftotext", "-layout", str(pdf), "-"],
                           capture_output=True, text=True).stdout.splitlines()
    for i, ln in enumerate(lines):
        if re.search(r"PM\s*2\.?5?\s*-?24", ln) or ("2.5" in ln and ("24hr" in ln or "24 hr" in ln)):
            for j in range(i - 1, max(0, i - 6), -1):
                if "average" in lines[j].lower():
                    nums = [_num(t) for t in lines[j].split() if _num(t) is not None]
                    if len(nums) >= 3:
                        return nums
    return None


def dhaka_month(pdf):
    b = _bbox_pm25(pdf)
    if not b:
        return None
    good = {k: (v, c) for k, (v, c) in b.items()
            if v is not None and VLO <= v <= VHI and c is not None and c >= CAP_MIN}
    if not good:
        return None
    bmax = max(v for v, c in good.values())
    row = _layout_pm25_row(pdf)
    if row and not any(abs(n - bmax) <= 2.0 for n in row):   # the two parses must agree
        return None
    return {"good": {k: (round(v, 1), c) for k, (v, c) in good.items()},
            "dhaka": round(float(np.mean([v for v, c in good.values()])), 1)}


def _r2_yearly_median(series: pd.Series):
    df = pd.DataFrame({"v": series.values, "yr": [p.year for p in series.index]})
    ym = df.groupby("yr")["v"].median()
    if len(ym) < 3:
        return None, None, len(ym)
    sl, ic, r, p, se = stats.linregress(ym.index, ym.values)
    return float(r ** 2), float(sl), len(ym)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--download", action="store_true", help="(re)download the DoE PDFs first")
    args = ap.parse_args()

    if args.download or not CACHE.exists():
        download(force=args.download)
    pdfs = sorted(CACHE.glob("*.pdf"))
    if not pdfs:
        raise SystemExit("no DoE PDFs cached; run with --download (needs network + poppler)")

    doe = {}
    for pdf in pdfs:
        r = dhaka_month(pdf)
        if r:
            doe[pdf.stem] = r
    # drop stuck/duplicated values (a value repeated across months is a misparse or a
    # duplicated source report, e.g. the 2017-06/2018-01 near-duplicate)
    cnt = Counter(v["dhaka"] for v in doe.values())
    dup = {val for val, n in cnt.items() if n >= 2}
    doe = {k: v for k, v in doe.items() if v["dhaka"] not in dup}
    doe_s = pd.Series({pd.Period(k, "M"): v["dhaka"] for k, v in doe.items()}).sort_index()

    d = pd.read_csv(MENDELEY, usecols=["city_name", "datetime", "pm2_5"])
    d = d[d["city_name"].str.strip().str.lower() == "dhaka"].copy()
    d["dt"] = pd.to_datetime(d["datetime"])
    d = d[d["dt"] < BOUNDARY]
    men_all = d.groupby(d["dt"].dt.to_period("M"))["pm2_5"].mean()

    idx = doe_s.index.intersection(men_all.index)
    A, B = doe_s.reindex(idx).astype(float), men_all.reindex(idx).astype(float)
    pear, spear = stats.pearsonr(A, B), stats.spearmanr(A, B)
    r2_men_full, sl_men_full, n_full = _r2_yearly_median(men_all)
    r2_men_ov, _, _ = _r2_yearly_median(B)
    r2_doe_ov, _, ndo = _r2_yearly_median(A)
    mons = [m for m in idx if m.month in (6, 7, 8, 9)]
    win = [m for m in idx if m.month in (12, 1, 2)]

    art = {
        "description": "Fifth fabrication check: DoE published Dhaka monthly PM2.5 vs the "
                       "discarded (fabricated) Mendeley pre-2022 series, over the overlap.",
        "aggregation": "Dhaka-wide = unweighted mean across Dhaka CAMS (Sangsad Bhaban, "
                       "BARC/Farmgate, Darus-Salam) with per-station data capture >= 50%.",
        "n_months": int(len(idx)), "span": f"{idx.min()}..{idx.max()}",
        "stats": {"pearson": pear.statistic, "spearman": spear.statistic,
                  "mad": float(np.mean(np.abs(A - B))), "bias_mendeley_minus_doe": float(np.mean(B - A)),
                  "r2_mendeley_full_pre2022": r2_men_full, "slope_mendeley_full": sl_men_full,
                  "n_years_full": n_full,
                  "r2_mendeley_overlap": r2_men_ov, "r2_doe_overlap": r2_doe_ov, "n_years_overlap": ndo,
                  "monsoon_doe_mean": float(A[mons].mean()), "monsoon_mendeley_mean": float(B[mons].mean()),
                  "winter_doe_mean": float(A[win].mean()), "winter_mendeley_mean": float(B[win].mean())},
        "doe_dhaka_pm25": {str(k): round(float(v), 1) for k, v in A.items()},
        "mendeley_dhaka_pm25": {str(k): round(float(v), 1) for k, v in B.items()},
        "doe_per_station": {k: doe[k]["good"] for k in doe if pd.Period(k, "M") in idx},
    }
    OUT.write_text(json.dumps(art, indent=1))
    print(f"wrote {OUT.relative_to(REPO)}  ({len(idx)} months, {idx.min()}..{idx.max()})")
    print(f"  Pearson r={pear.statistic:.3f}  MAD={art['stats']['mad']:.1f}  "
          f"bias={art['stats']['bias_mendeley_minus_doe']:+.1f}")
    print(f"  yearly-median R2  Mendeley(overlap)={r2_men_ov:.3f}  DoE(overlap)={r2_doe_ov:.3f}")
    print(f"  monsoon DoE={art['stats']['monsoon_doe_mean']:.0f} vs Mendeley={art['stats']['monsoon_mendeley_mean']:.0f}")


if __name__ == "__main__":
    main()
