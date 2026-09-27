# Dhaka pre-2022 fabrication — a fifth check against real government data

**This is a fifth signature added to the four already in
[`bangladesh_validation.md`](bangladesh_validation.md), not a new standalone finding,
and it does not reopen the pre-2022 data for modelling.** That data remains excluded.
The purpose here is only to test the exclusion against an independent, real-world
reference: the Department of Environment's own published monthly CAMS averages for
Dhaka. The question is narrow — *does the discarded Mendeley pre-2022 Dhaka series agree,
month by month, with what the government actually measured over the same period?*

Regenerate with `python -m src.preprocessing.dhaka_monthly_crosscheck` (data in
`reports/metrics/dhaka_monthly_crosscheck.json`).

## The two series

**Fabricated (already discarded).** The Mendeley Dhaka rows before 2022-08-05 —
198,048 hourly rows, the same rows the four existing signatures flagged. Aggregated here
to a monthly mean PM2.5 per calendar month.

**Real (the reference).** The DoE publishes a *Monthly Air Quality Monitoring Report*
whose Table 3 gives, per CAMS, the monthly Average/Max/Min for each criteria pollutant
in µg/m³ with a data-capture rate. These are public static PDFs
(`doe.gov.bd`; index in `docs/reference/doe_monthly_report_index.json`). For Dhaka the
CAMS are Sangsad Bhaban, BARC/Farmgate and Darus-Salam.

### How the DoE numbers were extracted, and the coverage that survived

Reading numbers out of 81 layout PDFs is error-prone, and a wrong number here would
manufacture a false comparison, so extraction was deliberately conservative:

- Each report's Dhaka PM2.5 table was parsed **twice by independent methods** — once
  from word coordinates (`pdftotext -bbox-layout`), once from column order
  (`pdftotext -layout`) — and a month was **used only when the two agree**.
- **Dhaka-wide monthly value = simple unweighted mean** across the Dhaka CAMS whose
  **data capture ≥ 50%** that month. No other weighting.
- Months whose Dhaka-wide value was byte-identical to another month were dropped as
  either a stuck parse or a duplicated source report (see caveats).

Of the 81 reports (Nov 2011 – Jun 2019), **53 months (2014-03 → 2019-06) survived** and
overlap the fabricated series. The rest were dropped honestly: the 2011–2013 reports
publish PM2.5 only as a prose range, not a per-station table; two reports are near
duplicates; three were rejected because the two parses disagreed. 53 of a possible ~90
overlap months is the real coverage, and the small-*n* limits below apply.

## What the comparison shows

| Measure | Value |
| --- | --- |
| Overlapping months | **53** (2014-03 → 2019-06) |
| Pearson correlation | **r = 0.756** (p ≈ 6 × 10⁻¹¹) |
| Spearman correlation | ρ = 0.582 |
| Mean absolute difference | **47.4 µg/m³** |
| Mean bias (fabricated − DoE) | **+42.7 µg/m³** (fabricated runs high) |

A correlation of 0.76 sounds like agreement, but it is carried almost entirely by the
coarse winter-high / rest-of-year-lower contrast that any Dhaka series would share. Two
structural comparisons show the fabricated series is not the same object as the real
one.

### Signature 5a — the fabricated series has no monsoon washout

Real Dhaka PM2.5 collapses every monsoon. In the DoE data the **June–September** mean is
**35 µg/m³** (individual months 19–54). The fabricated series over the same monsoon
months averages **112 µg/m³** and never drops below ~99 — it is **+77 µg/m³** too high
in exactly the season a real Dhaka series is cleanest, and it drifts *upward* year on
year instead of crashing. Winter, by contrast, roughly matches (DoE 157 vs fabricated
170 µg/m³). The fabrication reproduced the winter that a casual check would look at and
invented a monsoon roughly three times too high.

Within-year max–min range makes the same point: **DoE ≈ 141 µg/m³**, fabricated **≈ 58
µg/m³**. The real city swings far more within each year than the fabricated data does.

### Signature 5b — the fabricated trend is linear where the real one is not

The original audit found the fabricated yearly-median PM2.5 was near-linear in year,
R² = 0.992. This reproduces on the monthly aggregation: over the **full pre-2022** span
the fabricated yearly-median trend is **R² = 0.993, slope +5.19 µg/m³/yr** — matching the
audit. The new evidence is what the *real* data does over the **same six overlap years**:

| Yearly-median PM2.5, 2014–2019 | R² of a linear fit |
| --- | --- |
| Fabricated (Mendeley pre-2022) | **0.999** |
| Real (DoE published) | **0.132** |

Over the identical window and city, the fabricated series lies almost perfectly on a
straight line while the government's own measurements show no linear yearly trend at all.
A real city's yearly medians do not march up a line; the fabricated ones do. This is the
fourth signature (linearity) confirmed against an external ground truth rather than only
by the internal shape of the series.

## Caveats (stated, not worked around)

- **Coverage: 53 of ~90 overlap months.** The pre-2014 reports give only prose ranges;
  three months were dropped for parse disagreement; two were dropped as duplicates. The
  comparison is over the months that could be extracted with two agreeing parses, not the
  full overlap.
- **Small *n* for the trend contrast:** the yearly-median R² comparison rests on **6
  years**. The R² gap (0.999 vs 0.13) is large, but with six points it is descriptive
  support for the existing signature, not an independent significance test.
- **The DoE monthly averages are themselves aggregates** of hourly CAMS data with **50–
  100% capture** (≥50% required here), outlier flagging, and analyser downtime; they are
  not gap-free truth, only an independent real-world reference.
- **A duplicate in the source reports:** the June-2017 and January-2018 reports carry
  byte-identical Dhaka tables (and June-2017's prose reports 30 exceedance days, which is
  a winter figure). Both were dropped. This is a data-quality flaw in the DoE PDFs, noted
  because it was found, and it does not affect the 53 retained months.
- **Aggregation is deliberately plain:** unweighted mean across Dhaka CAMS with capture
  ≥ 50%, no distance or population weighting, so the number is not tuned to produce any
  particular comparison.

## Conclusion

The discarded Mendeley pre-2022 Dhaka series disagrees with the government's own
published measurements in exactly the ways fabrication predicts: it is ~43 µg/m³ high on
average, it misses the monsoon minimum by ~77 µg/m³, its within-year variability is less
than half the real city's, and its yearly trend is a near-perfect line where the real
one is flat. This is a **fifth, external** confirmation, consistent with the four
internal signatures already documented, that the pre-2022 data was correctly excluded.
**Nothing here changes that exclusion; the pre-2022 data stays out of every model.**
