# DoE published air-quality reports — what is actually retrievable, and what it could support

**Status: investigation only. Nothing here is wired into the pipeline.** This documents
what the Department of Environment publishes as static files, exactly what those files
contain, and — for the monthly series specifically — what a *coarser* external check
could and could not scientifically establish. No dataset was integrated; that decision
is deferred pending review.

Retrieved 2026-09-27 by direct download of public static files. No API token, no login.
The DoE real-time API remains untouched.

---

## 1. The six URLs supplied are dead (domain decommissioned)

All six `doe.portal.gov.bd` URLs return HTTP 200 but serve an **identical 884-byte
stub**: *"Domain is not available: doe.portal.gov.bd — This website is temporarily
unavailable."* The `doe.portal.gov.bd` host has been retired; the department's live
site is `doe.gov.bd`. So the specific files could not be fetched from those URLs — as
the brief anticipated — and the archive listing pages were used instead.

## 2. Two live archives exist on `doe.gov.bd`, backed by Oracle object storage

Both listing pages load (with a browser User-Agent; the server also sends an incomplete
TLS chain, so certificate verification had to be relaxed — normal for `*.gov.bd`). The
files themselves sit on `objectstorage.ap-dcc-gazipur-1.oraclecloud15.com` and download
cleanly as real PDFs.

| Archive | Listing page | Entries | Span | Cadence |
| --- | --- | --- | --- | --- |
| **Daily AQI reports** | `/site/page/8efde0a3-…` | 1,229 (712 PDF + 515 .doc/.docx) | **2023-02-13 → 2026-09-15** | daily, 1,227 distinct dates = **93.6%** of calendar days in span |
| **Monthly AQ reports** | `/site/page/09bafc68-…` | 81 monthly reports (+ ~23 study docs) | **Nov 2011 / Jan 2013 → June 2019** | monthly |

(The listing-page titles are swapped relative to their content: the page that reads
"CASE Project" holds the *monthly* reports, and the page titled "Daily AQI Report" holds
the daily ones. The table above is by verified content, not by page title. Full
machine-readable indices of both, with per-file titles, dates and storage URLs, were
captured during the survey.)

## 3. What each report type actually contains

### Daily AQI reports (2023–2026) — a derived index, one value per city per day

Verified against three samples (2023-02-13, 2025-01-31, 2026-09-15), format consistent
across the span:

- **One AQI value per city per day.** Not hourly. Not a concentration — the *Air Quality
  Index* (0–500+), the derived category index.
- **Columns:** City · AQI · Responsible Pollutant · AQI Category · Comments (occasionally
  an "AQI Range" for Dhaka/Chattogram).
- **Responsible pollutant is PM2.5 in essentially every row**, every city, every date
  sampled. In practice the daily series is single-pollutant (PM2.5-driven AQI).
- **Each city's value is itself a daily average** across that city's 1–5 stations —
  footnotes state e.g. "Based on 03 CAMS and 02 C-CAMS AQI Average in Dhaka."
- **City count grew over time:** ~11 cities in early 2023 → ~27 cities/locations by 2025
  (Dhaka, Chattogram, Gazipur, Narayanganj, Sylhet, Khulna, Rajshahi, Barishal, Savar,
  Mymensingh, Rangpur, Cumilla, Narsingdi, Bogura, Brahmanbaria, Cox's Bazar, Faridpur,
  Feni, Gopalganj, Jashore, Noakhali, Rampal, Shyamnagar, Tangail, Tongi, …).
- **"DNA" (Data Not Available) is frequent**, and increasingly so in 2026 (the
  2026-09-15 report has DNA for most cities outside Dhaka).

### Monthly AQ reports (2012–2019) — genuine multi-pollutant concentrations, monthly

Verified against June 2019 (13 pp) and January 2013. These are substantive documents:

- **Six criteria pollutants in real units:** PM2.5, PM10, CO, SO2, NOx, O3 (µg/m³, CO in
  mg/m³), monitored by the CASE-DoE CAMS network.
- **Per-station monthly statistics** — Table 3 gives **Average / Max / Min** for each
  pollutant at each CAMS, plus **data-capture rate** and **number of days exceeding**
  the Bangladesh National Ambient Air Quality Standard.
- **11 CAMS across 8 cities** in the 2019 era (Dhaka ×3, Chattogram ×2, Gazipur,
  Narayanganj, Khulna, Rajshahi, Sylhet, Barishal), with station coordinates listed.
- Data quality is stated honestly in the reports themselves: **50–90% capture**, outlier
  flagging (3rd/97th percentile), and analysers "not functional for some days due to
  maintenance" — so some station-months are partial or absent (DNA) per pollutant.
- Time-series plots appear in annexes as images, but the **numbers are in extractable
  tables**, not only figures.

## 4. What this means, stated precisely (item 3)

Neither archive is the *hourly, multi-pollutant, per-station ground-truth* series that
Phase 11 needed and could not find. But the monthly archive in particular is real
multi-pollutant instrument data in µg/m³, which the project has not had before, so it is
worth being exact about what a coarser check built on it could and could not claim.

### What monthly per-station averages (2012–2019) *could* support

- **A reanalysis-vs-instrument bias check at monthly scale.** The strongest honest use:
  compare Phase 10's reanalysis-derived monthly means against these instrument monthly
  means, per city, per pollutant. That would say something real about whether the
  reanalysis inputs the deployed model trusts are biased against ground truth — a
  question currently unanswered.
- **Seasonal-cycle and multi-year-trend validation** across 8 cities: whether the
  winter/monsoon cycle and the PM2.5/PM10 relationship in the modelled data match
  instruments.
- **Cross-pollutant relationships** (e.g. PM2.5:PM10 ratios) against real measurements.

### What monthly averages *cannot* support — and this is the important half

- **Not the Phase 11 advisory task.** That task is hourly AQI-category classification,
  and its entire point is the rare, safety-critical **Hazardous** class, which is defined
  at sub-daily scale. A monthly mean averages the Hazardous mornings away: a month can be
  "Unhealthy" on average while containing both clean afternoons and hazardous peaks.
  Monthly data cannot validate, or refute, anything about hazardous-hour detection.
- **Not rolling-origin CV, the persistence floor, or short-horizon (h = 6 h) forecasting**
  — all of these are defined on the hourly series and have no monthly analogue.
- **No recovery of variance or extremes.** The reports publish Average/Max/Min per
  station-month; the underlying hourly distribution is not in them, so nothing about
  within-month spread can be reconstructed beyond the min/max envelope.

### Statistical limits to keep in view

- **Sample size is small at monthly scale:** roughly 72–78 months × 8–11 stations, with
  50–90% capture and DNA gaps, so per-station monthly series have real holes and uneven
  reliability. Any monthly comparison is a modest-n exercise, not a large-sample test.
- **Era and geography mismatch:** the monthly archive ends **June 2019** and the daily
  archive begins **February 2023** — there is a ~3.5-year gap, and neither overlaps the
  deployed Bangladesh model's own period cleanly. The monthly series also predates the
  reanalysis window used in Phase 10 in part.
- **Extraction cost is real but bounded:** the monthly numbers live in layout PDF tables
  (parseable with care); the daily archive is 712 PDFs + 515 .doc/.docx of a small table
  each. Both are machine-extractable, neither is a clean CSV.

### The daily archive, honestly

The 2023–2026 daily archive is tempting because it is recent and 94% complete, but it is
**AQI category, not concentration, and effectively PM2.5-only** (the responsible
pollutant is PM2.5 throughout). At best it could support a *daily, single-pollutant,
per-city AQI-category* series — a different and coarser task than Phase 11's hourly work,
and one that would be validating a model against a derived index rather than against
instrument concentrations. Its frequent DNA in the most recent period further limits it.

## 5. Bottom line

- The DoE **does** publish air-quality reports openly as static files: a long **monthly**
  series (2012–2019, six pollutants, per-station µg/m³) and a recent **daily** series
  (2023–2026, AQI index, PM2.5-driven, per city).
- **Neither is a drop-in replacement for the missing hourly multi-pollutant ground
  truth.** The monthly series is the more scientifically valuable of the two and could
  support a genuine, clearly-scoped *monthly* reanalysis-vs-instrument validation — which
  is a real contribution, but must not be described as validating the hourly advisory
  classes.
- **No integration has been done.** If the monthly reanalysis-vs-instrument check is
  worth pursuing, it should be scoped and approved as its own task, with the limits in §4
  stated up front. The permission-based route to any finer-grained data remains the DoE
  data-request email drafted at `docs/outreach/doe_data_request_email.md`.
