# Phase 11c — OpenAQ survey for a multi-pollutant Dhaka source

Phase 11b validated the advisory classes on the US Embassy reference monitor, but that
source is **PM2.5 only**. It therefore cannot answer the question the deployed model's
design depends on: **do PM10 and CO actually earn their place on real ground-truth
data, or only on reanalysis data?**

This phase asked OpenAQ whether a suitable station exists near Dhaka.

**Acceptance bar, fixed before looking:** PM2.5 **and** at least one of
{PM10, CO}, more than **1 year** of
overlap, better than **70%** hourly completeness.

---

## 1. What is there

22 locations within 25 km of
(23.8103, 90.4125).

| Provider | Stations |
| --- | --- |
| AirGradient | 18 |
| Spartan | 1 |
| StateAir Dhaka | 1 |
| AirNow | 1 |
| SPARTAN Network | 1 |

| Parameter | Stations reporting it |
| --- | --- |
| pm25 | 22 |
| pm1 | 18 |
| um003 | 18 |
| relativehumidity | 17 |
| temperature | 17 |
| o3 | 1 |
| pm10 | 1 |

| ID | Station | Provider | Parameters | Date range | Max records |
| --- | --- | --- | --- | --- | --- |
| 8415 | Dhaka | AirNow | o3, pm25 | 2016-11-09..2025-03-24 | 66,366 |
| 3194367 | Dept. of Public Health and Informatics,  | AirGradient | pm1, pm10, pm25, relativehumidity, temperature, um003 | 2024-10-28..2026-09-01 | 8,513 |
| 6157905 | RAJUK Uttara Apartment Project, Sector - | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2025-12-09..2026-09-20 | 6,809 |
| 2445 | US Diplomatic Post: Dhaka | StateAir Dhaka | pm25 | 2016-03-10..2016-11-09 | 5,656 |
| 6234078 | Kaliganj \| Gazipur \| Smart Air Banglades | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-02-09..2026-09-20 | 5,220 |
| 6236590 | Hazaribagh, Jigatola \| Dhaka \| Smart Air | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-02-11..2026-09-20 | 5,189 |
| 6234363 | Gulshan Society Lake Park \| Dhaka \| Smar | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-02-09..2026-09-20 | 5,052 |
| 6242079 | Moghbazar; Bhodro Goli \| Dhaka \| Smart A | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-02-16..2026-09-19 | 5,012 |
| 6242232 | Baridhara, Park Road \| Dhaka l Smart Air | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-02-16..2026-09-20 | 4,839 |
| 6240773 | Kalshi, Mirpur-12 \| Dhaka l Smart Air Ba | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-02-15..2026-09-20 | 4,164 |
| 6240023 | Dhanmondi, Road No. 7 \| Dhaka l Smart Ai | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-02-14..2026-08-13 | 4,014 |
| 6271076 | Baridhara Lakeside | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-03-14..2026-09-20 | 3,998 |
| 6251395 | Mirpur, Shewrapara \| Dhaka \| Smart Air B | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-02-24..2026-08-17 | 805 |
| 6322838 | Ahmedbagh, Sabujbagh \| Dhaka \| Smart Air | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-04-26..2026-09-20 | 227 |
| 6528096 | Dept. of Public Health and Informatics,  | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-09-07..2026-09-19 | 186 |
| 5105944 | smart air bangladesh office | AirGradient | pm1, pm25, um003 | 2025-07-17..2025-10-22 | 94 |
| 6219798 | Kalijung,Gazipur  | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-01-29..2026-02-05 | 13 |
| 6313504 | Demra \| Dhaka \| Smart Air Bangladesh | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-04-19..2026-09-20 | 11 |
| 6219691 | Abdullahbag, North Badda | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-01-29..2026-01-29 | 6 |
| 6322839 | Sensor No -04 | AirGradient | pm1, pm25, relativehumidity, temperature, um003 | 2026-04-26..2026-04-26 | 3 |
| 22 | SPARTAN - Dhaka University | Spartan | pm25 | — | 0 |
| 1285342 | SPARTAN - Dhaka University | SPARTAN Network | pm25 | — | 0 |

**Notable absences.** No **CO** sensor exists anywhere in the radius. No **Bangladesh
Department of Environment** station is registered on OpenAQ — the DoE network is
referenced in earlier phases as a possible source and it is *not* available here.
The US Embassy record appears as two entries (`2445`, `8415`) and is the same
instrument already used in Phase 11.

---

## 2. Assessment against the bar

**1 of 22 stations** report PM2.5
together with PM10 or CO.

| ID | Station | Provider | Companion | PM2.5 span / n | Companion span / n | Passes ≥1 yr? |
| --- | --- | --- | --- | --- | --- | --- |
| 3194367 | Dept. of Public Health and Informa | AirGradient | pm10 | 1.84 yr / 8,513 | 0.21 yr / 1,552 | **no** |

### The one candidate, measured directly

Rather than infer overlap from the sensor metadata, the two series were downloaded and
joined hourly.

|  | Value | Bar | Passes |
| --- | --- | --- | --- |
| Joint PM2.5 + PM10 hours | **1,547** | — |  |
| Overlap window | 2024-10-28 .. 2025-01-13 | — |  |
| Overlap span | **77 days (0.21 years)** | > 1 year | **NO** |
| Hourly completeness in window | 83.2% | > 70% | yes |
| PM2.5 range (µg/m³) | 13.4 – 440.0 | — |  |
| PM10 range (µg/m³) | 13.8 – 494.0 | — |  |
| Hazardous hours in window | 334 | — |  |

The data itself is not bad — 83% complete and carrying
334 Hazardous hours, which is a denser rare-class signal than
anything in the Mendeley product. **It is simply far too short.** At
77 days it is a single winter fragment, and the protocol this
project uses is rolling-origin CV with seasonal rotation: no fold design can rotate
seasons inside two and a half months. Fitting the Phase 11b protocol to it would
produce a number with no meaning.

---

## 3. Verdict

**No suitable multi-pollutant station exists. Phase 11c stops here rather than forcing
a result.**

Precisely what was found and why it is insufficient:

1. **No CO at all** within 25 km of Dhaka, on any network. The
   deployed model's CO channel cannot be validated against ground truth here at any
   duration.
2. **Exactly one station carries PM10** (`3194367 — Dept. of Public Health and Informatics, `,
   AirGradient), and its PM10 sensor ran for
   **77 days**
   (0.21 years) against a 1-year bar. Its
   parent PM2.5 series is
   8,513 readings over 1.84
   years ≈ 53%
   hourly completeness, also under the 70% bar.
3. **Everything else is PM2.5-only.** The AirGradient network has expanded across
   Dhaka rapidly, but almost entirely with particulate-only units, and most began
   reporting in 2026 — too recent for a seasonal protocol.

### What this means for the thesis

- **The Phase 11b result stands as the advisory-class evidence**, and its stated
  limitation — one pollutant, one station — cannot currently be lifted with public
  data.
- **The deployed 7-channel model's multi-pollutant design remains validated only on
  reanalysis data** (Phase 10), never on ground truth. That is now a documented
  fact about data availability in Bangladesh, not an oversight in this project.
- **This is a concrete, citable finding for the write-up:** reference-grade
  multi-pollutant monitoring in Dhaka is effectively unavailable via public APIs.
  A deployment programme would need either the DoE's own archive (not on OpenAQ),
  a data-sharing agreement, or its own instrumentation. That is an infrastructure
  conclusion, and it is worth stating plainly in a thesis about a device intended
  for this population.

Reproduce with `python -m src.preprocessing.openaq_survey`.
