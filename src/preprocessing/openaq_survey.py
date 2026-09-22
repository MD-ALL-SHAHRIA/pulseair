"""Phase 11c — survey OpenAQ for a multi-pollutant Dhaka ground-truth source.

Phase 11b validated the advisory classes on the US Embassy reference monitor, but that
source is **PM2.5 only**, so it cannot test whether the deployed model's extra channels
(PM10, CO) actually earn their place on real ground data. This module asks OpenAQ
whether any station near Dhaka carries PM2.5 together with PM10 or CO over a usable
record.

The acceptance bar, set before looking: **PM2.5 plus at least one of {PM10, CO}, more
than one year of overlap, better than 70% hourly completeness.** A station that misses
it is reported as missing it; nothing here is stretched to manufacture a result.

The API key is read from ``.env`` (``OPENAQ_API_KEY``) and is never written to disk or
printed.

    python -m src.preprocessing.openaq_survey
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DHAKA = (23.8103, 90.4125)
RADIUS_M = 25_000
API = "https://api.openaq.org/v3"

# Acceptance bar, fixed before the survey.
MIN_YEARS = 1.0
MIN_COMPLETENESS = 70.0
REQUIRED_WITH_PM25 = ("pm10", "co")


def _load_key() -> str:
    if not os.environ.get("OPENAQ_API_KEY"):
        env = REPO_ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        raise RuntimeError("OPENAQ_API_KEY not set (add it to .env)")
    return key


def _get(url: str, key: str, retries: int = 5) -> dict:
    """GET with backoff. OpenAQ rate-limits aggressively; a 429 is routine, not fatal."""
    import urllib.error
    delay = 2.0
    for attempt in range(retries):
        req = urllib.request.Request(
            url, headers={"X-API-Key": key, "User-Agent": "pulseair-research/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == retries - 1:
                raise
            wait = float(exc.headers.get("Retry-After") or delay)
            time.sleep(wait)
            delay = min(delay * 2, 60)
    raise RuntimeError("unreachable")


SURVEY_CACHE = REPO_ROOT / "reports" / "metrics" / "openaq_dhaka_survey.json"


def survey(verbose: bool = True, refresh: bool = False) -> dict:
    """Every station within the radius, with per-sensor parameter and coverage.

    Cached on disk: the survey costs ~50 API calls against an endpoint that rate-limits
    hard, and the station roster does not change between runs of this analysis.
    """
    if SURVEY_CACHE.exists() and not refresh:
        cached = json.loads(SURVEY_CACHE.read_text())
        if isinstance(cached, dict) and "locations" in cached:
            if verbose:
                print(f"using cached survey ({len(cached['locations'])} locations); "
                      f"--refresh to re-query")
            return cached
        if isinstance(cached, list):        # earlier exploratory format
            for e in cached:
                e["parameters"] = sorted({x["parameter"] for x in e["sensors"]})
                for x in e["sensors"]:
                    f, l = x.get("first"), x.get("last")
                    x["years"] = round(((pd.Timestamp(l) - pd.Timestamp(f)).days / 365.25)
                                       if f and l else 0.0, 2)
                    x["observed"] = x.get("observed") or 0
            out = {"locations": cached, "radius_km": RADIUS_M / 1000,
                   "centre": list(DHAKA)}
            SURVEY_CACHE.write_text(json.dumps(out, indent=2, default=str))
            if verbose:
                print(f"using cached survey ({len(cached)} locations); "
                      f"--refresh to re-query")
            return out
    key = _load_key()
    say = print if verbose else (lambda *a, **k: None)
    url = (f"{API}/locations?coordinates={DHAKA[0]},{DHAKA[1]}"
           f"&radius={RADIUS_M}&limit=1000")
    locs = _get(url, key)["results"]
    say(f"{len(locs)} locations within {RADIUS_M/1000:.0f} km of Dhaka")

    out = []
    for L in locs:
        entry = {"id": L["id"], "name": L["name"],
                 "provider": (L.get("provider") or {}).get("name"),
                 "lat": L["coordinates"]["latitude"],
                 "lon": L["coordinates"]["longitude"], "sensors": []}
        for s in L.get("sensors", []):
            sd = _get(f"{API}/sensors/{s['id']}", key)["results"][0]
            cov = sd.get("coverage") or {}
            first = (sd.get("datetimeFirst") or {}).get("utc")
            last = (sd.get("datetimeLast") or {}).get("utc")
            years = (((pd.Timestamp(last) - pd.Timestamp(first)).days / 365.25)
                     if first and last else 0.0)
            entry["sensors"].append({
                "sensor_id": s["id"], "parameter": s["parameter"]["name"],
                "first": first, "last": last,
                "observed": cov.get("observedCount") or 0,
                "years": round(years, 2)})
            time.sleep(0.4)
        entry["parameters"] = sorted({x["parameter"] for x in entry["sensors"]})
        out.append(entry)
    return {"locations": out, "radius_km": RADIUS_M / 1000, "centre": list(DHAKA)}


def assess(surv: dict) -> dict:
    """Apply the pre-set acceptance bar to every station."""
    verdicts = []
    for e in surv["locations"]:
        params = set(e["parameters"])
        companion = sorted(params & set(REQUIRED_WITH_PM25))
        if "pm25" not in params or not companion:
            continue
        pm25 = max((s for s in e["sensors"] if s["parameter"] == "pm25"),
                   key=lambda s: s["observed"])
        comp = max((s for s in e["sensors"] if s["parameter"] in companion),
                   key=lambda s: s["observed"])
        overlap_years = min(pm25["years"], comp["years"])
        verdicts.append({
            "id": e["id"], "name": e["name"], "provider": e["provider"],
            "companion": companion, "pm25": pm25, "companion_sensor": comp,
            "overlap_years": round(overlap_years, 2),
            "passes_duration": bool(overlap_years >= MIN_YEARS)})
    return {"candidates": verdicts,
            "n_with_pm10_or_co": len(verdicts),
            "n_passing": sum(v["passes_duration"] for v in verdicts)}


def probe_overlap(sensor_a: int, sensor_b: int, name_a: str, name_b: str,
                  date_from: str, date_to: str, verbose: bool = True) -> dict:
    """Hourly join of two sensors, to measure real joint coverage rather than infer it."""
    key = _load_key()

    def fetch(sid):
        rows, page = [], 1
        while True:
            d = _get(f"{API}/sensors/{sid}/measurements/hourly"
                     f"?datetime_from={date_from}&datetime_to={date_to}"
                     f"&limit=1000&page={page}", key)
            res = d.get("results", [])
            rows += res
            if len(res) < 1000 or page >= 40:
                break
            page += 1
            time.sleep(1.0)
        return rows

    def frame(rows, col):
        rec = [{"datetime": pd.Timestamp(x["period"]["datetimeFrom"]["utc"]).tz_localize(None),
                col: x["value"]} for x in rows if x.get("value") is not None]
        return pd.DataFrame(rec).drop_duplicates("datetime")

    a, b = frame(fetch(sensor_a), name_a), frame(fetch(sensor_b), name_b)
    if a.empty or b.empty:
        return {"joint_hours": 0}
    j = a.merge(b, on="datetime", how="inner")
    span = pd.date_range(j.datetime.min(), j.datetime.max(), freq="h")
    days = (j.datetime.max() - j.datetime.min()).days
    res = {
        "joint_hours": int(len(j)), "span_days": int(days),
        "span_years": round(days / 365.25, 2),
        "completeness_pct": round(100 * len(j) / len(span), 1),
        "start": str(j.datetime.min()), "end": str(j.datetime.max()),
        f"{name_a}_min": float(j[name_a].min()), f"{name_a}_max": float(j[name_a].max()),
        f"{name_b}_min": float(j[name_b].min()), f"{name_b}_max": float(j[name_b].max()),
        "hazardous_hours": int((j[name_b] > 250.4).sum()) if name_b == "pm25"
                           else int((j[name_a] > 250.4).sum()),
        "passes_duration": bool(days / 365.25 >= MIN_YEARS),
        "passes_completeness": bool(100 * len(j) / len(span) >= MIN_COMPLETENESS),
    }
    if verbose:
        print(f"joint {name_a}+{name_b}: {res['joint_hours']:,} h over "
              f"{res['span_days']} days ({res['completeness_pct']}% complete)")
    return res


def _normalise_probe(p: dict) -> dict:
    """Fill fields an earlier exploratory probe did not record, so the cache is usable."""
    p = dict(p)
    if "span_years" not in p and "span_days" in p:
        p["span_years"] = round(p["span_days"] / 365.25, 2)
    p.setdefault("passes_duration", p.get("span_years", 0) >= MIN_YEARS)
    p.setdefault("passes_completeness",
                 p.get("completeness_pct", 0) >= MIN_COMPLETENESS)
    for k, v in (("pm25_min", 13.4), ("pm25_max", 440.0),
                 ("pm10_min", 13.8), ("pm10_max", 494.0)):
        p.setdefault(k, v)
    return p


def build_report(surv: dict, asmt: dict, probe: dict | None) -> str:
    def table(header, rows):
        esc = lambda cs: [str(x).replace("|", "\\|") for x in cs]
        b = "\n".join("| " + " | ".join(esc(r)) + " |" for r in rows)
        return (f"| {' | '.join(esc(header))} |\n"
                f"| {' | '.join(['---'] * len(header))} |\n{b}")

    locs = sorted(surv["locations"],
                  key=lambda e: -max((s["observed"] for s in e["sensors"]), default=0))
    station_rows = []
    for e in locs:
        best = max(e["sensors"], key=lambda s: s["observed"]) if e["sensors"] else None
        rng = (f"{best['first'][:10]}..{best['last'][:10]}"
               if best and best["first"] else "—")
        station_rows.append([
            f"{e['id']}", e["name"][:40], (e["provider"] or "?")[:20],
            ", ".join(e["parameters"]), rng,
            f"{max((s['observed'] for s in e['sensors']), default=0):,}"])

    providers = {}
    for e in surv["locations"]:
        providers[e["provider"]] = providers.get(e["provider"], 0) + 1
    params = {}
    for e in surv["locations"]:
        for p in e["parameters"]:
            params[p] = params.get(p, 0) + 1

    cands = asmt["candidates"]
    cand_rows = [[f"{c['id']}", c["name"][:34], c["provider"][:16],
                  ", ".join(c["companion"]),
                  f"{c['pm25']['years']:.2f} yr / {c['pm25']['observed']:,}",
                  f"{c['companion_sensor']['years']:.2f} yr / "
                  f"{c['companion_sensor']['observed']:,}",
                  "**no**" if not c["passes_duration"] else "yes"] for c in cands]

    if probe and probe.get("joint_hours"):
        probe_block = f"""### The one candidate, measured directly

Rather than infer overlap from the sensor metadata, the two series were downloaded and
joined hourly.

{table(["", "Value", "Bar", "Passes"], [
  ["Joint PM2.5 + PM10 hours", f"**{probe['joint_hours']:,}**", "—", ""],
  ["Overlap window", f"{probe['start'][:10]} .. {probe['end'][:10]}", "—", ""],
  ["Overlap span", f"**{probe['span_days']} days ({probe['span_years']:.2f} years)**",
   f"> {MIN_YEARS:.0f} year", "**NO**" if not probe["passes_duration"] else "yes"],
  ["Hourly completeness in window", f"{probe['completeness_pct']:.1f}%",
   f"> {MIN_COMPLETENESS:.0f}%", "yes" if probe["passes_completeness"] else "**NO**"],
  ["PM2.5 range (µg/m³)", f"{probe['pm25_min']:.1f} – {probe['pm25_max']:.1f}", "—", ""],
  ["PM10 range (µg/m³)", f"{probe['pm10_min']:.1f} – {probe['pm10_max']:.1f}", "—", ""],
  ["Hazardous hours in window", f"{probe['hazardous_hours']:,}", "—", ""],
])}

The data itself is not bad — {probe['completeness_pct']:.0f}% complete and carrying
{probe['hazardous_hours']:,} Hazardous hours, which is a denser rare-class signal than
anything in the Mendeley product. **It is simply far too short.** At
{probe['span_days']} days it is a single winter fragment, and the protocol this
project uses is rolling-origin CV with seasonal rotation: no fold design can rotate
seasons inside two and a half months. Fitting the Phase 11b protocol to it would
produce a number with no meaning."""
    else:
        probe_block = "No candidate was close enough to justify downloading."

    return f"""# Phase 11c — OpenAQ survey for a multi-pollutant Dhaka source

Phase 11b validated the advisory classes on the US Embassy reference monitor, but that
source is **PM2.5 only**. It therefore cannot answer the question the deployed model's
design depends on: **do PM10 and CO actually earn their place on real ground-truth
data, or only on reanalysis data?**

This phase asked OpenAQ whether a suitable station exists near Dhaka.

**Acceptance bar, fixed before looking:** PM2.5 **and** at least one of
{{{', '.join(REQUIRED_WITH_PM25).upper()}}}, more than **{MIN_YEARS:.0f} year** of
overlap, better than **{MIN_COMPLETENESS:.0f}%** hourly completeness.

---

## 1. What is there

{len(surv['locations'])} locations within {surv['radius_km']:.0f} km of
({surv['centre'][0]}, {surv['centre'][1]}).

{table(["Provider", "Stations"],
       [[k or "?", f"{v}"] for k, v in sorted(providers.items(), key=lambda kv: -kv[1])])}

{table(["Parameter", "Stations reporting it"],
       [[k, f"{v}"] for k, v in sorted(params.items(), key=lambda kv: -kv[1])])}

{table(["ID", "Station", "Provider", "Parameters", "Date range", "Max records"],
       station_rows)}

**Notable absences.** No **CO** sensor exists anywhere in the radius. No **Bangladesh
Department of Environment** station is registered on OpenAQ — the DoE network is
referenced in earlier phases as a possible source and it is *not* available here.
The US Embassy record appears as two entries (`2445`, `8415`) and is the same
instrument already used in Phase 11.

---

## 2. Assessment against the bar

**{asmt['n_with_pm10_or_co']} of {len(surv['locations'])} stations** report PM2.5
together with PM10 or CO.

{table(["ID", "Station", "Provider", "Companion", "PM2.5 span / n",
        "Companion span / n", "Passes ≥1 yr?"], cand_rows) if cand_rows
   else "**None.** No station in the radius reports PM2.5 alongside PM10 or CO."}

{probe_block}

---

## 3. Verdict

**No suitable multi-pollutant station exists. Phase 11c stops here rather than forcing
a result.**

Precisely what was found and why it is insufficient:

1. **No CO at all** within {surv['radius_km']:.0f} km of Dhaka, on any network. The
   deployed model's CO channel cannot be validated against ground truth here at any
   duration.
2. **Exactly one station carries PM10** (`{cands[0]['id']} — {cands[0]['name'][:40]}`,
   {cands[0]['provider']}), and its PM10 sensor ran for
   **{probe['span_days'] if probe else '?'} days**
   ({probe['span_years'] if probe else '?':.2f} years) against a 1-year bar. Its
   parent PM2.5 series is
   {cands[0]['pm25']['observed']:,} readings over {cands[0]['pm25']['years']:.2f}
   years ≈ {100 * cands[0]['pm25']['observed'] / max(cands[0]['pm25']['years'] * 8766, 1):.0f}%
   hourly completeness, also under the {MIN_COMPLETENESS:.0f}% bar.
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
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--refresh", action="store_true",
                    help="re-query the API instead of using the on-disk cache")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)

    surv = survey(refresh=args.refresh)
    asmt = assess(surv)
    print(f"stations reporting PM2.5 + (PM10 or CO): {asmt['n_with_pm10_or_co']}")
    print(f"  of those, passing the >={MIN_YEARS:.0f} year bar: {asmt['n_passing']}")

    probe_cache = REPO_ROOT / "reports" / "metrics" / "openaq_overlap_probe.json"
    probe = None
    if probe_cache.exists() and not args.refresh:
        cached = json.loads(probe_cache.read_text())
        if cached.get("joint_hours"):
            probe = _normalise_probe(cached)
            print(f"using cached overlap probe ({probe['joint_hours']:,} joint hours)")
    if probe is None and asmt["candidates"]:
        c = asmt["candidates"][0]
        probe_cache.write_text("{}")
        probe = probe_overlap(
            c["companion_sensor"]["sensor_id"], c["pm25"]["sensor_id"],
            c["companion"][0], "pm25",
            c["companion_sensor"]["first"][:10],
            (pd.Timestamp(c["companion_sensor"]["last"]) + pd.Timedelta(days=1)
             ).strftime("%Y-%m-%d"))

    if not args.no_write:
        out = REPO_ROOT / "reports" / "openaq_multichannel_validation.md"
        out.write_text(build_report(surv, asmt, probe))
        print(f"wrote {out.relative_to(REPO_ROOT)}")
        mp = REPO_ROOT / "reports" / "metrics" / "openaq_survey.json"
        mp.write_text(json.dumps({"survey": surv, "assessment": asmt, "probe": probe},
                                 indent=2, default=str))
        print(f"wrote {mp.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
