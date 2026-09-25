"""Run ``pulsebench.dataset_audit`` on the unfiltered Mendeley Bangladesh file.

This is the package's validation against a real case, and it is the one number in
the project that had no committed producer: the result in
``reports/metrics/integrity_audit_bangladesh.json`` was originally written by an
ad-hoc command. Anything the thesis cites has to be re-runnable, so it lives here.

Nothing about the known defect is passed in. The audit is given the raw file, the
time column, the value column and the city column, and has to find the boundary,
the affected city and the verdict on its own -- which is the only way the result
says anything about the method rather than about this dataset.

    python -m src.models.integrity_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pulsebench import dataset_audit

RAW = Path("data/raw/bangladesh_aqi.csv")
OUT = Path("reports/metrics/integrity_audit_bangladesh.json")

# What manual inspection of this file found, for the report to measure against.
# It is never given to the audit.
MANUAL_BOUNDARY = pd.Timestamp("2022-08-05")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    df = pd.read_csv(args.raw, usecols=["city_name", "datetime", "pm2_5"])
    print(f"{len(df):,} rows from {args.raw}")

    out = dataset_audit(df, time_col="datetime", value_col="pm2_5",
                        group_col="city_name")

    boundary = out["suspected_boundary"]
    if boundary is not None:
        out["manual_boundary"] = str(MANUAL_BOUNDARY)
        out["boundary_error_days"] = abs((pd.Timestamp(boundary)
                                          - MANUAL_BOUNDARY).days)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))

    print(f"verdict: {out['verdict']} ({out['n_flagged']}/4 checks)")
    print(f"focus city: {out['focus_group']}  (unprompted)")
    print(f"boundary: {str(boundary)[:10]}  vs 2022-08-05 by hand "
          f"({out.get('boundary_error_days', '-')} days)")
    for name, c in out["checks"].items():
        print(f"  {'FLAG' if c['flagged'] else '    '}  {name}: {c['note']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
