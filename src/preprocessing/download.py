"""Fetch the Beijing Multi-Site Air Quality dataset (UCI id=501).

``ucimlrepo.fetch_ucirepo(id=501)`` does **not** work for this dataset: the UCI API
reports it as "exists in the repository, but is not available for import", because it
ships as a nested zip of 12 per-station CSVs rather than a single flat table.

So we pull the archive directly and concatenate the station files ourselves. The result
is the combined 420,768-row table written to ``data/raw/beijing_multisite.csv``.

    python -m src.preprocessing.download
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd
import urllib.request

UCI_ID = 501
URL = "https://archive.ics.uci.edu/static/public/501/beijing+multi+site+air+quality+data.zip"
EXPECTED_ROWS = 420_768
EXPECTED_STATIONS = 12
DEFAULT_OUT = Path("data/raw/beijing_multisite.csv")


def _station_frames(archive: bytes):
    """Yield one DataFrame per station from the nested zip."""
    with zipfile.ZipFile(io.BytesIO(archive)) as outer:
        # The payload is a zip inside the zip.
        inner_name = next(n for n in outer.namelist() if n.endswith(".zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
            members = sorted(
                n for n in inner.namelist()
                if n.endswith(".csv") and "PRSA_Data_" in n and not n.startswith("__MACOSX")
            )
            for name in members:
                yield pd.read_csv(io.BytesIO(inner.read(name)))


def download(out_path: Path = DEFAULT_OUT, force: bool = False) -> pd.DataFrame:
    if out_path.exists() and not force:
        print(f"{out_path} already exists; reading it (use --force to re-download)")
        return pd.read_csv(out_path)

    print(f"downloading UCI id={UCI_ID} ...")
    with urllib.request.urlopen(URL) as resp:
        archive = resp.read()

    frames = list(_station_frames(archive))
    df = pd.concat(frames, ignore_index=True)

    # The row/station counts are a fixed property of this dataset; a mismatch means the
    # archive changed or the extraction dropped a file, and nothing downstream is valid.
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS:,} rows, got {len(df):,}")
    if df["station"].nunique() != EXPECTED_STATIONS:
        raise ValueError(f"expected {EXPECTED_STATIONS} stations, got {df['station'].nunique()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(df):,} rows x {df.shape[1]} cols, {df['station'].nunique()} stations)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--force", action="store_true", help="re-download even if the CSV exists")
    a = ap.parse_args()
    download(a.out, a.force)
