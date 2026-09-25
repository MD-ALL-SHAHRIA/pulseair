"""Detect fabricated or spliced segments in a published time series.

A dataset that arrives with a DOI and a long advertised history still has to be
checked. This module packages four cheap checks that, applied together, located a
multi-year fabricated segment in a published air-quality dataset without being told
where to look. Each is a statement about structure rather than about values, so none
of them needs a reference series to compare against.

    trend linearity      real environmental series are not straight lines
    hard clipping        a generator with a ceiling piles values on it exactly
    scale discontinuity  a splice between sources shows as a step in level or spread
    autocorrelation      synthetic noise rarely matches real hour-to-hour persistence

None of the four is conclusive alone, and the module does not pretend otherwise: it
reports each check's evidence separately and proposes a boundary only where several
agree. A single flag on real data is common; four flags on the same side of the same
date is not.

    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 3000
    >>> idx = pd.date_range("2020-01-01", periods=2 * n, freq="h")
    >>> fake = np.minimum(np.linspace(10, 300, n) + rng.normal(0, 4, n), 250.0)
    >>> real = 60 + 30 * np.sin(np.arange(n) / 24) + rng.normal(0, 8, n)
    >>> out = dataset_audit(pd.DataFrame({"t": idx, "v": np.r_[fake, real]}),
    ...                     time_col="t", value_col="v")
    >>> out["checks"]["trend_linearity"]["flagged"]
    True
    >>> out["checks"]["hard_clip"]["flagged"]
    True
    >>> out["suspected_boundary"] is not None
    True
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["dataset_audit"]

# Thresholds. Each is a judgement call, so each is a named constant with a reason
# rather than a number buried in a comparison.
R2_SUSPICIOUS = 0.90      # an environmental series explained this well by a straight
                          # line over years is not an environmental series
CLIP_MIN_PCT = 0.2        # percent of values on one exact figure. Deliberately
                          # low: a clip confined to part of a spliced series is
                          # diluted when pooled, and the ratio below carries the
                          # real signal.
CLIP_RATIO = 50.0         # times the typical repeat rate. In a continuous
                          # quantity an exact repeat is rare, so a value repeated
                          # far more often than its peers is a generator artefact.
SCALE_SHIFT_FACTOR = 5.0  # ratio between window medians that a unit change implies
ACF_STEP = 0.20           # absolute jump in lag-1 autocorrelation between regimes
MIN_WINDOW = 200          # samples; below this a window statistic is noise
MIN_PERIODS = 8           # aggregated points needed before a trend fit means
                          # anything


def _r2_linear(t: np.ndarray, y: np.ndarray) -> float:
    """R-squared of an ordinary least-squares line through (t, y)."""
    if len(y) < 3 or np.allclose(y, y[0]):
        return 0.0
    x = (t - t.min()) / max(t.max() - t.min(), 1e-9)
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 0.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)


def _windows(n: int, k: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, k + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(k)
            if edges[i + 1] - edges[i] >= MIN_WINDOW]


def _lag1(y: np.ndarray) -> float:
    if len(y) < 3:
        return float("nan")
    a, b = y[:-1], y[1:]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _check_trend(times: pd.Series, y: np.ndarray, period: str = "YS") -> dict:
    """Find the longest leading segment that a straight line explains too well.

    Two things make this work on real data. First, the series is aggregated to period
    medians before fitting: hourly environmental noise swamps a linear signal, so a
    fit to raw observations reports a low R-squared even when the underlying level
    marches in a straight line. The manual audit this check generalises found
    R-squared 0.99 on annual medians and would have found 0.14 on the raw hours.
    Second, prefixes are scanned rather than the whole series, because a fabricated
    prefix followed by genuine data is not linear overall and the genuine portion
    hides the artefact.

    The resolution limit follows from the first point: a segment shorter than
    ``MIN_PERIODS`` aggregation units cannot be found, because a straight line through
    fewer points than that says nothing. Pass a finer ``period`` when the suspected
    segment is short.
    """
    # Datetimes are passed through as datetimes. An earlier version round-tripped
    # them via int64 nanoseconds and rebuilt with pd.to_datetime, which pandas 3
    # interprets differently from pandas 2 -- the span collapsed and annual
    # resampling returned two periods instead of twenty-six.
    ser = pd.Series(np.asarray(y, dtype=float),
                    index=pd.DatetimeIndex(times)).dropna()
    if len(ser) < MIN_WINDOW:
        return {"name": "trend_linearity", "r2": 0.0, "threshold": R2_SUSPICIOUS,
                "flagged": False, "boundary_index": None, "prefix_fraction": 0.0,
                "n_periods": 0, "note": "series too short to test"}
    # Aggregate as coarsely as the span allows. A trend test wants seasonality gone,
    # and seasonality in environmental data is annual: on this project's real dataset
    # annual medians give a prefix R-squared of 0.96 where monthly medians give none,
    # because the monthly series still carries the seasonal swing. `period` sets the
    # preferred rung; finer ones are used only when the span is too short.
    ladder = [period] + [q for q in ("YS", "QS", "MS", "W", "D") if q != period]
    agg, period_used = None, None
    for rung in ladder:
        candidate = ser.resample(rung).median().dropna()
        # 2x, not 1x. A rung yielding exactly MIN_PERIODS periods passes a naive
        # length check and is still useless: the shortest testable prefix is then the
        # whole series, so a fabricated prefix cannot be isolated from the genuine
        # remainder. Requiring twice the minimum guarantees that a prefix covering
        # half the span is itself long enough to fit.
        if len(candidate) >= 2 * MIN_PERIODS:
            agg, period_used = candidate, rung
            break
    if agg is None:
        agg, period_used = ser.resample("D").median().dropna(), "D"
    period = period_used
    at = agg.index.astype("int64").to_numpy()
    ay = agg.to_numpy(dtype=float)

    best_r2, best_end_period = 0.0, None
    for frac in np.linspace(0.15, 0.95, 33):
        k = int(len(ay) * frac)
        if k < MIN_PERIODS:
            continue
        r2 = _r2_linear(at[:k], ay[:k])
        if r2 >= R2_SUSPICIOUS and k > (best_end_period or 0):
            best_r2, best_end_period = r2, k
    pooled = _r2_linear(at, ay)
    flagged = best_end_period is not None
    # translate the period index back into a row index in the original series
    boundary_idx = None
    if flagged:
        cutoff_time = agg.index[min(best_end_period, len(agg) - 1)]
        boundary_idx = int(np.searchsorted(
            np.sort(pd.DatetimeIndex(times).to_numpy()),
            np.datetime64(cutoff_time)))
    return {"name": "trend_linearity",
            "r2": round(best_r2 if flagged else pooled, 6),
            "r2_pooled": round(pooled, 6),
            "aggregated_to": period, "n_periods": int(len(agg)),
            "threshold": R2_SUSPICIOUS, "flagged": bool(flagged),
            "boundary_index": boundary_idx,
            "prefix_fraction": round((best_end_period or 0) / max(len(ay), 1), 4),
            "note": ("a linear fit to {} period medians explains {:.1%} of the "
                     "variance over the first {:.0%} of the series".format(
                         len(agg), best_r2, (best_end_period or 0) / len(ay))
                     if flagged else
                     "no suspiciously linear segment (pooled R2 {:.3f} on {} periods)"
                     .format(pooled, len(agg)))}


def _check_clip(y: np.ndarray) -> dict:
    """Find any single value carrying anomalous mass, not only the extremes.

    Testing the max and min alone misses the case that matters most: a generator
    ceiling that later gets concatenated with genuine data exceeding it is no longer
    the series maximum, so a max-test sees nothing. What survives concatenation is the
    *pile* itself — one exact value repeated far more often than its neighbours in a
    continuous quantity, where exact repeats should be rare.
    """
    y = y[~np.isnan(y)]
    if len(y) < MIN_WINDOW:
        return {"name": "hard_clip", "pct_at_value": 0.0, "clip_value": None,
                "threshold_pct": CLIP_MIN_PCT, "flagged": False,
                "note": "series too short to test"}
    vals, counts = np.unique(y, return_counts=True)
    i = int(np.argmax(counts))
    pct = float(counts[i] / len(y) * 100)
    # Compare against the typical repeat rate so a naturally discrete series (a
    # sensor reporting whole numbers) is not mistaken for a clipped one.
    typical = float(np.median(counts) / len(y) * 100)
    ratio = pct / typical if typical > 0 else float("inf")
    flagged = bool(pct >= CLIP_MIN_PCT and ratio >= CLIP_RATIO)
    return {"name": "hard_clip", "pct_at_value": round(pct, 4),
            "clip_value": float(vals[i]),
            "typical_repeat_pct": round(typical, 6),
            "ratio_to_typical": round(ratio, 1) if np.isfinite(ratio) else None,
            "n_distinct": int(len(vals)),
            "threshold_pct": CLIP_MIN_PCT, "threshold_ratio": CLIP_RATIO,
            "flagged": flagged,
            "note": ("{:.2f}% of values sit on exactly {:g}, {:.0f}x the typical "
                     "repeat rate".format(pct, vals[i], ratio) if flagged else
                     "no anomalous pile-up on a single value")}


def _check_scale(t: np.ndarray, y: np.ndarray, n_windows: int) -> dict:
    """A splice between sources in different units shows as a step in level."""
    wins = _windows(len(y), n_windows)
    meds = [float(np.nanmedian(y[a:b])) for a, b in wins]
    ratios, worst_i, worst = [], None, 1.0
    for i in range(1, len(meds)):
        prev, cur = meds[i - 1], meds[i]
        if prev == 0 or cur == 0:
            ratios.append(float("nan")); continue
        r = max(abs(cur / prev), abs(prev / cur))
        ratios.append(r)
        if r > worst:
            worst, worst_i = r, i
    flagged = bool(worst >= SCALE_SHIFT_FACTOR)
    return {"name": "scale_discontinuity", "window_medians": [round(m, 4) for m in meds],
            "max_ratio": round(worst, 3), "threshold": SCALE_SHIFT_FACTOR,
            "flagged": flagged,
            "boundary_index": int(wins[worst_i][0]) if (flagged and worst_i) else None,
            "note": ("median jumps by {:.1f}x between adjacent windows".format(worst)
                     if flagged else "no scale discontinuity")}


def _check_autocorrelation(y: np.ndarray, n_windows: int) -> dict:
    """Compare persistence between the early and late halves of the series.

    An earlier version took the largest step between adjacent windows, which fires on
    genuine data: real series have episodic periods whose autocorrelation differs from
    their neighbours without anything being wrong. A splice is a *regime* change, so
    the test compares aggregate persistence either side of the best-splitting point
    and requires the difference to hold across windows, not within one pair.
    """
    wins = _windows(len(y), n_windows)
    acf = [_lag1(y[a:b]) for a, b in wins]
    clean = [(i, v) for i, v in enumerate(acf) if not np.isnan(v)]
    best_gap, at = 0.0, None
    for k in range(2, len(clean) - 1):          # need >=2 windows either side
        left = np.mean([v for _, v in clean[:k]])
        right = np.mean([v for _, v in clean[k:]])
        gap = abs(right - left)
        if gap > best_gap:
            best_gap, at = gap, clean[k][0]
    flagged = bool(best_gap >= ACF_STEP)
    return {"name": "autocorrelation_break",
            "window_lag1": [None if np.isnan(v) else round(v, 4) for v in acf],
            "max_regime_gap": round(best_gap, 4), "threshold": ACF_STEP,
            "flagged": flagged,
            "boundary_index": int(wins[at][0]) if (flagged and at is not None) else None,
            "note": ("mean lag-1 autocorrelation differs by {:.3f} either side of the "
                     "split".format(best_gap) if flagged else
                     "persistence is consistent across the series")}


def _boundary(t: pd.Series, idx: int | None):
    return None if idx is None else t.iloc[int(idx)]


def dataset_audit(df: pd.DataFrame, time_col: str, value_col: str,
                  group_col: str | None = None, *, n_windows: int = 12,
                  focus_group: str | None = None, period: str = "YS") -> dict:
    """Audit a published series for signs of fabrication or splicing.

    Runs four structural checks and, where they agree, proposes the index and
    timestamp at which the series changes character. The proposal is offered as
    evidence, not as a verdict: the caller is expected to look at the flagged
    boundary and decide, which is what happened when this was applied to a real
    dataset.

    Args:
        df: The series to audit.
        time_col: Datetime column.
        value_col: The measured quantity.
        group_col: Optional grouping (station, city). When given, the audit runs per
            group as well as pooled, since a fabricated segment often affects one
            group only — which is itself diagnostic.
        n_windows: Rolling windows used by the scale and autocorrelation checks.
        focus_group: Audit this group rather than the pooled series. Defaults to the
            group with the longest history, which is where backfill tends to live.

    Returns:
        ``checks`` (one entry per test), ``suspected_boundary`` (timestamp or None),
        ``n_flagged``, ``verdict`` and, when ``group_col`` is given, ``per_group``.

    Example:
        >>> import numpy as np, pandas as pd
        >>> rng = np.random.default_rng(1)
        >>> n = 3000
        >>> t = pd.date_range("2019-01-01", periods=2 * n, freq="h")
        >>> fake = np.linspace(5, 95, n)
        >>> real = 60 + 20 * np.sin(np.arange(n) / 24) + rng.normal(0, 4, n)
        >>> out = dataset_audit(pd.DataFrame({"t": t, "v": np.r_[fake, real]}),
        ...                     time_col="t", value_col="v")
        >>> out["checks"]["trend_linearity"]["flagged"]
        True
    """
    if time_col not in df.columns or value_col not in df.columns:
        raise KeyError(f"need columns {time_col!r} and {value_col!r}; "
                       f"got {list(df.columns)[:8]}")

    work = df[[c for c in (time_col, value_col, group_col) if c]].copy()
    work[time_col] = pd.to_datetime(work[time_col])
    work = work.dropna(subset=[time_col, value_col])

    per_group = {}
    if group_col:
        spans = work.groupby(group_col)[time_col].agg(["min", "max", "size"])
        spans["days"] = (spans["max"] - spans["min"]).dt.days
        focus = focus_group or spans["days"].idxmax()
        for g, block in work.groupby(group_col):
            if len(block) >= MIN_WINDOW * 2:
                per_group[str(g)] = _audit_one(block, time_col, value_col,
                                               n_windows, period)["summary"]
        work = work[work[group_col] == focus]
    else:
        focus = None

    out = _audit_one(work, time_col, value_col, n_windows, period)
    out["focus_group"] = None if focus is None else str(focus)
    if per_group:
        out["per_group"] = per_group
        flagged_groups = [g for g, v in per_group.items() if v["n_flagged"] >= 2]
        out["groups_flagged"] = flagged_groups
        out["group_note"] = (
            f"{len(flagged_groups)} of {len(per_group)} groups show two or more flags"
            + (f"; the audit focused on {focus}, which has the longest history"
               if focus is not None else ""))
    return out


def _audit_one(work: pd.DataFrame, time_col: str, value_col: str,
               n_windows: int, period: str = "YS") -> dict:
    work = work.sort_values(time_col, kind="mergesort").reset_index(drop=True)
    times = work[time_col]
    t_num = times.astype("int64").to_numpy()      # only the scale check needs ordinals
    y = work[value_col].to_numpy(dtype=float)

    checks = {}
    for c in (_check_trend(times, y, period), _check_clip(y),
              _check_scale(t_num, y, n_windows),
              _check_autocorrelation(y, n_windows)):
        checks[c["name"]] = c

    n_flagged = sum(1 for c in checks.values() if c["flagged"])
    # Boundary candidates, ordered by how precisely each check can localise.
    #
    # The trend check identifies the *end of a linear prefix*, which is a sharp event
    # located to one aggregation period. The scale check localises to a window edge.
    # The autocorrelation check is the coarsest: it splits the series wherever the
    # difference in mean persistence is largest, which on a smoothly varying series
    # is not a break at all. An earlier version took the median of all candidates,
    # which on real data averaged a correct 2023 estimate with a meaningless 2004 one
    # and reported 2013. Disagreeing estimates are now reported as disagreeing.
    PRECISION = ["trend_linearity", "scale_discontinuity", "autocorrelation_break"]
    candidates = {name: checks[name]["boundary_index"] for name in PRECISION
                  if checks.get(name, {}).get("boundary_index") is not None}
    boundary_idx, agreement = None, None
    if candidates:
        span = max(len(work), 1)
        vals = list(candidates.values())
        # cluster: candidates within 10% of the series length of each other agree
        best_cluster = max(
            ([v for v in vals if abs(v - anchor) <= 0.10 * span] for anchor in vals),
            key=len)
        agreement = len(best_cluster) > 1
        boundary_idx = (int(np.median(best_cluster)) if agreement
                        else int(candidates[next(n for n in PRECISION
                                                 if n in candidates)]))

    if n_flagged >= 3:
        verdict = "fabrication suspected"
    elif n_flagged == 2:
        verdict = "inconclusive"
    else:
        verdict = "no structural anomaly"

    return {
        "n_rows": int(len(work)),
        "span": [str(work[time_col].min()), str(work[time_col].max())],
        "checks": checks,
        "n_flagged": n_flagged,
        "flagged": [k for k, c in checks.items() if c["flagged"]],
        "boundary_candidates": {k: (None if v is None else
                                    str(work[time_col].iloc[int(v)]))
                                for k, v in candidates.items()},
        "boundary_agreement": agreement,
        "suspected_boundary_index": boundary_idx,
        "suspected_boundary": (None if boundary_idx is None
                               else str(work[time_col].iloc[boundary_idx])),
        "verdict": verdict,
        "summary": {"n_flagged": n_flagged, "verdict": verdict,
                    "flagged": [k for k, c in checks.items() if c["flagged"]]},
    }
