"""The persistence floor: what a model must beat to have demonstrated anything."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

__all__ = ["persistence_floor", "seasonal_naive_floor"]


def persistence_floor(df: pd.DataFrame, target_col: str, horizon: int, *,
                      time_col: str | None = None, freq: str = "h",
                      group_col: str | None = None,
                      labels: list | None = None) -> dict:
    """Score the zero-parameter rule "the value at *t+h* equals the value at *t*".

    This is the baseline most seasonal forecasting papers omit, and omitting it is how
    a model that has learned to echo its input gets reported as a forecaster. On the
    air-quality data this package was extracted from, a 1-hour-ahead AQI classifier
    scored 0.7994 macro-F1 against a persistence floor of 0.7933 — a gain of +0.006
    that vanished entirely once the floor was reported alongside it.

    Pairs are formed by an explicit time join, not a positional shift, so gaps in the
    series do not silently create pairs that span them.

    Args:
        df: Rows indexed by time, or carrying a datetime column named by ``time_col``.
        target_col: Categorical column to forecast.
        horizon: How far ahead, in units of ``freq``.
        time_col: Datetime column. Defaults to the index.
        freq: Pandas offset alias for one step. Default hourly.
        group_col: Optional grouping (station, city, sensor). Pairs never cross groups.
        labels: Class labels in order. Inferred from the data when omitted.

    Returns:
        ``macro_f1``, ``accuracy``, ``label_unchanged_pct``, ``n_pairs``,
        ``per_class_f1`` and ``support``.

    Example:
        >>> import pandas as pd, numpy as np
        >>> idx = pd.date_range("2024-01-01", periods=200, freq="h")
        >>> rng = np.random.default_rng(0)
        >>> y = pd.Series(rng.integers(0, 3, len(idx)), index=idx)
        >>> out = persistence_floor(y.to_frame("risk"), "risk", horizon=6)
        >>> out["n_pairs"]
        194
        >>> 0.0 <= out["macro_f1"] <= 1.0
        True
        >>> sorted(out["support"])
        [0, 1, 2]
    """
    d = df.copy()
    if time_col is None:
        if not isinstance(d.index, pd.DatetimeIndex):
            raise TypeError("index must be a DatetimeIndex, or pass time_col=")
        d = d.reset_index(names="__t")
        time_col = "__t"
    d[time_col] = pd.to_datetime(d[time_col])

    step = pd.tseries.frequencies.to_offset(freq) * horizon
    keys = [time_col] + ([group_col] if group_col else [])
    left = d[keys + [target_col]].copy()
    left["__join"] = left[time_col] + step
    right = d[keys + [target_col]].rename(columns={target_col: "__future"})

    merged = left.merge(right, left_on=["__join"] + ([group_col] if group_col else []),
                        right_on=keys, how="inner", suffixes=("", "__r"))
    if merged.empty:
        raise ValueError("no (t, t+horizon) pairs — check horizon, freq and gaps")

    now = merged[target_col].to_numpy()
    future = merged["__future"].to_numpy()
    return _score_pairs(now, future, labels)


def seasonal_naive_floor(df: pd.DataFrame, target_col: str, horizon: int, *,
                         season_length: int, time_col: str | None = None,
                         freq: str = "h", group_col: str | None = None,
                         labels: list | None = None) -> dict:
    """Predict t+h from the latest matching seasonal time at or before t.

    ``horizon`` and ``season_length`` are positive integer counts of ``freq``.
    The prediction is y[t+h-k*season_length], where k=ceil(h/season_length),
    including horizons longer than one season without using future observations.
    Both the origin-to-target and seasonal-source matches are time joins within
    ``group_col`` (when given); missing timestamps are never replaced by row shifts.
    Other arguments and returned metrics match :func:`persistence_floor`.
    ``label_unchanged_pct`` here compares the seasonal source with the target.
    Pair counts can differ between baselines when seasonal history is unavailable.

    >>> idx = pd.date_range("2024-01-01", periods=12, freq="h")
    >>> df = pd.DataFrame({"y": [0, 1, 2] * 4}, index=idx)
    >>> out = seasonal_naive_floor(df, "y", 1, season_length=3)
    >>> out["n_pairs"], out["macro_f1"]
    (9, 1.0)
    """
    for name, value in (("horizon", horizon), ("season_length", season_length)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    d = df.copy()
    if time_col is None:
        if not isinstance(d.index, pd.DatetimeIndex):
            raise TypeError("index must be a DatetimeIndex, or pass time_col=")
        d = d.reset_index(names="__t")
        time_col = "__t"
    d[time_col] = pd.to_datetime(d[time_col])
    offset = pd.tseries.frequencies.to_offset(freq)
    groups = [group_col] if group_col else []
    keys = [time_col] + groups
    origins = d[keys].copy()
    origins[time_col] = origins[time_col] + offset * horizon
    targets = origins.merge(d[keys + [target_col]], on=keys, how="inner")
    lag = ((int(horizon) - 1) // int(season_length) + 1) * int(season_length)
    sources = d[keys + [target_col]].rename(columns={target_col: "__seasonal"})
    sources[time_col] = sources[time_col] + offset * lag
    pairs = targets.merge(sources, on=keys, how="inner")
    if pairs.empty:
        raise ValueError("no seasonal pairs — check horizon, season_length, freq and gaps")
    return _score_pairs(pairs["__seasonal"].to_numpy(), pairs[target_col].to_numpy(), labels)


def _score_pairs(now: np.ndarray, future: np.ndarray, labels: list | None) -> dict:
    # .tolist() matters: np.unique hands back numpy scalars, which become numpy
    # scalars in the returned dicts' KEYS. That makes the result unserializable by
    # json.dump and, since numpy 2 changed scalar repr, makes it print differently
    # depending on which numpy is installed. Callers get plain Python values.
    classes = list(labels) if labels is not None else sorted(
        set(np.unique(now).tolist()) | set(np.unique(future).tolist()))
    idx = list(range(len(classes)))
    code = {c: i for i, c in enumerate(classes)}
    a = np.array([code[v] for v in now])
    b = np.array([code[v] for v in future])

    return {
        "n_pairs": int(len(now)),
        "macro_f1": float(f1_score(b, a, labels=idx, average="macro", zero_division=0)),
        "accuracy": float((a == b).mean()),
        "label_unchanged_pct": float((a == b).mean() * 100),
        "per_class_f1": {classes[i]: float(f1_score(b == i, a == i, zero_division=0))
                         for i in idx},
        "support": {classes[i]: int((b == i).sum()) for i in idx},
        "labels": classes,
    }
