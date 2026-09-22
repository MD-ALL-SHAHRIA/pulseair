"""Rolling-origin (walk-forward) cross-validation with embargo and per-fold scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

__all__ = ["make_folds", "rolling_origin_cv", "aggregate_folds"]


def make_folds(times: np.ndarray, n_folds: int, initial_fraction: float) -> list[dict]:
    """Expanding-window fold boundaries over the sorted distinct timestamps."""
    stamps = np.sort(np.unique(times))
    n = len(stamps)
    edges = np.linspace(int(n * initial_fraction), n, n_folds + 1).astype(int)
    return [{"fold": k + 1,
             "cutoff": pd.Timestamp(stamps[edges[k]]),
             "eval_end": pd.Timestamp(stamps[edges[k + 1] - 1])}
            for k in range(n_folds)]


def _score(y, p, labels) -> dict:
    idx = list(range(len(labels)))
    return {"macro_f1": float(f1_score(y, p, labels=idx, average="macro",
                                       zero_division=0)),
            "accuracy": float((y == p).mean()),
            **{f"f1_{l}": float(f1_score(y == i, p == i, zero_division=0))
               for i, l in enumerate(labels)},
            **{f"n_{l}": int((y == i).sum()) for i, l in enumerate(labels)}}


def rolling_origin_cv(df: pd.DataFrame, model_fn, n_folds: int = 5,
                      embargo_hours: int = 24, min_class_support: int | None = None, *,
                      feature_cols: list[str] | None = None, target_col: str = "y",
                      time_col: str | None = None, target_time_col: str | None = None,
                      horizon: int = 0, freq: str = "h",
                      initial_fraction: float = 0.4,
                      scale_cols: list[str] | None = None,
                      observed_col: str | None = None,
                      baseline_col: str | None = None,
                      labels: list | None = None, verbose: bool = False) -> dict:
    """Expanding-window CV that rotates the evaluation block through the series.

    A single chronological split answers "does this model beat the baseline on *this*
    quarter?" — which, on seasonal data, is not the question anyone means. This rotates
    the evaluation block so every fold lands on a different part of the year, and
    reports **how many folds the model wins** alongside the aggregate.

    Three details that are easy to get wrong and are handled here:

    * **Embargo.** Training uses rows whose *target* precedes the cutoff; evaluation
      uses rows whose feature window begins at least ``embargo_hours`` after it.
      Without the gap, an evaluation row's lagged features contain hours the model saw
      as training labels.
    * **Per-fold scaling.** Any scaler is refit inside each fold. Scaling the whole
      frame once leaks future distribution information into every fold.
    * **Sparse classes.** With ``min_class_support``, a class with too few examples in
      an evaluation block is recorded as unevaluable rather than scored 0.0 — an F1 of
      zero on an absent class reads like a model failure and is not one.

    Args:
        df: One row per sample, with features, target, and a time column.
        model_fn: Zero-argument factory returning an unfitted sklearn-like estimator
            (``fit``/``predict``). Called fresh for every fold.
        n_folds: Number of evaluation blocks.
        embargo_hours: Gap between a fold's training end and its evaluation start.
        min_class_support: Minimum examples for a class to be scored in a fold.
        feature_cols: Defaults to every column except target/time/bookkeeping.
        target_col: Integer-coded class column.
        time_col: Datetime column; defaults to the index.
        target_time_col: When the target is at a different time from the row, the
            column holding it. Otherwise derived as ``time + horizon``.
        horizon: Forecast horizon in ``freq`` units, used when ``target_time_col`` is
            absent.
        scale_cols: Columns to standardise per fold. ``None`` scales nothing.
        observed_col: Boolean column; ``True`` rows are excluded from evaluation
            (e.g. imputed labels). Training is unaffected.
        baseline_col: Integer-coded per-row baseline prediction (e.g. persistence) to
            compare against fold by fold.
        labels: Class labels in order; inferred when omitted.

    Returns:
        ``folds`` (per-fold scores, support counts and dates) and ``aggregate``
        (mean ± std, per-fold deltas, folds-won, Wilcoxon p-values).

    Example:
        >>> import numpy as np, pandas as pd
        >>> from sklearn.dummy import DummyClassifier
        >>> idx = pd.date_range("2022-01-01", periods=900, freq="h")
        >>> rng = np.random.default_rng(0)
        >>> d = pd.DataFrame({"x": rng.normal(size=len(idx)),
        ...                   "y": rng.integers(0, 3, len(idx))}, index=idx)
        >>> d["naive"] = d["y"].shift(1).bfill().astype(int)   # a baseline to beat
        >>> out = rolling_origin_cv(d, lambda: DummyClassifier(strategy="most_frequent"),
        ...                         n_folds=3, embargo_hours=1, horizon=1,
        ...                         feature_cols=["x"], baseline_col="naive")
        >>> len(out["folds"])
        3
        >>> out["aggregate"]["macro_f1"]["wins"] in (0, 1, 2, 3)
        True
        >>> out["aggregate"]["resolution"]["can_reach_alpha_05"]   # 3 folds cannot
        False
    """
    d = df.copy()
    if time_col is None:
        if not isinstance(d.index, pd.DatetimeIndex):
            raise TypeError("index must be a DatetimeIndex, or pass time_col=")
        d = d.reset_index(names="__t")
        time_col = "__t"
    d[time_col] = pd.to_datetime(d[time_col])

    if target_time_col is None:
        target_time_col = "__target_t"
        d[target_time_col] = d[time_col] + pd.tseries.frequencies.to_offset(freq) * horizon
    else:
        d[target_time_col] = pd.to_datetime(d[target_time_col])

    reserved = {time_col, target_time_col, target_col, observed_col, baseline_col}
    if feature_cols is None:
        feature_cols = [c for c in d.columns if c not in reserved and c is not None]
    if labels is None:
        labels = sorted(pd.unique(d[target_col]))

    folds_meta = make_folds(d[time_col].to_numpy(), n_folds, initial_fraction)
    embargo = pd.Timedelta(hours=embargo_hours)
    out_folds = []

    for meta in folds_meta:
        tr = d[d[target_time_col] < meta["cutoff"]]
        ev_raw = d[(d[time_col] >= meta["cutoff"] + embargo)
                   & (d[time_col] <= meta["eval_end"])]
        ev = ev_raw[ev_raw[observed_col].astype(bool)] if observed_col else ev_raw
        if tr.empty:
            raise ValueError(
                f"fold {meta['fold']} has no training rows before {meta['cutoff']}; "
                f"raise initial_fraction or reduce n_folds")
        if ev.empty:
            # Distinguish the two causes: an empty block is a fold-layout problem,
            # while a block emptied by the filter is a data problem the caller needs
            # to hear about in those terms.
            cause = (f"all {len(ev_raw):,} rows were excluded by observed_col="
                     f"{observed_col!r}" if observed_col and len(ev_raw)
                     else f"the block is empty after a {embargo_hours}h embargo")
            raise ValueError(f"fold {meta['fold']} has no evaluable rows: {cause}")

        Xtr = tr[feature_cols].to_numpy(dtype=float)
        Xev = ev[feature_cols].to_numpy(dtype=float)
        if scale_cols:
            pos = [feature_cols.index(c) for c in scale_cols]
            sc = StandardScaler().fit(Xtr[:, pos])
            Xtr[:, pos] = sc.transform(Xtr[:, pos])
            Xev[:, pos] = sc.transform(Xev[:, pos])

        model = model_fn()
        model.fit(Xtr.astype(np.float32), tr[target_col].to_numpy())
        pred = model.predict(Xev.astype(np.float32))
        y = ev[target_col].to_numpy()

        entry = {"fold": meta["fold"], "cutoff": str(meta["cutoff"]),
                 "eval_start": str(ev[time_col].min()),
                 "eval_end": str(ev[time_col].max()),
                 "months": sorted({int(m) for m in ev[time_col].dt.month.unique()}),
                 "n_train": int(len(tr)), "n_eval": int(len(ev)),
                 "support": {l: int((y == i).sum()) for i, l in enumerate(labels)},
                 "model": _score(y, pred, labels)}
        if min_class_support is not None:
            entry["unevaluable"] = [l for l, n in entry["support"].items()
                                    if n < min_class_support]
        if baseline_col is not None:
            entry["baseline"] = _score(y, ev[baseline_col].to_numpy(), labels)
        out_folds.append(entry)
        if verbose:
            b = (f" vs {entry['baseline']['macro_f1']:.4f}"
                 if baseline_col is not None else "")
            print(f"  fold {meta['fold']}: train {len(tr):,} eval {len(ev):,} | "
                  f"macro-F1 {entry['model']['macro_f1']:.4f}{b}")

    agg = (aggregate_folds(out_folds, labels, min_class_support)
           if baseline_col is not None else {})
    return {"folds": out_folds, "aggregate": agg, "n_folds": n_folds,
            "embargo_hours": embargo_hours, "initial_fraction": initial_fraction,
            "labels": list(labels)}


def aggregate_folds(folds: list[dict], labels: list,
                    min_class_support: int | None = None) -> dict:
    """Mean ± std, folds-won and a signed-rank test on the per-fold deltas.

    The unit of analysis is the **fold**, not the sample. A percentile bootstrap is
    wrong here — with a handful of folds it can only resample the same handful of
    numbers — so the paired comparison is a Wilcoxon signed-rank test.

    The test has a hard floor: with *n* paired folds the smallest reachable two-sided
    p is ``2**(1-n)``. At 5 folds that is 0.0625, so **no five-fold result can reach
    α = 0.05 however clean it is**. ``resolution`` reports the floor so a reader is
    never invited to read a null as evidence of absence.
    """
    metrics = ["macro_f1"] + [f"f1_{l}" for l in labels]
    out = {}
    for m in metrics:
        a = np.array([f["model"][m] for f in folds], dtype=float)
        b = np.array([f["baseline"][m] for f in folds], dtype=float)
        if min_class_support is not None and m.startswith("f1_"):
            lab = m[3:]
            keep = np.array([f["support"].get(lab, 0) >= min_class_support
                             for f in folds])
        else:
            keep = np.ones(len(a), dtype=bool)
        d = (a - b)[keep]
        if len(d) == 0:
            out[m] = {"evaluable_folds": 0, "note": "no fold met min_class_support"}
            continue
        try:
            p2 = float(wilcoxon(d).pvalue)
            p1 = float(wilcoxon(d, alternative="greater").pvalue)
        except ValueError:
            p2 = p1 = 1.0
        out[m] = {
            "model_mean": float(a[keep].mean()), "model_std": float(a[keep].std(ddof=1))
            if keep.sum() > 1 else 0.0,
            "baseline_mean": float(b[keep].mean()),
            "mean_delta": float(d.mean()),
            "std_delta": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "wins": int((d > 0).sum()), "n_folds": int(keep.sum()),
            "evaluable_folds": int(keep.sum()),
            "p_two_sided": p2, "p_one_sided": p1,
            "sig_two_sided": bool(p2 < 0.05), "sig_one_sided": bool(p1 < 0.05),
            "per_fold_delta": [float(x) for x in d]}
    n = len(folds)
    out["resolution"] = {"n_pairs": n,
                         "min_p_two_sided": float(2 ** (1 - n)),
                         "min_p_one_sided": float(2 ** -n),
                         "can_reach_alpha_05": bool(2 ** (1 - n) < 0.05)}
    return out
