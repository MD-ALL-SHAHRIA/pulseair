"""Multiple-comparisons correction for a family of baseline comparisons."""

from __future__ import annotations

__all__ = ["bonferroni_report"]


def bonferroni_report(comparisons, alpha: float = 0.05, *,
                      n_resamples: int | None = None) -> dict:
    """Bonferroni-correct a family of comparisons and format the result.

    A study that reports one model against a baseline is making one comparison. A
    study that reports eleven is making eleven, and the family-wise error rate is not
    what the individual p-values say. This applies ``alpha / k`` and reports which
    claims survive — including, explicitly, the ones that were significant before
    correction and are not after, since those are the ones a reader needs flagged.

    ``n_resamples`` matters when the p-values came from a bootstrap or permutation
    test: such a test cannot resolve a two-sided p below ``2 / n_resamples``. Values at
    that floor are reported as ``< floor`` rather than as an exact number that the
    procedure could not have produced.

    Args:
        comparisons: ``{name: {"p": float, "delta": float}}`` or a list of dicts each
            carrying ``name``, ``p`` and ``delta``.
        alpha: Family-wise error rate.
        n_resamples: Resamples behind the p-values, if resampling-based.

    Returns:
        ``k``, ``alpha``, ``corrected_alpha``, ``rows`` (sorted by p, each flagged
        ``survives`` / ``lost_to_correction``), ``survivors``, ``lost`` and
        ``resolution_floor``.

    Example:
        >>> out = bonferroni_report({"model A": {"p": 0.002, "delta": 0.05},
        ...                          "model B": {"p": 0.038, "delta": 0.01}},
        ...                         n_resamples=1000)
        >>> out["corrected_alpha"]
        0.025
        >>> out["survivors"]
        ['model A']
        >>> out["lost"]
        ['model B']
    """
    if isinstance(comparisons, dict):
        items = [{"name": k, **v} for k, v in comparisons.items()]
    else:
        items = [dict(c) for c in comparisons]
    if not items:
        raise ValueError("no comparisons supplied")
    for it in items:
        if "p" not in it or "name" not in it:
            raise KeyError("each comparison needs 'name' and 'p'")

    k = len(items)
    corrected = alpha / k
    floor = (2.0 / n_resamples) if n_resamples else None

    rows = []
    for it in sorted(items, key=lambda x: x["p"]):
        p = float(it["p"])
        at_floor = bool(floor is not None and p <= floor)
        rows.append({
            "name": it["name"],
            "delta": it.get("delta"),
            "p": p,
            "p_display": (f"<{floor:.4f}" if at_floor else f"{p:.4f}"),
            "at_resolution_floor": at_floor,
            "significant_uncorrected": bool(p < alpha),
            "survives": bool(p < corrected),
            "lost_to_correction": bool(alpha > p >= corrected),
            "direction": ("better" if (it.get("delta") or 0) > 0 else
                          "worse" if (it.get("delta") or 0) < 0 else "unchanged"),
        })

    survivors = [r["name"] for r in rows if r["survives"]]
    lost = [r["name"] for r in rows if r["lost_to_correction"]]
    return {
        "k": k, "alpha": alpha, "corrected_alpha": corrected,
        "resolution_floor": floor, "rows": rows,
        "survivors": survivors, "lost": lost,
        "summary": (f"{len(survivors)} of {k} comparisons survive alpha/k = "
                    f"{corrected:.4f}"
                    + (f"; {len(lost)} were significant at {alpha} and are not after "
                       f"correction: {', '.join(lost)}" if lost else "")),
    }


def format_markdown(report: dict) -> str:
    """Render :func:`bonferroni_report` output as a markdown table.

    Example:
        >>> r = bonferroni_report({"a": {"p": 0.001, "delta": 0.2}})
        >>> print(format_markdown(r).splitlines()[0])
        | Comparison | Delta | p | p < 0.0500 | p < 0.0500 (corrected) |
    """
    a, c = report["alpha"], report["corrected_alpha"]
    head = ["Comparison", "Delta", "p", f"p < {a:.4f}",
            f"p < {c:.4f} (corrected)"]
    lines = ["| " + " | ".join(head) + " |",
             "| " + " | ".join(["---"] * len(head)) + " |"]
    for r in report["rows"]:
        d = "—" if r["delta"] is None else f"{r['delta']:+.4f}"
        lines.append("| " + " | ".join([
            str(r["name"]), d, r["p_display"],
            "yes" if r["significant_uncorrected"] else "no",
            "**yes**" if r["survives"] else "no"]) + " |")
    return "\n".join(lines)
