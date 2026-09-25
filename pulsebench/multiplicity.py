"""Multiple-comparisons correction for a family of baseline comparisons."""

from __future__ import annotations

__all__ = ["bonferroni_report", "format_markdown"]


def bonferroni_report(comparisons, alpha: float = 0.05, *,
                      n_resamples: int | None = None,
                      method: str = "bonferroni") -> dict:
    """Bonferroni-correct a family of comparisons and format the result.

    A study that reports one model against a baseline is making one comparison. A
    study that reports eleven is making eleven, and the family-wise error rate is not
    what the individual p-values say. This applies ``alpha / k`` and reports which
    claims survive — including, explicitly, the ones that were significant before
    correction and are not after, since those are the ones a reader needs flagged.

    ``n_resamples`` matters when the p-values came from a bootstrap or permutation
    test: such a test cannot resolve a two-sided p below ``2 / n_resamples``. Values at
    that floor are reported as ``< floor`` rather than as an exact number that the
    procedure could not have produced. This is independent of the correction: Holm does
    not let a signed-rank test over five folds produce a p below ``2 ** (1 - 5)``.

    ``method`` selects the correction. ``"bonferroni"`` (the default) compares every
    p against ``alpha / k``. ``"holm"`` applies the step-down procedure: sort ascending
    and compare the i-th against ``alpha / (k - i)``, stopping at the first failure and
    rejecting nothing after it. Holm controls the same family-wise error rate and is
    uniformly at least as powerful, so it never rejects fewer hypotheses on the same
    input. Bonferroni remains the default because the numbers already published in
    this project were computed with it.

    Args:
        comparisons: ``{name: {"p": float, "delta": float}}`` or a list of dicts each
            carrying ``name``, ``p`` and ``delta``.
        alpha: Family-wise error rate.
        n_resamples: Resamples behind the p-values, if resampling-based.
        method: ``"bonferroni"`` or ``"holm"``.

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

        Holm is uniformly at least as powerful. Here it rejects one hypothesis that
        plain Bonferroni does not:

        >>> fam = {"A": {"p": 0.004, "delta": 0.05}, "B": {"p": 0.02, "delta": 0.03},
        ...        "C": {"p": 0.9, "delta": 0.001}}
        >>> bonferroni_report(fam)["survivors"]
        ['A']
        >>> bonferroni_report(fam, method="holm")["survivors"]
        ['A', 'B']
    """
    method = str(method).lower()
    if method not in ("bonferroni", "holm"):
        raise ValueError(f"method must be 'bonferroni' or 'holm', got {method!r}")
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
    corrected = alpha / k          # the Bonferroni threshold, reported either way
    floor = (2.0 / n_resamples) if n_resamples else None

    ordered = sorted(items, key=lambda x: x["p"])
    # Holm steps down: the i-th smallest p is compared against alpha / (k - i), and
    # the first failure stops the procedure -- nothing after it is rejected, even if
    # its own threshold would have passed. That monotonicity is what preserves the
    # family-wise error rate.
    holm_ok, still_rejecting = [], True
    for i, it in enumerate(ordered):
        thresh = alpha / (k - i)
        if still_rejecting and float(it["p"]) < thresh:
            holm_ok.append(True)
        else:
            still_rejecting = False
            holm_ok.append(False)

    rows = []
    for i, it in enumerate(ordered):
        p = float(it["p"])
        at_floor = bool(floor is not None and p <= floor)
        thresh = alpha / (k - i) if method == "holm" else corrected
        survives = holm_ok[i] if method == "holm" else bool(p < corrected)
        rows.append({
            "name": it["name"],
            "delta": it.get("delta"),
            "p": p,
            "p_display": (f"<{floor:.4f}" if at_floor else f"{p:.4f}"),
            "at_resolution_floor": at_floor,
            "significant_uncorrected": bool(p < alpha),
            "threshold": thresh,
            "survives": bool(survives),
            "lost_to_correction": bool(p < alpha and not survives),
            "direction": ("better" if (it.get("delta") or 0) > 0 else
                          "worse" if (it.get("delta") or 0) < 0 else "unchanged"),
        })

    survivors = [r["name"] for r in rows if r["survives"]]
    lost = [r["name"] for r in rows if r["lost_to_correction"]]
    threshold_text = (f"alpha/k = {corrected:.4f}" if method == "bonferroni"
                      else f"the Holm step-down thresholds "
                           f"(alpha/{k} to alpha/1)")
    return {
        "k": k, "alpha": alpha, "corrected_alpha": corrected,
        "method": method, "resolution_floor": floor, "rows": rows,
        "survivors": survivors, "lost": lost,
        "summary": (f"{len(survivors)} of {k} comparisons survive {threshold_text} "
                    f"under {method.capitalize()} correction"
                    + (f"; {len(lost)} were significant at {alpha} and are not after "
                       f"correction: {', '.join(lost)}" if lost else "")),
    }


def format_markdown(report: dict) -> str:
    """Render :func:`bonferroni_report` output as a markdown table.

    The header names the correction that was applied, and for Holm the per-row
    threshold is shown, since it differs by rank rather than being one number.

    Example:
        >>> r = bonferroni_report({"a": {"p": 0.001, "delta": 0.2}})
        >>> print(format_markdown(r).splitlines()[0])
        | Comparison | Delta | p | p < 0.0500 | Survives Bonferroni (p < 0.0500) |
        >>> h = bonferroni_report({"a": {"p": 0.001}, "b": {"p": 0.02}}, method="holm")
        >>> print(format_markdown(h).splitlines()[0])
        | Comparison | Delta | p | p < 0.0500 | Holm threshold | Survives Holm |
    """
    a, c = report["alpha"], report["corrected_alpha"]
    holm = report.get("method", "bonferroni") == "holm"
    head = ["Comparison", "Delta", "p", f"p < {a:.4f}"]
    head += (["Holm threshold", "Survives Holm"] if holm
             else [f"Survives Bonferroni (p < {c:.4f})"])
    lines = ["| " + " | ".join(head) + " |",
             "| " + " | ".join(["---"] * len(head)) + " |"]
    for r in report["rows"]:
        d = "—" if r["delta"] is None else f"{r['delta']:+.4f}"
        cells = [str(r["name"]), d, r["p_display"],
                 "yes" if r["significant_uncorrected"] else "no"]
        if holm:
            cells.append(f"{r.get('threshold', c):.4f}")
        cells.append("**yes**" if r["survives"] else "no")
        lines.append("| " + " | ".join(cells) + " |")
    note = [""]
    if report.get("resolution_floor"):
        note.append(
            f"A resampling test cannot resolve a two-sided p below "
            f"{report['resolution_floor']:.4f}; values at that floor are shown as "
            f"`<{report['resolution_floor']:.4f}` rather than as an exact figure. "
            f"This bound is a property of the test and is unaffected by the choice "
            f"of correction.")
    return "\n".join(lines + note).rstrip()
