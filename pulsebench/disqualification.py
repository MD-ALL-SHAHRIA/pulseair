"""The protected-class disqualification rule."""

from __future__ import annotations

__all__ = ["advisory_disqualification"]


def advisory_disqualification(results_by_class: dict, protected_classes,
                              practical_threshold: float = 0.01, *,
                              aggregate_delta: float | None = None,
                              aggregate_significant: bool | None = None) -> dict:
    """Reject an intervention that helps on average by hurting a class that matters.

    Aggregate metrics average classes with equal weight. Safety-critical applications
    do not: a missed hazardous-air warning and a missed moderate-air warning are not
    equally costly, and macro-F1 cannot tell them apart.

    This encodes the rule that settled several comparisons in the project this package
    came from. Both CTGAN augmentation and SMOTE produced a statistically significant
    macro-F1 gain while significantly degrading *both* safety-critical classes; both
    were rejected on that basis, and a one-word class-weight change that left those
    classes alone was preferred.

    Precedence is deliberate: **a significant regression on a protected class
    disqualifies, whatever the aggregate did.** Only then is the aggregate gain
    checked against ``practical_threshold``, because with large test sets a bootstrap
    resolves differences far below anything that changes a decision.

    Args:
        results_by_class: ``{class: {"delta": float, "significant": bool}}`` —
            per-class change versus the incumbent, with significance already decided
            (bootstrap, permutation, whatever the study used).
        protected_classes: Names that must not regress.
        practical_threshold: Minimum aggregate gain worth acting on.
        aggregate_delta: Aggregate metric change; enables the second test.
        aggregate_significant: Whether that change was significant.

    Returns:
        ``disqualified``, ``verdict`` (``"disqualified"`` / ``"accepted"`` /
        ``"insufficient"`` / ``"no_aggregate_test"``), ``harmed``, ``helped`` and a
        human-readable ``reason``.

    Example:
        >>> res = {"Hazardous": {"delta": -0.019, "significant": True},
        ...        "Very unhealthy": {"delta": -0.029, "significant": True},
        ...        "Moderate": {"delta": 0.09, "significant": True}}
        >>> out = advisory_disqualification(res, ["Hazardous", "Very unhealthy"],
        ...                                 aggregate_delta=0.0039,
        ...                                 aggregate_significant=True)
        >>> out["disqualified"]
        True
        >>> out["harmed"]
        ['Hazardous', 'Very unhealthy']
    """
    protected = list(protected_classes)
    missing = [c for c in protected if c not in results_by_class]
    if missing:
        raise KeyError(f"protected classes absent from results: {missing}")

    harmed = [c for c in protected
              if results_by_class[c].get("significant") and results_by_class[c]["delta"] < 0]
    helped = [c for c in protected
              if results_by_class[c].get("significant") and results_by_class[c]["delta"] > 0]

    if harmed:
        detail = "; ".join(f"{c} {results_by_class[c]['delta']:+.4f}" for c in harmed)
        return {"disqualified": True, "verdict": "disqualified",
                "harmed": harmed, "helped": helped,
                "protected_classes": protected,
                "practical_threshold": practical_threshold,
                "reason": (f"significantly degrades {detail} — a protected class. "
                           f"Aggregate performance does not override this.")}

    if aggregate_delta is None or aggregate_significant is None:
        return {"disqualified": False, "verdict": "no_aggregate_test",
                "harmed": [], "helped": helped, "protected_classes": protected,
                "practical_threshold": practical_threshold,
                "reason": ("no protected class regressed; no aggregate comparison was "
                           "supplied, so acceptance is not decided")}

    if aggregate_significant and aggregate_delta >= practical_threshold:
        return {"disqualified": False, "verdict": "accepted",
                "harmed": [], "helped": helped, "protected_classes": protected,
                "practical_threshold": practical_threshold,
                "reason": (f"aggregate {aggregate_delta:+.4f} is significant and at or "
                           f"above the {practical_threshold} practical threshold, with "
                           f"no protected-class regression")}

    why = ("not significant" if not aggregate_significant
           else f"below the {practical_threshold} practical threshold")
    return {"disqualified": False, "verdict": "insufficient",
            "harmed": [], "helped": helped, "protected_classes": protected,
            "practical_threshold": practical_threshold,
            "reason": (f"no protected class regressed, but the aggregate gain "
                       f"({aggregate_delta:+.4f}) is {why}")}
