"""pulsebench — evaluation protocol for seasonal, imbalanced time-series classification.

Extracted from a thesis on wearable air-quality forecasting, where the headline finding
was methodological rather than architectural: most of the reported gains in that
literature do not survive being measured against a zero-parameter baseline, across
rotating seasonal folds, with protected classes checked separately.

Four pieces, usable on any dataset with a datetime axis and a categorical target:

* :func:`persistence_floor` — the baseline a model must beat to have demonstrated
  anything.
* :func:`rolling_origin_cv` — expanding-window CV with embargo, per-fold scaling, and
  sparse-class handling.
* :func:`advisory_disqualification` — reject an intervention that helps on average by
  hurting a safety-critical class.
* :func:`bonferroni_report` — family-wise correction over a set of baseline
  comparisons, with the resampling resolution floor made explicit.

See ``pulsebench/README.md`` for the reasoning behind each.
"""

from pulsebench.cv import aggregate_folds, make_folds, rolling_origin_cv
from pulsebench.disqualification import advisory_disqualification
from pulsebench.multiplicity import bonferroni_report, format_markdown
from pulsebench.persistence import persistence_floor, seasonal_naive_floor

__version__ = "0.1.0"
__all__ = ["persistence_floor", "seasonal_naive_floor", "rolling_origin_cv", "make_folds", "aggregate_folds",
           "advisory_disqualification", "bonferroni_report", "format_markdown",
           "__version__"]
