# Selective prediction — Beijing, horizon 6 h

Phase 6 turned point predictions into conformal prediction **sets**. A set with one or
two categories is a usable forecast; a set with four is barely a forecast. That
suggests an operating mode a device could actually adopt — answer when the set is
small, say nothing when it is not — and this section measures what that buys.

**Report-only. Nothing is retrained.** The saved model and the committed Mondrian
thresholds are applied to the test split exactly as Phase 6 applied them
(`conformal_h6.pkl Mondrian thresholds + baseline_h6.pkl`), and the predictions are partitioned by set size. "Confident" means
the conformal set holds at most **2** of the 6 categories.

---

## 1. Confident against full

| Subset | Share of predictions | n | Accuracy | Macro-F1 | Very unhealthy F1 | Hazardous F1 | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **All predictions** | 100% | 61,530 | 0.5793 | 0.5173 | 0.5265 | 0.6098 | 0.8865 |
| Confident (set ≤ 2) | 47.4% | 29,191 | 0.6289 | 0.5160 | 0.5581 | 0.6473 | 0.8563 |
| Abstained (set > 2) | 52.6% | 32,339 | 0.5346 | 0.4199 | 0.4474 | 0.3737 | 0.9139 |

Coverage is the share of rows whose conformal set contained the true category.
Accuracy and macro-F1 are computed on the **point** prediction the device would show.

## 2. By set size

| Set size | Share | n | Accuracy | Macro-F1 | Coverage |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.1% | 1,299 | 0.7575 | 0.4193 | 0.7575 |
| 2 | 45.3% | 27,892 | 0.6229 | 0.5075 | 0.8609 |
| 3 | 43.0% | 26,442 | 0.5474 | 0.4382 | 0.9069 |
| 4 | 8.6% | 5,311 | 0.4724 | 0.2836 | 0.9441 |
| 5 | 0.9% | 566 | 0.5265 | 0.1435 | 0.9541 |
| 6 | 0.0% | 20 | 0.3000 | 0.0769 | 1.0000 |

## 3. What this establishes

- **47.4% of predictions qualify as confident** at a set size
  of 2 or fewer. The remaining 52.6% would be withheld.
- On that subset accuracy moves **+0.0496** and macro-F1 **-0.0013** against
  the full test split.

**The Hazardous class is where the trade bites.** Restricting to confident predictions leaves 3,038 Hazardous samples of 3,687, and Hazardous F1 moves from 0.6098 to 0.6473. A device that abstains on the readings it finds hard is abstaining disproportionately on the readings that matter, which is the opposite of what an advisory is for.

**This is a trade, not a gain.** Abstention improves the reported metric on the rows
that remain by removing the rows the model finds hard; it does not make those rows
easier. The relevant question for a wearable is whether a forecast withheld is better
than a forecast hedged, and the answer depends on what the wearer does with silence.
The advisory layer in Phase 6 takes the other option — it reports the ambiguity rather
than suppressing the reading — and this analysis quantifies what the alternative would
have cost.

Regenerate with `python -m src.models.selective_prediction`.
