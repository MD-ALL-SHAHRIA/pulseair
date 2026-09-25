# Selective prediction — Bangladesh (deployed model), horizon 6 h

Phase 6 turned point predictions into conformal prediction **sets**. A set with one or
two categories is a usable forecast; a set with four is barely a forecast. That
suggests an operating mode a device could actually adopt — answer when the set is
small, say nothing when it is not — and this section measures what that buys.

**Report-only. Nothing is retrained.** The saved model and the committed Mondrian
thresholds are applied to the test split exactly as Phase 6 applied them
(`bangladesh_rf_h6.pkl + conformal_thresholds_h6_bd.json`), and the predictions are partitioned by set size. "Confident" means
the conformal set holds at most **2** of the 6 categories.

---

## 1. Confident against full

| Subset | Share of predictions | n | Accuracy | Macro-F1 | Very unhealthy F1 | Hazardous F1 | Coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **All predictions** | 100% | 17,356 | 0.6703 | 0.4075 | 0.0000 | 0.0000 | 0.9175 |
| Confident (set ≤ 2) | 25.0% | 4,345 | 0.8608 | 0.4423 | 0.0000 | 0.0000 | 0.8608 |
| Abstained (set > 2) | 75.0% | 13,011 | 0.6067 | 0.3790 | 0.0000 | 0.0000 | 0.9364 |

Coverage is the share of rows whose conformal set contained the true category.
Accuracy and macro-F1 are computed on the **point** prediction the device would show.

## 2. By set size

| Set size | Share | n | Accuracy | Macro-F1 | Coverage |
| --- | --- | --- | --- | --- | --- |
| 2 | 25.0% | 4,345 | 0.8608 | 0.4423 | 0.8608 |
| 3 | 71.2% | 12,351 | 0.6136 | 0.3864 | 0.9387 |
| 4 | 3.8% | 660 | 0.4773 | 0.2407 | 0.8939 |

## 3. What this establishes

- **25.0% of predictions qualify as confident** at a set size
  of 2 or fewer. The remaining 75.0% would be withheld.
- On that subset accuracy moves **+0.1904** and macro-F1 **+0.0348** against
  the full test split.

**The Hazardous class is where the trade bites.** Restricting to confident predictions leaves 0 Hazardous samples of 0, and Hazardous F1 moves from 0.0000 to 0.0000. A device that abstains on the readings it finds hard is abstaining disproportionately on the readings that matter, which is the opposite of what an advisory is for.

**This is a trade, not a gain.** Abstention improves the reported metric on the rows
that remain by removing the rows the model finds hard; it does not make those rows
easier. The relevant question for a wearable is whether a forecast withheld is better
than a forecast hedged, and the answer depends on what the wearer does with silence.
The advisory layer in Phase 6 takes the other option — it reports the ambiguity rather
than suppressing the reading — and this analysis quantifies what the alternative would
have cost.

Regenerate with `python -m src.models.selective_prediction`.
