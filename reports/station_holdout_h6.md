# Station-held-out cross-validation — horizon 6 h

Rolling-origin cross-validation asks whether a model holds up **over time**. It says
nothing about **space**. A wearable is carried where the training data does not reach,
so the deployment question is whether a model trained on eleven Beijing monitoring
stations works at a twelfth it has never seen.

12 leave-one-station-out folds. Train on the other eleven, evaluate on the held-out
station, `RandomForest (class_weight=balanced)`, observed labels only. The scaler is fitted on the eleven
training stations alone, because fitting it on all twelve would leak the held-out
station's distribution into its own evaluation.

**Each station is compared against its own persistence floor, not a global one.** A
station whose air is stickier is easier to predict by doing nothing, and scoring it
against another station's baseline would flatter or punish it for geography rather
than for modelling.

---

## 1. Per-station results

| Held-out station | Observed n | Model macro-F1 | Its own floor | Difference | Model Haz F1 | Floor Haz F1 | Haz difference |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dongsi | 34,248 | 0.5479 | 0.4830 | +0.0649**\*** | 0.6543 | 0.5740 | +0.0802**\*** |
| Aotizhongxin | 34,061 | 0.5576 | 0.4969 | +0.0607**\*** | 0.6693 | 0.5972 | +0.0721**\*** |
| Wanshouxigong | 34,281 | 0.5527 | 0.4936 | +0.0591**\*** | 0.6797 | 0.6161 | +0.0636**\*** |
| Guanyuan | 34,361 | 0.5614 | 0.5029 | +0.0584**\*** | 0.6597 | 0.5870 | +0.0727**\*** |
| Nongzhanguan | 34,349 | 0.5418 | 0.4866 | +0.0552**\*** | 0.6356 | 0.5815 | +0.0541**\*** |
| Tiantan | 34,300 | 0.5472 | 0.4925 | +0.0546**\*** | 0.6482 | 0.5817 | +0.0665**\*** |
| Wanliu | 34,598 | 0.5604 | 0.5112 | +0.0492**\*** | 0.6634 | 0.6161 | +0.0473**\*** |
| Shunyi | 34,064 | 0.5131 | 0.4666 | +0.0464**\*** | 0.5963 | 0.5286 | +0.0677**\*** |
| Gucheng | 34,331 | 0.5403 | 0.5000 | +0.0404**\*** | 0.6476 | 0.5974 | +0.0502**\*** |
| Changping | 34,205 | 0.5227 | 0.5070 | +0.0157**\*** | 0.5626 | 0.5479 | +0.0147 |
| Huairou | 34,032 | 0.5231 | 0.5146 | +0.0084**\*** | 0.5468 | 0.5287 | +0.0181 |
| Dingling | 34,198 | 0.5427 | 0.5388 | +0.0039 | 0.6314 | 0.6230 | +0.0084 |

`*` marks a difference significant at 0.05 under a paired bootstrap
(1,000 resamples on identical rows).

## 2. Verdict

**Generalisation holds.** The model beats the held-out station's own persistence floor at every station. Generalisation to an unseen site holds.

- Beats its own floor on macro-F1 at **12 of 12** stations
  (significantly at 11).
- Beats its own floor on Hazardous F1 at **12 of 12**
  (significantly at 9).
- Mean difference **+0.0431** macro-F1
  (sd 0.0215, range +0.0039 to
  +0.0649).
- Wilcoxon signed-rank over the 12 station-level differences:
  two-sided p = **0.0005**. The floor for 12 stations is
  0.0005, so that value is interpretable rather
  than bounded.

**No station is a clear outlier.** No station's difference against its own floor sits more than two standard deviations from the mean, so the spread below is ordinary variation rather than one site behaving unlike the others.

## 3. Station profiles

Whether a station is hard is partly a property of its air, so the profiles are
reported alongside the results rather than left implicit.

| Station | Median PM2.5 | 95th pct | Category unchanged over h | Hazardous n |
| --- | --- | --- | --- | --- |
| Dongsi | 61.0 | 260.0 | 50.8% | 1,924 |
| Aotizhongxin | 60.0 | 248.0 | 51.7% | 1,650 |
| Gucheng | 60.0 | 248.0 | 52.1% | 1,641 |
| Wanshouxigong | 60.0 | 259.0 | 51.5% | 1,826 |
| Guanyuan | 59.0 | 244.0 | 52.8% | 1,560 |
| Nongzhanguan | 59.0 | 262.0 | 50.8% | 1,902 |
| Tiantan | 59.0 | 244.0 | 51.5% | 1,577 |
| Wanliu | 59.0 | 248.0 | 52.8% | 1,655 |
| Shunyi | 55.0 | 244.0 | 49.4% | 1,529 |
| Changping | 47.0 | 222.0 | 53.1% | 1,095 |
| Huairou | 47.0 | 215.0 | 54.3% | 972 |
| Dingling | 41.0 | 222.0 | 55.8% | 1,104 |

## 4. Why this is an easier test than rolling-origin CV

**These two results are not in conflict, and the difference between them is
instructive.** Rolling-origin CV found that no Beijing-trained model beats persistence
in a majority of chronological folds. This section may find the opposite. The reason is
that holding out a *station* leaves the *time axis intact*: the model trains on eleven
stations across the whole 2013–2017 record and is evaluated on a twelfth over the same
period. It has therefore already seen every pollution episode, every winter and every
synoptic event in the evaluation window — just measured somewhere else in the same
city.

That is a genuinely easier problem than forecasting a period it has never seen. Beijing
stations are tens of kilometres apart in one airshed, and a haze episode arrives at all
of them; a model that has learned what such an episode looks like at eleven sites is
not being asked to extrapolate when it meets the twelfth.

**So this section bounds spatial transfer within a shared period, not deployment.** A
device carried somewhere new, forecasting a time nobody has seen, faces both problems
at once. The rolling-origin result is the binding one for that case, and nothing here
softens it.

A stricter version of this test would hold out a station *and* the later part of the
record simultaneously. That was not run, and the claim below is limited accordingly.

## 5. What this adds to the thesis

This is a different axis of generalisation from the rolling-origin result, and it is
worth stating which is which. Rolling-origin CV showed that no Beijing-trained model
beats persistence in a majority of chronological folds — a statement about *time*.
This section holds the time axis fixed and varies *place*.

The two together bound the claim a deployed model can make. A model that generalised
across stations but not across time would be a seasonal artefact; one that generalised
across time but not stations would be site-specific. The persistence floor is computed
per station here for the same reason it is computed per dataset elsewhere: a baseline
borrowed from somewhere else is not a baseline.

Regenerate with `python -m src.models.station_holdout_cv`.
