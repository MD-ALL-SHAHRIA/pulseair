# Contributing

This repository has two halves with different expectations.

**`src/` is thesis code.** It reproduces a specific set of published results, so its
numbers are effectively frozen: `reports/` and `reports/metrics/*.json` are committed
precisely so a reader can check any claim without re-running anything. Changes that alter
those numbers are not bug fixes to be merged quietly — if you find a genuine error, please
open an issue with the reproduction rather than a silent PR, so the affected reports can be
regenerated and the change recorded honestly.

**`pulsebench/` is built for reuse, and contributions there are welcome.** It depends on
nothing in `src/`, works on any dataframe with a datetime axis and a categorical target, and
is the part of this project most likely to be useful to someone else. Issues and pull
requests against it — new baselines, better sparse-class handling, other correction methods,
bug reports from your own data — are the ones we most want.

---

## Reporting a bug

Open an issue with:

- what you ran (the exact `python -m ...` command),
- what happened, including the full traceback,
- your Python version and OS, and whether you installed from `requirements.txt` or
  `requirements.lock.txt`.

For a numerical discrepancy, include the metrics JSON you got and the one in `reports/metrics/`
that it disagrees with.

## Pull requests

1. Work from a branch, not `main`.
2. Keep the change focused — one concern per PR.
3. Run the tests:
   ```bash
   pytest pulsebench/tests/ -q
   pytest src/preprocessing/test_pipeline.py -q
   ```
   `pulsebench/tests/test_retrofit_phase9.py` pins Phase 9's pre-extraction numbers. If it
   fails, the change altered evaluation behaviour — that may be correct, but say so
   explicitly in the PR rather than updating the fixture in passing.
4. New `pulsebench` functions need a docstring with a runnable doctest; the suite is checked
   with `python -m doctest pulsebench/*.py`.
5. Say in the PR description whether any committed report or metrics file needs regenerating.

## Style

- Python 3.11, type hints on public functions, `from __future__ import annotations`.
- Match the surrounding code: this codebase explains *why* in comments and keeps the *what*
  in the code. Comments that restate the line above get dropped in review.
- Hyperparameters and thresholds belong in `configs/default.yaml`, not inline.

## Evaluation conventions

If your change touches how anything is measured, these are the house rules, and a PR that
breaks one should say why:

- **Report the persistence floor next to every headline metric.** A macro-F1 alone does not
  establish that a model works.
- **Select on validation; report on test.** Even when both splits agree — establishing that
  they agree requires looking at test, which is the thing being avoided.
- **Splits are chronological, never random.** Random splits leak future information through
  seasonal and diurnal structure.
- **Compute selection metrics on observed rows only** (`is_imputed_pm25 == False`).
  Forward-filled labels inflate the rare classes.
- **A gain in an aggregate metric that comes with a significant loss on a protected class is
  a disqualification, not a result.** See `pulsebench.advisory_disqualification`.
- **Correct for multiple comparisons** across a family of baseline tests, and state the
  resampling resolution floor (`2^(1-n)` for a signed-rank test over *n* folds) when a
  *p*-value is near it.
- **Never remove or soften a negative result.** Several of this project's most useful
  findings are failures, and they are load-bearing.

## Secrets

`.env` is gitignored and must stay that way. Keys go in `.env`; only their variable names
belong in `.env.example`. Never commit a key, and never print one in report output or a
traceback.

## License

Contributions are accepted under the MIT License, the same terms as the rest of the
repository — see [`LICENSE`](LICENSE).
