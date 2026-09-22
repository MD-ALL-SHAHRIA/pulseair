## What this changes

<!-- One or two sentences. One concern per PR, please. -->

## Why

<!-- The problem it solves. Link an issue if there is one. -->

## Checklist

- [ ] `pytest pulsebench/tests/ -q` passes
- [ ] `pytest src/preprocessing/test_pipeline.py -q` passes
- [ ] New `pulsebench` functions have a docstring with a runnable doctest
- [ ] Hyperparameters and thresholds went into `configs/default.yaml`, not inline

## Does this change any reported number?

- [ ] No — behaviour-preserving
- [ ] Yes, and I've said which reports need regenerating, below

<!--
If test_retrofit_phase9.py fails, this changed evaluation behaviour. That may be
correct — but say so explicitly here rather than updating the fixture in passing.
That pin exists to make silent changes loud.
-->
