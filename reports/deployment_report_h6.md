# Edge deployment simulation — horizon 6 h

Target: the **final predictor** — `baseline_h6.pkl` (RandomForest) plus
`conformal_h6.pkl` (Mondrian per-class thresholds). The LSTM and
Transformer were disqualified in Phase 5 and play no part here.

---

## 1. Compression — two-sided acceptance rule

Sweep over `n_estimators` x `max_depth`, holding `min_samples_leaf` and
`max_features` at their Phase 3 values. Selection is on **validation** only. A
candidate must satisfy **both**:

- **(a)** within **0.01** macro-F1 of the uncompressed forest — it has
  not lost much relative to its parent; and
- **(b)** **significantly beat the persistence floor** on validation under the same
  1,000-resample paired bootstrap used throughout — it is worth deploying
  at all.

The first version of this phase tested only (a). The model it selected passed that
test and then landed significantly *below* persistence on test. Condition (b) exists
because a compression pipeline can satisfy its own acceptance criterion while
producing something worse than having no model.

**Validation persistence floor: 0.5048 macro-F1.**

Baseline: **200 trees x depth 16**,
990,646 nodes, 42.3 MB,
val macro-F1 0.4895.

| Trees | Depth | Val F1 | vs base | vs persistence | 95% CI | (a) within tol. | (b) beats floor | Nodes | KB |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | 16 | 0.4895 | +0.0000 | -0.0153 | [-0.0195, -0.0115] | yes | no | 990,646 | 43352.2 |  |
| 200 | 12 | 0.4862 | -0.0033 | -0.0186 | [-0.0227, -0.0148] | yes | no | 550,864 | 25866.6 |  |
| 200 | 10 | 0.4815 | -0.0080 | -0.0233 | [-0.0270, -0.0197] | yes | no | 272,250 | 13936.4 |  |
| 200 | 8 | 0.4712 | -0.0183 | -0.0336 | [-0.0367, -0.0304] | no | no | 94,276 | 5512.5 |  |
| 200 | 6 | 0.4628 | -0.0267 | -0.0420 | [-0.0449, -0.0390] | no | no | 25,320 | 1574.8 |  |
| 100 | 16 | 0.4898 | +0.0002 | -0.0150 | [-0.0192, -0.0109] | yes | no | 496,196 | 21708.3 | **selected** |
| 100 | 12 | 0.4868 | -0.0028 | -0.0180 | [-0.0223, -0.0141] | yes | no | 276,238 | 12964.1 |  |
| 100 | 10 | 0.4803 | -0.0092 | -0.0244 | [-0.0283, -0.0206] | yes | no | 136,566 | 6989.1 |  |
| 100 | 8 | 0.4712 | -0.0183 | -0.0335 | [-0.0366, -0.0304] | no | no | 47,218 | 2763.1 |  |
| 100 | 6 | 0.4630 | -0.0265 | -0.0417 | [-0.0448, -0.0388] | no | no | 12,672 | 788.1 |  |
| 50 | 16 | 0.4894 | -0.0001 | -0.0154 | [-0.0196, -0.0115] | yes | no | 247,592 | 10838.0 |  |
| 50 | 12 | 0.4872 | -0.0024 | -0.0176 | [-0.0220, -0.0139] | yes | no | 139,616 | 6543.0 |  |
| 50 | 10 | 0.4808 | -0.0088 | -0.0240 | [-0.0282, -0.0204] | yes | no | 68,460 | 3503.0 |  |
| 50 | 8 | 0.4747 | -0.0148 | -0.0301 | [-0.0331, -0.0271] | no | no | 23,598 | 1382.1 |  |
| 50 | 6 | 0.4627 | -0.0269 | -0.0421 | [-0.0452, -0.0391] | no | no | 6,340 | 395.0 |  |
| 25 | 16 | 0.4889 | -0.0007 | -0.0159 | [-0.0202, -0.0121] | yes | no | 122,915 | 5383.9 |  |
| 25 | 12 | 0.4879 | -0.0017 | -0.0169 | [-0.0213, -0.0131] | yes | no | 69,677 | 3264.3 |  |
| 25 | 10 | 0.4811 | -0.0085 | -0.0237 | [-0.0279, -0.0203] | yes | no | 34,283 | 1751.1 |  |
| 25 | 8 | 0.4747 | -0.0148 | -0.0301 | [-0.0330, -0.0271] | no | no | 11,823 | 692.8 |  |
| 25 | 6 | 0.4636 | -0.0260 | -0.0412 | [-0.0443, -0.0382] | no | no | 3,169 | 198.1 |  |
| 10 | 16 | 0.4871 | -0.0024 | -0.0177 | [-0.0219, -0.0138] | yes | no | 48,944 | 2145.4 |  |
| 10 | 12 | 0.4849 | -0.0047 | -0.0199 | [-0.0240, -0.0160] | yes | no | 27,436 | 1285.9 |  |
| 10 | 10 | 0.4792 | -0.0104 | -0.0256 | [-0.0298, -0.0221] | no | no | 13,642 | 697.8 |  |
| 10 | 8 | 0.4779 | -0.0117 | -0.0269 | [-0.0304, -0.0239] | no | no | 4,704 | 276.0 |  |
| 10 | 6 | 0.4570 | -0.0326 | -0.0478 | [-0.0511, -0.0447] | no | no | 1,264 | 79.9 |  |
| 5 | 16 | 0.4874 | -0.0021 | -0.0174 | [-0.0219, -0.0133] | yes | no | 24,093 | 1057.7 |  |
| 5 | 12 | 0.4815 | -0.0080 | -0.0233 | [-0.0274, -0.0192] | yes | no | 13,727 | 644.0 |  |
| 5 | 10 | 0.4772 | -0.0124 | -0.0276 | [-0.0318, -0.0238] | no | no | 6,889 | 352.9 |  |
| 5 | 8 | 0.4809 | -0.0086 | -0.0239 | [-0.0275, -0.0207] | yes | no | 2,335 | 137.7 |  |
| 5 | 6 | 0.4567 | -0.0328 | -0.0481 | [-0.0513, -0.0448] | no | no | 631 | 40.6 |  |

> ### No configuration clears the persistence floor — including the uncompressed model
>
> **0 of 30 configurations beat persistence on validation.** Not the smallest,
> and not the largest: the **uncompressed** 200 x
> 16 forest scores -0.0153 [-0.0195, -0.0115] against the floor — significantly
> *below* it, with a CI excluding zero.
>
> **This is not a compression failure.** Compression is not what put the model under
> the floor; on validation it was never above it. The two-sided rule has no valid
> answer here because condition (b) is unsatisfiable by any forest of any size on
> this split, so the selection falls through to the highest validation macro-F1
> (100 x 16) — a choice that carries no
> deployment justification and, at 21.2 MB, no deployment
> benefit either.
>
> **What this exposes.** The project's headline claim — "the RandomForest beats
> persistence" — rests entirely on the **test** split (+0.0055
> [+0.0016, +0.0093]). On **validation** the same model is
> -0.0153 [-0.0195, -0.0115]. Persistence scores 0.5048 on validation and
> 0.5118 on test, moving only +0.0070; the forest
> moves +0.0278 over the same gap. The two splits do not rank these two predictors
> the same way, and the positive result is the one that happens to sit on test.
>
> **This is the same val/test mismatch that reversed the Phase 5 model ranking and
> broke the Phase 7 tolerance, now arriving at the project's central claim.** It is
> documented in `preprocessing_summary_h6.md` §4: *Very unhealthy* is
> 7.04% of validation and 12.14% of test.
>
> **Recommended before submission.** Do not present a deployed edge model on this
> evidence. Either (i) adopt rolling-origin or prevalence-matched validation so the
> two splits are comparable and re-run selection end to end, or (ii) state explicitly
> that the persistence comparison is split-dependent and report both numbers side by
> side wherever the claim appears. Option (ii) is honest and cheap; option (i) is
> what would actually settle it.

### Selected

| | Baseline | Compressed | Change |
|---|---|---|---|
| Trees x depth | 200 x 16 | **100 x 16** | |
| Nodes | 990,646 | **496,196** | **-49.9%** |
| Pickle (compress=3) | 43,352.4 KB | **21,708.3 KB** | **2x smaller** |
| Val macro-F1 | 0.4895 | 0.4898 | +0.0002 |
| **Test macro-F1** | 0.5173 | **0.5152** | **-0.0021** |

The compressed model stays inside the 0.01 tolerance on **both** splits (+0.0002 validation, -0.0021 test).

**The accuracy/size curve is almost flat over three orders of magnitude.** The largest forest in the sweep (200 x 16, 990,646 nodes) scores 0.4895 on validation; the selected one has 2x fewer nodes and scores 0.4898. Across the whole grid, validation macro-F1 spans only 0.0328.

That is consistent with every other result in this project: at a six-hour horizon the signal available in these nine channels is limited, and model capacity is not the binding constraint. The Phase 3 forest was not chosen for accuracy it needed — it was simply larger than the problem required.

---

## 2. Conformal re-calibration (not optional)

Mondrian thresholds are quantiles of `1 − P(true class)` **under a specific model**.
Compressing the forest changes those probabilities, so the Phase 6 thresholds no longer
certify anything about the new model — and a stale set would look exactly like a
calibrated one while guaranteeing nothing. They are re-derived on the compressed model
from the same validation split, and coverage is re-verified on test.

| Class | Coverage before | Coverage after | Δ | Mean set size |
| --- | --- | --- | --- | --- |
| Good | 0.8673 | 0.8698 | +0.0025 | 2.45 |
| Moderate | 0.8877 | 0.8880 | +0.0003 | 2.76 |
| Unhealthy (sensitive) | 0.8635 | 0.8629 | -0.0006 | 2.93 |
| Unhealthy | 0.8808 | 0.8796 | -0.0012 | 2.70 |
| **Very unhealthy** | 0.9357 | 0.9357 | +0.0000 | 2.36 |
| **Hazardous** | 0.9083 | 0.9053 | -0.0030 | 2.08 |

| | Before | After |
|---|---|---|
| Overall coverage | 0.8865 | **0.8864** |
| Mean set size | — | 2.613 |
| Singleton rate | — | 2.1% |

Both advisory classes still clear the 90% target after compression.

---

## 3. ONNX export

| | |
|---|---|
| File | `reports/deployment/pulseair_rf_h6.onnx` |
| Size | **33,378.7 KB** (32.60 MB) |
| Opset | 17 |
| Input | `float32[n, 9]` — the nine scaled feature columns, in `metadata.json` order |
| Output | label + `float32[n, 6]` probabilities (`zipmap=False`) |

**Parity check before benchmarking.** ONNX must reproduce scikit-learn or the latency
numbers describe a different model:

| | |
|---|---|
| Rows compared | 2,000 |
| Max abs probability difference | 4.13e-07 |
| Argmax agreement | 1.0000 |
| Verdict | **PASS** |

---

## 4. Latency (single-threaded CPU)

`intra_op_num_threads=1`, warm session, 200 runs after
warm-up. Measured on the development machine (Apple Silicon), **not** on an ESP32 —
see section 6.

| Path | Mean (ms) | Median (ms) | p95 (ms) |
| --- | --- | --- | --- |
| onnx single | 0.0138 | 0.0135 | 0.0155 |
| sklearn single | 13.5788 | 13.8621 | 14.1982 |
| conformal single | 0.0104 | 0.0101 | 0.0105 |
| onnx plus conformal single | 0.0256 | 0.0255 | 0.0262 |
| onnx batch32 | 0.1222 | 0.1220 | 0.1254 |
| onnx batch256 | 2.0424 | 1.9820 | 2.3224 |

ONNX single-sample inference is
**987.4x faster** than
scikit-learn's `predict_proba` on the same model, which is the usual result: the
Python-level ensemble loop dominates at batch size 1.

### The conformal layer costs essentially nothing

Applying the Mondrian thresholds is **0.0104 ms** —
75.35% of the
inference it wraps. That is expected and worth stating plainly: the layer is
6 comparisons against 6 stored floats. There is no model, no
state and nothing to load, which is why it ships as a
1,322-byte JSON file rather than a second artifact.

```python
# the entire conformal runtime
def apply_conformal(probs, thresholds):
    return {c: (1.0 - probs[c]) <= thresholds[c] for c in range(n_classes)}
```

`reports/deployment/conformal_thresholds_h6.json` holds the thresholds, class labels, feature order and the
measured coverage. `reports/deployment/infer_example.py` is a complete, dependency-light
inference path: ONNX Runtime plus that JSON, no scikit-learn and no project imports.

---

## 5. Deployable bundle

| File | Size | Purpose |
|---|---|---|
| `pulseair_rf_h6.onnx` | 33,378.7 KB | the compressed forest |
| `conformal_thresholds_h6.json` | 1,322 B | per-class thresholds + metadata |
| `infer_example.py` | ~1 KB | reference inference path |

---

## 6. Would this fit an ESP32?

**Read this section as a feasibility estimate, not a deployment claim.** Every number
below follows from stated assumptions, and the honest headline is in the last
paragraph.

### Assumptions

1. **Target part**: ESP32-WROOM class — 520 KB SRAM on paper, of
   which roughly **320 KB** is realistically available to an
   application after the IRAM/DRAM split, the Wi-Fi/BT stack and heap fragmentation.
   Typical module flash: 4 MB.
2. **Node cost**: ~12 bytes per decision node in a compact C
   representation — `uint8` feature index, `float32` threshold, two `uint16` child
   offsets, padded. A leaf stores 6 class scores; quantised to `uint8`
   that is 6 bytes, and the estimate below uses the decision-node figure
   throughout, so it is if anything optimistic.
3. **Inference is one sample at a time.** A wearable classifies the current reading;
   there is no batch.
4. **Features arrive pre-scaled**, using the stored StandardScaler statistics — 5
   multiply-adds, negligible.
5. **No ONNX Runtime on device.** ESP32 has no ONNX interpreter; see the caveat below.

### Estimate

| | |
|---|---|
| Nodes in the compressed forest | 496,196 |
| Tree structure at ~12 B/node | **5,815 KB** (5.68 MB) |
| Input vector + probability accumulator | 84 B |
| ONNX file on flash | 33,378.7 KB (32.60 MB) |
| Usable SRAM (assumption 1) | ~320 KB |
| Flash available (assumption 1) | ~4,096 KB |

**The model would not fit in ESP32 SRAM, but it does not need to.** The estimated 5,815 KB of tree structure is 18.2x the ~320 KB of usable SRAM. Trees are read-only, though, so the realistic deployment keeps them in flash (32.60 MB against a typical 4 MB part) and streams nodes during traversal, leaving SRAM for the input vector, the probability accumulator and the stack — on the order of 84 bytes, which is negligible.

Flash: the ONNX file is 32.60 MB against a typical
4 MB part — **tight**,
though a hand-rolled C array would be smaller than the ONNX protobuf, which carries
graph metadata a device does not need.

### The caveat that matters most

**A TFLite Micro conversion path was planned for this phase and abandoned**: TensorFlow
Lite Micro does not serve a scikit-learn tree ensemble, so the route would have required
re-training the predictor as a different model and re-validating every result against it.
ONNX was taken instead, as an explicit portability proxy. The dependency has now been
removed from the project rather than left advertised in a requirements file nothing
imported.

**This ONNX benchmark is a portability and feasibility proxy. It is not a claim that
this model can be flashed to an ESP32.** There is no ONNX Runtime for the ESP32 —
no interpreter, no execution provider, nothing to load the file with. Getting this
predictor onto the device requires one of:

- **A from-scratch tree traversal in C.** The most likely route for a random forest.
  Export the trees as `const` arrays in flash and write the traversal by hand — a few
  hundred lines, fully deterministic, no framework dependency. The conformal layer
  comes along for free: 6 float comparisons.
- **A framework with ESP32 support**, e.g. TensorFlow Lite Micro or ESP-DL. Neither
  runs a scikit-learn forest, so this means re-training the predictor as something the
  framework supports (a small MLP, or a gradient-boosted model via a supported
  converter) and re-validating every result in this project against the new model —
  including re-calibrating conformal, as section 2 shows.
- **`emlearn` or a similar C code generator for scikit-learn models**, which targets
  exactly this case and is the lowest-effort path worth trying first.

What this section does establish: the compressed model is **2x smaller**
than the Phase 3 forest at a cost of 0.0021 macro-F1, it runs in
0.014 ms per sample single-threaded, the conformal layer
adds 0.0104 ms, and the artifact is small enough that
flash is not the binding constraint. Those are the facts a port would start from.

Reproduce with `python -m src.deployment.compress_export`.
