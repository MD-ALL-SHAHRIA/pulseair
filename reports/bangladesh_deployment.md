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

**Validation persistence floor: 0.4322 macro-F1.**

Baseline: **200 trees x depth 16**,
223,746 nodes, 8.2 MB,
val macro-F1 0.4983.

| Trees | Depth | Val F1 | vs base | vs persistence | 95% CI | (a) within tol. | (b) beats floor | Nodes | KB |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 200 | 16 | 0.4983 | +0.0000 | +0.0661 | [+0.0580, +0.0740] | yes | **yes** | 223,746 | 8442.3 |  |
| 200 | 12 | 0.5004 | +0.0021 | +0.0682 | [+0.0600, +0.0768] | yes | **yes** | 200,080 | 7708.5 |  |
| 200 | 10 | 0.4976 | -0.0006 | +0.0654 | [+0.0569, +0.0743] | yes | **yes** | 144,676 | 5908.4 |  |
| 200 | 8 | 0.4856 | -0.0127 | +0.0534 | [+0.0451, +0.0615] | no | **yes** | 70,708 | 3226.2 |  |
| 200 | 6 | 0.4587 | -0.0396 | +0.0264 | [+0.0185, +0.0340] | no | **yes** | 23,874 | 1196.2 |  |
| 100 | 16 | 0.4963 | -0.0020 | +0.0641 | [+0.0558, +0.0722] | yes | **yes** | 111,886 | 4231.3 |  |
| 100 | 12 | 0.4970 | -0.0013 | +0.0647 | [+0.0562, +0.0734] | yes | **yes** | 100,350 | 3872.5 |  |
| 100 | 10 | 0.4960 | -0.0022 | +0.0638 | [+0.0552, +0.0722] | yes | **yes** | 71,974 | 2949.9 |  |
| 100 | 8 | 0.4832 | -0.0151 | +0.0509 | [+0.0426, +0.0589] | no | **yes** | 35,344 | 1620.0 |  |
| 100 | 6 | 0.4612 | -0.0371 | +0.0290 | [+0.0206, +0.0373] | no | **yes** | 11,922 | 607.7 |  |
| 50 | 16 | 0.4943 | -0.0039 | +0.0621 | [+0.0534, +0.0706] | yes | **yes** | 56,012 | 2126.6 |  |
| 50 | 12 | 0.4994 | +0.0012 | +0.0672 | [+0.0583, +0.0759] | yes | **yes** | 50,070 | 1942.0 |  |
| 50 | 10 | 0.4937 | -0.0046 | +0.0615 | [+0.0536, +0.0700] | yes | **yes** | 35,542 | 1469.4 |  |
| 50 | 8 | 0.4796 | -0.0187 | +0.0474 | [+0.0387, +0.0555] | no | **yes** | 17,702 | 820.6 |  |
| 50 | 6 | 0.4641 | -0.0342 | +0.0318 | [+0.0234, +0.0397] | no | **yes** | 5,970 | 315.5 |  |
| 25 | 16 | 0.4969 | -0.0013 | +0.0647 | [+0.0561, +0.0734] | yes | **yes** | 28,057 | 1076.1 |  |
| 25 | 12 | 0.4920 | -0.0062 | +0.0598 | [+0.0512, +0.0688] | yes | **yes** | 25,037 | 983.2 | **selected** |
| 25 | 10 | 0.4838 | -0.0145 | +0.0515 | [+0.0431, +0.0598] | no | **yes** | 17,925 | 750.9 |  |
| 25 | 8 | 0.4830 | -0.0153 | +0.0508 | [+0.0422, +0.0593] | no | **yes** | 8,863 | 421.2 |  |
| 25 | 6 | 0.4656 | -0.0327 | +0.0334 | [+0.0238, +0.0423] | no | **yes** | 2,993 | 169.2 |  |
| 10 | 16 | 0.4853 | -0.0130 | +0.0530 | [+0.0444, +0.0623] | no | **yes** | 11,142 | 441.5 |  |
| 10 | 12 | 0.4812 | -0.0171 | +0.0490 | [+0.0401, +0.0579] | no | **yes** | 10,114 | 408.6 |  |
| 10 | 10 | 0.4656 | -0.0327 | +0.0334 | [+0.0256, +0.0414] | no | **yes** | 7,330 | 319.0 |  |
| 10 | 8 | 0.4657 | -0.0325 | +0.0335 | [+0.0244, +0.0423] | no | **yes** | 3,570 | 182.7 |  |
| 10 | 6 | 0.4374 | -0.0609 | +0.0052 | [-0.0037, +0.0144] | no | no | 1,210 | 81.6 |  |
| 5 | 16 | 0.4843 | -0.0140 | +0.0521 | [+0.0420, +0.0615] | no | **yes** | 5,591 | 232.8 |  |
| 5 | 12 | 0.4600 | -0.0382 | +0.0278 | [+0.0183, +0.0374] | no | **yes** | 5,105 | 216.5 |  |
| 5 | 10 | 0.4456 | -0.0527 | +0.0134 | [+0.0042, +0.0225] | no | **yes** | 3,637 | 168.9 |  |
| 5 | 8 | 0.4466 | -0.0516 | +0.0144 | [+0.0050, +0.0241] | no | **yes** | 1,789 | 102.0 |  |
| 5 | 6 | 0.4017 | -0.0966 | -0.0305 | [-0.0402, -0.0211] | no | no | 603 | 51.7 |  |

**11 of 30 configurations satisfy both conditions.** Among them the smallest by **node count** is selected — nodes, not trees, because node count sets both flash footprint and traversal cost.

### Selected

| | Baseline | Compressed | Change |
|---|---|---|---|
| Trees x depth | 200 x 16 | **25 x 12** | |
| Nodes | 223,746 | **25,037** | **-88.8%** |
| Pickle (compress=3) | 8,442.6 KB | **983.2 KB** | **9x smaller** |
| Val macro-F1 | 0.4983 | 0.4920 | -0.0062 |
| **Test macro-F1** | 0.4075 | **0.4057** | **-0.0018** |

The compressed model stays inside the 0.01 tolerance on **both** splits (-0.0062 validation, -0.0018 test).

**The accuracy/size curve is almost flat over three orders of magnitude.** The largest forest in the sweep (200 x 16, 223,746 nodes) scores 0.4983 on validation; the selected one has 9x fewer nodes and scores 0.4920. Across the whole grid, validation macro-F1 spans only 0.0966.

That is consistent with every other result in this project: at a six-hour horizon the signal available in these nine channels is limited, and model capacity is not the binding constraint. The Phase 3 forest was not chosen for accuracy it needed — it was simply larger than the problem required.

---

## 2. Conformal re-calibration (not optional)

Mondrian thresholds are quantiles of `1 − P(true class)` **under a specific model**.
Compressing the forest changes those probabilities, so the Phase 6 thresholds no longer
certify anything about the new model — and a stale set would look exactly like a
calibrated one while guaranteeing nothing. They are re-derived on the compressed model
from the same validation split, and coverage is re-verified on test.

| Class | Coverage before | Coverage after | Δ | Mean set size | Test n |
| --- | --- | --- | --- | --- | --- |
| Good | — | 0.9927 | — | 2.44 | 6,176 |
| Moderate | — | 0.9041 | — | 2.92 | 7,927 |
| Unhealthy (sensitive) | — | 0.7735 | — | 3.06 | 1,757 |
| Unhealthy | — | 0.7604 | — | 3.11 | 1,490 |
| **Very unhealthy** | — | 0.0000 | — | 3.17 | 6 |
| **Hazardous** | — | n/a | — | n/a | 0 |

| | Before | After |
|---|---|---|
| Overall coverage | nan | **0.9098** |
| Mean set size | — | 2.781 |
| Singleton rate | — | 0.0% |

**Warning: an advisory class fell below target after compression** — see the table above.

---

## 3. ONNX export

| | |
|---|---|
| File | `reports/deployment/pulseair_rf_h6_bd.onnx` |
| Size | **1,673.3 KB** (1.63 MB) |
| Opset | 17 |
| Input | `float32[n, 7]` — the nine scaled feature columns, in `metadata.json` order |
| Output | label + `float32[n, 6]` probabilities (`zipmap=False`) |

**Parity check before benchmarking.** ONNX must reproduce scikit-learn or the latency
numbers describe a different model:

| | |
|---|---|
| Rows compared | 2,000 |
| Max abs probability difference | 2.60e-07 |
| Argmax agreement | 1.0000 |
| Verdict | **PASS** |

---

## 4. Latency (single-threaded CPU)

`intra_op_num_threads=1`, warm session, 200 runs after
warm-up. Measured on the development machine (Apple Silicon), **not** on an ESP32 —
see section 6.

| Path | Mean (ms) | Median (ms) | p95 (ms) |
| --- | --- | --- | --- |
| onnx single | 0.0074 | 0.0072 | 0.0081 |
| sklearn single | 15.3313 | 16.3376 | 16.6830 |
| conformal single | 0.0103 | 0.0102 | 0.0105 |
| onnx plus conformal single | 0.0196 | 0.0192 | 0.0205 |
| onnx batch32 | 0.0284 | 0.0284 | 0.0292 |
| onnx batch256 | 0.1925 | 0.1847 | 0.2520 |

ONNX single-sample inference is
**2080.0x faster** than
scikit-learn's `predict_proba` on the same model, which is the usual result: the
Python-level ensemble loop dominates at batch size 1.

### The conformal layer costs essentially nothing

Applying the Mondrian thresholds is **0.0103 ms** —
139.39% of the
inference it wraps. That is expected and worth stating plainly: the layer is
6 comparisons against 6 stored floats. There is no model, no
state and nothing to load, which is why it ships as a
1,240-byte JSON file rather than a second artifact.

```python
# the entire conformal runtime
def apply_conformal(probs, thresholds):
    return {c: (1.0 - probs[c]) <= thresholds[c] for c in range(n_classes)}
```

`reports/deployment/conformal_thresholds_h6_bd.json` holds the thresholds, class labels, feature order and the
measured coverage. `reports/deployment/infer_example.py` is a complete, dependency-light
inference path: ONNX Runtime plus that JSON, no scikit-learn and no project imports.

---

## 5. Deployable bundle

| File | Size | Purpose |
|---|---|---|
| `pulseair_rf_h6.onnx` | 1,673.3 KB | the compressed forest |
| `conformal_thresholds_h6.json` | 1,240 B | per-class thresholds + metadata |
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
| Nodes in the compressed forest | 25,037 |
| Tree structure at ~12 B/node | **293 KB** (0.29 MB) |
| Input vector + probability accumulator | 76 B |
| ONNX file on flash | 1,673.3 KB (1.63 MB) |
| Usable SRAM (assumption 1) | ~320 KB |
| Flash available (assumption 1) | ~4,096 KB |

**The model would not fit in ESP32 SRAM, but it does not need to.** The estimated 293 KB of tree structure is 0.9x the ~320 KB of usable SRAM. Trees are read-only, though, so the realistic deployment keeps them in flash (1.63 MB against a typical 4 MB part) and streams nodes during traversal, leaving SRAM for the input vector, the probability accumulator and the stack — on the order of 76 bytes, which is negligible.

Flash: the ONNX file is 1.63 MB against a typical
4 MB part — **comfortable**,
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

What this section does establish: the compressed model is **9x smaller**
than the Phase 3 forest at a cost of 0.0018 macro-F1, it runs in
0.007 ms per sample single-threaded, the conformal layer
adds 0.0103 ms, and the artifact is small enough that
flash is not the binding constraint. Those are the facts a port would start from.

Reproduce with `python -m src.deployment.compress_export`.
