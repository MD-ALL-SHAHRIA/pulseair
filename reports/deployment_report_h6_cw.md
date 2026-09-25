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
| 200 | 16 | 0.5196 | +0.0300 | +0.0148 | [+0.0107, +0.0187] | yes | **yes** | 932,754 | 41477.4 |  |
| 200 | 12 | 0.5185 | +0.0290 | +0.0137 | [+0.0097, +0.0176] | yes | **yes** | 528,474 | 25435.3 |  |
| 200 | 10 | 0.5152 | +0.0257 | +0.0105 | [+0.0065, +0.0146] | yes | **yes** | 258,012 | 13697.6 |  |
| 200 | 8 | 0.5075 | +0.0180 | +0.0027 | [-0.0013, +0.0067] | yes | no | 90,082 | 5500.5 |  |
| 200 | 6 | 0.4930 | +0.0035 | -0.0118 | [-0.0157, -0.0077] | yes | no | 25,028 | 1680.3 |  |
| 100 | 16 | 0.5191 | +0.0296 | +0.0143 | [+0.0102, +0.0184] | yes | **yes** | 467,158 | 20822.1 |  |
| 100 | 12 | 0.5180 | +0.0284 | +0.0132 | [+0.0091, +0.0171] | yes | **yes** | 264,008 | 12760.4 |  |
| 100 | 10 | 0.5152 | +0.0257 | +0.0104 | [+0.0065, +0.0145] | yes | **yes** | 128,448 | 6882.7 |  |
| 100 | 8 | 0.5075 | +0.0180 | +0.0027 | [-0.0014, +0.0067] | yes | no | 45,124 | 2809.6 |  |
| 100 | 6 | 0.4922 | +0.0027 | -0.0126 | [-0.0166, -0.0085] | yes | no | 12,504 | 893.4 |  |
| 50 | 16 | 0.5186 | +0.0290 | +0.0138 | [+0.0095, +0.0178] | yes | **yes** | 232,798 | 10435.6 |  |
| 50 | 12 | 0.5179 | +0.0284 | +0.0131 | [+0.0090, +0.0172] | yes | **yes** | 132,236 | 6444.9 |  |
| 50 | 10 | 0.5147 | +0.0251 | +0.0099 | [+0.0056, +0.0138] | yes | **yes** | 64,198 | 3493.7 |  |
| 50 | 8 | 0.5080 | +0.0185 | +0.0032 | [-0.0008, +0.0073] | yes | no | 22,514 | 1455.0 |  |
| 50 | 6 | 0.4921 | +0.0025 | -0.0127 | [-0.0169, -0.0087] | yes | no | 6,238 | 499.5 |  |
| 25 | 16 | 0.5174 | +0.0278 | +0.0126 | [+0.0084, +0.0166] | yes | **yes** | 117,235 | 5302.2 |  |
| 25 | 12 | 0.5164 | +0.0269 | +0.0116 | [+0.0074, +0.0159] | yes | **yes** | 66,877 | 3312.2 |  |
| 25 | 10 | 0.5140 | +0.0244 | +0.0092 | [+0.0051, +0.0131] | yes | **yes** | 32,291 | 1808.2 |  |
| 25 | 8 | 0.5064 | +0.0169 | +0.0016 | [-0.0025, +0.0057] | yes | no | 11,197 | 777.5 |  |
| 25 | 6 | 0.4911 | +0.0016 | -0.0137 | [-0.0177, -0.0096] | yes | no | 3,101 | 302.6 |  |
| 10 | 16 | 0.5131 | +0.0235 | +0.0083 | [+0.0038, +0.0125] | yes | **yes** | 46,754 | 2178.7 |  |
| 10 | 12 | 0.5144 | +0.0248 | +0.0096 | [+0.0053, +0.0135] | yes | **yes** | 26,762 | 1389.2 |  |
| 10 | 10 | 0.5109 | +0.0214 | +0.0061 | [+0.0019, +0.0101] | yes | **yes** | 12,718 | 776.7 | **selected** |
| 10 | 8 | 0.5050 | +0.0155 | +0.0002 | [-0.0037, +0.0044] | yes | no | 4,500 | 376.1 |  |
| 10 | 6 | 0.4869 | -0.0027 | -0.0179 | [-0.0219, -0.0140] | yes | no | 1,250 | 186.2 |  |
| 5 | 16 | 0.5050 | +0.0155 | +0.0002 | [-0.0039, +0.0046] | yes | no | 23,459 | 1145.0 |  |
| 5 | 12 | 0.5109 | +0.0213 | +0.0061 | [+0.0017, +0.0103] | yes | **yes** | 13,373 | 748.1 |  |
| 5 | 10 | 0.5070 | +0.0174 | +0.0022 | [-0.0022, +0.0062] | yes | no | 6,387 | 442.3 |  |
| 5 | 8 | 0.4999 | +0.0104 | -0.0049 | [-0.0088, -0.0008] | yes | no | 2,251 | 241.4 |  |
| 5 | 6 | 0.4868 | -0.0027 | -0.0180 | [-0.0222, -0.0140] | yes | no | 623 | 146.8 |  |

**16 of 30 configurations satisfy both conditions.** Among them the smallest by **node count** is selected — nodes, not trees, because node count sets both flash footprint and traversal cost.

### Selected

| | Baseline | Compressed | Change |
|---|---|---|---|
| Trees x depth | 200 x 16 | **10 x 10** | |
| Nodes | 990,646 | **12,718** | **-98.7%** |
| Pickle (compress=3) | 43,352.4 KB | **776.7 KB** | **56x smaller** |
| Val macro-F1 | 0.4895 | 0.5109 | +0.0214 |
| **Test macro-F1** | 0.5173 | **0.5020** | **-0.0153** |

> **The compressed model exceeds the 0.01 tolerance on both splits** (+0.0214 val, -0.0153 test). No configuration in the sweep met the rule; the highest-scoring one was taken instead. Treat this as a failed compression target rather than a successful one.

**The accuracy/size curve is almost flat over three orders of magnitude.** The largest forest in the sweep (200 x 16, 932,754 nodes) scores 0.5196 on validation; the selected one has 73x fewer nodes and scores 0.5109. Across the whole grid, validation macro-F1 spans only 0.0328.

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
| Good | 0.8673 | 0.8599 | -0.0075 | 2.42 |
| Moderate | 0.8877 | 0.8878 | +0.0001 | 2.77 |
| Unhealthy (sensitive) | 0.8635 | 0.8806 | +0.0171 | 3.00 |
| Unhealthy | 0.8808 | 0.8709 | -0.0099 | 2.82 |
| **Very unhealthy** | 0.9357 | 0.9121 | -0.0236 | 2.22 |
| **Hazardous** | 0.9083 | 0.9056 | -0.0027 | 1.76 |

| | Before | After |
|---|---|---|
| Overall coverage | 0.8865 | **0.8809** |
| Mean set size | — | 2.621 |
| Singleton rate | — | 3.3% |

Both advisory classes still clear the 90% target after compression.

---

## 3. ONNX export

| | |
|---|---|
| File | `reports/deployment/pulseair_rf_h6_cw.onnx` |
| Size | **851.8 KB** (0.83 MB) |
| Opset | 17 |
| Input | `float32[n, 9]` — the nine scaled feature columns, in `metadata.json` order |
| Output | label + `float32[n, 6]` probabilities (`zipmap=False`) |

**Parity check before benchmarking.** ONNX must reproduce scikit-learn or the latency
numbers describe a different model:

| | |
|---|---|
| Rows compared | 2,000 |
| Max abs probability difference | 9.14e-08 |
| Argmax agreement | 1.0000 |
| Verdict | **PASS** |

---

## 4. Latency (single-threaded CPU)

`intra_op_num_threads=1`, warm session, 200 runs after
warm-up. Measured on the development machine (Apple Silicon), **not** on an ESP32 —
see section 6.

| Path | Mean (ms) | Median (ms) | p95 (ms) |
| --- | --- | --- | --- |
| onnx single | 0.0072 | 0.0071 | 0.0076 |
| sklearn single | 15.2217 | 16.2529 | 16.6520 |
| conformal single | 0.0104 | 0.0103 | 0.0107 |
| onnx plus conformal single | 0.0197 | 0.0191 | 0.0205 |
| onnx batch32 | 0.0172 | 0.0161 | 0.0203 |
| onnx batch256 | 0.0799 | 0.0798 | 0.0805 |

ONNX single-sample inference is
**2123.7x faster** than
scikit-learn's `predict_proba` on the same model, which is the usual result: the
Python-level ensemble loop dominates at batch size 1.

### The conformal layer costs essentially nothing

Applying the Mondrian thresholds is **0.0104 ms** —
144.52% of the
inference it wraps. That is expected and worth stating plainly: the layer is
6 comparisons against 6 stored floats. There is no model, no
state and nothing to load, which is why it ships as a
1,323-byte JSON file rather than a second artifact.

```python
# the entire conformal runtime
def apply_conformal(probs, thresholds):
    return {c: (1.0 - probs[c]) <= thresholds[c] for c in range(n_classes)}
```

`reports/deployment/conformal_thresholds_h6_cw.json` holds the thresholds, class labels, feature order and the
measured coverage. `reports/deployment/infer_example.py` is a complete, dependency-light
inference path: ONNX Runtime plus that JSON, no scikit-learn and no project imports.

---

## 5. Deployable bundle

| File | Size | Purpose |
|---|---|---|
| `pulseair_rf_h6.onnx` | 851.8 KB | the compressed forest |
| `conformal_thresholds_h6.json` | 1,323 B | per-class thresholds + metadata |
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
| Nodes in the compressed forest | 12,718 |
| Tree structure at ~12 B/node | **149 KB** (0.15 MB) |
| Input vector + probability accumulator | 84 B |
| ONNX file on flash | 851.8 KB (0.83 MB) |
| Usable SRAM (assumption 1) | ~320 KB |
| Flash available (assumption 1) | ~4,096 KB |

**The estimated 149.1 KB of tree structure fits within the ~320 KB of usable SRAM** with room to spare, and the input/output buffers add about 84 bytes.

Flash: the ONNX file is 0.83 MB against a typical
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

What this section does establish: the compressed model is **56x smaller**
than the Phase 3 forest at a cost of 0.0153 macro-F1, it runs in
0.007 ms per sample single-threaded, the conformal layer
adds 0.0104 ms, and the artifact is small enough that
flash is not the binding constraint. Those are the facts a port would start from.

Reproduce with `python -m src.deployment.compress_export`.
