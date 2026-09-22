"""Minimal inference path: ONNX probabilities -> conformal prediction set.

This is the entire runtime contract. No scikit-learn, no pickle, no project imports.

    python infer_example.py
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).parent
CONF = json.loads((HERE / "conformal_thresholds_h6.json").read_text())
SESS = ort.InferenceSession(str(HERE / CONF["model"]),
                            providers=["CPUExecutionProvider"])
THRESH = {int(k): v for k, v in CONF["thresholds_by_index"].items()}


def predict_set(x: np.ndarray) -> list[list[str]]:
    """x: (n, 9) float32, scaled with the training scaler. Returns one set per row."""
    probs = np.asarray(SESS.run(None, {SESS.get_inputs()[0].name:
                                       x.astype(np.float32)})[1])
    labels = CONF["class_labels"]
    return [[labels[c] for c in range(len(labels))
             if (1.0 - row[c]) <= THRESH[c] + 1e-12] for row in probs]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    demo = rng.normal(size=(3, len(CONF["feature_columns"]))).astype(np.float32)
    for i, s in enumerate(predict_set(demo)):
        print(f"row {i}: {s or 'EMPTY (no category met the threshold)'}")
