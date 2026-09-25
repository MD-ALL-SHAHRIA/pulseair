"""Edge deployment: ONNX export, compression sweep and latency benchmarking.

A TFLite / TFLite Micro conversion path was planned for the ESP32 and
abandoned. The deployed predictor is a tree ensemble, which TFLite Micro does
not serve well, and the project reports the ONNX benchmark as an explicit
portability proxy rather than a flashability claim. See
``reports/deployment_report_h6.md`` for what that does and does not establish.
"""
