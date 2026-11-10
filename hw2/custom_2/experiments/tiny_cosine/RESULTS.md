# Tiny CNN: cosine schedule and early stopping

- Trained **71** epochs; stop reason: `early_stopping`.
- Best checkpoint: epoch **68**.
- Clean validation accuracy: **90.80%**.
- 13-view mean validation accuracy: **89.94%**.
- 13-view prediction consistency: **83.40%**.
- Diagnostic stress mean: **87.97%** (not used in selection).
- ONNX float weights: **510,917**.
- Local CPU median inference latency: **0.2559 ms**.

Original models and exports were not replaced.
Validation is measured every epoch, versus every five epochs in the previous sweep;
more checkpoint selection opportunities can affect apparent improvements.
Actual hidden accuracy and grader latency remain unknown.
