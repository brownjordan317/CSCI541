# Tiny CNN: plateau schedule and minimum-LR-gated early stopping

- Trained **80** epochs; stop reason: `maximum_epochs`.
- Last training LR: **0.00001563**; reached minimum: **False**.
- Best checkpoint: epoch **73**.
- Clean validation accuracy: **89.40%**.
- 13-view mean validation accuracy: **89.62%**.
- 13-view prediction consistency: **83.60%**.
- Diagnostic stress mean: **88.23%** (not used in selection).
- ONNX float weights: **510,917**.
- Local CPU median inference latency: **0.2517 ms**.

Original models and exports were not replaced.
Validation is measured every epoch, versus every five epochs in the previous sweep;
more checkpoint selection opportunities can affect apparent improvements.
Actual hidden accuracy and grader latency remain unknown.
