# Tiny CNN: plateau schedule and minimum-LR-gated early stopping

- Trained **114** epochs; stop reason: `early_stopping`.
- Last training LR: **0.00001000**; reached minimum: **True**.
- Best checkpoint: epoch **99**.
- Clean validation accuracy: **90.40%**.
- 13-view mean validation accuracy: **89.83%**.
- 13-view prediction consistency: **83.40%**.
- Diagnostic stress mean: **87.85%** (not used in selection).
- ONNX float weights: **510,917**.
- Local CPU median inference latency: **0.2568 ms**.

Original models and exports were not replaced.
Validation is measured every epoch, versus every five epochs in the previous sweep;
more checkpoint selection opportunities can affect apparent improvements.
Actual hidden accuracy and grader latency remain unknown.
