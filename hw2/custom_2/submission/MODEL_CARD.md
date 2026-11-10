# Recommended: tiny cosine CNN

- **Submission file:** `recommended_tiny.onnx`.
- **Source:** `../experiments/tiny_cosine/tiny_cosine_cnn.onnx`; identical bytes.
- **Architecture:** residual CNN, BatchNorm, global average pooling;
  stage widths 16, 32, 64, 96; 256-unit hidden classifier; five output logits.
- **Input:** `image`, float32 `[1,3,64,64]`, RGB values in `[0,1]`.
- **Normalization:** inside the graph; do not normalize inputs a second time.
- **Learned parameters:** 511,957. **Exported float weights:** 510,917.
- **Training:** robust augmentation, Adam, cosine LR taper; trained 71 epochs,
  selected epoch 68. This earlier run predates the corrected minimum-80 policy.
- **Local validation:** 90.80% clean, 89.94% mean across 13 views;
  83.40% all-view prediction consistency; 87.97% diagnostic stress mean.
- **Local single-thread CPU median inference:** approximately 0.256 ms;
  excludes loading and preprocessing. Not evaluator latency.
- **ONNX:** opset 17, four-dimensional fixed input, five logits, no ensemble.
- **User-reported external hidden accuracy:** **88.7%**.
- **User-reported actual grading score:** **0.638**, versus **0.612** for the
  original custom model (90.6% hidden accuracy).

Recommended for the supplied parameter/latency-penalized score: the external
results confirm a **0.026 score improvement**, despite a **1.9-percentage-point
hidden-accuracy decrease**. These results were reported by the user, not
independently retrieved from the evaluator. Submit the original
`../../custom/custom_cnn.onnx` if prioritizing raw hidden accuracy rather than
the grading score. Actual evaluator latency was not supplied.

For a tiny model that follows the corrected minimum-80 training policy, use
`../experiments/tiny_plateau_min80/tiny_plateau_cnn.onnx` instead (114 epochs,
best checkpoint 99, 89.83% local 13-view mean).
