# Local score-proxy comparison

**Not actual submission scores.** Accuracy below is a validation proxy, not hidden accuracy.
Latency is local single-thread ONNX Runtime CPU inference, not evaluator hardware.
The fraction-accuracy score assumes the formula uses 0.906 rather than 90.6.

| Model | ONNX float values | Local median ms | 13-view accuracy | Score proxy (fraction accuracy) |
|---|---:|---:|---:|---:|
| tiny_robust_aug | 510,917 | 0.2557 | 89.78% | 0.64204 |
| compact_robust_low_lr | 1,128,101 | 0.3767 | 90.55% | 0.62412 |
| compact_robust_aug | 1,128,101 | 0.3770 | 90.48% | 0.62333 |
| compact_original_aug | 1,128,101 | 0.3764 | 90.25% | 0.62106 |
| original_custom | 1,986,949 | 0.4972 | 90.46% | 0.60488 |

Timing: 30 warmup calls per model, three interleaved rounds of 300 timed calls by default.
Input is contiguous float32 [1,3,64,64]; session creation/loading and preprocessing are excluded.
Parameter penalty uses exported float initializer elements, matching the supplied local checker.
The evaluator's exact parameter-count definition and latency configuration need confirmation.
Actual hidden scores remain unknown for all new candidates.
