# Score-oriented architecture sweep

**All new results are local proxies, not external scores.**
Confirmed baseline: tiny cosine, 88.7% hidden accuracy and 0.638 actual score.

| Candidate | ONNX float weights | Epochs / best | Clean | 13-view mean | Paired latency ms | Proxy score gain |
|---|---:|---:|---:|---:|---:|---:|
| mini_gap | 174,373 | 91 / 79 | 91.00% | 89.89% | 0.2319 | +0.02505 |
| micro_gap | 98,333 | 89 / 84 | 88.80% | 87.95% | 0.2010 | +0.02121 |
| shallow_gap | 77,381 | 99 / 64 | 87.00% | 86.66% | 0.1565 | +0.01897 |
| nano_gap | 43,925 | 102 / 48 | 85.80% | 84.46% | 0.1655 | +0.00812 |

Local score-proxy leader: **mini_gap**; export: `mini_gap/model.onnx`.
The confirmed submission was not replaced.

Proxy gain = validation-accuracy change − 0.05 log10(parameter ratio) − 0.05 log10(local latency ratio).
The reference is re-timed alongside each candidate. Timings use single-thread ONNX Runtime CPU
session.run with batch 1, 30 warmups, three interleaved rounds of 300 calls.
Accuracy changes may not transfer to hidden images; CPU latency ratios may differ on the grader.
Do not interpret 0.638 plus this proxy gain as a measured or guaranteed external score.
