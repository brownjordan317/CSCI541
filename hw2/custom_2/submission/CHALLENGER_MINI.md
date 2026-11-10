# Next external-evaluation candidate: mini GAP CNN

**File:** `challenger_mini.onnx`, a byte-identical copy of
`../experiments/score_sweep_v2/mini_gap/model.onnx`.

This is the highest-ranked new candidate under the local score proxy.
It is **not yet a proven replacement** for `recommended_tiny.onnx`, whose
actual reported grading score is 0.638.

| Metric | Mini challenger | Confirmed tiny cosine |
|---|---:|---:|
| Exported float weights | 174,373 | 510,917 |
| Clean local validation | 91.00% | 90.80% |
| 13-view local validation mean | 89.89% | 89.94% |
| Diagnostic stress mean | 88.12% | 87.97% |
| Paired local CPU median latency | 0.2319 ms | 0.2562 ms |
| Actual hidden accuracy | **Unknown** | 88.7% |
| Actual grading score | **Unknown** | 0.638 |

Mini's local score-proxy gain is **+0.02505**, combining validation-accuracy
change and size/local-latency ratios. This is **not** a measured external
score improvement. Do not assume its actual score will be 0.663.

Architecture: three residual stages, widths 16/32/64, two blocks each,
BatchNorm, global average pooling, and a direct linear five-class head.
It uses conventional convolutional/residual/global-pooling methods from
the lecture topics, not pretrained models, ensembles, or test-time augmentation.

Training: same source split and robust augmentation as preceding tiny trials;
Adam with plateau-triggered LR reductions; minimum 80 epochs, no default cap;
stopped after **91 epochs** at LR **0.00001**. Selected best weights from
**epoch 79**. The overall run meets the minimum training duration even though
the best validation checkpoint occurred earlier. ONNX checks pass.

Input remains raw RGB float32 `[1,3,64,64]` in `[0,1]`, named `image`;
normalization is inside the graph. Output is five logits named `logits`.

Submit this challenger next to test whether the score-proxy advantage
transfers. Retain the externally tested tiny cosine as the fallback.
