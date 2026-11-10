# External evaluation: confirmed score trade-off

Results below were **reported by the user** following submission; they were
not independently retrieved from the evaluator.

| Model | Artifact | Hidden accuracy | Grading score |
|---|---|---:|---:|
| **Tiny cosine** | `../submission/recommended_tiny.onnx` | **88.7%** | **0.638** |
| Original custom | `../../custom/custom_cnn.onnx` | 90.6% | 0.612 |

The tiny model loses **1.9 percentage points** of hidden accuracy but gains
**0.026** in score. This confirms the recommendation to submit tiny cosine
when optimizing the stated grading objective:

```text
score = hidden_accuracy - 0.05*log10(params) - 0.05*log10(median_latency_ms)
```

The size/latency penalty reduction outweighs the accuracy decrease. This does
not mean tiny cosine has better hidden accuracy; the original wins that metric.
Exact evaluator parameter counts and latency measurements were not provided,
so no measured evaluator latency is inferred here from the rounded scores.

`SCORE_COMPARISON.md` and experiment JSON files remain historical **local
validation/latency proxy** records. Their values should not be confused with
these actual submission results. No further model training or selection was
performed when recording this feedback.
