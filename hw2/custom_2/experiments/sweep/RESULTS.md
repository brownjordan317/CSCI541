# Custom 2 experiment results

All values below are local validation proxies, not hidden accuracy.

| Model | Learned parameters | Clean | 13-view mean | 13-view consistency | Diagnostic stress mean |
|---|---:|---:|---:|---:|---:|
| Original custom | 1,989,029 | 91.60% | 90.46% | 84.80% | 85.03% |
| compact_original_aug | 1,129,661 | 90.80% | 90.25% | 83.20% | 86.42% |
| compact_robust_aug | 1,129,661 | 91.00% | 90.48% | 85.00% | 88.87% |
| tiny_robust_aug | 511,957 | 90.40% | 89.78% | 83.40% | 87.72% |
| compact_robust_low_lr | 1,129,661 | 91.60% | 90.55% | 83.00% | 89.40% |

Selected candidate: **compact_robust_low_lr**.
Smallest candidate within 0.3 percentage points of the best candidate's 13-view mean.
Stress results and reported hidden scores do not enter selection.

The original model's 90.6% hidden accuracy was supplied by the user.
The new model requires external evaluation; retain the original as a fallback.
