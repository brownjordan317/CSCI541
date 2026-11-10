# Custom 2 — start here

**The user identified challenger mini as best.** Its exact model, checkpoints,
and standalone code are now collected in [`../p3_best/`](../p3_best/).
Submit `../p3_best/p3_best.onnx` for that model. The numerical external result
was not provided. The sections below preserve the earlier experiment history
and tiny-cosine recommendation; they are superseded by this selection.

## Recommended submission

**[submission/recommended_tiny.onnx](submission/recommended_tiny.onnx)**

This is an unchanged copy of the **tiny cosine-LR model, best epoch 68**.
It has the highest local 13-view validation accuracy among the tiny models,
while retaining their low parameter count and latency. See its
[model card](submission/MODEL_CARD.md) for qualifications.

The recommendation is now supported by **user-reported external evaluation**:

| Model | Hidden accuracy | Actual grading score |
|---|---:|---:|
| **Tiny cosine (recommended)** | **88.7%** | **0.638** |
| Original custom | 90.6% | 0.612 |

Tiny cosine sacrifices **1.9 percentage points** of hidden accuracy but gains
**0.026** in the actual grading score. It is the confirmed better choice for
the score objective among these two submissions. The original remains better
for raw hidden accuracy and is untouched at `../custom/custom_cnn.onnx`.
See [external evaluation results](reports/EXTERNAL_RESULTS.md).

**Next candidate to evaluate:**
[`submission/challenger_mini.onnx`](submission/challenger_mini.onnx).
The completed score sweep found nearly the same local transformed accuracy
as tiny cosine with **66% fewer exported weights** and lower local latency.
It leads the new candidates on the local score proxy, but has **no actual
hidden accuracy or grading score yet**. Keep the confirmed 0.638 submission
until the challenger has been externally evaluated. Details:
[challenger model card](submission/CHALLENGER_MINI.md).

## Folder map

```text
custom_2/
├── README.md                 ← this index
├── submission/               ← recommended ONNX and its model card
├── experiments/
│   ├── sweep/                ← four original trials and their comparison
│   ├── tiny_cosine/          ← 71-epoch run; best checkpoint at 68
│   ├── tiny_plateau_capped/  ← earlier 80-epoch-capped run (superseded policy)
│   └── tiny_plateau_min80/   ← corrected minimum-80 run; stopped at 114
├── reports/                  ← score benchmark, historical notes, move manifest
├── code/                     ← training, benchmarking, LR controller, tests
└── logs/                     ← combined console logs for every run
```

No checkpoints, logs, or experiment results were deleted. Artifact moves were
verified with SHA-256 hashes; [cleanup_manifest.json](reports/cleanup_manifest.json)
records old/new paths. The recommendation is a byte-identical copy, not a
re-exported or retrained model. No training was launched during cleanup.

## Results at a glance

**All accuracies below are local validation results, not hidden accuracy.**
“13-view” means the same 500 held-out photos under 13 fixed transformations.

| Model | Learned parameters | Clean | 13-view mean | Stress mean | Location |
|---|---:|---:|---:|---:|---|
| Original custom reference | 1,989,029 | 91.60% | 90.46% | 85.03% | `../custom/` |
| Compact, original augmentation | 1,129,661 | 90.80% | 90.25% | 86.42% | `experiments/sweep/runs/compact_original_aug/` |
| Compact, robust augmentation | 1,129,661 | 91.00% | 90.48% | 88.87% | `experiments/sweep/runs/compact_robust_aug/` |
| Compact, robust, lower LR | 1,129,661 | 91.60% | **90.55%** | **89.40%** | `experiments/sweep/runs/compact_robust_low_lr/` |
| Tiny, original step schedule | 511,957 | 90.40% | 89.78% | 87.72% | `experiments/sweep/runs/tiny_robust_aug/` |
| **Tiny, cosine schedule** | **511,957** | **90.80%** | **89.94%** | **87.97%** | **`experiments/tiny_cosine/`** |
| Tiny, plateau, capped at 80 | 511,957 | 89.40% | 89.62% | 88.23% | `experiments/tiny_plateau_capped/` |
| Tiny, plateau, minimum 80 | 511,957 | 90.40% | 89.83% | 87.85% | `experiments/tiny_plateau_min80/` |

Tiny exports contain **510,917 float initializer values**. Learned-parameter
and exported-initializer counts differ slightly because of BatchNorm fusion.
All exported models satisfy the 3,000,000-float budget.

## Why recommend tiny rather than the highest-accuracy compact model?

```text
score = hidden_accuracy - 0.05*log10(params) - 0.05*log10(median_latency_ms)
```

If accuracy uses fractions, halving parameters adds approximately **0.0151**
to the score; halving latency adds the same amount. A small accuracy sacrifice
can therefore be worthwhile. The tiny architecture removes about **74%** of
the original custom model's weights and locally runs at about half its latency.
Compared with the compact low-LR model, its local validation loss is about
0.62 percentage points, but its size/latency penalties are substantially lower.

The tiny cosine run slightly outperformed the other tiny runs locally at
essentially the same size and speed. These small accuracy differences are
not statistically established: this is one seed, repeated selection on 500
photos, and the later runs evaluated checkpoints more frequently.

**Training-policy caveat:** the cosine model came from the earlier run that
stopped at 71 epochs before reaching minimum LR. If you require the submitted
model itself to follow the corrected “at least 80 epochs, then stop only at
minimum LR” policy, use
[`experiments/tiny_plateau_min80/tiny_plateau_cnn.onnx`](experiments/tiny_plateau_min80/tiny_plateau_cnn.onnx)
instead. That is the 114-epoch run, best checkpoint at epoch 99. Its local
13-view mean is only 0.11 percentage points below cosine's.

Local CPU timing is **not** evaluator timing. The earlier score proxies assumed
fractional accuracy. If the grader instead uses
percentage accuracy (`90.6`), accuracy dominates the penalties and this
proxy formula must be reconsidered. Regardless, the actual reported scores
above directly establish the tiny model's advantage under the real grader.
For highest currently **confirmed hidden accuracy**, retain the original custom model.

## Where to look next

- [Tiny cosine results](experiments/tiny_cosine/RESULTS.md).
- [Corrected 114-epoch results](experiments/tiny_plateau_min80/RESULTS.md).
- [Original sweep comparison](experiments/sweep/RESULTS.md).
- [Local score benchmark](reports/SCORE_COMPARISON.md): original sweep only;
  it predates cosine/plateau follow-ups.
- [Historical notes](reports/EXPERIMENT_NOTES.md): full decisions and superseded
  paths/policies, retained for provenance. Historical logs/JSON configuration
  snapshots may contain pre-cleanup paths; use this index for current locations.

## Code and commands

Run from the parent `hw2/` directory. Imports and output paths were updated
for the new layout.

```bash
# Fast tests; no training:
.venv/bin/python -m unittest custom_2.code.test_custom_2 -v

# Current training policy: fresh tiny model, minimum 80 epochs, no default cap:
.venv/bin/python custom_2/code/train_tiny_cosine.py --schedule plateau --epochs 80

# Optional explicit safety cap, distinct from early stopping:
.venv/bin/python custom_2/code/train_tiny_cosine.py --schedule plateau --epochs 80 --max-epochs 160

# Re-benchmark the original sweep candidates (overwrites benchmark reports):
.venv/bin/python custom_2/code/benchmark_score.py

# Check the recommended export's structure/budget:
.venv/bin/python export_onnx.py --check custom_2/submission/recommended_tiny.onnx --part part3
```

Existing completed experiment folders are protected against silent overwrites;
use `--output custom_2/experiments/<new_name>` for a genuinely new tiny run.
Training does not automatically update `submission/recommended_tiny.onnx`.
That file is an explicitly selected snapshot, so experiments cannot silently
replace the recommended submission.

## Completed: optimize the actual score, not just accuracy

`code/optimize_score.py` tests four smaller CNNs in
`experiments/score_sweep_v2/`. These retain familiar residual blocks, BatchNorm,
and global pooling (Lectures 13–14); no new exotic operators are introduced.

| Candidate | Learned parameters | Residual widths | Blocks per stage | Head |
|---|---:|---|---:|---|
| mini_gap | 174,933 | 16, 32, 64 | 2 | GAP → Dropout(0.2) → Linear to 5 |
| micro_gap | 98,753 | 12, 24, 48 | 2 | same |
| nano_gap | 44,205 | 8, 16, 32 | 2 | same |
| shallow_gap | 77,717 | 16, 32, 64 | 1 | same |

All four passed forward/backward and temporary ONNX export checks before
training. Nine automated tests pass, including the minimum-LR stopping gate.

Removing the fourth residual stage and the hidden classifier reduces parameters
and computation. Width tests probe the accuracy/size trade-off; the shallow
variant probes whether reducing layer overhead gives a better latency trade-off.
This is an architecture search, not a claim of purely isolated ablations.

All use the same seed-0 split, robust augmentation, batch 64, cross-entropy,
Adam LR 0.001, and weight decay 0.0001. Validation plateaus halve LR to a floor
of 0.00001. Runs have **minimum 80 epochs, no default maximum**, and early stop
only after a fresh 15-stale-epoch minimum-LR window. Best weights are restored.
Unlike older runs, each epoch also saves optimizer/scheduler/RNG state in
`last.pt` for future explicit recovery; automatic resume is not implemented.

```bash
.venv/bin/python -u custom_2/code/optimize_score.py > custom_2/logs/score_optimization.log 2>&1
```

Each candidate export receives the provided ONNX check and paired local CPU
latency measurement against the **externally tested tiny cosine**. Ranking uses:

```text
proxy score gain = (candidate validation mean − tiny cosine validation mean)
                   − 0.05*log10(candidate parameters / tiny cosine parameters)
                   − 0.05*log10(candidate local latency / tiny cosine local latency)
```

This assumes fractional accuracy and that local differences transfer to the
evaluator. Adding the gain to the reference's 0.638 produces a **calibrated
proxy**, not a measured or guaranteed score. Diagnostic stress is reported
separately, not used in ranking. All trials reuse the same 500 validation source
photos; repeated model selection can overfit them. New external submissions
are needed to establish actual improvements.

The confirmed submission is **never automatically replaced**. The sweep writes
its local shortlist to `experiments/score_sweep_v2/RESULTS.md`. Each candidate's
file is `<candidate>/model.onnx`; the user can evaluate the proxy leader before
promoting it. This experiment does not change any existing model's weights.

Completed results:

| Candidate | Exported float weights | 13-view mean | Local latency | Proxy score gain vs tiny cosine |
|---|---:|---:|---:|---:|
| **Mini** | **174,373** | **89.89%** | **0.232 ms** | **+0.02505** |
| Micro | 98,333 | 87.95% | 0.201 ms | +0.02121 |
| Shallow | 77,381 | 86.66% | 0.157 ms | +0.01897 |
| Nano | 43,925 | 84.46% | 0.166 ms | +0.00812 |

All runs trained at least 80 epochs and stopped only after minimum-LR
patience. Mini trained 91 epochs, selected epoch 79, and had 91.00% clean
validation accuracy. Its selection-view accuracy trails tiny cosine by only
0.046 percentage points. Most of the proxy improvement comes from parameter
reduction, not latency. A smaller model was not automatically better: micro,
shallow, and nano lost more accuracy than their additional savings justified
relative to mini in this local comparison. Full results:
[`experiments/score_sweep_v2/RESULTS.md`](experiments/score_sweep_v2/RESULTS.md).
