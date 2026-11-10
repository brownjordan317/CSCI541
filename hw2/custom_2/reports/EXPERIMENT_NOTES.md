# Historical experiment notes

These notes preserve the experiment history, including superseded policies
and **paths from before cleanup**. Use [the current index](../README.md) for
current locations and commands. No recorded result was removed during cleanup.

## Custom 2: compact-model and augmentation experiments

Goal: attempt to improve on the original custom model's **user-reported 90.6%
hidden accuracy**, while reducing parameter count. This is a controlled local
search, not a claim that a new model is already better on the hidden set.

All new code and outputs live here. `custom/`, `gap/`, and `export_onnx.py`
remain unchanged. The script imports unchanged residual blocks, original
augmentation, geometry utilities, and evaluation utilities from
`../custom/train_custom.py`; keep that file alongside this experiment.

## What is tested

Four full runs, sequentially on the available accelerator:

| Trial | Residual stage widths | Augmentation | Initial learning rate |
|---|---|---|---:|
| compact_original_aug | 24, 48, 96, 144 | Original custom policy | 0.001 |
| compact_robust_aug | 24, 48, 96, 144 | New mixture policy | 0.001 |
| tiny_robust_aug | 16, 32, 64, 96 | New mixture policy | 0.001 |
| compact_robust_low_lr | 24, 48, 96, 144 | New mixture policy | 0.0005 |

The compact architecture has **1,129,661 learned parameters** (43.2% fewer
than the original's 1,989,029); the tiny architecture has **511,957** (74.3%
fewer). Both passed temporary ONNX export/budget checks before training.
Five automated tests cover reconstruction of the original architecture at
original widths, gradients/shapes/budgets, clean-example preservation,
augmentation bounds, and deterministic stress views. Run them with:

```bash
.venv/bin/python -m unittest custom_2.test_custom_2 -v
```

This gives direct comparisons for augmentation (trials 1/2), width (2/3), and
learning rate (2/4). The original selected custom checkpoint is evaluated on
the same views as a reference, without retraining. Comparing trial 1 with that
reference is not a perfectly isolated width study: the original checkpoint
used a different data-loader RNG coupling and its epoch was already selected.

All candidates retain eight two-convolution residual blocks, BatchNorm,
global average pooling, a 256-unit hidden head, and dropout 0.5. Narrowing
channels reduces most convolutional parameters approximately quadratically,
without removing depth. No pretrained weights, new source images, ensembles,
or test-time augmentation are used.

The methods stay within the prior CNN/ResNet/BatchNorm, regularization,
optimization, and preprocessing themes. Exact coverage of individual
augmentation operations should be checked against lecture slides if required.

## Augmentation changes

The original always geometrically transforms training examples. The new policy
uses a mixture, so training does not consist entirely of interpolated photos:

- All examples may be horizontally flipped with probability 0.5.
- **20%:** otherwise completely clean, with no interpolation/color/noise.
- **30%:** native-resolution integer translations.
- **50%:** scale 0.65–1.35 and rotation ±20°, with translations.
- Movements are constrained so the nominal 32×32 photo remains inside the
  canvas, and never exceed 20 pixels. Padding is gray, not wraparound.
- On non-clean examples: moderate contrast/saturation/brightness jitter,
  grayscale with 10% probability, 3×3 average blur with 10% probability,
  and Gaussian noise up to standard deviation 0.03. Pixel values stay `[0,1]`.

The clean/geometric mixture is the main idea; simply increasing every
augmentation's strength could make the task unrealistic and hurt clean
accuracy. Nevertheless, broader augmentation remains an empirical assumption:
it may improve or worsen generalization and must be evaluated.

## Shared recipe

- Same ZIP, labels, normalization inside forward, and fixed seed-0 source split:
  4,500 train / 500 validation. No validation source enters gradient training.
- Each trial starts from scratch with seed 0, not the previous trained model.
- Batch 64, cross-entropy, Adam, weight decay 0.0001, 80 epochs.
- Learning rate multiplied by 0.2 after epochs 40 and 60.
- Independent seeded data-loader RNG ensures the same training order across
  candidates, irrespective of augmentation RNG consumption.
- Clean validation accuracy every epoch; original 13-view mean evaluated at
  epoch 1, every fifth epoch, and final epoch, to select each trial's checkpoint.
- All float state values are asserted below 3,000,000; the exported graph's
  count is checked separately by the provided helper.

This is a **single-seed** study. Small differences are not evidence of a
statistically established improvement, especially after trying several models
on the same 500 source photos.

## Selection and diagnostics

**Predeclared selection rule:** among the new candidates, choose the smallest
model within **0.3 percentage points** of the best candidate's 13-view mean.
This explicitly trades a tiny validation difference for a material size
reduction. The original remains available as a fallback even if every new
candidate loses to it. The rule does not guarantee improvement over the original.

Each selected trial checkpoint and the original reference also receive the
same 12-view diagnostic stress evaluation: four diagonal shifts, rotations
±20°, scales 0.6/1.5, grayscale, low contrast, fixed Gaussian noise, and blur.
Those scores are **not** used to select checkpoints or the final candidate.
Some stress operations are in the training policy; others use different
parameters/combinations. They are not unseen *source images* beyond the
existing validation split, nor a faithful reconstruction of hidden variants.

Repeated views of 500 photos are correlated, not thousands of independent
examples. Both selection and diagnostic views come from the same validation
source set. The user-reported hidden score is context only; there is no access
to hidden images and no optimization against hidden labels. Actual hidden
accuracy requires a fresh external evaluation.

## Running and artifacts

From `hw2/`:

```bash
.venv/bin/python -u custom_2/train_custom_2.py > custom_2/sweep.log 2>&1
```

The four 80-epoch runs take substantially longer than one training run; they
run sequentially to avoid contention on the same GPU. Available controls:

```bash
# A specific trial (writes the final model from that trial):
.venv/bin/python custom_2/train_custom_2.py --trials compact_robust_aug

# Reuse completed trials after an interrupted sweep:
.venv/bin/python custom_2/train_custom_2.py --reuse-completed
```

Completed trials are not overwritten silently. Reuse checks config, seed, and
epoch count, but does not hash dataset contents or code; only reuse unchanged
experiments. An incomplete trial restarts from scratch rather than resuming
optimizer state. Per-trial logs append if an incomplete trial is restarted.

Outputs:

- `runs/<trial>/best.pt`, `train.log`, `history.json`, `metrics.json`.
- `reference_metrics.json`: original model evaluated under this protocol.
- `comparison.json` and `RESULTS.md`: side-by-side results and selected trial.
- `custom_2_cnn.pt`: selected candidate's weights/config, not a full optimizer resume.
- `custom_2_cnn.onnx`: selected candidate, opset 17, fixed `[1,3,64,64]` input,
  five logits; checked with `check_onnx(part="part3")`.
- `sweep.log`: combined console output, including final ONNX checks.

Keep `custom/custom_cnn.onnx`: a smaller candidate is not automatically a
better classifier, and the established 90.6% hidden result remains valuable.

## Follow-up: train a late-improving candidate longer

The initial sweep finished with `compact_robust_low_lr` achieving its highest
13-view validation mean at epoch 80. The other three candidates peaked at
epochs 65–70 and did not show sustained late validation gains. Only the
lower-learning-rate candidate is extended, rather than spending more training
on every model because training losses fell.

`extend_training.py` trains that candidate for epochs 81–120. The original
checkpoint contains weights but no optimizer state, so it explicitly starts
**fresh Adam** at the final learning rate 0.00002, then reduces to 0.000004
after epoch 100. This is weight-only continuation, not an exact resume. The
same source-image split and 13-view selection metric are retained; diagnostic
stress scores do not select the checkpoint. Epoch 80 remains eligible, so
the saved result cannot lose to it on that selection score.

```bash
.venv/bin/python -u custom_2/extend_training.py > custom_2/extension.log 2>&1
```

Outputs live in `extended_low_lr/`. Only if validation mean improves, the
checked extended export replaces the top-level `custom_2_cnn.onnx` and its
weights; previous top-level files are backed up inside `extended_low_lr/`.
The initial sweep's `comparison.json` stays an 80-epoch record; extension
metrics are in `extended_low_lr/metrics.json` and appended to `RESULTS.md`.
Do not rerun the initial sweep after promotion unless you intend to replace
the extended top-level artifact with an initial-sweep checkpoint again.

## Score-aware comparison

The user subsequently supplied the grading formula:

```text
score = hidden_accuracy - 0.05*log10(params) - 0.05*log10(median_latency_ms)
```

Accuracy units matter: 0.906 and 90.6 produce radically different trade-offs.
`benchmark_score.py` reports both numerical interpretations in JSON, but its
Markdown ranking explicitly assumes fractional accuracy. It exports every
candidate to `runs/<trial>/model.onnx`, evaluates local single-thread ONNX
Runtime CPU latency, and combines that latency with the existing 13-view
validation mean as a **score proxy**, not an actual grader score.

```bash
.venv/bin/python -u custom_2/benchmark_score.py > custom_2/benchmark.log 2>&1
```

Results: `SCORE_COMPARISON.md` and `score_benchmark.json`. Local measurements
are hardware/thread/runtime dependent. The evaluator's parameter-count
definition and latency configuration also need confirmation. The timing loop
excludes preprocessing, loading, and session construction.

The original sweep's top-level `custom_2_cnn.onnx` is **accuracy-selected**;
benchmarking does not silently replace it. The smaller score-proxy leader can
be submitted separately as `runs/tiny_robust_aug/model.onnx` if fractional
accuracy is confirmed. Its actual hidden accuracy is still unknown. The
40-epoch low-LR extension script is prepared but, at the time of adding this
section, has not been launched pending clarification of score units.

## Tiny model with tapered LR and early stopping

`train_tiny_cosine.py` starts the same 511,957-parameter tiny architecture
from scratch, with the same seed, source-image split, robust augmentation,
batch size, dropout, weight decay, and Adam optimizer as the tiny sweep trial.
It replaces stepwise LR reductions with a smooth epoch-dependent cosine:

```text
lr(epoch) = 0.00001 + 0.5*(0.001 - 0.00001)
                     * (1 + cos(pi*(epoch - 1)/(80 - 1)))
```

Epoch 1 starts at 0.001 and epoch 80, if reached, uses 0.00001. The schedule
does not restart or depend on noisy batch loss. Training lasts at most 80
epochs. Early stopping monitors the same 13-view validation mean **every
epoch**, with patience 15, minimum improvement 0.001 (0.1 percentage points),
and a minimum of 40 training epochs. Patience can accumulate before epoch
40, but stopping cannot occur before then. It resets when the score improves
by more than min-delta relative to its last significant-improvement reference;
the exact highest-score checkpoint is saved even for smaller improvements.
Best weights are restored before export. Diagnostic stress is evaluated only
after selection and does not control stopping.

```bash
.venv/bin/python -u custom_2/train_tiny_cosine.py > custom_2/tiny_cosine.log 2>&1
```

Outputs go to `tiny_cosine_earlystop/`: `best.pt`, `history.json`, `metrics.json`,
`RESULTS.md`, and **`tiny_cosine_cnn.onnx`**. Previous exports are not replaced.
Completed or checkpointed runs are not overwritten silently. CLI options
control epochs (maximum 80), seed, patience, min-epochs, min-delta, and data.

Evaluating every epoch provides more selection opportunities than the earlier
five-epoch protocol, so any comparison is not a pure schedule-only statistical
test. The run also reports local CPU latency and a score proxy explicitly
assuming fractional accuracy; neither is a hidden/evaluator result.

### Correction: reduce LR before permitting early stopping

The original cosine experiment stopped at epoch 71 at LR **0.0000413664**,
before reaching its 0.00001 floor. That historical result is preserved, but
its early-stop policy did not match the requested ordering. The trainer now
gates stopping on actual training at minimum LR for **both** scheduler modes.
With pure cosine over the full budget, that means the epoch cap, rather than
early stopping, normally ends the run.

The new `--schedule plateau` mode responds directly to stalled validation:

- Start at LR 0.001; use `ReduceLROnPlateau(mode="max")` on 13-view mean.
- Halve LR after five consecutive validations without an improvement exceeding
  0.001 (0.1 percentage points). `--lr-patience 4` is the PyTorch setting that
  gives five bad epochs before a reduction.
- Never reduce below 0.00001.
- Do not accumulate early-stop patience while training above that floor.
- After an epoch actually trained at the floor, begin a fresh 15-stale-epoch
  stopping window; significant improvements reset it. The validation that
  first reduces the next epoch's LR to the floor does not itself count.
- Retain the 80-epoch maximum as an independent compute limit. If LR has not
  reached the floor by epoch 80, report a **budget stop**, not early stopping.

Run from scratch, retaining the same tiny architecture/data/augmentation:

```bash
.venv/bin/python -u custom_2/train_tiny_cosine.py --schedule plateau > custom_2/tiny_plateau.log 2>&1
```

Outputs are separate in `tiny_plateau_earlystop/`, including
`tiny_plateau_cnn.onnx`, `history.json`, and `metrics.json`. History records
the LR actually used and the LR for the next epoch, plus minimum-LR patience.
Tests explicitly verify that reductions precede stopping and that improvement
at minimum LR resets patience. Existing cosine-run artifacts are unchanged;
the old historical policy above describes that earlier run, not the corrected
code's stopping behavior.

### Final clarification: 80 is a minimum, not a maximum

The user clarified that **80 epochs is a minimum**. Current trainer behavior
supersedes the earlier capped-run descriptions above:

- `--epochs 80` (alias `--min-epochs 80`) means train at least 80 epochs.
- There is **no maximum by default**. Continue until minimum-LR patience is
  exhausted and the minimum epoch count has been met.
- Plateau mode reduces LR when progress stalls. Early stopping remains disabled
  above 0.00001; a fresh 15-stale-epoch window is measured at that floor.
- Cosine mode, if used, tapers over the first 80 epochs and stays at its floor
  afterward. It does not rise again or stop automatically at epoch 80.
- An optional `--max-epochs` safety cap exists only when explicitly supplied;
  it must be at least the minimum and is reported separately from early stopping.

The corrected fresh run uses:

```bash
.venv/bin/python -u custom_2/train_tiny_cosine.py --schedule plateau --epochs 80 > custom_2/tiny_plateau_min80.log 2>&1
```

Outputs: `tiny_plateau_min80/`, including `tiny_plateau_cnn.onnx`. Prior cosine
and capped plateau artifacts remain untouched. A fresh run is necessary for
an uninterrupted optimizer trajectory: the earlier capped run saved its best
weights, not its final optimizer state, so it cannot be exactly resumed.
