# Custom model: robustness-oriented Part 3

`train_custom.py` starts from the Part 2 workflow in `../gap/train_gap.py`:
the same supplied data loader, five labels, seed-0 4,500/500 source-image
split, input normalization inside the model, global-pooling head, and provided
ONNX helpers. The original baseline, GAP trainer, and export helper are untouched.

## Goal and limits

Improve generalization to transformed images, especially position changes.
The baseline's reported 54.5% hidden accuracy and 4.0% all-variant consistency
motivate this work; they are **not** measurements of this custom model.
Hidden images and their precise transformation distribution are unavailable.
No improvement in actual hidden accuracy can be guaranteed from local results.

Lecture headlines establish broad topics, not every technique in the slides.
The design uses conventional methods within the listed topics; if your course
requires a specific slide citation for augmentation types or learning-rate
schedules, confirm those against the actual decks before submission.

## Changes and lecture connections

| Change | Reason | Relevant lecture topic |
|---|---|---|
| Deeper convolutional feature extractor | Learn richer features than the four-convolution backbone | 11–12: CNNs, deeper CNNs |
| Residual connections | Help optimization through identity/projection paths | 14: ResNet |
| BatchNorm after convolutions | Stabilize activation scales during training; uses fixed running statistics at inference | 14: BatchNorm |
| Global average pooling retained | Reduce explicit reliance on feature position | 13: NiN/GoogLeNet |
| Dropout 0.5 retained; Adam weight decay 0.0001 added | Regularization with a small training dataset | 10: generalization/regularization |
| Horizontal flips, translations, modest resizing/rotation, mild color/noise augmentation | Train on label-preserving variation instead of only centered photos | 3 and 10: preprocessing/generalization; confirm detailed coverage in decks |
| 80 epochs; scheduled learning-rate reductions | Allow learning from augmented examples, then refine with smaller updates | Optimization/training pipeline; confirm schedule coverage in decks |

No pretrained data, external images, attention, transformers, distillation,
test-time augmentation, or ensembles are used. One ordinary CNN is exported.

## Architecture

Input `[B,3,64,64]` → `(x - 0.5) / 0.25` → Conv3×3(32), BN, ReLU,
MaxPool2 → two residual blocks at each stage width **32, 64, 128, 192**.
The first block of each new width downsamples with stride 2. A block has two
3×3 convolutions with BN; a 1×1 convolution and BN project the shortcut when
the width or stride changes. ReLU follows the addition.

The final features are `[B,192,4,4]`. The head is:

```text
AdaptiveAvgPool2d(1) → Flatten → Linear(192,256)
→ ReLU → Dropout(0.5) → Linear(256,5)
```

Convolutions preceding BN omit biases. BN supplies a learned shift and scale.
Training asserts that all float state values, including running statistics,
fit under 3,000,000. The provided ONNX helper checks the exported initializer
count as well; BN fusion can change that count from the PyTorch count.

Pre-training checks passed: **1,989,029 learned parameters**, **1,993,189 float
state values**, valid forward/backward passes, augmentation geometry/range
checks, and successful temporary ONNX export. The untrained smoke export
reported 1,985,285 float initializer values; the trained export is checked again
because fusion and initializer sharing can affect the final count.

## Augmentation details

Only the 4,500 training source images are augmented for gradient updates.
Each batch receives independent random transformations:

- Horizontal flip with probability 0.5.
- Scale between 0.75× and 1.25×.
- Rotation between −15° and +15°.
- Horizontal/vertical movement up to 16 pixels, constrained by the transformed
  central photo's bounding box so that the nominal photo stays on the canvas.
- Contrast/saturation factors 0.8–1.2 and brightness offsets ±0.1, applied to
  80% of examples; Gaussian noise with standard deviation sampled from 0–0.02.
- Clamp pixels to `[0,1]`.

Geometric transforms use bilinear interpolation with gray padding, not black
padding or wraparound. Photometric transforms subsequently affect the canvas
too, deliberately varying its appearance. There is no fixed center crop at
inference: the whole input is classified.

These transformations are assumptions about useful invariances, not knowledge
of the hidden set. Excessively strong or inappropriate augmentations can harm
accuracy; this is a reason to retain clean-image validation too.

## Recipe and checkpoint selection

- Seed 0 by default; same fixed split as Parts 1 and 2.
- Batch 64, cross-entropy on raw logits.
- Adam, initial learning rate 0.001, weight decay 0.0001.
- Default 80 epochs; multiply learning rate by 0.2 after epochs 40 and 60.
- Print training loss and clean validation accuracy each epoch.
- Evaluate robustness at epoch 1, every fifth epoch, and the final epoch.
- Save the checkpoint with highest **mean validation accuracy across 13 views**:
  center, shifts ±8 and ±16 in each axis, scales 0.75/1.25, brightness ±0.15.
- Restore that checkpoint before export. Ties retain the earlier checkpoint.

The 13 views are generated from the same 500 held-out source images. They are
not 6,500 independent photos, and none is used in gradient updates. Because
they select the checkpoint, they remain validation data, not an unbiased test.

The report includes accuracy per view, their equally weighted mean,
the fraction of photos with identical predictions across **these 13 views**,
and the fraction correct on all views. Consistency alone is not accuracy:
a model can consistently predict the wrong class.

These views do not reproduce the evaluator's unknown variants. Their
consistency score is not directly comparable with the reported 4.0% hidden
consistency. A higher robustness proxy is not proof of better hidden accuracy.

## Run and outputs

From the parent `hw2/` directory:

```bash
.venv/bin/python -u custom/train_custom.py > custom/train_custom.log 2>&1
```

Outputs in this folder:

- `custom_cnn.onnx`: fixed `[1,3,64,64]` input and five logits, opset 17.
- `custom_cnn.pt`: selected PyTorch weights, including BN statistics and metrics.
- `metrics.json`: configuration summary, per-epoch history, selected metrics.
- `train_custom.log`: training and `check_onnx(part="part3")` output.

The `.pt` file does not include Adam state and is not a full resume checkpoint.
Re-running overwrites these outputs. Use `--seed`, `--data`, or `--epochs` to
override defaults; changing epoch count also adjusts the schedule milestones.

Actual public/hidden improvements must be measured by submitting the exported
model. Fair local comparisons should evaluate all models on the same views;
do not compare this model's 13-view mean with a baseline's center-only accuracy.
