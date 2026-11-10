# Part 3 best: mini GAP CNN

**Submit [`p3_best.onnx`](p3_best.onnx).** This is the exact challenger-mini
artifact the user identified as best, copied without re-exporting or retraining.
The latest external accuracy/score numbers were not supplied; the local score
proxy in the archived metrics is **not** an actual submission score.

## Contents

```text
p3_best/
├── p3_best.onnx        ← winning submission, byte-identical to challenger mini
├── model.py            ← only the winning architecture
├── train.py            ← standalone single-model trainer
├── training_utils.py   ← original augmentation, validation, LR/stop policy
├── export_onnx.py      ← byte-identical copy of the provided helper
├── requirements.txt
├── checkpoints/
│   ├── best.pt         ← selected epoch 79; weights used by the ONNX
│   └── last.pt         ← epoch 91; model, optimizer, scheduler and RNG state
└── results/
    ├── metrics.json    ← original local metrics, preserved
    ├── history.json    ← full epoch history
    ├── train.log
    └── provenance.json ← original artifact paths and verified SHA-256 hashes
```

All inference/training code is local to this folder; it does not import
`custom/` or `custom_2/`. The dataset remains in the parent folder to avoid
duplicating it. Original experiments are preserved as provenance.

## Architecture and interface

- Input: `image`, RGB float32 `[1,3,64,64]` in `[0,1]`.
- Normalization inside forward: `(x - 0.5) / 0.25`.
- Conv3×3(16) → BatchNorm → ReLU → MaxPool2.
- Three residual stages at widths **16, 32, 64**, **two blocks per stage**.
- New-width stages downsample with stride 2 and project their shortcuts.
- Global average pooling → Flatten → Dropout(0.2) → Linear(64,5).
- Output: `logits`, `[1,5]`; no softmax. Classes: airplane, automobile, frog,
  horse, ship, in that order.
- **174,933 learned parameters; 174,373 exported float weights.**

## Winning run

Seed 0, fixed 4,500/500 source-image split; robust translation/scale/rotation/
color/noise augmentation with a clean-example mixture. Batch 64, cross-entropy,
Adam LR 0.001 and weight decay 0.0001. Validation plateaus halve LR to 0.00001.
Training requires at least 80 epochs and has no default maximum. Early stopping
requires 15 stalled validations after actual training at minimum LR.

The run stopped after **91 epochs** and restored its best checkpoint from
**epoch 79**. A best checkpoint before epoch 80 is compatible with a run that
trained at least 80 epochs.

Local validation: **91.00% clean**, **89.89% 13-view mean**, **88.12% stress mean**.
These are not hidden-set measurements. The user subsequently reported this
challenger was best, without providing its numerical external result.

## Use

From the parent `hw2/` directory, using the existing environment:

```bash
# Check the preserved submission; no training:
.venv/bin/python p3_best/export_onnx.py --check p3_best/p3_best.onnx --part part3

# Test copied artifact hashes, PyTorch/ONNX agreement, and training utilities:
.venv/bin/python -m unittest p3_best.test_model -v

# Reproduce the recipe from scratch; writes only to a NEW run folder:
.venv/bin/python p3_best/train.py --output p3_best/runs/reproduction
```

Or run from this folder with an appropriate Python environment:

```bash
python train.py --data ../pj2_lite_cifar_train.zip --output runs/reproduction
```

The trainer defaults to `runs/new_run`, refuses nonempty output folders, and
never overwrites the packaged winner. `--seed`, `--min-epochs`, `--data`, and
an optional explicit `--max-epochs` are available. There is no automatic resume
mode; `last.pt` retains state for explicit recovery if needed.

Retraining may not reproduce identical bits across different devices/library
versions. Use the packaged ONNX when submitting the exact proven artifact.
