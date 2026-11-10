"""Controlled compact-model/augmentation sweep derived from custom/train_custom.py."""

import argparse
import gc
import json
import math
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "custom_2" / "experiments" / "sweep"
sys.path.insert(0, str(ROOT))
from export_onnx import check_onnx, export_onnx, load_lite_cifar
# Reuse unchanged building blocks and the exact original validation protocol.
# No files in custom/ are written by this script.
from custom.train_custom import (
    CustomCNN, ResidualBlock, augment as original_augment,
    evaluate_views, predict, spatial_transform, validation_views,
)


@dataclass(frozen=True)
class Config:
    name: str
    widths: tuple
    augmentation: str
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.5


CONFIGS = (
    Config("compact_original_aug", (24, 48, 96, 144), "original"),
    Config("compact_robust_aug", (24, 48, 96, 144), "robust"),
    Config("tiny_robust_aug", (16, 32, 64, 96), "robust"),
    Config("compact_robust_low_lr", (24, 48, 96, 144), "robust", lr=5e-4),
)


class CompactCNN(nn.Module):
    """Same residual-block counts and GAP head as custom; narrower channels."""

    def __init__(self, widths=(24, 48, 96, 144), dropout=0.5):
        super().__init__()
        layers = [nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                  nn.BatchNorm2d(widths[0]), nn.ReLU(), nn.MaxPool2d(2)]
        previous = widths[0]
        for i, width in enumerate(widths):
            layers += [ResidualBlock(previous, width, 1 if i == 0 else 2),
                       ResidualBlock(width, width)]
            previous = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(widths[-1], 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 5),
        )

    def forward(self, x):
        return self.classifier(self.features((x - 0.5) / 0.25))


def robust_augment(images):
    """Mixture of clean examples, integer shifts, and broader affine/color views."""
    n = len(images)
    # Keep 20% of examples fully clean (apart from optional horizontal flip).
    flip = torch.rand(n) < 0.5
    images[flip] = images[flip].flip(-1)
    original = images.clone()
    kind = torch.rand(n)
    clean = kind < 0.2
    integer_shift = (kind >= 0.2) & (kind < 0.5)

    # 30% receive native-resolution integer shifts rather than always being
    # rotated/rescaled. This avoids training exclusively on smoothed photos.
    scale = 0.65 + 0.7 * torch.rand(n)
    angle = (torch.rand(n) * 2 - 1) * math.pi / 9  # ±20 degrees
    scale[integer_shift | clean] = 1
    angle[integer_shift | clean] = 0
    half_extent = 16 * scale * (angle.cos().abs() + angle.sin().abs())
    max_shift = (31.5 - half_extent).clamp(min=0, max=20)
    dx = (torch.rand(n) * 2 - 1) * max_shift
    dy = (torch.rand(n) * 2 - 1) * max_shift
    dx[integer_shift] = dx[integer_shift].round()
    dy[integer_shift] = dy[integer_shift].round()
    dx[clean], dy[clean] = 0, 0
    images = spatial_transform(images, dx, dy, scale, angle)

    apply = (torch.rand(n, 1, 1, 1) < 0.8).float()
    contrast = 1 + apply * (torch.rand(n, 1, 1, 1) * 0.6 - 0.3)
    brightness = apply * (torch.rand(n, 1, 1, 1) * 0.3 - 0.15)
    images = (images - 0.5) * contrast + 0.5 + brightness
    gray = images.mean(1, keepdim=True)
    saturation = 1 + apply * (torch.rand(n, 1, 1, 1) * 0.8 - 0.4)
    saturation[torch.rand(n) < 0.1] = 0
    images = gray + saturation * (images - gray)
    # Low-probability mild blur, rather than blurring every example.
    blur = torch.rand(n) < 0.1
    if blur.any():
        images[blur] = F.avg_pool2d(F.pad(images[blur], (1, 1, 1, 1), mode="replicate"), 3, 1)
    images += torch.randn_like(images) * (0.03 * torch.rand(n, 1, 1, 1))
    images = images.clamp(0, 1)
    images[clean] = original[clean]
    return images


def stress_views(images):
    """Fixed diagnostic suite, never used to select checkpoints or trials."""
    views = {}
    for name, dx, dy in (("top_left", -12, -12), ("top_right", 12, -12),
                         ("bottom_left", -12, 12), ("bottom_right", 12, 12)):
        views[name] = spatial_transform(images, dx=dx, dy=dy)
    for degrees in (-20, 20):
        views[f"rotation_{degrees}"] = spatial_transform(images, angle=math.radians(degrees))
    for scale in (0.6, 1.5):
        views[f"scale_{scale}"] = spatial_transform(images, scale=scale)
    views["grayscale"] = images.mean(1, keepdim=True).expand_as(images).contiguous()
    views["low_contrast"] = ((images - 0.5) * 0.6 + 0.5).clamp(0, 1)
    generator = torch.Generator().manual_seed(193)
    views["noise_0.04"] = (images + torch.randn(images.shape, generator=generator) * 0.04).clamp(0, 1)
    views["blur_3"] = F.avg_pool2d(F.pad(images, (1, 1, 1, 1), mode="replicate"), 3, 1)
    return views


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def clear_device(device):
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def train_trial(config, args, train_images, train_labels, X_val, y_val, views, device):
    directory = OUT / "runs" / config.name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "metrics.json").exists():
        raise FileExistsError(f"{directory} already contains a completed run; refusing to overwrite")
    torch.manual_seed(args.seed)
    # Independent loader RNG prevents augmentation RNG consumption from changing
    # batch order between augmentation ablations. Different widths still have
    # different initializations; a seed is not a multi-seed significance test.
    loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=64,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    model = CompactCNN(config.widths, config.dropout).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    float_state = sum(t.numel() for t in model.state_dict().values() if t.is_floating_point())
    assert float_state <= 3_000_000
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    milestones = sorted(set([max(1, args.epochs // 2), max(1, 3 * args.epochs // 4)]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.2)
    criterion = nn.CrossEntropyLoss()
    augmentation = original_augment if config.augmentation == "original" else robust_augment
    best_score, best_epoch, history = -1, 0, []
    def log(message):
        print(f"[{config.name}] {message}", flush=True)
        with (directory / "train.log").open("a") as stream:
            stream.write(message + "\n")
    log(f"config={asdict(config)}; parameters={parameters:,}; float state={float_state:,}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        lr = optimizer.param_groups[0]["lr"]
        for images, labels in loader:
            images, labels = augmentation(images).to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        scheduler.step()
        clean_accuracy = (predict(model, X_val, device) == y_val).float().mean().item()
        record = {"epoch": epoch, "learning_rate": lr, "loss": total_loss / len(train_labels),
                  "clean_accuracy": clean_accuracy}
        log(f"Epoch {epoch:02d}/{args.epochs}; loss={record['loss']:.4f}; "
            f"clean={clean_accuracy:.4f}; lr={lr:.6f}")
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            metrics = evaluate_views(model, views, y_val, device)
            record["validation"] = metrics
            score = metrics["mean_accuracy"]
            log(f"13-view mean={score:.4f}; consistency={metrics['all_views_same_prediction']:.4f}")
            if score > best_score:
                best_score, best_epoch = score, epoch
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"config": asdict(config), "model_state_dict": state,
                            "epoch": epoch, "seed": args.seed, "metrics": metrics}, directory / "best.pt")
        history.append(record)
        dump_json(directory / "history.json", history)
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    report = {"config": asdict(config), "epochs": args.epochs, "seed": args.seed,
              "parameters": parameters, "float_state_values": float_state,
              "best_epoch": best_epoch, "validation": checkpoint["metrics"]}
    # Report stress only after selection of the epoch, never use it as a score.
    diagnostics = stress_views(X_val)
    report["diagnostic_stress"] = evaluate_views(model, diagnostics, y_val, device)
    dump_json(directory / "metrics.json", report)
    log(f"Selected epoch {best_epoch}, 13-view mean {best_score:.4f}")
    del model, optimizer, scheduler, diagnostics, checkpoint
    clear_device(device)
    return report


def compare_reference(args, X_val, y_val, views, device):
    path = ROOT / "custom" / "custom_cnn.pt"
    if not path.exists():
        raise FileNotFoundError(f"Original reference checkpoint is required: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = CustomCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    reference = {"name": "custom_reference", "parameters": sum(p.numel() for p in model.parameters()),
                 "reported_hidden_accuracy": 0.906,
                 "validation": evaluate_views(model, views, y_val, device),
                 "diagnostic_stress": evaluate_views(model, stress_views(X_val), y_val, device)}
    dump_json(OUT / "reference_metrics.json", reference)
    del model, checkpoint
    clear_device(device)
    return reference


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "pj2_lite_cifar_train.zip")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", nargs="+", choices=[c.name for c in CONFIGS],
                        default=[c.name for c in CONFIGS])
    parser.add_argument("--reuse-completed", action="store_true",
                        help="Explicitly reuse completed trials with matching config/seed/epochs")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    device = get_device()
    print(f"Device: {device}; {len(args.trials)} sequential trials × {args.epochs} epochs", flush=True)
    images, labels = load_lite_cifar(args.data)
    assert images.shape == (5000, 3, 64, 64) and labels.shape == (5000,)
    indices = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))
    X_val, y_val = images[indices[:500]], labels[indices[:500]]
    train_images, train_labels = images[indices[500:]], labels[indices[500:]]
    del images, labels
    views = validation_views(X_val)
    reference = compare_reference(args, X_val, y_val, views, device)
    reports = []
    for config in CONFIGS:
        if config.name not in args.trials:
            continue
        existing = OUT / "runs" / config.name / "metrics.json"
        if args.reuse_completed and existing.exists():
            report = json.loads(existing.read_text())
            expected_config = json.loads(json.dumps(asdict(config)))
            if report["config"] != expected_config or report["seed"] != args.seed or report["epochs"] != args.epochs:
                raise ValueError(f"Completed run configuration does not match: {existing}")
        else:
            report = train_trial(config, args, train_images, train_labels, X_val, y_val, views, device)
        reports.append(report)
    # Predeclared trade-off: smallest model within 0.3 percentage points of
    # the best candidate's validation mean. Never use hidden or stress scores.
    best_mean = max(r["validation"]["mean_accuracy"] for r in reports)
    contenders = [r for r in reports if r["validation"]["mean_accuracy"] >= best_mean - 0.003]
    winner = min(contenders, key=lambda r: (r["parameters"], -r["validation"]["mean_accuracy"]))
    name = winner["config"]["name"]
    checkpoint_path = OUT / "runs" / name / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = CompactCNN(tuple(winner["config"]["widths"]), winner["config"]["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    shutil.copy2(checkpoint_path, OUT / "custom_2_cnn.pt")
    summary = {"selection_rule": "smallest candidate within 0.003 of best 13-view validation mean",
               "selected": name, "reference": reference, "trials": reports,
               "hidden_accuracy": None,
               "warning": "Validation/model selection and diagnostic proxies; not an independent hidden-set result."}
    dump_json(OUT / "comparison.json", summary)
    lines = ["# Custom 2 experiment results", "",
             "All values below are local validation proxies, not hidden accuracy.", "",
             "| Model | Learned parameters | Clean | 13-view mean | 13-view consistency | Diagnostic stress mean |",
             "|---|---:|---:|---:|---:|---:|"]
    for label, report in [("Original custom", reference)] + [(r["config"]["name"], r) for r in reports]:
        val = report["validation"]
        lines.append(f"| {label} | {report['parameters']:,} | {val['view_accuracy']['center']:.2%} | "
                     f"{val['mean_accuracy']:.2%} | {val['all_views_same_prediction']:.2%} | "
                     f"{report['diagnostic_stress']['mean_accuracy']:.2%} |")
    lines += ["", f"Selected candidate: **{name}**.",
              "Smallest candidate within 0.3 percentage points of the best candidate's 13-view mean.",
              "Stress results and reported hidden scores do not enter selection.", "",
              "The original model's 90.6% hidden accuracy was supplied by the user.",
              "The new model requires external evaluation; retain the original as a fallback.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)
    output = OUT / "custom_2_cnn.onnx"
    export_onnx(model, output)
    check_onnx(output, part="part3", images=X_val, labels=y_val)


if __name__ == "__main__":
    main()
