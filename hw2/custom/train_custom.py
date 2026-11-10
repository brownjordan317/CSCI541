"""Part 3: residual CNN with GAP and position/scale augmentation.

Derived from gap/train_gap.py; see README.md for the changes and rationale.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from export_onnx import check_onnx, export_onnx, load_lite_cifar


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = nn.Identity() if in_channels == out_channels and stride == 1 else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.main(x) + self.skip(x))


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, 2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 128),
            ResidualBlock(128, 192, 2),
            ResidualBlock(192, 192),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5),
        )

    def forward(self, x):
        return self.classifier(self.features((x - 0.5) / 0.25))


def spatial_transform(images, dx=0.0, dy=0.0, scale=1.0, angle=0.0):
    """Move/resize/rotate content about the canvas center; fill with gray.

    Runs on CPU, outside the exported model. Positive dx/dy move content
    right/down. grid_sample needs the inverse, output-to-input mapping.
    """
    n = len(images)
    def vector(value):
        return torch.as_tensor(value, dtype=images.dtype).expand(n)
    scale, angle = vector(scale), vector(angle)
    c, s = torch.cos(angle) / scale, torch.sin(angle) / scale
    tx, ty = 2 * vector(dx) / images.shape[-1], 2 * vector(dy) / images.shape[-2]
    theta = torch.zeros(n, 2, 3, dtype=images.dtype)
    theta[:, 0, 0], theta[:, 0, 1] = c, s
    theta[:, 1, 0], theta[:, 1, 1] = -s, c
    theta[:, 0, 2] = -c * tx - s * ty
    theta[:, 1, 2] = s * tx - c * ty
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images - 0.5, grid, align_corners=False, padding_mode="zeros") + 0.5


def augment(images):
    n = len(images)
    flip = torch.rand(n) < 0.5
    images[flip] = images[flip].flip(-1)
    # Scale 0.75–1.25, rotate ±15 degrees, and shift without clipping
    # the nominal 32×32 photo's bounding box.
    scale = 0.75 + 0.5 * torch.rand(n)
    angle = (torch.rand(n) * 2 - 1) * math.pi / 12
    half_extent = 16 * scale * (angle.cos().abs() + angle.sin().abs())
    max_shift = (31 - half_extent).clamp(min=0, max=16)
    dx = (torch.rand(n) * 2 - 1) * max_shift
    dy = (torch.rand(n) * 2 - 1) * max_shift
    images = spatial_transform(images, dx, dy, scale, angle)
    # Mild color variation and noise; leave 20% of examples unaltered here.
    apply = (torch.rand(n, 1, 1, 1) < 0.8).float()
    contrast = 1 + apply * (torch.rand(n, 1, 1, 1) * 0.4 - 0.2)
    brightness = apply * (torch.rand(n, 1, 1, 1) * 0.2 - 0.1)
    images = (images - 0.5) * contrast + 0.5 + brightness
    saturation = 1 + apply * (torch.rand(n, 1, 1, 1) * 0.4 - 0.2)
    gray = images.mean(dim=1, keepdim=True)
    images = gray + saturation * (images - gray)
    images += torch.randn_like(images) * (0.02 * torch.rand(n, 1, 1, 1))
    return images.clamp(0, 1)


def validation_views(images):
    views = {"center": images}
    for distance in (8, 16):
        for name, dx, dy in (("left", -distance, 0), ("right", distance, 0),
                             ("up", 0, -distance), ("down", 0, distance)):
            views[f"{name}_{distance}"] = spatial_transform(images, dx, dy)
    for scale in (0.75, 1.25):
        views[f"scale_{scale}"] = spatial_transform(images, scale=scale)
    views["darker"] = (images - 0.15).clamp(0, 1)
    views["brighter"] = (images + 0.15).clamp(0, 1)
    return views


@torch.no_grad()
def predict(model, images, device):
    model.eval()
    return torch.cat([
        model(batch.to(device)).argmax(1).cpu()
        for batch in images.split(64)
    ])


@torch.no_grad()
def evaluate_views(model, views, labels, device):
    predictions = torch.stack([predict(model, images, device) for images in views.values()])
    accuracies = {
        name: (prediction == labels).float().mean().item()
        for name, prediction in zip(views, predictions)
    }
    return {
        "view_accuracy": accuracies,
        "mean_accuracy": sum(accuracies.values()) / len(accuracies),
        "all_views_same_prediction": (predictions == predictions[0]).all(0).float().mean().item(),
        "all_views_correct": (predictions == labels.unsqueeze(0)).all(0).float().mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=ROOT / "pj2_lite_cifar_train.zip")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    images, labels = load_lite_cifar(args.data)
    assert images.shape == (5000, 3, 64, 64) and labels.shape == (5000,)
    indices = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))
    X_val, y_val = images[indices[:500]], labels[indices[:500]]
    loader = DataLoader(TensorDataset(images[indices[500:]], labels[indices[500:]]),
                        batch_size=64, shuffle=True)
    views = validation_views(X_val)
    model = CustomCNN().to(device)
    params = sum(p.numel() for p in model.parameters())
    # Include BN running statistics conservatively; the exported count is
    # checked separately by the provided helper (BN may be fused into Conv).
    float_values = sum(t.numel() for t in model.state_dict().values() if t.is_floating_point())
    assert float_values <= 3_000_000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=sorted(set([max(1, args.epochs // 2), max(1, 3 * args.epochs // 4)])),
        gamma=0.2,
    )
    criterion = nn.CrossEntropyLoss()
    best_score, best_epoch, best_state = -1.0, 0, None
    print(f"Device: {device}; seed: {args.seed}; parameters: {params:,}; "
          f"float state values: {float_values:,}; train: 4500; validation: 500", flush=True)
    print(f"Selection: mean accuracy across {len(views)} fixed validation views, "
          "evaluated every 5 epochs and at epoch 1/final. Not hidden accuracy.", flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        lr = optimizer.param_groups[0]["lr"]
        for batch_images, batch_labels in loader:
            batch_images = augment(batch_images).to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_images), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_images)
        scheduler.step()
        clean = (predict(model, X_val, device) == y_val).float().mean().item()
        record = {"epoch": epoch, "train_loss": total_loss / 4500,
                  "learning_rate": lr, "clean_accuracy": clean}
        print(f"Epoch {epoch:02d}/{args.epochs}  lr: {lr:.6f}  "
              f"train loss: {record['train_loss']:.4f}  clean accuracy: {clean:.4f}", flush=True)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            metrics = evaluate_views(model, views, y_val, device)
            record["robustness_proxy"] = metrics
            score = metrics["mean_accuracy"]
            print(f"  proxy mean accuracy: {score:.4f}; all-view prediction consistency: "
                  f"{metrics['all_views_same_prediction']:.4f}", flush=True)
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save({"model_state_dict": best_state, "epoch": epoch,
                            "seed": args.seed, "metrics": metrics}, OUT / "custom_cnn.pt")
        history.append(record)

    model.load_state_dict(best_state)
    metrics = evaluate_views(model, views, y_val, device)
    report = {"seed": args.seed, "epochs": args.epochs, "parameters": params,
              "float_state_values": float_values, "best_epoch": best_epoch,
              "selection_metric": "mean accuracy over 13 fixed validation views",
              "metrics": metrics, "history": history,
              "warning": "Local validation proxies, not public or hidden evaluation results."}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Selected epoch {best_epoch}; final local validation metrics:\n"
          f"{json.dumps(metrics, indent=2)}", flush=True)
    output_path = OUT / "custom_cnn.onnx"
    export_onnx(model, output_path)
    check_onnx(output_path, part="part3", images=X_val, labels=y_val)


if __name__ == "__main__":
    main()
