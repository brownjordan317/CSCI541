"""Winning augmentation, validation, and LR policy; no experiment imports."""

import math

import torch
from torch.nn import functional as F


def spatial_transform(images, dx=0.0, dy=0.0, scale=1.0, angle=0.0):
    """Move/resize/rotate about the canvas center, with gray padding."""
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


def robust_augment(images):
    n = len(images)
    flip = torch.rand(n) < 0.5
    images[flip] = images[flip].flip(-1)
    original = images.clone()
    kind = torch.rand(n)
    clean = kind < 0.2
    integer_shift = (kind >= 0.2) & (kind < 0.5)
    scale = 0.65 + 0.7 * torch.rand(n)
    angle = (torch.rand(n) * 2 - 1) * math.pi / 9
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
    blur = torch.rand(n) < 0.1
    if blur.any():
        images[blur] = F.avg_pool2d(F.pad(images[blur], (1, 1, 1, 1), mode="replicate"), 3, 1)
    images += torch.randn_like(images) * (0.03 * torch.rand(n, 1, 1, 1))
    images = images.clamp(0, 1)
    images[clean] = original[clean]
    return images


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


def stress_views(images):
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


class PlateauController:
    def __init__(self, optimizer, minimum=1e-5, lr_patience=4,
                 stop_patience=15, min_delta=0.001):
        self.optimizer = optimizer
        self.minimum = minimum
        self.stop_patience = stop_patience
        self.min_delta = min_delta
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=lr_patience,
            threshold=min_delta, threshold_mode="abs", min_lr=minimum,
        )
        self.floor_reference = None
        self.floor_stale = 0
        self.floor_epochs = 0

    def update(self, score, trained_lr):
        trained_at_floor = math.isclose(trained_lr, self.minimum, rel_tol=1e-8, abs_tol=1e-12)
        if trained_at_floor:
            self.floor_epochs += 1
            if self.floor_reference is None or score > self.floor_reference + self.min_delta:
                self.floor_reference = score
                self.floor_stale = 0
            else:
                self.floor_stale += 1
        else:
            self.floor_reference = None
            self.floor_stale = 0
        self.scheduler.step(score)
        return trained_at_floor and self.floor_stale >= self.stop_patience


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value
