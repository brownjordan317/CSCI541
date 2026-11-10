"""Train and export the Part 1 Flatten-head baseline."""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from export_onnx import check_onnx, export_onnx, load_lite_cifar


class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 5),
        )

    def forward(self, x):
        x = (x - 0.5) / 0.25
        return self.classifier(self.features(x))


@torch.no_grad()
def validation_accuracy(model, loader, device):
    model.eval()
    correct = 0
    for images, labels in loader:
        predictions = model(images.to(device)).argmax(dim=1)
        correct += (predictions == labels.to(device)).sum().item()
    return correct / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data", type=Path,
        default=Path(__file__).resolve().parent / "pj2_lite_cifar_train.zip",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    images, labels = load_lite_cifar(args.data)
    assert images.shape == (5000, 3, 64, 64)
    assert labels.shape == (5000,)
    indices = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))
    val_indices, train_indices = indices[:500], indices[500:]
    X_val, y_val = images[val_indices], labels[val_indices]
    train_loader = DataLoader(
        TensorDataset(images[train_indices], labels[train_indices]),
        batch_size=64, shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

    model = BaselineCNN().to(device)
    assert sum(p.numel() for p in model.parameters()) == 2_341_061
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_accuracy = -1.0
    best_state = None
    best_epoch = 0
    print(f"Device: {device}; seed: {args.seed}; train: 4500; validation: 500", flush=True)

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0
        for batch_images, batch_labels in train_loader:
            # Independently flip each training image with probability 0.5.
            flip = torch.rand(len(batch_images)) < 0.5
            batch_images[flip] = batch_images[flip].flip(-1)
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_images), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_images)

        accuracy = validation_accuracy(model, val_loader, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        print(
            f"Epoch {epoch:02d}/50  train loss: {total_loss / 4500:.4f}  "
            f"validation accuracy: {accuracy:.4f}",
            flush=True,
        )

    model.load_state_dict(best_state)
    print(
        f"Best validation accuracy: {best_accuracy:.4f} "
        f"({best_accuracy:.2%}), epoch {best_epoch}",
        flush=True,
    )
    export_onnx(model, "baseline_cnn.onnx")
    check_onnx("baseline_cnn.onnx", part="part1", images=X_val, labels=y_val)


if __name__ == "__main__":
    main()
