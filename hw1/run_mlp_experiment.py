"""Controlled MLP variants of the sub_5 run; no server submission is made."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sample_mlp import (
    BUDGET,
    CopybookDirectory,
    N_CLASSES,
    export_mlp,
    grouped_holdout,
)


VARIANTS = {
    "baseline": {"pool": 4, "first_width": 128, "second_width": 64,
                 "initialization": "xavier", "changed_factor": "reference sub_5 configuration"},
    "pool8": {"pool": 8, "first_width": 128, "second_width": 64,
              "initialization": "xavier", "changed_factor": "pooling factor 4 to 8"},
    "width96": {"pool": 4, "first_width": 96, "second_width": 64,
                "initialization": "xavier", "changed_factor": "first hidden width 128 to 96"},
    "kaiming": {"pool": 4, "first_width": 128, "second_width": 64,
                "initialization": "kaiming", "changed_factor": "Xavier to Kaiming initialization"},
}
BASELINE = {"pool": 4, "first_width": 128, "second_width": 64,
            "initialization": "xavier"}
LR = 0.01
MOMENTUM = 0.9
BATCH = 256
EPOCHS = 25
SEED = 0
SPLIT_SEED = 42


class VariantMLP(nn.Module):
    def __init__(self, pool, first_width, second_width):
        super().__init__()
        self.pool = pool
        self.fc1 = nn.Linear((64 // pool) ** 2, first_width)
        self.fc2 = nn.Linear(first_width, second_width)
        self.fc3 = nn.Linear(second_width, N_CLASSES)

    def forward(self, x):
        x = (0.5 - F.avg_pool2d(x, self.pool)) * 2.0
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def make_model(config):
    torch.manual_seed(SEED)
    model = VariantMLP(config["pool"], config["first_width"],
                       config["second_width"])
    for layer in (model.fc1, model.fc2, model.fc3):
        if config["initialization"] == "xavier":
            nn.init.xavier_uniform_(layer.weight)
        else:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        nn.init.zeros_(layer.bias)
    count = sum(parameter.numel() for parameter in model.parameters())
    assert count + 2 <= BUDGET, f"{count + 2} ONNX floats exceeds budget"
    return model


def evaluate(model, loader):
    model.eval()
    correct = 0
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            correct += int((logits.argmax(1) == labels).sum())
            loss_sum += float(F.cross_entropy(logits, labels, reduction="sum"))
            count += len(labels)
    return correct / count, loss_sum / count


def fit(model, dataset, epochs, phase, validation=None):
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(SEED))
    val_loader = (DataLoader(validation, batch_size=512)
                  if validation is not None else None)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=LR * 0.1
    )
    loss_fn = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, seen = 0.0, 0
        current_lr = optimizer.param_groups[0]["lr"]
        for batch_index, (images, labels) in enumerate(loader, 1):
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            seen += len(labels)
            if batch_index % 250 == 0:
                print(f"[{phase}] epoch {epoch}/{epochs} "
                      f"batch {batch_index}/{len(loader)}", flush=True)
        row = {"phase": phase, "epoch": epoch, "learning_rate": current_lr,
               "train_loss": loss_sum / seen}
        if val_loader is not None:
            row["val_accuracy"], row["val_loss"] = evaluate(model, val_loader)
        history.append(row)
        print(f"[{phase}] {row}", flush=True)
        scheduler.step()
    return history


def float_count(graph):
    count = sum(numpy_helper.to_array(item).size for item in graph.graph.initializer
                if numpy_helper.to_array(item).dtype.kind == "f")
    for node in graph.graph.node:
        if node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.TENSOR:
                    value = numpy_helper.to_array(attribute.t)
                    if value.dtype.kind == "f":
                        count += value.size
    return count


def local_latency_ms(model_path, dataset):
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    image, _ = dataset[0]
    value = image.numpy()[None]
    for _ in range(20):
        session.run(None, {input_name: value})
    timings = []
    for _ in range(100):
        start = time.perf_counter_ns()
        session.run(None, {input_name: value})
        timings.append((time.perf_counter_ns() - start) / 1_000_000)
    return float(np.median(timings))


def save_curve(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    search = [row for row in history if row["phase"] == "selection"]
    refit = [row for row in history if row["phase"] == "refit"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot([row["epoch"] for row in search],
                 [row["train_loss"] for row in search], label="selection")
    axes[0].plot([row["epoch"] for row in refit],
                 [row["train_loss"] for row in refit], label="full-data refit")
    axes[0].set(title="Training loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[0].legend()
    axes[1].plot([row["epoch"] for row in search],
                 [row["val_accuracy"] for row in search])
    axes[1].set(title="Held-out capture accuracy", xlabel="Epoch",
                ylabel="Accuracy", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset_path = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = VARIANTS[args.variant]
    with (dataset_path / "labels.json").open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    train_indices, val_indices, val_groups = grouped_holdout(
        metadata, seed=SPLIT_SEED
    )
    train = CopybookDirectory(dataset_path, metadata, train_indices)
    validation = CopybookDirectory(dataset_path, metadata, val_indices)
    full = CopybookDirectory(dataset_path, metadata)
    print(f"variant={args.variant} changed={config['changed_factor']}", flush=True)
    print(f"captures held out={len(val_groups)}, train={len(train)}, "
          f"validation={len(validation)}, full={len(full)}", flush=True)

    model = make_model(config)
    search_history = fit(model, train, EPOCHS, "selection", validation)
    winner = max(search_history, key=lambda row:
                 (row["val_accuracy"], -row["val_loss"]))
    best_epoch = winner["epoch"]
    print(f"selected epoch {best_epoch} with validation accuracy "
          f"{winner['val_accuracy']:.4f}", flush=True)
    del model

    model = make_model(config)
    refit_history = fit(model, full, best_epoch, "refit")
    model_path = output / "mlp_model.onnx"
    export_mlp(model, str(model_path))
    graph = onnx.load(model_path)
    onnx.checker.check_model(graph)
    dims = [dim.dim_value for dim in graph.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 1, 64, 64]
    params = float_count(graph)
    assert params <= BUDGET
    latency = local_latency_ms(model_path, full)
    history = search_history + refit_history
    with (output / "training_history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "phase", "epoch", "learning_rate", "train_loss",
            "val_accuracy", "val_loss",
        ))
        writer.writeheader()
        writer.writerows(history)
    save_curve(history, output / "training_curve.png")
    results = {
        "variant": args.variant,
        "baseline": BASELINE,
        "configuration": config,
        "training": {"epochs_searched": EPOCHS, "learning_rate": LR,
                     "momentum": MOMENTUM, "batch": BATCH,
                     "split_seed": SPLIT_SEED,
                     "train_images": len(train), "validation_images": len(validation),
                     "final_train_images": len(full)},
        "best_validation_accuracy": winner["val_accuracy"],
        "best_validation_loss": winner["val_loss"],
        "best_epoch": best_epoch,
        "onnx_float_parameters": params,
        "local_median_latency_ms": latency,
        "server_accuracy": None,
        "server_score": None,
    }
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"DONE {args.variant}: val={winner['val_accuracy']:.4f} "
          f"params={params} local_latency_ms={latency:.4f}", flush=True)


if __name__ == "__main__":
    main()
