"""Train only the winning mini GAP CNN; never overwrite the packaged winner."""

import argparse
import itertools
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

if __package__:
    from .export_onnx import check_onnx, export_onnx, load_lite_cifar
    from .model import MiniGAPCNN
    from .training_utils import PlateauController, cpu_tree, evaluate_views, robust_augment, stress_views, validation_views
else:
    from export_onnx import check_onnx, export_onnx, load_lite_cifar
    from model import MiniGAPCNN
    from training_utils import PlateauController, cpu_tree, evaluate_views, robust_augment, stress_views, validation_views

HERE = Path(__file__).resolve().parent
CONFIG = {"name": "mini_gap", "widths": (16, 32, 64), "blocks_per_stage": 2}


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE.parent / "pj2_lite_cifar_train.zip")
    parser.add_argument("--output", type=Path, default=HERE / "runs" / "new_run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args()
    if args.min_epochs < 1 or (args.max_epochs is not None and args.max_epochs < args.min_epochs):
        parser.error("minimum epochs must be positive; optional maximum cannot be below minimum")
    directory = args.output.resolve()
    if directory == HERE or directory == (HERE / "checkpoints").resolve():
        parser.error("use a separate run directory, not the packaged model/checkpoint directory")
    if directory.exists() and any(directory.iterdir()):
        parser.error(f"refusing to overwrite nonempty output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    images, labels = load_lite_cifar(args.data)
    assert images.shape == (5000, 3, 64, 64) and labels.shape == (5000,)
    indices = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))
    X_val, y_val = images[indices[:500]], labels[indices[:500]]
    train_images, train_labels = images[indices[500:]], labels[indices[500:]]
    del images, labels
    views = validation_views(X_val)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=64,
                        shuffle=True, generator=generator)
    model = MiniGAPCNN().to(device)
    assert sum(p.numel() for p in model.parameters()) == 174_933
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    controller = PlateauController(optimizer, minimum=1e-5, lr_patience=4,
                                   stop_patience=15, min_delta=0.001)
    criterion = nn.CrossEntropyLoss()
    best_score, best_epoch, history = -1.0, 0, []

    def log(message):
        print(message, flush=True)
        with (directory / "train.log").open("a") as stream:
            stream.write(message + "\n")

    log(f"mini_gap; device={device}; seed={args.seed}; minimum epochs={args.min_epochs}; "
        f"maximum epochs={args.max_epochs}; parameters=174,933")
    for epoch in itertools.count(1):
        model.train()
        lr = optimizer.param_groups[0]["lr"]
        total_loss = 0.0
        for images, labels in loader:
            images, labels = robust_augment(images).to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        validation = evaluate_views(model, views, y_val, device)
        score = validation["mean_accuracy"]
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"config": CONFIG, "epoch": epoch, "seed": args.seed,
                        "metrics": validation, "model_state_dict": cpu_tree(model.state_dict())},
                       directory / "best.pt")
        ready = controller.update(score, trained_lr=lr)
        record = {"epoch": epoch, "loss": total_loss / len(train_labels),
                  "learning_rate": lr, "next_learning_rate": optimizer.param_groups[0]["lr"],
                  "validation": validation, "minimum_lr_stale_epochs": controller.floor_stale,
                  "epochs_trained_at_minimum_lr": controller.floor_epochs}
        history.append(record)
        dump_json(directory / "history.json", history)
        state = {"config": CONFIG, "epoch": epoch, "seed": args.seed,
                 "minimum_epochs": args.min_epochs,
                 "model_state_dict": cpu_tree(model.state_dict()),
                 "optimizer_state_dict": cpu_tree(optimizer.state_dict()),
                 "scheduler_state_dict": controller.scheduler.state_dict(),
                 "floor_reference": controller.floor_reference,
                 "floor_stale": controller.floor_stale, "floor_epochs": controller.floor_epochs,
                 "best_epoch": best_epoch, "best_score": best_score,
                 "cpu_rng_state": torch.get_rng_state(), "loader_rng_state": generator.get_state()}
        if device.type == "mps":
            state["device_rng_state"] = torch.mps.get_rng_state()
        elif device.type == "cuda":
            state["device_rng_state"] = torch.cuda.get_rng_state()
        torch.save(state, directory / "last.pt")
        log(f"Epoch {epoch}; lr={lr:.8f}; loss={record['loss']:.4f}; "
            f"clean={validation['view_accuracy']['center']:.4f}; mean={score:.5f}; "
            f"floor patience={controller.floor_stale}/15")
        if epoch >= args.min_epochs and ready:
            stop_reason = "minimum_lr_early_stopping"
            break
        if args.max_epochs is not None and epoch >= args.max_epochs:
            stop_reason = "explicit_safety_cap"
            break
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    diagnostics = evaluate_views(model, stress_views(X_val), y_val, device)
    path = directory / "p3_best.onnx"
    export_onnx(model, path)
    count = check_onnx(path, part="part3", images=X_val, labels=y_val)
    dump_json(directory / "metrics.json", {
        "config": CONFIG, "parameters": 174_933, "onnx_float_weights": count,
        "seed": args.seed, "minimum_epochs": args.min_epochs, "maximum_epochs": args.max_epochs,
        "epochs_run": epoch, "best_epoch": best_epoch, "stop_reason": stop_reason,
        "last_training_lr": history[-1]["learning_rate"], "validation": checkpoint["metrics"],
        "diagnostic_stress": diagnostics, "actual_hidden_accuracy": None, "actual_score": None,
    })


if __name__ == "__main__":
    main()
