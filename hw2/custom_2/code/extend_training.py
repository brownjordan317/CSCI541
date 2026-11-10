"""Extend the late-improving low-LR candidate from epoch 80 to epoch 120.

The original sweep saved weights but no optimizer state. This intentionally
starts fresh Adam at the final learning rate, rather than claiming exact resume.
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from custom_2.code.train_custom_2 import (
    CompactCNN, OUT, clear_device, dump_json, evaluate_views, get_device,
    predict, robust_augment, stress_views, validation_views,
)
from export_onnx import check_onnx, export_onnx, load_lite_cifar


def main():
    name = "compact_robust_low_lr"
    source = OUT / "runs" / name
    destination = OUT / "extended_low_lr"
    destination.mkdir(exist_ok=True)
    if (destination / "metrics.json").exists():
        raise FileExistsError("Extension already completed; refusing to overwrite")
    checkpoint = torch.load(source / "best.pt", map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 80, "This extension is specifically for the selected epoch-80 weights"
    config = checkpoint["config"]
    device = get_device()
    torch.manual_seed(80)
    images, labels = load_lite_cifar(ROOT / "pj2_lite_cifar_train.zip")
    indices = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))
    X_val, y_val = images[indices[:500]], labels[indices[:500]]
    loader = DataLoader(TensorDataset(images[indices[500:]], labels[indices[500:]]),
                        batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(80))
    views = validation_views(X_val)
    model = CompactCNN(tuple(config["widths"]), config["dropout"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"] * 0.04,
                                 weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=0.2)
    criterion = nn.CrossEntropyLoss()
    initial = evaluate_views(model, views, y_val, device)
    best_score = initial["mean_accuracy"]
    best_epoch = 80
    torch.save(checkpoint, destination / "best.pt")
    history = []
    print(f"Extending {name}; device={device}; epochs 81–120; fresh Adam at 0.000020, "
          "then 0.000004 after epoch 100. No exact optimizer resume.", flush=True)
    print(f"Starting validation mean: {best_score:.6f}", flush=True)
    for epoch in range(81, 121):
        model.train()
        loss_sum = 0.0
        lr = optimizer.param_groups[0]["lr"]
        for batch_images, batch_labels in loader:
            batch_images = robust_augment(batch_images).to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_images), batch_labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_images)
        scheduler.step()
        clean = (predict(model, X_val, device) == y_val).float().mean().item()
        record = {"epoch": epoch, "train_loss": loss_sum / 4500,
                  "clean_accuracy": clean, "learning_rate": lr}
        print(f"Epoch {epoch}/120; loss={record['train_loss']:.4f}; clean={clean:.4f}; lr={lr:.6f}",
              flush=True)
        if epoch % 5 == 0:
            metrics = evaluate_views(model, views, y_val, device)
            record["validation"] = metrics
            print(f"  13-view mean={metrics['mean_accuracy']:.6f}", flush=True)
            if metrics["mean_accuracy"] > best_score:
                best_score, best_epoch = metrics["mean_accuracy"], epoch
                torch.save({"config": config, "epoch": epoch, "seed": 80,
                            "original_training_seed": checkpoint["seed"],
                            "continuation": "weights from epoch 80; fresh Adam",
                            "model_state_dict": {k: v.detach().cpu().clone()
                                                 for k, v in model.state_dict().items()},
                            "metrics": metrics}, destination / "best.pt")
        history.append(record)
        dump_json(destination / "history.json", history)

    selected = torch.load(destination / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model_state_dict"])
    final = evaluate_views(model, views, y_val, device)
    stress = evaluate_views(model, stress_views(X_val), y_val, device)
    report = {"source_trial": name, "starting_epoch": 80, "ending_epoch": 120,
              "selected_epoch": best_epoch, "optimizer_restarted": True,
              "initial_validation": initial, "selected_validation": final,
              "diagnostic_stress": stress,
              "improved": best_score > initial["mean_accuracy"],
              "hidden_accuracy": None}
    path = destination / "custom_2_extended.onnx"
    export_onnx(model, path)
    check_onnx(path, part="part3", images=X_val, labels=y_val)
    # Promote only a checked improvement, preserving the prior 80-epoch files.
    if report["improved"]:
        for filename in ("custom_2_cnn.onnx", "custom_2_cnn.pt"):
            original = OUT / filename
            backup = destination / ("before_extension_" + filename)
            if original.exists() and not backup.exists():
                shutil.copy2(original, backup)
        shutil.copy2(path, OUT / "custom_2_cnn.onnx")
        shutil.copy2(destination / "best.pt", OUT / "custom_2_cnn.pt")
    dump_json(destination / "metrics.json", report)
    summary = (
        "\n## Additional training: lower-learning-rate candidate\n\n"
        "Epochs 81–120 started from the epoch-80 weights with fresh Adam at "
        "0.00002, reduced to 0.000004 after epoch 100. This was not an exact optimizer resume.\n\n"
        f"- Selected epoch: **{best_epoch}**.\n"
        f"- Initial / selected 13-view mean: **{initial['mean_accuracy']:.2%} / {best_score:.2%}**.\n"
        f"- Selected clean accuracy: **{final['view_accuracy']['center']:.2%}**.\n"
        f"- Selected 13-view consistency: **{final['all_views_same_prediction']:.2%}**.\n"
        f"- Diagnostic stress mean: **{stress['mean_accuracy']:.2%}** (not a selection score).\n"
        f"- Top-level ONNX updated: **{report['improved']}**.\n\n"
        "The original 90.6%-hidden custom model remains untouched. New hidden accuracy is unknown.\n"
    )
    with (OUT / "RESULTS.md").open("a") as stream:
        stream.write(summary)
    print(summary, flush=True)
    print(json.dumps(report, indent=2), flush=True)
    clear_device(device)


if __name__ == "__main__":
    main()
