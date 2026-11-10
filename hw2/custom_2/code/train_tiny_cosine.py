"""Fresh tiny-model run with cosine or plateau-driven learning-rate control."""

import argparse
import itertools
import json
import math
import statistics
import sys
import time
from pathlib import Path

import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from custom_2.code.train_custom_2 import (
    CompactCNN, dump_json, evaluate_views, get_device, robust_augment,
    stress_views, validation_views,
)
from export_onnx import check_onnx, export_onnx, load_lite_cifar
from custom_2.code.lr_control import PlateauController

OUT = ROOT / "custom_2" / "experiments" / "tiny_cosine"


def learning_rate(epoch, max_epochs, initial=1e-3, minimum=1e-5):
    """Epoch 1 uses initial; the last allowed epoch uses minimum."""
    progress = min(1.0, (epoch - 1) / max(1, max_epochs - 1))
    return minimum + 0.5 * (initial - minimum) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", "--min-epochs", dest="epochs", type=int, default=80,
                        help="Minimum training epochs; also the cosine taper horizon (not a cap)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Optional explicit safety cap; no maximum by default")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--schedule", choices=("cosine", "plateau"), default="cosine")
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "pj2_lite_cifar_train.zip")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.min_delta < 0 or args.lr_patience < 0:
        parser.error("minimum epochs/patience must be positive; min-delta/lr-patience nonnegative")
    if args.max_epochs is not None and args.max_epochs < args.epochs:
        parser.error("optional max-epochs must be at least the minimum epochs")
    out = args.output or OUT.parent / f"tiny_{args.schedule}_min{args.epochs}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best.pt").exists() or (out / "metrics.json").exists():
        raise FileExistsError(f"Existing run artifacts in {out}; refusing to overwrite")

    torch.manual_seed(args.seed)
    device = get_device()
    images, labels = load_lite_cifar(args.data)
    assert images.shape == (5000, 3, 64, 64) and labels.shape == (5000,)
    indices = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))
    X_val, y_val = images[indices[:500]], labels[indices[:500]]
    loader = DataLoader(TensorDataset(images[indices[500:]], labels[indices[500:]]),
                        batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    views = validation_views(X_val)
    model = CompactCNN((16, 32, 64, 96), dropout=0.5).to(device)
    parameters = sum(p.numel() for p in model.parameters())
    assert sum(t.numel() for t in model.state_dict().values() if t.is_floating_point()) <= 3_000_000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    controller = PlateauController(optimizer, minimum=1e-5, lr_patience=args.lr_patience,
                                   stop_patience=args.patience, min_delta=args.min_delta)
    criterion = nn.CrossEntropyLoss()
    best_score, best_epoch = -1.0, 0
    # Keep exact best weights, but reset patience only for meaningful changes.
    patience_reference = None
    cosine_floor_epochs = 0
    stale_epochs = 0
    history = []
    stop_reason = None
    print(f"Device: {device}; seed: {args.seed}; learned parameters: {parameters:,}", flush=True)
    print(f"Fresh training, {args.schedule} 0.001→0.00001, minimum {args.epochs} epochs; "
          f"patience={args.patience}; min_delta={args.min_delta}; "
          f"maximum epochs={args.max_epochs if args.max_epochs is not None else 'none'}", flush=True)
    if args.schedule == "plateau":
        print(f"Halve LR after {args.lr_patience + 1} stagnant validations; "
              "early-stop patience starts only after training at minimum LR.", flush=True)

    for epoch in itertools.count(1):
        if args.schedule == "cosine":
            lr = learning_rate(epoch, args.epochs)
            for group in optimizer.param_groups:
                group["lr"] = lr
        else:
            lr = optimizer.param_groups[0]["lr"]
        model.train()
        loss_sum = 0.0
        for batch_images, batch_labels in loader:
            batch_images = robust_augment(batch_images).to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_images), batch_labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(batch_images)
        metrics = evaluate_views(model, views, y_val, device)
        score = metrics["mean_accuracy"]
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"model_state_dict": {k: v.detach().cpu().clone()
                                             for k, v in model.state_dict().items()},
                        "epoch": epoch, "seed": args.seed, "metrics": metrics,
                        "widths": (16, 32, 64, 96), "dropout": 0.5,
                        "schedule": args.schedule, "minimum_epochs": args.epochs,
                        "maximum_epochs": args.max_epochs}, out / "best.pt")
        if args.schedule == "plateau":
            should_stop = controller.update(score, trained_lr=lr)
            stale_epochs = controller.floor_stale
        else:
            # For cosine too, do not accumulate early-stop patience before
            # training at the floor. Cosine reaches its floor at the minimum
            # epoch horizon and then stays there while patience is assessed.
            at_floor = math.isclose(lr, 1e-5, rel_tol=1e-8, abs_tol=1e-12)
            if not at_floor:
                patience_reference, stale_epochs = None, 0
            elif patience_reference is None or score > patience_reference + args.min_delta:
                patience_reference, stale_epochs = score, 0
            else:
                stale_epochs += 1
            cosine_floor_epochs += int(at_floor)
            should_stop = at_floor and stale_epochs >= args.patience
        history.append({"epoch": epoch, "learning_rate": lr, "train_loss": loss_sum / 4500,
                        "next_learning_rate": optimizer.param_groups[0]["lr"],
                        "validation": metrics, "stale_epochs_at_min_lr": stale_epochs,
                        "epochs_trained_at_min_lr": controller.floor_epochs if args.schedule == "plateau" else cosine_floor_epochs})
        dump_json(out / "history.json", history)
        print(f"Epoch {epoch:02d} (minimum {args.epochs}); lr={lr:.8f}; loss={loss_sum / 4500:.4f}; "
              f"clean={metrics['view_accuracy']['center']:.4f}; mean={score:.5f}; "
              f"consistency={metrics['all_views_same_prediction']:.4f}; stale={stale_epochs}/{args.patience}",
              flush=True)
        if epoch >= args.epochs and should_stop:
            stop_reason = "early_stopping"
            print(f"Early stopping at epoch {epoch}; restoring best epoch {best_epoch}", flush=True)
            break
        if args.max_epochs is not None and epoch >= args.max_epochs:
            stop_reason = "explicit_safety_cap"
            print(f"Explicit safety cap at epoch {epoch}; not an early stop", flush=True)
            break

    checkpoint = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    selected = evaluate_views(model, views, y_val, device)
    stress = evaluate_views(model, stress_views(X_val), y_val, device)
    path = out / f"tiny_{args.schedule}_cnn.onnx"
    export_onnx(model, path)
    onnx_weights = check_onnx(path, part="part3", images=X_val, labels=y_val)
    # Same local single-thread inference scope as benchmark_score.py.
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    inputs = X_val.contiguous().numpy()
    for i in range(30):
        session.run(None, {"image": inputs[i:i+1]})
    timings = []
    for i in range(900):
        x = inputs[i % 500:i % 500 + 1]
        start = time.perf_counter_ns()
        session.run(None, {"image": x})
        timings.append((time.perf_counter_ns() - start) / 1e6)
    median_ms = statistics.median(timings)
    report = {"config": vars(args) | {"data": str(args.data), "output": str(out)}, "parameters": parameters,
              "onnx_float_weights": onnx_weights, "epochs_run": len(history),
              "best_epoch": best_epoch, "stop_reason": stop_reason,
              "minimum_learning_rate": 1e-5,
              "last_training_learning_rate": history[-1]["learning_rate"],
              "reached_minimum_learning_rate": any(math.isclose(r["learning_rate"], 1e-5, rel_tol=1e-8) for r in history),
              "validation": selected, "diagnostic_stress": stress,
              "local_median_latency_ms": median_ms,
              "local_fraction_score_proxy": selected["mean_accuracy"]
                  - 0.05 * math.log10(onnx_weights) - 0.05 * math.log10(median_ms),
              "hidden_accuracy": None,
              "warning": "Local validation and CPU latency proxies, not evaluator results. "
                         "Daily validation selects among more checkpoints than the prior five-epoch protocol."}
    prior_path = OUT.parent / "sweep" / "runs" / "tiny_robust_aug" / "metrics.json"
    if prior_path.exists():
        previous = json.loads(prior_path.read_text())
        report["previous_tiny_validation"] = previous["validation"]
    dump_json(out / "metrics.json", report)
    lines = [f"# Tiny CNN: {args.schedule} schedule and minimum-LR-gated early stopping", "",
             f"- Trained **{len(history)}** epochs; stop reason: `{stop_reason}`.",
             f"- Last training LR: **{history[-1]['learning_rate']:.8f}**; reached minimum: **{report['reached_minimum_learning_rate']}**.",
             f"- Best checkpoint: epoch **{best_epoch}**.",
             f"- Clean validation accuracy: **{selected['view_accuracy']['center']:.2%}**.",
             f"- 13-view mean validation accuracy: **{selected['mean_accuracy']:.2%}**.",
             f"- 13-view prediction consistency: **{selected['all_views_same_prediction']:.2%}**.",
             f"- Diagnostic stress mean: **{stress['mean_accuracy']:.2%}** (not used in selection).",
             f"- ONNX float weights: **{onnx_weights:,}**.",
             f"- Local CPU median inference latency: **{median_ms:.4f} ms**.",
             "", "Original models and exports were not replaced.",
             "Validation is measured every epoch, versus every five epochs in the previous sweep;",
             "more checkpoint selection opportunities can affect apparent improvements.",
             "Actual hidden accuracy and grader latency remain unknown.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
