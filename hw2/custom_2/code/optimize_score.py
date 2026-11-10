"""Score-oriented architecture sweep. Preserve the externally tested submission."""

import argparse
import gc
import itertools
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from custom.train_custom import ResidualBlock
from custom_2.code.lr_control import PlateauController
from custom_2.code.train_custom_2 import (
    clear_device, dump_json, evaluate_views, get_device, robust_augment,
    stress_views, validation_views,
)
from export_onnx import check_onnx, export_onnx, load_lite_cifar

OUT = ROOT / "custom_2" / "experiments" / "score_sweep_v2"
REFERENCE = ROOT / "custom_2" / "submission" / "recommended_tiny.onnx"
REFERENCE_METRICS = ROOT / "custom_2" / "experiments" / "tiny_cosine" / "metrics.json"


@dataclass(frozen=True)
class Config:
    name: str
    widths: tuple
    blocks_per_stage: int = 2


CONFIGS = (
    Config("mini_gap", (16, 32, 64)),
    Config("micro_gap", (12, 24, 48)),
    Config("nano_gap", (8, 16, 32)),
    Config("shallow_gap", (16, 32, 64), blocks_per_stage=1),
)


class ScoreCNN(nn.Module):
    """Conventional residual CNN with a NiN-style GAP/direct-logit head."""

    def __init__(self, widths=(16, 32, 64), blocks_per_stage=2):
        super().__init__()
        layers = [nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                  nn.BatchNorm2d(widths[0]), nn.ReLU(), nn.MaxPool2d(2)]
        previous = widths[0]
        for stage, width in enumerate(widths):
            for block in range(blocks_per_stage):
                stride = 2 if stage > 0 and block == 0 else 1
                layers.append(ResidualBlock(previous, width, stride))
                previous = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(widths[-1], 5),
        )

    def forward(self, x):
        return self.classifier(self.features((x - 0.5) / 0.25))


def float_weights(path):
    graph = onnx.load(str(path))
    return sum(onnx.numpy_helper.to_array(t).size for t in graph.graph.initializer
               if t.data_type == onnx.TensorProto.FLOAT)


def paired_latency(candidate, reference, images, samples=300, rounds=3):
    """Local single-thread CPU session.run, interleaved order across rounds."""
    sessions, timings = {}, {"candidate": [], "reference": []}
    for name, path in (("candidate", candidate), ("reference", reference)):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sessions[name] = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        for i in range(30):
            sessions[name].run(None, {"image": images[i % len(images):i % len(images) + 1]})
    rng = np.random.default_rng(0)
    for _ in range(rounds):
        for name in rng.permutation(["candidate", "reference"]):
            for i in range(samples):
                x = images[i % len(images):i % len(images) + 1]
                start = time.perf_counter_ns()
                sessions[name].run(None, {"image": x})
                timings[name].append((time.perf_counter_ns() - start) / 1e6)
    return {name: {"median_ms": statistics.median(values),
                   "p90_ms": float(np.percentile(values, 90))}
            for name, values in timings.items()}


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


def train(config, args, train_images, train_labels, X_val, y_val, views, device):
    directory = OUT / config.name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "metrics.json").exists():
        if args.reuse_completed:
            report = json.loads((directory / "metrics.json").read_text())
            if report["config"] != json.loads(json.dumps(asdict(config))) or report["seed"] != args.seed or report["minimum_epochs"] != args.min_epochs:
                raise ValueError(f"Mismatched completed experiment: {directory}")
            return report
        raise FileExistsError(f"Completed experiment exists: {directory}")
    if (directory / "last.pt").exists():
        raise FileExistsError(f"Incomplete experiment in {directory}; refusing to overwrite its checkpoints")
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TensorDataset(train_images, train_labels), batch_size=64,
                        shuffle=True, generator=generator)
    model = ScoreCNN(config.widths, config.blocks_per_stage).to(device)
    params = sum(p.numel() for p in model.parameters())
    assert sum(t.numel() for t in model.state_dict().values() if t.is_floating_point()) <= 3_000_000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    controller = PlateauController(optimizer, minimum=1e-5, lr_patience=4,
                                   stop_patience=15, min_delta=0.001)
    criterion = nn.CrossEntropyLoss()
    best_score, best_epoch, history = -1.0, 0, []
    def log(message):
        print(f"[{config.name}] {message}", flush=True)
        with (directory / "train.log").open("a") as stream:
            stream.write(message + "\n")
    log(f"Parameters={params:,}; min epochs={args.min_epochs}; no default maximum; {asdict(config)}")
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
            torch.save({"config": asdict(config), "epoch": epoch, "seed": args.seed,
                        "metrics": validation, "model_state_dict": cpu_tree(model.state_dict())},
                       directory / "best.pt")
        ready = controller.update(score, trained_lr=lr)
        record = {"epoch": epoch, "loss": total_loss / len(train_labels),
                  "learning_rate": lr, "next_learning_rate": optimizer.param_groups[0]["lr"],
                  "validation": validation, "minimum_lr_stale_epochs": controller.floor_stale,
                  "epochs_trained_at_minimum_lr": controller.floor_epochs}
        history.append(record)
        dump_json(directory / "history.json", history)
        # Preserve optimizer/scheduler/RNG state too, unlike the earlier runs.
        # No automatic resume is attempted; this permits future explicit recovery.
        state = {"config": asdict(config), "epoch": epoch, "seed": args.seed,
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
    path = directory / "model.onnx"
    export_onnx(model, path)
    count = check_onnx(path, part="part3", images=X_val, labels=y_val)
    # Release accelerator resources before timing CPU inference.
    del model, optimizer, controller, criterion, state
    clear_device(device)
    latency = paired_latency(path, REFERENCE, X_val.contiguous().numpy())
    reference_metrics = json.loads(REFERENCE_METRICS.read_text())
    reference_mean = reference_metrics["validation"]["mean_accuracy"]
    parameter_gain = -0.05 * math.log10(count / float_weights(REFERENCE))
    latency_gain = -0.05 * math.log10(latency["candidate"]["median_ms"] / latency["reference"]["median_ms"])
    accuracy_change = best_score - reference_mean
    proxy_gain = accuracy_change + parameter_gain + latency_gain
    report = {"config": asdict(config), "parameters": params, "onnx_float_weights": count,
              "seed": args.seed, "minimum_epochs": args.min_epochs, "maximum_epochs": args.max_epochs,
              "epochs_run": epoch, "best_epoch": best_epoch, "stop_reason": stop_reason,
              "last_training_lr": history[-1]["learning_rate"], "validation": checkpoint["metrics"],
              "diagnostic_stress": diagnostics, "local_paired_latency": latency,
              "reference_actual_hidden_accuracy": 0.887, "reference_actual_score": 0.638,
              "proxy_accuracy_change": accuracy_change, "proxy_parameter_gain": parameter_gain,
              "proxy_latency_gain": latency_gain, "local_proxy_score_gain": proxy_gain,
              "calibrated_score_proxy_not_actual": 0.638 + proxy_gain,
              "actual_hidden_accuracy": None, "actual_score": None,
              "warning": "Score proxy assumes fractional accuracy, comparable validation-to-hidden changes, "
                         "and local-to-evaluator latency ratios. None is guaranteed. Not an actual submission result."}
    dump_json(directory / "metrics.json", report)
    log(f"FINISHED: epoch {epoch}, selected {best_epoch}; local proxy score gain {proxy_gain:+.5f}")
    gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trials", nargs="+", choices=[c.name for c in CONFIGS],
                        default=[c.name for c in CONFIGS])
    parser.add_argument("--reuse-completed", action="store_true")
    args = parser.parse_args()
    if args.min_epochs < 1 or (args.max_epochs is not None and args.max_epochs < args.min_epochs):
        parser.error("minimum epochs must be positive; an explicit maximum must not be below minimum")
    if not REFERENCE.exists() or not REFERENCE_METRICS.exists():
        raise FileNotFoundError("The tested tiny-cosine submission and metrics are required as reference")
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    images, labels = load_lite_cifar(ROOT / "pj2_lite_cifar_train.zip")
    assert images.shape == (5000, 3, 64, 64) and labels.shape == (5000,)
    indices = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))
    X_val, y_val = images[indices[:500]], labels[indices[:500]]
    train_images, train_labels = images[indices[500:]], labels[indices[500:]]
    del images, labels
    views = validation_views(X_val)
    reports = []
    for config in CONFIGS:
        if config.name in args.trials:
            reports.append(train(config, args, train_images, train_labels, X_val, y_val, views, device))
    ranked = sorted(reports, key=lambda r: r["local_proxy_score_gain"], reverse=True)
    winner = ranked[0]
    dump_json(OUT / "comparison.json", {
        "ranked_trials": ranked, "local_proxy_leader": winner["config"]["name"],
        "confirmed_submission": "custom_2/submission/recommended_tiny.onnx",
        "confirmed_submission_score": 0.638,
        "warning": "Ranking uses local proxies; do not replace the externally tested submission without evaluation."})
    lines = ["# Score-oriented architecture sweep", "",
             "**All new results are local proxies, not external scores.**",
             "Confirmed baseline: tiny cosine, 88.7% hidden accuracy and 0.638 actual score.", "",
             "| Candidate | ONNX float weights | Epochs / best | Clean | 13-view mean | Paired latency ms | Proxy score gain |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in ranked:
        lines.append(f"| {r['config']['name']} | {r['onnx_float_weights']:,} | {r['epochs_run']} / {r['best_epoch']} | "
                     f"{r['validation']['view_accuracy']['center']:.2%} | {r['validation']['mean_accuracy']:.2%} | "
                     f"{r['local_paired_latency']['candidate']['median_ms']:.4f} | {r['local_proxy_score_gain']:+.5f} |")
    name = winner["config"]["name"]
    lines += ["", f"Local score-proxy leader: **{name}**; export: `{name}/model.onnx`.",
              "The confirmed submission was not replaced.", "",
              "Proxy gain = validation-accuracy change − 0.05 log10(parameter ratio) − 0.05 log10(local latency ratio).",
              "The reference is re-timed alongside each candidate. Timings use single-thread ONNX Runtime CPU",
              "session.run with batch 1, 30 warmups, three interleaved rounds of 300 calls.",
              "Accuracy changes may not transfer to hidden images; CPU latency ratios may differ on the grader.",
              "Do not interpret 0.638 plus this proxy gain as a measured or guaranteed external score.", ""]
    (OUT / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
