"""Compare local latency/validation score proxies; NOT evaluator scores."""

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from custom_2.code.train_custom_2 import CompactCNN, OUT, dump_json
from export_onnx import check_onnx, export_onnx, load_lite_cifar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1 or args.rounds < 1:
        parser.error("samples and rounds must be positive")
    comparison = json.loads((OUT / "comparison.json").read_text())
    entries = [("original_custom", ROOT / "custom" / "custom_cnn.onnx", comparison["reference"])]
    for trial in comparison["trials"]:
        config = trial["config"]
        directory = OUT / "runs" / config["name"]
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        model = CompactCNN(tuple(config["widths"]), config["dropout"])
        model.load_state_dict(checkpoint["model_state_dict"])
        path = directory / "model.onnx"
        export_onnx(model, path)
        check_onnx(path, part="part3")
        entries.append((config["name"], path, trial))
    images, _ = load_lite_cifar(ROOT / "pj2_lite_cifar_train.zip")
    indices = torch.randperm(len(images), generator=torch.Generator().manual_seed(0))[:500]
    inputs = images[indices].contiguous().numpy()
    del images
    sessions, times = {}, {name: [] for name, _, _ in entries}
    for name, path, _ in entries:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sessions[name] = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        for i in range(30):
            sessions[name].run(None, {"image": inputs[i:i+1]})
    # Interleave model order across rounds to reduce systematic warmup/order bias.
    rng = np.random.default_rng(0)
    for _ in range(args.rounds):
        for index in rng.permutation(len(entries)):
            name = entries[index][0]
            session = sessions[name]
            for i in range(args.samples):
                x = inputs[i % len(inputs):i % len(inputs) + 1]
                start = time.perf_counter_ns()
                session.run(None, {"image": x})
                times[name].append((time.perf_counter_ns() - start) / 1e6)
    rows = []
    for name, path, trial in entries:
        graph = onnx.load(str(path))
        parameters = sum(onnx.numpy_helper.to_array(t).size for t in graph.graph.initializer
                         if t.data_type == onnx.TensorProto.FLOAT)
        latency = statistics.median(times[name])
        accuracy = trial["validation"]["mean_accuracy"]
        clean = trial["validation"]["view_accuracy"]["center"]
        penalty = 0.05 * math.log10(parameters) + 0.05 * math.log10(latency)
        rows.append({"name": name, "onnx_path": str(path.relative_to(ROOT)),
                     "learned_parameters": trial["parameters"], "onnx_float_values": parameters,
                     "median_latency_ms": latency,
                     "p90_latency_ms": float(np.percentile(times[name], 90)),
                     "validation_13_view_mean": accuracy, "clean_accuracy": clean,
                     "score_proxy_fraction_accuracy": accuracy - penalty,
                     "score_proxy_percent_accuracy": 100 * accuracy - penalty,
                     "hidden_score": None})
    rows.sort(key=lambda row: row["score_proxy_fraction_accuracy"], reverse=True)
    report = {"formula": "accuracy - 0.05*log10(params) - 0.05*log10(median_latency_ms)",
              "platform": platform.platform(), "processor": platform.machine(),
              "onnxruntime": ort.__version__, "provider": "CPUExecutionProvider",
              "intra_op_threads": 1, "inter_op_threads": 1,
              "warmup_calls": 30, "calls_per_model": args.samples * args.rounds,
              "timing_scope": "session.run, batch=1; excludes session construction and preprocessing",
              "accuracy_proxy": "13-view mean on the existing 500-photo selection set",
              "warning": "NOT hidden accuracy or evaluator latency; do not interpret as actual submission scores.",
              "rows": rows}
    reports = ROOT / "custom_2" / "reports"
    reports.mkdir(exist_ok=True)
    dump_json(reports / "score_benchmark.json", report)
    lines = ["# Local score-proxy comparison", "",
             "**Not actual submission scores.** Accuracy below is a validation proxy, not hidden accuracy.",
             "Latency is local single-thread ONNX Runtime CPU inference, not evaluator hardware.",
             "The fraction-accuracy score assumes the formula uses 0.906 rather than 90.6.", "",
             "| Model | ONNX float values | Local median ms | 13-view accuracy | Score proxy (fraction accuracy) |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['name']} | {row['onnx_float_values']:,} | {row['median_latency_ms']:.4f} | "
                     f"{row['validation_13_view_mean']:.2%} | {row['score_proxy_fraction_accuracy']:.5f} |")
    lines += ["", "Timing: 30 warmup calls per model, three interleaved rounds of 300 timed calls by default.",
              "Input is contiguous float32 [1,3,64,64]; session creation/loading and preprocessing are excluded.",
              "Parameter penalty uses exported float initializer elements, matching the supplied local checker.",
              "The evaluator's exact parameter-count definition and latency configuration need confirmation.",
              "Actual hidden scores remain unknown for all new candidates.", ""]
    (reports / "SCORE_COMPARISON.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
