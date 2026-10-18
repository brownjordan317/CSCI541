"""Summarize completed local controlled runs after the background queue exits."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ("baseline", "pool8", "width96", "kaiming")
HYPOTHESES = {
    "pool8": "Larger pooling blocks reduce parameters and latency, but may erase distinguishing strokes.",
    "width96": "A narrower first hidden layer may retain accuracy with fewer parameters.",
    "kaiming": "Kaiming initialization may improve ReLU training without changing model size.",
}
results = {}
for name in NAMES:
    path = ROOT / f"sub5_{name}" / "results.json"
    if path.exists():
        results[name] = json.loads(path.read_text())

lines = [
    "# Controlled MLP experiments from the sub_5 configuration",
    "",
    "All runs use the same 1,327,326-image dataset, capture-group split (seed 42),",
    "batch size 256, SGD learning rate 0.01 and momentum 0.9, cosine taper,",
    "25-epoch selection, and full-data refit for the selected epoch.",
    "The reference sub_5 server result was 96.1% hidden accuracy, 41,869 float",
    "parameters, 0.012 ms median latency, and Score 0.795. New variants have",
    "**local** validation and latency only until separately evaluated by the server.",
    "",
    "| Run | Best grouped-val accuracy | Val loss | ONNX floats | Local median latency | Server Score |",
    "| --- | ---: | ---: | ---: | ---: | ---: |",
]
for name in NAMES:
    r = results.get(name)
    if r:
        lines.append(f"| {name} | {r['best_validation_accuracy']:.2%} | "
                     f"{r['best_validation_loss']:.4f} | "
                     f"{r['onnx_float_parameters']:,} | "
                     f"{r['local_median_latency_ms']:.4f} ms | pending |")
    else:
        lines.append(f"| {name} | pending | pending | pending | pending | pending |")
lines.append("")
base = results.get("baseline")
for name in NAMES[1:]:
    lines += [f"## {name}", "", f"**Changed factor:** {name if name != 'width96' else 'first-layer width'}.  ",
              f"**Hypothesis:** {HYPOTHESES[name]}"]
    r = results.get(name)
    if r and base:
        diff = r["best_validation_accuracy"] - base["best_validation_accuracy"]
        if diff > 0:
            verdict = f"higher by {diff:.2%}"
        elif diff < 0:
            verdict = f"lower by {-diff:.2%}"
        else:
            verdict = "equal"
        lines += [f"**Result:** Grouped-validation accuracy was {r['best_validation_accuracy']:.2%}, "
                  f"{verdict} relative to the rerun baseline; the ONNX graph has "
                  f"{r['onnx_float_parameters']:,} floats and local median latency "
                  f"{r['local_median_latency_ms']:.4f} ms.",
                  "**Conclusion:** This is a local comparison. Hidden accuracy and server "
                  "Score require separate evaluation; tied validation accuracy should "
                  "be interpreted with validation loss and the confusion matrix."]
    else:
        lines.append("**Result and conclusion:** pending completion of this run and the baseline.")
    lines += ["", f"Training curve and printout: `sub5_{name}/training_curve.png` "
               f"and `sub5_{name}/training.log`.", ""]
(ROOT / "comparison.md").write_text("\n".join(lines) + "\n")
print(f"wrote {ROOT / 'comparison.md'}")
