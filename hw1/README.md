# MLP experiments from sub_5

The queue runs one job at a time to avoid CPU and disk contention. `baseline` reruns the sub_5 settings; each following job changes exactly one factor:

| Folder | Change from sub_5 | Hypothesis |
| --- | --- | --- |
| `sub5_baseline` | none | Control under the current dataset and split |
| `sub5_pool8` | pooling block 4×4 → 8×8 | Fewer parameters may cost fine stroke detail |
| `sub5_width96` | first hidden width 128 → 96 | Smaller model may preserve accuracy |
| `sub5_kaiming` | Xavier → Kaiming initialization | ReLU training may improve without adding parameters |

Each folder receives `mlp_model.onnx`, `results.json`, `training_history.csv`, `training_curve.png`, and `training.log` when its run finishes. `queue_status.tsv` records completion. `comparison.md` is generated after the queue exits. Local validation and latency are not server hidden accuracy, latency, or Score.
