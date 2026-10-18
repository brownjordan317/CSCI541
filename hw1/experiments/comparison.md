# Controlled MLP experiments from the sub_5 configuration

All runs use the same 1,327,326-image dataset, capture-group split (seed 42),
batch size 256, SGD learning rate 0.01 and momentum 0.9, cosine taper,
25-epoch selection, and full-data refit for the selected epoch.
The reference sub_5 server result was 96.1% hidden accuracy, 41,869 float
parameters, 0.012 ms median latency, and Score 0.795. New variants have
**local** validation and latency only until separately evaluated by the server.

| Run | Best grouped-val accuracy | Val loss | ONNX floats | Local median latency | Server Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 99.90% | 0.0035 | 41,869 | 0.0099 ms | pending |
| pool8 | 99.72% | 0.0092 | 17,293 | 0.0081 ms | pending |
| width96 | 99.89% | 0.0039 | 31,597 | 0.0091 ms | pending |
| kaiming | 99.87% | 0.0043 | 41,869 | 0.0098 ms | pending |

## pool8

**Changed factor:** pool8.  
**Hypothesis:** Larger pooling blocks reduce parameters and latency, but may erase distinguishing strokes.
**Result:** Grouped-validation accuracy was 99.72%, lower by 0.18% relative to the rerun baseline; the ONNX graph has 17,293 floats and local median latency 0.0081 ms.
**Conclusion:** This is a local comparison. Hidden accuracy and server Score require separate evaluation; tied validation accuracy should be interpreted with validation loss and the confusion matrix.

Training curve and printout: `sub5_pool8/training_curve.png` and `sub5_pool8/training.log`.

## width96

**Changed factor:** first-layer width.  
**Hypothesis:** A narrower first hidden layer may retain accuracy with fewer parameters.
**Result:** Grouped-validation accuracy was 99.89%, lower by 0.01% relative to the rerun baseline; the ONNX graph has 31,597 floats and local median latency 0.0091 ms.
**Conclusion:** This is a local comparison. Hidden accuracy and server Score require separate evaluation; tied validation accuracy should be interpreted with validation loss and the confusion matrix.

Training curve and printout: `sub5_width96/training_curve.png` and `sub5_width96/training.log`.

## kaiming

**Changed factor:** kaiming.  
**Hypothesis:** Kaiming initialization may improve ReLU training without changing model size.
**Result:** Grouped-validation accuracy was 99.87%, lower by 0.03% relative to the rerun baseline; the ONNX graph has 41,869 floats and local median latency 0.0098 ms.
**Conclusion:** This is a local comparison. Hidden accuracy and server Score require separate evaluation; tied validation accuracy should be interpreted with validation loss and the confusion matrix.

Training curve and printout: `sub5_kaiming/training_curve.png` and `sub5_kaiming/training.log`.

