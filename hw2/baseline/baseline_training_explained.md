# Understanding the baseline CNN and its training

This guide explains the actual implementation in `train_baseline.py`, the requirements in `baseline_cnn_spec.md`, and the results in `train_baseline.log`. It covers what the network computes, where its **2,341,061 learned parameters** come from, how training changes them, and what the results do—and do not—tell us.

## 1. The big picture

The task is **supervised image classification**. Each example contains:

- An RGB image: the input.
- A class ID: the correct answer, supplied during training.

The class IDs are:

| ID | Class |
|---|---|
| 0 | airplane |
| 1 | automobile |
| 2 | frog |
| 3 | horse |
| 4 | ship |

The model takes an image and produces five numerical scores called **logits**. The highest-scoring class is its prediction.

Initially, the model's weights are randomly initialized using PyTorch's layer defaults. They do not encode useful knowledge about these classes. Training repeatedly compares predictions with correct labels and adjusts weights to reduce the prediction error.

The full workflow is:

```text
Read ZIP and labels
       ↓
Create fixed training/validation split
       ↓
Construct randomly initialized CNN
       ↓
Train for 50 epochs; evaluate after each epoch
       ↓
Restore weights with the best validation accuracy
       ↓
Export inference graph to baseline_cnn.onnx
       ↓
Check graph and measure ONNX validation accuracy
```

## 2. The data and its representation

### Image dimensions

`load_lite_cifar()` loads 5,000 images into a tensor with shape:

```text
[5000, 3, 64, 64]
   N   C   H   W
```

Here, `N` is the number of examples, `C` is the number of color channels, and `H` and `W` are height and width. This arrangement is called **NCHW**.

Each image contains a 32×32 photo centered on a gray 64×64 canvas. The canvas is part of the input: the script does not crop it away or resize the image.

The helper reads pixel values and divides them by 255, producing `float32` values in `[0, 1]`. Labels are `int64` class IDs, not one-hot vectors.

### Training versus validation

The script creates the permutation required by the spec:

```python
indices = torch.randperm(
    len(images), generator=torch.Generator().manual_seed(0)
)
val_indices = indices[:500]
train_indices = indices[500:]
```

- **4,500 training examples** supply the gradients that update weights.
- **500 validation examples** measure performance without updating weights.

The full dataset has 1,000 examples per class. The split is randomly permuted, not stratified, so the validation set is not guaranteed to contain exactly 100 examples of each class.

The split uses its own generator with seed 0. It stays the same even when the command-line training seed changes. That makes comparisons between training runs more meaningful: they are evaluated on the same images.

**Important distinction:** validation labels do not enter backpropagation, but validation accuracy does determine which epoch is selected. This makes validation a model-selection set, not a completely untouched test set.

## 3. Parameters versus hyperparameters

These terms describe different things.

**Parameters** are numbers learned by training:

- Convolution kernel weights.
- Convolution biases.
- Linear-layer weights.
- Linear-layer biases.

**Hyperparameters** are choices made before or around training:

- Number of layers and channels.
- Kernel sizes and padding.
- Learning rate.
- Batch size.
- Number of epochs.
- Dropout probability.
- Augmentation policy.

Adam changes learned parameters. It does not decide to add another convolution or change the batch size. For this assignment, the architecture and training hyperparameters are fixed by the Part 1 specification.

## 4. The architecture, step by step

The model deliberately has two separate `nn.Sequential` attributes:

- `features`: the convolutional backbone.
- `classifier`: the Flatten-based classification head.

For a batch of size `B`, the shapes are:

| Operation | Output shape | Learned parameters |
|---|---|---:|
| Input | `B × 3 × 64 × 64` | 0 |
| Normalize | `B × 3 × 64 × 64` | 0 |
| Conv 5×5, 3 → 32 | `B × 32 × 64 × 64` | 2,432 |
| ReLU | `B × 32 × 64 × 64` | 0 |
| MaxPool 2×2 | `B × 32 × 32 × 32` | 0 |
| Conv 3×3, 32 → 64 | `B × 64 × 32 × 32` | 18,496 |
| ReLU | `B × 64 × 32 × 32` | 0 |
| MaxPool 2×2 | `B × 64 × 16 × 16` | 0 |
| Conv 3×3, 64 → 128 | `B × 128 × 16 × 16` | 73,856 |
| ReLU | `B × 128 × 16 × 16` | 0 |
| MaxPool 2×2 | `B × 128 × 8 × 8` | 0 |
| Conv 3×3, 128 → 128 | `B × 128 × 8 × 8` | 147,584 |
| ReLU | `B × 128 × 8 × 8` | 0 |
| Flatten | `B × 8192` | 0 |
| Linear 8192 → 256 | `B × 256` | 2,097,408 |
| ReLU | `B × 256` | 0 |
| Dropout, probability 0.5 | `B × 256` | 0 |
| Linear 256 → 5 | `B × 5` | 1,285 |
| **Total** | | **2,341,061** |

### 4.1 Normalization

The first computation in `forward()` is:

```python
x = (x - 0.5) / 0.25
```

For example:

| Original pixel | Normalized value |
|---:|---:|
| 0.00 | -2.0 |
| 0.25 | -1.0 |
| 0.50 | 0.0 |
| 0.75 | 1.0 |
| 1.00 | 2.0 |

This shifts and rescales inputs. It often helps optimization by avoiding an entirely positive input range and controlling the scale of the first layer's inputs.

The constants are specified values, not statistics estimated from this dataset. Normalization does not guarantee that the actual dataset has exactly zero mean and unit variance. It is also not BatchNorm: there are no learned normalization parameters or running statistics.

Normalization happens **inside the model** so it is included in ONNX. The consumer of the exported model supplies raw `[0, 1]` pixels and must not apply the same normalization again.

### 4.2 Convolution: shared pattern detectors

A convolution applies learned kernels across an image or feature map. Each output channel has its own kernel and bias.

One output value can be written schematically as:

```text
output[channel, row, column]
    = bias[channel]
      + sum of input values × kernel weights
        across the kernel's spatial window and all input channels
```

The first layer has 32 filters, each shaped `3 × 5 × 5`. A filter spans all three input color channels, not just one. Each filter produces one output feature map.

The same filter weights are reused at every spatial location. This **weight sharing** is why a convolution can recognize a pattern in different positions without learning a separate detector for every pixel.

Early layers can learn simple color and edge patterns; deeper layers can combine them into more complex patterns. That is a useful intuition, not a claim that we inspected these particular filters and identified what each one detects.

### 4.3 Kernel size, stride, and padding

For these convolutions, which use dilation 1, the output size along one axis is:

```text
output size = floor((input size + 2 × padding - kernel size) / stride) + 1
```

The first convolution uses kernel 5, padding 2, and stride 1:

```text
floor((64 + 4 - 5) / 1) + 1 = 64
```

The other convolutions use kernel 3, padding 1, and stride 1, so they also preserve spatial dimensions.

- **Larger kernels** directly inspect larger neighborhoods but require more weights and computation at the same channel counts.
- **Stride** determines how far the kernel moves between output positions. Larger stride generally reduces spatial resolution.
- **Padding** adds values around the boundary. Here it is zero-padding, used to preserve spatial size through each convolution.
- **Output channels** determine how many distinct learned feature maps a layer produces.

### 4.4 ReLU: making the network nonlinear

ReLU computes:

```text
ReLU(x) = max(0, x)
```

Without nonlinearities, a composition of convolutional and linear transformations alone could be collapsed into a single affine transformation. ReLU lets the network represent more complex relationships.

ReLU has no learned weights. Its gradient is 1 for positive inputs and 0 for negative inputs; PyTorch uses 0 at the boundary. Consequently, some pathways do not receive a gradient on a particular example.

### 4.5 Max pooling: reducing spatial resolution

`nn.MaxPool2d(2)` takes the maximum in each 2×2 window, with stride 2 by default:

```text
64×64 → 32×32 → 16×16 → 8×8
```

Pooling reduces spatial computation in later layers and can make features less sensitive to small shifts. It also discards information: the exact arrangement of all four values in each window is not retained.

Pooling has no learned parameters, but it affects learning. During backpropagation, the gradient through a max-pooling window follows the selected maximum.

### 4.6 Receptive field

A feature's **receptive field** is the region of the original input that can influence it.

Using this architecture's kernel sizes and strides, the nominal receptive field grows as follows:

| After operation | Receptive field | Spacing between adjacent features in input coordinates |
|---|---:|---:|
| First convolution | 5×5 | 1 |
| First pool | 6×6 | 2 |
| Second convolution | 10×10 | 2 |
| Second pool | 12×12 | 4 |
| Third convolution | 20×20 | 4 |
| Third pool | 24×24 | 8 |
| Fourth convolution | 40×40 | 8 |

Thus one final backbone feature can depend on a nominal 40×40 input region. Near boundaries, some of that region involves padding. This is the theoretical support, not a guarantee that every input pixel contributes equally strongly. The dense head then combines features across all 8×8 positions.

### 4.7 Flatten and the dense head

The backbone produces 128 feature maps of size 8×8:

```text
128 × 8 × 8 = 8192 values per image
```

Flatten rearranges these values into a vector. It does not learn weights and does not average or discard feature values.

The first linear layer computes:

```text
h = W × flattened_features + b
```

Its weight matrix has shape `[256, 8192]`. Each of its 256 outputs can combine information from every channel and spatial position. This makes the head flexible and position-sensitive, but expensive in parameter count.

The final linear layer maps the 256 hidden activations to five class logits.

## 5. Exactly where the parameters come from

### Convolution formula

For these ordinary convolutions, with one bias per output channel:

```text
parameters = output_channels × input_channels × kernel_height × kernel_width
             + output_channels
```

Applying it:

```text
Conv 1: 32 × 3 × 5 × 5 + 32       =   2,432
Conv 2: 64 × 32 × 3 × 3 + 64      =  18,496
Conv 3: 128 × 64 × 3 × 3 + 128    =  73,856
Conv 4: 128 × 128 × 3 × 3 + 128   = 147,584
                                          --------
Backbone total                            242,368
```

Notice that image height and width do not appear in this formula. A convolution reuses its weights at different locations. Larger feature maps still require more computation and activation memory, even if the number of learned weights is unchanged.

### Linear-layer formula

```text
parameters = input_features × output_features + output_features
```

Therefore:

```text
Linear 1: 8192 × 256 + 256 = 2,097,408
Linear 2: 256 × 5 + 5     =     1,285
                                  ---------
Head total                        2,098,693

Overall total = 242,368 + 2,098,693 = 2,341,061
```

Almost **90% of all parameters are in the head**, and approximately **89.6% are in the first linear layer alone**.

That explains why changing only the head can dramatically change model size. For comparison, the Part 2 global-average-pooling head reduces each of the 128 feature maps to one number. Its first linear layer needs only:

```text
128 × 256 + 256 = 33,024 parameters
```

With the unchanged backbone and final layer, Part 2 totals 276,677 parameters. This comparison is explanatory only; `train_baseline.py` implements Part 1 and does not use global average pooling.

### Parameters are not the same as memory use

At four bytes per `float32`, the learned values alone occupy:

```text
2,341,061 × 4 = 9,364,244 bytes ≈ 8.93 MiB
```

An ONNX file also contains graph information and constants. During training, memory use is much larger because it includes gradients, Adam state, intermediate activations, input batches, and a CPU copy of the best weights. Parameter count alone does not determine speed or total memory use.

## 6. What happens during one training batch

The script uses batch size 64. Each batch goes through the following sequence.

### Step 1: Random horizontal flips

```python
flip = torch.rand(len(batch_images)) < 0.5
batch_images[flip] = batch_images[flip].flip(-1)
```

Each image is independently flipped left-to-right with probability 0.5. The last dimension is width, so `.flip(-1)` reverses columns, not color channels or image rows.

The class remains unchanged: a left-facing horse is still a horse. This encourages the model not to depend unnecessarily on orientation.

Augmentation creates variation each time an example is visited; it does not permanently double the dataset. There are no crops, rotations, color changes, or other augmentations. Validation images are not flipped.

### Step 2: Move the batch to the compute device

The script selects CUDA if available, otherwise Apple's MPS accelerator if available, otherwise CPU. The completed run used MPS.

The model and its current input batch must be on compatible devices. Moving data changes where computations happen; it does not change which architecture is being trained.

### Step 3: Clear old gradients

```python
optimizer.zero_grad(set_to_none=True)
```

PyTorch normally accumulates gradients. We clear the previous batch's gradients so the next update uses only the current batch. Using `set_to_none=True` avoids explicitly filling existing gradient buffers with zeros; gradients are populated during backpropagation.

### Step 4: Forward pass and loss

```python
loss = criterion(model(batch_images), batch_labels)
```

The forward pass produces one five-logit vector per example. `nn.CrossEntropyLoss()` measures how well those logits match the true class IDs.

### Step 5: Backpropagation

```python
loss.backward()
```

Autograd uses the chain rule to calculate a derivative for every trainable parameter involved in the computation:

```text
gradient = ∂loss / ∂parameter
```

A gradient describes how a small change in a parameter would affect the current batch's loss, locally. It is not the parameter's new value and does not directly say how validation accuracy will change.

Gradients flow backward through the dense head, Flatten, convolutions, ReLUs, pooling, and normalization. Non-learned operations still affect gradient flow even though they have no parameters of their own.

### Step 6: Adam updates the weights

```python
optimizer.step()
```

Adam uses the gradients and its running state to adjust each parameter. The next batch is processed using those updated weights.

## 7. Cross-entropy, logits, and accuracy

### Why the model does not end in softmax

For logits `z`, the conceptual probability of class `k` is:

```text
p[k] = exp(z[k]) / sum_j exp(z[j])
```

For an example with true label `y`, cross-entropy is:

```text
loss = -log(p[y])
     = -z[y] + log(sum_j exp(z[j]))
```

`CrossEntropyLoss` combines these operations in a numerically stable implementation. It expects **raw logits**, so the model must not apply softmax before passing outputs into this loss.

Softmax is unnecessary for choosing the predicted class too: it preserves ordering, so `argmax(logits)` gives the same class as `argmax(softmax(logits))`.

### How loss behaves

If all five classes have equal predicted probability, the loss is:

```text
-log(1/5) = log(5) ≈ 1.609
```

Higher probability for the correct class lowers loss. Confidently predicting the wrong class can produce very large loss.

For one example, the derivative with respect to a logit is:

```text
∂loss / ∂z[k] = p[k] - 1[k == y]
```

Here `1[k == y]` is 1 for the true class and 0 otherwise. This gives the learning signal: promote the true class relative to competing classes. The default batch loss averages across examples.

### Loss and accuracy measure different things

Accuracy asks only whether the largest logit belongs to the correct class. It ignores how close the competing logits were.

Cross-entropy cares about the entire probability assigned to the correct class. A model can become more confident without changing any predicted labels, lowering loss while leaving accuracy unchanged. Conversely, a few very confident mistakes can raise loss even if accuracy improves.

This is why training optimizes a smooth loss rather than trying to backpropagate through the discrete correct/incorrect count.

## 8. Adam and the learning rate

The optimizer is:

```python
torch.optim.Adam(model.parameters(), lr=1e-3)
```

The learning rate is `0.001`. Other settings use PyTorch defaults, including `betas=(0.9, 0.999)`, `eps=1e-8`, and `weight_decay=0`.

Conceptually, Adam maintains two moving averages for each parameter:

```text
m_t = beta1 × m_(t-1) + (1 - beta1) × gradient
v_t = beta2 × v_(t-1) + (1 - beta2) × gradient²
```

After bias correction, the update is approximately:

```text
parameter ← parameter - learning_rate × corrected_m / (sqrt(corrected_v) + epsilon)
```

The first moment smooths gradient direction; the second tracks gradient magnitude. Different parameters therefore receive differently scaled updates.

The learning rate is **not** the fraction of each weight removed per step. The update also depends on gradients and Adam's history.

- Too large a learning rate can cause instability or prevent settling into a useful solution.
- Too small a learning rate can make learning slow or leave the model undertrained within a fixed epoch budget.
- There is no learning-rate scheduler in this implementation: the configured learning rate remains `0.001`.
- There is no explicit weight decay or gradient clipping.

## 9. Epochs, batch size, and shuffling

An **epoch** is one pass through all 4,500 training examples.

With batch size 64:

```text
4500 = 70 × 64 + 20
```

There are 71 optimizer updates per epoch: 70 full batches and one batch of 20. The final batch is kept because `drop_last` is not enabled.

Across 50 epochs:

```text
71 × 50 = 3,550 optimizer updates
4500 × 50 = 225,000 training-example presentations
```

Those presentations are not 225,000 distinct source images. They revisit the same 4,500 examples, potentially with different flip choices.

`shuffle=True` gives a new training order each epoch. Since each update changes weights, changing example order can change the optimization trajectory.

### What batch size affects

- **Memory:** larger batches usually require more activation memory.
- **Gradient noise:** smaller batches typically give noisier estimates of the full training-set gradient.
- **Updates per epoch:** larger batches mean fewer updates for the same dataset size.
- **Throughput:** hardware may process larger batches more efficiently, up to its limits.
- **Generalization:** batch size can affect the solution reached; larger is not automatically better.

Batch size does not change this model's parameter count.

### How the displayed training loss is calculated

The script multiplies each mean batch loss by the actual batch size, sums these values, and divides by 4,500:

```python
total_loss += loss.item() * len(batch_images)
epoch_loss = total_loss / 4500
```

This correctly weights the small final batch. It reports losses from training-mode forward passes at different moments during the epoch, while the weights are changing. It is not a separate evaluation of the final epoch weights on the entire training set.

## 10. Dropout and overfitting

The classifier includes:

```python
nn.Dropout(0.5)
```

During training, each of the 256 hidden activations is independently zeroed with probability 0.5. Surviving activations are multiplied by `1 / (1 - 0.5) = 2`, preserving each activation's expected value.

Dropout encourages the network not to rely too heavily on particular hidden activations always being present. It changes activations temporarily; it does not delete learned weights or reduce parameter count.

During evaluation, dropout is disabled and all hidden activations are used without that training-time rescaling.

**Overfitting** means learning details that improve training performance without improving performance on new examples. This network has many parameters relative to 4,500 training images, especially in its dense head. Horizontal flips, dropout, and selecting a good validation checkpoint help address that risk, but do not guarantee its absence.

## 11. Training mode versus evaluation mode

At the start of each training epoch:

```python
model.train()
```

During validation:

```python
model.eval()
```

These methods control mode-dependent layer behavior. In this model, the relevant difference is dropout.

Validation also uses `@torch.no_grad()`, which disables gradient recording for those computations. This reduces memory and overhead.

These are separate controls:

- `eval()` changes layer behavior but does not by itself disable autograd.
- `no_grad()` disables gradient tracking but does not by itself disable dropout.

Correct validation uses both. No validation optimizer step is performed.

## 12. Keeping the best epoch

After each epoch, validation accuracy is compared with the best value seen so far. On a strict improvement, the script saves a copy of the model's state:

```python
best_state = {
    name: tensor.detach().cpu().clone()
    for name, tensor in model.state_dict().items()
}
```

The clone is essential: saving references to live parameter storage would allow later training updates to change the supposedly saved checkpoint. Moving the copy to CPU also avoids retaining it on the accelerator.

The comparison uses `>`, not `>=`, so an equal accuracy does not replace the earlier checkpoint.

All 50 epochs are still trained. This is **best-checkpoint selection**, not early stopping.

At the end:

```python
model.load_state_dict(best_state)
```

The selected weights are restored before export. The exported model therefore need not be the model from epoch 50.

The best state is kept in memory during the run; the script does not write a separate PyTorch training checkpoint. The ONNX file contains the selected model for inference, not Adam's state for resuming training.

## 13. Understanding the actual results

The run used seed 0 and MPS. Selected log entries were:

| Epoch | Training loss | Validation accuracy |
|---:|---:|---:|
| 1 | 1.2674 | 64.60% |
| 10 | 0.3863 | 85.80% |
| 20 | 0.1309 | 85.60% |
| 30 | 0.0668 | 86.00% |
| 40 | 0.0383 | **88.40%** |
| 50 | 0.0447 | 86.80% |

The best checkpoint was **epoch 40**, with:

```text
442 correct / 500 images = 0.884 = 88.40%
58 incorrect / 500 images
```

Each validation example changes accuracy by `1 / 500 = 0.002`, or **0.2 percentage points**. For example, 88.2% versus 88.4% is a difference of just one prediction.

### What the trajectory suggests

- The model learned useful information quickly: accuracy rose substantially over the first ten epochs.
- Training loss continued falling while validation accuracy improved much more slowly and fluctuated.
- The later training/validation pattern is consistent with diminishing generalization gains and possible overfitting, but the log does not include validation loss or a separately evaluated training accuracy, so it cannot support every diagnostic conclusion.
- Epoch 50 was not the best validation model; restoring epoch 40 mattered.
- Loss need not fall monotonically because batches, augmentations, dropout masks, and optimization updates vary.

### What 88.40% does not establish

It is not a guarantee of 88.40% on every future dataset. It is accuracy on this specific 500-image validation set, which also selected the winning epoch.

Repeatedly trying many architectures or seeds and choosing whichever scores best on this same validation set can overfit model selection to the validation set. An independent test set is needed for a less selection-biased final assessment.

The log also does not tell us which classes are confused most often, whether predictions are well calibrated, or how performance changes under distribution shift. A confusion matrix, per-class metrics, and an untouched test set would answer different questions.

## 14. Reproducibility and the seed

The script calls:

```python
torch.manual_seed(args.seed)
```

before building the model. This controls PyTorch random-number generation relevant to initialization and stochastic training operations, subject to backend behavior.

The seed influences:

- Initial layer weights and biases.
- Training-data shuffling.
- Random flip decisions.
- Dropout masks.

The validation split remains fixed by its separate seed-0 generator.

A fixed seed helps reproducibility, but does not promise bit-for-bit identical training across CUDA, MPS, CPU, different library versions, or nondeterministic kernels. The script does not enable strict deterministic algorithms.

Example commands, using the existing environment:

```bash
.venv/bin/python train_baseline.py
.venv/bin/python train_baseline.py --seed 1
```

These commands perform a fresh full training run. They write `baseline_cnn.onnx` in the current working directory, replacing the previous file at that path. The default dataset path is relative to the script's directory; `--data` can override it.

## 15. Exporting to ONNX

The script uses the supplied helper without modifying it:

```python
export_onnx(model, "baseline_cnn.onnx")
check_onnx(
    "baseline_cnn.onnx",
    part="part1",
    images=X_val,
    labels=y_val,
)
```

The export helper moves the model to CPU, switches to evaluation mode, and exports using the TorchScript-based exporter with ONNX opset 17.

The exported contract for this five-class model is:

```text
Input name:    image
Input type:    float32
Input shape:   [1, 3, 64, 64]
Input values:  RGB pixels in [0, 1]

Output name:   logits
Output shape:  [1, 5]
Prediction:    index of largest logit
```

The batch dimension is fixed to 1 in this export. Although PyTorch trained with batches of up to 64, the exported graph is intended for one image per inference call.

**Note about the supplied helper:** some comments and an assertion message mention 10 output classes, but its actual shape assertion uses `len(CLASSES)`, which is 5. The spec and implemented model correctly use five logits.

### Why dropout is absent from the ONNX operation list

Dropout is inactive in evaluation mode. It therefore need not remain as a stochastic operation in the exported inference graph. This does not mean it was omitted during training.

### What the ONNX operators mean

| ONNX operator | Role in this model |
|---|---|
| `Constant` | Fixed values used by operations such as normalization |
| `Sub` | Subtract 0.5 from input values |
| `Div` | Divide by 0.25 |
| `Conv` | Learned convolutions |
| `Relu` | Nonlinear activation |
| `MaxPool` | Spatial downsampling |
| `Flatten` | Reshape features into a vector |
| `Gemm` | Matrix multiplication with bias for the linear layers |

ONNX export does not retrain the network. It serializes the selected inference computation and learned values in another representation.

## 16. What the ONNX check confirms

The completed run printed:

```text
wrote baseline_cnn.onnx  (2,341,061 parameters)
baseline_cnn.onnx: OK  input [1, 3, 64, 64]  params 2,341,061  convs 4
  ops: Constant, Conv, Div, Flatten, Gemm, MaxPool, Relu, Sub
  onnxruntime accuracy on 500 given images: 0.884
```

The supplied `check_onnx()` function:

1. Loads the graph and runs the ONNX checker.
2. Asserts the required input shape.
3. Counts float-valued initializer elements and checks the 3,000,000 budget.
4. Prints the convolution count and operation types.
5. Warns if global pooling appears in a Part 1 graph.
6. Runs ONNX Runtime inference on the supplied validation images and measures accuracy.

The 0.884 ONNX accuracy matches the selected PyTorch validation accuracy. That is reassuring, but equality of aggregate accuracy alone does not prove every logit—or even every individual prediction—is identical between runtimes.

Also, `check_onnx()` is not a complete local enforcement of the architecture table. Its documentation says the server performs additional architecture checks. That is why matching the specified layers in the Python implementation matters even when the local check says `OK`.

## 17. What affects what: a practical map

The changes below are conceptual explanations, **not changes permitted in the fixed Part 1 submission**.

| Choice | Main effects | Important caveat |
|---|---|---|
| More channels | More feature capacity, parameters, and computation | Can increase overfitting and memory use |
| Larger kernels | Larger local receptive fields; more weights at fixed channel counts | Not automatically better than stacked small kernels |
| More pooling | Smaller feature maps and dense heads | Loses spatial detail |
| Larger input resolution | More activation computation; larger Flatten head if its input grows | Existing linear dimensions must still match |
| Wider hidden linear layer | More head capacity and parameters | The head already dominates this model's size |
| Global average pooling | Drastically fewer head weights and less explicit spatial dependence | Changes the required Part 1 architecture |
| Higher learning rate | Larger configured update scale | May destabilize training |
| More epochs | More optimization steps and data revisits | Can improve training fit without helping validation |
| Larger batch size | More memory, fewer steps per epoch, different gradient noise | Does not change learned parameter count |
| More dropout | Stronger random activation masking | Too much can hinder learning |
| More augmentation | Encourages invariance to chosen transformations | Invalid transformations can destroy label information |
| Weight decay | Penalizes or shrinks weights depending on optimizer formulation | This run uses none |
| Different seed | Different initialization and stochastic trajectory | Not a meaningful architectural improvement by itself |

Many choices interact. For example, increasing batch size while keeping 50 epochs reduces the number of optimizer steps; increasing resolution can dramatically inflate a Flatten head; stronger augmentation may need more training to fit. A controlled experiment changes one factor at a time and keeps evaluation consistent.

## 18. Key takeaways

1. **The CNN learns from gradients, not from manually written visual rules.** Labels provide the training signal; backpropagation and Adam update weights.
2. **The exact parameter count follows from tensor dimensions.** Convolutions share weights spatially; the Flatten-to-256 linear layer accounts for most of the model's parameters.
3. **Architecture, optimization, and regularization have different jobs.** Layers define the function family, Adam searches within it, and flips/dropout help discourage brittle fitting.
4. **Validation is separate from weight updates but still guides model selection.** The exported weights came from epoch 40, not the final epoch.
5. **Training quality and export correctness are separate concerns.** This run achieved 88.40% validation accuracy, then successfully exported and ran under ONNX Runtime with the same measured accuracy.
