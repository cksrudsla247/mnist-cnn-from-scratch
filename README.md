# MNIST CNN in NumPy — learning code and the KV260 reference network

**Conv / MaxPool / BatchNorm forward *and* backward implemented in NumPy, first with raw nested loops, then vectorised with im2col — and finally the VGG-style network that the [KV260 CNN accelerator](https://github.com/cksrudsla247/kv260-cnn-accelerator) runs in hardware.**

Winter Intern, EASY Lab · 2026.01 – 2026.02

The repository also contains later hardware-reference and array-exploration
code; the internship dates do not date every file in this repository.

### 코드 출처와 AI 활용 / Code provenance and AI assistance

이 저장소에는 교재 기반 코드와 프로젝트에서 추가·수정한 코드가 함께 있습니다. 아래 Credits에서 재사용한 공통 계층·optimizer·loader와 확장 부분을 구분합니다.

PE 배열 비교용 `analyze_hw.py`는 AI를 활용해 작성했고, 후보 결과의 비교와 8×32 배열 선택은 직접 수행했습니다. `from scratch`는 NumPy 기반 구현을 뜻하며, 모든 코드를 참고 자료나 AI 도움 없이 단독 작성했다는 의미로 사용하지 않습니다. 프로젝트 추가 파일이라는 분류도 개별 코드의 단독 수작업 작성을 뜻하지 않습니다.

---

## 1. What is implemented

```
04_CNN/
  layers.py             Conv / Pooling / Affine / ReLU / SoftmaxWithLoss
                        BatchNorm_MLP / BatchNorm_CNN  — all forward + backward
  simple_convnet.py     Conv-BN-ReLU-Pool-Affine-BN-ReLU-Affine
  convnet_hw.py         VGG-style 6-layer net for the RTL accelerator
  run_hw.py             train + evaluate convnet_hw, saves params_hw.pkl
  training_convnet.py   training loop for simple_convnet
  test_convnet.py       inference loop for simple_convnet
  analyze_hw.py         cost model / PE-array-shape exploration
  optimization.py       SGD / Momentum / Adam
dataset/mnist.py        MNIST loader (downloads the idx files on first use)
params_hw.pkl           trained weights of convnet_hw (the accelerator's network)
```

The network uses NumPy without a deep-learning framework. The following sections
explain the forward and backward computations; code origins are listed in Credits.

### 1.1 Convolution backward

Three gradients are needed: `dW`, `db`, `dx`.

- `dW` — each filter pixel was multiplied by a specific input during the forward
  pass, so `dW` is the convolution of the input with the output gradient.
- `db` — bias was broadcast over `(N, OH, OW)` in the forward pass, so `db` sums
  the output gradient over exactly those axes.
- `dx` — cannot be obtained by a plain convolution of `dout` with the filter; it
  needs `dout` **zero-padded and convolved with the 180°-rotated filter**.

Implementing `dW` and `dx` as two separate sliding-window loops would be slow and
would need the filter flipped. Instead **both gradients are computed from a single
sweep over `dout` pixels**: for each output-gradient pixel, multiply it by the
corresponding input window to accumulate into `dW`, and by the whole filter to
accumulate into `dx`. One loop nest, no filter flip.

If the forward pass zero-padded `x`, `dx` is returned cropped back to the unpadded
shape.

### 1.2 Max pooling backward

`arg_max` from the forward pass stores the **1-D** index of the winning element in
each window. Backward routes `dout` to exactly that position and zeros everything
else — so the 1-D index has to be converted back to 2-D: `quotient = row`,
`remainder = column`, dividing by the pool width.

### 1.3 BatchNorm — MLP and CNN versions

- **`BatchNorm_MLP`** — forward and backward drawn as a computation graph and
  transcribed step by step. The subtlety: the mean reduces `(N, D)` to `(D,)`, so
  backward must restore the shape — hence the `np.ones((N, D))` broadcasts at
  steps 6 and 8. `dbeta` and `dgamma` sum over the batch axis for the same reason.
- **`BatchNorm_CNN`** — same maths, 4-D input. Transpose `(N,C,H,W) → (N,H,W,C)`,
  reshape to `(N*H*W, C)`, run the MLP version, then restore the shape. The
  broadcast becomes `np.ones((N*H*W, D))`.

`BatchNorm_CNN` also keeps **running mean / variance** for inference. This matters
for the hardware: the accelerator runs at batch size 1, and without running stats
a single image would be normalised by its own mean and could never match the
software model. `run_hw.py` prints batch-1 and batch-200 accuracy side by side for
exactly this reason; they must be equal.

---

## 2. Results

The tables in 2.1–2.3 are from the project report (Korean, not included in this
repository). They were measured with 3,000 training and 1,000 test images unless
stated otherwise.

### 2.1 Inference with the textbook's pretrained parameters

`Conv → ReLU → Pool → Affine → ReLU → Affine → Softmax`, weights from the book's
`params.pkl` (see *Credits*):

| test set size | accuracy |
|---|---|
| 1000 | **98.7%** |
| 3000 | 98.1% |

`simple_convnet.py` was later extended with two BatchNorm layers, which the book's
`params.pkl` does not contain, so this exact result is not reproducible with the
code in this repository.

### 2.2 The cost of not using im2col

The first implementation deliberately used **four nested for-loops** for both
convolution and pooling, to make the operation explicit before optimising it.

| run | train acc | test acc | time (h:mm) |
|---|---|---|---|
| pretrained `params.pkl`, inference only | — | 98.7% | 0:04 |
| for-loop training, init std 0.01 | 12.9% | 14.2% | 1:56 |
| for-loop training, **He init** | 85.5% | 79.3% | 1:56 |

The initialisation change alone moved test accuracy from 14.2% to 79.3%. The
for-loop version took almost two hours to train; after switching to im2col, a
comparable run (lr 0.03, 30 filters, see 2.3) trained in **1 min 01 s**.

### 2.3 Architecture and hyper-parameter study

**Adding a second conv layer** (`Conv→ReLU→Conv→ReLU→Pool→Affine`) — more time,
slightly lower accuracy at unchanged hyper-parameters:

| | before | after |
|---|---|---|
| train / test | 0.876 / 0.835 | 0.873 / 0.826 |
| time (h:mm) | 1:17 | 5:57 |

**Learning rate** (30 filters, SGD, for-loop version):

| lr | 0.01 | **0.03** | 0.05 |
|---|---|---|---|
| train / test | 0.855 / 0.793 | **0.886 / 0.852** | 0.879 / 0.832 |
| time (h:mm) | 1:56 | 1:57 | 3:03 |

At 0.05 accuracy drops near the end of training — the step overshoots the optimum.

**Filter count** (lr 0.03, im2col):

| filters | 15 | 30 | **45** |
|---|---|---|---|
| train / test | 0.891 / 0.853 | 0.906 / 0.869 | **0.908 / 0.873** |
| time (m:ss) | 0:39 | 1:01 | 1:24 |

More filters capture more features, at a proportional cost in compute.

**Batch size** (lr 0.03, 45 filters):

| batch | **100** | 300 | 500 |
|---|---|---|---|
| train / test | **0.934 / 0.891** | 0.910 / 0.872 | 0.901 / 0.862 |
| time (m:ss) | 2:15 | 1:25 | 1:21 |

Smaller batches gave better accuracy but more weight updates and therefore more
time — the same accuracy-vs-throughput trade-off that reappears in the hardware.

**Optimiser** (lr 0.03, 45 filters, batch 100):

| | SGD | **Momentum** | Adam |
|---|---|---|---|
| train / test | 0.934 / 0.891 | **0.937 / 0.906** | 0.927 / 0.891 |
| time (m:ss) | 2:15 | 1:25 | 1:29 |

Best combination overall: **He init + lr 0.03 + 45 filters + batch 100 + Momentum
+ BatchNorm.**

---

## 3. `convnet_hw.py` — the network the accelerator runs

A VGG-style network designed **for the hardware**, not just for accuracy. Every
channel count is a multiple of 32 so the accelerator's output-channel tile loop
divides evenly, and pooling sits only at the end of a stage, matching VGG's
conv-conv-pool block.

```
Conv1_1    1 ->  32   3x3 pad1   28x28 -> 28x28   BN ReLU
Conv1_2   32 ->  32   3x3 pad1   28x28 -> 28x28   BN ReLU  Pool2 -> 14x14
Conv2_1   32 ->  64   3x3 pad1   14x14 -> 14x14   BN ReLU
Conv2_2   64 ->  64   3x3 pad1   14x14 -> 14x14   BN ReLU  Pool2 ->  7x7
Conv3     64 -> 128   3x3 pad0    7x7  ->  5x5    BN ReLU  Pool2 ->  2x2
Affine   512 ->  10                                        (128 * 2 * 2)
```

**MNIST accuracy with `params_hw.pkl`: 99.13%** (float, all 10,000 test images),
trained by `run_hw.py` (Momentum, He init, lr 0.03, batch 100, 8 epochs on 20,000
images). The Python INT8 inference model scores **99.17%** on the 10,000-image
MNIST test set. This is separate from the board's single-image output-word check.

Note `Conv3` uses **pad 0**, giving `7x7 → 5x5 → pool → 2x2`. Pooling a 5×5 map
gives `int((5-2)/2 + 1) = 2`, so the last row and column are discarded — an
odd-dimension pooling case that later became part of the RTL regression set.

**→ This network, quantised to INT8, is what
[kv260-cnn-accelerator](https://github.com/cksrudsla247/kv260-cnn-accelerator)
runs — a six-layer single-image run whose 928 surviving output words match in
simulation and on a Kria KV260 board.**

### 3.1 `analyze_hw.py` — PE array shape cost model

A cycle and resource model for the accelerator, calibrated against a Task 1
synthesis report, used to choose the PE array shape. Excerpt of
`python analyze_hw.py` (model estimates, not measurements):

**Model version:** the script retains an earlier network with Conv3 pad1 and
Affine input 1152. The final `convnet_hw.py` instead uses Conv3 pad0 and Affine
input 512. The model also uses simplified 1/2-cycle compute and memory assumptions;
the current RTL pixel loop has four states. These estimates explain the earlier
array comparison and are not final RTL cycle counts, board measurements or a
measurement-backed speed-up claim. The historical model is retained so the table
can still be traced to its assumptions.

| array (rows x cols) | PE | model cycles | model LUT (XC7Z020) |
|---|---|---|---|
| 32 x 32 | 1024 | 105,933 | 173% (does not fit) |
| 16 x 32 | 512 | 138,250 | 91% |
| 16 x 16 | 256 | 247,159 | 51% |
| **8 x 32** | **256** | **222,935** | **51%** |
| 32 x 8 | 256 | 319,302 | 51% |

For this model at 256 PEs, 16x16 needs approximately 11% and 32x8 approximately
43% more estimated cycles than 8x32.

---

## 4. Run

```bash
cd 04_CNN
python run_hw.py            # train convnet_hw, save params_hw.pkl, evaluate
python training_convnet.py  # train simple_convnet
python analyze_hw.py        # PE array shape sweep
```

Requires Python 3, NumPy, tqdm and (for `training_convnet.py`) matplotlib. MNIST
is downloaded into `dataset/` on first use.

⚠️ `run_hw.py` overwrites `params_hw.pkl` (it looks for it in the working
directory). The accelerator's golden data is derived from the committed file, so
retraining invalidates every golden value in `kv260-cnn-accelerator`.

---

## Credits

This work started from the code of **"Deep Learning from Scratch"** (Koki Saitoh,
O'Reilly Japan; Korean edition 「밑바닥부터 시작하는 딥러닝」),
<https://github.com/oreilly-japan/deep-learning-from-scratch>, released under the
MIT License:

```
The MIT License (MIT)
Copyright (c) 2016 Koki Saitoh

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

- From the book, largely unchanged: `dataset/mnist.py` (download URL moved to a
  working mirror), the `SGD` / `Momentum` / `Adam` optimisers, the `im2col` /
  `col2im` helpers, and the `Relu`, `Affine` and `SoftmaxWithLoss` layers.
- From the book, extended here: `simple_convnet.py` (BatchNorm added), and the
  training and test loops.
- Project-specific additions and modifications (including AI-assisted code):
  the four-nested-loop `Convolution` / `Pooling`
  forward and backward, `BatchNorm_MLP` / `BatchNorm_CNN` with running
  statistics, `convnet_hw.py`, `run_hw.py`, `analyze_hw.py`.

Other references used for the derivations:
[Backpropagation in convolution](https://steady-programming.tistory.com/93),
[Understanding the backward pass through Batch Normalization](https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html).
