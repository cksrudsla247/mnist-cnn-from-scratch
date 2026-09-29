# coding: utf-8
"""
run_hw.py

Train the accelerator's reference network and evaluate it in one shot.

Prints, in order:
  1. the per-layer output shape table (catches a wrong pad / pool early)
  2. the training curve, one line per epoch
  3. accuracy after reloading params_hw.pkl
  4. accuracy at batch_size = 1   <-- the number that matters for the RTL
  5. per-image predictions and a misclassification summary

Requires convnet_hw.py (6-layer VGG-style net) and the running-stats version
of BatchNorm_CNN in layers.py.
"""
import sys, os, time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataset.mnist import load_mnist
from convnet_hw import ConvNetHW
from optimization import Momentum
from tqdm import tqdm

np.random.seed(41)

# ------------------------------------------------------------------ settings
TRAIN_N = 20000        # training images actually used
TEST_N  = 2000         # quick eval set used between epochs
BATCH   = 100
EPOCHS  = 8
LR      = 0.03
PKL     = "params_hw.pkl"

# --------------------------------------------------------------------- data
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, one_hot_label=True)
x_tr, t_tr = x_train[:TRAIN_N], t_train[:TRAIN_N]
x_te, t_te = x_test[:TEST_N],  t_test[:TEST_N]

net = ConvNetHW(output_size=10)

# ------------------------------------------------------------ shape report
print("=" * 64)
print("  layer                     output shape")
print("-" * 64)
try:
    for name, shp in net.shape_report():
        print(f"  {name:<22}    {shp}")
except AttributeError:
    # older convnet_hw.py without shape_report : do it inline
    x = np.zeros((1, 1, 28, 28))
    for key, layer in net.layers.items():
        x = layer.forward(x, False) if key.startswith('BatchNorm') else layer.forward(x)
        print(f"  {key:<22}    {tuple(x.shape)}")

nparam = sum(v.size for k, v in net.params.items() if k[0] in 'Wb')
print("-" * 64)
print(f"  weights + biases : {nparam:,}")
print("=" * 64)

# -------------------------------------------------------------------- train
opt = Momentum(lr=LR, momentum=0.9)
iter_per_epoch = max(TRAIN_N // BATCH, 1)
iters_num = EPOCHS * iter_per_epoch

loss_list, tr_acc_list, te_acc_list = [], [], []
t0 = time.time()

for i in tqdm(range(iters_num), ncols=78):
    mask = np.random.choice(TRAIN_N, BATCH)
    opt.update(net.params, net.gradient(x_tr[mask], t_tr[mask]))
    loss_list.append(net.last_layer.loss)          # gradient() already computed it

    if (i + 1) % iter_per_epoch == 0:
        ep = (i + 1) // iter_per_epoch
        tr = net.accuracy(x_tr[:2000], t_tr[:2000])
        te = net.accuracy(x_te, t_te)
        tr_acc_list.append(tr)
        te_acc_list.append(te)
        avg = float(np.mean(loss_list[-iter_per_epoch:]))
        tqdm.write(f"  epoch {ep}/{EPOCHS}   loss {avg:.4f}   "
                   f"train {tr:.4f}   test {te:.4f}")

train_time = time.time() - t0
net.save_params(PKL)

print("=" * 64)
print(f"  trained in {train_time:.1f} s   ->  {PKL} saved")
print("=" * 64)

# ------------------------------------------------- reload and run inference
net2 = ConvNetHW(output_size=10)
net2.load_params(PKL)

labels = np.argmax(t_test, axis=1)

acc_full = net2.accuracy(x_test, labels)
acc_b1   = net2.accuracy(x_test[:200], labels[:200], batch_size=1)
acc_b200 = net2.accuracy(x_test[:200], labels[:200], batch_size=200)

print(f"  train acc (last epoch) : {tr_acc_list[-1]:.4f}")
print(f"  test  acc (last epoch) : {te_acc_list[-1]:.4f}")
print(f"  reloaded, full 10000   : {acc_full:.4f}")
print("-" * 64)
print("  batch-size independence check (BN running stats)")
print(f"    first 200, batch=200 : {acc_b200:.4f}")
print(f"    first 200, batch=1   : {acc_b1:.4f}")
print("    -> these two MUST match; the accelerator runs B=1")
print("=" * 64)

# ------------------------------------------------------ per-image inference
print("  idx  label  pred        top-3 logits")
print("-" * 64)
for idx in range(10):
    logit = net2.predict(x_test[idx:idx+1], train_flg=False)[0]
    pred = int(np.argmax(logit))
    top3 = np.argsort(-logit)[:3]
    mark = "OK" if pred == labels[idx] else "XX"
    detail = "  ".join(f"{c}:{logit[c]:7.2f}" for c in top3)
    print(f"  {idx:3d}    {labels[idx]}      {pred}   {mark}   {detail}")
print("=" * 64)

# ----------------------------------------------------- misclassification set
pred_all = np.concatenate([
    np.argmax(net2.predict(x_test[i:i+200], train_flg=False), axis=1)
    for i in range(0, x_test.shape[0], 200)
])
wrong = np.where(pred_all != labels)[0]
print(f"  misclassified {len(wrong)} / {x_test.shape[0]}"
      f"   ({100.0*len(wrong)/x_test.shape[0]:.2f} %)")
if len(wrong):
    print("  first 10 (idx, label, pred) :",
          [(int(i), int(labels[i]), int(pred_all[i])) for i in wrong[:10]])
print("=" * 64)