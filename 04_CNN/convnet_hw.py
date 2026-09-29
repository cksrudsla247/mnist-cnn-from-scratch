# coding: utf-8
import pickle
import numpy as np
from collections import OrderedDict
from layers import *


class ConvNetHW:
    """
    VGG-style reduced net for the 32x32 weight-stationary accelerator.

      Conv1_1  1 -> 32   3x3 pad1   28x28 -> 28x28   BN ReLU
      Conv1_2 32 -> 32   3x3 pad1   28x28 -> 28x28   BN ReLU  Pool2 -> 14x14
      Conv2_1 32 -> 64   3x3 pad1   14x14 -> 14x14   BN ReLU
      Conv2_2 64 -> 64   3x3 pad1   14x14 -> 14x14   BN ReLU  Pool2 ->  7x7
      Conv3   64 ->128   3x3 pad0    7x7  ->  5x5    BN ReLU  Pool2 ->  2x2
      Affine  512 -> 10

    Every channel count is a multiple of 32 so the og/ig tile loops divide
    evenly. Pool sits at the end of a stage only, matching VGG's conv-conv-pool
    block.
    """
    # (name, in_ch, out_ch, pad, pool_after)
    CFG = [
        ('Conv1_1',   1,  32, 1, False),
        ('Conv1_2',  32,  32, 1, True ),
        ('Conv2_1',  32,  64, 1, False),
        ('Conv2_2',  64,  64, 1, True ),
        ('Conv3',    64, 128, 0, True ),
    ]
    FC_IN = 128 * 2 * 2      # 512

    def __init__(self, output_size=10):
        self.nconv = len(self.CFG)
        self.params = {}

        for i, (_, cin, cout, _, _) in enumerate(self.CFG, 1):
            scale = np.sqrt(2.0 / (cin * 3 * 3))
            self.params[f'W{i}']     = scale * np.random.randn(cout, cin, 3, 3)
            self.params[f'b{i}']     = np.zeros(cout)
            self.params[f'gamma{i}'] = np.ones(cout)
            self.params[f'beta{i}']  = np.zeros(cout)

        L = self.nconv + 1                      # 6 : the affine layer
        self.params[f'W{L}'] = np.sqrt(2.0/self.FC_IN) * np.random.randn(self.FC_IN, output_size)
        self.params[f'b{L}'] = np.zeros(output_size)

        self.layers = OrderedDict()
        for i, (name, _, _, pad, pool) in enumerate(self.CFG, 1):
            self.layers[f'Conv{i}'] = Convolution(
                self.params[f'W{i}'], self.params[f'b{i}'], stride=1, pad=pad)
            self.layers[f'BatchNorm{i}'] = BatchNorm_CNN(
                self.params[f'gamma{i}'], self.params[f'beta{i}'])
            self.layers[f'Relu{i}'] = Relu()
            if pool:
                self.layers[f'Pool{i}'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine1'] = Affine(self.params[f'W{L}'], self.params[f'b{L}'])
        self.last_layer = SoftmaxWithLoss()

    # ---------------------------------------------------------------- forward
    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if key.startswith('BatchNorm'):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):
        return self.last_layer.forward(self.predict(x, train_flg), t)

    def accuracy(self, x, t, batch_size=200):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        hit = 0
        for i in range(0, x.shape[0], batch_size):
            y = np.argmax(self.predict(x[i:i+batch_size], train_flg=False), axis=1)
            hit += np.sum(y == t[i:i+batch_size])
        return hit / x.shape[0]

    # --------------------------------------------------------------- backward
    def gradient(self, x, t):
        self.loss(x, t, train_flg=True)

        dout = self.last_layer.backward(1)
        for layer in reversed(list(self.layers.values())):
            dout = layer.backward(dout)

        grads = {}
        for i in range(1, self.nconv + 1):
            grads[f'W{i}']     = self.layers[f'Conv{i}'].dW
            grads[f'b{i}']     = self.layers[f'Conv{i}'].db
            grads[f'gamma{i}'] = self.layers[f'BatchNorm{i}'].dgamma
            grads[f'beta{i}']  = self.layers[f'BatchNorm{i}'].dbeta
        L = self.nconv + 1
        grads[f'W{L}'] = self.layers['Affine1'].dW
        grads[f'b{L}'] = self.layers['Affine1'].db
        return grads

    # ------------------------------------------------------------ persistence
    def save_params(self, file_name="params_hw.pkl"):
        p = {k: v for k, v in self.params.items()}
        for i in range(1, self.nconv + 1):
            bn = self.layers[f'BatchNorm{i}']
            p[f'running_mean{i}'] = bn.running_mean
            p[f'running_var{i}']  = bn.running_var
        with open(file_name, 'wb') as f:
            pickle.dump(p, f)

    def load_params(self, file_name="params_hw.pkl"):
        with open(file_name, 'rb') as f:
            p = pickle.load(f)
        for i in range(1, self.nconv + 1):
            for k in (f'W{i}', f'b{i}', f'gamma{i}', f'beta{i}'):
                self.params[k] = p[k]
            self.layers[f'Conv{i}'].W = self.params[f'W{i}']
            self.layers[f'Conv{i}'].b = self.params[f'b{i}']
            bn = self.layers[f'BatchNorm{i}']
            bn.gamma = self.params[f'gamma{i}']
            bn.beta  = self.params[f'beta{i}']
            bn.running_mean = p[f'running_mean{i}']
            bn.running_var  = p[f'running_var{i}']
        L = self.nconv + 1
        self.params[f'W{L}'] = p[f'W{L}']
        self.params[f'b{L}'] = p[f'b{L}']
        self.layers['Affine1'].W = self.params[f'W{L}']
        self.layers['Affine1'].b = self.params[f'b{L}']

    # ------------------------------------------------------------ shape check
    def shape_report(self, x_shape=(1, 1, 28, 28)):
        """Run one dummy image and print every layer's output shape."""
        x = np.zeros(x_shape)
        rows = []
        for key, layer in self.layers.items():
            x = layer.forward(x, False) if key.startswith('BatchNorm') else layer.forward(x)
            rows.append((key, tuple(x.shape)))
        return rows