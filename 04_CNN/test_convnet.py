# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # allow imports from the parent directory
import numpy as np
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet

class Tester:
    """Runs inference with a trained network
    """
    def __init__(self, network, x_test, t_test, verbose=True):
        self.network = network
        self.x_test = x_test
        self.t_test = t_test
        self.verbose = verbose
        self.test_acc_list = []

    def test(self):
        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

# load the data
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = SimpleConvNet(input_dim=(1,28,28), 
                       conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                       hidden_size=100, output_size=10, weight_init_std=0.01)

x_test = x_test[:3000]
t_test = t_test[:3000]

tester = Tester(network, x_test, t_test)

network.load_params("params.pkl")
tester.test()

