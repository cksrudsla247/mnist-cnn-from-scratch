# coding: utf-8
import sys, os
import pickle
import numpy as np
from collections import OrderedDict
from layers import *
from tqdm import tqdm
class SimpleConvNet:
    """Simple convolutional network
    
    conv - relu - pool - affine - relu - affine - softmax
    
    Parameters
    ----------
    input_size : input size (784 for MNIST)
    hidden_size_list : neurons per hidden layer (e.g. [100, 100, 100])
    output_size : output size (10 for MNIST)
    activation : activation function - 'relu' or 'sigmoid'
    weight_init_std : weight standard deviation (e.g. 0.01)
        'relu' or 'he' selects He initialisation
        'sigmoid' or 'xavier' selects Xavier initialisation
    """
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # weight initialisation
        self.params = {}
        scale1 = np.sqrt(2.0 / (input_dim[0] * filter_size * filter_size))
        #self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['W1'] = scale1 * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['gamma1'] = np.ones(filter_num) 
        self.params['beta1'] = np.zeros(filter_num)
        
        scale2 = np.sqrt(2.0 / pool_output_size)
        #self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['W2'] = scale2 * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['gamma2'] = np.ones(hidden_size)
        self.params['beta2'] = np.zeros(hidden_size) 
        
        
        scale3 = np.sqrt(2.0 / hidden_size)
        #self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['W3'] = scale3 * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)        

        # build the layers
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['BatchNorm1'] = BatchNorm_CNN(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['BatchNorm2'] = BatchNorm_MLP(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        #for layer in tqdm(self.layers.values()):
            #x = layer.forward(x)
        
        return x

    def loss(self, x, t):
        
        y = self.predict(x)
        return self.last_layer.forward(y,t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t,axis=1)
        
        batch_size = x.shape[0]
        accuracy = np.sum(y==t)/batch_size
        return accuracy
    
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
    
    def gradient(self, x, t):
        
        self.loss(x,t)
        
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
            
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].db
        grads['gamma1'] = self.layers['BatchNorm1'].dgamma
        grads['beta1'] = self.layers['BatchNorm1'].dbeta

        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].db
        grads['gamma2'] = self.layers['BatchNorm2'].dgamma
        grads['beta2'] = self.layers['BatchNorm2'].dbeta
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].db
        
        return grads