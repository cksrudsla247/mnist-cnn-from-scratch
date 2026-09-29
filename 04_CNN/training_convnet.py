# coding: utf-8
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # allow imports from the parent directory
import numpy as np
from dataset.mnist import load_mnist    
from simple_convnet import SimpleConvNet
from optimization import Momentum
from tqdm import tqdm 
import matplotlib.pyplot as plt


# load the data
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, one_hot_label = True)

network = SimpleConvNet(input_dim=(1,28,28), 
                       conv_param = {'filter_num': 45, 'filter_size': 5, 'pad': 0, 'stride': 1},
                       hidden_size=100, output_size=10, weight_init_std=0.01)

x_train_sample = x_train[:5000] 
t_train_sample = t_train[:5000]
x_test_sample = x_test[:1000]
t_test_sample = t_test[:1000]

train_size = x_train_sample.shape[0]
batch_size = 100
#optimizer = SGD(lr=0.03)
optimizer = Momentum(lr=0.03, momentum = 0.9)
#optimizer = Nesterov(lr=0.01, momentum=0.9)
#optimizer = AdaGrad(lr = 0.01)
#optimizer = RMSprop(lr=0.01, decay_rate = 0.99)
#optimizer = Adam(lr=0.003, beta1=0.9, beta2=0.999)


#learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

epochs = 5
iter_per_epoch = int(max(train_size / batch_size, 1))
iters_num = int(epochs * iter_per_epoch)

for i in tqdm(range(iters_num)):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train_sample[batch_mask]
    t_batch = t_train_sample[batch_mask]
    
    # compute gradients
    #grad = network.numerical_gradient(x_batch, t_batch) # numerical differentiation
    grad = network.gradient(x_batch, t_batch) # backpropagation
    
    # update
    
    optimizer.update(network.params, grad)
    
    #for key in ('W1', 'b1', 'W2', 'b2'):
    #    network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train_sample, t_train_sample)
        test_acc = network.accuracy(x_test_sample, t_test_sample)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(" Train Accuracy: ", round(train_acc,3), " Test Accuracy: ", round(test_acc,3))
        
print("=============== Final Result ===============")
final_train_acc = train_acc_list[-1]
final_test_acc = test_acc_list[-1]
print(f" Final Train Acc: {final_train_acc:.3f}")
print(f" Final Test Acc: {final_test_acc:.3f}")
