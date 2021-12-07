import numpy as np
import os
import sys
from loss import *
from activation import *
from linear import *
from conv import *


class CNN_SimpleScanningMLP():
    def __init__(self):
        # <---------------------
        # given 128 x 24 input (128 time steps, with a 24-dimensional vector
        # MLP evaulates 8 contiguous input vectors at a time
        # MLP "stride" forward 4 time instants, no pad
        # MLP has 3 layers, 8 neurons in 1st layer, 16 in the second layer, 4 in third layer. No bias term
        # Architecture: [Flatten(), Linear(8 * 24, 8), ReLU(), Linear(8, 16), ReLU(), Linear(16, 4)]
        self.conv1 = Conv1D(in_channel=24, out_channel=8, kernel_size=8, stride=4)
        self.conv2 = Conv1D(in_channel=8, out_channel=16, kernel_size=1, stride=1)
        self.conv3 = Conv1D(in_channel=16, out_channel=4, kernel_size=1, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]
        

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN
        # (192, 8) (8, 16) (16, 4)
        # self.W = weight_init_fn(out_channel, in_channel, kernel_size)
        w1,w2,w3 = weights

        H1 = w1.flatten(order='F').reshape(8, 24, 8)
        H2 = w2.flatten(order='F').reshape(16, 8, 1)
        H3 = w3.flatten(order='F').reshape(4, 16, 1)
        for i in range(8):
            H1[i] = H1[i].flatten().reshape((24,8), order='F')
        for i in range(16):
            H2[i] = H2[i].flatten().reshape((8, 1), order='F')
        for i in range(4):
            H3[i] = H3[i].flatten().reshape((16,1), order='F')
        
        self.conv1.W = H1
        self.conv2.W = H2
        self.conv3.W = H3

    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta


class CNN_DistributedScanningMLP():
    def __init__(self):
        self.conv1 = Conv1D(in_channel=24, out_channel=2, kernel_size=2, stride=2)
        self.conv2 = Conv1D(in_channel=2, out_channel=8, kernel_size=2, stride=2)
        self.conv3 = Conv1D(in_channel=8, out_channel=4, kernel_size=2, stride=1)
        self.layers = [self.conv1, ReLU(), self.conv2, ReLU(), self.conv3, Flatten()]

    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def init_weights(self, weights):
        # Load the weights for your CNN from the MLP Weights given
        # w1, w2, w3 contain the weights for the three layers of the MLP
        # Load them appropriately into the CNN

        w1, w2, w3 = weights
        w1 = w1[:2*24, :2]
        w2 = w2[:4,:8]
        w3 = w3
        H1 = w1.flatten(order='F').reshape(2, 24, 2)
        H2 = w2.flatten(order='F').reshape(8, 2, 2)
        H3 = w3.flatten(order='F').reshape(4, 8, 2)
        for i in range(2):
            H1[i] = H1[i].flatten().reshape((24,2), order='F')
        for i in range(8):
            H2[i] = H2[i].flatten().reshape((2, 2), order='F')
        for i in range(4):
            H3[i] = H3[i].flatten().reshape((8,2), order='F')
        self.conv1.W = H1
        self.conv2.W = H2
        self.conv3.W = H3

    def forward(self, x):
        """
        Do not modify this method

        Argument:
            x (np.array): (batch size, in channel, in width)
        Return:
            out (np.array): (batch size, out channel , out width)
        """

        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, delta):
        """
        Do not modify this method

        Argument:
            delta (np.array): (batch size, out channel, out width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """

        for layer in self.layers[::-1]:
            delta = layer.backward(delta)
        return delta

class CNN(object):

    """
    A simple convolutional neural network

    Here you build implement the same architecture described in Section 3.3
    You need to specify the detailed architecture in function "get_cnn_model" below
    The returned model architecture should be same as in Section 3.3 Figure 3
    """

    def __init__(self, input_width, num_input_channels, num_channels, kernel_sizes, strides,
                 num_linear_neurons, activations, conv_weight_init_fn, bias_init_fn,
                 linear_weight_init_fn, criterion, lr):
        """
        input_width           : int    : The width of the input to the first convolutional layer
        num_input_channels    : int    : Number of channels for the input layer
        num_channels          : [int]  : List containing number of (output) channels for each conv layer
        kernel_sizes          : [int]  : List containing kernel width for each conv layer
        strides               : [int]  : List containing stride size for each conv layer
        num_linear_neurons    : int    : Number of neurons in the linear layer
        activations           : [obj]  : List of objects corresponding to the activation fn for each conv layer
        conv_weight_init_fn   : fn     : Function to init each conv layers weights
        bias_init_fn          : fn     : Function to initialize each conv layers AND the linear layers bias to 0
        linear_weight_init_fn : fn     : Function to initialize the linear layers weights
        criterion             : obj    : Object to the criterion (SoftMaxCrossEntropy) to be used
        lr                    : float  : The learning rate for the class

        You can be sure that len(activations) == len(num_channels) == len(kernel_sizes) == len(strides)
        """

        self.train_mode = True
        self.nlayers = len(num_channels)

        self.activations = activations
        self.criterion = criterion

        self.lr = lr
        self.convolutional_layers = []
        last_out_channel = num_input_channels
        last_out = input_width

        for out_channel, kernel_size, stride in zip(num_channels, kernel_sizes, strides):
            cnn_layer = Conv1D(last_out_channel, out_channel, kernel_size, stride, conv_weight_init_fn, bias_init_fn)
            self.convolutional_layers.append(cnn_layer)
            last_out_channel = out_channel
            last_out = (last_out - kernel_size) // stride + 1

        linear_in_channel = last_out * last_out_channel
        self.flatten = Flatten()
        self.linear_layer = Linear(linear_in_channel, num_linear_neurons, linear_weight_init_fn, bias_init_fn)
        


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, num_input_channels, input_width)
        Return:
            out (np.array): (batch_size, num_linear_neurons)
        """
        self.output = x
        for cnn_layer, activation_layer in zip(self.convolutional_layers, self.activations):
            self.output = cnn_layer(self.output)
            self.output = activation_layer(self.output)
        self.output = self.flatten(self.output)
        self.output = self.linear_layer(self.output)
        return self.output

    def backward(self, labels):
        """
        Argument:
            labels (np.array): (batch_size, num_linear_neurons)
        Return:
            grad (np.array): (batch size, num_input_channels, input_width)
        """

        m, _ = labels.shape
        self.loss = self.criterion(self.output, labels).sum()
        grad = self.criterion.derivative()
        grad = self.linear_layer.backward(grad)
        grad = self.flatten.backward(grad)
        for cnn_layer, activation_layer in zip(self.convolutional_layers[::-1], self.activations[::-1]):
            grad *= activation_layer.derivative()
            grad = cnn_layer.backward(grad)
        return grad


    def zero_grads(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].dW.fill(0.0)
            self.convolutional_layers[i].db.fill(0.0)

        self.linear_layer.dW.fill(0.0)
        self.linear_layer.db.fill(0.0)

    def step(self):
        # Do not modify this method
        for i in range(self.nlayers):
            self.convolutional_layers[i].W = (self.convolutional_layers[i].W -
                                              self.lr * self.convolutional_layers[i].dW)
            self.convolutional_layers[i].b = (self.convolutional_layers[i].b -
                                  self.lr * self.convolutional_layers[i].db)

        self.linear_layer.W = (self.linear_layer.W - self.lr * self.linear_layers.dW)
        self.linear_layers.b = (self.linear_layers.b -  self.lr * self.linear_layers.db)


    def __call__(self, x):
        # Do not modify this method
        return self.forward(x)

    def train(self):
        # Do not modify this method
        self.train_mode = True

    def eval(self):
        # Do not modify this method
        self.train_mode = False
