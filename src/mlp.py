import numpy as np
import os
import sys
from loss import *
from activation import *
from batchnorm import *
from linear import *


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn,
                 bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        hiddens = [input_size] + hiddens + [output_size]
        self.linear_layers = [Linear(in_size, out_size, weight_init_fn, bias_init_fn) for in_size, out_size in zip(hiddens, hiddens[1:])]

        # If batch norm, add batch norm layers into the list 'self.bn_layers'
        if self.bn:
            self.bn_layers = [BatchNorm(in_size) for in_size in hiddens[1:1+self.num_bn_layers]]


    def forward(self, x):
        """
        Argument:
            x (np.array): (batch size, input_size)
        Return:
            out (np.array): (batch size, output_size)
        """
        # Complete the forward pass through entire MLP.
        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)
            if self.bn and i < self.num_bn_layers:
                x = self.bn_layers[i](x, eval= not self.train_mode)
            x = self.activations[i](x)
        self.out = x
        return self.out

    def zero_grads(self):
        # Use numpyArray.fill(0.0) to zero out your backpropped derivatives in each
        # of your linear and batchnorm layers.
        for i in range(len(self.linear_layers)):
            layer = self.linear_layers[i]
            layer.dW.fill(0.0)
            layer.db.fill(0.0)
        if self.bn:
            for i in range(len(self.bn_layers)):
                bn_layer = self.bn_layers[i]
                bn_layer.dgamma.fill(0.0)
                bn_layer.dbeta.fill(0.0)

    def step(self):
        # Apply a step to the weights and biases of the linear layers.
        # Apply a step to the weights of the batchnorm layers.
        # (You will add momentum later in the assignment to the linear layers only
        # not the batchnorm layers)

        for i in range(len(self.linear_layers)):
            # Update weights and biases here
            layer = self.linear_layers[i]
            if layer.momentum_W.any():
                layer.momentum_W = self.momentum * layer.momentum_W - self.lr * layer.dW
                layer.momentum_b = self.momentum * layer.momentum_b - self.lr * layer.db
            else:
                layer.momentum_W =  - self.lr * layer.dW
                layer.momentum_b =  - self.lr * layer.db
            layer.W += layer.momentum_W
            layer.b += layer.momentum_b
        # Do the same for batchnorm layers
        if self.bn:
            for i in range(len(self.bn_layers)):
                bn_layer = self.bn_layers[i]
                bn_layer.gamma -= self.lr * bn_layer.dgamma
                bn_layer.beta -= self.lr * bn_layer.dbeta

    def backward(self, labels):
        # Backpropagate through the activation functions, batch norm and
        # linear layers.
        # Be aware of which return derivatives and which are pure backward passes
        # i.e. take in a loss w.r.t it's output.
        self.criterion.forward(self.out, labels)
        loss = self.criterion.derivative()
        for i in range(len(self.linear_layers)-1, -1, -1):
            loss *= self.activations[i].derivative()
            if self.bn and 0 <= i < self.num_bn_layers:
                loss = self.bn_layers[i].backward(loss)
            loss = self.linear_layers[i].backward(loss)
        
    def error(self, labels):
        return (np.argmax(self.output, axis = 1) != np.argmax(labels, axis = 1)).sum()

    def total_loss(self, labels):
        return self.criterion(self.output, labels).sum()

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False
