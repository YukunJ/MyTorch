import numpy as np
import os

class Criterion(object):
    """
    Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

class SoftmaxCrossEntropy(Criterion):
    """
    Softmax loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Argument:
            x (np.array): (batch size, X)
            y (np.array): (batch size, X)
        Return:
            out (np.array): (batch size, )
        """
        self.logits = x
        self.labels = y

        # use LogSumExp trick to ensure numerical stability
        logits_extreme = np.max(self.logits, axis=1)
        logits_normize = self.logits - logits_extreme[:, None]
        exp_normize = np.exp(logits_normize)
        
        # use intermediate variable to store the probability predicton
        self.probability = exp_normize /  np.sum(exp_normize, axis=1)[:, None]
        
        self.loss = -np.sum(y * np.log(self.probability), axis=1)
        return self.loss

    def derivative(self):
        """
        Return:
            out (np.array): (batch size, X)
        """

        return self.probability - self.labels
