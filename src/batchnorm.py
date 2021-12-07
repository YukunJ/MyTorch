import numpy as np

class BatchNorm(object):

    def __init__(self, in_feature, alpha=0.9):
        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, in_feature))
        self.mean = np.zeros((1, in_feature))

        self.gamma = np.ones((1, in_feature))
        self.dgamma = np.zeros((1, in_feature))

        self.beta = np.zeros((1, in_feature))
        self.dbeta = np.zeros((1, in_feature))

        # inference parameters
        self.running_mean = np.zeros((1, in_feature))
        self.running_var = np.ones((1, in_feature))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):
        """
        Argument:
            x (np.array): (batch_size, in_feature)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, in_feature)

        NOTE: The eval parameter is to indicate whether we are in the
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        self.x = x
        if eval:
            self.norm = (x - self.running_mean[None, :]) / np.sqrt(self.running_var+self.eps)[None, :]
            self.out = np.squeeze(self.gamma * self.norm + self.beta)
            return self.out
            
        self.mean = np.mean(x, axis=0)
        self.var = np.var(x, axis=0)
        
        self.norm = (x - self.mean[None, :]) / np.sqrt(self.var+self.eps)[None, :]
        self.out = self.gamma * self.norm + self.beta

        self.running_mean = self.alpha * self.running_mean + (1-self.alpha) * self.mean
        self.running_var = self.alpha * self.running_var +(1-self.alpha) * self.var

        return self.out


    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in feature)
        Return:
            out (np.array): (batch size, in feature)
        """
        # get batch size
        batch_size = delta.shape[0]
        
        # repeative usage
        sqrt_var_eps = np.sqrt(self.var+self.eps)
        
        # find the beta and gamma derivatives for later gradient descend
        self.dbeta = np.sum(delta, axis=0, keepdims=True)
        self.dgamma = np.sum(delta * self.norm, axis=0, keepdims=True)
        
        # find the graident of the norm
        self.dnorm = self.gamma * delta
        
        # derivative of the variance
        self.dvar = -0.5 * (np.sum(self.dnorm*(self.x-self.mean[None, :])/(sqrt_var_eps)**3,axis=0))
        
        # derivative of the mean
        first_term_dmean = -(np.sum(self.dnorm/sqrt_var_eps, axis=0))
        second_term_dmean = -(2/batch_size) * (self.dvar) * (np.sum(self.x-self.mean[None, :], axis=0))
        self.dmean = first_term_dmean + second_term_dmean
        
        # final answer calculation
        first = self.dnorm/sqrt_var_eps
        second = self.dvar * (2/batch_size) * (self.x-self.mean[None, :])
        third = self.dmean / batch_size
        self.out = first + second + third
        
        return self.out
