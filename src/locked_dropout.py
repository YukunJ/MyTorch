import torch.nn as nn
from torch.autograd import Variable

class MyLockedDropout(nn.Module):
    def forward(self, x, dropout=0.2):
        """
        @param: x[tensor] : the tensor we want to apply Locked Dropout upon
                            should be of shape (batch_size, seq_len, frequency)
        """
        # if it's not in the training mode or 0 dropout required, directly return
        if (not self.training) or (dropout == 0):
            return x
        
        # change x to (seq_len, batch_size, frequency)
        x = torch.permute(x, (1, 0, 2))
        
        # construct a mask of size (frequency) that will be applied to every timestamp
        # keep the auxiliary axis for broadcasting
        mask = Variable(x.data.new(1, x.shape[1], x.shape[2]).bernoulli(p=1-dropout) / (1-dropout), requires_grad=False)
        
        # masking off samely for every timestamp and permute back the dimension
        return torch.permute(mask * x, (1, 0, 2))
