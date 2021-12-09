import torch.nn as nn
from torch.autograd import Variable

class MyEmbeddingDropout(nn.Module):
    def forward(self, embedding, dropout=0.2):
        """
        @param: embedding[tensor] : the embedding of a batch,
                                    should be (batch_size, seq_len, embedding_size)
        """
        # if it's not in the training mode or 0 dropout required, directly return
        if (not self.training) or (dropout == 0):
            return embedding
        
        # change embedding to (seq_len, batch_size, embedding_size)
        embedding = torch.permute(embedding, (1, 0, 2))
        
        # construct a mask of size (embedding_size) that will be applied to every timestamp
        # the a single word will disappear completely from a sequence
        mask = Variable(embedding.data.new(1, embedding.shape[1], embedding.shape[2]).bernoulli(p=1-dropout) / (1-dropout), requires_grad=False)
        
        # masking off the same word in a whole sequence and permute back the dimension
        return torch.permute(mask * embedding, (1, 0, 2))
