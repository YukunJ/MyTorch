import sys
import numpy as np
from matplotlib import pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tests import test_prediction, test_generation
from locked_dropout import MyLockedDropout
from embedding_dropout import MyEmbeddingDropout
from tqdm import tqdm

class LanguageModelDataLoader(DataLoader):
    """
        TODO: Define data loader logic here
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.dataset = dataset
        self.shuffle = shuffle

    def __iter__(self):
        # concatenate your articles and build into batches
        chunk_X, chunk_Y = self.build_batches()
        for X, Y in zip(chunk_X, chunk_Y):
            yield torch.Tensor(X).long(), torch.Tensor(Y).long()
    
    def get_totalBatchs(self):
        chunk_X, chunk_Y = self.build_batches()
        return len(chunk_X)
    
    def build_batches(self, seq_len=5):
        # build two lists of train_seq and target_seq in batches
        start = time.time()
        if self.shuffle:
            articles = np.random.permutation(self.dataset)
        else:
            articles = self.dataset
        X, Y = [], []
        for article in articles:
            n = len(article)
            if n <= seq_len:
                # not long enough to generate a training sample
                continue
            for begin in range(n-seq_len):
                X.append(article[begin:begin+seq_len])
                Y.append(article[begin+1:begin+1+seq_len])
        chunk_X, chunk_Y = [], []
        for i in range(0, len(X), self.batch_size):
            chunk_X.append(X[i:i+self.batch_size])
            chunk_Y.append(Y[i:i+self.batch_size])
        return chunk_X, chunk_Y
        
# model
class LanguageModel(nn.Module):
    """
        TODO: Define your model here
        (In the paper, it says using three-layer LSTM model with 1150 units in the hidden layer
        and an embedding size of 400. All embedding weights were uniformly initialized in the interval
        [-0.1, 0.1], and all other weights were initialized between [-1/sqrt(H), 1/sqrt(H)], where H=hidden size)
    """
    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.num_layers = 3
        self.hidden_size = 1150
        self.embedding_size = 400
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embedding_size)
        self.linear_layer = nn.Linear(self.hidden_size, self.vocab_size)
        self.lstm_layers = []
        self.locked_dropout = MyLockedDropout()
        self.embedding_dropout = MyEmbeddingDropout()
        for i in range(self.num_layers):
            if i == 0:
                self.lstm_layers.append(nn.LSTM(self.embedding_size, self.hidden_size, num_layers=1, batch_first=True))
            else:
                self.lstm_layers.append(nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True))
        self.lstm_layers = nn.Sequential(*self.lstm_layers)
        self.init_weight()
        print("Finishing initialization of [LanguageModel] class with {} lstm layers with embed_size={} and hidden={} of vocab={}".\
              format(self.num_layers, self.embedding_size, self.hidden_size, self.vocab_size))
        
    def forward(self, x, require_hidden=False, input_lengths=None):
        # x should be (batch_size, seq_len)
        x = self.embedding(x)
        x = self.embedding_dropout(x)
        last_hidden = []
        # x should be (batch_size, seq_len, embedding_size)
        for i, lstm_layer in enumerate(self.lstm_layers):
            x, (h, c) = lstm_layer(x)
            if require_hidden:
                last_hidden.append((h, c))
            if i != len(self.lstm_layers)-1:
                x = self.locked_dropout(x)
        # x should be (batch_size, seq_len, hidden_size)
        output = self.linear_layer(x)
        # x should be (batch_size, seq_len, vocab_size)
        if input_lengths is not None:
            return output, input_lengths
        if require_hidden:
            return output, last_hidden
        return output
    
    def forward_one_timestep(self, x, last_hidden):
        x = self.embedding(x)
        next_hidden = []
        for i, lstm_layer in enumerate(self.lstm_layers):
            x, (h, c) = lstm_layer(x, last_hidden[i])
            next_hidden.append((h, c))
        output = self.linear_layer(x)
        return output, next_hidden
        
    def init_weight(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear_layer.weight.data.uniform_(-1/np.sqrt(self.hidden_size), 1/np.sqrt(self.hidden_size))
        for lstm_layer in self.lstm_layers:
            for weight in lstm_layer._all_weights:
                if 'weight' in weight:
                    nn.init.uniform_(lstm_layer.__getattr__(weight), -1/np.sqrt(self.hidden_size), 1/np.sqrt(self.hidden_size))
                    
# model trainer
class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):
        """
            Use this class to train your model
        """
        # feel free to add any other parameters here
        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id
        
        # TODO: Define your optimizer and criterion here
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4, weight_decay=5e-6)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        self.model.train() # set to training mode
        epoch_loss = 0
        num_batches = 0
        for batch_num, (inputs, targets) in enumerate(tqdm(self.loader, total=self.loader.get_totalBatchs())):
            epoch_loss += float(self.train_batch(inputs, targets))
        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs + 1, self.max_epochs, epoch_loss))
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        """
            TODO: Define code for training a single batch of inputs
        
        """
        self.optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        logits = self.model(inputs)
        # logits: (batch_size, seq_len, vocab_len)
        # targets: (batch_size, seq_len)
        logits = torch.permute(logits, (0, 2, 1))
        loss = self.criterion(logits, targets)
        """
        loss = None
        for i in range(targets.shape[1]):
            if loss is None:
                loss = self.criterion(logits[:, i, :], targets[:, i])
            else:
                loss += self.criterion(logits[:, i, :], targets[:, i])
        """
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def test(self):
        # don't change these
        self.model.eval() # set to eval mode
        predictions = TestLanguageModel.prediction(fixtures_pred['inp'], self.model) # get predictions
        self.predictions.append(predictions)
        generated_logits = TestLanguageModel.generation(fixtures_gen, 10, self.model) # generated predictions for 10 words
        generated_logits_test = TestLanguageModel.generation(fixtures_gen_test, 10, self.model)
        nll = test_prediction(predictions, fixtures_pred['out'])
        generated = test_generation(fixtures_gen, generated_logits, vocab)
        generated_test = test_generation(fixtures_gen_test, generated_logits_test, vocab)
        self.val_losses.append(nll)
        
        self.generated.append(generated)
        self.generated_test.append(generated_test)
        self.generated_logits.append(generated_logits)
        self.generated_logits_test.append(generated_logits_test)
        
        # generate predictions for test data
        predictions_test = TestLanguageModel.prediction(fixtures_pred_test['inp'], self.model) # get predictions
        self.predictions_test.append(predictions_test)
            
        print('[VAL]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs + 1, self.max_epochs, nll))
        return nll

    def save(self):
        # don't change these
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pkl'.format(self.epochs))
        torch.save({'state_dict': self.model.state_dict()},
            model_path)
        np.save(os.path.join('experiments', self.run_id, 'predictions-{}.npy'.format(self.epochs)), self.predictions[-1])
        np.save(os.path.join('experiments', self.run_id, 'predictions-test-{}.npy'.format(self.epochs)), self.predictions_test[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-{}.npy'.format(self.epochs)), self.generated_logits[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-test-{}.npy'.format(self.epochs)), self.generated_logits_test[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}-test.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated_test[-1])

class TestLanguageModel:
    def prediction(inp, model):
        """
            TODO: write prediction code here
            
            :param inp: [batch_size, seq_len]
            :return: a np.ndarray of logits
        """
        inputs = torch.Tensor(inp).long().to(device)
        logits = model(inputs)[:, -1, :]
        return logits.detach().cpu().numpy()

        
    def generation(inp, forward, model):
        """
            TODO: write generation code here

            Generate a sequence of words given a starting sequence.
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """
        batch_size = len(inp)
        seq_len = len(inp[0])
        generated_words = torch.zeros((batch_size, forward), device=device)
        inputs = torch.Tensor(inp).long().to(device)
        output, last_hidden = model.forward(inputs, require_hidden=True)
        logits = output[:, -1, :] # of shape (batch_size, vocab_size)
        inputs = torch.argmax(logits, dim=1, keepdim=True)
        for i in range(forward):
            output, last_hidden = model.forward_one_timestep(inputs, last_hidden)
            logits = output[:, -1, :]
            inputs = torch.argmax(logits, dim=1, keepdim=True)
            generated_words[:, i] = inputs.squeeze()
        return generated_words.cpu().numpy().astype(np.int)
