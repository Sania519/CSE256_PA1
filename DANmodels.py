import torch
import torch.nn as nn
from sentiment_data import *
from utils import Indexer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence

# word_embeddings = read_word_embeddings("data/glove.6B.50d-relativized.txt")

class DeepAveragingNetwork(nn.Module):
    def __init__(self, word_embeddings, input_dim, hidden_dim, num_classes, num_layers=2, randomise=False):
        super(DeepAveragingNetwork, self).__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        if randomise:
            self.embedded_layer = nn.Embedding(14923, input_dim)
        else:
            self.embedded_layer = word_embeddings.get_initialized_embedding_layer(frozen=False)
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.Relu = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.num_classes)
        self.softmax = nn.LogSoftmax()
        self.dropout = nn.Dropout(p=0.5)
        self.feedforward = nn.Sequential(self.fc1, self.Relu, self.dropout,self.fc2, self.Relu, self.dropout, self.fc3, self.softmax)
        # self.feedforward = nn.Sequential(self.fc1, self.Relu,self.fc2, self.Relu, self.dropout, self.fc3, self.softmax)
        # self.feedforward = nn.Sequential(self.fc1, self.Relu, self.dropout, self.fc3, self.softmax)
        # self.feedforward = nn.Sequential(self.fc1, self.Relu,self.fc2, self.Relu, self.fc3, self.softmax)

    def forward(self, x):
        x = self.embedded_layer(x)
        x = torch.mean(x,dim=1)
        output = self.feedforward(x)
        return output

