import torch
import torch.nn as nn
from sentiment_data import *
from utils import Indexer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import re
from collections import defaultdict


def read_data(file_path):
    data = []
    train_exs = read_sentiment_examples(file_path)
    for exs in train_exs:
        # print(exs.words)
        data += exs.words
    return data

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = ' '.join(pair)
    p = ' '.join(pair)
    for word in v_in:
        w_out = word.replace(bigram, p)
        v_out[w_out] = v_in[word]
    return v_out

def get_vocab(data):
    vocab = defaultdict(int)
    for word in data:
            vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def byte_pair_encoding(data, n):
    vocab = get_vocab(data)
    merged_pairs = {}
    for i in range(n):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merged_pairs[best] = ''.join(best)
    unique_tokens = sorted(set(token for word in vocab for token in word.split()))
    return merged_pairs, unique_tokens

def encode(text, merged_pairs, token_to_index):
    encoded = []
    for word in text.split():
        word = ' '.join(list(word)) + ' </w>'
        while True:
            bigrams = [b for b in zip(word.split()[:-1], word.split()[1:])]
            if not bigrams:
                break
            bigram = max(bigrams, key=lambda b: merged_pairs.get(''.join(b), float('-inf')))
            if ''.join(bigram) not in merged_pairs:
                break
            word = re.sub(r'(?<!\S)' + re.escape(' '.join(bigram)) + r'(?!\S)', merged_pairs[''.join(bigram)], word)
        encoded.extend([token_to_index.get(token, token_to_index['<UNK>']) for token in word.split()])
    return encoded

# class DeepAveragingNetwork(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=2):
#         super(DeepAveragingNetwork, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
#         self.fc1 = nn.Linear(embedding_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, num_classes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p=0.3)
#         self.softmax = nn.LogSoftmax(dim=1)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = torch.mean(x, dim=1)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return self.softmax(x)

# class DeepAveragingNetwork(nn.Module):
#     def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, num_layers=3, dropout_rate=0.3):
#         super(DeepAveragingNetwork, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
#         self.layers = nn.ModuleList([
#             nn.Linear(embedding_dim if i == 0 else hidden_dim, hidden_dim)
#             for i in range(num_layers)
#         ])
        
#         self.batch_norms = nn.ModuleList([
#             nn.BatchNorm1d(hidden_dim)
#             for _ in range(num_layers)
#         ])
        
#         self.fc_out = nn.Linear(hidden_dim, num_classes)
#         self.dropout = nn.Dropout(p=dropout_rate)
#         self.activation = nn.LeakyReLU(negative_slope=0.01)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = torch.mean(x, dim=1)
        
#         residual = x
#         for i, (layer, bn) in enumerate(zip(self.layers, self.batch_norms)):
#             x = layer(x)
#             x = bn(x)
#             x = self.activation(x)
#             x = self.dropout(x)
#             if i > 0:  # Apply residual connection after the first layer
#                 x = x + residual
#             residual = x
        
#         x = self.fc_out(x)
#         return F.log_softmax(x, dim=1)
    
def prepare_input_tensor_BPE(train_exs, merged_pairs, token_to_index):
    tensors = []
    labels = []
    for ex in train_exs:
        word_indices = encode(' '.join(ex.words), merged_pairs, token_to_index)
        if word_indices:
            example_tensor = torch.tensor(word_indices, dtype=torch.long)
            tensors.append(example_tensor)
            labels.append(ex.label)

    if not tensors:
        raise ValueError("No valid sequences found after encoding")

    tensors_padded = pad_sequence(tensors, batch_first=True, padding_value=token_to_index.get('<PAD>', 0))
    return tensors_padded, torch.tensor(labels, dtype=torch.long)

def train_subword_dan(train_exs, dev_exs, epochs, lr, merged_pairs, token_to_index, batch_size):
    vocab_size = len(token_to_index)
    embedding_dim = 300
    hidden_dim = 512
    num_classes = 2

    model = DeepAveragingNetwork(vocab_size, embedding_dim, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    inputs, labels = prepare_input_tensor_BPE(train_exs, merged_pairs, token_to_index)
    train_dataset = TensorDataset(inputs, labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    inputs, labels = prepare_input_tensor_BPE(dev_exs, merged_pairs, token_to_index)
    test_dataset = TensorDataset(inputs, labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    all_train_accuracy = []
    all_test_accuracy = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        all_preds = []
        all_labels = []
        for batch in train_dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())



        train_loss = train_loss / len(train_dataloader)
        train_accuracy = accuracy_score(all_labels, all_preds)

        all_train_accuracy.append(train_accuracy)
        all_train_accuracy.append(train_accuracy)

            # Evaluation on test set
        model.eval()
        test_loss = 0
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for batch_inputs, batch_labels in test_dataloader:
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                
                test_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate test metrics
        test_loss /= len(test_dataloader)
        test_accuracy = accuracy_score(test_labels, test_preds)
        all_test_accuracy.append(test_accuracy)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return all_train_accuracy, all_test_accuracy