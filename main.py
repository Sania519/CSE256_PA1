# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import DeepAveragingNetwork
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score
from BPEModel import *

# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy

def prepare_input_tensor(train_exs, word_embeddings):
       tensors = []
       labels = []
       for ex in train_exs:
           # Convert words to indices
           word_indices = [word_embeddings.word_indexer.index_of(word) for word in ex.words]
           # Use a default index (e.g., UNK) for words not in the vocabulary
           word_indices = [idx if idx != -1 else word_embeddings.word_indexer.index_of("UNK") for idx in word_indices]
           
           example_tensor = torch.tensor(word_indices, dtype=torch.long)
           tensors.append(example_tensor)
           labels.append(ex.label)

       tensors_list = pad_sequence(tensors, batch_first=True, padding_value=word_embeddings.word_indexer.index_of("PAD"))
       input_tensor = torch.stack(tuple(tensors_list))
       return input_tensor, torch.tensor(labels, dtype=torch.long)

def train_DAN(model, train_loader, test_loader, num_epochs, lr):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    all_train_accuracy = []
    all_test_accuracy = []

    for epoch in range(num_epochs):
    
    # Prepare inputs and labels
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch_inputs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())



        train_loss = total_loss / len(train_loader)
        train_accuracy = accuracy_score(all_labels, all_preds)

        all_train_accuracy.append(train_accuracy)

            # Evaluation on test set
        model.eval()
        test_loss = 0
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for batch_inputs, batch_labels in test_loader:
                outputs = model(batch_inputs)
                loss = loss_fn(outputs, batch_labels)
                
                test_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                test_preds.extend(preds.cpu().numpy())
                test_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate test metrics
        test_loss /= len(test_loader)
        test_accuracy = accuracy_score(test_labels, test_preds)
        all_test_accuracy.append(test_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return all_train_accuracy, all_test_accuracy
    
def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., BOW)')
    parser.add_argument('--randomise', type=str, required=False, help='Model type to train (e.g., BOW)')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load dataset
    start_time = time.time()

    train_data = SentimentDatasetBOW("data/train.txt")
    dev_data = SentimentDatasetBOW("data/dev.txt")
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loaded in : {elapsed_time} seconds")


    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DAN":
        train_exs = read_sentiment_examples("data/train.txt")
        dev_exs = read_sentiment_examples("data/dev.txt")

        word_embeddings = read_word_embeddings("data/glove.6B.50d-relativized.txt")
        
        num_epochs = 20
        batch_size = 16
        lr = 0.001

        inputs, labels = prepare_input_tensor(train_exs, word_embeddings)
        train_dataset = SentimentDatasetDAN(inputs, labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        inputs, labels = prepare_input_tensor(dev_exs, word_embeddings)
        test_dataset = TensorDataset(inputs, labels)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        print(word_embeddings.get_embedding_length())

        if args.randomise == False:
            model = DeepAveragingNetwork(word_embeddings, word_embeddings.get_embedding_length(), hidden_dim=100, num_classes=2, randomise=False)
        else:
            model = DeepAveragingNetwork(word_embeddings, word_embeddings.get_embedding_length(), hidden_dim=100, num_classes=2, randomise=True)

        DAN_train_accuracy, DAN_test_accuracy = train_DAN(model, train_dataloader, test_dataloader, num_epochs, lr)

    elif args.model == "SUBWORDDAN":
        data = read_data("data/train.txt")
        test_data = read_data("data/dev.txt")
        data = data+test_data
        merged_pairs, unique_tokens = byte_pair_encoding(data, 20000)
        token_to_index = {token: i for i, token in enumerate(unique_tokens)}
        token_to_index['<UNK>'] = len(token_to_index)
        token_to_index['<PAD>'] = len(token_to_index)
        
     
        train_exs = read_sentiment_examples("data/train.txt")
        dev_exs = read_sentiment_examples("data/dev.txt")

        num_epochs = 100
        batch_size = 16
        lr=0.001

        inputs, labels = prepare_input_tensor_BPE(train_exs, merged_pairs, token_to_index)
        train_dataset = TensorDataset(inputs, labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        inputs, labels = prepare_input_tensor_BPE(dev_exs, merged_pairs, token_to_index)
        test_dataset = TensorDataset(inputs, labels)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        vocab_size = len(token_to_index)

        model = DeepAveragingNetwork(vocab_size, 50, hidden_dim=100, num_classes=2, randomise=True)

        DAN_train_accuracy, DAN_test_accuracy = train_DAN(model, train_dataloader, test_dataloader, num_epochs, lr)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(DAN_train_accuracy, label='Train Accuracy')
        plt.plot(DAN_test_accuracy, label='Test Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('SUBWORDDAN')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy_SUBWORDDAN.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")


if __name__ == "__main__":
    main()
