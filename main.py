from utils import get_labels, name2tensor
from data import split_dataset
from models import RNN, GRUModel
from train import evaluate_GRU, train, train_GRU, evaluate, predict
import torch
from torch import nn
from string import ascii_letters

def main():
    DIR_NAME = "./data/names"

    # Get Names and Labels
    lang2labels = get_labels(DIR_NAME)
    label2lang = {v.item(): k for k, v in lang2labels.items()}
    num_languages = len(lang2labels)
    char2idx = {c: i for i, c in enumerate(ascii_letters + ".,:;-")}
    num_chars = len(char2idx)

    # Create dataset
    train_loader, val_loader = split_dataset(DIR_NAME)

    # Initialize model
    # Uncomment the model you want to use

    # RNN model
    # model = RNN(input_size=num_chars, hidden_size=256, output_size=num_languages)
    
    # GRU model
    # num_layers = 2  # Example number of layers
    model = GRUModel(num_layers=2, input_size=num_chars, hidden_size=256, output_size=num_languages)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train RNN model
    # train(model, train_loader, criterion, optimizer, num_epochs=2)

    # Train GRU model
    train_GRU(model, train_loader, criterion, optimizer, num_epochs=2)

    # Evaluate RNN model
    # evaluate(model, val_loader)

    # Evaluate GRU model
    evaluate_GRU(model, val_loader)

if __name__ == "__main__":
    main()