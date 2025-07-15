from utils import get_labels, name2tensor
from data import split_dataset
from models import RNN
from train import train, evaluate, predict
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
    model = RNN(input_size=num_chars, hidden_size=256, output_size=num_languages)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train(model, train_loader, criterion, optimizer, num_epochs=2)

    # Evaluate the model
    evaluate(model, val_loader)

if __name__ == "__main__":
    main()