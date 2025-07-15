import random
import torch
from torch import nn
from utils import name2tensor

random.seed(42)

def train(model, train_dataset, criterion, optimizer, num_epochs, print_every=1000):
    random.shuffle(train_dataset)
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, targets) in enumerate(train_dataset):
            optimizer.zero_grad()
            hidden_state = model.init_hidden()
            for x in inputs:
                hidden_state, output = model(x, hidden_state)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            if (i + 1) % print_every == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataset)}], Loss: {loss.item():.4f}')

def evaluate(model, test_dataset):
    num_correct = 0
    num_samples = len(test_dataset)
    model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataset:
            hidden_state = model.init_hidden()
            for x in inputs:
                hidden_state, output = model(x, hidden_state)
            _, predicted = torch.max(output, 1)
            num_correct += bool(predicted == targets)
    print(f'Test Accuracy: {num_correct / num_samples:.4f}')

def predict(model, name, label_map):
    model.eval()
    input_data = name2tensor(name)
    with torch.no_grad():
        hidden_state = model.init_hidden()
        for x in input_data:
            hidden_state, output = model(x, hidden_state)
        _, predicted = torch.max(output, 1)
    model.train()
    output = label_map[predicted.item()]
    return output
