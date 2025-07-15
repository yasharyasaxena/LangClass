# Language Classification with RNN

A PyTorch-based project for classifying names by their language of origin using Recurrent Neural Networks (RNN) and Gated Recurrent Units (GRU).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a character-level RNN model to predict the language origin of names. Given a person's name as input, the model classifies it into one of 18 supported languages. The model processes names character by character and uses the final hidden state to make predictions.

**Supported Languages:**

- Arabic, Chinese, Czech, Dutch, English, French, German, Greek
- Irish, Italian, Japanese, Korean, Polish, Portuguese, Russian
- Scottish, Spanish, Vietnamese

## ✨ Features

- **Multiple RNN Architectures**: Includes both vanilla RNN and GRU implementations
- **Character-level Processing**: Processes names at the character level for better generalization
- **One-hot Encoding**: Converts characters to one-hot encoded vectors
- **Train/Validation Split**: Automatic stratified splitting of dataset
- **Model Evaluation**: Comprehensive training and evaluation pipeline
- **Prediction Interface**: Easy-to-use prediction function for new names

## 📊 Dataset

The project uses a collection of names from 18 different languages, stored in individual text files under `data/names/`. Each file contains names specific to that language:

```
data/names/
├── Arabic.txt
├── Chinese.txt
├── Czech.txt
├── Dutch.txt
├── English.txt
├── French.txt
├── German.txt
├── Greek.txt
├── Irish.txt
├── Italian.txt
├── Japanese.txt
├── Korean.txt
├── Polish.txt
├── Portuguese.txt
├── Russian.txt
├── Scottish.txt
├── Spanish.txt
└── Vietnamese.txt
```

## 🏗️ Model Architecture

### RNN Model

- **Input Size**: 57 characters (ASCII letters + punctuation: `.,:;-`)
- **Hidden Size**: 256 units
- **Output Size**: 18 classes (one for each language)
- **Architecture**: Single-layer RNN with sigmoid activation
- **Processing**: Character-by-character sequential processing

### Model Components

1. **Input Layer**: One-hot encoded character vectors
2. **Hidden Layer**: Linear transformation with sigmoid activation
3. **Output Layer**: Linear transformation to class probabilities
4. **Hidden State Initialization**: Kaiming uniform initialization

## 🚀 Installation

### Prerequisites

- Python 3.7+
- PyTorch
- scikit-learn
- unidecode

### Setup

1. Clone or download the project:

```bash
git clone https://github.com/yasharyasaxena/LangClass.git
cd LangClass
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch scikit-learn unidecode
```

3. Extract the dataset (since data directory is not included in the repository):

```bash
# On Windows
powershell Expand-Archive -Path data.zip -DestinationPath .

# Or using Python
python -c "import zipfile; zipfile.ZipFile('data.zip').extractall('.')"

# On Linux/Mac
unzip data.zip
```

This will create the `data/` directory with the required name files for all 18 languages.

## 💻 Usage

### Training the Model

Run the main script to train the model:

```bash
python main.py
```

This will:

1. Load and preprocess the dataset
2. Split data into training (90%) and validation (10%) sets
3. Initialize the RNN model
4. Train for 2 epochs
5. Evaluate on the validation set

### Making Predictions

```python
from train import predict
from utils import get_labels

# Load label mapping
lang2labels = get_labels("./data/names")
label2lang = {v.item(): k for k, v in lang2labels.items()}

# Predict language for a name
predicted_language = predict(model, "Jackson", label2lang)
print(f"Predicted language: {predicted_language}")
```

### Custom Training

```python
from models import RNN, GRU
from data import split_dataset
from train import train, evaluate
import torch.nn as nn
import torch.optim as optim

# Load data
train_loader, val_loader = split_dataset("./data/names")

# Initialize model (choose RNN or GRU)
model = RNN(input_size=57, hidden_size=256, output_size=18)
# or
# model = GRU(input_size=57, hidden_size=256, output_size=18)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train and evaluate
train(model, train_loader, criterion, optimizer, num_epochs=5)
evaluate(model, val_loader)
```

## 📁 Project Structure

```
LangClass/
├── main.py              # Main execution script
├── models.py            # RNN and GRU model definitions
├── data.py              # Dataset loading and splitting
├── train.py             # Training, evaluation, and prediction functions
├── utils.py             # Utility functions for data processing
├── README.md            # Project documentation
├── __init__.py          # Package initialization
├── data/
│   ├── names/           # Name datasets by language
│   └── eng-fra.txt      # Additional language data
└── __pycache__/         # Python cache files
```

### Key Files Description

- **`models.py`**: Contains RNN and GRU model classes with forward pass and initialization methods
- **`data.py`**: Handles dataset creation and train/validation splitting with stratification
- **`train.py`**: Implements training loop, evaluation metrics, and prediction interface
- **`utils.py`**: Provides utility functions for label mapping, character encoding, and dataset creation
- **`main.py`**: Orchestrates the entire pipeline from data loading to model evaluation

## 📝 Examples

### Example 1: Basic Training

```python
python main.py
```

Output:

```
Epoch [1/2], Step [1000/15000], Loss: 2.1234
Epoch [2/2], Step [1000/15000], Loss: 1.8765
Test Accuracy: 0.6750
```

### Example 2: Predicting Name Origins

```python
# Example predictions
names = ["Rodriguez", "O'Connor", "Zhang", "Mueller", "Dubois"]
for name in names:
    prediction = predict(model, name, label2lang)
    print(f"{name} -> {prediction}")
```

Expected Output:

```
Rodriguez -> Spanish
O'Connor -> Irish
Zhang -> Chinese
Mueller -> German
Dubois -> French
```

## 📈 Results

The model typically achieves:

- **Training Accuracy**: ~70-80% after 2 epochs
- **Validation Accuracy**: ~65-75%
- **Best Performance**: On common European languages (English, French, German, Spanish)
- **Challenging Cases**: Asian languages and names with mixed origins

### Performance Notes

- Model performs well on names with distinctive character patterns
- Longer names generally yield better predictions
- Some names may be ambiguous across similar languages

## 🛠️ Customization

### Model Architecture Changes

#### Switching Between RNN and GRU Models

The project includes both vanilla RNN and GRU implementations. You can easily switch between them:

**Using Vanilla RNN (Default):**

```python
from models import RNN
model = RNN(input_size=57, hidden_size=256, output_size=18)
```

**Using GRU Model:**

```python
from models import GRUModel
model = GRUModel(
    num_layers=1,       # Number of GRU layers
    input_size=57,      # Character vocabulary size
    hidden_size=256,    # Hidden state dimension
    output_size=18      # Number of language classes
)
```

**Performance Comparison:**

- **RNN**: Simpler, faster training, good baseline performance
- **GRU**: Better at capturing long-term dependencies, may achieve higher accuracy
- **Recommendation**: Start with RNN, then try GRU if you need better performance

#### Multi-layer GRU Configuration

```python
# Deeper GRU model for better performance
model = GRUModel(
    num_layers=2,       # Add more layers for complexity
    input_size=57,
    hidden_size=256,
    output_size=18
)
```

### Hyperparameter Tuning

```python
# Adjust model parameters
model = RNN(
    input_size=57,      # Character vocabulary size
    hidden_size=512,    # Increase for more capacity
    output_size=18      # Number of languages
)

# Modify training parameters
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Lower learning rate
train(model, train_loader, criterion, optimizer, num_epochs=10)  # More epochs
```

### Training Configuration Changes

#### Adjusting Training Parameters

```python
# In main.py, modify these parameters:

# Training epochs
train(model, train_loader, criterion, optimizer, num_epochs=5)  # More epochs

# Learning rate options
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)     # Default
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)    # Lower LR
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)       # SGD optimizer

# Different loss functions
criterion = nn.CrossEntropyLoss()              # Default
criterion = nn.NLLLoss()                       # Negative log likelihood
criterion = nn.CrossEntropyLoss(weight=weights) # Weighted for imbalanced data
```

#### Batch Processing (Optional Enhancement)

```python
# For larger datasets, consider adding batch processing
from torch.utils.data import DataLoader, TensorDataset

def create_dataloader(dataset, batch_size=32):
    # Convert to tensors and create DataLoader
    inputs, labels = zip(*dataset)
    dataset = TensorDataset(torch.stack(inputs), torch.stack(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Usage
train_loader = create_dataloader(train_dataset, batch_size=64)
```

### Adding New Languages

1. Add a new `.txt` file with names in `data/names/`
2. Update the model's `output_size` parameter
3. Retrain the model

### Model Comparison Script

```python
# Compare RNN vs GRU performance
def compare_models():
    # Load data
    train_loader, val_loader = split_dataset("./data/names")

    # Test RNN
    rnn_model = RNN(input_size=57, hidden_size=256, output_size=18)
    train(rnn_model, train_loader, criterion, optimizer, num_epochs=3)
    print("RNN Results:")
    evaluate(rnn_model, val_loader)

    # Test GRU
    gru_model = GRUModel(num_layers=1, input_size=57, hidden_size=256, output_size=18)
    train(gru_model, train_loader, criterion, optimizer, num_epochs=3)
    print("GRU Results:")
    evaluate(gru_model, val_loader)

# Run comparison
compare_models()
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset sources for name collections by language
- PyTorch team for the deep learning framework
- Contributors to the unidecode library for character normalization

---

**Note**: This project is designed for educational purposes and demonstrates basic RNN implementation for text classification. For production use, consider more advanced architectures like LSTMs, Transformers, or pre-trained language models.
