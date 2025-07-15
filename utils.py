import os
import torch
from string import ascii_letters
from unidecode import unidecode

DATA_DIR = './data/names'

def get_labels(dir):
    """    Get a dictionary mapping language names to tensor labels.
    Args:
        dir (str): Directory containing language files.
    Returns:
        dict: A dictionary where keys are language names and values are tensors of labels.
    """
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory {dir} does not exist.")
    if not os.path.isdir(dir):
        raise NotADirectoryError(f"{dir} is not a directory.")
    if not os.listdir(dir):
        raise ValueError(f"Directory {dir} is empty.")
    lang2label = {
        file_name.split('.')[0]: torch.tensor([i], dtype=torch.long) for i, file_name in enumerate(os.listdir(dir))
    }
    return lang2label

def name2tensor(name):
    """Convert a name to a tensor of character indices.
    
    Args:
        name (str): The name to convert.
    
    Returns:
        torch.Tensor: A tensor representing the name, where each character is one-hot encoded.
    """
    if not isinstance(name, str):
        raise TypeError("Input must be a string.")
    
    char2idx = { c: i for i, c in enumerate(ascii_letters+".,:;-") }
    num_chars = len(char2idx)
    tensor = torch.zeros(len(name), 1, num_chars)

    for i, c in enumerate(name):
        if c in char2idx:
            tensor[i][0][char2idx[c]] = 1

    return tensor

def create_dataset(dir):
    """Create a dataset from the specified directory.
    
    Args:
        dir (str): Directory containing language files.
    
    Returns:
        tuple: A tuple containing two lists, one for data and one for labels.
    """
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory {dir} does not exist.")
    
    lang2labels = get_labels(dir)
    target_tensor = []
    target_labels = []

    for file in os.listdir(dir):
        with open(os.path.join(dir, file), encoding='utf-8') as f:
            lang = file.split('.')[0]
            names = [unidecode(line.rstrip()) for line in f]
            for name in names:
                try:
                    tensor = name2tensor(name)
                    target_tensor.append(tensor)
                    target_labels.append(lang2labels[lang])
                except Exception as e:
                    print(f"Error processing name '{name}' in language '{lang}': {e}")
                    pass
    return target_tensor, target_labels

if __name__ == "__main__":
    labels = get_labels(DATA_DIR)
    print(labels)