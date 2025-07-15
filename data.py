import os
from utils import create_dataset
from sklearn.model_selection import train_test_split

def split_dataset(dir):
    """Get test and train datasets from the specified directory.
    Args:
        dir (str): Directory containing language files.
    Returns:
        tuple: A tuple containing two lists, one for training data and one for testing data.
    """
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory {dir} does not exist.")
    
    target_tensor, target_labels = create_dataset(dir)

    train_idx, test_idx = train_test_split(range(len(target_labels)), test_size=0.1, shuffle=True, stratify=target_labels, random_state=42)

    train_dataset = [(target_tensor[i], target_labels[i]) for i in train_idx]
    test_dataset = [(target_tensor[i], target_labels[i]) for i in test_idx]

    return (train_dataset, test_dataset)

if __name__ == "__main__":
    train_data, test_data = split_dataset('./data/names')
    print(f"Train dataset size: {len(train_data)}")
    print(f"Test dataset size: {len(test_data)}")
