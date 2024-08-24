import torch
from collections import Counter
import os


def test_data():
    # Load the saved .pt file
    data_file = '../data/processed/mnist/PSG/train/client_1_train.pt'  # Replace with the actual path to your .pt file
    dataset = torch.load(data_file)

    # Assuming the dataset is a PyTorch Dataset object
    labels = [label for _, label in dataset]

    # Count the occurrences of each label
    label_counts = Counter(labels)

    # Display the number of data points for each group
    for label, count in label_counts.items():
        print(f"Label {label}: {count} data points")

if __name__ == '__main__':
    test_data()