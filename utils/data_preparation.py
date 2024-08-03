import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets.mnist import MNIST

NUM_CLASSES_DICT = {
    'mnist': 10,
    'fashion_mnist': 10,
    'emnist': 47  # using 'balanced' split
}

def load_data(args):
    """Load dataset data and apply transformations."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if args.data == 'mnist':
        train_data = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        test_data = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    elif args.data == 'fashion_mnist':
        train_data = datasets.FashionMNIST(args.data_dir, train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(args.data_dir, train=False, download=True, transform=transform)
    elif args.data == 'emnist':
        train_data = datasets.EMNIST(args.data_dir, split='balanced', train=True, download=True, transform=transform)
        test_data = datasets.EMNIST(args.data_dir, split='balanced', train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {args.data}")
    
    return train_data, test_data

def distribute_data_psg(data, num_clients=10, seed=42):
    """Distribute data partially to simulate Partial access to Sensitive Groups (PSG)."""
    np.random.seed(seed)
    targets = data.targets.numpy()
    client_indices = {i: [] for i in range(num_clients)}

    # Partially distribute each class to a subset of clients
    for label in range(10):
        indices = np.where(targets == label)[0]
        np.random.shuffle(indices)
        # Split indices among a subset of clients
        subset_clients = np.random.choice(num_clients, size=max(2, num_clients // 2), replace=False)
        split_indices = np.array_split(indices, len(subset_clients))
        for client, idx in zip(subset_clients, split_indices):
            client_indices[client].extend(idx.tolist())
    return client_indices

def distribute_data_ssg(data, num_clients=10):
    """Distribute data to simulate Single Sensitive Group (SSG) access."""
    targets = data.targets.numpy()
    client_indices = {i: [] for i in range(num_clients)}

    # Assign each digit to a different client, vary sizes randomly
    for label in range(10):
        indices = np.where(targets == label)[0]
        client_id = label % num_clients
        client_indices[client_id].extend(indices.tolist())
    return client_indices

def _partition_data(data, train_ratio=0.8, seed=42):
    """Partition the data into training and validation sets."""
    np.random.seed(seed)
    num_samples = len(data)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    split = int(train_ratio * num_samples)
    train_indices, val_indices = indices[:split], indices[split:]
    
    train_subset = Subset(data, train_indices)
    val_subset = Subset(data, val_indices)
    
    return train_subset, val_subset

def save_data(args, train_data, test_data, indices_dict):
    """Save data into subfolders under processed based on scenario, including partitioning the training data."""
    base_dir = os.path.join(args.data_dir, 'processed', args.data, args.data_setting)
    
    # Create necessary directories if they do not exist
    os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'test'), exist_ok=True)
    
    for client_id, indices in indices_dict.items():
        client_data = Subset(train_data, indices)
        client_train_subset, client_val_subset = _partition_data(client_data)

        train_path = os.path.join(base_dir, 'train', f'client_{client_id}_train.pt')
        val_path = os.path.join(base_dir, 'validation', f'client_{client_id}_validation.pt')

        torch.save(client_train_subset, train_path)
        torch.save(client_val_subset, val_path)

    # Save the entire test dataset in one file
    test_path = os.path.join(base_dir, 'test', 'test_data.pt')
    torch.save(test_data, test_path)

def prepare_data(args):
    train_data, test_data = load_data(args)
    num_clients = NUM_CLASSES_DICT[args.data]

    if args.data_setting == 'PSG':
        psg_indices = distribute_data_psg(train_data, num_clients)
        save_data(args, train_data, test_data, psg_indices)
    elif args.data_setting == 'SSG':
        ssg_indices = distribute_data_ssg(train_data, num_clients)
        save_data(args, train_data, test_data, ssg_indices)
    else:
        raise ValueError("Invalid data setting provided.")

    print("Data preprocessing and saving completed.")
