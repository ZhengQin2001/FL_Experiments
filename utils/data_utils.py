import torch
import os
from torch.utils.data import DataLoader
import numpy as np

def load_client_data(args, client_id):
    train_path = os.path.join(args.data_dir, 'processed', args.data, args.data_setting, 'train', f'client_{client_id}_train.pt')
    val_path = os.path.join(args.data_dir, 'processed', args.data, args.data_setting, 'validation', f'client_{client_id}_validation.pt')

    train_data = torch.load(train_path)
    val_data = torch.load(val_path)

    # print(f"Client {client_id} - Training data size: {np.shape(train_data)}")
    # print(f"Client {client_id} - Validation data size: {np.shape(val_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader

def load_test_data(args):
    test_path = os.path.join(args.data_dir, 'processed', args.data, args.data_setting, 'test', 'test_data.pt')
    test_data = torch.load(test_path)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    return test_loader