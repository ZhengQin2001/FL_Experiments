from parameters import get_args
import torch
import os
from torch.utils.data.dataset import Subset
from torchvision.datasets.mnist import MNIST
from utils.data_preparation import prepare_data
from training.train import federated_training
from training.dpmcf_training import dpmcf_training


def main(args):
    prepare_data(args)
    if args.federated_type in ['fedavg', 'afl', 'dpsgd', 'dpmcf']:
        federated_training(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
