from parameters import get_args
import torch
import os
from torch.utils.data.dataset import Subset
from torchvision.datasets.mnist import MNIST
from utils.data_preparation import prepare_data
from training import fedavg_training


def main(args):
    prepare_data(args)
    fedavg_training(args)

if __name__ == "__main__":
    args = get_args()
    main(args)
