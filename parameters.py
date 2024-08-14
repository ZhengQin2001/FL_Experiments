"""
Define parameters needed for training a model using distributed or federated schema
"""
import argparse
from os.path import join

NUM_CLASSES_DICT = {
    'mnist': 10,
    'fashion_mnist': 10,
    'emnist': 47  # using 'balanced' split
}

def get_args():
    # feed them to the parser.
    parser = argparse.ArgumentParser(
        description='Parameters for running training on FedTorch package.')

    # add arguments.
    # dataset.
    parser.add_argument('-d', '--data', default='mnist',
                        choices=['mnist','fashion_mnist', 'emnist'],
                            help='Dataset name.')
    
    parser.add_argument('-p', '--data_dir', default='./data/',
                        help='path to dataset')
    
    parser.add_argument('-ds', '--data_setting', default='PSG',
                        choices=['PSG', 'SSG'],
                            help='federated learning settings')
    
    parser.add_argument('--federated_type', default='fedavg', type=str,
                        choices=['fedavg', 'afl', 'dpmcf'],
                        help="Types of federated learning algorithm and/or training procedure.")

    # training.
    parser.add_argument('-lr', default=0.005, type=float,
                        help='Learning rate')
    
    parser.add_argument('-r', '--rounds', default=1000, type=int,
                        help='Number of rounds')

    parser.add_argument('-b', '--batch_size', default=32, type=int,
                        help='Batch size for client training')
    
    parser.add_argument('--model_dir', default='./model/',
                        help='path to model')
    
    parser.add_argument('--num_runs', default=1, type=int,
                        help='Number of runs')

    # Differential Privacy Minimax Client Fairness (DP-MCF) specific arguments
    parser.add_argument('--epsilon', default=10, type=float,
                        help='Privacy budget (epsilon) for DP-MCF')
    
    parser.add_argument('--delta', default=1e-6, type=float,
                        help='Privacy parameter (delta) for DP-MCF')
    
    parser.add_argument('--sensitivity', default=1, type=float,
                        help='Sensitivity bound (C) for DP-MCF')
    
    parser.add_argument('--lr_mu', default=0.0005, type=float,
                        help='Learning rate for updating client weights (mu) in DP-MCF')

    parser.add_argument('--minibatch_size', default=128, type=int,
                        help='minibatch sample size in DPMCF')
    
    args = parser.parse_args()

    # Set num_clients based on the dataset
    args.num_clients = NUM_CLASSES_DICT[args.data]

    return args

def print_args(args):
    print('parameters: ')
    for arg in vars(args):
        print(arg, getattr(args, arg))


if __name__ == '__main__':
    args = get_args()