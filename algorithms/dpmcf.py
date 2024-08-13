import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from models.cnn import CNN
from utils.data_utils import load_client_data, load_test_data
from utils.train_utils import train_client, evaluate_model

class DPMCF:
    def __init__(self, args, global_model):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_model = global_model.to(self.device)
        self.client_weights = np.ones(args.num_clients) / args.num_clients  # Initialize client weights uniformly
        self.sigma = self.compute_sigma()
        self.U = self.compute_U()
        self.client_data = {}  # Dictionary to store positive and negative DataLoaders for each client

    def compute_sigma(self):
        """ Compute the noise standard deviation for differential privacy. """
        C = self.args.sensitivity
        delta = self.args.delta
        epsilon = self.args.epsilon
        return C * np.sqrt(2 * np.log(1 / delta)) / epsilon

    def compute_U(self):
        """ Compute the upper bound U used for clipping in the DP mechanism. """
        C = self.args.sensitivity
        T = self.args.rounds
        return C / 2 + 2 * self.sigma * np.log(T)

    def clip_gradient(self, gradient, C):
        """ Clip the gradients to have a bounded L2 norm. """
        norm = torch.norm(gradient)
        return gradient * min(1, C / norm)

    def gradient_update(self, client_model, mini_batch_size):
        """ Perform gradient update for client model with clipping and noise addition. """
        for param_global, param_client in zip(self.global_model.parameters(), client_model.parameters()):
            if param_client.grad is None:
                continue
            # Clip the gradients
            clipped_grad = self.clip_gradient(param_client.grad.data, self.args.sensitivity)
            # Add Gaussian noise
            noised_grad = clipped_grad + torch.normal(0, self.sigma, clipped_grad.size()).to(self.device)
            # Perform the update step (adjusting for mini-batch size)
            param_global.data -= self.args.lr * noised_grad / mini_batch_size

    def dpmcf(self, client_model, client_loss, mini_batch_size):
        """ Perform the Differential Privacy Minimax Client Fairness Optimization. """
        normalized_loss = client_loss / sum(self.client_weights)

        # Update global model with the client's model update
        self.gradient_update(client_model, mini_batch_size)

        # Update client weights using exponential mechanism
        G_t = np.random.normal(0, self.sigma)
        tilde_loss = self.U - normalized_loss + G_t
        updated_weight = self.client_weights * np.exp(-self.args.lr_mu * tilde_loss / self.client_weights)
        
        # Normalize the weights
        self.client_weights = updated_weight / np.sum(updated_weight)
