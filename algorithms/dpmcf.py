import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from models.cnn import CNN
import torch.optim as optim
from copy import deepcopy

class DPMCF:
    def __init__(self, args, global_model, criterion):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion
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
        # return C + 2 * C * np.log(T)/self.sigma

    def clip_gradient(self, gradient, C):
        """ Clip the gradients to have a bounded L2 norm. """
        norm = torch.norm(gradient)
        return gradient * min(1, C / norm)

    # 1st version
    def client_update(self, client_model, data_loader):
        """ Perform the client update with differential privacy. """
        client_model.train()
        optimizer = optim.SGD(client_model.parameters(), lr=self.args.lr)

        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = self.criterion(output, target)
            loss.backward()

            # Clip gradients, average them, and add noise
            for param in client_model.parameters():
                if param.grad is not None:
                    # Clip the gradient
                    clipped_grad = self.clip_gradient(param.grad, self.args.sensitivity)

                    # Average the gradient over the batch size
                    clipped_grad = clipped_grad / self.args.minibatch_size

                    # Add noise to the averaged gradient
                    param.grad = clipped_grad + torch.normal(0, self.sigma, clipped_grad.size()).to(self.device)

            optimizer.step()
        return client_model

    def gradient_update(self, client_models):
        """ Aggregate gradients from clients and then update the global model. """
        aggregated_model = deepcopy(self.global_model)
        
        with torch.no_grad():
            for param in aggregated_model.parameters():
                param.grad = torch.zeros_like(param.data)

            # Aggregate gradients from each client
            for client_index, client_model in enumerate(client_models.values()):
                for agg_param, client_param in zip(aggregated_model.parameters(), client_model.parameters()):
                    if client_param.grad is not None:
                        agg_param.grad.add_(client_param.grad * self.client_weights[client_index])

        # Apply the aggregated gradients to update the global model
        optimizer = optim.SGD(self.global_model.parameters(), lr=self.args.lr)
        optimizer.zero_grad()  # Clear any existing gradients
        for param, agg_param in zip(self.global_model.parameters(), aggregated_model.parameters()):
            if agg_param.grad is not None:
                param.grad = agg_param.grad  # Set the gradient to the aggregated gradient
        optimizer.step()

    def dpmcf(self, client_index, client_models, client_losses):
        """ Perform the Differential Privacy Minimax Client Fairness Optimization. """

        # Update the selected client's weight using the exponential mechanism
        G_t = np.random.normal(0, self.sigma)
        tilde_loss = self.U - client_losses[client_index] + G_t
        updated_weight = self.client_weights[client_index] * np.exp(-self.args.lr_mu * tilde_loss / self.client_weights[client_index])
        
        # Only update the weight for the selected client
        self.client_weights[client_index] = updated_weight

        # Normalize the weights
        self.client_weights /= np.sum(self.client_weights)

        # Debugging: Ensure weights are normalized and have the correct size
        if len(self.client_weights) != self.args.num_clients:
            raise ValueError(f"Weight size mismatch: expected {self.args.num_clients} but got {len(self.client_weights)}")
        if not np.isclose(np.sum(self.client_weights), 1.0):
            raise ValueError(f"Weights not normalized properly: sum is {np.sum(self.client_weights)}")
        
