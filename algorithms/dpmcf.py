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
        self.global_model = global_model
        self.client_weights = [1.0 / args.num_clients] * args.num_clients  # Initialize client weights
        self.sigma = self.compute_sigma()
        self.U = self.compute_U()

    def compute_sigma(self):
        C = self.args.sensitivity
        delta = self.args.delta
        epsilon = self.args.epsilon
        return C * np.sqrt(2 * np.log(1 / delta)) / (2 * epsilon)

    def compute_U(self):
        C = self.args.sensitivity
        T = self.args.rounds
        return C / 2 + 2 * self.sigma * np.log(T)

    def clip_gradient(self, gradient, C):
        norm = torch.norm(gradient)
        return gradient * min(1, C / norm)
    
    def split_data(self, dataset):
        data_size = len(dataset)
        indices = list(range(data_size))
        np.random.shuffle(indices)
        split = int(np.floor(0.5 * data_size))
        return Subset(dataset, indices[:split]), Subset(dataset, indices[split:])

    def aggregation(self, client_models, client_losses):
        total_loss = sum(client_losses)
        normalized_losses = [loss / total_loss for loss in client_losses]
        
        # Clip gradients and add noise for differential privacy
        clipped_grads = []
        for client_model in client_models:
            grads = []
            for param in client_model.parameters():
                grad = self.clip_gradient(param.grad.data, self.args.sensitivity)
                grad += torch.normal(0, self.sigma, grad.size()).to(self.device)  # Add Gaussian noise
                grads.append(grad)
            clipped_grads.append(grads)
        
        # Aggregate clipped gradients
        with torch.no_grad():
            for i, param in enumerate(self.global_model.parameters()):
                param.data.zero_()
                for weight, grads in zip(self.client_weights, clipped_grads):
                    param.data.add_(grads[i] * weight)

        # Update client weights based on the exponential mechanism
        updated_client_weights = []
        # print(normalized_losses)
        mean_norm_loss = np.mean(normalized_losses)
        for weight, norm_loss in zip(self.client_weights, normalized_losses):
            tilde_loss = self.U - norm_loss + np.random.normal(0, self.sigma)
            tilde_loss = np.clip(tilde_loss, -50, 50)  # Clip the values to prevent overflow/underflow
            updated_weight = weight * np.exp(-self.args.lr_mu * tilde_loss / (weight + 1e-10))  # Add small constant to avoid division by zero
            updated_client_weights.append(updated_weight)

        # Normalize updated client weights
        weight_sum = sum(updated_client_weights)
        self.client_weights = [weight / weight_sum for weight in updated_client_weights]

        return self.global_model
