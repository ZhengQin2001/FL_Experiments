import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

def clip_gradients(model, C):
    total_norm = 0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = C / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)

def train_client(args, model, train_loader, optimizer, criterion, device, return_loss=False, private=False, with_sample=False):
    model.train()
    model.to(device)
    total_loss = 0
    total_samples = 0

    if with_sample:
        # Sampling probability (assume this is passed as an argument or calculated earlier)
        sampling_prob = args.sampling_prob
        
        # Apply the sampling probability to the dataset
        if sampling_prob < 1.0:
            dataset = train_loader.dataset
            num_samples = int(len(dataset) * sampling_prob)
            sampled_indices = np.random.choice(len(dataset), num_samples, replace=False)
            sampled_dataset = Subset(dataset, sampled_indices)
            new_train_loader = DataLoader(sampled_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        new_train_loader = train_loader

    for data, target in new_train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if private:
            if args.federated_type in ['dpsgd', 'dpmcf','fedavg', 'afl']:
                clip_gradients(model, args.clip_norm)
            for param in model.parameters():
                if param.grad is not None:
                    # delta = args.delta
                    # epsilon = args.epsilon
                    # sigma = np.sqrt(2 * np.log(1 / delta)) / epsilon
                    # Add noise to the averaged gradient
                    param.grad = param.grad + torch.normal(0, args.sigma, param.grad.size()).to(device)

        if return_loss:
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    if return_loss:
        average_loss = total_loss / total_samples
        return average_loss


def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    return accuracy

def evaluate_model_loss(model, val_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total_samples += data.size(0)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    average_loss = total_loss / total_samples
    return average_loss, accuracy