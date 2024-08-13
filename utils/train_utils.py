import torch

def train_client(model, train_loader, optimizer, criterion, device, return_loss=False):
    model.train()
    model.to(device)
    total_loss = 0
    total_samples = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
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