import torch
import numpy as np
from copy import deepcopy

def afl_aggregation(global_model, client_models, client_losses, client_weights, alpha=0.3):
    """
    Agnostic Federated Learning aggregation.

    Parameters:
    global_model: The global model to be updated.
    client_models: List of client models.
    client_losses: List of losses for each client.
    client_weights: List of weights for each client.
    alpha: Learning rate for updating client weights.

    Returns:
    Updated global model and client weights.
    """
    total_loss = sum(client_losses)
    normalized_losses = [loss / total_loss for loss in client_losses]

    # print(f"Client losses: {client_losses}")
    # print(f"Total loss: {total_loss}")
    # print(f"Normalized losses: {normalized_losses}")

    # Zero out the parameters in the aggregated model
    aggregated_model = deepcopy(global_model)
    with torch.no_grad():
        for param in aggregated_model.parameters():
            param.data.zero_()

        # Aggregate the differences from the global model
        for client_model, weight, norm_loss in zip(client_models, client_weights, normalized_losses):
            for agg_param, client_param, global_param in zip(aggregated_model.parameters(), client_model.parameters(), global_model.parameters()):
                agg_param.data.add_((client_param.data - global_param.data) * weight * norm_loss)

        # Add the original global model parameters back
        for agg_param, global_param in zip(aggregated_model.parameters(), global_model.parameters()):
            agg_param.data.add_(global_param.data)

    # Update client weights based on the normalized losses
    mean_norm_loss = np.mean(normalized_losses)
    updated_client_weights = [weight * (1 + alpha * (norm_loss - mean_norm_loss)) for weight, norm_loss in zip(client_weights, normalized_losses)]

    # print(f"Mean normalized loss: {mean_norm_loss}")
    # print(f"Updated client weights before normalization: {updated_client_weights}")

    # Normalize updated client weights
    weight_sum = sum(updated_client_weights)
    updated_client_weights = [weight / weight_sum for weight in updated_client_weights]

    print(f"Updated client weights after normalization: {updated_client_weights:2f}")

    return aggregated_model, updated_client_weights
