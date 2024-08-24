import torch
import numpy as np
from copy import deepcopy

def afl_aggregation(args, global_model, client_models, client_losses, client_weights, alpha=0.1, noisy=False, use_afl=False, use_dpmcf=False):
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
    if noisy:
        updated_client_weights = []
        for weight, norm_loss in zip(client_weights, normalized_losses):
            # Compute the new weight without noise
            new_weight = weight * (1 + alpha * (norm_loss - mean_norm_loss))
            
            # Add Gaussian noise to the weight
            noise = np.random.normal(0, args.sigma)
            noisy_weight = new_weight + noise
            
            updated_client_weights.append(noisy_weight)
    elif use_dpmcf:
        updated_client_weights = []
        # Update the selected client's weight using the exponential mechanism
        G_t = np.random.normal(0, args.sigma)
        U = compute_U(args)
        for weight, norm_loss in zip(client_weights, normalized_losses):
            tilde_loss = U - norm_loss + G_t
            updated_weight = norm_loss * np.exp(-args.lr_mu * tilde_loss / norm_loss)

            updated_client_weights.append(updated_weight)
    elif use_afl:
        weighted_sum = 0
        for weight, norm_loss in zip(client_weights, normalized_losses):
            weighted_sum += weight * np.exp(alpha * norm_loss)
            
        updated_client_weights = []
        for weight, norm_loss in zip(client_weights, normalized_losses):
            # Calculate the updated weight
            updated_weight = (weight * np.exp(alpha * norm_loss)) / weighted_sum
            
            # Append the updated weight to the list
            updated_client_weights.append(updated_weight)

    else:
        updated_client_weights = [weight * (1 + alpha * (norm_loss - mean_norm_loss)) for weight, norm_loss in zip(client_weights, normalized_losses)]

    # print(f"Mean normalized loss: {mean_norm_loss}")
    # print(f"Updated client weights before normalization: {updated_client_weights}")

    # Normalize updated client weights
    weight_sum = sum(updated_client_weights)
    updated_client_weights = [weight / weight_sum for weight in updated_client_weights]

    print(f"Updated client weights after normalization: {updated_client_weights}")

    return aggregated_model, updated_client_weights

def compute_U(args):
    """ Compute the upper bound U used for clipping in the DP mechanism. """
    C = args.sensitivity
    T = args.rounds
    return C + 2 * C * args.sigma * np.log(T)