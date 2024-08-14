import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from models.cnn import CNN
from utils.data_utils import load_client_data, load_test_data
from utils.train_utils import train_client, evaluate_model, evaluate_model_loss
from algorithms.dpmcf import DPMCF
import os
from copy import deepcopy

def dpmcf_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Track accuracies and fairness metrics across runs
    test_accuracies = []
    client_fairness_values = []

    for run in range(args.num_runs):
        print(f"Run {run+1}/{args.num_runs}")
        torch.manual_seed(run)
        np.random.seed(run)

        # Initialize the global model
        global_model = CNN(args.data).to(device)
        criterion = torch.nn.CrossEntropyLoss()

        # Reset the global model for each run
        global_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

        # Initialize DPMCF algorithm with the global model
        dp_mcf = DPMCF(args, global_model, criterion)
        
        # Initialize the client model dictionary outside the loop
        client_models = {i: deepcopy(global_model) for i in range(args.num_clients)}
        client_losses = {}

        # Preload data for all clients
        for i_t in range(args.num_clients):
            # Load the client's data
            train_loader, val_loader = load_client_data(args, i_t)

            # Store loaders for each client
            dp_mcf.client_data[i_t] = (val_loader, train_loader) # (positive set, negative set)
        
        for t in tqdm(range(args.rounds), desc="Rounds"):
            # Step 3: Sample a client based on current weights
            client_indices = np.arange(args.num_clients)

            i_t = np.random.choice(client_indices, p=dp_mcf.client_weights)
            
            # Step 4: Load the split loaders for the sampled client
            positive_loader, negative_loader = dp_mcf.client_data[i_t]

            # Use the constant mini-batch size from args for gradient updates
            mini_batch_indices = np.random.choice(len(negative_loader.dataset), args.minibatch_size, replace=True)
            mini_batch_subset = Subset(negative_loader.dataset, mini_batch_indices)
            mini_batch_loader = DataLoader(mini_batch_subset, batch_size=args.batch_size, shuffle=False)

            # Train the client model using the CNN training batch
            client_model = client_models[i_t]

            # Perform the client update with differential privacy, returning the updated model and loss
            updated_model = dp_mcf.client_update(client_model, mini_batch_loader)
            client_models[i_t] = updated_model

            # Evaluate the client's model on its positive subset
            positive_loss, accuracy = evaluate_model_loss(updated_model, positive_loader, device, criterion)
            client_losses[i_t] = positive_loss
            

            # Step 5: Update global model and client weights using DPMCF with the aggregated mini-batch
            dp_mcf.dpmcf(i_t, client_models, client_losses)

            if t%100 == 0:
                print(f"Client {i_t} Accuracy on positive loader: {accuracy:.2f}, {positive_loss:.4f}")
                print(f'Client weights: {dp_mcf.client_weights}')
                dp_mcf.gradient_update(client_models)

                # Evaluate global model on test data
                test_loader = load_test_data(args)
                global_accuracy = evaluate_model(dp_mcf.global_model, test_loader, device)
                test_accuracies.append(global_accuracy)
                print(f"Round {t+1} completed. Global Model Test Accuracy: {global_accuracy:.4f}")
                # print(f'Client losses: ', client_losses)


        # Save the global model after all rounds
        model_base_dir = os.path.join(args.model_dir, args.data, args.data_setting)
        os.makedirs(model_base_dir, exist_ok=True)
        torch.save(global_model.state_dict(), os.path.join(model_base_dir, f'global_model_{run}.pt'))

    mean_accuracy = np.mean(test_accuracies)
    std_accuracy = np.std(test_accuracies)

    print(f"Final Results - Average Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}")
    print("DPMCF training completed.")
