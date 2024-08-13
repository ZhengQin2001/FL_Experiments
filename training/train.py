import os
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from models.cnn import CNN
from utils.data_utils import load_client_data, load_test_data
from utils.train_utils import train_client, evaluate_model
from algorithms.fedavg import fedavg_aggregation
from algorithms.afl import afl_aggregation

def compute_fairness(client_losses, global_loss, client_weights):
    fairness = sum([weight * (client_loss - global_loss) ** 2 for client_loss, weight in zip(client_losses, client_weights)])
    return fairness

def federated_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the global model
    global_model = CNN(args.data).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    client_weights = [1.0 / args.num_clients] * args.num_clients  # Assuming equal weights initially

    # Track accuracies and fairness metrics across runs
    test_accuracies = []
    client_fairness_values = []

    for run in range(args.num_runs):
        print(f"Run {run+1}/{args.num_runs}")
        torch.manual_seed(run)
        np.random.seed(run)

        # Reset the global model for each run
        global_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

        for round in range(args.rounds):
            print(f"Round {round+1}/{args.rounds}")
            client_models = []
            client_losses = []
            client_accuracies = {}

            for client_id in tqdm(range(args.num_clients), desc="Training clients"):
                client_model = CNN(args.data).to(device)
                client_model.load_state_dict(global_model.state_dict())
                train_loader, val_loader = load_client_data(args, client_id)

                optimizer = optim.SGD(client_model.parameters(), lr=args.lr)
                train_client(client_model, train_loader, optimizer, criterion, device, return_loss=False)

                # Validate the client model after training
                accuracy = evaluate_model(client_model, val_loader, device)
                client_accuracies[client_id] = accuracy
                # print(f"Client {client_id} validation accuracy: {accuracy:.2f}")

                # Calculate the loss for client fairness
                client_loss = 0.0
                total_samples = 0
                client_model.eval()
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = client_model(data)
                        loss = criterion(output, target).item()
                        client_loss += loss * data.size(0)
                        total_samples += data.size(0)
                client_loss /= total_samples
                client_losses.append(client_loss)

                client_models.append(client_model)

            # Select aggregation method based on args.federated_type
            if args.federated_type == 'fedavg':
                global_model = fedavg_aggregation(global_model, client_models)
            elif args.federated_type == 'afl':
                global_model, client_weights = afl_aggregation(global_model, client_models, client_losses, client_weights)
            else:
                raise ValueError(f"Unsupported federated type: {args.federated_type}")

            print('Client weights: ', client_weights)
            # Calculate client fairness (worst-off client loss)
            client_fairness = max(client_losses)
            client_fairness_values.append(client_fairness)

            # Calculate global test accuracy
            test_loader = load_test_data(args)
            global_accuracy = evaluate_model(global_model, test_loader, device)
            print(f"Round {round+1} completed. Global Model Test Accuracy: {global_accuracy:.4f}. Client Model Accuracy: {client_accuracies}")

        # Store the global test accuracy for this run
        test_accuracies.append(global_accuracy)

        # Ensure the directory exists and save the global model
        model_base_dir = os.path.join(args.model_dir, args.data, args.data_setting)
        os.makedirs(model_base_dir, exist_ok=True)
        torch.save(global_model.state_dict(), os.path.join(model_base_dir, f'global_model_{run}.pt'))

    # Calculate mean and std for test accuracy and fairness metrics
    mean_accuracy = np.mean(test_accuracies)
    std_accuracy = np.std(test_accuracies)
    mean_client_fairness = np.mean(client_fairness_values)
    std_client_fairness = np.std(client_fairness_values)

    print(f"Final Results - Average Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.6f}")
    print(f"Final Results - Client Fairness: {mean_client_fairness:.2f} ± {std_client_fairness:.6f}")

    print("Federated training completed.")
