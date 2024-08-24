import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
import torch.optim as optim
from models.cnn import CNN
from models.mlp import MLP
from models.robust_logistic_regression import RobustLogisticRegression
from utils.data_utils import load_client_data, load_test_data
from utils.train_utils import train_client, evaluate_model
from algorithms.fedavg import fedavg_aggregation
from algorithms.afl import afl_aggregation
from collections import defaultdict
from copy import deepcopy

def save_metrics(metrics, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dict(metrics), f)

def calculate_fairness(client_losses, client_weights):
    """Calculate fairness as the weighted variance of client losses."""
    global_loss = np.sum([weight * loss for weight, loss in zip(client_weights, client_losses)])
    fairness = np.sum([weight * (loss - global_loss) ** 2 for weight, loss in zip(client_weights, client_losses)])
    return fairness

def federated_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.base_model == 'cnn':
        global_model = CNN(args.data).to(device)
    elif args.base_model == 'mlp':
        global_model = MLP(args.data, 2, 128, 0.3).to(device)
    elif args.base_model == 'rlr':
        global_model = RobustLogisticRegression(args.data).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    sigma_values = np.linspace(start=1, stop=args.sigma_max, num=args.num_runs)  # Sigma changes per run
    
    # Early stopping parameters
    patience = args.patience  # Number of rounds to wait for improvement
    min_delta = args.min_delta  # Minimum change to qualify as an improvement

    # Prepare data structures for logging
    run_metrics = defaultdict(lambda: defaultdict(list))

    for run in range(args.num_runs):
        args.sigma = sigma_values[run]  # Set sigma for the entire run
        print(f"Run {run+1}/{args.num_runs} with sigma={args.sigma:.4f}")
        torch.manual_seed(run)
        np.random.seed(run)
        global_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

        best_accuracy = 0.0
        rounds_no_improvement = 0

        for round in range(args.rounds):
            print(f"Round {round+1}/{args.rounds}")
            client_models = []
            client_losses = []
            client_accuracies = {}
            client_weights = [1.0 / args.num_clients] * args.num_clients
            for client_id in tqdm(range(args.num_clients), desc="Training clients"):
                client_model = deepcopy(global_model)
                train_loader, val_loader = load_client_data(args, client_id)
                optimizer = optim.SGD(client_model.parameters(), lr=args.lr)

                args.sampling_prob = 0.6 * client_weights[client_id]
                train_client(args, client_model, train_loader, optimizer, criterion, device, return_loss=False, private=True, with_sample=False)

                accuracy = evaluate_model(client_model, val_loader, device)
                client_accuracies[client_id] = accuracy

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
            if args.federated_type in ['fedavg']:
                global_model = fedavg_aggregation(global_model, client_models)
            elif args.federated_type == 'afl':
                global_model, client_weights = afl_aggregation(args, global_model, client_models, client_losses, client_weights, use_afl=True, alpha=0.3)
            elif args.federated_type in ['dpsgd']: 
                global_model, client_weights = afl_aggregation(args, global_model, client_models, client_losses, client_weights, noisy=False)
            elif args.federated_type in ['dpmcf']:
                global_model, client_weights = afl_aggregation(args, global_model, client_models, client_losses, client_weights, use_dpmcf=True)

            # client_fairness = np.std(client_losses)
            client_fairness = calculate_fairness(client_losses, client_weights)
            run_metrics[run]['fairness'].append(client_fairness)
            run_metrics[run]['best accuracy'].append(max(client_accuracies))
            run_metrics[run]['worst accuracy'].append(min(client_accuracies))
            print(f'Client_accuracies: {client_accuracies}')
            run_metrics[run]['best client loss'].append(max(client_losses))
            run_metrics[run]['worst client loss'].append(min(client_losses))
            test_loader = load_test_data(args)
            global_accuracy = evaluate_model(global_model, test_loader, device)
            run_metrics[run]['accuracy'].append(global_accuracy)
            run_metrics[run]['sigma'].append(args.sigma)
            print(f'\n Round {round+1} completed. Global test accuracy: {global_accuracy}, Fairness: {client_fairness}')
            # Early stopping check
            if global_accuracy - best_accuracy > min_delta:
                best_accuracy = global_accuracy
                rounds_no_improvement = 0
            else:
                rounds_no_improvement += 1

            if rounds_no_improvement >= patience:
                print(f"Early stopping triggered after {round+1} rounds.")
                break

            # Save metrics after each round
            save_metrics(run_metrics, os.path.join('logs', f'training_metrics_{args.federated_type}_{args.data}_{args.base_model}_{args.data_setting}_1545.pkl'))

        model_base_dir = os.path.join(args.model_dir, args.data, args.data_setting)
        os.makedirs(model_base_dir, exist_ok=True)
        torch.save(global_model.state_dict(), os.path.join(model_base_dir, f'global_model_{run}.pt'))

        # Save metrics after each run
        save_metrics(run_metrics, os.path.join('logs', f'training_metrics_{args.federated_type}_{args.data}_{args.base_model}_{args.data_setting}_1545.pkl'))

    print("Federated training completed. Metrics saved for analysis.")
