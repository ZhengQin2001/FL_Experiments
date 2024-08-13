import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from models.cnn import CNN
from utils.data_utils import load_client_data, load_test_data
from utils.train_utils import train_client, evaluate_model
from algorithms.dpmcf import DPMCF
import os

def dpmcf_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the global model
    global_model = CNN(args.data).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # Initialize DPMCF algorithm with the global model
    dp_mcf = DPMCF(args, global_model)

    # Track accuracies and fairness metrics across runs
    test_accuracies = []
    client_fairness_values = []

    for run in range(args.num_runs):
        print(f"Run {run+1}/{args.num_runs}")
        torch.manual_seed(run)
        np.random.seed(run)

        # Reset the global model for each run
        global_model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

        for i_t in range(args.num_clients):
            # Step 1: Load the client's data
            train_loader, val_loader = load_client_data(args, i_t)

            # Step 2: Further split the training data into positive and negative subsets
            train_data = train_loader.dataset
            total_size = len(train_data)
            split_size = total_size // 2

            # Randomly split the dataset into two equal parts
            positive_subset, negative_subset = random_split(train_data, [split_size, total_size - split_size])

            # Create DataLoaders for the split subsets
            positive_loader = DataLoader(positive_subset, batch_size=args.batch_size, shuffle=True)
            negative_loader = DataLoader(negative_subset, batch_size=args.batch_size, shuffle=True)

            # Store loaders for each client
            dp_mcf.client_data[i_t] = (positive_loader, negative_loader)

        for t in tqdm(range(args.rounds), desc="Rounds"):
            client_losses = []

            # Step 3: Sample a client based on current weights
            i_t = np.random.choice(args.num_clients, p=dp_mcf.client_weights)

            # Step 4: Load the split loaders for the sampled client
            positive_loader, negative_loader = dp_mcf.client_data[i_t]

            # Sample mini-batches from the positive and negative datasets
            negative_batch = next(iter(negative_loader))
            positive_batch = next(iter(positive_loader))

            # Train the client model
            client_model = CNN(args.data).to(device)
            client_model.load_state_dict(global_model.state_dict())

            optimizer = optim.SGD(client_model.parameters(), lr=args.lr)

            # We use the negative batch for training (as indicated by the algorithm)
            training_loss = train_client(client_model, negative_loader, optimizer, criterion, device, return_loss=True)
            client_losses.append(training_loss)

            accuracy = evaluate_model(client_model, positive_loader, device)
            print(f"Client {i_t} Validation Accuracy: {accuracy:.2f}")

            # Step 5: Update global model and client weights using DPMCF
            dp_mcf.dpmcf(client_model, training_loss, len(negative_batch))

            # Evaluate global model on test data
            test_loader = load_test_data(args)
            global_accuracy = evaluate_model(global_model, test_loader, device)
            test_accuracies.append(global_accuracy)
            print(f"Round {t+1} completed. Global Model Test Accuracy: {global_accuracy:.4f}")

        # Save the global model after all rounds
        model_base_dir = os.path.join(args.model_dir, args.data, args.data_setting)
        os.makedirs(model_base_dir, exist_ok=True)
        torch.save(global_model.state_dict(), os.path.join(model_base_dir, f'global_model_{run}.pt'))

    mean_accuracy = np.mean(test_accuracies)
    std_accuracy = np.std(test_accuracies)

    print(f"Final Results - Average Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.2f}")
    print("DPMCF training completed.")
