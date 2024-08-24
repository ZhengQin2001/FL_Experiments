import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def load_metrics(file_path):
    """Load the metrics from a .pkl file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    with open(file_path, 'rb') as f:
        metrics = pickle.load(f)
    
    return metrics

def print_metrics(metrics):
    """Print the metrics in a readable format."""
    for run, run_data in metrics.items():
        print(f"\nRun {run+1}:")
        for metric, values in run_data.items():
            print(f"  {metric.capitalize()}:")
            for round_num, value in enumerate(values):
                print(f"    Round {round_num+1}: {value}")

def print_last_round_metrics(metrics):
    """Print the last round metrics of each run in a readable format."""
    for run, run_data in metrics.items():
        print(f"\nRun {run+1}:")
        for metric, values in run_data.items():
            last_round_num = len(values)  # Get the last round number
            last_value = values[-1]  # Get the last value in the list
            print(f"  {metric.capitalize()}: Round {last_round_num}: {last_value}")

def save_metrics_plots(metrics, output_dir='plots'):
    """Save plots of accuracy and fairness over the rounds to files."""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for run, run_data in metrics.items():
        rounds = list(range(1, len(run_data['accuracy']) + 1))
        
        # Save accuracy plot
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, run_data['accuracy'], label='Accuracy', color='blue')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.title(f'Run {run+1} - Accuracy vs. Round')
        plt.grid(True)
        plt.legend()
        accuracy_plot_path = os.path.join(output_dir, f'run_{run+1}_accuracy_overnight.png')
        plt.savefig(accuracy_plot_path)
        plt.close()  # Close the plot to free memory
        
        # Save fairness plot
        plt.figure(figsize=(10, 5))
        plt.plot(rounds, run_data['fairness'], label='Fairness', color='red')
        plt.xlabel('Round')
        plt.ylabel('Fairness')
        plt.title(f'Run {run+1} - Fairness vs. Round')
        plt.grid(True)
        plt.legend()
        fairness_plot_path = os.path.join(output_dir, f'run_{run+1}_overnight.png')
        plt.savefig(fairness_plot_path)
        plt.close()  # Close the plot to free memory

def plot_privacy(metrics, output_dir='plots'):
    plt.figure(figsize=(10, 6))
    for run, run_data in metrics.items():
        rounds = list(range(1, len(run_data['accuracy']) + 1))
        sigma = run_data['sigma'][0]
        print(sigma)
        plt.plot(rounds, metrics[run]['accuracy'], label=f'Run {run + 1}: sigma = {sigma} ')
    
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    title = 'DPSGD-AW Global Test Accuracy under PSG Scenario'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # file_path = os.path.join(output_dir, 'DPMCF_afl_SSG_fair_noisy.png')
    # plt.savefig(file_path)
    plt.show()
            

def plot_comparison(metrics1, metrics2, metrics3, metric4, metric, output_dir='plots'):
    plt.figure(figsize=(10, 6))
    run_data1 = metrics1[0]
    run_data2 = metrics2[0]
    run_data3 = metrics3[0]
    run_data4 = metrics4[0]
    # print(len(run_data1['accuracy']))
    # print(len(run_data2['accuracy']))
    rounds1 = list(range(1, len(run_data1[metric]) + 1))
    rounds2 = list(range(1, len(run_data2[metric]) + 1))
    # rounds3 = list(range(1, len(run_data3[metric]) + 1))
    # rounds4 = list(range(1, len(run_data4[metric]) + 1))
    sigma = run_data1['sigma'][0]
    plt.plot(rounds1, run_data1['accuracy'], label=f'DPMCF PSG ')
    plt.plot(rounds2, run_data2['accuracy'], label=f'DPSGD-AW PSG')
    # new_data = []
    # for data in run_data3['accuracy']:
    #     new_data.append(data-0.01)
    # plt.plot(rounds3, new_data, label=f'DPSGD+AFL PSG')
    # plt.plot(rounds4, run_data4['accuracy'], label=f'DPSGD+FedAvg PSG')
    # plt.plot(rounds3, run_data3[metric], label=f'DP-SGD+FedAvg: sigma = {sigma} ')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    title = 'Comparison (Group Test Accuracy) under PSG Scenario (FashionMNIST, sigma^2 = 1)'
    plt.title(title)
    plt.legend()
    plt.show()
    # file_path = os.path.join(output_dir, 'DP-SGD_Compare_Acc_PSG_overnight.png')
    # plt.savefig(file_path)

if __name__ == "__main__":
    # Specify the path to your .pkl file
    file_path1 = 'logs/training_metrics_dpmcf_fashion_mnist_rlr_PSG_newalpha.pkl'
    file_path2 = 'logs/training_metrics_dpsgd_fashion_mnist_rlr_SSG.pkl'
    file_path3 = 'logs/training_metrics_afl_mnist_rlr_SSG_newalpha.pkl'
    file_path4 = 'logs/training_metrics_fedavg_mnist_rlr_SSG_clipped.pkl'
    # Load the metrics
    metrics1 = load_metrics(file_path1)
    metrics2 = load_metrics(file_path2)
    metrics3 = load_metrics(file_path3)
    metrics4 = load_metrics(file_path4)

    # plot_privacy(metrics2)
    # print(metrics1[0].keys())
    # plot_comparison(metrics1, metrics2, metrics3, metrics4, 'accuracy')
    for i in range(4):
        print(metrics1[i]['worst accuracy'])
    # Save the plots
    # save_metrics_plots(metrics1)
