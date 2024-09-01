## FL_EXPERIMENTS
### Overview

This repository contains the implementation of various Federated Learning (FL) algorithms aimed at addressing client fairness and privacy in FL settings, particularly under non-IID data distributions. The repository includes code for algorithms, model definitions, synthetic data generation, and training scripts. The main focus is on implementing and experimenting with the DP-SGD-AW and DPMCF algorithms.

### Repository Structure

- **algorithms**: Contains the implementation of different federated learning algorithms.
  - `afl.py`: Implements Adaptive Federated Learning (AFL). Multiple options are available to adjust the AFL for specific needs, for example incorporating Differential Privacy.
  - `dpmcf.py`: Contains the implementation of the Differential Privacy Minimax Client Fairness (DPMCF) algorithm.
  - `fedavg.py`: Implements the Federated Averaging (FedAvg) algorithm, which serves as a baseline for comparison.

- **models**: This folder contains base model definitions used in the experiments. All the Python files in this folder have been copied from the [FedTorch repository](https://github.com/MLOPTPSU/FedTorch/tree/main), which provides a collection of base models for federated learning.
  - `cnn.py`: Defines a Convolutional Neural Network (CNN) model.
  - `logistic_regression.py`: Implements a logistic regression model.
  - `mlp.py`: Defines a Multi-Layer Perceptron (MLP) model.
  - `robust_logistic_regression.py`: Contains the implementation of a robust logistic regression model.

- **plots**: (Not described as it is not part of the request, but likely contains scripts or data for generating visualizations of results.)

- **synthetic**: Contains scripts related to synthetic data generation and experiments.
  - `experiment1.py`: Runs experiments using synthetic data.
  - `synthetic_data.py`: Generates synthetic datasets used for training and testing the models.

- **training**: Contains the main training script used to conduct federated learning experiments.
  - `train.py`: The primary script that manages the training process across different FL algorithms.

- **utils**: Utility scripts that assist with data preparation, model training, and evaluation.
  - `data_preparation.py`: Prepares datasets for training and evaluation.
  - `data_utils.py`: Contains various utility functions related to data handling.
  - `test_data_utils.py`: Utilities specifically for handling test data.
  - `train_utils.py`: Provides helper functions for training models, such as metrics calculation and model evaluation.

- **main.py**: The main entry point for running the experiments. This script sets up the environment, initializes parameters, and triggers the training process.

- **parameters.py**: Contains configurable parameters for the experiments, including hyperparameters, dataset configurations, and algorithm-specific settings.

- **read_results.py**: A script for reading and processing the results of the experiments.

- **run_experiments.py**: Automates the process of running multiple experiments, possibly with varying configurations, and logging the results.

### How to Use

1. **Set Up**: Ensure that all dependencies are installed as specified in the requirements (not listed here, so make sure to check the relevant files or documentation).

2. **Running Experiments**:
   - Configure your experiment parameters in `parameters.py`.
   - Use `main.py` to run an individual experiment.
   - Use `run_experiments.py` to automate running a series of experiments with different settings.

### Acknowledgments

The base models in the `models` directory have been adapted from the [FedTorch repository](https://github.com/MLOPTPSU/FedTorch/tree/main). FedTorch provides a comprehensive suite of tools and models for federated learning research. 

This repository builds on those models to explore advanced federated learning techniques with a focus on fairness and privacy.
