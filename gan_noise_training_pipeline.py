import itertools
import subprocess
import os

# Define noise types
noise_types = ['normal', 'uniform', 'lognormal', 'exponential', 'gamma', 'poisson', 'random_binary']

# Define ranges for parameters
noise_params = {
    # 'normal': {'noise_std': [1.0], 'noise_mean': [0.0]},
    'normal': {'noise_std': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], 'noise_mean': [0.0, 1.0, -1.0, 5.0, -5.0]},
    'uniform': {'noise_min': [-1.0, 0.0], 'noise_max': [0.5, 1.0]},
    'lognormal': {'noise_std': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0], 'noise_mean': [0.0, 1.0, -1.0, 5.0, -5.0]},
    'exponential': {'noise_lambda': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]},
    'gamma': {'noise_alpha': [1.0, 2.0, 3.0], 'noise_beta': [1.0, 2.0]},
    'poisson': {'noise_lambda': [1.0, 2.0, 5.0]},
    'random_binary': {}  # No parameters to vary for random_binary
}


# Set default configurations based on dataset
dataset_configs = {
    'MNIST': {
        'project_wandb': 'MNIST-noise-investigation',
        'latent_size': 64,
        'hidden_size': 256,
        'image_size': 784,  # Flattened 28x28 image
        'batch_size': 50,
        'learning_rate': 0.0002,
        'num_epochs': 200,
        'hidden_layers_d': 1,
        'hidden_layers_g': 2
    },
    'BAS': {
        'project_wandb': 'BAS-noise-investigation-dim-02',
        'img_dim': 2,  # Set the Bars-and-Stripes image dimension (e.g., 4x4)
        'latent_size': 2,
        'hidden_size': 32,
        'batch_size': 20,
        'learning_rate': 0.0002,
        'num_epochs': 200,
        'n_samples': 1_000,
        'hidden_layers_d': 3,
        'hidden_layers_g': 3,
        'reduction_factor': 0.5 ,
        'log_wandb': True ,
        'penalty_weight': 0,
        'penalty_weight_entropy': 0.5,
        'tolerance': 0.3
    }
}

# Calculate image_size based on img_dim
dataset_configs['BAS']['image_size'] = dataset_configs['BAS']['img_dim'] ** 2  # 4 for img_dim = 2


# Select dataset and get configurations
dataset = 'BAS'  # Change this to 'BAS' for Bars-and-Stripes
config = dataset_configs[dataset]

# Generate commands for each noise type with all combinations of parameter values
for noise_type in noise_types:
    param_keys = noise_params[noise_type].keys()
    if param_keys:  # If there are parameters for the noise type
        param_values = itertools.product(*noise_params[noise_type].values())
        
        for param_combination in param_values:
            params = {key: value for key, value in zip(param_keys, param_combination)}
            # Create the command
            cmd = f"python train.py --noise_type {noise_type} --dataset {dataset}"
            
            # Append each parameter to the command
            for param, value in params.items():
                cmd += f" --{param} {value}"
            
            # Append the dataset-specific configurations to the command
            for param, value in config.items():
                cmd += f" --{param} {value}"

            # Run the command using subprocess
            print(f"Running command: {cmd}")
            os.system(cmd)
    else:
        # If there are no parameters (e.g., random_binary)
        cmd = f"python train.py --noise_type {noise_type} --dataset {dataset}"
        
        # Append the dataset-specific configurations to the command
        for param, value in config.items():
            cmd += f" --{param} {value}"

        print(f"Running command: {cmd}")
        os.system(cmd)
