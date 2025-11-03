# Configuration parameters for the AWAC adaptive optics project

# Hyperparameters for training
HYPERPARAMS = {
    'learning_rate': 3e-4,
    'discount_factor': 0.99,
    'tau': 0.005,
    'batch_size': 128,
    'num_epochs': 50,
    'eval_freq': 5000,
    'training_steps': 50000,
}

# Environment settings
ENV_SETTINGS = {
    'max_steps': 200,
    'gainCL': 0.6,
    'sampling_rate': 1000,
    'exposure_time': 1.0,
    'clock_rate': 1000,
    'light_ratio': 0.3,
}

# Paths for saving models and logs
PATHS = {
    'model_save_dir': './models/',
    'log_dir': './logs/',
    'offline_data_path': './data/offline_dataset.npz',
}

# Ensure directories exist
import os

for path in PATHS.values():
    os.makedirs(os.path.dirname(path), exist_ok=True)