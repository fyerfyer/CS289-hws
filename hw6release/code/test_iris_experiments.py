#!/usr/bin/env python3
"""
Iris Dataset Experiments for Two-Layer Fully Connected Network
Testing different combinations of learning rate and hidden layer size
"""

import os
import sys
import numpy as np
from neural_networks.utils import AttrDict
from neural_networks.models import initialize_model
from neural_networks.datasets import initialize_dataset
from neural_networks.logs import Logger

def run_experiment(learning_rate, hidden_size, experiment_name, epochs=100, seed=42):
    """
    Run a single experiment with given hyperparameters
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment_name}")
    print(f"Learning Rate: {learning_rate}, Hidden Size: {hidden_size}")
    print(f"{'='*60}")
    
    # Set random seed
    np.random.seed(seed)
    
    # Define layer arguments
    fc1 = AttrDict({
        "name": "fully_connected",
        "activation": "relu",
        "weight_init": "xavier_uniform",
        "n_out": hidden_size,
    })

    fc_out = AttrDict({
        "name": "fully_connected",
        "activation": "softmax",
        "weight_init": "xavier_uniform",
        "n_out": None  # Will be set by dataset (3 classes for iris)
    })

    layer_args = [fc1, fc_out]

    # Define optimizer arguments
    optimizer_args = AttrDict({
        "name": "SGD",
        "lr": learning_rate,
        "lr_scheduler": "constant",
        "lr_decay": 0.99,
        "stage_length": 1000,
        "staircase": True,
        "clip_norm": 1.0,
        "momentum": 0.9,
    })

    # Define model arguments
    model_args = AttrDict({
        "name": "feed_forward",
        "loss": "cross_entropy",
        "layer_args": layer_args,
        "optimizer_args": optimizer_args,
        "seed": seed,
    })

    # Define data arguments
    data_args = AttrDict({
        "name": "iris",
        "batch_size": 25,
    })

    # Define log arguments
    log_args = AttrDict({
        "save": True,
        "plot": True,
        "save_dir": "experiments/",
    })

    # Create model name for saving
    model_name = f"iris_{experiment_name}_lr{learning_rate}_hidden{hidden_size}_seed{seed}"

    # Initialize logger
    logger = Logger(
        model_name=model_name,
        model_args=model_args,
        data_args=data_args,
        save=log_args.save,
        plot=log_args.plot,
        save_dir=log_args.save_dir,
    )

    # Initialize model
    model = initialize_model(
        name=model_args.name,
        loss=model_args.loss,
        layer_args=model_args.layer_args,
        optimizer_args=model_args.optimizer_args,
        logger=logger,
    )

    # Initialize dataset
    dataset = initialize_dataset(
        name=data_args.name,
        batch_size=data_args.batch_size,
    )

    print(f"Training {model_args.name} neural network on {data_args.name} dataset...")
    print(f"Architecture: Input -> Hidden({hidden_size}) -> Output(3)")
    print(f"Optimizer: {optimizer_args.name} with lr={learning_rate}, momentum={optimizer_args.momentum}")
    
    # Train model
    model.train(dataset, epochs=epochs)
    
    # Test model
    test_results = model.test(dataset)
    
    print(f"Experiment {experiment_name} completed!")
    print(f"Results saved to: experiments/{model_name}/")
    
    return test_results, model_name

def main():
    """
    Run multiple experiments with different hyperparameter combinations
    """
    print("Starting Iris Dataset Experiments")
    print("Testing different learning rates and hidden layer sizes")
    
    # Change to the correct directory
    os.chdir('/home/fyerfyer/Downloads/cs289/hw6release/code')
    
    # Define hyperparameter combinations to test
    experiments = [
        # (learning_rate, hidden_size, experiment_name)
        (0.01, 10, "exp1_low_lr_small_hidden"),
        (0.01, 25, "exp2_low_lr_medium_hidden"), 
        (0.01, 50, "exp3_low_lr_large_hidden"),
        (0.1, 10, "exp4_high_lr_small_hidden"),
        (0.1, 25, "exp5_high_lr_medium_hidden"),
        (0.05, 25, "exp6_medium_lr_medium_hidden"),
    ]
    
    results = {}
    
    # Run all experiments
    for lr, hidden_size, exp_name in experiments:
        try:
            test_results, model_name = run_experiment(
                learning_rate=lr, 
                hidden_size=hidden_size, 
                experiment_name=exp_name,
                epochs=100,
                seed=42
            )
            results[exp_name] = {
                'learning_rate': lr,
                'hidden_size': hidden_size,
                'test_results': test_results,
                'model_name': model_name
            }
        except Exception as e:
            print(f"Error in experiment {exp_name}: {str(e)}")
            continue
    
    # Print summary of results
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    
    best_error = float('inf')
    best_experiment = None
    
    for exp_name, result in results.items():
        lr = result['learning_rate']
        hidden = result['hidden_size']
        # Note: test_results format depends on your model.test() implementation
        print(f"{exp_name:30} | LR: {lr:6.3f} | Hidden: {hidden:3d} | Model: {result['model_name']}")
        
        # You may need to adjust this based on what your model.test() returns
        # if 'error' in result['test_results']:
        #     error = result['test_results']['error']
        #     if error < best_error:
        #         best_error = error
        #         best_experiment = exp_name
    
    if best_experiment:
        print(f"\nBest performing experiment: {best_experiment}")
        print(f"Best test error: {best_error:.4f}")
    
    print(f"\nAll experiment results saved to: experiments/")
    print("Check the individual folders for detailed logs and plots!")

if __name__ == "__main__":
    main()