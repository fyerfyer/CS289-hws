#!/usr/bin/env python3
"""
Simple Iris Dataset Test - Single Experiment
Quick test to verify the neural network implementation works
"""

import os
import sys
import numpy as np

# Change to the correct directory first
os.chdir('/home/fyerfyer/Downloads/cs289/hw6release/code')

from neural_networks.utils import AttrDict
from neural_networks.models import initialize_model
from neural_networks.datasets import initialize_dataset
from neural_networks.logs import Logger

def test_single_experiment():
    """
    Run a single experiment to test the implementation
    """
    print("Testing Iris Dataset with Two-Layer Neural Network")
    print("="*50)
    
    # Set random seed for reproducibility
    seed = 42
    np.random.seed(seed)
    
    # Define network architecture
    # Layer 1: Fully connected with ReLU activation
    fc1 = AttrDict({
        "name": "fully_connected",
        "activation": "relu",
        "weight_init": "xavier_uniform",
        "n_out": 20,  # 20 hidden units
    })

    # Output layer: Fully connected with softmax for classification
    fc_out = AttrDict({
        "name": "fully_connected",
        "activation": "softmax",
        "weight_init": "xavier_uniform",
        "n_out": None  # Will be set to 3 for iris (3 classes)
    })

    layer_args = [fc1, fc_out]

    # Define optimizer (SGD with momentum)
    optimizer_args = AttrDict({
        "name": "SGD",
        "lr": 0.05,  # Learning rate
        "lr_scheduler": "constant",
        "lr_decay": 0.99,
        "stage_length": 1000,
        "staircase": True,
        "clip_norm": 1.0,
        "momentum": 0.9,
    })

    # Model configuration
    model_args = AttrDict({
        "name": "feed_forward",
        "loss": "cross_entropy",
        "layer_args": layer_args,
        "optimizer_args": optimizer_args,
        "seed": seed,
    })

    # Data configuration
    data_args = AttrDict({
        "name": "iris",
        "batch_size": 15,  # Small batch size for iris dataset
    })

    # Logging configuration
    log_args = AttrDict({
        "save": True,
        "plot": True,
        "save_dir": "experiments/",
    })

    # Create experiment name
    model_name = f"iris_test_lr{optimizer_args.lr}_hidden{fc1.n_out}_seed{seed}"

    print(f"Experiment: {model_name}")
    print(f"Architecture: Input(4) -> Hidden({fc1.n_out}, ReLU) -> Output(3, Softmax)")
    print(f"Learning Rate: {optimizer_args.lr}")
    print(f"Batch Size: {data_args.batch_size}")
    print("-" * 50)

    # Initialize components
    try:
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
        
        print("Successfully initialized model and dataset!")
        print(f"Training samples: {dataset.train.n_samples}")
        print(f"Validation samples: {dataset.validate.n_samples}")
        print(f"Test samples: {dataset.test.n_samples}")
        print(f"Input features: {dataset.train.data_.shape[1]}")
        
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        return False

    # Train the model
    try:
        print("\nStarting training...")
        epochs = 50  # Fewer epochs for quick test
        model.train(dataset, epochs=epochs)
        print("Training completed!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return False

    # Test the model
    try:
        print("\nEvaluating on test set...")
        test_results = model.test(dataset)
        print("Testing completed!")
        
        print(f"\nResults saved to: experiments/{model_name}/")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_single_experiment()
    if success:
        print("\n" + "="*50)
        print("SUCCESS: Neural network test completed!")
        print("Check the experiments/ folder for detailed results and plots.")
    else:
        print("\n" + "="*50)
        print("FAILED: There were errors during the test.")