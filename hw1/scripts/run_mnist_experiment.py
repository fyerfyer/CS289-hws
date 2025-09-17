import numpy as np

from load import load_mnist_data
from utils import shuffle_and_split
from preprocess import process_mnist_features
from evaluate import evaluate_best_C
from visualization import plot_performance, plot_confusion_matrix

if __name__ == '__main__':
    print("--- Step 1: Loading MNIST Data ---")
    X_all, y_all, _ = load_mnist_data('../data/mnist-data.npz')
    # N_SUBSET = 10000
    # print(f"Using a subset of {N_SUBSET} samples for the experiment for performance reasons.")
    # X_subset = X_all[:N_SUBSET]
    # y_subset = y_all[:N_SUBSET]

    X_train_raw, X_val_raw, y_train, y_val = shuffle_and_split(X_all, y_all, validation_size=0.2, seed=42)

    X_train = process_mnist_features(X_train_raw)
    X_val = process_mnist_features(X_val_raw)


    print("--- Data preparation complete ---")
    print(f"Train features shape: {X_train.shape}")
    print(f"Validation features shape: {X_val.shape}")

    print("\n--- Evaluating hyperparameters ---")
    C_values_to_test = [0.01, 0.1, 1, 10, 100] 

    best_model, best_acc, history = evaluate_best_C(
        X_train, y_train, X_val, y_val, C_values_to_test
    )

    print(f"\nBest C found: {best_model.C if best_model else 'None'}")
    print(f"Validation accuracy with best C: {best_acc:.4f}")

    print("\n--- Visualizing results ---")

    if history:
        plot_performance(history, title='MNIST SVM Performance vs. C value')

    if best_model:
        print("\nGenerating confusion matrix for the best model...")
        best_predictions = best_model.predict(X_val)

        class_names = [str(i) for i in range(10)]
        plot_confusion_matrix(y_val, best_predictions,
                              class_names=class_names,
                              title=f'MNIST Confusion Matrix (C={best_model.C})')
