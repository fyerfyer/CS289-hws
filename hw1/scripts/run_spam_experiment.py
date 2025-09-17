import numpy as np
from load import load_spam_data
from utils import shuffle_and_split
from preprocess import fit_spam_scaler, apply_spam_scaler
from evaluate import evaluate_best_C
from visualization import plot_confusion_matrix, plot_performance

if __name__ == '__main__':
    print("--- Step 1: Loading MNIST Data ---")
    X_all, y_all, _ = load_spam_data('../data/spam-data.npz')
    # N_SUBSET = 10000
    # print(f"Using a subset of {N_SUBSET} samples for the experiment for performance reasons.")
    # X_subset = X_all[:N_SUBSET]
    # y_subset = y_all[:N_SUBSET]

    X_train_raw, X_val_raw, y_train, y_val  = shuffle_and_split(X_all, y_all, validation_size=0.2, seed=42)

    scaler = fit_spam_scaler(X_train_raw)

    X_train_scaled = apply_spam_scaler(X_train_raw, scaler)
    X_val_scaled = apply_spam_scaler(X_val_raw, scaler)

    print("--- Data preparation complete ---")
    print(f"Train features shape: {X_train_scaled.shape}")
    print(f"Validation features shape: {X_val_scaled.shape}")

    print("\n--- Evaluating hyperparameters ---")
    C_values_to_test = [0.01, 0.1, 1, 10, 100]

    best_model, best_acc, history = evaluate_best_C(
        X_train_scaled, y_train, X_val_scaled, y_val, C_values_to_test
    )

    print(f"\nBest C found: {best_model.C if best_model else 'None'}")
    print(f"Validation accuracy with best C: {best_acc:.4f}")

    print("\n--- Visualizing results ---")

    if history:
        plot_performance(history, title='Spam SVM Performance vs. C value')

    if best_model:
        best_predictions = best_model.predict(X_val_scaled)
        plot_confusion_matrix(y_val, best_predictions,
                              class_names=['Not Spam', 'Spam'],
                              title=f'Confusion Matrix (C={best_model.C})')

