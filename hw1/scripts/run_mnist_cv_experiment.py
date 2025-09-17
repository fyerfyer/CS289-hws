from load import load_mnist_data
from evaluate import perform_mnist_k_fold_cv 
from visualization import plot_performance

if __name__ == '__main__':
    X_all, y_all, _ = load_mnist_data('../data/mnist-data.npz')
    print("--- Data loading complete ---")

    C_values_to_test = [0.01, 0.1, 1, 10, 100]
    K_FOLDS = 5

    print(f"\n--- Starting {K_FOLDS}-Fold Cross-Validation for MNIST ---")
    print(f"Testing C values: {C_values_to_test}")

    performance_history = {}

    for c in C_values_to_test:
        print(f"\n--- Evaluating for C = {c} ---")
        avg_accuracy = perform_mnist_k_fold_cv(X_all, y_all, C_value=c, k=K_FOLDS)

        print(f"  => Average accuracy for C = {c}: {avg_accuracy:.4f}")
        performance_history[c] = avg_accuracy

    best_C = max(performance_history, key=performance_history.get)
    best_accuracy = performance_history[best_C]

    print("-" * 30)
    print("Cross-Validation finished!")
    print(f"The best performing C value is: {best_C}")
    print(f"It achieved an average accuracy of {best_accuracy:.4f} on the subset.")

    if performance_history:
        plot_performance(performance_history, title=f'{K_FOLDS}-Fold CV Performance for MNIST SVM')
