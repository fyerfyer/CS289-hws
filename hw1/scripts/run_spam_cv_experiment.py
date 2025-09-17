from load import load_spam_data
from evaluate import perform_spam_k_fold_cv
from visualization import plot_performance

if __name__ == '__main__':
    X_all, y_all, _ = load_spam_data('../data/spam-data.npz')
    print("--- Data loading complete ---")

    C_values_to_test = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4]
    K_FOLDS = 5 

    print(f"\n--- Starting {K_FOLDS}-Fold Cross-Validation ---")
    print(f"Testing C values: {C_values_to_test}")

    performance_history = {}

    for c in C_values_to_test:
        avg_accuracy = perform_spam_k_fold_cv(X_all, y_all, C_value=c, k=K_FOLDS)

        print(f"  Average accuracy for C = {c}: {avg_accuracy:.4f}")
        performance_history[c] = avg_accuracy

    best_C = max(performance_history, key=performance_history.get)
    best_accuracy = performance_history[best_C]

    print("-" * 30)
    print("Cross-Validation finished!")
    print(f"The best performing C value is: {best_C}")
    print(f"It achieved an average accuracy of {best_accuracy:.4f}")

    if performance_history:
        plot_performance(performance_history, title=f'{K_FOLDS}-Fold CV Performance for Spam SVM')