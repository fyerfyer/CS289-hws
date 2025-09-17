from typing import Any, Dict, List, Tuple
import numpy as np 
from utils import create_k_folds, evaluate
from preprocess import fit_spam_scaler, apply_spam_scaler, process_mnist_features
from svm_model import train_svm_model

def evaluate_best_C(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C_values: List[float]
) -> Tuple[Any, float, Dict]:
    best_accuracy = -1.0
    best_C = None 
    best_model = None 
    performance_history = {}

    for c in C_values:
        model = train_svm_model(X_train, y_train, C=c)
        y_pred = model.predict(X_val)
        accuracy = evaluate(y_pred, y_val)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = c 
            best_model = model

        performance_history[c] = accuracy
    
    return best_model, best_C, performance_history

def perform_spam_k_fold_cv(
    X_all: np.ndarray,
    y_all: np.ndarray,
    C_value: float,
    k: int
) -> float:
    k_folds = create_k_folds(X_all, y_all, k=k)
    fold_accuracies = []

    for i in range(k):
        X_val_raw, y_val = k_folds[i]
        X_train_raw = np.concatenate([k_folds[j][0] for j in range(k) if j != i])
        y_train = np.concatenate([k_folds[j][1] for j in range(k) if j != i])

        scaler = fit_spam_scaler(X_train_raw)
        X_train_scaled = apply_spam_scaler(X_train_raw, scaler)
        X_val_scaled = apply_spam_scaler(X_val_raw, scaler)

        model = train_svm_model(X_train_scaled, y_train, C_value)
        y_pred = model.predict(X_val_scaled)
        accuracy = evaluate(y_pred, y_val)
        fold_accuracies.append(accuracy)

    return np.mean(fold_accuracies)


def perform_mnist_k_fold_cv(
    X_all: np.ndarray,
    y_all: np.ndarray,
    C_value: float,
    k: int
) -> float:
    k_folds = create_k_folds(X_all, y_all, k=k)
    fold_accuracies = []

    for i in range(k):
        X_val_raw, y_val = k_folds[i]
        X_train_raw = np.concatenate([k_folds[j][0] for j in range(k) if j != i])
        y_train = np.concatenate([k_folds[j][1] for j in range(k) if j != i])

        X_train = process_mnist_features(X_train_raw)
        X_val = process_mnist_features(X_val_raw)

        model = train_svm_model(X_train, y_train, C_value)
        y_pred = model.predict(X_val)
        accuracy = evaluate(y_pred, y_val)
        fold_accuracies.append(accuracy)
        print(f"  Fold {i+1}/{k} accuracy for C={C_value}: {accuracy:.4f}")

    return np.mean(fold_accuracies)

