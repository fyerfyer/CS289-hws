import numpy as np
from typing import List, Optional, Union, Tuple

def shuffle_and_split(
    X: np.ndarray, 
    y: np.ndarray, 
    validation_size: Union[float, int] = 0.2,
    seed: Optional[int] = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if X.shape[0] != y.shape[0]:
        raise ValueError('data shape error')
    
    n_sample = X.shape[0]
    
    rng = np.random.default_rng(seed=seed)
    shuffled_indices = rng.permutation(n_sample)

    X_shuffle = X[shuffled_indices]
    y_shuffle = y[shuffled_indices]

    if isinstance(validation_size, float):
        if not (0.0 <= validation_size <= 1.0):
            raise ValueError('split percentage error')
        split_index = int(n_sample * (1 - validation_size))
    elif isinstance(validation_size, int):
        if not(0 <= validation_size <= n_sample):
            raise ValueError('validation data size error')
        split_index = n_sample - validation_size

    X_train = X_shuffle[:split_index]
    y_train = y_shuffle[:split_index]
    X_val = X_shuffle[split_index:]
    y_val = y_shuffle[split_index:]

    return X_train, X_val, y_train, y_val

def evaluate(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError('data shape error')
    return np.mean(y_pred == y_true)
 
def create_k_folds(X: np.ndarray, y: np.ndarray, k: int, seed: int=42)->List[Tuple[np.ndarray, np.ndarray]]:
    if X.shape[0] != y.shape[0]:
        raise ValueError('data shape error')
    
    n_sample = X.shape[0]
    rng = np.random.default_rng(seed=seed)
    shuffled_indices = rng.permutation(n_sample)

    X_shuffle = X[shuffled_indices]
    y_shuffle = y[shuffled_indices]
    X_folds = np.array_split(X_shuffle, k)
    y_folds = np.array_split(y_shuffle, k)

    folds = list(zip(X_folds, y_folds))
    return folds

def evaluate_with_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float,
    kernel: str = 'rbf',
    gamma: str = 'scale',
    degree: int = 3,
    coef0: float = 0.0,
    class_weight=None,
    k: int = 5,
    seed: int = 42
) -> float:
    """
    Evaluate SVM parameters using k-fold cross-validation.
    Returns the mean accuracy across folds.
    """
    from svm_model import train_svm_model
    from preprocess import fit_spam_scaler, apply_spam_scaler
    
    k_folds = create_k_folds(X_train, y_train, k=k, seed=seed)
    fold_accuracies = []
    
    for i in range(k):
        X_val_raw, y_val = k_folds[i]
        X_fold_train_raw = np.concatenate([k_folds[j][0] for j in range(k) if j != i])
        y_fold_train = np.concatenate([k_folds[j][1] for j in range(k) if j != i])
        
        # Scale features for this fold
        scaler = fit_spam_scaler(X_fold_train_raw)
        X_fold_train_scaled = apply_spam_scaler(X_fold_train_raw, scaler)
        X_val_scaled = apply_spam_scaler(X_val_raw, scaler)
        
        # Train model with current parameters
        model = train_svm_model(
            X_fold_train_scaled, y_fold_train,
            C=C, kernel=kernel, gamma=gamma
        )
        
        # Manually set additional parameters if needed
        if hasattr(model, 'degree'):
            model.degree = degree
        if hasattr(model, 'coef0'):
            model.coef0 = coef0
        if hasattr(model, 'class_weight'):
            model.class_weight = class_weight
            
        # Re-fit with all parameters
        if kernel == 'poly':
            from sklearn.svm import SVC
            model = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, class_weight=class_weight)
            model.fit(X_fold_train_scaled, y_fold_train)
        elif kernel == 'sigmoid':
            from sklearn.svm import SVC
            model = SVC(C=C, kernel=kernel, gamma=gamma, coef0=coef0, class_weight=class_weight)
            model.fit(X_fold_train_scaled, y_fold_train)
        elif class_weight is not None:
            from sklearn.svm import SVC
            model = SVC(C=C, kernel=kernel, gamma=gamma, class_weight=class_weight)
            model.fit(X_fold_train_scaled, y_fold_train)
        
        y_pred = model.predict(X_val_scaled)
        accuracy = evaluate(y_pred, y_val)
        fold_accuracies.append(accuracy)
    
    return np.mean(fold_accuracies)