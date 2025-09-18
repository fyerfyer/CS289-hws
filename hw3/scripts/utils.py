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

def calculate_per_digit_error(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    errors = {}
    for digit in range(len(y_pred)):
        digit_mask = (y_true == digit)
        if np.sum(digit_mask) == 0:
            continue
        accuracy = np.mean(y_pred[digit_mask] == y_true[digit_mask])
        errors[digit] = 1.0 - accuracy
    return errors
