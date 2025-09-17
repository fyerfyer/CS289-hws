from typing import Tuple
import numpy as np 
from sklearn.preprocessing import StandardScaler

def process_mnist_features(X: np.ndarray) -> np.ndarray:
    # reshape
    X = X.reshape(X.shape[0], -1)   # (60000, 784)

    # normalization
    X = X.astype('float64') / 255.0 # [0, 1]

    return X

def fit_spam_scaler(X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler

def apply_spam_scaler(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(X)