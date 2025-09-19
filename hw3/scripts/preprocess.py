import numpy as np 

def process_mnist_features(X: np.ndarray) -> np.ndarray:
    # reshape
    X = X.reshape(X.shape[0], -1)   # (60000, 784)

    # normalization
    X = X.astype('float64') / 255.0 # [0, 1]

    return X

def process_spam_features(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms += 1e-8
    X = X / norms
    return X 