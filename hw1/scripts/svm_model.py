import numpy as np 
from sklearn.svm import SVC

def train_svm_model(
    X: np.ndarray, 
    y: np.ndarray,
    C: float = 1.0,
    kernel: str = 'rbf',
    gamma: str = 'scale'
) -> SVC:
    svc_model = SVC(C=C, kernel=kernel, gamma=gamma)
    svc_model.fit(X, y)
    return svc_model
