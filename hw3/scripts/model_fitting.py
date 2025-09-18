import numpy as np 

def fit_gaussian_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    model_params = {}
    N, num_features = X_train.shape
    classes = np.unique(y_train)

    for c in classes:
        # calculate prior probability
        X_c = X_train[y_train == c]
        N_c = X_c.shape[0]
        prior = N_c / N 

        # calculate mean value 
        c_mean = np.mean(X_c, axis=0)

        # calculate cov matrix 
        c_cov = np.cov(X_c, rowvar=False)
        c_cov += np.eye(num_features) * 1e-6

        model_params[c] = {
            'prior': prior,
            'mean': c_mean,
            'covariance': c_cov
        }

    return model_params
