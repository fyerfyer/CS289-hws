import numpy as np 
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal


class BaseClassifier(ABC):
    def __init__(self, params: dict):
        self.params = params

    def predict(self, X: np.ndarray) -> np.ndarray:
        score = self._get_class_scores(X)
        prediction = np.argmax(score, axis=1)
        return prediction

    @abstractmethod
    def _get_class_scores(self, X: np.ndarray) -> np.ndarray:
        pass

class LDAClassifier(BaseClassifier):
    def __init__(self, params: dict, N_train: int, y_train: np.ndarray):
        super().__init__(params)
        K = len(params.keys())
        N = params[y_train[0]]['mean'].shape[0]
        pooled_cov_numerator = np.zeros((N, N))
        
        for c, c_params in self.params.items():
            N_c = np.sum(y_train == c)
            c_cov = c_params['covariance']
            pooled_cov_numerator += (N_c - 1) * c_cov
        
        denominator = N_train - K 
        self.pooled_cov = pooled_cov_numerator / denominator 

    def _get_class_scores(self, X: np.ndarray) -> np.ndarray:
        classes = self.params.keys()
        N = X.shape[0]
        scores = np.zeros((N, len(classes)))

        for c in classes:
            prior = self.params[c]['prior']
            c_mean = self.params[c]['mean']
            log_prior = np.log(prior)
            log_likeliness = multivariate_normal.logpdf(X, mean=c_mean, cov=self.pooled_cov)
            scores[:, c] = log_prior + log_likeliness
        
        return scores
            

class QDAClassifier(BaseClassifier):
    def __init__(self, params: dict):
        super().__init__(params)
    
    def _get_class_scores(self, X: np.ndarray) -> np.ndarray:
        classes = self.params.keys()
        N = X.shape[0]
        scores = np.zeros((N, len(classes)))

        for c in classes:
            prior = self.params[c]['prior']
            c_mean = self.params[c]['mean']
            c_cov = self.params[c]['covariance']
            log_prior = np.log(prior)
            log_likelihood = multivariate_normal.logpdf(X, mean=c_mean, cov=c_cov)
            scores[:, c] = log_prior + log_likelihood

        return scores
            
