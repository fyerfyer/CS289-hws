from typing import Tuple
import numpy as np 

def load_mnist_data(path: str='../data/mnist-data.npz') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = np.load(path)
    
    return (dataset['training_data'], dataset['training_labels'], dataset['test_data'])

def load_spam_data(path: str='../data/spam-data.npz') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = np.load(path)
    return (dataset['training_data'], dataset['training_labels'], dataset['test_data'])