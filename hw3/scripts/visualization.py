import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict

def visualize_covariance(cov_matrix: np.ndarray, class_name: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_matrix, cmap='viridis', square=True, xticklabels=False, yticklabels=False)
    plt.title(f'Covariance Matrix for Digit "{class_name}"', fontsize=16)
    plt.xlabel('Pixel Index')
    plt.ylabel('Pixel Index')
    plt.show()
