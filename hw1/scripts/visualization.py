import os
from typing import Dict
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import seaborn as sns

def plot_svm_2d_boundary(X: np.ndarray, y: np.ndarray, model: SVC, title: str):
    w = model.coef_[0]
    b = model.intercept_[0]
    support_vector = model.support_vectors_
    support_idx = model.support_

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = np.where(y == 1, 'C0', 'C1')
    ax.scatter(X[:, 0], X[:, 1], c=colors, s=30, label='data')

    # decision boundary and margin
    xmin, xmax = (X[:, 0].min() - 1.0, X[:, 0].max() + 1.0)
    ymin, ymax = (X[:, 1].min() - 1.0, X[:, 1].max() + 1.0)

    # draw boundary
    t = np.linspace(xmin, xmax, 400)
    if abs(w[1]) > 1e-8:
        y_decision = -(w[0] * t + b) / w[1]
        y_plus = -(w[0] * t + b - 1.0) / w[1]
        y_minus = -(w[0] * t + b + 1.0) / w[1]
        ax.plot(t, y_decision, 'k-', label='decision boundary')
        ax.plot(t, y_plus, 'k--', label='margin +1')
        ax.plot(t, y_minus, 'k--', label='margin -1')
    else:
        t_y = np.linspace(ymin, ymax, 400)
        x_decision = -(w[1] * t_y + b) / w[0]
        x_plus = -(w[1] * t_y + b - 1.0) / w[0]
        x_minus = -(w[1] * t_y + b + 1.0) / w[0]
        ax.plot(x_decision, t_y, 'k-', label='decision boundary')
        ax.plot(x_plus, t_y, 'k--', label='margin +1')
        ax.plot(x_minus, t_y, 'k--', label='margin -1')

    # mark support vectors
    ax.scatter(support_vector[:, 0], support_vector[:, 1], 
               facecolors='none', edgecolors='k', s=140, linewidths=1.5,
               label='support vectors')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.legend(loc='best')
    plt.show()

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: list, title: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_performance(performance_history: Dict[float, float], title: str):
    C_values = list(performance_history.keys())
    accuracies = list(performance_history.values())

    plt.figure(figsize=(10, 6))
    plt.plot(C_values, accuracies, marker='o', linestyle='-')

    plt.xscale('log')

    plt.title(title)
    plt.xlabel('C value (log scale)')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.savefig(f'{title.replace(" ", "_").lower()}_performance.png')
    print(f"Performance plot saved to {title.replace(' ', '_').lower()}_performance.png")
    plt.show()

