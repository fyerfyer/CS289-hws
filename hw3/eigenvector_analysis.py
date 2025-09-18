"""
Eigenvectors of the Gaussian Covariance Matrix

This program analyzes the eigenvectors and eigenvalues of a sample covariance matrix
from bivariate Gaussian data where X2 depends on X1.

Random seed: 42 (using numpy.random)
Random number generator: numpy.random (default Generator)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Set random seed for reproducibility
np.random.seed(42)
print("Random seed used: 42")
print("Random number generator: numpy.random")
print()

# Parameters
n = 100
mu1 = 3      # Mean of X1
sigma1_sq = 9  # Variance of X1
sigma1 = np.sqrt(sigma1_sq)  # Standard deviation of X1

mu2_offset = 4   # Offset for X2
sigma2_sq = 4    # Variance of noise in X2
sigma2 = np.sqrt(sigma2_sq)  # Standard deviation of noise in X2

print("=== Problem Setup ===")
print(f"X1 ~ N({mu1}, {sigma1_sq}) (mean={mu1}, variance={sigma1_sq})")
print(f"X2 = 0.5 * X1 + N({mu2_offset}, {sigma2_sq})")
print(f"Sample size: n = {n}")
print()

# Generate sample data
# First generate X1 from N(3, 9)
X1 = np.random.normal(mu1, sigma1, n)

# Then generate X2 = 0.5 * X1 + N(4, 4)
noise = np.random.normal(mu2_offset, sigma2, n)
X2 = 0.5 * X1 + noise

# Combine into sample matrix
X = np.column_stack([X1, X2])

print("=== Part 1: Sample Mean ===")
sample_mean = np.mean(X, axis=0)
print(f"Sample mean: [{sample_mean[0]:.6f}, {sample_mean[1]:.6f}]")
print()

print("=== Part 2: Sample Covariance Matrix ===")
# Compute sample covariance matrix using ML estimator (divide by n)
centered_X = X - sample_mean
sample_cov = (centered_X.T @ centered_X) / n
print("Sample covariance matrix (ML estimator):")
print(f"[[{sample_cov[0,0]:.6f}, {sample_cov[0,1]:.6f}],")
print(f" [{sample_cov[1,0]:.6f}, {sample_cov[1,1]:.6f}]]")
print()

print("=== Part 3: Eigenvalues and Eigenvectors ===")
# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = eigh(sample_cov)

# Sort in descending order of eigenvalues
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("Eigenvalues (in descending order):")
print(f"λ1 = {eigenvalues[0]:.6f}")
print(f"λ2 = {eigenvalues[1]:.6f}")
print()

print("Eigenvectors (columns):")
print("v1 (first eigenvector):")
print(f"[{eigenvectors[0,0]:.6f}, {eigenvectors[1,0]:.6f}]")
print("v2 (second eigenvector):")
print(f"[{eigenvectors[0,1]:.6f}, {eigenvectors[1,1]:.6f}]")
print()

# Verify eigenvectors are unit vectors
print("Verification - Eigenvector magnitudes:")
print(f"|v1| = {np.linalg.norm(eigenvectors[:,0]):.6f}")
print(f"|v2| = {np.linalg.norm(eigenvectors[:,1]):.6f}")
print()

print("=== Part 4: Original Data Plot ===")
# Create the first plot
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 8))

# Plot data points
ax1.scatter(X[:, 0], X[:, 1], alpha=0.6, s=30, color='blue', label='Data points')

# Plot mean
ax1.plot(sample_mean[0], sample_mean[1], 'ro', markersize=8, label='Sample mean')

# Plot eigenvector arrows (magnitude = eigenvalue)
arrow1_end = sample_mean + eigenvalues[0] * eigenvectors[:, 0]
arrow2_end = sample_mean + eigenvalues[1] * eigenvectors[:, 1]

ax1.arrow(sample_mean[0], sample_mean[1], 
          eigenvalues[0] * eigenvectors[0, 0], eigenvalues[0] * eigenvectors[1, 0],
          head_width=0.5, head_length=0.5, fc='red', ec='red', linewidth=2,
          label=f'1st eigenvector (λ={eigenvalues[0]:.2f})')

ax1.arrow(sample_mean[0], sample_mean[1], 
          eigenvalues[1] * eigenvectors[0, 1], eigenvalues[1] * eigenvectors[1, 1],
          head_width=0.5, head_length=0.5, fc='green', ec='green', linewidth=2,
          label=f'2nd eigenvector (λ={eigenvalues[1]:.2f})')

ax1.set_xlim(-15, 15)
ax1.set_ylim(-15, 15)
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_title('Original Data with Eigenvectors')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_aspect('equal')  # Ensure square aspect ratio

plt.tight_layout()
plt.savefig('./original_data_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("=== Part 5: Rotated Data Plot ===")
# Create rotation matrix U^T where U = [v1 v2]
U = eigenvectors  # U has eigenvectors as columns
U_T = U.T  # Rotation matrix

# Center the data and rotate
centered_data = X - sample_mean
rotated_data = (U_T @ centered_data.T).T

print("Rotation matrix U^T:")
print(f"[[{U_T[0,0]:.6f}, {U_T[0,1]:.6f}],")
print(f" [{U_T[1,0]:.6f}, {U_T[1,1]:.6f}]]")
print()

# Verify rotation: compute covariance of rotated data
rotated_cov = (rotated_data.T @ rotated_data) / n
print("Covariance matrix of rotated data (should be diagonal):")
print(f"[[{rotated_cov[0,0]:.6f}, {rotated_cov[0,1]:.6f}],")
print(f" [{rotated_cov[1,0]:.6f}, {rotated_cov[1,1]:.6f}]]")
print()

# Create the second plot
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))

# Plot rotated data points
ax2.scatter(rotated_data[:, 0], rotated_data[:, 1], alpha=0.6, s=30, color='blue', label='Rotated data points')

# Plot origin (rotated mean)
ax2.plot(0, 0, 'ro', markersize=8, label='Mean (origin)')

ax2.set_xlim(-15, 15)
ax2.set_ylim(-15, 15)
ax2.set_xlabel('Rotated X1 (along 1st eigenvector)')
ax2.set_ylabel('Rotated X2 (along 2nd eigenvector)')
ax2.set_title('Data Rotated to Eigenvector Coordinate System')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_aspect('equal')  # Ensure square aspect ratio

plt.tight_layout()
plt.savefig('./rotated_data_plot.png', dpi=150, bbox_inches='tight')
plt.show()

print("=== Summary ===")
print(f"Sample mean: [{sample_mean[0]:.6f}, {sample_mean[1]:.6f}]")
print(f"Eigenvalues: λ1 = {eigenvalues[0]:.6f}, λ2 = {eigenvalues[1]:.6f}")
print(f"First eigenvector: [{eigenvectors[0,0]:.6f}, {eigenvectors[1,0]:.6f}]")
print(f"Second eigenvector: [{eigenvectors[0,1]:.6f}, {eigenvectors[1,1]:.6f}]")
print("Plots saved as 'original_data_plot.png' and 'rotated_data_plot.png'")