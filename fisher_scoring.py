
import numpy as np

def fisher_scoring_poisson(X, y, max_iter=100, epsilon=1e-6):
    n, p = X.shape
    theta = np.zeros(p)
    mu = np.exp(X @ theta)
    W = np.diag(mu)
    z = X @ theta + np.linalg.inv(W) @ (y - mu)
    
    converged = False
    while not converged:
        theta_new = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ z)
        mu = np.exp(X @ theta_new)
        W = np.diag(mu)
        z = X @ theta_new + np.linalg.inv(W) @ (y - mu)

        if np.linalg.norm(theta_new - theta) / (np.linalg.norm(theta) + epsilon) < epsilon:
            converged = True
        theta = theta_new
    return theta, converged

# Example usage with simulated data
np.random.seed(45)
n, p = 100, 2
X = np.random.normal(size=(n, p))
beta = np.array([0.2, -0.4])
mu = np.exp(X @ beta)
y = np.random.poisson(mu)

theta_est, converged = fisher_scoring_poisson(X, y)
print("Estimated coefficients:", theta_est)
print("Convergence status:", converged)
