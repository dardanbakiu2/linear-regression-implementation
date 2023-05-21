import numpy as np
import matplotlib.pyplot as plt

def feature_normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    square_err = np.power((predictions - y), 2)
    cost = 1 / (2 * m) * np.sum(square_err)
    return cost

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)

    for iter in range(num_iters):
        predictions = X.dot(theta)
        errors = np.dot(X.transpose(), (predictions - y))
        theta = theta - (alpha / m) * errors
        J_history[iter] = compute_cost(X, y, theta)

    return theta, J_history

# Load the dataset
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :-1]
y = data[:, -1]
m = len(y)

# Feature normalization
X, mean, std = feature_normalize(X)

# Add intercept term to X
X = np.column_stack((np.ones(m), X))

# Initialize theta parameters
theta = np.zeros(X.shape[1])

# Set hyperparameters
alpha = 0.01
num_iters = 400

# Compute and display initial cost
initial_cost = compute_cost(X, y, theta)
print("Initial cost:", initial_cost)

# Run gradient descent to get optimized theta values
theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
print("Theta found by gradient descent:", theta)

# Plot the cost history
plt.plot(range(num_iters), J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()
