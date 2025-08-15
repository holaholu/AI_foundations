import numpy as np
import matplotlib.pyplot as plt

def generate_random_data(n_samples=100, noise=0.5):
    """Generate random linear data with some noise"""
    np.random.seed(42)  # For reproducibility
    X = 2 * np.random.rand(n_samples, 1)  # Random features between 0 and 2
    y = 4 + 3 * X + np.random.randn(n_samples, 1) * noise  # y = 4 + 3x + noise
    return X, y

def compute_cost(X, y, theta):
    """Compute the mean squared error cost"""
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1000, print_cost=False):
    """Perform gradient descent to minimize the cost function"""
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        # Calculate predictions and errors
        predictions = X.dot(theta)
        errors = predictions - y
        
        # Update parameters
        gradients = (1/m) * X.T.dot(errors)
        theta = theta - learning_rate * gradients
        
        # Store cost for plotting
        cost_history[i] = compute_cost(X, y, theta)
        
        # Print cost every 100 iterations
        if print_cost and i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost_history[i]:.4f}")
    
    return theta, cost_history

# Generate random data
X, y = generate_random_data(100, 0.5)

# Add bias term (x0 = 1) to X
X_b = np.c_[np.ones((len(X), 1)), X]

# Initialize parameters
theta = np.random.randn(2, 1)

# Hyperparameters
learning_rate = 0.1
iterations = 1000

# Run gradient descent
theta_optimized, cost_history = gradient_descent(
    X_b, y, theta, learning_rate, iterations, print_cost=True
)

# Print results
print(f"\nOptimized parameters:")
print(f"Theta0 (bias): {theta_optimized[0][0]:.4f}")
print(f"Theta1 (weight): {theta_optimized[1][0]:.4f}")

# Plot the results
plt.figure(figsize=(12, 4))

# Plot 1: Data and best fit line
plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.5)
plt.plot(X, X_b.dot(theta_optimized), 'r-', label='Best fit')
plt.title('Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Plot 2: Cost function over iterations
plt.subplot(1, 2, 2)
plt.plot(cost_history)
plt.title('Cost Function over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')

plt.tight_layout()
plt.show()