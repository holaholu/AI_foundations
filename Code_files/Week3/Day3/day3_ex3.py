import numpy as np

# Define the gradient descent function
def gradient_descent(X, y, theta, learning_rate, iterations): # Gradient Descent Function
    m = len(y) # Number of samples
    for _ in range(iterations):# Number of iterations
        predictions = np.dot(X, theta) # Predictions
        errors = predictions - y # Errors
        gradients = (1/m) * np.dot(X.T, errors) # Gradients. This is the gradient of the cost function with respect to the parameters. X.T is the transpose of X
        theta -= learning_rate * gradients # Update parameters
    return theta

# Sample Data
X = np.array([[1, 1], [1, 2], [1, 3]]) # Feature matrix
y = np.array([2, 2.5, 3.5]) # Target vector
theta = np.array([0.1, 0.1]) # Initial parameters   
learning_rate = 0.1 # Learning rate
iterations = 1000 # Number of iterations

# Perform gradient descent
optimized_theta = gradient_descent(X, y, theta, learning_rate, iterations) # Perform gradient descent

print("Optimized Parameters: ", optimized_theta)