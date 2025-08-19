import numpy as np
import matplotlib.pyplot as plt

# Generate data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3X + noise

# Visualize data
# plt.scatter(X, y, color="blue")
# plt.title('Generated Dataset')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.grid()
# plt.show()

# Initialize parameters
m = 100 # number of samples
theta = np.random.rand(2, 1) # random initial parameters. this corresponds to the weights and bias.rand(2,1) is used to create a 2x1 array of random numbers
learning_rate = 0.1 # learning rate
iterations = 1000 # number of iterations

# Add bias term to X
X_b = np.c_[np.ones((m, 1)), X] # np.c_ is used to concatenate the bias term to the input features. np.ones((m, 1)) creates a column of ones. this is used to add the intercept term to the model.

# Gradient Descent
for iteration in range(iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y) # calculate the gradient of the loss function with respect to the parameters    
    theta -= learning_rate * gradients # update the parameters

print("Optimized Parameters (Theta): \n ", theta)