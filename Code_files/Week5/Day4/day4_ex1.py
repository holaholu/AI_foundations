import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Generate synthetioc dataset
np.random.seed(42)
n_samples = 200
X = np.random.rand(n_samples, 2) * 10 #random.rand() generates random numbers from a uniform distribution. * 10 scales the values between 0 and 10
y = (X[:, 0] * 1.5 + X[:, 1] > 15).astype(int) # creates a binary target variable based on a linear combination of the features

# Create a Dataframe
df = pd.DataFrame(X, columns=['Age', 'Salary'])
df['Purchase'] = y

# Split Data
X_train, X_test, y_train, y_test = train_test_split(df[['Age', 'Salary']], df['Purchase'], test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() -1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1)) # creates a grid of points

# Predict probabilities for grid points
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) # ravel() flattens the array
Z = Z.reshape(xx.shape) # reshapes the array to match the grid shape

# Plot
plt.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm") # creates a filled contour plot
plt.scatter(X_test['Age'], X_test['Salary'], c=y_test, edgecolor="k", cmap="coolwarm") # plots the test data
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()