import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

# Generate Synthetic Data
np.random.seed(42)
X = np.random.rand(100, 1) * 10 #random.rand() generates random numbers from a uniform distribution
y = 3 * X**2 + 2 * X + np.random.randn(100, 1) * 5 #randn() generates random numbers from a normal distribution

# Transform features to polynomial
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

#Ridge Regression
ridge_model = Ridge(alpha=1) #alpha is the regularization strength.This controls the amount of shrinkage: the higher the value of alpha, the greater the shrinkage and thus the coefficients become more robust to collinearity/ overfitting.
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)

# Lasso Regression
lasso_model = Lasso(alpha=1) #alpha is the regularization strength
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)

# Evaluate Ridge
ridge_mse = mean_squared_error(y_test, ridge_predictions) # a good model has a low mse. low range values include 0-10.
print("Ridge Regression MSE:", ridge_mse)

# Evaluate Lasso
lasso_mse = mean_squared_error(y_test, lasso_predictions)
print("Lasso Regression MSE:", lasso_mse)






