#Description
#Ridge and Lasso Regression are types of supervised learning algorithms that can be used for both classification and regression tasks. They work by adding a regularization term to the loss function, which helps prevent overfitting by penalizing large coefficients.Ridge ( adds L2 penalty: sum of squared coefficients) and Lasso regression ( adds L! Penalty: sum of absolute coefficients) are regularization techniques applied to linear Regression to prevent overfitting by penalizing large coefficients.

#import necessary libraries
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

#Sample data (e.g, house size vs. house price)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([2, 4, 6, 9, 10, 13, 14, 17, 18, 20])

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Ridge Regression
ridge_model = Ridge(alpha=1) # alpha is the regularization strength
ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_predictions)
print("Ridge Regression MSE: ", ridge_mse)

#Lasso Regression
lasso_model = Lasso(alpha=0.1) # alpha is the regularization strength
lasso_model.fit(X_train, y_train)
lasso_predictions = lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_predictions)
print("Lasso Regression MSE: ", lasso_mse)

