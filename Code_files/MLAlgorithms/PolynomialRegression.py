#import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # for adding polynomial features to the model 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

#Sample data(e.g experience vs salary)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([4, 8, 15, 16, 23, 42, 16, 32, 35, 59])

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Add polynomial features to the model
poly = PolynomialFeatures(degree=2, include_bias=False) # degree is the degree of the polynomial
X_train_poly = poly.fit_transform(X_train) # fit and transform the training data
X_test_poly = poly.transform(X_test) # transform the test data

#Initialize and train the model
model = LinearRegression()
model.fit(X_train_poly, y_train)

#Make predictions
y_pred = model.predict(X_test_poly)

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)
print("Predicted values: ", y_pred)
print("Actual values: ", y_test)





