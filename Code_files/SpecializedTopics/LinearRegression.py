#import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

#Sample data 9e.g, house size vs. house price)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
# output should be random multiple of input
y = np.array([2, 4, 6, 9, 10, 13, 14, 17, 18, 20]) 

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error: ", mse)
print("Predicted values: ", y_pred)
print("Actual values: ", y_test)

