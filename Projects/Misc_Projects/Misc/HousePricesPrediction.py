import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.datasets import fetch_california_housing

#Load the dataset
housing = fetch_california_housing(as_frame=True)

#Create a dataframe from dataset
df=housing.frame
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model
model = LinearRegression()
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error: {mse}")
print(f"R2 score: {r2}") #r2 score is a measure of how well the model fits the data.R2 score ranges from 0 to 1, with higher values indicating a better fit.

print ("Model Coefficients:")   
print(model.coef_)
print ("Model Intercept:")
print(model.intercept_)

coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient']) #Create a dataframe from the model coefficients
print("Coefficients for each feature:")
print(coef_df)


#Test the model

new_data = pd.DataFrame({
    'MedInc': [5],
    'HouseAge': [30],
    'AveRooms': [6],
    'AveBedrms': [1],
    'Population': [500],
    'AveOccup': [3],
    'Latitude': [34.05],
    'Longitude': [-118.25]
})

predicted_price = model.predict(new_data)
print(f"\n\n Predicted price: {predicted_price[0]:.2f}")

