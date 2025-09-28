#importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

#Sample data with 10 car models
data ={
    'make': ['Toyota','Honda','Ford','Chevrolet','BMW','Audi','Volkswagen','Jaguar','Mercedes','Lexus'],
    'model': ['Camry','Civic','Mustang','Camaro','M3','RS5','Golf','XE','S-Class','IS'],
    'year': [2022,2021,2020,2019,2018,2022,2021,2020,2019,2018],
    'mileage': [50000,60000,70000,80000,90000,50000,60000,70000,80000,90000],
    'price': [25000,22000,28000,26000,35000,50000,60000,70000,80000,90000]
}

#Creating a DataFrame
df = pd.DataFrame(data)

#Converting categorical variables to numeric using one-hot encoding.
df_encoded = pd.get_dummies(df, columns=['make','model'])
# print(df_encoded.head())

#Defining the features and target variable
X = df_encoded.drop('price',axis=1)
y = df_encoded['price']

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#initializing the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100,random_state=42)

#Training the model
model.fit(X_train,y_train)


#Making predictions on the test set
y_pred = model.predict(X_test)

#Evaluating the model
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

