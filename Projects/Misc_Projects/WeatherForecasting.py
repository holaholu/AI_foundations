import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = {
    'Day': [1, 2, 3, 4, 5, 6, 7,8,9,10],
    'Temperature': [22, 24, 19, 23, 25, 21, 20, 22, 24, 26],
    'Humidity': [80, 82, 78, 85, 90, 87, 84, 82, 80, 78],
    'WindSpeed': [10, 12, 15, 14, 16, 13, 11, 10, 12, 14],
    'Precipitation': [0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'Next Day Temperature': [24, 26, 21, 24, 26, 22, 21, 23, 25, 27],
}

# Convert the dictionary to a DataFrame
df = pd.DataFrame(data)

# Split the data into features (X) and target (y)
X = df[['Temperature', 'Humidity', 'WindSpeed', 'Precipitation']]
y = df['Next Day Temperature']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}") #Closer to 1 is better


new_data = pd.DataFrame({
    'Temperature': [30],
    'Humidity': [60],
    'WindSpeed': [10],
    'Precipitation': [0],
})

prediction = model.predict(new_data)
print(f"\n\nPredicted Temperature: {prediction[0]:.2f}C")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, color='blue', marker='o', label='Actual Temperatures')
plt.plot(y_pred, color='red', marker='x', label='Predicted Temperatures')
plt.title('Actual vs Predicted Temperatures')
plt.xlabel('Test Sample Index')
plt.ylabel('Temperature')
plt.legend()
plt.show()


    

