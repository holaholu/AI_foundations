#importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

#Sample historical stock price data
data = {
    'Date': pd.date_range(start='2025-01-01', periods=10, freq='D'),
    'Close': [100,101,102,103,104,105,106,107,108,109]   
}

#Convert the data to a DataFrame
df = pd.DataFrame(data)

df['Date'] = df['Date'].map(pd.Timestamp.toordinal) #Convert the date to ordinal

# print("Stock Price Data:")
# print(df.head())

X = df[['Date']]
y = df['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


future_date = pd.Timestamp('2025-01-11').toordinal()
future_date = pd.DataFrame({'Date': [future_date]})
predicted_price = model.predict(future_date)
print("Predicted Stock Price for 2025-01-11:", predicted_price[0])


plt.figure(figsize=(10,6))
plt.plot(X_test, y_test, color='blue', marker='o', label="Actual Stock Price")
plt.plot(X_test, y_pred, color='red', marker='x', label="Predicted Stock Price")
plt.xlabel('Date')
plt.ylabel('Stock Price($USD)')
plt.title('Actual vs Predicted Stock Price')
plt.legend()
plt.show()




