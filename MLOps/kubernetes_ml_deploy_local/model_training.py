import joblib 
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
housing = fetch_california_housing()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) #fit_transform not used on test data to prevent data leakage

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Save the model
joblib.dump(model, 'model.pkl') #model.pkl is a binary file that contain a serialized version of the model

#A .pkl file is a file saved in the Python pickle format. It contains serialized Python objects, meaning that Python objects (like lists, dictionaries, custom class instances, or even entire machine learning models) have been converted into a byte stream. This byte stream can then be stored on disk or transmitted, and later, deserialized back into their original Python object form. 

print("Model saved to 'model.pkl'")
