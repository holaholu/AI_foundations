#import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#Load the dataset
df = pd.read_csv('./data/CustomerChurn.csv')
# print(df.head())

#Basic data pre-processing: handling missing values
df = df.dropna()

#Convert categorical columns into numeric using one-hot encoding if needed
# df_encoded = pd.get_dummies(df, columns = ['Area', 'ContractRenewal', 'DataPlan', 'PhoneService', 'MultipleLines'])

#Define Features and Target
X = df.drop('Churn', axis = 1)
y = df['Churn']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

#Feature Scaling
scaler = StandardScaler() # Scales to have mean = 0 and std = 1
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Initialize the model
model = LogisticRegression()

#Train the model
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred, zero_division = 1))




