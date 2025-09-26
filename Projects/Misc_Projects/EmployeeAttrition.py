#import necessary libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#load the dataset
df = pd.read_csv('./data/EmployeeAttrition.csv')
#print(df.head())

#Preprocess the data
#Drop irrelavant columns
df.drop(['EmployeeCount', 'Over18', 'StandardHours'], axis=1, inplace=True)

#Encode categorical variables
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

#Split the data into features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss') #use_label_encoder=False to avoid duplicated label encoding
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report: \n', class_report)
print('Confusion Matrix: \n', conf_matrix)

#Visualize the feature importances
plt.figure(figsize=(10,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')    
plt.show()

