#import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

#load the dataset
df = pd.read_csv('./data/heart_disease.csv')
#print(df.head())

#check for missing values
#print("Missing values:", df.isnull().sum()) #count the number of missing values in each column

#features scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.drop('target', axis=1))
X = pd.DataFrame(scaled_features, columns=df.columns[:-1])
y = df['target']

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Train multiple models

#1.Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_pred = log_model.predict(X_test)
log_accuracy = accuracy_score(y_test, log_pred)
print(f"Logistic Regression Accuracy: {log_accuracy:.2f}")

#2.Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")

#Evaluate best Model
best_model = rf_model if rf_accuracy > log_accuracy else log_model
best_pred = rf_pred if best_model == rf_model else log_pred

print("\nBest Model Metrics:")
print("Accuracy Score", accuracy_score(y_test, best_pred))
print("Classification Report\n", classification_report(y_test, best_pred))
print("Confusion Matrix\n", confusion_matrix(y_test, best_pred))

#Visualize the confusion matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, best_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Make predictions on new data
new_data = pd.DataFrame({
    'age': [45],
    'sex': [1],
    'cp': [2],
    'trestbps': [130],
    'chol': [230],
    'fbs': [0],
    'restecg': [1],
    'thalach': [150],
    'exang': [0],
    'oldpeak': [0.5],
    'slope': [2],
    'ca': [0],
    'thal': [2]
})

#Scale the new data
new_data_scaled = scaler.transform(new_data)

#Make predictions
prediction = best_model.predict(new_data_scaled)
print("\nPredicion for New Data:","At risk of Heart Disease" if prediction[0] == 1 else "Not at risk of Heart Disease")









