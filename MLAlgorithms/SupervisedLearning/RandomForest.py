#Description
#Random Forest is a type of supervised learning algorithm that can be used for both classification and regression tasks. It works by building an ensemble of decision trees and using their predictions to make a final prediction. It is a powerful algorithm that can handle both numerical and categorical features, and it is less sensitive to overfitting than other algorithms.Each tree is trained with a random subset of data. Final prediction is averaged (for regression) or voted (for classification) from the predictions of individual trees. Reduces overfitting and improves generalization.

#import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np

#Sample data (e.g, hours studied and grades vs. pass/fail)  
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) # 0 = fail, 1 = pass

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy)
print("Confusion Matrix: ", conf_matrix)