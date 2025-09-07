#Description
#KNN is a type of supervised learning algorithm that can be used for both classification and regression tasks. It works by finding the k nearest neighbors of a data point in the feature space and using their labels to make a prediction. Useful for smaller datasets

#import necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np

#Sample data (e.g, hours studied and prior grades vs. pass/fail)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) # 0 = fail, 1 = pass

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and train the model
model = KNeighborsClassifier(n_neighbors=3) # n_neighbors is the number of nearest neighbors to consider
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy)
print("Confusion Matrix: ", conf_matrix)


