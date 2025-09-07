#Description
#Naive Bayes is a probabilistic supervised learning classification algorithm that, despite its "naive" assumption that all features are independent of each other, works well for real-world applications like spam filtering and document categorization. It functions by using Bayes' Theorem to calculate the posterior probability of a data point belonging to each class and then assigns it to the class with the highest probability.

#import necessary libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import numpy as np

#Sample data (e.g, hours studied and grades vs. pass/fail)  
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]) # 0 = fail, 1 = pass

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and train the model
model = GaussianNB()
model.fit(X_train, y_train)

#Make predictions
y_pred = model.predict(X_test)

#Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy: ", accuracy)
print("Confusion Matrix: ", conf_matrix)
