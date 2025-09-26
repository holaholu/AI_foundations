#Import the libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
import matplotlib.pyplot as plt

#Load the iris dataset
iris = load_iris()

#create a dataframe
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df["species"] = iris.target #Adds the species column to the dataframe

#display the first 5 rows of the dataframe
print(df.head())

#Split the dataset into training and testing sets
X = df.drop("species", axis=1)
y = df["species"] #Target (Species: 0 = setosa, 1 = versicolor, 2 = virginica)

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train the model
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

#Make predictions
y_pred = classifier.predict(X_test)

#Evaluate the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:", metrics.classification_report(y_test, y_pred))

#Plot the decision tree
plt.figure(figsize=(12,8))
plot_tree(classifier, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree for Iris Flower Classification")
plt.show()


