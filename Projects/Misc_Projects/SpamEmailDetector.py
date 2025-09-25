import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score,recall_score,f1_score
import seaborn as sns
import matplotlib.pyplot as plt

#Load the dataset ans split it into training and testing sets
data = pd.read_csv('.data/spam.csv')
X= data.drop('spam',axis=1)
y=data['spam']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Train the model
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print(X_test)

#Evaluate the model
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Precision:",precision_score(y_test,y_pred))
print("Recall:",recall_score(y_test,y_pred))
print("F1 Score:",f1_score(y_test,y_pred))

#Visualize the confusion matrix using seaborn heatmap
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
