import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df=pd.read_csv("./data/har_data.csv")

#Preprocess the dataset
X=df.drop("Activity",axis=1)
y=df["Activity"]

#Split the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Train the model
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

#Make predictions
y_pred=model.predict(X_test)

#Evaluate the model
accuracy=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred,average="macro")
recall=recall_score(y_test,y_pred,average="macro")
f1=f1_score(y_test,y_pred,average="macro")

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"Recall: {recall*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")


#Visualize the confusion matrix
conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()





