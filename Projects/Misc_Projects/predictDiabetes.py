#importing the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Pima Indians Diabetes dataset
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
column_names=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
df=pd.read_csv(url,names=column_names)

#Display the first 5 rows of the dataset
# print("diabetes dataset")
# print(df.head())

X=df.drop("Outcome",axis=1)
y=df["Outcome"]

#Split the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Train the model
model=LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

#Make predictions
y_pred=model.predict(X_test)

#Evaluate the model
print("Accuracy:",accuracy_score(y_test,y_pred))
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))


#Predict the outcome for a new patient
new_data = pd.DataFrame({
    "Pregnancies": [5],
    "Glucose": [120],
    "BloodPressure": [72],
    "SkinThickness": [35],
    "Insulin": [80],
    "BMI": [32.0],
    "DiabetesPedigreeFunction": [0.5],
    "Age": [42]
})

predicted_outcome=model.predict(new_data)
print(f"Predicted Output:{'Diabetic' if predicted_outcome[0] == 1 else 'Not Diabetic'}")

#Visualize the results
sns.heatmap(confusion_matrix(y_test,y_pred),annot=True,fmt="d",cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

