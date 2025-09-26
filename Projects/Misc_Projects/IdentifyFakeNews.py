#importing the required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report

#loading the dataset
df = pd.read_csv('./data/fakenews.csv')
print(df.head())

#defining the features and the target variable
X = df['title']
y = df['real']

#splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify=y)

#vectorizing the data
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7) #converts the text data into numerical data
X_train = vectorizer.fit_transform(X_train) #fits and transforms the training data
X_test = vectorizer.transform(X_test) #Only transforms the testing data

#training the model
model = MultinomialNB() #naive bayes classifier
model.fit(X_train,y_train) #fits the model  

#making predictions
y_pred = model.predict(X_test)

#evaluating the model
accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test,y_pred, zero_division=1)) #zero_division=1 to avoid division by zero



