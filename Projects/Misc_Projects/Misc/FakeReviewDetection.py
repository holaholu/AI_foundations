#import the required libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#load the dataset
#Assuming the dataset is in a CSV file with columns 'text' and 'label'
df = pd.read_csv("./data/fakereviews.csv")

#display the first few rows of the dataset
print(df.head())

#define feature and target
X = df["text"]
y = df["label"]

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) #stratify=y ensures that the training and testing sets have the same distribution of labels

#initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(max_df=0.7, stop_words="english") #max_df=0.7 ensures that words that appear in more than 70% of the documents are removed

#fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

#transform the testing data
X_test_tfidf = vectorizer.transform(X_test)

#initialize the LogisticRegression model
model = LogisticRegression(max_iter=1000)

#train the model
model.fit(X_train_tfidf, y_train)

#make predictions on the testing set
y_pred = model.predict(X_test_tfidf)

#evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:", classification_report(y_test, y_pred))





