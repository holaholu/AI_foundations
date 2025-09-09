#Description
#Self training is a semi-supervised learning technique where a model is trained on a small amount of labeled data and then used to generate predictions for a larger amount of unlabeled data. The predictions are then used to train the model on the unlabeled data. This process is repeated (using confidence threshold/predictions - those with high confidence are used to train the model) until the model converges or a maximum number of iterations is reached. Self training is useful when there is a large amount of unlabeled data and a small amount of labeled data.

#import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#Generate sample data
X, y = make_classification(n_samples=200, n_features=5, random_state=42)

#Split data into training and testing sets
X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=0.7, random_state=42)

#Initialize and train the model
model = RandomForestClassifier()
model.fit(X_labeled, y_labeled)

#Perform self training on unlabeled data
for _ in range(5): # repeat 5 times for iterative training
    #Predict probabilities for unlabeled data
    probs = model.predict_proba(X_unlabeled) #probs is the predicted probabilities for the unlabeled data
    #Select samples with highest confidence
    high_confidence_idx = np.where(np.max(probs, axis=1) > 0.9)[0] #high_confidence_idx is the array of samples with highest confidence

    #Add selected samples to labeled data
    X_labeled = np.vstack([X_labeled, X_unlabeled[high_confidence_idx]]) #X_labeled is the array of labeled data
    y_labeled = np.hstack([y_labeled, model.predict(X_unlabeled[high_confidence_idx])]) #y_labeled is the array of labels for the labeled data

    #remove selected samples from unlabeled data
    X_unlabeled = np.delete(X_unlabeled, high_confidence_idx, axis=0) #X_unlabeled is the array of unlabeled data


    #Re-train the model on updated labeled data
    model.fit(X_labeled, y_labeled)

#Evaluate the model on original test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)    
    
    