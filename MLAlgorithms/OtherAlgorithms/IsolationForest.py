#Description
#Isolation Forest is an unsupervised ensemble learning algorithm that is used for anomaly detection. It is based on the idea of isolating anomalies by randomly selecting features and splitting the data into subset rather than profiling normal data points. The algorithm randomly selects a feature and a split value to partition the data, creating trees where anomalies are easier to isolate due to their sparse distribution. Anomalies are identified based on their short path length in the trees as they are isolated faster than normal data points. 

#import necessary libraries
from sklearn.ensemble import IsolationForest
import numpy as np


#Sample data (e.g, normal data points clustered around 0)
X = 0.3*np.random.randn(100,2) #100 data points in 2D space.0.3 is the standard deviation from the mean of zero.
X_train = np.r_[X+2, X-2] #200 data points in 2D space around two clusters. r_ is used to stack arrays in sequence vertically (row wise).

#New test data including both normal and anomaly data
X_test = np.r_[X+2, X-2, np.random.uniform(low=-6, high=6, size=(20,2))] #220 data points in 2D space around two clusters. np.random.uniform(low=-6, high=6, size=(20,2)) is used to generate random data points in the range [-6,6] in 2D space as anomalies.

#initialize and train the model
model = IsolationForest(contamination=0.1,random_state=42) # contamination is the amount of outliers in the data. It is a value between 0 and 1. random_state is used to ensure reproducibility of the results.
model.fit(X_train)

#Make predictions on test data (-1 = anomaly, 1 = normal)
predictions = model.predict(X_test)

#Print predictions
print("Predictions: ", predictions)


