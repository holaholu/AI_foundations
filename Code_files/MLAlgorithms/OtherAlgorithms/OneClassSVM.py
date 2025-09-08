#Description
#OneClassSVM (Support Vector Machine) is an algorithm for anomaly detection by identifying the boundary of normal data points and classifying any data point that falls outside of this boundary as an anomaly. It separates data into two classes: normal ( high density) and anomaly (low density). It is useful for anomaly detection in time series data, image data, and other types of data.

#import necessary libraries
from sklearn.svm import OneClassSVM
import numpy as np

#Sample data (e.g, normal data points clustered around 0)
X = 0.3*np.random.randn(100,2) #100 data points in 2D space.0.3 is the standard deviation
X_train = np.r_[X+2, X-2] #200 data points in 2D space around two clusters. r_ is used to stack arrays in sequence vertically (row wise).

#New test data including both normal and anomaly data
X_test = np.r_[X+2, X-2, np.random.uniform(low=-6, high=6, size=(20,2))] #220 data points in 2D space around two clusters. np.random.uniform(low=-6, high=6, size=(20,2)) is used to generate random data points in the range [-6,6] in 2D space.

#initialize and train the model
model = OneClassSVM(gamma="auto",nu=0.1) # gamma is the kernel coefficient. nu is the nu parameter of the OneClassSVM. "auto" is used to automatically calculate the gamma value. nu means the amount of outliers in the data. It is a value between 0 and 1.
model.fit(X_train)

#Make predictions
predictions = model.predict(X_test)

#Print the predictions
print("Predictions: ", predictions)



