#Description
#GMM (Gaussian Mixture Model) is a type of unsupervised probabilistic clustering algorithm that assumes data is generated from a mixture of several Gaussian distributions with unknown parameters.This is soft clustering as GMM assigns each data point to the cluster with the highest probability.It is useful when cluster have different shapes or densities.



#import necessary libraries
from sklearn.mixture import GaussianMixture
import numpy as np

#Sample data (e.g, points in 2D space)
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

#initialize and train the model
model = GaussianMixture(n_components=2, random_state=42) # n_components is the number of clusters to find
model.fit(X)  

#Get the cluster labels and the probability of each data point belonging to each cluster
labels = model.predict(X)
probabilities = model.predict_proba(X)

#Print the cluster labels and the probability of each data point belonging to each cluster
print("Labels: ", labels)
print("Probabilities: ", probabilities)     
