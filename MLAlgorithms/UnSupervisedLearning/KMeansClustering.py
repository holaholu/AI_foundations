#Description
#KMeans Clustering is a type of unsupervised learning algorithm that can be used for both classification and regression tasks. It works by finding the k clusters of data points in the feature space and using their labels to make a prediction. Useful for smaller datasets. Each cluster is defined by its centroid (mean of the data points in the cluster) and each data point is assigned to the cluster with the closest centroid. The algorithm iteratively updates the centroids until the clusters converge.

#import necessary libraries
from sklearn.cluster import KMeans
import numpy as np
#Sample data (e.g, points in 2D space)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])

#initialize and train the model
model = KMeans(n_clusters=2, random_state=42) # n_clusters is the number of clusters to find
model.fit(X)

#Get the cluster centers and labels
cluster_centers = model.cluster_centers_
labels = model.labels_

#Print the cluster centers and labels
print("Cluster Centers: ", cluster_centers)
print("Labels: ", labels)




