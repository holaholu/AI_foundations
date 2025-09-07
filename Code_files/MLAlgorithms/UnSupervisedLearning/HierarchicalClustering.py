#Description
#Hierarchical Clustering is a type of unsupervised learning algorithm that builds a hierarchy of clusters by iteratively merging or splitting the data points based on their similarity/distance from each other forming a tree-like structure called a dendrogram. The hierarchy can be used to choose a suitable number of clusters by cutting the dendrogram at a desired height/level. It is a powerful algorithm that can handle both numerical and categorical features, and it is less sensitive to overfitting than other algorithms.

#import necessary libraries
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import numpy as np

#Sample data (e.g, points in 2D space)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])

#perform hierarchical/agglomerative clustering
Z = linkage(X, 'ward') # 'ward' is the linkage method. other linkage methods include 'single' (min distance), 'complete' (max distance), 'average' (mean distance). Z stores the hierarchical clustering information as a matrix of distances between clusters.

#plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(Z) # a dendrogram is a tree-like structure that shows the hierarchical clustering of the data points
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Data points')
plt.ylabel('Distance')
plt.show()



