#Description
#tSNE (t-Distributed Stochastic Neighbor Embedding) is a type of unsupervised learning algorithm that is used for dimensionality reduction. It works by finding the k t-distributed neighbors of the data and using them to represent the data in a lower-dimensional space. It is useful for visualizing high-dimensional data in a 2D or 3D space. Each t-distributed neighbor is defined by its distance (euclidean distance) and its importance (probability). The algorithm iteratively updates the t-distributed neighbors until the data is well-represented in the lower-dimensional space.It preserves the local structure of the data while sacrificing the global structure of the data.Best suited for smaller datasets.

#import necessary libraries
from sklearn.manifold import TSNE
import numpy as np

#Sample data (e.g, points in high-dimensional space)
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])

#initialize and train the model
model = TSNE(n_components=2,perplexity=4, random_state=42) # n_components is the number of principal components to find. here we are reducing the dimensionality from 3 to 2. Perplexity is the number of nearest neighbors to consider
X_reduced = model.fit_transform(X)
 
#Print the reduced data
print("Reduced Data: ", X_reduced)


