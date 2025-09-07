#Description
#PCA (Principal Component Analysis) is a type of unsupervised learning algorithm that is used for dimensionality reduction. It works by finding the k principal components of the data and using them to represent the data in a lower-dimensional space. it identifies directions (principal components) that capture the most variance in the data. Useful for larger datasets, data visualization, noise reduction, speeding up machine learning algorithms by reducing the number of features, and feature extraction. Each principal component is defined by its direction (eigenvector) and its importance (eigenvalue). The algorithm iteratively updates the principal components until the data is well-represented in the lower-dimensional space.

#import necessary libraries
from sklearn.decomposition import PCA
import numpy as np

#Sample data (e.g, points in 3D space)  
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])

#initialize and train the model
model = PCA(n_components=2) # n_components is the number of principal components to find. here we are reducing the dimensionality from 3 to 2
X_reduced = model.fit_transform(X)

#Print the reduced data
print("Reduced Data: ", X_reduced)
print("Explained Variance: ", model.explained_variance_ratio_) # explained variance ratio is the ratio of the variance of the data along each principal component to the total variance of the data


