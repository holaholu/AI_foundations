from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Sample Data
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) #reshape(-1, 1) is used to convert the array to a column vector
y = np.array([2, 4, 5, 8, 10])

#Pearson Correlation
# pearson_corr, _ = pearsonr(x, y)
# print("Pearson Correlation coefficient: ", pearson_corr)

# #Spearman Correlation
# spearman_corr, _ = spearmanr(x, y)
# print("Spearman Correlation coefficient: ", spearman_corr)


# Fit Linear Regression
model = LinearRegression()
model.fit(x, y)

print("Slope: ", model.coef_[0])
print("Intercept: ", model.intercept_)
print("R-Squared: ", model.score(x, y))