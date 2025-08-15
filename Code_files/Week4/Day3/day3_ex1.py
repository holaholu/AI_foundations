import numpy as np
from scipy.stats import norm

# Generate random sample data
data = np.random.normal(loc=50, scale=10, size=100) #this is a sample of size 100 from a normal distribution with mean 50 and standard deviation 10

# Sample statistics
mean = np.mean(data)
std = np.std(data, ddof=1)
n = len(data)

# 95% Confidence Interval
z_value = norm.ppf(0.975) # z-value for 95% confidence interval
margin_of_error = z_value * (std / np.sqrt(n))
ci = (mean - margin_of_error, mean + margin_of_error)

print("Sample Mean: ", mean)
print("95% Confidence Interval: ", ci)