import numpy as np

np.random.seed(42) # for reproducibility. It makes the random numbers the same every time we run the code

random_array = np.random.rand(3, 3)
print("Random Array: \n", random_array)


random_integers = np.random.randint(0, 10, size=(2,3))
print("Random Integers: \n", random_integers)

arr = np.array([[1,2,3],[4,5,6]])
print("Sum: ", np.sum(arr))
print("Mean: ", np.mean(arr))
print("Standard Deviation: ", np.std(arr))
print("Min: ", np.min(arr))
print("Max: ", np.max(arr))
print("Median: ", np.median(arr))
print("Variance: ", np.var(arr))
print("Sum along rows: ", np.sum(arr, axis=1))
print("Sum along columns: ", np.sum(arr, axis=0))
print("Transpose: ", arr.T)
print("Reshape: ", arr.reshape(6,1))
print("Flatten: ", arr.flatten())

arr[arr>3] = 0
print("Modified Array: \n", arr)