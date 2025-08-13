import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# print("Addition: \n", A + B)
# print("Subtraction: \n", B - A)

C = 2 * A
# print("Scalar Multiplication \n", C)

result = np.dot(A, B)

# print("Matrix Multiplication \n", result)

I = np.eye(5) # Identity Matrix. This is a square matrix with 1s on the diagonal and 0s elsewhere
# print("Identity Matrix \n", I)

Z = np.zeros((2, 3)) # Zero Matrix. This is a matrix with all elements as 0
# print("Zero Matrix \n", Z)

D = np.diag([1, 2, 3]) # Diagonal Matrix. This is a square matrix with non-zero elements on the diagonal and 0s elsewhere
print("Diagonal Matrix\n", D)




