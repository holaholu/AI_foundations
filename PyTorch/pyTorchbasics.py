#PyTorch is a deep learning framework that is used for building and training models. It is a Python-based framework developed by Facebook. It is a more recent framework than TensorFlow and is based on the idea of autograd (automatic differentiation). Pytorch operates on tensors (multi-dimensional arrays similar to numpy arrays but for GPU acceleration(parallel processing)) and provides a set of tools for building and training models. It uses dynamic computation graphs ( tensorflow uses static graphs prior to version 2`s Eager Execution) which allows for more flexibility in building and training models.

#Provides libraries such as Torchvision (for computer vision tasks), Torchtext (for natural language processing tasks), Torchaudio (for audio processing tasks), TorchData (for data loading and preprocessing tasks) and TorchServe (for model deployment).

#Import necessary libraries
import torch #provides a set of tools for deep learning

#Create two random tensors
x = torch.rand(3, 3) #create a 3x3 tensor with random values between 0 and 1
y = torch.rand(3, 3) #create a 3x3 tensor with random values between 0 and 1

#Perform matrix multiplication
z = torch.matmul(x, y) #perform matrix multiplication

#Print the tensors
# print("x:")
# print(x)
# print("y:")
# print(y)
# print("z:")
# print(z)


#if GPU is available, move the tensors to GPU
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    z = z.cuda()

# print("x:")
# print(x)
# print("y:")
# print(y)
# print("z:")
# print(z)


#Tensor Rank is the number of dimensions of the tensor. Rank 0 is a scalar, rank 1 is a vector, rank 2 is a matrix, rank 3 is a 3D tensor, and so on.

#Tensor Shape is a tuple of integers that specify the size of the tensor in each dimension. A 2D tensor has a shape of (rows, columns).

#Tensor Data Type is the type of data stored in the tensor. Common data types are float32, float64, int32, int64, bool and uint8.

#Tensor Device is the device on which the tensor is stored. Common devices are CPU and GPU.

#Tensor requires_grad is a boolean that specifies whether the tensor requires gradient calculation. It is used for backpropagation (computing gradients for optimization).

#Create Tensor from list or numpy array
import numpy as np
x = torch.tensor([1, 2, 3]) #create a 1D tensor from a list
numpy_array = np.array([1, 2, 3]) #create a numpy array
y = torch.from_numpy(numpy_array) #create a tensor from a numpy array
z = torch.tensor(numpy_array) #create a tensor from a numpy array

#print the tensors
# print("x:")
# print(x)
# print("y:")
# print(y)
# print("z:")
# print(z)

#Create Zeros, Ones and random tensors
x = torch.zeros(3, 3) #create a 3x3 tensor of zeros
y = torch.ones(3, 3) #create a 3x3 tensor of ones
z = torch.rand(3, 3) #create a 3x3 tensor of random values between 0 and 1
z_normal = torch.randn(3, 3) #create a 3x3 tensor of random values from a normal distribution with mean 0 and standard deviation 1

#print the tensors
# print("x:")
# print(x)
# print("y:")
# print(y)
# print("z:")
# print(z)
# print("z_normal:")
# print(z_normal)

#create tensor while specifying data type and device
#x = torch.zeros(3, 3, dtype=torch.float32, device="cuda") #create a 3x3 tensor of zeros
#y = torch.tensor([1,2,3], dtype=torch.float32, device="cuda") #create a 1D tensor from a list

#elementwise operations
z = x + y #elementwise addition
# print("z:")
# print(z)

#matrix operations
z = torch.matmul(x, y) #matrix multiplication
# print("z:")
# print(z)

#indexing and slicing
x=torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print("x:")
# print(x)
# print("x[0]:") #print the first row
# print(x[0])
# print("x[:, 0]:") #print the first column
# print(x[:, 0])
#print first and last row together 
selected_row =x[[0,2],:]
# print("selected_row:")
# print(selected_row)

#broadcasting is a feature that allows tensors of different shapes to be used in elementwise operations
x = torch.tensor([1, 2, 3])
y = torch.tensor([[4, 5, 6], [1, 2, 3]])
z = x + y
# print("z:")
# print(z)

#In-place operations
x.add_(5) #add y to x in-place
print("x:")
print(x)

