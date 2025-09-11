#Autograd is a feature that allows PyTorch to automatically compute gradients for backpropagation. It is used for training models. It is a Pytorch automatic differentiation engine  that records operations performed on tensors to create a computational graph. This graph is then used to compute gradients for optimization.

#Computational Graph is a graph that represents the operations performed on tensors.This tracks the dependencies between tensors and allows PyTorch to compute gradients for optimization.
#Backward Pass is the process of computing gradients for optimization. Autograd traverses the computational graph in reverse to compute the gradients.

#Gradient is the partial derivative of the loss with respect to the parameters of the model. It is used to update the parameters of the model to minimize the loss. once computed, it is stored in the .grad attribute of the tensor. Gradients are computed by default when we call the backward() method. however, calling the backward() method multiple times will raise an error. to compute the gradient multiple times, we need to call the zero_grad() method to reset the gradients.

import torch 
x= torch.tensor(2.0, requires_grad=True) #requires_grad=True means that we want to compute the gradient of x with respect to the loss
y = x**2 #y is a function of x
y.backward() #compute the gradient of y with respect to x
print(x.grad) #print the gradient of x with respect to the loss

#detach is used to detach the tensor from the computational graph. This is used to prevent the tensor from being tracked for gradient computation.
detached_x = x.detach()


