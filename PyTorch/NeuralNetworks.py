#Neural Network consists of layers of neurons Each layer has a set of weights and biases that are used to compute the output of the layer. The output of one layer is the input of the next layer. The final output of the network is the output of the last layer. The network is trained by adjusting the weights and biases to minimize the loss. It is the backbone of deep learning models.

# A neuron is a basic unit of a neural network. It takes in a set of inputs, multiplies them by a set of weights, adds a bias, and outputs a value. it performs a weighted sum of inputs, applies an activation function, and outputs a value. The activation function is a non-linear function that introduces non-linearity into the network to allow the network to learn complex patterns (e.g ReLU(Rectified Linear Unit - Most popular), Sigmoid(for binary classification), Tanh(for hyperbolic tangent), Softmax(for multi-class classification)). 

#Neuron Operation

# y = a(w1x1 + w2x2 + b)

#where a is the activation function, w1 and w2 are the weights, x1 and x2 are the inputs, and b is the bias.

# 3 layers of neurons Input Layer, Hidden Layer and Output Layer

#Forward Pass: The input is passed through the network layer by layer. The output of one layer is the input of the next layer. The final output of the network is the output of the last layer. The output is then compared to the target value to calculate the loss. 


#Defining a Neural Network Class. Two methods __init__ (to define the layers) and forward (to define how the data flows through the network. First stage is the forward pass) needed
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50) #First Fully Connected Layer
        self.fc2 = nn.Linear(50, 1) # Output Layer
    
    def forward(self, x):
        x = torch.relu(self.fc1(x)) # Apply ReLU activation function to the output of the first fully connected layer
        x = self.fc2(x) # Apply linear activation function to the output of the second fully connected layer
        return x

#Create an instance of the SimpleNN class
model = SimpleNN()
input = torch.randn(1, 10) #Random input of size 10
output = model(input) #Forward pass
print(output)   
print(model.parameters())


#Backward Pass (backpropagation) involves computing the gradient of the loss with respect to the parameters of the model. This is done using the backward() method. The gradient is then used to update the parameters of the model to minimize the loss.  Backward pass is used to compute the gradient of the loss with respect to the parameters of the model.

#loss.backward() 
#Compute the gradient of the loss with respect to the parameters of the model

#Epoch is the number of times the entire training dataset is passed through the network. 
#Iteration is one pass through a single batch of data. 
# size of dataset = batch size * number of iterations
#loss function is a function that measures the difference between the predicted output and the actual output. Types include Mean Squared Error, Cross Entropy, Binary Cross Entropy, etc.   
# optimizer is used to update the parameters of the model to minimize the loss. Types include Stochastic Gradient Descent (updates parameters by moving them in the direction of the negative gradient), Adam (adaptive learning rate results in faster convergence), etc.     
      