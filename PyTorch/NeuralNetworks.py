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
#print(output)   
#print(model.parameters())


#Backward Pass (backpropagation) involves computing the gradient of the loss with respect to the parameters of the model. This is done using the backward() method. The gradient is then used to update the parameters of the model to minimize the loss.  Backward pass is used to compute the gradient of the loss with respect to the parameters of the model.

#loss.backward() 
#Compute the gradient of the loss with respect to the parameters of the model

#Epoch is the number of times the entire training dataset is passed through the network. 
#Iteration is one pass through a single batch of data. 
# size of dataset = batch size * number of iterations
#loss function is a function that measures the difference between the predicted output and the actual output. Types include Mean Squared Error, Cross Entropy, Binary Cross Entropy, etc.   
# optimizer is used to update the parameters of the model to minimize the loss. Types include Stochastic Gradient Descent (updates parameters by moving them in the direction of the negative gradient), Adam (adaptive learning rate results in faster convergence), etc.     


#Transfer learning is a technique where a pre-trained model is used as a starting point for training a new model. The pre-trained model is used to extract features from the data, which are then used to train the new model. This is useful when there is a large amount of data and a small amount of labeled data. Two strategies are used: 

#1.Feature Extraction (Freezing the early layers of the model and only the final classification layer is trained on new dataset). Used when there is a small amount of labeled data and data is similar to the pre-trained model. 

#2. Finetuning is a technique where a pre-trained model is used to extract features from the data, which are then used to train the new model. It involves freezing the early layers of the model (which capture general features) and training the later layers (which capture task-specific features) on the new dataset. Used when there is a small amount of labeled data and data is different from the pre-trained model.

#Example of transfer learning using torchvision
import torchvision.models as models
resnet = models.resnet50(pretrained=True)

m_classes = 10
#finetuning
for param in resnet.parameters():
    param.requires_grad = False

#replace final layer with a new one for the specific task
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

#print(resnet)

#Example of transfer learning using huggingface transformers
from transformers import BertModel,BertTokenizer #transformers was built by huggingface

model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)
#print(outputs)

#torchvision.transform is used to transform the data.examples include resize, crop, normalize, etc.

#tokenization is used to convert text into a format that can be processed by deep learning models. It is used in natural language processing tasks. tokens are the basic unit of meaning. methods include word piece, character level, byte pair encoding, etc.

#Text Data Preprocessing includes padding (adding extra tokens to make all sequences the same length), truncation (removing extra tokens to make all sequences the same length), long sequence handling (splitting long sequences into smaller ones), etc.

# to save a model, we use torch.save(model.state_dict(), "model.pth")
# to load a model, we use model.load_state_dict(torch.load("model.pth"))

#TorchScript is a way to convert a PyTorch model into a static graph that can be executed on any device. It is used to deploy models on mobile devices, web browsers, and other devices. It is also used to convert models into other formats such as ONNX (Open Neural Network Exchange) and TensorFlow.

#Example of torchscript 
scripted_model = torch.jit.script(model)
scripted_model.save("scripted_model.pt")
scripted_model = torch.jit.load("scripted_model.pt")
scripted_model.eval()
output = scripted_model(input)
#print(output)

#ONNX is a format for representing machine learning models. It is used to deploy models on mobile devices, web browsers, and other devices. It is also used to convert models into other formats such as TensorFlow and PyTorch.

#other deployment serving options include Flask,Django FastAPI, AWS lambda, Google Cloud Run, Heroku, etc.

#pytorch debugger (pdb) is used to debug pytorch models. It is used to debug models during training and inference. It is used to debug models by setting breakpoints and inspecting the values of variables. It is used to debug models by using the pdb.set_trace() function.

#torch.clamp is used to clamp the values of a tensor to a range. It is used to prevent the values from going out of bounds. 

#torch.nn.parallel.distributedDataParallel is used to parallelize the training of a model. It is used to train models on multiple GPUs. It is used to train models on multiple machines.

#gradient accumulation is used when desired batch size is larger than the available memory. It is used to simulate a larger batch size by accumulating gradients over multiple smaller batches. 

#Dropout is a regularization technique used to prevent overfitting. It is used to randomly set a fraction of input units to 0 at each update during training time. It is used to prevent overfitting by making the model more robust to noise in the data.

#Weight decay is a regularization technique used to prevent overfitting. It is used to add a penalty to the loss function to discourage the model from learning complex patterns in the data. It is used to prevent overfitting by making the model more robust to noise in the data.

