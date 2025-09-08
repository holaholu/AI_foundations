#Description
#Autoencoders are a type of neural network that is used for dimensionality reduction and feature extraction. They work by encoding the input data into a lower-dimensional space and then decoding it back to the original space. They are useful for noise reduction, data compression, anomaly detection,pre-training other neural networks, and feature extraction. 

#import necessary libraries
from tensorflow.keras.models import Model #Model is the base class for all neural network models in Keras. It is used to create and train the autoencoder.
from tensorflow.keras.layers import Input, Dense #Input layer is the input layer of the autoencoder. Dense layer is the hidden layer of the autoencoder.Dense layers are fully connected layers meaning each neuron is connected to every other neuron in the previous layer.
import numpy as np

#Sample Data (e.g points in 5-dimensional space)
X = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]])

#Define the autoencoder model
input_dim = X.shape[1] #input dimension is the number of features in the input data. 1 specifies the number of column of the input data derived from the shape of the input data
encoding_dim = 2 #encoding dimension is the number of features in the encoded data. 2 specifies the number of column of the encoded data

#Encoder
input_layer = Input(shape=(input_dim,)) #input layer is the input layer of the autoencoder. 
encoded = Dense(encoding_dim, activation='relu')(input_layer) #encoded is the encoded data. activation='relu' is the activation function used in the encoded layer.ReLU (Rectified Linear Unit) is a non-linear activation function that is used to introduce non-linearity into the model. It is defined as f(x) = max(0, x).

#Decoder
decoded = Dense(input_dim, activation='sigmoid')(encoded) #decoded is the decoded data. activation='sigmoid' is the activation function used in the decoded layer.Sigmoid is a non-linear activation function that is used to introduce non-linearity into the model. It is defined as f(x) = 1 / (1 + e^(-x)).

#Autoencoder
autoencoder = Model(input_layer, decoded) #autoencoder is the autoencoder model. It is created by connecting the input layer to the decoded layer. the decoded layer has the encoded data (see definition above).

autoencoder.compile(optimizer='adam', loss='mse') #autoencoder is compiled using the Adam optimizer and mean squared error (MSE) loss function.Adam is an optimization algorithm that is used to update the weights of the model during training. It is a variant of stochastic gradient descent (SGD) that is used to optimize the weights of the model. It is defined as f(x) = 1 / (1 + e^(-x)).

#Train the autoencoder
autoencoder.fit(X, X, epochs=100, batch_size=2, verbose=0) #autoencoder is trained using the input data X and same target data X. epochs is the number of times the model is trained on the input data. batch_size is the number of samples processed before the model is updated. verbose is the level of detail of the training process. 0 means no detail.

#GEt the encoded (compressed) representation of the input data
encoder = Model(input_layer, encoded) #encoder is the encoder model. It is created by connecting the input layer to the encoded layer. 
X_compressed = encoder.predict(X) #X_compressed is the compressed data.

print("Compressed Representation:\n", X_compressed)





