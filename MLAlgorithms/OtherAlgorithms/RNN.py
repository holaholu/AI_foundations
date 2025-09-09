#Description
#RNN is a type of neural network that is used to process sequential data (data that is ordered in time or space) such as time series data, text, and speech. RNNs have connections that form directed cycles allowing them to maintain information from previous steps in the sequence.  A common variant of RNN is the Long Short-Term Memory (LSTM) network which which helps to address the vanishing gradient problem (long-term dependencies).

#import necessary libraries
import tensorflow as tf #provides a set of tools for deep learning
from tensorflow.keras import layers,models #provides a set of tools for building and training neural networks
from tensorflow.keras.datasets import imdb #provides a set of pre-built datasets for machine learning.imdb is a dataset of movie reviews.
from tensorflow.keras.preprocessing import sequence #provides a set of tools for preprocessing data

#load and preprocess the data
max_features = 10000 #number of words to consider as features/vocabulary size/most common words
max_len = 500 #max number of words to consider in each review

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) #load the data
x_train = sequence.pad_sequences(x_train, maxlen=max_len) #pad the sequences to be the same length
x_test = sequence.pad_sequences(x_test, maxlen=max_len) #pad the sequences to be the same length

#Define the model
model = models.Sequential([ #create a sequential model]   
    layers.Embedding(max_features, 32, input_length=max_len), #embedding layer.This layer maps the input data into a dense vector space of fixed size (32 in this case). 
    layers.SimpleRNN(32), #simpleRNN layer. This layer is used to process the input data and extract features from it.32 is the number of units (neurons) in the layer.
    layers.Dense(1, activation='sigmoid') #dense layer. This layer is used to output the final prediction. 1 is the number of units (neurons) in the layer. activation='sigmoid' is the activation function used in the dense layer.Sigmoid is a non-linear activation function that is used to introduce non-linearity into the model. It is defined as f(x) = 1 / (1 + e^(-x)).
    ]) 

#compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #adam optimizer works by minimizing the loss function using gradient descent. binary_crossentropy works by comparing the predicted and actual values to calculate the loss. metrics=['accuracy'] is the metric used to evaluate the performance of the model.

#train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2) #train the model. epochs is the number of times the model is trained on the input data. batch_size is the number of samples processed before the model is updated. validation_split is the fraction of the training data to be used as validation data ( allows the model to be evaluated on a separate set of data during training).

#evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")










