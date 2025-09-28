#import the required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical #<-- for one-hot encoding required for classification tasks
import matplotlib.pyplot as plt #<-- for visualization

#Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

#Preprocess the data to normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

#Reshape the images to (28,28,1) [height, width, channels] as they are grayscale images which have a single channel
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

#Convert the labels to one-hot encoded vectors .E.g [0,0,0,0,0,0,0,0,0,1] for the digit 9
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Build the CNN model
model = models.Sequential()

#First Convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) #32 filters, 3x3 filter size, ReLU activation, input shape (28,28,1)
model.add(layers.MaxPooling2D((2, 2))) #2x2 pooling size which means it will take a 2x2 window and return the maximum value in that window

#Second Convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

#Third Convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

#Flatten the 3D output to 1D and add a dense layer
model.add(layers.Flatten()) #Flatten the 3D output to 1D
model.add(layers.Dense(64, activation='relu')) #Add a dense layer with 64 units and ReLU activation

#Output layer with 10 neurons (for 10 digit classes)
model.add(layers.Dense(10, activation='softmax')) #Add a dense layer with 10 units and softmax activation

#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Train the model
print("DEBUG: start training")
model.fit(train_images, train_labels, epochs=5,batch_size=64, validation_data=(test_images, test_labels))
print("DEBUG: finished training")
#Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc*100:.2f}%")

#Make predictions
predictions = model.predict(test_images)
print(f"Prediction for first test image: {np.argmax(predictions[0])}")

plt.imshow(test_images[0].reshape(28,28), cmap='gray')
plt.title(f"Predicted Label: {predictions[0].argmax()}")
plt.show()




