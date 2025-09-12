#Description
#Convolutional Neural Network (CNN) is a type of deep learning algorithm that is used for image classification. It is based on the idea of convolving the input image with a set of filters to capture spatial hierarchies and extract features (like edges, shapes, textures, etc.) from the image. The features are then used to train a classifier to predict the class of the image. CNNs are useful in computer vision tasks such as image classification, object detection, image segmentation, and image classification. Common CNNs include LeNet, AlexNet, VGG, ResNet and Inception.

#Stride is the number of pixels that the filter moves over the input image. A stride of 1 means that the filter moves one pixel at a time. A stride of 2 means that the filter moves two pixels at a time.

#Padding is the number of pixels added to the input image to make it a multiple of the filter size. Padding is used to preserve the spatial dimensions of the feature maps.

#Max Pooling is used to reduce the spatial dimensions of the feature maps by taking the maximum value in each pool.Average Pooling is used to reduce the spatial dimensions of the feature maps by taking the average value in each pool.

#Fully Connected Layer is a layer that is used to perform a weighted sum of the inputs and outputs a single value. It is used to perform a classification task.

#Import necessary libraries
import tensorflow as tf #provides a set of tools for deep learning
from tensorflow.keras import layers, models #provides a set of tools for building and training neural networks
from tensorflow.keras.datasets import mnist #provides a set of pre-built datasets for machine learning


#load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#normalize the pixel values to be between 0 and 1 since the data ranges from 0 to 255
x_train = x_train / 255.0
x_test = x_test / 255.0

#Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1) #reshape the training data to be 28x28 pixels with 1 channel (grayscale). -1 means that the number of samples is unknown.
x_test = x_test.reshape(-1, 28, 28, 1)

#Define the model
model = models.Sequential([    #create a sequential model which is a linear stack of layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), #first layer. 32 is the number of filters, (3, 3) is the size of the filter/kernel to detect features, activation='relu' is the activation function, input_shape is the shape of the input data
        layers.MaxPooling2D((2, 2)), #second layer. 2 is the size of the pool to reduce the spatial dimensions (by half) of the feature maps by taking the maximum value in each pool
        layers.Conv2D(64, (3, 3), activation='relu'), #third layer
        layers.MaxPooling2D((2, 2)), #fourth layer
        layers.Conv2D(64, (3, 3), activation='relu'), #fifth layer
        layers.Flatten(), #sixth layer. flatten is used to convert the 3D feature maps into a 1D vector to feed into a dense layer (fully connected layer)
        layers.Dense(64, activation='relu'), #seventh layer. dense is used to create a fully connected layer (fully connected layer)
        layers.Dense(10, activation='softmax') #eighth layer. softmax is used to convert the output of the network into a probability distribution over the possible classes (10 classes in this case for digits 0-9)
    ])

#compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train the model
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2) #validation_split is the fraction of the training data to be used as validation data ( allows the model to be evaluated on a separate set of data during training). batch_size is the number of samples processed before the model is updated. epochs is the number of times the model is trained on the input data.

#evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")








