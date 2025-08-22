import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data() # load the dataset

# Normalize data
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0 # normalize the data. This is used to convert the data to a format that can be used by the model. The data is converted to a float32 format and then normalized to a range of 0 to 1.
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# One-hot encode labels
y_train = to_categorical(y_train, 10) # convert the labels to one-hot encoded format. This is used to convert the labels to a format that can be used by the model. The labels are converted to a binary matrix where each row has a 1 in the column corresponding to the label.
y_test = to_categorical(y_test, 10)

print(f"Training Data Shape: {X_train.shape}")
print(f"Test Data Shape: {X_test.shape}")

# Build the model
model = Sequential([ #sequential model is a linear stack of layers. This is used to build the model.
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)), #conv2d is a 2d convolutional layer. This is used to extract features from the input data. The input shape is (28, 28, 1) which means the input data is a 28x28 image with 1 channel.
    MaxPooling2D(2, 2), #maxpooling is a pooling layer. This is used to reduce the spatial dimensions of the input data. The pool size is (2, 2) which means the input data is reduced to a 14x14 image.
    Flatten(), #flatten is a layer that converts the 2d data to a 1d array. This is used to convert the 2d data to a 1d array.
    Dense(128, activation="relu"), #dense is a fully connected layer. This is used to extract features from the input data. The activation function is relu which means the output is the maximum of 0 and the input.
    Dropout(0.5), #dropout is a regularization technique. This is used to prevent overfitting. The dropout rate is 0.5 which means 50% of the input data is dropped.
    Dense(10, activation="softmax")   #dense is a fully connected layer. This is used to extract features from the input data. The activation function is softmax which means the output is the maximum of 0 and the input.
])

# Display model architecture
model.summary()

# Compile the model
model.compile(
    optimizer="adam", #adam is an optimizer. This is used to update the weights of the model during training.
    loss="categorical_crossentropy", #categorical_crossentropy is a loss function. This is used to measure the difference between the predicted and actual values.
    metrics=['accuracy'] #accuracy is a metric. This is used to measure the performance of the model.
)

# Train the model
history = model.fit(
    X_train, y_train, #training data
    epochs=10, #number of epochs.This is the number of times the model will be trained on the data.
    batch_size=32, #batch size. This is the number of samples processed before the model is updated.
    validation_split=0.2 #validation split. This is the percentage of the data that will be used for validation.
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model
model.save('mnist_classifier.h5')

# Load the model
from tensorflow.keras.models import load_model
loaded_model = load_model('mnist_classifier.h5')

# Verify loaded model performance
loss, accuracy = loaded_model.evaluate(X_test, y_test)
print(f"Loaded model Accuracy: {accuracy:.4f}")
