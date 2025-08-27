from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the data in the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# One-hot encode the labels into 10 classes of 0-9
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Training Data Shape: {X_train.shape}, Label Shapes: {y_train.shape}")
print(f"Test Data Shape: {X_test.shape}, Label Shapes: {y_test.shape}")

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32,32,3)), # First convolutional layer
    MaxPooling2D((2, 2)), # Pooling layer to reduce the spatial dimensions
    Conv2D(64, (3, 3), activation='relu'), # Second convolutional layer
    MaxPooling2D((2, 2)), # Pooling layer to reduce the spatial dimensions
    Flatten(), # Flatten the output to feed into a dense layer
    Dense(128, activation='relu'), # Fully connected layer
    Dropout(0.5), # Dropout to prevent overfitting
    Dense(10, activation='softmax') # Output layer with 10 classes
])

model.summary()

model.compile(
    optimizer='adam', # Use Adam optimizer
    loss='categorical_crossentropy', # Use categorical cross entropy loss
    metrics=['accuracy'] # Track accuracy
)

# TRain the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2
)

# Evaluate on the test dataset
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

import matplotlib.pyplot as plt

# Plot Accuracy
plt.plot(history.history['accuracy'], label="Training Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# Plot Loss
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()