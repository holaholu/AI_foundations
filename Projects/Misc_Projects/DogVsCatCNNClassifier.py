#download data from https://www.kaggle.com/datasets/tongpython/cat-and-dog

#importing the libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

#define paths to dataset
train_dir = './data/cats_and_dogs/training_set'
validation_dir = './data/cats_and_dogs/test_set'
test_image = './data/cats_and_dogs/cat.4007.jpg'

#Define ImageDataGenerator for data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255, #rescale the pixel values to the range [0, 1]
    rotation_range=40,#random rotation of images
    width_shift_range=0.2,#random horizontal shift of images
    height_shift_range=0.2,#random vertical shift of images
    shear_range=0.2,#random shearing of images
    zoom_range=0.2,#random zoom of images
    horizontal_flip=True,#random horizontal flip of images
    fill_mode='nearest'#fill mode for new pixels
)                               
#For the validation set, we only need to rescale the pixel values
validation_datagen = ImageDataGenerator(rescale=1./255)

#Load and preprocess the training data and validation data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),#resize the images to 150x150
    batch_size=32,#batch size for training
    class_mode='binary'#binary classification (cat or dog)
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),#resize the images to 150x150
    batch_size=32,#batch size for validation
    class_mode='binary'#binary classification (cat or dog)
)

#Define the CNN model
model = models.Sequential()

#First convolutional layer
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))

#Second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

#Third convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

#Fourth convolutional layer
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

#Flatten the output of the last convolutional layer
model.add(layers.Flatten())

#Add a dense/fully connected layer
model.add(layers.Dense(512, activation='relu'))

#Add the output layer
model.add(layers.Dense(1, activation='sigmoid'))

#Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Print the model summary
model.summary()

#Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100, #number of batches per epoch
    epochs=20,
    validation_data=validation_generator,
    validation_steps=50
)

#Plot the training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#Test the model with new image
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(model, img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) #convert the image to a numpy array
    img_array = np.expand_dims(img_array, axis=0) #add a batch dimension
    img_array /= 255.0 #rescale the pixel values to the range [0, 1]
    prediction = model.predict(img_array)
    if prediction[0] > 0.5:
        print(f"The image is a dog with a confidence of {prediction[0][0]:.2f}")
    else:
        print(f"The image is a cat with a confidence of {1-prediction[0][0]:.2f}")

#Example: Test the model with a cat image
predict_image(model, test_image)

    
    

