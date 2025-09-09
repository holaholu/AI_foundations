#Save model to HDF5 file
model.save('model.h5')

#Load model from HDF5 file
model = tf.keras.models.load_model('model.h5')

#Save model architecture to JSON file
json_config = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(json_config)

#Load model architecture from JSON file
with open('model.json', 'r') as json_file:
    json_config = json_file.read()
model = tf.keras.models.model_from_json(json_config)
model.summary() #Print model summary

#save model weights to HDF5 file
model.save_weights('model_weights.h5')

#load model weights from HDF5 file
model.load_weights('model_weights.h5')

#Tensorflow serving for model deployment in a production environment
#1.installation and setup
# run docker pull tensorflow/serving
#2.Exporting Tensorflow model
import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
tf.saved_model.save(model, 'model')
#3.Run tensorflow serving
#run docker run -it -p 8501:8501 -v /path/to/model:/models/model -e MODEL_NAME=model -t tensorflow/serving
#4.Serving Requests
#run curl -d @input.json http://localhost:8501/v1/models/model:predict


#Tensorflow Lite os used for model deployment on mobile and embedded devices.
#Convert the model to Tensorflow lite format
import tensorflow as tf
model = tf.keras.models.load_model('model.h5')
tflite_model = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = tflite_model.convert()
#Save the model to a file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)


#Tensorflow lite interpreter for model deployment on mobile and embedded device.
import tensorflow as tf
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input details:", input_details)
print("Output details:", output_details)
#Run inference
input_data = np.array(np.random.random_sample(input_details[0]['shape']), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Output data:", output_data)

#Tensorflow Extended (TFX) is used for model deployment in a production environment.

#GANs(Generative Adversarial Networks) are a class of deep learning models that are used for generating new data that is similar to the training data. They are made up of two networks: the generator and the discriminator. The generator generates new data, while the discriminator tries to distinguish between real and generated data. Applications include image generation, video generation, and audio generation. 






