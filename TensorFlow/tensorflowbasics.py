import tensorflow as tf

# Enable eager execution (default in TF2)
tf.config.run_functions_eagerly(True)

# Define TensorFlow constants
x = tf.constant(2) #values cannot be changed once defined
y = tf.constant(3)
z = tf.Variable(4) #values can be changed
z.assign(5) #change the value of z
print("z:",z)
z2 =tf.random.normal(shape=(3,3)) #create a random tensor
print("z2:",z2)
z2 =tf.random.normal(shape=(3,3)) #create another random tensor
print("z2:",z2)

# Perform operations (eager execution)
add_result = tf.add(x, y)
sub_result = tf.subtract(x, y)
mul_result = tf.multiply(x, y)
div_result = tf.divide(x, y)

# Print results directly (no session needed in TF2 eager execution)
print("Addition:", add_result.numpy())
print("Subtraction:", sub_result.numpy())
print("Multiplication:", mul_result.numpy())
print("Division:", div_result.numpy())

#Other arithmetic operations include tf.mod(x, y) for modulo, tf.pow(x, y) for power, tf.sqrt(x) for square root,tf.square(x) for square, tf.exp(x) for exponential, tf.log(x) for natural log,
# Reduction operations include tf.reduce_sum(x) for sum, tf.reduce_mean(x) for mean, tf.reduce_max(x) for max, tf.reduce_min(x) for min

#Matrix operations include tf.matmul(x, y) for matrix multiplication,tf.transpose(x) for transpose,tf.linalg.inv(x) for inverse, tf.tensordot(x, y, axes=2) for tensor dot product, tf.einsum('ij,jk->ik', x, y) for Einstein summation convention

#indexing and slicing examples include
tensor_a =tf.constant([[1,2,3],[4,5,6]])
element =tensor_a[0,1] #access element at row 0 and column 1
row =tensor_a[0,:] #access row 0
column =tensor_a[:,1] #access column 1

#Broadcasting allows operations between tensors of different shapes. The smaller tensor is broadcasted to match the shape of the larger tensor
#Example
a =tf.constant([[1,2,3],[4,5,6]])
b =tf.constant([1,0,1])
result_broadcast =tf.add(a,b) # b is broadcasted to match the shape of a

#===================================================================
# Visualization of computational graph using TensorFlow 2.x
# Create a simple function to trace
@tf.function
def simple_function(x, y):
    return tf.add(x, y)

# Create a summary writer
logdir = "logs"
writer = tf.summary.create_file_writer(logdir)

# Call the function to ensure it's traced
result = simple_function(tf.constant(2.0), tf.constant(3.0))

# Write the graph to the summary
with writer.as_default():
    tf.summary.trace_on(graph=True, profiler=False)
    # Call the function again to trace it
    simple_function(tf.constant(2.0), tf.constant(3.0))
    tf.summary.trace_export(name="my_func_trace", step=0)

# Close the writer
writer.close()

print("Graph has been written to the logs directory. You can visualize it using TensorBoard.")
print("To view the graph, run: tensorboard --logdir=logs")

#===================================================================

# In TensorFlow 2.x, we can use eager execution directly
# No need for sessions or graphs in most cases
x = tf.constant(2)
y = tf.constant(3)
z = tf.add(x, y)
print("Result:", z.numpy())  # .numpy() converts tensor to a Python scalar

#Activation functions are used to introduce non-linearity into the model
#Sigmoid activation function converts to values between 0 and 1.Makes it suitable for binary classification
sigmoid = tf.keras.activations.sigmoid
print("Sigmoid:", sigmoid(z).numpy())

#hyberbolic tangent activation function converts to values between -1 and 1. Making it possible to handle both positive and negative values/hidden layers
hyperbolic_tangent = tf.keras.activations.tanh
print("Hyperbolic Tangent:", hyperbolic_tangent(z).numpy())

#Rectified Linear Unit (ReLU) activation function converts to values between 0 and infinity. Making it suitable for hidden layers
relu = tf.keras.activations.relu
print("ReLU:", relu(z).numpy())

#Leaky ReLU activation function converts to values between 0 and infinity. Making it suitable for hidden layers
leaky_relu = tf.keras.activations.leaky_relu
print("Leaky ReLU:", leaky_relu(z).numpy())

#Softmax activation function converts to values between 0 and 1. Making it suitable for multi-class classification
softmax = tf.keras.activations.softmax
print("Softmax:", softmax(z).numpy())


#Loss functions are used to measure the difference between the predicted and actual values
#Binary Cross-Entropy loss function is used for binary classification
binary_crossentropy = tf.keras.losses.binary_crossentropy
print("Binary Cross-Entropy:", binary_crossentropy(z, z).numpy())

#Categorical Cross-Entropy loss function is used for multi-class classification.Accepts one-hot encoded labels
categorical_crossentropy = tf.keras.losses.categorical_crossentropy
print("Categorical Cross-Entropy:", categorical_crossentropy(z, z).numpy())

#Sparse Categorical Cross-Entropy loss function is used for multi-class classification.Accepts integer labels
sparse_categorical_crossentropy = tf.keras.losses.sparse_categorical_crossentropy
print("Sparse Categorical Cross-Entropy:", sparse_categorical_crossentropy(z, z).numpy())


#Mean Squared Error loss function is used for regression
mean_squared_error = tf.keras.losses.mean_squared_error
print("Mean Squared Error:", mean_squared_error(z, z).numpy())

#Mean Absolute Error loss function is used for regression
mean_absolute_error = tf.keras.losses.mean_absolute_error
print("Mean Absolute Error:", mean_absolute_error(z, z).numpy())

#Optimizer are used to update the model's weights to minimize the loss function
#Optimizer include tf.keras.optimizers.SGD for Stochastic Gradient Descent, tf.keras.optimizers.Adam for Adaptive Moment Estimation, tf.keras.optimizers.RMSprop for Root Mean Square Propagation, tf.keras.optimizers.Adagrad for Adaptive Gradient.

#Keras is a high-level neural networks API that is used to build and train models.

#Key components of Keras include layers,activations,loss functions,metrics, models, optimizers, and callbacks.

#Steps to Train and Evaluate a model include: data preprocessing, model definition, model compilation, model training, model prediction,model evaluation, model visualization / deployment,  model save, and continuous improvement.

#CNNs are used for object detection, image segmentation, and image classification. Kernel/filter is a small matrix that is used to extract features from the image. Pooling is used to reduce the spatial dimensions of the feature maps by taking the maximum value in each pool.

#RNNs are used for sequence prediction.Architecture includes an input layer, an output layer, and a hidden layer. Types of RNNs include Simple RNN, LSTM, and GRU. Applications include time series prediction (stock market prediction,weatherforecasting, demand forecasting, healthcare forecasting), natural language processing(Text Generation, Machine Translation, Sentiment Analysis), and speech recognition.

#LSTMs are used for time series prediction.

#Transformers are used for natural language processing.

#Transfer Learning is used to transfer the knowledge of a pre-trained model to a new model. Benefits include faster training, better performance, and less data required. Strategies in transfer learning include feature extraction(base_model.trainable = False), fine-tuning(base_model.trainable = True), and transfer learning with data augmentation(base_model.trainable = True).