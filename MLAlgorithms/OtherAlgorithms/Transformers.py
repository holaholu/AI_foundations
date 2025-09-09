#Description
#Transformers are deep learning architectures designed for handling sequential data without the need for recurrent connections (RNNs) or convolutions (CNNs). They are based on the idea of self-attention, which allows the model to focus on different parts of the input data at different times, making it more effective at capturing long-range dependencies in the data.It processes all token in the input sequence simultaneously, allowing it to capture dependencies between tokens regardless of their distance/position in the sequence.It is the foundation of many NLP tasks and models, including BERT, GPT, and T5.

#import necessary libraries
import tensorflow as tf #provides a set of tools for deep learning
from tensorflow.keras import layers #provides a set of tools for building and training neural networks
from tensorflow.keras.datasets import imdb #provides a set of pre-built datasets for machine learning.imdb is a dataset of movie reviews.
from tensorflow.keras.preprocessing import sequence #provides a set of tools for preprocessing data

#load and preprocess the data
max_features = 10000 #number of words to consider as features/vocabulary size/most common words
max_len = 200 #max number of words to consider in each review

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) #load the data
x_train = sequence.pad_sequences(x_train, maxlen=max_len) #pad the sequences to be the same length
x_test = sequence.pad_sequences(x_test, maxlen=max_len) #pad the sequences to be the same length

#Define a Transformer Block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1): #embed_dim is the dimension of the embedding space, num_heads is the number of attention heads, ff_dim is the dimension of the feed forward network, rate is the dropout rate (to prevent overfitting)
        super(TransformerBlock, self).__init__() #inherit from the parent(this refers to the current class) class
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim) #this layer implements the self-attention mechanism
        self.ffn = tf.keras.Sequential([#this layer implements the feed forward network
            layers.Dense(ff_dim, activation="relu"), #first layer
            layers.Dense(embed_dim), #second layer
            ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6) #this layer implements the layer normalization to stabilize and improve training.epsilon is a small constant to prevent division by zero
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6) #this layer implements the layer normalization to stabilize and improve training
        self.dropout1 = layers.Dropout(rate) #this layer implements the dropout to prevent overfitting
        self.dropout2 = layers.Dropout(rate) #this layer implements the dropout to prevent overfitting
    
    def call(self, inputs, training=None): #call the parent class constructor. None is used to indicate that the parameter is optional
        attn_output = self.att(inputs, inputs) #self-attention mechanism
        attn_output = self.dropout1(attn_output, training=training) #apply dropout to the attention output
        out1 = self.layernorm1(inputs + attn_output) #adds the input and the attention output and apply layer normalization
        ffn_output = self.ffn(out1) #passes the normalized output through the feed forward network
        ffn_output = self.dropout2(ffn_output, training=training) #apply dropout to the feed forward output
        return self.layernorm2(out1 + ffn_output) #adds the output of the feed forward network and the normalized output and apply layer normalization

#Define the Transformer Model
embed_dim = 32 #Embedding dimension
num_heads = 2 #Number of attention heads
ff_dim = 32 #Feed forward dimension
inputs = layers.Input(shape=(max_len,)) #Input layer
embedding_layer = layers.Embedding(input_dim=max_features, output_dim=embed_dim, input_length=max_len)#Embedding layer
x = embedding_layer(inputs) #Embedding layer
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim) #Transformer block
x = transformer_block(x, training=True) #Explicitly pass the training parameter to the transformer block
x = layers.GlobalAveragePooling1D()(x) #Reduces the dimensionality of the input by taking the average of the values in each feature map/time axis
x = layers.Dropout(0.1)(x) #Dropout layer
x = layers.Dense(20, activation="relu")(x) #Dense layer
x = layers.Dropout(0.1)(x) #Dropout layer
outputs = layers.Dense(1, activation="sigmoid")(x) #Dense layer with sigmoid for binary classification
model = tf.keras.Model(inputs=inputs, outputs=outputs) #Keras Model

#Compile and train the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]) #Compile the model

#Train the model
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2) #Train the model

#Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")









        
