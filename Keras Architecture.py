"""
Created on Thu Mar 29 17:05:48 2018

KERAS, Multi-Layer Perceptron network
"""
'''
1. The combination of two new perceptrons is w1*0.4 + w2*0.6 + b. 
We'll explore the following values for the weights and the bias that would result in the final probability of the point to be 0.88:
'''
w1 = 2; w2 = 6; b = -2
w1 = 3; w2 = 5; b = -2.2
w1 = 5; w2 = 4; b = -3
x = w1*0.4 + w2*0.6 + b

import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
sigmoid(x)

# Neural Networks in Keras
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation

# X has shape (num_rows, num_cols), where the training data are stored
# as row vectors
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

# y must have an output vector for each input vector
y = np.array([[0], [0], [0], [1]], dtype=np.float32)

# Create the Sequential model
model = Sequential()

# 1st Layer - Add an input layer of 32 nodes with the same input shape as
# the training samples in X
model.add(Dense(32, input_dim=X.shape[1]))

# Add a softmax activation layer
model.add(Activation('softmax'))

# 2nd Layer - Add a fully connected output layer
model.add(Dense(1))

# Add a sigmoid activation layer
model.add(Activation('sigmoid'))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])

model.summary()

model.fit(X, y, nb_epoch=1000, verbose=0)
model.evaluate()

'''
2. We will build a simple multi-layer feedforward neural network to solve the XOR problem in the following way:

1) Set the first layer to a Dense() layer with an output width of 8 nodes and the input_dim set to the size of the training samples (in this case 2).
2) Add a tanh activation function.
3) Set the output layer width to 1, since the output has only two classes. We can use 0 for one class an 1 for the other
4) Use a sigmoid activation function after the output layer.
5) Run the model for 50 epochs.

With defaults we can get an accuracy of 50%. Out of 4 input points, we're correctly classifying only 2 of them. 
When we change some parameters around to improve, for example, increasing the number of epochs, we can get 75% accuracy or even 100%
'''
# Accuracy: 0.5
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
# Using TensorFlow 1.0.0; use tf.python_io in later versions
tf.python.control_flow_ops = tf

# Set random seed
np.random.seed(42)

# Our data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
# One-hot encoding the output
y = np_utils.to_categorical(y)

# Building the model
xor = Sequential()
# Add required layers
xor.add(Dense(32, input_dim=2))
xor.add(Activation("tanh"))
xor.add(Dense(2))
xor.add(Activation("sigmoid"))
# Specify loss as "binary_crossentropy", optimizer as "adam",
# and add the accuracy metric
xor.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy'])

xor.summary()

# Fitting the model
history = xor.fit(X, y, nb_epoch=50, verbose=0)

# Scoring the model
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

# Checking the predictions
print("\nPredictions:")
print(xor.predict_proba(X))

# Another SOLUTION: Accuracy: 0.75 but if binary_crossentropy Accuracy = 100%
import numpy as np
from keras.utils import np_utils
import tensorflow as tf
tf.python.control_flow_ops = tf

# Set random seed
np.random.seed(42)

# Our data
X = np.array([[0,0],[0,1],[1,0],[1,1]]).astype('float32')
y = np.array([[0],[1],[1],[0]]).astype('float32')

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten

# One-hot encoding the output
y = np_utils.to_categorical(y)

# Building the model
xor = Sequential()
xor.add(Dense(32, input_dim=2))
xor.add(Activation("sigmoid"))
xor.add(Dense(2))
xor.add(Activation("sigmoid"))

xor.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ['accuracy'])

xor.summary()

# Fitting the model
history = xor.fit(X, y, nb_epoch=1000, verbose=0)

# Scoring the model
score = xor.evaluate(X, y)
print("\nAccuracy: ", score[-1])

# Checking the predictions
print("\nPredictions:")
print(xor.predict_proba(X))

'''
Using TensorFlow backend.
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
dense_1 (Dense)                  (None, 32)            96          dense_input_1[0][0]              
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 32)            0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 2)             66          activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 2)             0           dense_2[0][0]                    
====================================================================================================
Total params: 162
Trainable params: 162
Non-trainable params: 0
____________________________________________________________________________________________________

4/4 [==============================] - 0s

Accuracy:  1.0

Predictions:

4/4 [==============================] - 0s
[[0.5658225  0.48416695]
 [0.44882667 0.5023724 ]
 [0.44079846 0.5049914 ]
 [0.54609394 0.4963648 ]]
'''


