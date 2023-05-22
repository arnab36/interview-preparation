# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:44:32 2023

@author: 01927Z744

"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np

#%%

# Loading the data
(x_train, y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
#%%

# Checking the data
plt.imshow(x_train[0], cmap='gray')
print(x_train[0].shape)

#%%

print(x_train[0])

x_train = x_train/255
x_test = x_test/255

#%%

# Scaling pixels between 0 and 1
print(x_train[0])

#%%

# Creating the encoder

# The roiginal pics are 28x28
encoder_input = keras.Input(shape=(28,28,1), name='img')
x = keras.layers.Flatten() (encoder_input)

# This encoder_out is the image that we can feed in neural network as it contains
# less feature/information. You can change 64 to some other value (you have to change 
# line 89 accordingly)
encoder_out = keras.layers.Dense(64, activation='relu') (x)

encoder = keras.Model(encoder_input, encoder_out, name='encoder')


#%%

# 28x28 = 784
decoder_input = keras.layers.Dense(784, activation='relu') (encoder_out)

decoder_outout = keras.layers.Reshape((28,28,1))(decoder_input)

opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

#%%

# Auto encoder

auto_encoder = keras.Model(encoder_input, decoder_outout, name='auto_encoder')

print(auto_encoder.summary())

#%%

# Training the auto-encoder
auto_encoder.compile(opt, loss='mse')
auto_encoder.fit(x_train,x_train, epochs=6, batch_size=32, validation_split=0.1)

#%%

# The whole point of this excercise is to reduce the features of image to lower 
# value thus by making it calculation feasible for neural networks

# Predict the encoder
example = encoder.predict(x_test[4].reshape(-1,28,28,1))[0]
print(example)

# The shape will tell us that instead of 28x28 = 784 features we have now only 64 features
print(example.shape)

# Printing the encoded image - it will look blurr but it has most of the informations of the 
# original image
plt.imshow(example.reshape(8,8), cmap='gray')


#%%
# Let us check what the testing image actually contains
plt.imshow(x_test[4], cmap='gray')


#%%

# Let us validate that the encoder actually worked

# Use the auto encoder (which contains both the encoder and decoder) to predict/regenerate
# the test image. The generated otuput i.e ae_out is the result of first converting x_test
# to encoded image of size 64 i.e 8x8 and then decoding it back to 784 i.e 28x28. 
# During this procedure we are going to definitely loose some information but the whole 
# purpose of it is to reduce the image so that the encoded image can be put to neural network 
# to make it calculation efficient and decoding the whole thing back is just to check 
# if we get back an image that is closely related to the actual input image. Remember the encoded
# image had less information and that is the reason decoded image is not 100% same as the input image
# but close enough.

ae_out = auto_encoder.predict(x_test[4].reshape(-1,28,28,1))[0]
plt.imshow(ae_out, cmap='gray')


#%%

# Now let us add some noise in the pic and see how this auto_encoder can remove 
# the noise successfully.

import random

def add_noise(img, random_chance=5):
    noisy = []
    for row in img:
        new_row = []
        for pix in row:
            if random.choice(range(100)) <= random_chance :
                new_val = random.uniform(0,1)
                new_row.append(new_val)
            else:
                new_row.append(pix)
        noisy.append(new_row)
                
    return np.array(noisy)


#%%

noisy = add_noise(x_test[0])
plt.imshow(noisy, cmap='gray')

#%%

# Basically what we are doing is adding some noise in the pic. Now, our auto encoder 
# was trained on the clean images (unsupervised way), so when you pass a noisy image
# it will first encode the image down to 64 features and then decode it back to 
# 784 features. while ding that it will remove the noise because it will understand that
# the noise was not part of the training data and the image belongs to one of the 10 
# different class of digits that we have already seen. As it does not recognize the noise 
# it will simply not encode the noise and/or not decode the noise and thus we get the 
# output without noise.

ae_out = auto_encoder.predict(noisy.reshape(-1,28,28,1))[0]
plt.imshow(ae_out, cmap='gray')









