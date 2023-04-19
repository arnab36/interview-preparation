# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:54:36 2023


@author: 01927Z744


@Data Path - C:/Users/ArnabBiswas/Documents/Python ML/Kaggle Dataset/cnn-data/
    
"""
 
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing import image
import keras.utils as image

path = 'C:/Users/ArnabBiswas/Documents/Python ML/Kaggle Dataset/cnn-data/Cat-Dog/'

modelName = "savedModels/model-cnn-cat-dog-1.h5"


#%%

def prepare_data():
    #As each pixel takes value from 0 to 255, we normalize by dividing all of them by 255
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, 
                                       horizontal_flip=True)
    
    training_set = train_datagen.flow_from_directory(path+'training_set', target_size=(64,64), batch_size=32,
                                                    class_mode='binary')    
    
    
    test_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, 
                                       horizontal_flip=True)    
    
    test_set = test_datagen.flow_from_directory(path+'test_set', target_size=(64,64), batch_size=32,
                                                    class_mode='binary')
    
    #Encode the result
    print(training_set.class_indices)
    
    return training_set, test_set

training_set, test_set = prepare_data()

#%%

def train_cnn_model(training_set, validation_set):
    
    #Initializing the CNN
    cnn = tf.keras.models.Sequential()
    
    #Convolution
    cnn.add(tf.keras.layers.Conv2D(filters=32, activation='relu', kernel_size=3,
                                   input_shape=[64,64,3]))
    #pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    
    #Adding a second convolution layer
    cnn.add(tf.keras.layers.Conv2D(filters=32, activation='relu', kernel_size=3))
    #pooling
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    #Flattening 
    cnn.add(tf.keras.layers.Flatten())
    
    # Add a fully connected layer
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    
    #Output layer
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    
    #Compiling the CNN
    cnn.compile(optimizer='adam',  loss='binary_crossentropy', metrics=['accuracy'])
    
    #Training the CNN
    cnn.fit(x= training_set, validation_data=validation_set, epochs=25)

    cnn.save(modelName)

train_cnn_model(training_set, test_set)

#%%

def load_model(modelName):
    cnn = tf.keras.models.load_model(modelName)
    return cnn

cnn = load_model(modelName)

#%%
def test_cnn_image(cnn,imgPath):    
    
    # Make as ingle prediction
    test_image = image.load_img(imgPath, target_size=(64,64))
    test_image = image.img_to_array(test_image)
    
    # Adding a fake/extra dimension
    test_image = np.expand_dims(test_image, axis=0)
    
    # Getting prediction 
    result = cnn.predict(test_image)
    
    prediction= ''
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
        
    print('Predicted class is = ' +prediction)

test_cnn_image(cnn, path+'single_prediction/cat_or_dog_1.jpg')

# for i in range(1,14):
#     test_cnn_image(cnn, path+'single_prediction/cat.' +str(2)+'.jpg')







