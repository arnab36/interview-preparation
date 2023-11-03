# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 09:05:46 2023

@author: 01927Z744


"""

import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing import image
import keras.utils as image

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve,classification_report
import matplotlib.pyplot as plt

path = 'C:/Users/ArnabBiswas/Documents/Python ML/Kaggle Dataset/cnn-data/nike-addidas/'

modelName = "savedModels/model-cnn-nike-addidas-1.h5"

#%%


def prepare_data():
    #As each pixel takes value from 0 to 255, we normalize by dividing all of them by 255
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, 
                                       horizontal_flip=True)
    
    training_set = train_datagen.flow_from_directory(path+'train', target_size=(64,64), batch_size=32,
                                                    class_mode='binary')    
    
    
    validation_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, 
                                       horizontal_flip=True)    
    
    validation_set = validation_datagen.flow_from_directory(path+'validation', target_size=(64,64), batch_size=32,
                                                    class_mode='binary')
    
    #Encode the result
    print(training_set.class_indices)
    
    return training_set, validation_set

training_set, validation_set = prepare_data()


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
    cnn.fit(x= training_set, validation_data=validation_set, epochs=50)

    cnn.save(modelName)

train_cnn_model(training_set, validation_set)


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
            
    return int(result[0][0])


#%%
def test_model():
    actual = []
    prediction = []
    
    for filename in os.listdir(path+'test/adidas/'):
        p = test_cnn_image(cnn,path+'test/adidas/'+str(filename))
        prediction.append(p)
        actual.append(0)
    
    for filename in os.listdir(path+'test/nike/'):
        p = test_cnn_image(cnn,path+'test/nike/'+str(filename))
        prediction.append(p)
        actual.append(1)
    
    print(confusion_matrix(actual, prediction))
    print(classification_report(actual, prediction))
    fpr, tpr, thresholds = roc_curve(actual, prediction)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    print("AUC score = {} ".format(roc_auc_score(actual, prediction)))
    
test_model()



