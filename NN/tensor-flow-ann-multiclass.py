# -*- coding: utf-8 -*-

"""
Created on Mon Jun  6 07:29:05 2022

@author: 01927Z744

In case of error download the dll file manually and put that inside system32

Then for more error run  pip install --ignore-installed --upgrade tensorflow --user

"""

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.utils.np_utils import to_categorical

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import classification_report

from sklearn.metrics import multilabel_confusion_matrix

#%%

modelName = "savedModels/model-ann-wine-quality.h5"

#%%


def prepare_wine_data():
    
    path = "C:/Users/ArnabBiswas/Documents/Python ML/Kaggle Dataset/classification-data/"
    filename = "WineQT.csv"
    
    df = pd.read_csv(path+filename)     
    x = df.iloc[:,:-2].values
    y = df.iloc[:,-2].values    
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = np_utils.to_categorical(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
  
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    
    return x_train, x_test, y_train, y_test 

x_train, x_test, y_train, y_test = prepare_wine_data()

#%%
def train_model(x_train, y_train, no_epochs, batch_size, modelName):

    ann = tf.keras.Sequential()    
    ann.add(tf.keras.layers.Dense(units=20, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=30, activation='relu'))
    
    ann.add(tf.keras.layers.Dense(units=6, activation='softmax'))    
    
    ann.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
    ann.fit(x_train, y_train, batch_size = batch_size, epochs = no_epochs)    
    
    ann.save(modelName)
    
train_model(x_train, y_train, 10, 32, modelName)


#%%

''' Need more corrctions. '''
def prediction(x_test, y_test, modelName):
    
    ann = tf.keras.models.load_model(modelName)
    
    y_prob = ann.predict(x_test)
    
    y_pred = np.zeros_like(y_prob)
    
    y_pred[np.arange(len(y_prob)), y_prob.argmax(1)] = 1
    
    print(classification_report(y_test, y_pred, digits=10))
    
       
    # cm = multilabel_confusion_matrix(y_test, y_pred)
    
    # print("Accuracy score = {} ".format(accuracy_score(y_test, y_pred)))
    
    # print("AUC score = {} ".format(roc_auc_score(y_test, y_pred)))
    
    # print("\n ------------------------------ \n")
    
    # print("Confusion Matrix \n {} ".format(cm))
    
    # print(" \n --------------ROC Curve ---------------- \n")
    
    # fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    # plt.plot(fpr, tpr)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')



prediction(x_test, y_test, modelName)



#%%
y_test = [0,0,0,1,1,2,2,2,1,0,0,0,1,2,0,0,0,1,1,2,2,2,1,0,0,0,1,2]
y_pred = [0,0,0,1,1,2,2,2,1,0,0,0,1,2,2,1,0,1,1,2,1,1,1,0,0,0,1,2]


print(classification_report(y_test, y_pred, digits=3))







