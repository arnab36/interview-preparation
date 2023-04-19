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

#%%

modelName = "savedModels/model-ann-wine-quality.h5"

#%%

''' Data Loading and preprocessing only for churn data.  '''

def prepare_data_churn_model():  
    
    path = "C:/Users/ArnabBiswas/Documents/Python ML/Kaggle Dataset/classification-data/"
    filename = "Churn_Modelling.csv"
    
    df = pd.read_csv(path+filename)    
    x = df.iloc[:,3:-1].values
    y = df.iloc[:,-1].values    
    
    ''' 
        Label encoder is for binary categorical variable i.e the fetaure that 
        takes only two values. One hot encoding is for other categorical variables which has 
        more than 2 different values.
    '''
    le = LabelEncoder()
    x[:, 2] = le.fit_transform(x[:, 2])
    
    
    transformer = ColumnTransformer( transformers=[('encoder',OneHotEncoder(),[1])], remainder='passthrough')
    x = np.array(transformer.fit_transform(x))
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=101)
    
    
    ''' Standard scaler is for scaling feature data '''
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    return x_train, x_test, y_train, y_test


# x_train, x_test, y_train, y_test = prepare_data_churn_model()

#%%


'''

About this file
    The file contains the different attributes of customers' reservation details. 
    The detailed data dictionary is given below.

Data Dictionary

    Booking_ID: unique identifier of each booking
    no_of_adults: Number of adults
    no_of_children: Number of Children
    no_of_weekend_nights: Number of weekend nights (Saturday or Sunday) the guest stayed or booked to stay at the hotel
    no_of_week_nights: Number of week nights (Monday to Friday) the guest stayed or booked to stay at the hotel
    type_of_meal_plan: Type of meal plan booked by the customer:
    required_car_parking_space: Does the customer require a car parking space? (0 - No, 1- Yes)
    room_type_reserved: Type of room reserved by the customer. The values are ciphered (encoded) by INN Hotels.
    lead_time: Number of days between the date of booking and the arrival date
    arrival_year: Year of arrival date
    arrival_month: Month of arrival date
    arrival_date: Date of the month
    market_segment_type: Market segment designation.
    repeated_guest: Is the customer a repeated guest? (0 - No, 1- Yes)
    no_of_previous_cancellations: Number of previous bookings that were canceled by the customer prior to the current booking
    no_of_previous_bookings_not_canceled: Number of previous bookings not canceled by the customer prior to the current booking
    avg_price_per_room: Average price per day of the reservation; prices of the rooms are dynamic. (in euros)
    no_of_special_requests: Total number of special requests made by the customer (e.g. high floor, view from the room, etc)
    booking_status: Flag indicating if the booking was canceled or not.

'''


def prepare_hotel_reservation_data():
    
    
    path = "C:/Users/ArnabBiswas/Documents/Python ML/Kaggle Dataset/classification-data/"
    filename = "Hotel Reservations.csv"
    
    df = pd.read_csv(path+filename)   
   
    x = df.iloc[:,1:-1].values
    y = df.iloc[:,-1].values    
    
    transformer = ColumnTransformer( transformers=[('encoder',OneHotEncoder(),[4])], remainder='passthrough')
    x = np.array(transformer.fit_transform(x))
    
    transformer = ColumnTransformer( transformers=[('encoder',OneHotEncoder(),[9])], remainder='passthrough')
    x = np.array(transformer.fit_transform(x))
    
    transformer = ColumnTransformer( transformers=[('encoder',OneHotEncoder(),[20])], remainder='passthrough')
    x = np.array(transformer.fit_transform(x))
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
   
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    
    return x_train, x_test, y_train, y_test



# x_train, x_test, y_train, y_test = prepare_hotel_reservation_data()

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


''' 
    Building ANN
    
    relu - Rectifier Linear Unit
    for 2 op classes -> sigmoid, for more classes -> softmax
    for 2 op classes -> binary_crossentropy, for more classes -> categorical_crossentropy
    
'''

def train_model_binary(x_train, y_train, no_epochs, batch_size, modelName):

    ann = tf.keras.Sequential()    
    ann.add(tf.keras.layers.Dense(units=60, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=80, activation='relu'))
    
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))    
    
    ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    ann.fit(x_train, y_train, batch_size = batch_size, epochs = no_epochs)    
    
    ann.save(modelName)


# train_model_binary(x_train, y_train, 200, 32, modelName)


def train_model(x_train, y_train, no_epochs, batch_size, modelName):

    ann = tf.keras.Sequential()    
    ann.add(tf.keras.layers.Dense(units=20, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=30, activation='relu'))
    
    ann.add(tf.keras.layers.Dense(units=6, activation='softmax'))    
    
    ann.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    
    ann.fit(x_train, y_train, batch_size = batch_size, epochs = no_epochs)    
    
    ann.save(modelName)
    
train_model(x_train, y_train, 200, 32, modelName)


#%%


'''  Predcition and Evaluations  '''

def prediction_binary(x_test, y_test, modelName):

    ann = tf.keras.models.load_model(modelName)
     
    y_prob = ann.predict(x_test)
    
    y_pred = [1 if x > 0.5 else 0 for x in y_prob]
    
    cm = confusion_matrix(y_test, y_pred)
    
    print("Accuracy score = {} ".format(accuracy_score(y_test, y_pred)))
    
    print("AUC score = {} ".format(roc_auc_score(y_test, y_pred)))
    
    print("\n ------------------------------ \n")
    
    print("Confusion Matrix \n {} ".format(cm))
    
    print(" \n --------------ROC Curve ---------------- \n")
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


# prediction_binary(x_test, y_test, modelName)

#%%


def prediction(x_test, y_test, modelName):

    encoder = LabelEncoder()
    
    ann = tf.keras.models.load_model(modelName)
    
    predictions = ann.predict_classes(x_test)
    
    predictions = np.argmax(to_categorical(predictions), axis = 1)
    predictions = encoder.inverse_transform(predictions)
     
    cm = confusion_matrix(y_test, predictions)
    
    print("Accuracy score = {} ".format(accuracy_score(y_test, predictions)))
    
    print("AUC score = {} ".format(roc_auc_score(y_test, predictions)))
    
    print("\n ------------------------------ \n")
    
    print("Confusion Matrix \n {} ".format(cm))
    
    print(" \n --------------ROC Curve ---------------- \n")
    
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')


prediction_binary(x_test, y_test, modelName)











