# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 07:10:36 2023

@author: 01927Z744


"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


path = 'C:/Users/ArnabBiswas/Documents/Python ML/Part 3 - Recurrent Neural Networks/'

#%%

# Import the training dataset

df = pd.read_csv(path+'Google_Stock_Price_Train.csv')
training_set = df.iloc[:, 1:2].values


# Feature scaling
sc = MinMaxScaler(feature_range=(0,1))
df_scaled = sc.fit_transform(training_set)


#%%

# Building the RNN - 60 timestep and 1 output. [a:b, 0] from a to b-1 wihout including index b

x_train = []
y_train = []

for i in range(60, 1258):
    x_train.append(df_scaled[i-60:i, 0])
    y_train.append(df_scaled[i, 0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)


# Reshape
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

#%%

# Building the RNN - as we are predicting a continuous value that is why we call it regressor

rnn_regressor = Sequential()

# Input layer
rnn_regressor.add(LSTM(units= 50, return_sequences=True, input_shape=(x_train.shape[1],1)))
rnn_regressor.add(Dropout(0.2))

# Hidden layers
rnn_regressor.add(LSTM(units= 50, return_sequences=True))
rnn_regressor.add(Dropout(0.2))

rnn_regressor.add(LSTM(units= 50, return_sequences=True))
rnn_regressor.add(Dropout(0.2))

# No return from fourth layer
rnn_regressor.add(LSTM(units= 50))
rnn_regressor.add(Dropout(0.2))

# Output Layer
rnn_regressor.add(Dense(units=1))

#Compile
rnn_regressor.compile(optimizer='adam',  loss='mean_squared_error')

# Train
rnn_regressor.fit(x_train, y_train, epochs = 100, batch_size = 32)


#%%












