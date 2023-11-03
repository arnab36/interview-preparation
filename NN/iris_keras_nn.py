# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 10:59:18 2023

@author: 01927Z744

The following will classify iris dataset with basic Keras Deep NN

"""

import numpy as np
from sklearn.datasets import load_iris
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score


#%%
# Load Data

iris = load_iris()
print(iris.DESCR)

X = iris.data
y = iris.target

print(X)
print(y)
#%%

# Transform Data

y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = MinMaxScaler()
scaler.fit(X_train)
scaled_X_train = scaler.transform(X_train)
scaled_X_test = scaler.transform(X_test)


#%%

# Create & Train Model

model = Sequential()
model.add(Dense(8, input_dim = 4, activation='relu'))
model.add(Dense(8, input_dim = 8, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, epochs=200, verbose=2)

#%%

#Prediction

predictions = model.predict(scaled_X_test)
predictions = predictions.argmax(axis=1)
y_test = y_test.argmax(axis=1)
#%%


print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

print(accuracy_score(y_test, predictions))








