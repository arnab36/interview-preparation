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

from sklearn.metrics import confusion_matrix, accuracy_score

#%%

''' Data Loading and preprocessing '''

df = pd.read_csv("Churn_Modelling.csv")

x = df.iloc[:,3:-1].values
y = df.iloc[:,-1].values

le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])


transformer = ColumnTransformer( transformers=[('encoder',OneHotEncoder(),[1])], remainder='passthrough')
x = np.array(transformer.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=101)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)



#%%


''' 
    Building ANN
    
    relu - Rectifier Linear Unit
    for 2 op classes -> sigmoid, for more classes -> softmax
    for 2 op classes -> binary_crossentropy, for more classes -> categorical_crossentropy
    
'''


ann = tf.keras.Sequential()

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(x_train, y_train, batch_size = 32, epochs = 25)


ann.save("model1.h5")


#%%


'''  Predcition and Evaluations  '''


y_pred = ann.predict(x_test)

y_pred = [1 if x > 0.5 else 0 for x in y_pred]

cm = confusion_matrix(y_test, y_pred)

print(accuracy_score(y_test, y_pred))








