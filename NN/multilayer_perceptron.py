# -*- coding: utf-8 -*-
"""
Created on Mon May 22 13:37:48 2023

@author: 01927Z744

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


#%%

filepath = 'C:/Users/ArnabBiswas/Documents/Data/Kaggle Dataset/classification-data/'
filename = 'Churn_Modelling.csv'

df = pd.read_csv(filepath+filename)

X = df.drop(['Exited','RowNumber','CustomerId','Surname'],axis=1)
y = df['Exited']

#%%

# Create One hot encoding
enc = OneHotEncoder()
encoded = pd.DataFrame(enc.fit_transform(X[['Geography', 'Gender']]).toarray())

# Drop the categorical variables
X = X.drop(['Geography','Gender'],axis=1)

# Concat one hot encoding with X
X = pd.concat([X, encoded], axis=1)

#%%

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=101)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#%%

mlp = MLPClassifier(hidden_layer_sizes=(10,20,10),max_iter=300)

mlp.fit(X_train, y_train)

#%%

y_pred = mlp.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# TODO from sklearn.model_selection import GridSearchCV to search for best parameter





