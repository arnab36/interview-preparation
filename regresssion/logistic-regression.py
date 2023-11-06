# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 07:18:36 2021

@author: 01927Z744

Logistci regression


"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import numpy as np
from sklearn.linear_model import LogisticRegression



filePath = "../../Data/Dataset/"
df = pd.read_csv(filePath+'framingham.csv')

cols = list(df)

# Filling out nan values with avg
for col in cols:
    df[col].fillna((df[col].mean()), inplace=True)

train, test = train_test_split(df, test_size=0.2)

train_x = train[train.columns[train.columns!='TenYearCHD'] ]
train_y = train["TenYearCHD"]

test_x = test[test.columns[test.columns!='TenYearCHD'] ]
test_y = test["TenYearCHD"]


model = LogisticRegression()
model.fit(train_x, train_y)

y_pred= model.predict(test_x)

df1 = pd.DataFrame({'Actual': test_y, 'Predicted': y_pred})

acc = accuracy_score(test_y,y_pred)


















