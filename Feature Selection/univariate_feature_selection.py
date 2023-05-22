# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:03:33 2023

@author: 01927Z744


"""
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_classif,mutual_info_classif

#%%

path = 'C:/Users/ArnabBiswas/Documents/Data/Kaggle Dataset/classification-data/'
filename = 'WineQT.csv'


#%%

df = pd.read_csv(path+filename)
print(df.head())

df = df.drop('Id',axis=1)
array = df.values
#%%

X = array[:,0:11]
y = array[:,11]


#%%

# test = SelectKBest(score_func=chi2, k=4)
test = SelectKBest(score_func=mutual_info_classif, k=4)
fit = test.fit(X, y)

features = fit.transform(X)

np.set_printoptions(precision=1)
a = fit.scores_
for i in a:
    print(i)


#%%



