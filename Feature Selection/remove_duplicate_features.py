# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:35:21 2023

@author: 01927Z744

"""
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

#%%

path = 'C:/Users/ArnabBiswas/Documents/Data/Dataset/'
filename = 'ChurnData.csv'

df = pd.read_csv(path+filename)
print(df.head())

#%%

# Adding duplicate features just for test
words = ['apple','orange','banana','cucumber','pomogrenete','grape']
newFeature = []
for i in range(0,200):
    newFeature.append(random.choice(words))

df['newFeature1'] = newFeature
df['newFeature2'] = newFeature
df['newFeature3'] = newFeature

#%%



