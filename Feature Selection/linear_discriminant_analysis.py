# -*- coding: utf-8 -*-
"""
Created on Thu May 11 08:48:34 2023

@author: 01927Z744

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#%%

path = 'C:/Users/ArnabBiswas/Documents/Data/Kaggle Dataset/classification-data/'
filename = 'WineQT.csv'

#%%

df = pd.read_csv(path+filename)
print(df.head())

df = df.drop('Id',axis=1)
#%%

X = df.drop('quality', axis=1)
y = df['quality']

#%%

# Only 3 components
lda = LinearDiscriminantAnalysis(n_components=3)

X_r2 = lda.fit(X,y).transform(X)

print('variance explained :', lda.explained_variance_ratio_)



