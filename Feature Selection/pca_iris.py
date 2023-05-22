# -*- coding: utf-8 -*-
"""
Created on Mon May  8 11:56:11 2023

@author: 01927Z744

"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler,scale
from sklearn.decomposition import PCA

#%%

df = sns.load_dataset('iris')
print(df.head())

#%%

op = df['species']
df = df.drop('species', axis=1)

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

#%%

pca = PCA(n_components=2)

pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

#%%

print(scaled_data.shape)
# print(pca.components_)
print(pca.explained_variance_ratio_)




