# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:08:04 2023

@author: 01927Z744

"""

import pandas as pd
import numpy as np
import seaborn as sns


#%%

filepath = 'C:/Users/ArnabBiswas/Documents/Data/Dataset/'
filename = 'sample_dataset.csv'

df = pd.read_csv(filepath+filename)

#%%

# Summarization

print(df.head(10))
print(df.dtypes)
print(df['area error'].unique())
print(df.info())

#%%

print(df.isna())
print(df.isna().sum())
print(df.isna().sum()/df.shape[0])

#%%

print(df.describe())

#%%

print(df.describe(percentiles=[0.05,0.1,0.9]))

#%%

# Visualization

print((df.isna().sum()/df.shape[0]).sort_values().plot(kind='bar'))

#%%

print(df.hist())

#%%

# Printing histogram for only first 4 features
print(df.iloc[:,0:4].hist(bins='rice'))

#%%

print(df.iloc[:,0:2].plot(kind='bar'))















