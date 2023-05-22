# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 12:15:29 2023

@author: 01927Z744

Multivariate analysis is the analysis of relation between more than one variable

"""

import pandas as pd
import numpy as np
import seaborn as sns


#%%

filepath = 'C:/Users/ArnabBiswas/Documents/Data/Dataset/'
filename = 'sample_dataset.csv'

df = pd.read_csv(filepath+filename)

#%%

print(sns.pairplot(df.iloc[:,0:4]))

#%%

# Kernel density estimation
print(sns.pairplot(df.iloc[:,0:4], kind='kde'))

#%%

print(sns.pairplot(df.iloc[:,0:4], corner=True))


#%%

print(sns.pairplot(df, x_vars=['mean radius','mean texture'], 
                   y_vars=['mean radius','mean texture','mean area'], hue='target'))


#%%

print(df.corr())

#%%

print(sns.heatmap(df.corr()))

#%%

print(sns.histplot(df, x='mean radius', hue='target', multiple='stack'))

