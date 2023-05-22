# -*- coding: utf-8 -*-
"""
Created on Thu May 11 10:17:39 2023

@author: 01927Z744

"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

#%%

path = 'C:/Users/ArnabBiswas/Documents/Data/Dataset/'
filename = 'ChurnData.csv'


#%%

df = pd.read_csv(path+filename)
print(df.head())
# df = df.drop('Id',axis=1)

#%%

array = df.values
X = array[:,0:27]
y = array[:,27]

#%%

model = LogisticRegression()
rfe = RFE(model,n_features_to_select= 10)

fit = rfe.fit(X,y)
#%%

print(fit.n_features_)
print(fit.ranking_)
print(fit.support_)





