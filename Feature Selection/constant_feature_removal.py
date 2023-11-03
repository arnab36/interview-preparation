# -*- coding: utf-8 -*-
"""
Created on Thu May 25 14:04:42 2023

@author: 01927Z744

"""

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

# Adding a constant feature just for test
newFeature = np.ones(200, dtype = int)
df['newFeature'] = newFeature

#%%

X = df.drop(labels=['churn'], axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

#%%

vth = VarianceThreshold(threshold=0)

vth.fit(X_train)

# The indices that are False are constant features, the ones that are true are
# not constant features.
print(vth.get_support())


#%%

# Standard deviation technique - if standard deviation is 0 then the feature is 
# Constant.

constent_features = [
        feat for feat in X_train.columns if X_train[feat].std() == 0
    ]

print(constent_features)

# NB - TO remove Quasi constant features i.e the features with almost identical
# values to almost all the rows, you have to increase the threshold from 0 to 
# 0.01 or slightly more. You can play around it according to your choice. 







