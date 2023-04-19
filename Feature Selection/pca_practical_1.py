# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 11:43:29 2023

@author: 01927Z744

The following code is about PCA
 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
#%%

data = load_breast_cancer()

print(data.keys())
print(data['DESCR'])

#%%

df = pd.DataFrame(data['data'], columns = data['feature_names'])
print(df.head())
print(np.shape(df))

#%%

scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

#%%

# We can either choose the number of components
pca = PCA(n_components=2)

# or we can choose the % variance we need to keep
# pca = PCA(0.95)

pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)

print(scaled_data.shape)
print(x_pca.shape)

#%%

# print(pca.components_)
print(pca.explained_variance_)
#%%

# Now print it

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0], x_pca[:,1],c=data['target'])
plt.xlabel('1st Principal Component')
plt.ylabel('2nd Principal Component')
plt.show()


#%%

# Now check the composition/importance of features in PCA

df_comp = pd.DataFrame(pca.components_, columns = data['feature_names'])

sns.heatmap(df_comp, cmap='plasma')












