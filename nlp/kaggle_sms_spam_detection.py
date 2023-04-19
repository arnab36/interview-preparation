# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 11:59:03 2023

@author: 01927Z744

Kaggle SMS  - Spam vs Ham

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report,confusion_matrix

#%%

datapath = 'C:/Users/ArnabBiswas/Documents/Data/Kaggle Dataset/classification-data/'
df = pd.read_csv(datapath+'spam.csv', encoding='ISO-8859-1')


#%%


print(df.head())
print(df.isnull().sum())
print(len(df))

print(df['v1'].unique())
print(df['v1'].value_counts())

#%%

X = df['v2']
y = df['v1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


#%%

text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])

text_clf.fit(X_train, y_train)

#%%

predictions = text_clf.predict(X_test)

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))





