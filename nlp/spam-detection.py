# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 08:11:19 2023

@author: 01927Z744

Feature extraction  - vectorization

"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer, TfidfVectorizer
from sklearn.svm import LinearSVC

#%%

datapath = 'C:/Users/ArnabBiswas/Documents/Data/UPDATED_NLP_COURSE/TextFiles/'
df = pd.read_csv(datapath+'smsspamcollection.tsv', sep = '\t')

#%%


print(df.head())
print(df.isnull().sum())
print(len(df))

print(df['label'].unique())
print(df['label'].value_counts())

#%%

X = df['message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

count_vect = CountVectorizer()

# Fit and transform together
X_train_count = count_vect.fit_transform(X_train)

tfidf_trans = TfidfTransformer()

X_train_tfidf = tfidf_trans.fit_transform(X_train_count)

#%%
# Vectorizer does the work of CountVectorizer & TfidfTransformer together.
vectorizer = TfidfVectorizer()

X_train_tfidf  = vectorizer.fit_transform(X_train) 

#%%

# Train the classifier

clf = LinearSVC()

clf.fit(X_train_tfidf, y_train)

#%%

# The whole above process can be done by pipeline in a single call

from sklearn.pipeline import Pipeline

text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])

text_clf.fit(X_train, y_train)

#%%

# Testing the classifier

from sklearn.metrics import classification_report,confusion_matrix

predictions = text_clf.predict(X_test)

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))






















