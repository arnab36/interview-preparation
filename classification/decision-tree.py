# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 06:53:30 2021

@author: 01927Z744

 Decision Tree
 
"""

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def change_to_numerical(colName,df):
    a = df[colName]
    s = set(a)
    size =  len(s)
    l = []
    d = dict()
    count = 0
    for val in s:
        d[val] = count
        count += 1
    # df[colName] = df[colName].apply(d)
    df[colName] = df[colName].map(d)
    return df

filePath = "Dataset/"
df = pd.read_csv(filePath+'car_evaluation.csv')

cols = list(df)
for col in cols:
    df = change_to_numerical(col,df)

train, test = train_test_split(df, test_size=0.2)

train_x = train[train.columns[train.columns!='ACCEPTABILITY'] ]
train_y = train["ACCEPTABILITY"]

test_x = test[test.columns[test.columns!='ACCEPTABILITY'] ]
test_y = test["ACCEPTABILITY"]




# Function to perform training with giniIndex.
def train_using_gini(X_train, y_train):
  
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
  
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini
      
# Function to perform training with entropy.
def tarin_using_entropy(X_train, y_train):
  
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100,
            max_depth = 3, min_samples_leaf = 5)
  
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
  
  
# Function to make predictions
def prediction(X_test, clf_object):
  
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred
      
# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
      
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
      
    print("Report : ",
    classification_report(y_test, y_pred))
    
    
model = train_using_gini(train_x, train_y)

y_pred = prediction(test_x, model)

acc = cal_accuracy(test_y,y_pred)






    