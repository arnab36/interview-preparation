# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 09:36:17 2021

@author: 01927Z744

Interview codes

"""

def hr_min():
    arr = [1,2,3,4,5]
    
    a = '12:05:45AM'
    
    b = a.split(':')
    
    s = b[2][0:2]
    
    t = b[2][2:4]
    
    hr = int(b[0])
    
    m = b[1]
    
    if t == 'PM':
        if hr != 12:
            hr = hr + 12
    else:
        if hr == 12:
            hr = 0
        
    if hr < 10:
        hr = '0' + str(hr)
    else:
        hr = str(hr)
        
    op = str(hr) +':' + m + ':'+s
    print(op)
    
    
    
    

#%%

'Find the permutation'

l = ['a','b','c','d','e']


def permute(l):
    if len(l) == 1:
        return [l]
    else:
        perm_list = []
        for a in l:            
            remaining_elements = [x for x in l if x != a]
            op = permute(remaining_elements)          
            for t in op:
                t.append(a)
                perm_list.append(t)
        return perm_list



perm_list = permute(l)




#%%




import numpy as np
from sklearn.linear_model import LinearRegression


model = LinearRegression()


x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])

model = LinearRegression()
model.fit(x, y)

r_sq = model.score(x, y)

y_pred = model.predict(x)

#%%

import matplotlib.pyplot as plt
import seaborn as sns


plt.scatter(x, y, color ='b')
plt.plot(x, y_pred, color ='k')
plt.show()

#%%

def species_to_numeric(sp):
    if sp == 'Bream':
        return 1
    elif sp == 'Roach':
        return 2
    elif sp == 'Whitefish':
        return 3
    elif sp == 'Parkki':
        return 4
    elif sp == 'Perch':
        return 5
    elif sp == 'Pike':
        return 6
    elif sp == 'Smelt':
        return 7
    else:
        return 0
    

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

filePath = "Dataset/"
df = pd.read_csv(filePath+'Fish.csv')

species = df.Weight.unique()

# df['Species'] = df['Species'].apply(species_to_numeric)

df = df[df.columns[df.columns!='Species'] ]

train, test = train_test_split(df, test_size=0.2)

train_x = train[train.columns[train.columns!='Weight'] ]
train_y = train["Weight"]

test_x = test[test.columns[test.columns!='Weight'] ]
test_y = test["Weight"]



model = LinearRegression()
model.fit(train_x, train_y)


y_pred= model.predict(test_x)

df1 = pd.DataFrame({'Actual': test_y, 'Predicted': y_pred})

mean_squared_error(test_y, y_pred)






