# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 23:53:45 2022

@author: 01927Z744

  This section is for matrix calculations.
  
  
"""


A = [[1,2,3],
     [4,5,6],
     [7,8,9]]


B = [[1,2,3],
     [4,5,6],
     [7,8,9]]

C = A+B
print(C)
D = [
     [0,0,0],
     [0,0,0],
     [0,0,0]
     ]

for i in range(0,len(A)):
    for j in range(0,len(A[0])):
        D[i][j] = A[i][j] + B[i][j]
        
print(D)


#%%

import numpy as np
 
A = np.array([[1,2,3],
     [4,5,6],
     [7,8,9]])

B = np.array([[1,2,3],
     [4,5,6],
     [7,8,9]])


C = A+B
print(C)

D = A * B
print(D)


E = A.dot(B)
print(E)

#%%


A = np.arange(4)
print(A)


B = np.arange(16).reshape(4,4)
print(B)



print(B.transpose())


print(B.trace())


print(np.linalg.det(B))
