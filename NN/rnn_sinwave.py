# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:53:23 2023

@author: 01927Z744

Needed error correction, never write model from scratch, use library

"""

import numpy as np
import math
import matplotlib.pyplot as plt

# Generating Data as sinewave

num_records = 200
seq_len = 50

sin_wave = np.array([math.sin(x) for x in np.arange(400)])

plt.plot(sin_wave[:50])

X = []
Y = []
for i in range(0,num_records - 50):
    X.append(sin_wave[i:i+seq_len])
    Y.append(sin_wave[i+seq_len])
    
X = np.array(X)
X = np.expand_dims(X, axis=2)

Y = np.array(Y)
Y = np.expand_dims(Y, axis=1)

#%%

X_val = []
Y_val = []

for i in range(num_records - 50, num_records):
    X_val.append(sin_wave[i:i+seq_len])
    Y_val.append(sin_wave[i+seq_len])


X_val = np.array(X_val)
X_val = np.expand_dims(X_val, axis=2)

Y_val = np.array(Y_val)
Y_val = np.expand_dims(Y_val, axis=1)

#%%

"""
Here,

U is the weight matrix for weights between input and hidden layers
V is the weight matrix for weights between hidden and output layers
W is the weight matrix for shared weights in the RNN layer (hidden layer)
Finally, we will define the activation function, sigmoid, to be used in the hidden layer:
    
"""


learning_rate = 0.0001    
nepoch = 25   
# length of sequence            
T = 50                   
hidden_dim = 100         
output_dim = 1

bptt_truncate = 5
min_clip_value = -10
max_clip_value = 10
# We will then define the weights of the network:

U = np.random.uniform(0, 1, (hidden_dim, T))
W = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
V = np.random.uniform(0, 1, (output_dim, hidden_dim))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
#%%


"""
Step 2: Train the Model
Now that we have defined our model, we can finally move on with training it on our sequence data. We can subdivide the training process into smaller steps, namely:

Step 2.1 : Check the loss on training data
Step 2.1.1 : Forward Pass
Step 2.1.2 : Calculate Error
Step 2.2 : Check the loss on validation data
Step 2.2.1 : Forward Pass
Step 2.2.2 : Calculate Error
Step 2.3 : Start actual training
Step 2.3.1 : Forward Pass
Step 2.3.2 : Backpropagate Error
Step 2.3.3 : Update weights

We need to repeat these steps until convergence. If the model starts to overfit, stop! 
Or simply pre-define the number of epochs.

"""


#%%

"""
Step 2.1: Check the loss on training data
We will do a forward pass through our RNN model and calculate the squared error for the predictions for all records in order to get the loss value.
"""


def training_loss():
    for epoch in range(nepoch):
        # check loss on train
        loss = 0.0
        
        # do a forward pass to get prediction
        for i in range(Y.shape[0]):
            # get input, output values of each record
            x, y = X[i], Y[i]   
    # here, prev-s is the value of the previous activation of hidden layer; 
    # which is initialized as all zeroes                
            prev_s = np.zeros((hidden_dim, 1))   
            for t in range(T):
                # we then do a forward pass for every timestep in the sequence
                new_input = np.zeros(x.shape)   
                # for this, we define a single input for that timestep
                new_input[t] = x[t]             
                mulu = np.dot(U, new_input)
                mulw = np.dot(W, prev_s)
                add = mulw + mulu
                s = sigmoid(add)
                mulv = np.dot(V, s)
                prev_s = s
    
        # calculate error 
            loss_per_record = (y - mulv)**2 / 2
            loss += loss_per_record
        loss = loss / float(y.shape[0])
        validation_loss(epoch,loss)
        
    
    
#%%

"""
Step 2.2: Check the loss on validation data
We will do the same thing for calculating the loss on validation data (in the same loop):
"""

# check loss on val

def validation_loss(epoch,loss):
    val_loss = 0.0
    for i in range(Y_val.shape[0]):
        x, y = X_val[i], Y_val[i]
        prev_s = np.zeros((hidden_dim, 1))
        for t in range(T):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            mulu = np.dot(U, new_input)
            mulw = np.dot(W, prev_s)
            add = mulw + mulu
            s = sigmoid(add)
            mulv = np.dot(V, s)
            prev_s = s
    
        loss_per_record = (y - mulv)**2 / 2
        val_loss += loss_per_record
    val_loss = val_loss / float(y.shape[0])
    print('Epoch: ', epoch + 1, ', Loss: ', loss, ', Val Loss: ', val_loss)



#%%

"""
Step 2.3: Start actual training
We will now start with the actual training of the network. In this, we will first do a 
forward pass to calculate the errors and a backward pass to calculate the gradients and 
update them. Let me show you these step-by-step so you can visualize how it works in your mind.


Step 2.3.1: Forward Pass
In the forward pass:

We first multiply the input with the weights between input and hidden layers
Add this with the multiplication of weights in the RNN layer. This is because we want to 
capture the knowledge of the previous timestep. Pass it through a sigmoid activation function
Multiply this with the weights between hidden and output layers.
At the output layer, we have a linear activation of the values so we do not explicitly 
pass the value through an activation layer. Save the state at the current layer and also 
the state at the previous timestep in a dictionary Here is the code for doing a forward 
pass (note that it is in continuation of the above loop):


"""


# train model

def train_model():
    for i in range(Y.shape[0]):
        x, y = X[i], Y[i]
    
        layers = []
        prev_s = np.zeros((hidden_dim, 1))
        dU = np.zeros(U.shape)
        dV = np.zeros(V.shape)
        dW = np.zeros(W.shape)
        
        dU_t = np.zeros(U.shape)
        dV_t = np.zeros(V.shape)
        dW_t = np.zeros(W.shape)
        
        dU_i = np.zeros(U.shape)
        dW_i = np.zeros(W.shape)
        
        # forward pass
        for t in range(T):
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            mulu = np.dot(U, new_input)
            mulw = np.dot(W, prev_s)
            add = mulw + mulu
            s = sigmoid(add)
            mulv = np.dot(V, s)
            layers.append({'s':s, 'prev_s':prev_s})
            prev_s = s
            back_propagation(x,y,layers,add,mulv,mulw,mulu,dW_t,dU_t,dV,dU,dW,U,V,W)


#%%


# Back-propagation


def back_propagation(x,y,layers,add,mulv,mulw,mulu,dW_t,dU_t,dV,dU,dW,U,V,W):
    # derivative of pred
    dmulv = (mulv - y)
    
    # backward pass
    for t in range(T):
        dV_t = np.dot(dmulv, np.transpose(layers[t]['s']))
        dsv = np.dot(np.transpose(V), dmulv)
    
        ds = dsv
        dadd = add * (1 - add) * ds
        
        dmulw = dadd * np.ones_like(mulw)
        
        dprev_s = np.dot(np.transpose(W), dmulw)
        
        
        for i in range(t-1, max(-1, t-bptt_truncate-1), -1):
            ds = dsv + dprev_s
            dadd = add * (1 - add) * ds
        
            dmulw = dadd * np.ones_like(mulw)
            dmulu = dadd * np.ones_like(mulu)
        
            dW_i = np.dot(W, layers[t]['prev_s'])
            dprev_s = np.dot(np.transpose(W), dmulw)
        
            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            dU_i = np.dot(U, new_input)
            dx = np.dot(np.transpose(U), dmulu)
        
            dU_t += dU_i
            dW_t += dW_i
            
        dV += dV_t
        dU += dU_t
        dW += dW_t


    #Update weights

    if dU.max() > max_clip_value:
        dU[dU > max_clip_value] = max_clip_value
    if dV.max() > max_clip_value:
        dV[dV > max_clip_value] = max_clip_value
    if dW.max() > max_clip_value:
        dW[dW > max_clip_value] = max_clip_value
    

    if dU.min() < min_clip_value:
        dU[dU < min_clip_value] = min_clip_value
    if dV.min() < min_clip_value:
        dV[dV < min_clip_value] = min_clip_value
    if dW.min() < min_clip_value:
        dW[dW < min_clip_value] = min_clip_value
            
    # update
    U -= learning_rate * dU
    V -= learning_rate * dV
    W -= learning_rate * dW


#%%
training_loss()

train_model()



