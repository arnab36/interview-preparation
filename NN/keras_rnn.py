# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:22:27 2023

@author: 01927Z744


"""

import spacy
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
from keras.layers import Dense,LSTM,Embedding
from pickle import dump,load
from keras_preprocessing.sequence import pad_sequences

filepath = 'C:/Users/ArnabBiswas/Documents/Data/text-data/rnn-text-file.txt'

#%%

def read_file(filepath):
    with open(filepath,encoding="utf-8") as f:
        str_text = f.read()
    return str_text


def separate_punc(doc_text):
    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n']

def create_tokenizer(text_sequence):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_sequence)       
    return tokenizer

def create_sequence(tokenizer, text_sequence):
    sequences = tokenizer.texts_to_sequences(text_sequence)
    sequences = np.array(sequences)
    return sequences


def create_model(vocabulary_size,seq_len):
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(vocabulary_size,activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    print(model.summary())
    return model


''' Generate text  - The model will predict the index of the next word which we will get 
 from the tokenzer. So it is important to save the tokenizer and use the same tokenizer that we
 used while training the model or it will be completely useless or there will be error.'''

def generate_text(model,tokenizer, seq_len, seed_text, num_gen_words):
    
    output_text = []
    
    input_text = seed_text
    
    for i in range(num_gen_words):
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen = seq_len, truncating ='pre')
        prediction=model.predict(pad_encoded) 
        prediction_word_index=np.argmax(prediction,axis=1)[0]      
        pred_word = tokenizer.index_word[prediction_word_index]
        input_text += ' '+ pred_word
        output_text.append(pred_word)
        
    return ' '.join(output_text)


def predict_single_word(model,tokenizer, seq_len, seed_text):
    try:
        input_text = seed_text
        encoded_text = tokenizer.texts_to_sequences([input_text])[0]
        pad_encoded = pad_sequences([encoded_text], maxlen = seq_len, truncating ='pre')
        prediction=model.predict(pad_encoded) 
        prediction_word_index=np.argmax(prediction,axis=1)[0]      
        return tokenizer.index_word[prediction_word_index]
    except Exception as e:
        print(str(e))
        return None


#%%

nlp = spacy.load('en_core_web_sm', disable=['parser','tagger','ner'])

nlp.max_length = 1198624

d = read_file(filepath)

tokens = separate_punc(d)

#%%

train_len = 25+1

text_sequence = []

for i in range(train_len,len(tokens)):
    seq = tokens[i-train_len:i]
    text_sequence.append(seq)
    
#%%


# Sequence is a list where each element is an array with all but the last element 
# are the input features i.e rnn sequence input and the last element of the array is 
# the target word output.

tokenizer = create_tokenizer(text_sequence)
sequences = create_sequence(tokenizer, text_sequence)

X = sequences[:,:-1]
y = sequences[:,-1]

vocabulary_size = len(tokenizer.word_counts)
y = to_categorical(y, num_classes=vocabulary_size+1)
seq_len = X.shape[1]


#%%

model = create_model(vocabulary_size+1,seq_len)

model.fit(X,y,batch_size=128, epochs=100, verbose=1)

model.save('../../savedModels/rnn-text-gen-100.h5')

dump(tokenizer, open('../../savedModels/my_simpletokenizer','wb'))

#%%

# Load moel, tokenizer and predict/generate the text

model = load_model('../../savedModels/rnn-text-gen-100.h5')
tokenizer = load(open('../../savedModels/my_simpletokenizer','rb'))
#%%
# seed_text = 'for now he will rest at home in Kanchrapara. Then you can return to normal activities if the doctors advise. Mukul was admitted to the'
# op = generate_text(model,tokenizer, 25, seed_text, 10) 

# seed_text = 'cases has increased but the number of hospitalizations has not increased So experts feel that there is nothing to worry about the Covid situation right'   
# print(predict_single_word(model,tokenizer, 25, seed_text))






