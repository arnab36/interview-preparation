# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 12:25:32 2023

@author: 01927Z744

"""

import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')

#%%

doc = nlp(u'I am Arnab Biswas and I am not cool.')

for token in doc:
    print(token.text, token.pos_, token.tag_)
    
# displacy.render(doc, style='dep')

#%%

import nltk
from nltk.stem.porter import PorterStemmer

#%%

p_stemmer = PorterStemmer()

words = ['run', 'runs', 'runner', 'running', 'ran']

for w in words:
    print(w + ' => \t '+ p_stemmer.stem(w))
    
#%%

from nltk.stem.snowball import SnowballStemmer
s_stemmer = SnowballStemmer(language='english')


for w in words:
    print(w + ' => \t '+ s_stemmer.stem(w))

#%%






  
    
    