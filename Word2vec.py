#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:34:40 2023

@author: myyntiimac
"""

import nltk
import gensim
from gensim.models import Word2Vec
from nltk.corpus import stopwords

import re

paragraph = """IIf the correlation score between a numerical variable and the target variable is 0, it means there is no linear relationship between them. In other words, there is no linear dependence or linear association between the numerical variable and the target variable. The variables are not correlated in a linear fashion.

However, it's important to note that a correlation score of 0 does not necessarily mean there is no relationship at all between the variables. There could still be other types of relationships, such as non-linear relationships or categorical dependencies, that are not captured by the correlation coefficient.

In such cases, it is recommended to explore other techniques or analysis methods to understand the relationship between the variables more comprehensively. This could include visualizing the data using scatter plots or applying other statistical techniques such as regression analysis or feature engineering to capture any non-linear or categorical relationships."""

# text Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]
    
    
    
# Training the Word2Vec model
model = Word2Vec(sentences, min_count=1)
#if the word is present < then 1 then use to skip the  conunt and as my data is very small 
#word2vec is applied for huge amount of data

# Finding Word Vectors
words = model.wv.index_to_key

# in this paragrapb if we want to find the vocalbulary & create a object called words
# if you select then each & every word there may be vectors and dimensions associated to it


# Finding Word Vectors
vector = model.wv['methods']
#if i want to find the vector of war word and if i want to find the relationship 
# Most similar words
similar = model.wv.most_similar('methods')
similar1 = model.wv.most_similar('techniques')