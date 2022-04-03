#!/usr/bin/env python
# coding: utf-8

# # News Classification Model

# ### Step 1. Read in the data

# In[120]:


# install all packages needed

#!pip install gensim
#!pip install pyLDAvis
#!pip3 install openpyxl --upgrade
#!pip3 install sklearn --upgrade


# In[121]:


# import all libraries needed

import pandas as pd
from pandas import read_csv
import os
import openpyxl
#import jupyter_resource_usage
import re
import math
import collections
import spacy
import de_core_news_sm
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import langid

# nltk used for parsing and cleaning text
import nltk
import unicodedata
import string
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from difflib import SequenceMatcher
from scipy import spatial
from itertools import combinations

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import Phrases
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import pyLDAvis
import pyLDAvis.gensim_models


import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.linear_model import LogisticRegression

import pickle

## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K

import warnings
warnings.filterwarnings('ignore')


# In[3]:


current_dir = os.getcwd()
current_dir 


# Data overview:
# 
# train.csv: A full training dataset with the following attributes:
# 
# - id: unique id for a news article
# - title: the title of a news article
# - author: author of the news article
# - text: the text of the article; could be incomplete
# - label: a label that marks the article as potentially unreliable
# 
# 1: unreliable
# 0: reliable
# 
# test.csv: A testing training dataset with all the same attributes at train.csv without the label.
# 
# submit.csv: A sample submission that you can

# In[4]:


# read in all three data sets

train = pd.read_csv(f'{current_dir }/data_fake_news/kaggle_news_dataset/train.csv')

test = pd.read_csv(f'{current_dir }/data_fake_news/kaggle_news_dataset/test.csv')

submit = pd.read_csv(f'{current_dir }/data_fake_news/kaggle_news_dataset/submit.csv')


# In[5]:


train.info()


# In[6]:


test.info()


# In[7]:


train = train[['text','label']]
train


# In[8]:


train.text.is_unique


# In[9]:


train.label.value_counts()


# 1: unreliable
# 0: reliable

# Since not all values are unique we will only keep observations.

# In[10]:


train.drop_duplicates(inplace = True)


# In[11]:


train.shape


# In[12]:


# function which checks if text is not in english
def text_is_english(text):
    
    if isinstance(text, str):
        
        if langid.classify(text)[0]!='en':
 
            return True
        


# In[13]:


# check if tweet is not in english ans sa
train['bool_true'] = train.text.apply(lambda x: text_is_english(x))

# get a list of all indicese that need to be dropped
drop_index = train.index[train.bool_true == True].to_list()


# In[14]:


# drop the indicese 
train.drop(drop_index, inplace = True)


# In[15]:


# drop the last column
train.drop(columns = 'bool_true', inplace = True)


# In[16]:


train.shape


# In[ ]:





# ### Step 2. Preprocess the text data

# In[17]:


# save list of all stopwords
english_stop_words = stopwords.words('english')


# In[18]:


# function to remove unwanted characters from as string
def remove_char(text):
    
    # list of unwated charaters 
    remove_characters = ["'",';', ':', '!', "*", '/', '\\',"(", ")",",",".","-", "&","\n",'“','@', '–', '"', '+', '=', '[',']', '?', '”']
    
    # loop through all unwated characters 
    for character in remove_characters:
                         
        text = text.replace(character, " ")
                         
    return text


# In[19]:


# function that replaces accentuated characters by their non-accentuated counterparts
# in a string
def remove_accents(text):
    
    text = unicodedata.normalize('NFKD', text)
    
    return "".join([c for c in text if not unicodedata.combining(c)]) 


# In[20]:


# function to clean a string and turn it into a uniform format
# we can either keep numbers or remove them
def clean_string(text):
    
    text = str(text)
    
    text = text.lower()
        
    text = text.replace("'","")
    
    text = remove_char(text)

    text = text.strip(' ')
    
    text = remove_accents(text)

    return text
   


# In[21]:


news_list_unprocessed = train.text.to_list()


# In[22]:


## build a vocab for cleaned/standardized keyword token, then count them 

# set for all tokens 
news_tokens = []

# loop through each keyword name 
for i in news_list_unprocessed:
    
    # only edit strings 
    if type(i) == str:
    
        # split based on white spaces and create a list of tokens
        tokens = i.split(' ')

        # loop through all the tokens
        for x in tokens:

            # clean the string
            x = clean_string(x)
             
            # check if a token is a stopword or non
            if x not in english_stop_words and x is not None:
                
                # check if a token is a number or larger than 1
                if x.isnumeric() == False and len(x) > 1:

                    # append the cleaned string
                    news_tokens.append(x)

    
# print lenght
print(len(news_tokens))

# get the count of each token accross all documents
token_frequencey = FreqDist(news_tokens)


# In[23]:


token_frequencey


# In[105]:


# function which turns an uncleaned sting containing a number of tokens into a list of cleaned tokens 
def futher_process_string(text):
    
    # ensure that the text is in string format
    text = str(text)
    
    # don't keep numbers
    if text.isnumeric() == False:
        
        # split the string into individual tokens
        text = text.split(' ')
        
        # save a list of strings
        final_string_list = []
        
        # loop through all tokens
        for token in text:
            
            # clean the string
            token = clean_string(token)
            
            # don't keep numbers
            if token.isnumeric() == False and token not in english_stop_words:

                # ensure that the token is not None
                if  token is not None and token != '':
                    
                    # lemmatize the token 
                    token = WordNetLemmatizer().lemmatize(token)

                    # append the cleaned and lemmatized token
                    final_string_list.append(token)
        
        
        # return the the final string list
        return final_string_list
        


# In[79]:


get_ipython().run_cell_magic('time', '', '# preprocess all news articles\nnews_list_processed = [futher_process_string(x)  for x in news_list_unprocessed]')


# In[ ]:





# ### Step 3. Vectorize text

# In[80]:


vectorizer = TfidfVectorizer(analyzer = 'word',
                             input = 'content',
                            lowercase = True,
                            token_pattern = '(?u)\\b\\w\\w+\\b',
                            min_df = 3,
                            ngram_range = (1,2))


# In[81]:


# combine all the strings into one string 
x_train_text = [' '.join(x) for x in news_list_processed]


# In[82]:


# statistical model used in this assignement
model = LogisticRegression()


# In[83]:


# vectorize the training text input 
x_train = vectorizer.fit_transform(x_train_text)


# In[84]:


# save the target data as a list
y_train = train.label.to_list()


# In[85]:


# fit the LogisticRegression using our training data
model = model.fit(x_train, y_train)


# In[119]:


# save the model to disk
filename = 'basic_news_logistic_regression.sav'
pickle.dump(model, open(filename, 'wb'))


# ### Step 4. Predict the news

# In[116]:


user_news_input = input('Paste your text here: ')
user_news_input_processed = futher_process_string(user_news_input)
user_news_input_processed = ' '.join(user_news_input_processed)
user_news_input_vec = vectorizer.transform([user_news_input_processed])


# In[117]:


# prediction of our target
prediction = model.predict(user_news_input_vec)

if prediction[0] == 1:
    print('The news is likely to be UNRELIABLE!')
else:
    print('The news is likely to be RELIABLE!')


# In[ ]:





# In[ ]:




