'''

Model_Deployment_01

Author: Armin Berger
First created:  03/04/2022
Last edited:    03/04/2022

OVERVIEW:
This file seeks to deploy a news classification model.
Pre-processed text will be read in and used to train a TF-IDF vectorizer.
The model will be read in using a pickle file.
We then use the model to predict whether the user input is reliable or not.

'''

### READ IN ALL REQUIRED PACKAGES AND FILES

# import the required packages
import pickle
import os
import sklearn as sk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
import nltk
import collections
import spacy
import unicodedata

# suppress common warnings
import warnings
warnings.filterwarnings("ignore")

# import file which has all text pre processing functions
from text_pre_processing import *     # import classes and functions



# main block to check if the code is working the way you expect
if __name__ == '__main__':

    ### CREATE A TFIDF VECTORIZER

    # set current directory
    current_dir = os.getcwd()

    # load in the trained vectorizer
    filename = f'{current_dir}/vectorizer.pk'
    vectorizer = pickle.load(open(filename, 'rb'))

    ### LOAD IN MODEL AND GET USER INPUT

    # load the model from disk
    filename = f'{current_dir}/basic_news_logistic_regression.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    # get user input to for prediction
    user_news_input = input('Paste your text here: ')

    # ensure that the user input is in string format
    if isinstance(user_news_input, str):

        # use try and except statement to catch faulty input
        try:

            # pre-process the user input in the same manner as the training data
            user_news_input_processed = futher_process_string(user_news_input)
            user_news_input_processed = ' '.join(user_news_input_processed)
            user_news_input_vec = vectorizer.transform([user_news_input_processed])

            # prediction of our target
            prediction = loaded_model.predict(user_news_input_vec)

            if prediction[0] == 1:
                print('The news is likely to be UNRELIABLE!')

            else:
                print('The news is likely to be RELIABLE!')

        except:

            print('Problem')
