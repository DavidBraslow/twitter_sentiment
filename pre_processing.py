import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer
import re

nltk.download("stopwords")
nltk.download("wordnet")

def tweet_preprocessing(tweet, stops = stopwords.words('english'), lemmatizer = nltk.stem.WordNetLemmatizer()):
    '''
    Preprocesses a tweet for sentiment analysis using a given list of stop words and lemmatizer
    '''
        
    # Remove case
    lower_tweet = tweet.lower()

    # Tokenize on spaces and apostrophes
    token_tweet = lower_tweet.replace("'", " ").split(" ")
    
    # Define stopword list
    stops = stops + ['sxsw']
    
    # Define regex pattern
    pattern = re.compile('[^a-zA-Z]+') 
    
    # Create processed tweet
    proc_tweet = []
    
    for i in range(len(token_tweet)):
        clean_word = token_tweet[i]
        
        # Remove usernames
        if '@' in clean_word:
            clean_word = ""

        # Keep only characters
        clean_word = pattern.sub('', clean_word)

        # Remove stopwords and words 2 chars or less
        if (clean_word in stops) | (len(clean_word) <= 2):
            clean_word = ""
        
        # Lemmatize
        clean_word = lemmatizer.lemmatize(clean_word)
        
        # Replace original word with clean word if it's not empty
        if clean_word != "":
            proc_tweet.append(clean_word) 
    
    # Return string version of tweet
    return " ".join(proc_tweet)