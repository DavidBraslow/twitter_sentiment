import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer
import re

nltk.download("stopwords")
nltk.download("wordnet")

SENTIMENT_VALUE_CODES =   {'No emotion toward brand or product': 0,
                            'Positive emotion': 1,
                            'Negative emotion': -1,
                            "I can't tell": np.nan}

DEFAULT_STOPS = stopwords.words('english') + ['sxsw']
DEFAULT_LEMMATIZER = nltk.stem.WordNetLemmatizer()

def data_prep(raw_df):
    '''
    Preprocess the CSV to ready it for text preprocessing
    '''
    df = raw_df.copy()
    df = df.rename({'is_there_an_emotion_directed_at_a_brand_or_product': 'sentiment'}, axis=1)
    df = df.drop('emotion_in_tweet_is_directed_at', axis=1)
    df['sentiment'] = df['sentiment'].map(SENTIMENT_VALUE_CODES)
    df.dropna(implace=True)
    df = df.astype({'sentiment': 'int8'})
    
    return df

def tweet_preprocessing(tweet, stops = DEFAULT_STOPS, lemmatizer = DEFAULT_LEMMATIZER):
    '''
    Preprocesses a tweet for sentiment analysis using a given list of stop words and lemmatizer
    '''
        
    # Remove case
    lower_tweet = tweet.lower()

    # Tokenize on spaces and apostrophes
    token_tweet = lower_tweet.replace("'", " ").split(" ")
    
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