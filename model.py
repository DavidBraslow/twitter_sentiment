import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer
import re
from pre_processing import data_prep, tweet_preprocessing

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.dummy import DummyClassifier

import pickle

raw_df = pd.read_csv("judge-1377884607_tweet_product_company.csv")
df = data_prep(raw_data)

tweet_corpus = df['tweet_text'].apply(tweet_preprocessing)

vec = CountVectorizer(min_df = 20)
X = vec.fit_transform(tweet_corpus)
y = df['sentiment']

filename = 'pickled_vectorizer.sav'
pickle.dump(vec, open(filename, 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=12345, test_size=0.2)
mnb = MultinomialNB()

final_model = mnb
final_model.fit(X_train, y_train)

filename = 'pickled_model.sav'
pickle.dump(final_model, open(filename, 'wb'))