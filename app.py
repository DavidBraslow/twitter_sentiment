# Adapted from https://towardsdatascience.com/how-to-easily-deploy-machine-learning-models-using-flask-b95af8fe34d4

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from pre_processing import tweet_preprocessing

app = Flask(__name__)
vectorizer = pickle.load(open('model/pickled_vectorizer.sav', 'rb'))
model = pickle.load(open('model/pickled_model.sav', 'rb'))

def label_output(output):
    if output == -1: return "Negative"
    elif output == 1: return "Positive"
    else: return "Neutral"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    data = request.form.values()
    raw_tweet = list(data)[0]
    prepped_tweet = tweet_preprocessing(raw_tweet)
    vectorized_tweet = vectorizer.transform([prepped_tweet])
    prediction = model.predict(vectorized_tweet)

    output = prediction[0]

    return render_template('index.html', prediction_text='Sentiment classified as {}'.format(label_output(output)))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    raw_tweet = list(data.values())[0]
    prepped_tweet = tweet_preprocessing(raw_tweet)
    vectorized_tweet = vectorizer.transform([prepped_tweet])
    prediction = model.predict(vectorized_tweet)

    output = prediction[0]
    return jsonify(tweet_sentiment=label_output(output))

if __name__ == "__main__":
    app.run(debug=True)