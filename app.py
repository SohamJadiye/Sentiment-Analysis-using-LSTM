from keras.preprocessing.text import Tokenizer
from flask import Flask, render_template, flash, request, url_for, redirect, session
import numpy as np
import pandas as pd
import re
import os
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


from keras.models import load_model

IMAGE_FOLDER = os.path.join('static', 'img_pool')
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
model = load_model('model.h5')
df = pd.read_csv("IMDB Dataset.csv")
total_words = 1000
max_length = 120
sentences = df['review']
labels = df['sentiment'].map({'negative': 0, 'positive': 1})
X_train, X_test, y_train, y_test = train_test_split(np.array(sentences), np.array(labels), train_size=0.8, random_state=42)
tokenizer = Tokenizer(num_words=total_words,oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route('/sentiment_analysis_prediction', methods = ['POST'])
def sent_anly_prediction():
    
    if request.method=='POST':
        text = request.form['textInput']
        print(text)
        Sentiment = ''
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text=re.sub(strip_special_chars, "", text.lower())

        words = text.split()
        print(len(tokenizer.word_index))
        
        x_test = [[word_index.get(word, 1) for word in words if word_index.get(word,1)<1000]]
        x_test = pad_sequences(x_test, maxlen=120) # Should be same which you used for training data
        vector = np.array([x_test.flatten()])
       
        
        probability = model.predict(array([vector][0]))[0][0]
        Sentiment =' '
        if(probability<0.6):
            Sentiment ='Negative'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Sad_Emoji.png')
            print(probability)
        else:
            Sentiment ='Positive'
            img_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Smiling_Emoji.png')
            print(probability)
       
    return render_template('home.html', text=text,sentiment = Sentiment ,probability=probability,image=img_filename)


if __name__ == "__main__":
    app.run(debug=True)
