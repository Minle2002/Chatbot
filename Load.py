import tensorflow as tf
from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPooling1D, Bidirectional, Dropout
from keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import json
from word_process import WordProcess
import os
import numpy as np
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model_path = f"models/model_cnn_v1.weights.h5"
tokenizer_path = f"models/tokenizer_cnn_v1.pkl"
class_path = f"models/disease_classes_cnn_v1.txt"

sent_length=20
voc_size = 50000
embedding_vector_features=40
model_loaded = None
num_classes = 30

model_loaded = load_model("saved_model.keras")

print(model_loaded.summary())

model_loaded.load_weights(model_path)

# Preprocessing function
nltk.download('stopwords')

tokenizer = pickle.load(open(tokenizer_path, 'rb'))
stop_words = stopwords.words('english')
stemmer = PorterStemmer()
wp = WordProcess()



def preprocess_text(text):
    """Preprocesses a single text sample for disease prediction."""
    # voc_size = 5000
    sent_length = 20
    processed_text = wp.process_sent2sent(text)

    # One-hot encoding and padding
    # print(processed_text)
    onehot_vector = tokenizer.texts_to_sequences([processed_text])
    # print('vector',onehot_vector)
    padded_vector = pad_sequences(onehot_vector, padding='pre', maxlen=sent_length)

    return padded_vector[0].tolist()

with open(class_path) as f:
    disease_classes = json.load(f)
disease_classes, len(disease_classes)

test_cases = [
"I have been sneezing frequently, accompanied by a mild headache, runny nose, and a general feeling of being unwell.",
"Experiencing a low-grade fever with chills, nasal congestion, and a scratchy throat.",
"Mild body aches with a runny nose, a few sneezes, and feeling slightly fatigued.",
"Congested nose with a sore throat, slight cough, and sneezing fits.",
"I am experiencing itching and irritation in the vaginal area, along with a white, clumpy discharge that resembles cottage cheese.",
"There's a burning sensation during urination and redness and swelling of the vulva.",
"Feeling soreness and experiencing painful intercourse, accompanied by a thick, odorless, white vaginal discharge.",
"Persistent itching and a thick white discharge, with slight redness around the external genitalia.",
"Feeling tired all the time and my bones ache, especially in the joints and back. There's also muscle weakness.",
"Noticing more hair falling out, general fatigue, and aching bones. I've been indoors most of the time.",
"Experiencing bone pain and muscle weakness, feeling depressed more frequently.",
"My doctor mentioned bone softening, and I feel persistently low energy and down in mood.",
"My stomach cramps after eating and I frequently have diarrhea or constipation, feeling bloated.",
"Experiencing abdominal pain, bloating, and an inconsistent stool pattern, swinging between diarrhea and constipation.",
"Frequent bloating and gas with episodes of constipation followed by sudden diarrhea.",
"Abdominal discomfort, altered bowel habits, with bouts of diarrhea and periods of constipation, including bloating."
]

for test in test_cases:
    print(test)
    ind = model_loaded.predict(np.array([preprocess_text(test)]),verbose=0).argmax()
    print( disease_classes[ind])