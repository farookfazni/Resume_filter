import pickle
import joblib
import torch
import numpy as np
import tensorflow as tf
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the model from the pickle file
# filename = 'F:/CVFilter/models/model_pk.pkl'
# with open(filename, 'rb') as file:
#     model = pickle.load(file)

# Load the saved model
# model = joblib.load('F:\CVFilter\models\model.joblib')

# Define the model name
model_name = "fazni/distilbert-base-uncased-career-path-prediction"


# Load the model
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Local model
# model = tf.keras.models.load_model('models\model.h5')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Local Tokenizer
# tokenfile = 'tokenized_words/tokenized_words.pkl'
# # Load the tokenized words from the pickle file
# with open(tokenfile, 'rb') as file:
#     loaded_tokenized_words = pickle.load(file)

# max_review_length = 200
# tokenizer = Tokenizer(num_words=10000,  #max no. of unique words to keep
#                       filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', 
#                       lower=True #convert to lower case
#                      )
# tokenizer.fit_on_texts(loaded_tokenized_words)

outcome_labels = ['Business Analyst', 'Cyber Security','Data Engineer','Data Science','DevOps','Machine Learning Engineer','Mobile App Developer','Network Engineer','Quality Assurance','Software Engineer']

def model_prediction(text, model=model, tokenizer=tokenizer, labels=outcome_labels):
    # for local model
    # seq = tokenizer.texts_to_sequences([text])
    # padded = pad_sequences(seq, maxlen=max_review_length)
    # pred = model.predict(padded)

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    # Get the predicted class probabilities
    probs = outputs.logits.softmax(dim=-1)

    # print("Probability distribution: ", pred)
    # print("Field ")
    return labels[torch.argmax(probs)]