import pickle
import joblib
import numpy as np
import tensorflow as tf
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

# Load the model from the pickle file
# filename = 'F:/CVFilter/models/model_pk.pkl'
# with open(filename, 'rb') as file:
#     model = pickle.load(file)

# Load the saved model
# model = joblib.load('F:\CVFilter\models\model.joblib')

model = tf.keras.models.load_model('models\model.h5')

tokenfile = 'tokenized_words/tokenized_words.pkl'
# Load the tokenized words from the pickle file
with open(tokenfile, 'rb') as file:
    loaded_tokenized_words = pickle.load(file)

max_review_length = 200
tokenizer = Tokenizer(num_words=10000,  #max no. of unique words to keep
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', 
                      lower=True #convert to lower case
                     )
tokenizer.fit_on_texts(loaded_tokenized_words)

outcome_labels = ['Business Analyst', 'Cyber Security','Data Engineer','Data Science','DevOps','Machine Learning Engineer','Mobile App Developer','Network Engineer','Quality Assurance','Software Engineer']

def model_prediction(text, model=model, tokenizer=tokenizer, labels=outcome_labels):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_review_length)
    pred = model.predict(padded)
    # print("Probability distribution: ", pred)
    # print("Field ")
    return labels[np.argmax(pred)]