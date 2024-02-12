# We first have to import every necessary library
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.stem import LancasterStemmer
USE_TF = True
if USE_TF:
  from tensorflow.keras.layers import Dropout
  from tensorflow.keras.callbacks import EarlyStopping
  from tensorflow.keras.regularizers import l2
  from tensorflow.keras.utils import to_categorical
  from tensorflow.keras.callbacks import EarlyStopping
  from tensorflow.keras.preprocessing.text import Tokenizer
  from tensorflow.keras.preprocessing.sequence import pad_sequences
else:
  from keras.layers import Dropout
  from keras.callbacks import EarlyStopping
  from keras.regularizers import l2
  from keras.utils import to_categorical
  from keras.callbacks import EarlyStopping
  from keras.preprocessing.text import Tokenizer
  from keras.preprocessing.sequence import pad_sequences

# Then, we define all our custom functions
  
# Clean text
def clean_text(text):
    text = text.lower()
    text = text.replace('\xa0', ' ') # Remove non-breaking spaces
    text = re.sub(r'http\S+|www.\S+', '', text) # Remove URLs
    text = re.sub(r'\S+@\S+', '', text) # Remove email addresses
    text = re.sub(r'^.*\b(this|post|published|site)\b.*$\n?', '', text, flags=re.MULTILINE) # Remove lines like "This post was published on the site"
    text = re.sub(r'\\(?!n|r)', '', text) # Remove anything but backslashes
    text = text.replace('[\r \n]\n', ' ') # Remove newlines
    text = re.sub(r'[\r\n]{2,}', ' ', text)
    text = re.sub(r'from[: ]* ', '', text) # Remove "from" at the beginning of the text
    text = re.sub(r'  ', ' ', text) # Remove double spaces
    text = re.sub(r'\(photo by .*\)', '', text) # Remove lines like "(photo by reuters)"
    return text

#  Import text
def import_text():
  df = pd.read_csv('drive/MyDrive/news-classification.csv')
  texts = df['content']
  labels = df['category_level_1']
  labels = pd.get_dummies(labels)
  return texts, labels

def preprocess_data(texts, labels, num_words = 20000, test_size = 0.2):
    # Split data into training and testing sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=test_size)
    
    # Tokenize texts
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    
    # Convert all texts to sequences
    all_sequences = tokenizer.texts_to_sequences(texts)
    
    # Get the length of the longest sequence
    max_sequence_length = max(len(sequence) for sequence in all_sequences)

    # Convert texts to sequences
    train_sequences = tokenizer.texts_to_sequences(train_texts)
    test_sequences = tokenizer.texts_to_sequences(test_texts)

    # Pad sequences
    max_length = max_sequence_length # Set the max length to the length of the longest sequence
    train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
    test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

    # Convert labels to one-hot encoding
    encoder = LabelEncoder()
    encoder.fit(labels)
    train_labels = to_categorical(encoder.transform(train_labels), num_classes=labels.nunique())
    test_labels  = to_categorical(encoder.transform(test_labels), num_classes=labels.nunique())

    return train_padded, test_padded, train_labels, test_labels, tokenizer, max_length

def train(model, train_texts, train_labels, test_texts, test_labels, epochs=30, batch_size=128):
	early_stopping = EarlyStopping(monitor='val_loss', patience=12)
	model.fit(train_texts, train_labels, validation_data=(test_texts, test_labels), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping], use_multiprocessing=True)

def evaluate_predict(model, test_texts, test_labels):
	loss, accuracy = model.evaluate(test_texts, test_labels)
	print(f'Loss: {loss}, Accuracy: {accuracy}')

	predictions = model.predict(test_texts)

	predicted_labels = np.argmax(predictions, axis=1)
	actual_labels = np.argmax(test_labels, axis=1)
	df = pd.DataFrame({
		'Predicted': predicted_labels,
		'Actual': actual_labels
	})
	print(df)

