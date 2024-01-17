import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
import datetime

import re

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

early_stopping = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

CSV_FILE = 'news-classification.csv'

df = pd.read_csv(CSV_FILE)
texts = df['content']
labels = df['category_level_1']

# Apply the function to the 'content' column
texts = texts.apply(clean_text)

# print(texts)
# exit()

# Tokenize the texts
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
# print(sequences[0])

# Pad the sequences
padded_sequences = pad_sequences(sequences, maxlen=200)
model = tf.keras.Sequential([
	tf.keras.layers.Embedding(10000, 16, input_length=200),
	tf.keras.layers.GlobalAveragePooling1D(),
	tf.keras.layers.Dense(300, activation='relu'),
	# tf.keras.layers.Dropout(0.5),
	tf.keras.layers.Dense(len(labels.unique()), activation='softmax')  # Adjust the number of neurons to match the number of unique labels
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Convert labels to numeric
label_tokenizer = Tokenizer()
labels_list = [[label] for label in labels]  # Make each label a sequence
label_tokenizer.fit_on_texts(labels_list)
training_label_seq = np.array([seq[0] for seq in label_tokenizer.texts_to_sequences(labels_list)]) - 1

# Train the model
model.fit(padded_sequences, training_label_seq, epochs=30, validation_split=0.2, callbacks=[tensorboard_callback])

# Evaluate the model
loss, accuracy = model.evaluate(padded_sequences, training_label_seq)
print(f'Loss: {loss}, Accuracy: {accuracy}')
