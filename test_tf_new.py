import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
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


# Load data
df = pd.read_csv('news-classification.csv')

# Split data into features and labels
texts = df['content']
labels = df['category_level_1']

texts = texts.apply(clean_text)

labels = pd.get_dummies(labels)

# Split data into training and testing sets
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2)

# Tokenize texts
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts_train)

# Convert texts to sequences
train_sequences = tokenizer.texts_to_sequences(texts_train)
test_sequences = tokenizer.texts_to_sequences(texts_test)

# Pad sequences
max_length = 100 # or any length you prefer
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')


# Create a sequential model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    # tf.keras.layers.Dense(1, activation='sigmoid')
    tf.keras.layers.Dense(len(labels.nunique()), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


num_epochs = 30
model.fit(train_padded, labels_train, epochs=num_epochs, validation_data=(test_padded, labels_test))


# Evaluate
loss, accuracy = model.evaluate(test_padded, labels_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Predict
predictions = model.predict(test_padded)
print(predictions)
