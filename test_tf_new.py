import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import keras

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
train_texts, test_texts, train_labels, test_labels = train_test_split(df['content'], df['category_level_1'], test_size=0.2)

# Tokenize texts
num_words = 20000
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
encoder.fit(train_labels)
# train_labels = to_categorical(encoder.transform(train_labels))
# test_labels = to_categorical(encoder.transform(test_labels))
train_labels = to_categorical(encoder.transform(train_labels), num_classes=len(labels.nunique()))
test_labels  = to_categorical(encoder.transform(test_labels), num_classes=len(labels.nunique()))


# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(num_words, 17, input_length=max_length)),
model.add(tf.keras.layers.Conv1D(128, 7, activation='swish')),
model.add(tf.keras.layers.Conv1D(64, 5, activation='swish')),
model.add(tf.keras.layers.GlobalMaxPooling1D()),
model.add(tf.keras.layers.Flatten()),
# model.add(tf.keras.layers.Dense(17, activation='relu')),
# model.add(tf.keras.layers.Dense(len(labels.nunique()), activation='softmax'))
model.add(tf.keras.layers.Dense(len(labels.nunique()), activation='softmax'))
# model.add(tf.keras.layers.Dense(1, activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=12)

num_epochs = 30
model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(test_padded, test_labels), batch_size=64, use_multiprocessing=True, callbacks=[early_stopping])

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.models import Sequential
# from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
# from keras.optimizers import Adam
# from keras.utils import to_categorical
# from sklearn.preprocessing import LabelEncoder

# # Load data
# df = pd.read_csv('news-classification.csv')

# # Split data into training and test sets
# train_texts, test_texts, train_labels, test_labels = train_test_split(df['content'], df['category_level_1'], test_size=0.2)

# num_words = 20000
# # Tokenize texts
# tokenizer = Tokenizer(num_words, oov_token='<OOV>')
# tokenizer.fit_on_texts(train_texts)
# train_sequences = tokenizer.texts_to_sequences(train_texts)
# test_sequences = tokenizer.texts_to_sequences(test_texts)

# # Pad sequences
# max_length = max(len(sequence) for sequence in train_sequences)
# train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
# test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# # Convert labels to one-hot encoding
# encoder = LabelEncoder()
# encoder.fit(train_labels)
# train_labels = to_categorical(encoder.transform(train_labels))
# test_labels = to_categorical(encoder.transform(test_labels))

# # Define the model
# model = Sequential()
# model.add(Embedding(20000, 100, input_length=max_length))
# model.add(Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.01)))   # Reduced complexity and added L2 regularization
# model.add(Conv1D(64, 5, activation='relu', kernel_regularizer=l2(0.01)))   # Reduced complexity and added L2 regularization
# #model.add(Dropout(0.5))  # Added Dropout layer
# model.add(GlobalMaxPooling1D())
# model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
# model.add(Dropout(0.5))  # Added Dropout layer
# model.add(Dense(train_labels.shape[1], activation='softmax'))  # The number of output units is equal to the number of categories

# # Compile the model
# optimizer = Adam(learning_rate=0.1)
# model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# # Define early stopping
# early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# # Train the model
# model.fit(train_padded, train_labels, epochs=30, validation_data=(test_padded, test_labels), batch_size=128, callbacks=[early_stopping], use_multiprocessing=True)


# Evaluate
loss, accuracy = model.evaluate(test_padded, test_labels)
print(f'Loss: {loss}, Accuracy: {accuracy}')

# Predict
predictions = model.predict(test_padded)

# Convert the predictions from probabilities to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Convert the one-hot encoded test labels back to class labels
actual_labels = np.argmax(test_labels, axis=1)

# Create a DataFrame that contains the predicted and actual labels
df = pd.DataFrame({
    'Predicted': predicted_labels,
    'Actual': actual_labels
})

print(df)
