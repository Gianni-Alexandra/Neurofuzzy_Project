import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
	text = text.lower()
	text = text.replace('\xa0', ' ') # Remove non-breaking spaces EDO
	text = re.sub(r'http\S+|www.\S+', '', text) # Remove URLs
	text = re.sub(r'\S+@\S+', '', text) # Remove email addresses
	text = text.replace('[\r \n]\n', ' ') # Remove newlines  EDO

	words = text.split() # Remove stopwords
	filtered_words = [word for word in words if word not in stop_words]
	text = ' '.join(filtered_words)

	return text


def import_text(fname, content_name='content', label_level_1_name='category_level_1', label_level_2_name='category_level_2'):
  df = pd.read_csv(fname)
  texts = df[content_name].apply(clean_text)
  labels_level_1 = df[label_level_1_name]
  labels_level_2 = df[label_level_2_name]
  return texts, labels_level_1, labels_level_2

def preprocess_data(texts, labels, test_size = 0.2, max_words=10000, max_len = 200):
  # Split data into training and testing sets
  train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=test_size)

  ros = RandomOverSampler(random_state=777)
  train_texts, train_labels = ros.fit_resample(train_texts.values.reshape(-1,1), train_labels)
  train_texts = pd.Series(train_texts.flatten())

  # Convert labels to one-hot encoding
  encoder = LabelEncoder()
  encoder.fit(labels)
  train_labels = to_categorical(encoder.transform(train_labels), num_classes=labels.nunique())
  test_labels  = to_categorical(encoder.transform(test_labels), num_classes=labels.nunique())

  tokenizer = Tokenizer(oov_token="<OOV>")
  tokenizer.fit_on_texts(train_texts)

  nunique_words = len(tokenizer.word_index) +1

  # Convert texts to sequences
  train_sequences = tokenizer.texts_to_sequences(train_texts)
  test_sequences = tokenizer.texts_to_sequences(test_texts)

  # Get the length of the longest sequence
  max_length = max(len(sequence) for sequence in train_sequences)

  # Pad sequences
  train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
  test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

  return train_padded, test_padded, train_labels, test_labels, tokenizer, max_length, nunique_words
