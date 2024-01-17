import re
import torch
import pandas as pd
from torchtext.data import get_tokenizer
import nltk

CSV_FILE = 'news-classification.csv'

def clean_text(text):
	text = text.lower()
	text = text.replace('\xa0', ' ') # Remove non-breaking spaces
	text = re.sub(r'http\S+|www.\S+', '', text) # Remove URLs
	text = re.sub(r'\S+@\S+', '', text) # Remove email addresses
	text = re.sub(r'^.*\b(this|post|published|site)\b.*$\n?', '', text, flags=re.MULTILINE) # Remove lines like "This post was published on the site"
	text = re.sub(r'\\(?!n|r)', '', text) # Remove anything but backslashes
	text = text.replace('[\r \n]\n', ' ') # Remove newlines
	text = re.sub(r'[\r \n]\n[\r \n]\n', ' ', text) # Remove double newlines
	text = re.sub(r'from[: ]* ', '', text) # Remove "from" at the beginning of the text
	text = re.sub(r'  ', ' ', text) # Remove double spaces
	text = re.sub(r'\(photo by .*\)', '', text) # Remove lines like "(photo by reuters)"
	return text

data = pd.read_csv(CSV_FILE)

data_for_pytorch = []

# Wrapper that translates the files to be PyTorch-friendly
for i in range(len(data)):
	temp_str = clean_text(data['content'][i])
	data_for_pytorch.append((data['category_level_1'][i], data['category_level_2'][i], temp_str))

print(data_for_pytorch[int(len(data)/2)])


tokenizer = get_tokenizer('basic_english')