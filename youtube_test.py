import os
import re
import shutil
import string

from collections import Counter

import pandas as pd
import numpy as np

import sklearn

from sklearn.model_selection import train_test_split

CSV_FILE = "news-classification.csv"

## Clean our text from unecessary characters etc
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

## Map category_level_1 label to numeric
def get_unique_num(sentiment, list_of_sentiment):
    for i in range(len(list_of_sentiment)):
        if sentiment == list_of_sentiment[i]:
            return i
    return -1

## Read our CSV_file
train_data= pd.read_csv(CSV_FILE)
train_data['Num_words_text'] = train_data['content'].apply(lambda x:len(str(x).split())) 
mask = train_data['Num_words_text'] >2

train_data = train_data[mask]

unique_word_list = train_data['category_level_1'].value_counts()

max_train_sentence_length  = train_data['Num_words_text'].max()

# Clean our text
train_data['content'] = train_data['content'].apply(clean_text)

# Generate unieque number for each category_level_1
train_data['label'] = train_data.apply(lambda row: get_unique_num(row['category_level_1'], unique_word_list.index), axis=1)

X = train_data['content'].tolist()
Y = train_data['label'].tolist()

# Split our data into train and validation
X_train, X_valid, Y_train, Y_valid= train_test_split(X,\
                                                    Y,\
                                                      test_size=0.2,\
                                                      stratify = Y,\
                                                      random_state=0)

train_dat =list(zip(Y_train,X_train))
valid_dat =list(zip(Y_valid,X_valid))

import torch
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate vocabulary
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

tokenizer = get_tokenizer('basic_english') # May need to change tokenizer
train_iter = train_dat
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) 

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        #  print(_label)
         label_list.append(label_pipeline(_label))
         processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
         text_list.append(processed_text)
         offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)

#train_iter =train_dat
#dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

from torch import nn
import torch.nn.functional as F

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim,64)
        self.fc2 = nn.Linear(64,16)
        self.fc3 = nn.Linear(16, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()
        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        x = F.relu(self.fc1(embedded))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
train_iter1 = train_dat
num_class = len(set([label for (label, text) in train_iter1]))
#print(num_class)
vocab_size = len(vocab)
emsize = 128
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)

import time

def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
# Hyperparameters
EPOCHS = 10 # epoch
LR =10  # learning rate
BATCH_SIZE = 16 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

train_iter2 = train_dat
# test_iter2 =test_dat 
valid_iter2= valid_dat




train_dataloader = DataLoader(train_iter2, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_iter2, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
# test_dataloader = DataLoader(test_iter2, batch_size=BATCH_SIZE,
#                              shuffle=True, collate_fn=collate_batch)

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)

    print('Checking the results of test dataset.')
# accu_test = evaluate(test_dataloader)
# print('test accuracy {:8.3f}'.format(accu_test))

# sentiment_label = {2:"Positive",
#                    1: "Negative",
#                    0: "Neutral"
#                   }

# def predict(text, text_pipeline):
#     with torch.no_grad():
#         text = torch.tensor(text_pipeline(text))
#         output = model(text, torch.tensor([0]))
#         return output.argmax(1).item() 
# ex_text_str = "soooooo wish i could, but im in school and myspace is completely blocked"
# model = model.to("cpu")

# print("This is a %s tweet" %sentiment_label[predict(ex_text_str, text_pipeline)])