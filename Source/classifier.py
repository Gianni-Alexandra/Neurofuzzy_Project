import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from import_text import import_text, preprocess_data
from build_train import build_model, train, evaluate_predict

texts, labels_level_1, labels_level_2 = import_text('../news-classification.csv')

print(f'info: Preprocessing level 1 data!')
train_texts_1, test_texts_1, train_labels_1, test_labels_1, tokenizer_1, max_length_1, nunique_words_1 = preprocess_data(texts, labels_level_1) # Level 1
print(f'info: Preprocessing level 2 data!')
train_texts_2, test_texts_2, train_labels_2, test_labels_2, tokenizer_2, max_length_2, nunique_words_2 = preprocess_data(texts, labels_level_2) # Level 2

print(f'info: Building level 1 neural network!')
model_level_1 = build_model(0.0015, nunique_words_1, max_length_1, labels_level_1.nunique())
print(f'info: Building level 2 neural network!')
model_level_2 = build_model(0.001, nunique_words_2, max_length_2, labels_level_2.nunique())

print('info: Training level 1:')
model_level_1 = train(model_level_1, train_texts_1, train_labels_1, test_texts_1, test_labels_1, batch_size=64, epochs=10)
print('\n\ninfo: Training level 2:')
model_level_2 = train(model_level_2, train_texts_2, train_labels_2, test_texts_2, test_labels_2, batch_size=64, epochs=10)

print('\n\nEvaluating level 1:')
evaluate_predict(model_level_1, test_texts_1, test_labels_1)
print('\n\nEvaluating level 2')
evaluate_predict(model_level_2, test_texts_2, test_labels_2)
