from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout

def build_model(lr, num_words, input_length, labels_unique_num):
  model = Sequential()
  model.add(Embedding(input_dim=int(num_words), output_dim=labels_unique_num, input_length=input_length))
  model.add(Conv1D(512, 20, activation=LeakyReLU(alpha=0.01)))
  model.add(GlobalMaxPooling1D())
  model.add(BatchNormalization()) # Remove me if wrong
  model.add(Dropout(0.2)) # paei analoga me to pososto split toy dataset
  model.add(Dense(labels_unique_num, activation='softmax'))

  optimizer = Adam(learning_rate=lr)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

def train(model, train_texts, train_labels, test_texts, test_labels, batch_size, epochs=40):
	# early_stopping = EarlyStopping(monitor='val_loss', patience=12)
	model.fit(train_texts, train_labels, validation_data=(test_texts, test_labels), epochs=epochs, batch_size=batch_size, use_multiprocessing=True)
	return model

def evaluate_predict(model, test_texts, test_labels):
	loss, accuracy = model.evaluate(test_texts, test_labels)
	print(f'Loss: {loss:.3}, Accuracy: {accuracy*100:.3}%')
