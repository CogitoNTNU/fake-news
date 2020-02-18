import json
import numpy as np
import pandas as pd
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM
from keras.utils import to_categorical
from keras.layers import Masking
import dataSetBoogaloo
data = ""
atestset = []
labels = []
output_size = 3500
categoric = ""
"""
with open('trumptweets.json') as json_file:
    data = json.load(json_file)
    i=0
    g=0
    for p in data:
        print(i,'text: ' + p['text'])
        atestset.append(p['text'])
        labels.append(i%4300)
        i+=1
    categoric = to_categorical(labels,32105)
    labels = categoric
"""
dataset = pd.read_json("./dataset.json")
print(len(dataset))
print(dataset.head(10))
X = dataset.iloc[:,:23709].values
Y = dataset.iloc[:,23709:23710].values
print(X,Y)
v_size = 3500
t = Tokenizer()
enc_docs = t.texts_to_sequences(dataset)
pad_docs = pad_sequences(enc_docs)
embeddings_index = dict()
f = open('glove/glove.twitter.27B.100d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((v_size, 100))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector
# define model
model = Sequential()
e = Embedding(v_size, 100, weights=[embedding_matrix], input_length=61, trainable=False)
model.add(e)
model.add(Masking(mask_value=0.0))
model.add(LSTM(50))
model.add(Dropout(0.75))
model.add(Dense(32105, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(pad_docs, labels, validation_data=(pad_docs, labels), epochs=2, verbose=0.2)
# evaluate the model
loss, accuracy = model.evaluate(pad_docs, labels, verbose=0.2)
print('Accuracy: %f' % (accuracy*100))