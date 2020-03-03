from keras.models import Sequential
from keras.layers import Dense,Dropout, BatchNormalization
from keras.layers import Flatten, Input, Concatenate,Model,LSTM
from keras.layers.embeddings import Embedding
from keras.layers import CuDNNLSTM
from keras.utils import to_categorical
from keras.layers import Masking
import json
import numpy as np
VOCABULARY_SIZE=3500
dataDic = []
with open("./dataDic.json") as f:
    dataDic = json.load(f)

EMBEDDING_FILE="C:/Program Files (x86)/Cogito Data/glove/glove.twitter.27B.100d.txt"
def loadEmbeddings():
    embeddings_index = np.zeros(shape=(VOCABULARY_SIZE, 100))

    f = open(EMBEDDING_FILE, encoding="UTF-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float16')
        try:
            embeddings_index[dataDic.index(word)] = coefs
        except Exception as e:
            pass
    f.close()
    return embeddings_index
def generateModel():
    embeddings_index = loadEmbeddings()
    model = Sequential()
    e = Embedding(VOCABULARY_SIZE, 100, weights=[embeddings_index], input_length=49, trainable=False)
    model.add(e)
    #model.add(Masking(mask_value=0.0))
    model.add(CuDNNLSTM(100,return_sequences=True))
    model.add(CuDNNLSTM(100))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.25))
    model.add(Dense(VOCABULARY_SIZE, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def generateModel2():
    embeddings_index = loadEmbeddings()

    model = Sequential()
    e = Embedding(VOCABULARY_SIZE, 100, weights=[embeddings_index], input_length=49, trainable=False)
    model.add(e)
    #model.add(Masking(mask_value=0.0))
    model.add(CuDNNLSTM(400,return_sequences=True))
    model.add(CuDNNLSTM(300))
    model.add(Dense(200,activation='relu'))
    #model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(VOCABULARY_SIZE, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def generateModel3():
    embeddings_index = loadEmbeddings()

    model = Sequential()
    e = Embedding(VOCABULARY_SIZE, 100, weights=[embeddings_index], input_length=49, trainable=False)
    model.add(e)
    #model.add(Masking(mask_value=0.0))
    model.add(CuDNNLSTM(50,return_sequences=True))
    model.add(CuDNNLSTM(100))
    model.add(Dense(100,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(VOCABULARY_SIZE, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def generateModel4():
    embeddings_index = loadEmbeddings()

    model = Sequential()
    e = Embedding(VOCABULARY_SIZE, 100, weights=[embeddings_index], input_length=49, trainable=False)
    model.add(e)
    #model.add(Masking(mask_value=0.0))
    model.add(CuDNNLSTM(128))
    model.add(Dense(200,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(VOCABULARY_SIZE, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def generateModel5():
    embeddings_index = loadEmbeddings()

    model = Sequential()
    e = Embedding(VOCABULARY_SIZE, 100, weights=[embeddings_index], input_length=49, trainable=False)
    model.add(e)
    #model.add(Masking(mask_value=0.0))
    model.add(CuDNNLSTM(128))
    model.add(Dense(200,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(VOCABULARY_SIZE, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def generateNSModel():
    embeddings_index = loadEmbeddings()

    ord = Input(shape=(49,))
    caps = Input(shape=(49,3))
    e = Embedding(VOCABULARY_SIZE, 100, weights=[embeddings_index], input_length=49, trainable=False)(ord)
    out = Concatenate(axis=2)([e,caps])
    out = LSTM(128)(out)
    out = Dense(256, activation='relu')(out)
    out = BatchNormalization()(out)
    out = Dense(VOCABULARY_SIZE+3, activation='sigmoid')(out)
    print(out.shape)
    model = Model(inputs = [ord,caps], output = out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model