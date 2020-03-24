from keras.models import Sequential,Model
from keras.layers import Dense,Dropout, BatchNormalization
from keras.layers import Flatten, Input, Concatenate,LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.layers import CuDNNLSTM, LSTM, Concatenate, Input
from keras.utils import to_categorical
from keras.layers import Masking
from keras import optimizers
import json
import numpy as np
from config import VOCABULARY_SIZE, TWEET_LENGTH, EMBEDDING_FILE_LOCATION

#Laster inn det ferdigdefinerte vokabulæret til trump som er VOCABULARY_SIZE langt
dataDic = []
with open("data/dataDic.json") as f:
    dataDic = json.load(f)

#Henter embeddingene definert i glove filen
def loadEmbeddings():
    #Starter med å definere alle som 0
    embeddings_index = np.zeros(shape=(VOCABULARY_SIZE, 100))
    #Definerer nullvektoren
    embeddings_dict = {"": 0}

    # Embeddings_index_predict er alle embeddingsene i hele glove filen, i motsetting til embeddings_index som kun er
    # Trump sitt vokabulær
    f = open(EMBEDDING_FILE_LOCATION, encoding="UTF-8")
    embeddings_index_predict = np.zeros(shape=(len([i for i in f])+1, 100))
    f.close()

    f = open(EMBEDDING_FILE_LOCATION, encoding="UTF-8")

    i = 0
    for line in f:
        #word er lik ordet til embeddingene
        #coefs er lik de faktiske verdiene knyttet til ordet
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float16')

        try:
            #Legger de riktige embeddingene
            if (word in dataDic):
                embeddings_index[dataDic.index(word)] = coefs
            embeddings_index_predict[i+1]=coefs
            embeddings_dict[word] = i+1
        except Exception as e:
            print(e)

        i+=1
        # Bryter av etter 400000 ord, ettersom det virker som embeddingene hovedsakelig er på andre språk etter dette,
        # Dette kan økes hvis man vil ha større vokabulær i applikasjonen
        # if i == 400000:
        #    break
    f.close()
    print(len(embeddings_index_predict))
    return embeddings_index,embeddings_index_predict,embeddings_dict

#Laster inn embeddingene
embeddings_index,embeddings_index_predict,embeddings_dict = loadEmbeddings()

#Denne funkjsonen generer embedding laget
# Når man trener, laster den kun inn de embeddingene man trenger for å trene, mens
# når man predikterer, bruker den alle embeddingene, slikt at man kan bruke ord applikasjonen ikke har trent på en gang
def generateEmbeddingLayer(training = True):
    if training:
        e = Embedding(VOCABULARY_SIZE, 100, weights=[embeddings_index], input_length=TWEET_LENGTH-1, trainable=False,name="embedding_train")
    else:
        e = Embedding(len(embeddings_index_predict), 100, weights=[embeddings_index_predict], input_length=TWEET_LENGTH-1, trainable=False, name="embedding_predict")

    return e


def generateModel(training = True):
    model = Sequential()
    e = generateEmbeddingLayer(training)
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

def generateModel2(training = True):

    model = Sequential()
    e = generateEmbeddingLayer(training)
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

def generateModel3(training = True):

    model = Sequential()
    e = generateEmbeddingLayer(training)
    model.add(e)
    #model.add(Masking(mask_value=0.0))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(100,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(VOCABULARY_SIZE, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def generateModel4(training = True):

    model = Sequential()
    e = generateEmbeddingLayer(training)
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

def generateModel5(training = True):
    embeddings_index = loadEmbeddings()

    model = Sequential()
    e = generateEmbeddingLayer(training)
    model.add(e)
    #model.add(Masking(mask_value=0.0))
    #model.add(CuDNNLSTM(128,return_sequences=True))
    model.add(CuDNNLSTM(178))
    model.add(Dense(200,activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.4))
    model.add(Dense(VOCABULARY_SIZE, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def generateModel6(training = True):
    model = Sequential()
    e = generateEmbeddingLayer(training)
    model.add(e)
    # model.add(Masking(mask_value=0.0))
    #model.add(CuDNNLSTM(128,return_sequences=True))
    model.add(CuDNNLSTM(256))
    #model.add(Dropout(0.4))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Dense(VOCABULARY_SIZE, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model

def generateModel30(training = True):

    model = Sequential()
    e = generateEmbeddingLayer(training)
    model.add(e)
    #model.add(Masking(mask_value=0.0))
    model.add(LSTM(256))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    adam = optimizers.adam(learning_rate=0.5)
    model.add(Dense(VOCABULARY_SIZE, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def generateModel31(training):

    model = Sequential()
    e = generateEmbeddingLayer(training)
    model.add(e)
    #model.add(Masking(mask_value=0.0))
    model.add(LSTM(256, return_sequences=True))
    model.add(GRU(256))
    model.add(Dense(512,activation='relu'))
    model.add(Dense(1024,activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    adam = optimizers.adam(learning_rate=0.5)
    model.add(Dense(VOCABULARY_SIZE, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def generateNSModel(training):
    embeddings_index = loadEmbeddings()

    ord = Input(shape=(49,))
    caps = Input(shape=(49,3))
    e = Embedding(VOCABULARY_SIZE, 100, weights=[embeddings_index], input_length=49, trainable=True)(ord)
    out = Concatenate(axis=2)([e,caps])
    out = LSTM(128)(out)
    out = Dense(256, activation='relu')(out)
    out = BatchNormalization()(out)
    out = Dense(VOCABULARY_SIZE+3, activation='sigmoid')(out)
    model = Model(inputs = [ord,caps], output = out)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model