from keras.models import Sequential,Model
from keras.layers import Dense,Dropout, BatchNormalization
from keras.layers import Flatten, Input, Concatenate,LSTM,GRU
from keras.layers.embeddings import Embedding
from keras.layers import CuDNNLSTM, LSTM, Concatenate, Input, Reshape
from keras.utils import to_categorical
from keras.layers import Masking
from keras.optimizers import Adagrad
from keras import optimizers
import json
import numpy as np
from config import VOCABULARY_SIZE, TWEET_LENGTH, EMBEDDING_FILE_LOCATION,LEARNING_RATE

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
    embeddings = np.load("data/embeddings.npy",allow_pickle=True)
    embeddings_index_predict = np.zeros(shape=(len(embeddings)+1, 100))
    words = np.load("data/embeddingWords.npy",allow_pickle=True)
    for i in range(len(embeddings)):
        #word er lik ordet til embeddingene
        #coefs er lik de faktiske verdiene knyttet til ordet
        try:
            #Legger de riktige embeddingene
            if (words[i] in dataDic):
                embeddings_index[dataDic.index(words[i])] = embeddings[i]
            embeddings_index_predict[i+1]=embeddings[i]
            embeddings_dict[words[i]] = i+1
        except Exception as e:
            print(e)

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
def generateEmbeddingLayer(training = True,trainable=False):
    if training:
        e = Embedding(VOCABULARY_SIZE, 100, weights=[embeddings_index], input_length=TWEET_LENGTH-1, trainable=trainable,name="embedding_train")
    else:
        e = Embedding(len(embeddings_index_predict), 100, weights=[embeddings_index_predict], input_length=TWEET_LENGTH-1, trainable=trainable, name="embedding_predict")

    return e



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


def generateModel6(training = True,trainable = False):
    model = Sequential()
    e = generateEmbeddingLayer(training)
    model.add(e)
    # model.add(Masking(mask_value=0.0))
    #model.add(CuDNNLSTM(128,return_sequences=True))
    model.add(LSTM(1024,name="lstm_0", trainable=trainable, return_sequences=False))
    #model.add(LSTM(1024,name="lstm_1", trainable=trainable, return_sequences=False))
    #model.add(LSTM(256,name="lstm_2", trainable=trainable))
    #model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu',name="dense_0"))
    model.add(BatchNormalization(name="batch_normalization_0"))
    #model.add(Dropout(0.2))
    model.add(Dense(VOCABULARY_SIZE, activation='sigmoid', name="dense_1"))
    # compile the model
    optimizer = optimizers.Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model
def generateModel9(training = True,trainable = True):
    ord = Input(shape=(49,))
    e = generateEmbeddingLayer(training,True)(ord)
    out = LSTM(512,name="lstm_0", return_sequences=True, trainable=trainable)(e)
    #e = Flatten(name="flatten_0")(e)
    out = Concatenate(axis=2, name="concatenate_0")([out,e])
    out = Dropout(0.2,name="dropout_0", trainable=True)(out)
    out = LSTM(512,name="lstm_1", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_1")([out, e])
    out = LSTM(512, name="lstm_2", trainable=trainable)(out)

    out = Dense(1024, activation='relu',name="dense_0")(out)
    out = BatchNormalization(name="batch_normalization_0")(out)
    out = Dense(VOCABULARY_SIZE, activation='sigmoid', name="dense_1")(out)
    model = Model(inputs = ord, output = out)
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def generateModel11(training = True,trainable=True):
    ord = Input(shape=(49,))
    e = generateEmbeddingLayer(training,True)(ord)
    out = LSTM(512,name="lstm_0", return_sequences=True, trainable=trainable)(e)
    #e = Flatten(name="flatten_0")(e)
    out = Concatenate(axis=2, name="concatenate_0")([out,e])
    out = Dropout(0.2,name="dropout_0", trainable=True)(out)
    out = LSTM(512,name="lstm_1", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_1")([out, e])
    out = LSTM(512, name="lstm_2", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_2")([out, e])
    out = LSTM(512, name="lstm_3", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_3")([out, e])
    out = LSTM(512, name="lstm_4", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_4")([out, e])
    out = LSTM(512, name="lstm_5", trainable=trainable)(out)

    out = Dense(1024, activation='relu',name="dense_0")(out)
    out = BatchNormalization(name="batch_normalization_0")(out)
    out = Dense(VOCABULARY_SIZE, activation='sigmoid', name="dense_1")(out)
    model = Model(inputs = ord, output = out)
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def generateModel12(training = True,trainable=True):
    ord = Input(shape=(49,))
    e = generateEmbeddingLayer(training,True)(ord)
    out = LSTM(512,name="lstm_0", return_sequences=True, trainable=trainable)(e)
    #e = Flatten(name="flatten_0")(e)
    out = Concatenate(axis=2, name="concatenate_0")([out,e])
    out = Dropout(0.2,name="dropout_0", trainable=True)(out)
    out = LSTM(512,name="lstm_1", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_1")([out, e])
    out = LSTM(512, name="lstm_2", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_2")([out, e])
    out = LSTM(512, name="lstm_3", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_3")([out, e])
    out = LSTM(512, name="lstm_4", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_4")([out, e])
    out = LSTM(512, name="lstm_5", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_5")([out, e])
    out = LSTM(512,name="lstm_6", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_6")([out, e])
    out = LSTM(512, name="lstm_7", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_7")([out, e])
    out = LSTM(512, name="lstm_8", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_8")([out, e])
    out = LSTM(512, name="lstm_9", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_9")([out, e])
    out = LSTM(512, name="lstm_10", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_10")([out, e])
    out = LSTM(512, name="lstm_11", trainable=True, return_sequences=True)(out)

    out = Concatenate(axis=2, name="concatenate_11")([out, e])
    out = LSTM(512, name="lstm_12", trainable=trainable)(out)

    out = Dense(1024, activation='relu',name="dense_0")(out)
    out = BatchNormalization(name="batch_normalization_0")(out)
    out = Dense(VOCABULARY_SIZE, activation='sigmoid', name="dense_1")(out)
    model = Model(inputs = ord, output = out)
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def generateModel10(training = True,trainable=True):

    ord = Input(shape=(49,))
    e = generateEmbeddingLayer(training)(ord)
    out = Dense(100, name="dense_0", activation="relu",trainable=trainable)(e)
    out = LSTM(512,name="lstm_0", return_sequences=True, trainable=trainable)(out)
    #e = Flatten(name="flatten_0")(e)
    out = Concatenate(axis=2, name="concatenate_0")([out,e])
    out = Dropout(0.2,name="dropout_0", trainable=trainable)(out)
    out = LSTM(512,name="lstm_1", trainable=trainable)(out)
    out = Dense(512, activation='relu',name="dense_1")(out)
    out = BatchNormalization(name="batch_normalization_0")(out)
    out = Dense(VOCABULARY_SIZE, activation='sigmoid', name="dense_2")(out)
    model = Model(inputs = ord, output = out)
    optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def generateModel8(training = True,trainable=True):

    ord = Input(shape=(49,))
    e = generateEmbeddingLayer(training)(ord)
    out = LSTM(512,name="lstm_0", return_sequences=True, trainable=trainable)(e)
    #e = Flatten(name="flatten_0")(e)
    out = Concatenate(axis=2, name="concatenate_0")([out,e])
    out = Dropout(0.2,name="dropout_0", trainable=trainable)(out)
    out = LSTM(512,name="lstm_1", trainable=trainable)(out)
    out = Dense(1024, activation='relu',name="dense_0")(out)
    out = BatchNormalization(name="batch_normalization_0")(out)
    out = Dense(VOCABULARY_SIZE, activation='sigmoid', name="dense_1")(out)
    model = Model(inputs = ord, output = out)
    optimizer = optimizers.Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
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

def generateNSModel(training=True):
    embeddings_index = loadEmbeddings()

    ord = Input(shape=(49,))
    caps = Input(shape=(49,3))
    e = generateEmbeddingLayer(training)(ord)
    out = Concatenate(axis=2)([e,caps])
    out = LSTM(128)(out)
    out = Dense(256, activation='relu')(out)
    out = BatchNormalization()(out)
    out = Dense(VOCABULARY_SIZE+3, activation='sigmoid')(out)
    model = Model(inputs = [ord,caps], output = out)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model

def generateCaseModel(training=True):
    model = Sequential()
    e = generateEmbeddingLayer(training)
    model.add(e)
    #model.add(Masking(mask_value=0.0))
    model.add(LSTM(256, return_sequences=True,name = "lstm_0"))
    model.add(GRU(256,name="gru_0"))
    model.add(Dense(512,activation='relu',name="dense_0"))
    model.add(Dense(1024,activation='relu',name="dense_1"))
    model.add(BatchNormalization(name="batch_normalization_0"))
    model.add(Dropout(0.2,name="dropout_0"))
    model.add(Dense(49*3, activation='sigmoid',name="dense_2"))
    model.add(Reshape((49,3)))
    # compile the model
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    return model
generateCaseModel()