import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.callbacks.callbacks import ModelCheckpoint
import model as md
from config import VOCABULARY_SIZE, TWEET_LENGTH, LOAD_WEIGHT, WEIGHT_FILE, EPOCHS, BATCH_SIZE

#Laster inn tweetene etter de har blitt behandlet i dataBehandling.py
dataset = pd.read_json("data/dataSet.json").to_numpy()
dataset = list(dataset)

wordCase = pd.read_json("data/wordCaseSet.json").to_numpy()
wordCase = list(wordCase)
#i den neste for løkken deles tweetene opp, slikt at i for eksempel en 50 ord lang tweet, trener vi på en som er 50
# ord lang, en som kun er de 49 første ordene, en som kun er de 48 første ordene osv.

# Når vi kutter ut ord, må vi fylle ut tomrommet med null-ordet, slikt at alle treningsdataene er samme lengde

new_dataset = []
labels = []
word_labels = []
# looper gjennom alle tweetsene datasettet
for i in range(len(dataset)):
    #Looper så lenge tweeten har flere ikke-null ord igjen
    while len(dataset[i]) > 1:
        if dataset[i][-1] == 0 and dataset[i][-2] == 0:
            break
        #Legger til tweeten i det nye datasettet, og fyller på tomrommet med null-ord
        new_dataset +=[[0]*(TWEET_LENGTH-len(dataset[i][:]))+list(dataset[i][:-1])]
        #new_wordCase += [[[1,0,0] for i in range(TWEET_LENGTH-len(wordCase[i][:]))]+list(wordCase[i][:-1])]
        #Legger til det siste ordet som en label, altså det nettverket prøver å gjette
        labels+=[dataset[i][-1]]
        #word_labels += [new_wordCase[i][-1]]
        #Kutter av det siste ordet
        dataset[i] = dataset[i][:-1]
#Gjør datasettet om til et numpy array fra en python liste
print(word_labels[0:100])
new_dataset = np.array(new_dataset)
#Printer ut formen til arrayet. Denne vil være på formen (Antall tweets å trene på, Antall ord i tweetene -1)
print("Form på datasettet", new_dataset.shape)

#One hot enkoder labelsene
labels = to_categorical(labels,VOCABULARY_SIZE)
#labels = np.concatenate((labels,np.array(word_labels)),axis=1)

#Generer modellen definert i model.py
model = md.generateModel11(training=True,trainable=True)
#Laster inn gamle vekter hvis dette er definert i config.py
if (LOAD_WEIGHT):
    model.load_weights(WEIGHT_FILE)

#Printer en oppsummering av modellen, slikt at man kan dobbelsjekke at man har valgt riktig
print(model.summary())

#Checkpoints evaluerer hvor bra modellen er etter hver epoke, og lagrer vektene kun hvis de har forbedret
# seg fra det beste resultatet
#Checkpoint 1 evaluerer value loss, mens checkpoint 2 evaluerer totalt loss.
checkpoint1 = ModelCheckpoint(filepath="weights/bestValLoss.h5",save_best_only=True, save_weights_only=True, verbose=1)
checkpoint2 = ModelCheckpoint(filepath="weights/best_trump_huge.h5",save_best_only=True, save_weights_only=True, verbose=1,monitor="loss")

#Trener modellen
model.fit(new_dataset, labels, validation_split=0.1, epochs=EPOCHS,batch_size=BATCH_SIZE,callbacks=[checkpoint1,checkpoint2])
print("finished!")
