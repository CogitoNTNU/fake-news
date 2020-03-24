#------------UNIVERSAL------------
#Antall forskjellige ord
VOCABULARY_SIZE = 3500
#antall ord pr tweet
TWEET_LENGTH = 50

#------------TRAINING------------

#Hvis False starter trener main en ny modell, mens hvis LOAD WEIGHT er True laster
# main inn vektene gitt i WEIGHT_FILE og fortsetter treningen derfra.
LOAD_WEIGHT = False
WEIGHT_FILE = "weights/weights1.h5"

#Antall epoker for å trene modellen
EPOCHS = 100
#Hvor mange tweets som trenes på parallellt.
BATCH_SIZE = 512


#------------MODEL------------
#der hvor glove embedding filen skal hentes fra
EMBEDDING_FILE_LOCATION = "C:\Cogito\glove\glove.twitter.27B.100d.txt"


#------------APPLIKASJON------------
# Velger hvilke vekter man skal bruke i applikasjonen
LOAD_WEIGHT_PREDICT = "weights/bestLoss.h5"