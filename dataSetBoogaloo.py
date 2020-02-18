import json
import random
import keras


DICTIONARY_LENGTH = 3500
TWEET_LENGTH = 50


def dataLoad(file):
    TrumpDump = open(file, encoding="UTF-8")
    data = json.load(TrumpDump)
    for i in range(len(data)):
        data[i] = data[i]["text"].split(" ")
        for j in range(len(data[i])):
            #print(data[i][j])
            if "http" in data[i][j]:
                data[i][j] = "thisisalink"
            elif "@" in data[i][j]:
                data[i][j] = "thisisamention"
            elif "#" in data[i][j]:
                data[i][j] = "thisisahashtag"
        data[i] = " ".join(data[i])
    return data


def createTokenizer(data):
    token = keras.preprocessing.text.Tokenizer(num_words=DICTIONARY_LENGTH, filters='!"“”#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
    token.fit_on_texts(data)
    return token

def getDataSet(data, token):
    data_matrix = token.texts_to_sequences(data)
    for i in range(len(data_matrix)):
        prevTweet = data_matrix[i]
        if len(prevTweet) < TWEET_LENGTH:
            newTweet = [0 for j in range(TWEET_LENGTH - len(prevTweet))]
            for value in prevTweet:
                newTweet.append(value)
            data_matrix[i] = newTweet
        elif len(prevTweet) > TWEET_LENGTH:
            data_matrix[i] = data_matrix[i][len(prevTweet)-TWEET_LENGTH:]
    return data_matrix

def getDict(token):
    numSequence = [[i] for i in range(DICTIONARY_LENGTH)]
    numberDic = token.sequences_to_texts(numSequence)
    return numberDic

def saveAsJSON(data, filename):
    json.dump(data, open(filename,"w"))


data = dataLoad("trumpTweets.json")
token = createTokenizer(data)
wordDic = getDict(token)
dataSet = getDataSet(data, token)
maxLength = 0
minLength = 999999
for tweet in dataSet:
    if len(tweet) > maxLength:
        maxLength = len(tweet)
    if len(tweet) < minLength:
        minLength = len(tweet)

#saveAsJSON(wordDic, "dataDic.json")

#TODO separate !, "," etc.
#TODO clean out unicode