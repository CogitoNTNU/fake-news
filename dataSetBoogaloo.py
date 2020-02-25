import json
import random
import keras
import re

DICTIONARY_LENGTH = 3500
TWEET_LENGTH = 50


def dataLoad(file):
    TrumpDump = open(file, encoding="UTF-8")
    data = json.load(TrumpDump)
    mentionSet = []
    linkSet = []
    hashtagSet = []
    for i in range(len(data)):
        data[i] = data[i]["text"].split(" ")
        oldTweet = data[i]
        newTweet=[]
        for word in oldTweet:
            word = word.replace("\u2019","'")
            if not re.search("[!.,?]$",word):
                if "http" in word:
                    linkSet.append(word)
                    word = "thisisalink"
                elif "@" in word:
                    mentionSet.append(word)
                    word = "thisisamention"
                elif "#" in word:
                    hashtagSet.append(word)
                    word = "thisisahashtag"
                newTweet.append(word)
            else:
                lastchar = word[-1]
                word = word[0:-1]
                if "http" in word:
                    linkSet.append(word)
                    word = "thisisalink"
                elif "@" in word:
                    mentionSet.append(word)
                    word = "thisisamention"
                elif "#" in word:
                    hashtagSet.append(word)
                    word = "thisisahashtag"
                newTweet.append(word)
                newTweet.append(lastchar)
        data[i] = " ".join(newTweet)
    print(linkSet)
    return data,linkSet,mentionSet,hashtagSet


def createTokenizer(data):
    token = keras.preprocessing.text.Tokenizer(num_words=DICTIONARY_LENGTH, filters='"“”#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
    token.fit_on_texts(data)
    return token

def getDataSet(data, token):
    data_matrix = token.texts_to_sequences(data)
    for i in range(len(data_matrix)):
        data_matrix[i].append(0)
        data_matrix[i] = [0]*(TWEET_LENGTH-len(data_matrix[i])) + data_matrix[i][0:min(TWEET_LENGTH,len(data_matrix[i]))]
    return data_matrix

def getDict(token):
    numSequence = [[i] for i in range(DICTIONARY_LENGTH)]
    numberDic = token.sequences_to_texts(numSequence)
    return numberDic

def saveAsJSON(data, filename):
    json.dump(data, open(filename,"w"))


data,linkSet,mentionSet,hashtagSet = dataLoad("trumpTweets.json")
token = createTokenizer(data)
wordDic = getDict(token)
dataSet = getDataSet(data, token)
saveAsJSON(wordDic, "dataDic.json")
saveAsJSON(dataSet,"dataSet.json")
saveAsJSON(linkSet,"linkSet.json")
saveAsJSON(mentionSet,"mentionSet.json")
saveAsJSON(hashtagSet,"hashtagSet.json")
maxLength = 0
minLength = 999999
"""
for tweet in dataSet:
    if len(tweet) > maxLength:
        maxLength = len(tweet)
    if len(tweet) < minLength:
        minLength = len(tweet)
"""
