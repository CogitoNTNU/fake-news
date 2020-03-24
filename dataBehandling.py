import json
import random
import keras
import re
from config import VOCABULARY_SIZE, TWEET_LENGTH


def dataLoad(file):
    TrumpDump = open(file, encoding="UTF-8")
    data = json.load(TrumpDump)
    mentionSet = []
    linkSet = []
    hashtagSet = []
    wordCaseSet = []
    for i in range(len(data)):
        wordCaseSet.append([])
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
                word = "".join(re.findall("[A-Za-z0-9.!,']", word))
                if word == "":
                    continue
                newTweet.append(word)
            else:
                lastchar = word[-1]
                word = word[0:-1]
                if "http" in word:
                    linkSet.append(word)
                    word = "<url>"
                elif "@" in word:
                    mentionSet.append(word)
                    word = "<user>"
                elif "#" in word:
                    hashtagSet.append(word)
                    word = "<hashtag>"
                word = "".join(re.findall("[A-Za-z0-9.!,']",word))
                if word == "":
                    newTweet.append(lastchar)
                    continue
                newTweet.append(word)
                newTweet.append(lastchar)
        data[i] = newTweet
    return data,linkSet,mentionSet,hashtagSet


def createTokenizer(data):
    token = keras.preprocessing.text.Tokenizer(num_words=VOCABULARY_SIZE, filters='"“”#$%&()*+-/:;<=>@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
    token.fit_on_texts(data)
    return token

def getDataSet(data, token, wordDic):
    data_matrix = token.texts_to_sequences(data)
    wordCaseSet = [[[1,0,0] for stuff in range(50)] for otherStuff in range(len(data_matrix))]
    for i in range(len(data_matrix)):

        data_matrix[i].append(0)
        data_matrix[i] = [0]*(TWEET_LENGTH-len(data_matrix[i])) + data_matrix[i][0:min(TWEET_LENGTH,len(data_matrix[i]))]

        oldTweet = data[i][::-1]
        oldTweetLower = [word.lower() for word in oldTweet]
        for j in range(len(data_matrix[i]) - 1 , -1 , -1):
            if data_matrix[i][j] != 0:
                wordIndex = oldTweetLower.index(wordDic[data_matrix[i][j]].lower())# next(k for k,v in enumerate(oldTweet) if v.lower() == wordDic[data_matrix[i][j]].lower())
                if oldTweet[wordIndex].isupper():
                    wordCaseSet[i][j] = [0,0,1]
                elif oldTweet[wordIndex][0].isupper():
                    wordCaseSet[i][j] = [0,1,0]
                del oldTweet[wordIndex]
                del oldTweetLower[wordIndex]



    return data_matrix, wordCaseSet

def getDict(token):
    numSequence = [[i] for i in range(VOCABULARY_SIZE)]
    numberDic = token.sequences_to_texts(numSequence)
    return numberDic

def saveAsJSON(data, filename):
    json.dump(data, open(filename,"w"))


data,linkSet,mentionSet,hashtagSet = dataLoad("data/trumpTweets.json")
token = createTokenizer(data)
wordDic = getDict(token)
dataSet, wordCaseSet = getDataSet(data, token, wordDic)

saveAsJSON(wordDic, "data/dataDic.json")
saveAsJSON(dataSet,"data/dataSet.json")
saveAsJSON(linkSet,"data/linkSet.json")
saveAsJSON(mentionSet,"data/mentionSet.json")
saveAsJSON(hashtagSet,"data/hashtagSet.json")
saveAsJSON(wordCaseSet, "data/wordCaseSet.json")
