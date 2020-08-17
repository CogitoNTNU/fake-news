import json
import random
import keras
import re
from config import VOCABULARY_SIZE, TWEET_LENGTH
import bz2
import os

def TrumpLoad(file):
    return json.load(open(file, encoding="UTF-8"))
def dataLoad(data):
    #TrumpDump = open(file, encoding="UTF-8")
    mentionSet = []
    linkSet = []
    hashtagSet = []
    wordCaseSet = []
    new_data = []
    for i in range(len(data)):
        try:
            wordCaseSet.append([])
            data[i] = data[i]["text"].split(" ")
            oldTweet = data[i]
            newTweet=[]
            for word in oldTweet:
                word = word.replace("\u2019","'")
                if not re.search("[!.,?]",word):
                    if "http" in word:
                        linkSet.append(word)
                        word = "<url>"
                    elif "@" in word:
                        mentionSet.append(word)
                        word = "<user>"
                    elif "#" in word:
                        hashtagSet.append(word)
                        word = "<hashtag>"
                    word = "".join(re.findall("[A-Za-z0-9.!,'<>]", word))
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
                    word = "".join(re.findall("[A-Za-z0-9.!,'<>]",word))
                    if word == "":
                        newTweet.append(lastchar)
                        continue
                    newTweet.append(word)
                    newTweet.append(lastchar)
            if len(newTweet) == 0:
                continue
            if newTweet[0] == "RT":
                continue
            new_data.append(newTweet)
        except:
            pass
    return new_data,linkSet,mentionSet,hashtagSet

def filter_generic_tweets(data, vocabulary):
    fordeling = [0]*50
    new_data = []
    for tweet in data:
        if (len(tweet) < 50):
            fordeling[len(tweet)] += 1
        if (len(tweet)/10) < random.random():
            continue
        skipping = False
        spamCount = 0
        for word in tweet:
            if (word.lower() not in vocabulary):
                skipping = True
                break
            if (word == "<user>" or word == "<hashtag>" or word == "." or word == "," or word == "<url>"):
                spamCount += 1
        if (spamCount/len(tweet) > 0.5):
            skipping = True
        if skipping:
            continue
        new_data.append(tweet)
    print("Fordeling:", fordeling)
    return new_data

def createTokenizer(data):
    token = keras.preprocessing.text.Tokenizer(num_words=VOCABULARY_SIZE, filters='"“”#$%&()*+-/:;=@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
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

def loadGenericTweet(filename,token,vocabulary):
    i = 0
    data = []
    filtered_data = []
    for path1 in os.listdir(filename)[3:]:
        old_path = "-01"
        for path2 in os.listdir(os.path.join(filename,path1)):
            for file in os.listdir(os.path.join(filename,path1,path2)):
                with bz2.open(os.path.join(filename,path1,path2,file), "r") as f:
                    for line in f:
                        data.append(json.loads(line))
            filtered_data += filter_generic_tweets(dataLoad(data)[0],vocabulary)
            data = []
            i+= 1
            print("loaded " + str(i) + " large set of tweets")
            print("Antall tweets hittil: " + str(len(filtered_data)))
            print("Den siste",filtered_data[-1])
            if(len(filtered_data) > 20000):
                dataset = getDataSet(filtered_data[:20000],token,wordDic)[0]
                json.dump(dataset,open("data/training/"+path1+"_"+ old_path  + "-" + path2 + ".json", "w"))
                old_path = path2
                filtered_data = filtered_data[20000:]
    dataset = getDataSet(filtered_data, token, wordDic)[0]
    json.dump(dataset, open("data/training/"+path1 + "_" + path2 +"-" +"end" + ".json", "w"))

def saveAsJSONInPieces(data, path,batch_size=20000):
    i = -1
    for i in range(len(data)//batch_size):
        json.dump(data[i*batch_size:(i+1)*batch_size],open(path+"tweets_" + str(i)+ ".json", "w"))
    i += 1
    json.dump(data[i * batch_size: -1], open(path + "tweets_" + str(i) + ".json", "w"))
if __name__ == "__main__":
    trumpDump = TrumpLoad("data/trumpTweets.json")
    data,linkSet,mentionSet,hashtagSet = dataLoad(trumpDump)
    token = createTokenizer(data)
    wordDic = getDict(token)
    dataSet, wordCaseSet = getDataSet(data, token, wordDic)

    #loadGenericTweet("../ekstra_data",token,wordDic)

    #getDataSet(generic_tweets, token, wordDic)[0]
    #print(len(filteredDataSet))
    #saveAsJSONInPieces(filteredDataSet,"data/training/")
    #print(filtered)
    print("finished")
    saveAsJSON(wordDic, "data/dataDic.json")
    saveAsJSON(dataSet,"data/dataSet.json")
    saveAsJSON(linkSet,"data/linkSet.json")
    saveAsJSON(mentionSet,"data/mentionSet.json")
    saveAsJSON(hashtagSet,"data/hashtagSet.json")
    saveAsJSON(wordCaseSet, "data/wordCaseSet.json")
