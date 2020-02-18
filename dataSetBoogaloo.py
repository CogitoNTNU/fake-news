import json
import random
import keras


DICTIONARY_LENGTH = 3500
TWEET_LENGTH = 50


TrumpDump = open("trumpTweets.json", encoding="UTF-8")
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
#print(data)
token = keras.preprocessing.text.Tokenizer(num_words=DICTIONARY_LENGTH, filters='!"“”#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
token.fit_on_texts(data)
data_matrix = token.texts_to_sequences(data)

for i in range(len(data_matrix)):
    prevTweet = data_matrix[i]
    newTweet = [0 for j in range(TWEET_LENGTH - len(prevTweet))]
    for value in prevTweet:
        newTweet.append(value)
    data_matrix[i] = newTweet

totValue = 0
for line in data_matrix:
    totValue = 0
    for value in line:
        totValue += value
    print(totValue)

print(data_matrix[28])
print(token.sequences_to_texts([data_matrix[28]]))
print(data[28])
print(len(data_matrix[28]))
print(token.sequences_to_texts([[i]for i in range(3500)]))
json.