import model as md
import numpy as np
import json
import random
from config import TWEET_LENGTH, VOCABULARY_SIZE, LOAD_WEIGHT_PREDICT,LOAD_CASE_WEIGHT_PREDICT

model = md.generateModel11(training=True)

model.load_weights(LOAD_WEIGHT_PREDICT,by_name=True)

caseModel = md.generateCaseModel(training=False)
caseModel.load_weights(LOAD_CASE_WEIGHT_PREDICT,by_name=True)
print(model.summary())
with open("data/mentionSet.json") as f:
    mentions = json.load(f)
with open("data/hashtagSet.json") as f:
    hashtags = json.load(f)
with open("data/linkSet.json") as f:
    links = json.load(f)

def generate_text(text = ["make america","", "i am","We need to","love","september 11","crooked hillary", "hello",".","my team", "we live in a"],cases = True):
    output = []
    for x in text:
        spaces = []
        if len(x) == 0:
            randomstart = True
        else:
            randomstart = False
        if not randomstart:
            y = x.split(" ")
            spaces = [" "]*(len(y)-1)
            x=x.lower()
            x=x.split(" ")
            #x = [md.embeddings_dict[i] for i in x]
            x = [md.dataDic.index(i) for i in x]
            x[0:0]= [0]*(TWEET_LENGTH-1-len(x))
            x = np.array(x)
        else:
            spaces = []
            #y =md.dataDic[random.randint(0,200)]
            y = random.randint(0, 200)
            x = [0] * (TWEET_LENGTH - 2) +[y]
            y =[md.dataDic[y]]
            #x = [0] * (TWEET_LENGTH - 2) + [md.embeddings_dict[y]]

        for i in range(TWEET_LENGTH):
            predict =model.predict(np.array([x]))
            #print(predict.argmax())
            #print(md.dataDic[predict.argmax()])
            try:
                print(md.embeddings_dict[md.dataDic[predict.argmax()]])
            except:
                pass
            #print("")
            weightList = []
            currentSum = 0
            for w in predict[0]:
                #print(w)
                currentSum += w**5
                weightList.append(currentSum)

            rWordValue = random.random()*currentSum
            #print(weightList)
            if (rWordValue < weightList[0]):
                wordindex = 0
            else:
                for w in range(len(weightList)):
                    if rWordValue > weightList[w]:
                        wordindex=w+1
                    else:
                        break
            if (wordindex == 0):
                break

            word = md.dataDic[wordindex]

            #try:
            #    wordindex = md.embeddings_dict[word]
            #except:
            #    wordindex = 0


            x = np.array(list(x[1:49]) + [wordindex])
            #if cases:
            #    casePredict = caseModel.predict(np.array([x]))[0][-1].argmax()
            #    #print(casePredict)
            #    if (casePredict ==1):
            #        word = word[0].upper()+word[1:]
            #    elif (casePredict==2):
            #        word = word.upper()
            if word in [",",".","!"]:
                spaces += [""]
                y += [word]
            elif word.lower() == "<user>":
                spaces += [" "]
                y += [random.choice(mentions)]
            elif word.lower() == "<url>":
                spaces += [" "]
                y += [random.choice(links)]
            elif word.lower() == "<hashtag>":
                spaces += [" "]
                y += [random.choice(hashtags)]
            else:
                spaces += [" "]
                y += [word]

                #break
            #print(md.dataDic[word])
            #print(predict[0][predict.argmax()])
        print(y)
        out = ""
        if cases:
            y_len = len(y)
            dif = TWEET_LENGTH-1-y_len
            casePredict = caseModel.predict(np.array([x]))[0]
            print(casePredict)
            for i,case in enumerate(casePredict):
                print(case)
                max = case.argmax()
                if (i >= dif):
                    if (max ==1):
                        y[i-dif] = y[i-dif][0].upper()+y[i-dif][1:]
                    elif (max==2):
                        y[i-dif] = y[i-dif].upper()


                    if (len(spaces) != i-dif):
                        out += y[i-dif] + spaces[i-dif]
                    else:
                        out += y[i-dif]
                print(out)
        output += [out]
    return output
if __name__ == "__main__":
    generate_text()