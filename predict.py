import model as md
import numpy as np
import json
import random
#from gtts import gTTS
#from playsound import playsound
#from io import BytesIO
model = md.generateModel30()

model.load_weights("weightsbigdenser17.h5")
with open("mentionSet.json") as f:
    mentions = json.load(f)
with open("hashtagSet.json") as f:
    hashtags = json.load(f)
with open("linkSet.json") as f:
    links = json.load(f)
def generate_text(text = ["make america","", "i am","We need to","love","september 11","crooked hillary", "hello",".","my team", "we live in a"]):
    output = []
    for x in text:
        y = x
        x=x.lower()
        x=x.split(" ")
        x = [md.dataDic.index(i) for i in x]
        x[0:0]= [0]*(49-len(x))
        x = np.array(x)
        for i in range(50):
            predict =model.predict(np.array([x]))
            wordindex = predict.argmax()
            if (wordindex == 0):
                break
            word = md.dataDic[wordindex]
            if word in [",",".","!"]:
                y += word
            elif word == "thisisamention":
                y += random.choice(mentions)
            elif word == "thisisalink":
                y += random.choice(links)
            elif word == "thisisahashtag":
                y += random.choice(hashtags)
            else:
                y += " " + word
            #print(md.dataDic[word])
            #print(predict[0][predict.argmax()])
            x = np.array(list(x[1:49])+[wordindex])
        print(y)
        output += [y]
        """
        mp3_fp = BytesIO()
        text = y
        language = "en"
        speech = gTTS(text=text, lang=language, slow=False)
        speech.save("temp.mp3")
        playsound("temp.mp3")
        """
    return output
if __name__ == "__main__":
    generate_text()