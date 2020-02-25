import model as md
import numpy as np
model = md.generateModel()

model.load_weights("weights0.h5")


text = ["make america","", "i am","We need to","love","september 11","crooked hillary"]
for x in text:
    y = x
    x=x.lower()
    x=x.split(" ")
    x = [md.dataDic.index(i) for i in x]
    x[0:0]= [0]*(49-len(x))
    x = np.array(x)
    for i in range(50):
        predict =model.predict(np.array([x]))
        word = predict.argmax()
        y += " " + md.dataDic[word]
        #print(md.dataDic[word])
        #print(predict[0][predict.argmax()])
        x = np.array(list(x[1:49])+[word])
    print(y)