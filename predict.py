import model as md
import numpy as np
model = md.generateWideModel()

model.load_weights("weights.h5")

x = "we need to build a"
x=x.lower()
x=x.split(" ")
x = [md.dataDic.index(i) for i in x]
x[0:0]= [0]*(49-len(x))
x = np.array(x)
print(x)
print(x.shape)
print(model.predict(np.zeros((2,49))).argmax())
x = model.predict(np.array([x])).argmax()
print(x)
print(md.dataDic[x])

x = "make america great"
x=x.lower()
x=x.split(" ")
x = [md.dataDic.index(i) for i in x]
x[0:0]= [0]*(49-len(x))
x = np.array(x)
for i in range(50):
    word = model.predict(np.array([x])).argmax()
    print(md.dataDic[word])
    x = np.array(list(x[1:49])+[word])