import json
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import model as md
data = ""
atestset = []
labels = []
output_size = 3500
WORD_COUNT = 50
categoric = ""

dataset = pd.read_json("./dataSet.json").to_numpy()
dataset = list(dataset)
v_size = 3500
new_dataset = []
labels = []
for i in range(12000):#range(len(dataset)):
    while len(dataset[i]) > 0:
        if dataset[i][-1] == 0:
            break
        #print(dataset[i][:].shape)
        #print([0]*(WORD_COUNT-len(dataset[i][:])))
        new_dataset +=[[0]*(WORD_COUNT-len(dataset[i][:]))+list(dataset[i][:-1])]
        labels+=[dataset[i][-1]]
        dataset[i] = dataset[i][:-1]
print(labels)
new_dataset = np.array(new_dataset)
print(new_dataset.shape)
#labels=[dataset[i][49] for i in range(len(dataset))]
#dataset = np.array([dataset[i][0:49] for i in range(len(dataset))])

labels = to_categorical(labels,v_size)
print(labels)
model = md.generateModel()
print(model.summary())
# fit the model
model.fit(new_dataset, labels, validation_split=0.1, epochs=2)

model.save_weights("weights.h5")