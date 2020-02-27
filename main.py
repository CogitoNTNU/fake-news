import json
import numpy as np
import pandas as pd
from keras.utils import to_categorical
import model as md
data = ""
atestset = []
labels = []
VOCABULARY_SIZE = 3500
WORD_COUNT = 50
LOAD_WEIGHT=False
WEIGHT_FILE = "weights1.h5"
categoric = ""

dataset = pd.read_json("./dataSet.json").to_numpy()
dataset = list(dataset)
new_dataset = []
labels = []
for i in range(len(dataset)):
    while len(dataset[i]) > 1:
        if dataset[i][-1] == 0 and dataset[i][-2] == 0:
            break
        #print(dataset[i][:].shape)
        #print([0]*(WORD_COUNT-len(dataset[i][:])))
        new_dataset +=[[0]*(WORD_COUNT-len(dataset[i][:]))+list(dataset[i][:-1])]
        labels+=[dataset[i][-1]]
        dataset[i] = dataset[i][:-1]
new_dataset = np.array(new_dataset)
print(new_dataset.shape)
#labels=[dataset[i][49] for i in range(len(dataset))]
#dataset = np.array([dataset[i][0:49] for i in range(len(dataset))])

labels = to_categorical(labels,VOCABULARY_SIZE)
model = md.generateModel5()
if (LOAD_WEIGHT):
    model.load_weights(WEIGHT_FILE)
print(model.summary())
# fit the model
for i in range(10):
    model.fit(new_dataset, labels, validation_split=0.1, epochs=2,batch_size=128)
    model.save_weights("weights" + str(i) + ".h5")