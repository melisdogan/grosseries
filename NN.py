from PIL import Image 
import numpy as np
import os 
import imageio
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers. normalization import BatchNormalization
from keras.datasets import mnist
from keras import regularizers


categoryNum = 25
encoding = {} 
f = open(".\\classid.txt", "r")
fileContents = f.read()
fileContents = fileContents.split('\n')
for i in range(len(fileContents)-1):
    fileContents[i] = fileContents[i].split(' ')
    encoding[fileContents[i][0]] = fileContents[i][1]

#def setLabel(name):
#    oneHot = np.zeros(categoryNum)
#    oneHot[int(name)] = 1
#    return oneHot

def importTrainData():
    trainData = []
    #label = []
    m = open(".\\splits\\train0.txt", "r")
    train = m.read()
    train = train.split('\n')
    for i in range(len(train)-1):
        train[i] = train[i].split(' ')
        #label = setLabel(train[i][1])
        #label.append(train[i][1])
        path = ".\\images\\" + train[i][0]
        image = Image.open(path)
        trainData.append([np.array(image), int(train[i][1])])
    return trainData

def importTestData():
    testData = []
    m = open(".\\splits\\test0.txt", "r")
    test = m.read()
    test = test.split('\n')
    for i in range(len(test)-1):
        test[i] = test[i].split(' ')
        #label = setLabel(test[i][1])
        path = ".\\images\\" + test[i][0]
        image = Image.open(path)
        testData.append([np.array(image), int(test[i][1])])
    return testData

trainData = importTrainData()
testData = importTestData()

labels = np.array([i[1] for i in trainData])

trainImages = np.array([i[0] for i in trainData])
#trainLabels = np.array([i[1] for i in trainData])
trainLabels = keras.utils.to_categorical(labels)



model = Sequential()
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01), input_shape=(256, 256, 3)))

model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))

model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))

model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01), bias_regularizer=regularizers.l2(0.01)))

model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(categoryNum, activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics = ['accuracy'])
model.fit(x = trainImages, y = trainLabels, epochs = 5, verbose = 1)

testImages = np.array([i[0] for i in testData])

testlabels = np.array([i[1] for i in testData])
testLabels = keras.utils.to_categorical(testlabels)

loss, acc = model.evaluate(testImages, testLabels, verbose = 0)
print(acc * 100)
