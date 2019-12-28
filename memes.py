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


categoryNum = 25
encoding = {} 
f = open(".\\classid.txt", "r")
fileContents = f.read()
fileContents = fileContents.split('\n')
for i in range(len(fileContents)-1):
    fileContents[i] = fileContents[i].split(' ')
    encoding[fileContents[i][0]] = fileContents[i][1]

def setLabel(name):
    oneHot = np.zeros(categoryNum)
    oneHot[int(name)] = 1
    return oneHot

def importTrainData():
    trainData = []
    m = open(".\\splits\\train0.txt", "r")
    train = m.read()
    train = train.split('\n')
    for i in range(len(train)-1):
        train[i] = train[i].split(' ')
        label = setLabel(train[i][1])
        path = ".\\images\\" + train[i][0]
        image = Image.open(path)
        trainData.append([np.array(image), label])
    return trainData

def importTestData():
    testData = []
    m = open(".\\splits\\test0.txt", "r")
    test = m.read()
    test = test.split('\n')
    for i in range(len(test)-1):
        test[i] = test[i].split(' ')
        label = setLabel(test[i][1])
        path = ".\\images\\" + test[i][0]
        image = Image.open(path)
        testData.append([np.array(image), label])
    return testData

trainData = importTrainData()
testData = importTestData()

trainImages = np.array([i[0] for i in trainData])
trainLabels = np.array([i[1] for i in trainData])

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.3))
model.add(Dense(25, activation = 'softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.fit(trainImages, trainLabels, batch_size = 50, epochs = 5, verbose = 1)

testImages = np.array([i[0] for i in testData])
testLabels = np.array([i[1] for i in testData])

loss, acc = model.evaluate(testImages, testLabels, verbose = 0)
print(acc * 100)
