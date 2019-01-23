from copy import deepcopy
import numpy as np
from sklearn.decomposition import PCA
import random
from matplotlib import pyplot as plt
import tkinter.filedialog


def dist(a, b):
    return (np.dot(a - b, a - b)) ** .5

### K nearest neighbour algorithm

def knnAlgo(train_data, label, point, k):
    list = []
    for i in range(0, len(train_data)):
        temp = dist(train_data[i], point)
        list.append([temp, i])
    list = sorted(list, key=lambda tup: tup[0])
    ind_dict = dict()
    for i in range(k):
        ind_dict.setdefault(label[list[i][1]], 0)
        ind_dict[label[list[i][1]]] += 1
    # print('Index dic is', ind_dict)
    max_label = max(ind_dict, key=ind_dict.get)
    return max_label


# Open a file dialog
kVal = int(input("Enter the K value = "))
# input_file = tkinter.filedialog.askopenfilename()
input_file = "../project3_dataset1.txt"
data = np.loadtxt(input_file, dtype=str)
data = np.array(data)

label = data[:, data.shape[1] - 1]
data = data[:, :data.shape[1] - 1]

data_norm = []
# dictlist = [dict() for x in range(len(data[0])-1)]
for i in range(data.shape[1]):
    col = data[0:data.shape[0], i]
    if col[0].isalpha():
        print('Alphabet')
        # fcol = [[0] for i in range()]
        fcol = np.zeros(shape=col.shape)
        keys = set(col)
        values = list(range(0, len(keys)))
        # print(type(values[0]))
        dictionary = dict(zip(keys, values))
        # dictlist[i] = dictionary
        # print('Dictlist:', dictlist)
        for idx in range(len(col)):
            fcol[idx] = (float(dictionary.get(col[idx])))
        mean = np.mean(fcol)
        sd = np.std(fcol)
        ncol = (fcol - mean) / sd
        data_norm.append(ncol)
    else:
        fcol = col.astype(np.float)
        mean = np.mean(fcol)
        sd = np.std(fcol)
        ncol = (fcol - mean) / sd
        # data_norm.append (fcol / fcol.max())
        data_norm.append(ncol)
data_norm = np.transpose(data_norm)


def split_data(data, iteration, no_of_folds):
    batch = len(data) / no_of_folds
    training_first = data[:int((iteration - 1) * batch)]
    testing_data = data[int((iteration - 1) * batch):int(iteration * batch)]
    training_last = data[int(iteration * batch):]
    # print(type(training_first))
    training_data = np.concatenate((training_first, training_last), axis=0)
    return training_data, testing_data


truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0
avgAccuracy = 0
avgPrecision = 0
avgRecall = 0
avgFMeasure = 0

for i in range(0, 10):
    train_data, test_data = split_data(data_norm, i + 1, 10)
    train_label, test_label = split_data(label, i + 1, 10)
    # Call for the Knn calculation for each point to identify the class
    predicted_class = []
    for tVal in test_data:
        d = knnAlgo(train_data, train_label, tVal, kVal)
        predicted_class.append(d)

    posCount = 0
    # print(predicted_class)
    for j in range(len(test_label)):
        if predicted_class[j] == test_label[j]:
            if predicted_class[j] == '0':
                trueNegative += 1
            else:
                truePositive += 1
        else:
            if predicted_class[j] == '0':
                falseNegative += 1
            else:
                falsePositive += 1
    Accuracy = float(truePositive + trueNegative) / (truePositive + trueNegative + falseNegative + falsePositive)
    Accuracy *= 100
    avgAccuracy += Accuracy
    Precision = float(truePositive / (truePositive + falsePositive))
    Precision *= 100
    avgPrecision += Precision
    Recall = float(truePositive / (truePositive + falseNegative))
    Recall *= 100
    avgRecall += Recall
    FMeasure = float((2 * truePositive) / ((2 * truePositive) + falseNegative + falsePositive))
    avgFMeasure += FMeasure

print('Accuracy:', avgAccuracy / 10)
print('Precision:', avgPrecision / 10)
print('Recall:', avgRecall / 10)
print('FMeasure:', avgFMeasure / 10)