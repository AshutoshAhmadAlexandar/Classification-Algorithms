import numpy as np
import math
def dist(a, b):
    return (np.dot(a - b, a - b)) ** .5

### K nearest neighbour algorithm

def knnAlgo(train_data, label, point, k):
    list = []
    for i in range(len(train_data)):
        temp = dist(train_data[i], point)
        list.append([temp, i])
    list = sorted(list, key=lambda tup: tup[0])
    ind_dict = dict()
    for i in range(k):
        ind_dict.setdefault(label[list[i][1]], 0)
        ind_dict[label[list[i][1]]] += 1
    max_label = max(ind_dict, key=ind_dict.get)
    return max_label


# dictlist = [dict() for x in range(len(data[0])-1)]
def normalizeData(data):
    data_norm = []
    train_mean = []
    train_sd = []
    for i in range(data.shape[1]):
        col = data[0:data.shape[0], i]
        if col[0].isalpha():
            print('Alphabet')
            fcol = np.zeros(shape=col.shape)
            keys = set(col)
            values = list(range(0, len(keys)))
            dictionary = dict(zip(keys, values))
            for idx in range(len(col)):
                fcol[idx] = (float(dictionary.get(col[idx])))
            mean = np.mean(fcol)
            sd = np.std(fcol)
            ncol = (fcol - mean) / sd
            data_norm.append(ncol)
            train_mean.append(mean)
            train_sd.append(sd)
        else:
            fcol = col.astype(np.float)
            mean = np.mean(fcol)
            sd = np.std(fcol)
            ncol = (fcol - mean) / sd
            data_norm.append(ncol)
            train_mean.append(mean)
            train_sd.append(sd)
    data_norm = np.transpose(data_norm)
    return data_norm, train_mean, train_sd


def normalizeDataTest(data, mean, sd):
    data_norm = []
    for i in range(data.shape[1]):
        col = data[0:data.shape[0], i]
        if col[0].isalpha():
            print('Alphabet')
            fcol = np.zeros(shape=col.shape)
            keys = set(col)
            values = list(range(0, len(keys)))
            dictionary = dict(zip(keys, values))
            for idx in range(len(col)):
                fcol[idx] = (float(dictionary.get(col[idx])))
            alist=[]
            for x in fcol:
                alist.append((x- mean[i])/sd[i])
            data_norm.append(alist)
        else:
            fcol = col.astype(np.float)
            alist = []
            for x in fcol:
                alist.append((x - mean[i]) / sd[i])
            data_norm.append(alist)
    data_norm = np.transpose(data_norm)
    return data_norm

input_file = "../project3_dataset3_train.txt"
kVal = int(input("Enter the K value = "))
data = np.loadtxt(input_file, dtype=str)
data = np.array(data)
train_label = data[:, data.shape[1] - 1]
data = data[:, :data.shape[1] - 1]

#reading the training file
train_data, train_mean_col, train_sd_col= normalizeData(data)

#reading the test file
input_file1 = "../project3_dataset3_test.txt"
data1 = np.loadtxt(input_file1, dtype=str)
data1 = np.array(data1)
test_label = data1[:, data1.shape[1] - 1]
data1 = data1[:, :data1.shape[1] - 1]

#reading the training file
test_data= normalizeDataTest(data1, train_mean_col, train_sd_col)

truePositive = 0
trueNegative = 0
falsePositive = 0
falseNegative = 0

predicted_class = []

for tVal in test_data:
    d = knnAlgo(train_data, train_label, tVal, kVal)
    predicted_class.append(d)

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
Precision = float(truePositive / (truePositive + falsePositive))
Precision *= 100
Recall = (float(truePositive) / (truePositive + falseNegative))
Recall *= 100
FMeasure = float((2 * truePositive) / ((2 * truePositive) + falseNegative + falsePositive))

print('Accuracy:', Accuracy)
print('Precision:', Precision)
print('Recall:', Recall)
print('FMeasure:', FMeasure)

