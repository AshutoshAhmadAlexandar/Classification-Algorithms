import numpy as np
import math


def attr_type(x):
    numeric_indexes = []
    non_numeric_indexes = []
    for i in range(len(x)):
        try:
            float(x[i])
            numeric_indexes.append(i)
        except ValueError:
            non_numeric_indexes.append(i)
    return numeric_indexes, non_numeric_indexes

def summarize(dataset):
    computed_mean_arr_0 = []
    standard_deviation_arr_0 = []
    computed_mean_arr_1 = []
    standard_deviation_arr_1 = []
    for i in range(len(numeric_indexes)):
        train_0 = dataset[np.where(dataset[:, -1] == "0"), numeric_indexes[i]]
        train_1 = dataset[np.where(dataset[:, -1] == "1"), numeric_indexes[i]]

        train_0 = np.array(train_0, dtype=float)
        train_1 = np.array(train_1, dtype=float)

        computed_mean_arr_0.append(np.mean(train_0))
        standard_deviation_arr_0.append(np.std(train_0))

        computed_mean_arr_1.append(np.mean(train_1))
        standard_deviation_arr_1.append(np.std(train_1))
    return computed_mean_arr_0, standard_deviation_arr_0, computed_mean_arr_1, standard_deviation_arr_1


def calculateProbability(mean_0,std_0,mean_1,std_1,testdata):

  pro_numeric_0 = np.ones(shape=(len(testdata)))
  pro_numeric_1 = np.ones(shape=(len(testdata)))
  for i in range(len(testdata)):
      for j in range(len(numeric_indexes)):
          x = np.float64(testdata[i, numeric_indexes[j]]) - mean_0[j]
          exp = math.exp(-(math.pow(x, 2) / (2 * math.pow(std_0[j], 2))))
          pro_numeric_0[i] = pro_numeric_0[i] * (
              (1 / (math.sqrt(2 * math.pi) * std_0[j])) * exp)

          x = np.float64(testdata[i, numeric_indexes[j]]) - mean_1[j]
          exp = math.exp(-(math.pow(x, 2) / (2 * math.pow(std_1[j], 2))))
          pro_numeric_1[i] = pro_numeric_1[i] * (
              (1 / (math.sqrt(2 * math.pi) * std_1[j])) * exp)

  return pro_numeric_0, pro_numeric_1

def getPredictions(class_0_prob, class_1_prob, prob_0_num, prob_1_num, pro_nominal_0, pro_nominal_1,
                                      testdata):
    predicted_labels = np.zeros(shape=(len(testdata)), dtype=int)
    for i in range(len(predicted_labels)):
        # print("pro_0: ",pro_0*pro_numeric_0[i]*pro_nominal_0[i],"prob_1: ",pro_1*pro_numeric_1[i]*pro_nominal_1[i])
        prob_sum = class_0_prob * prob_0_num[i] * pro_nominal_0[i] + class_1_prob * prob_1_num[i] * pro_nominal_1[i]
        prob_0 = class_0_prob * prob_0_num[i] * pro_nominal_0[i] / prob_sum
        prob_1 = class_1_prob * prob_1_num[i] * pro_nominal_1[i] / prob_sum
        if(prob_1>prob_0):
            predicted_labels[i]=1
        #print("prob_0:", prob_0, "prob_1:", prob_1)

    return predicted_labels

def split(dataset,start,end):
    copy = list(dataset)
    testdata = []
    count=start
    while(count<end):
        testdata.append(copy.pop(start))
        count =count+1
    return testdata,copy
def categeroical_prob(traindata, testdata, non_numeric_indexes):
    len_0 = traindata[:, -1].tolist().count("0")
    len_1 = traindata[:, -1].tolist().count("1")
    pro_categeroical_0 = np.ones(shape=(len(testdata)))
    pro_categeroical_1 = np.ones(shape=(len(testdata)))
    for j in range(len(testdata)):
        for i in range(len(non_numeric_indexes)):
            pro_categeroical_0[j] = pro_categeroical_0[j] * (
                traindata[np.where(traindata[:, -1] == "0")][:, non_numeric_indexes[i]].tolist().count(testdata[j, non_numeric_indexes[i]]) / len_0)
            pro_categeroical_1[j] = pro_categeroical_1[j] * (
                traindata[np.where(traindata[:, -1] == "1")][:, non_numeric_indexes[i]].tolist().count(testdata[j, non_numeric_indexes[i]]) / len_1)
    return pro_categeroical_0, pro_categeroical_1

def getClassProbability(traindata):
    class_0_prob = traindata[:, -1].tolist().count("0") / len(traindata)
    class_1_prob = traindata[:, -1].tolist().count("1") / len(traindata)
    return class_0_prob, class_1_prob

input_file = '../project3_dataset1.txt'
data1 = np.loadtxt(input_file, dtype=bytes).astype(str)
data = np.array(data1)


data_norm=[]
numeric_indexes, non_numeric_indexes = attr_type(data[0, :len(data[0]) - 1])
accuracy = precision = recall = f_measure = 0
l=int(len(data)/10)

avgAccuracy = 0
avgPrecision = 0
avgRecall = 0
avgFMeasure = 0

for i in range(0,10):
    testdata,traindata=split(data,i*l,i*l+l)
    testdata = np.array(testdata)
    traindata = np.array(traindata)
    class_0_prob,class_1_prob=getClassProbability(traindata)
    mean_0,std_0,mean_1,std_1=summarize(traindata)
    prob_0_num,prob_1_num=calculateProbability(mean_0,std_0,mean_1,std_1,testdata)
    if(len(non_numeric_indexes)>0):
        pro_categeroical_0, pro_categeroical_1 = categeroical_prob(traindata, testdata, non_numeric_indexes)
    else:
        pro_categeroical_0 = np.ones(shape=(len(testdata)))
        pro_categeroical_1 = np.ones(shape=(len(testdata)))
    predicted_labels = getPredictions(class_0_prob, class_1_prob, prob_0_num, prob_1_num, pro_categeroical_0, pro_categeroical_1,testdata)
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    for j in range(len(testdata)):
        if predicted_labels[j] == int(testdata[j][-1]):
            if predicted_labels[j] == 0:
                trueNegative += 1
            else:
                truePositive += 1
        else:
            if predicted_labels[j] == 0:
                falseNegative += 1
            else:
                falsePositive += 1

    Accuracy = float(truePositive + trueNegative) / (truePositive + trueNegative + falseNegative + falsePositive)
    avgAccuracy = avgAccuracy+Accuracy
    if truePositive+falsePositive == 0:
        precision=0
    else:
        Precision = float(truePositive / (truePositive + falsePositive))
        avgPrecision = avgPrecision+Precision
    if truePositive+falseNegative == 0:
        Recall=0
    else:
        Recall = float(truePositive / (truePositive + falseNegative))
        avgRecall = avgRecall+Recall
    if truePositive+falseNegative+falsePositive == 0:
        FMeasure=0
    else:
        FMeasure = float((2 * truePositive) / ((2 * truePositive) + falseNegative + falsePositive))
        avgFMeasure = avgFMeasure+FMeasure
# print(avg/10)
print('Accuracy:', avgAccuracy*100 / 10)
print('Precision:', avgPrecision*100 / 10)
print('Recall:', avgRecall*100 / 10)
print('FMeasure:', avgFMeasure / 10)