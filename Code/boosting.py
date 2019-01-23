import numpy as np
import numbers

# to identify the classes in the dataset
def classCount(data):
    dictCount = {}
    for row in data:
        label = row[-1]
        if label not in dictCount:
            dictCount[label] = 0
        dictCount[label] += 1
    return dictCount

def GINI(data):
    counts = classCount(data)
    giniVal = 1
    for lbl in counts:
        prob = counts[lbl] / float(len(data))
        giniVal -= prob ** 2
    return giniVal

def partition(data, col, val):
    right = []
    left = []
    for row in data:
        if isinstance(val, numbers.Number):
            val = float(val)
            col= int(col)
            if row[col] >= val:
                right.append(row)
            else:
                left.append(row)
        else:
            if row[col] == val:
                right.append(row)
            else:
                left.append(row)
    return left, right

def bestGainSplit(data):
    bestGain, bestCol, bestVal = 0, 0, 0
    if len(data) == 0:
        return bestGain, bestCol, bestVal

    curGINI = GINI(data)
    features = len(data[0]) - 1

    for col in range(features):
        values = set([row[col] for row in data])
        for val in values:
            leftClass, rightClass = partition(data, col, val)
            if isinstance(val, numbers.Number):
                val = float(val)
                col = int(col)
            if len(leftClass) == 0 or len(rightClass) == 0:
                continue
            total = (len(rightClass) + len(leftClass))
            probRight = float(len(rightClass) / total)
            probleft = float(len(leftClass) / total)
            gain = curGINI - (probleft * GINI(leftClass)) - (probRight * GINI(rightClass))

            if gain >= bestGain:
                bestGain = gain
                bestCol = col
                bestVal = val

    return bestGain, bestCol, bestVal


class TreeNode(object):
    def __init__(self, col, val, left, right, rightClass, leftClass):
        self.col = col
        self.val = val
        self.left = left
        self.right = right
        self.rightClass = rightClass
        self.leftClass = leftClass


def updateNode(data, col, val):
    rightClass = 0
    leftClass = 0
    for row in data:
        if int(row[-1]) == 1:
            rightClass += 1
        else:
            leftClass += 1
    return TreeNode(col, val, None, None, rightClass, leftClass)


def buildTree(data):
    gain, col, val = bestGainSplit(data)
    if gain == 0:
        return updateNode(data, None, None)
    leftClass, rightClass = partition(data, col, val)
    if isinstance(val, numbers.Number):
        val = float(val)
        col = int(col)
    node = TreeNode(col, val, None, None, -1, -1)
    node.left = buildTree(leftClass)
    node.right = buildTree(rightClass)
    return node


def predictClass(root, data):
    if root.left == None and root.right == None:
        if (root.rightClass == -1) and (root.leftClass == -1):
            print("")
        if (root.rightClass >= root.leftClass):
            return 1
        else:
            return 0
    if isinstance(root.val, numbers.Number):
        if (data[root.col] >= root.val):
            return predictClass(root.right, data)
        else:
            return predictClass(root.left, data)
    else:
        if data[root.col] == root.val:
            return predictClass(root.right, data)
        else:
            return predictClass(root.left, data)


def split_data(data, iteration, no_of_folds):
    batch = len(data) / no_of_folds
    training_first = data[:int((iteration - 1) * batch)]
    testing_data = data[int((iteration - 1) * batch):int(iteration * batch)]
    training_last = data[int(iteration * batch):]
    training_data = np.concatenate((training_first, training_last), axis=0)
    return training_data, testing_data

def weightBagging(weight,Data,bagSize):
    bag = np.random.choice(Data, bagSize, p=weight,replace=True)
    return bag

input_file = "../project3_dataset2.txt"
data = np.genfromtxt(input_file, dtype=None)
data = np.array(data)

fold = 10
bagSizeRatio = 1
numLearners = int(input("Enter the number of Learners = "))

totalAccuracy = []
totalPrecision = []
totalRecall = []
totalFMeasure = []

for i in range(fold):
    train_data, test_data = split_data(data, i + 1, 10)
    lengthTraining = len(train_data)
    bagSize = np.ceil(lengthTraining * bagSizeRatio)
    bagSize = int(bagSize)

    initialWeight = float(1 / lengthTraining)
    weights = []
    weights = initialWeight * np.ones(lengthTraining)

    allAlpha = []
    allLearners = []
    root = None
    L = 0
    while (L < numLearners):
        ##choose sample and call learner
        sampleData = []
        sampleData = weightBagging(weights, train_data, bagSize)
        root = buildTree(sampleData)
        allLearners.append(root)
        ##calculate error
        error = 0.0
        denominatorOfError = 0.0
        predictions = []
        for rowId in range(lengthTraining):
            denominatorOfError += weights[rowId]
            label = predictClass(root, train_data[rowId])
            predictions.append(label)
            if label != train_data[rowId][-1]:
                error += weights[rowId]

        error = float(error / denominatorOfError)
        if (error > 0.5):
            continue
        else:
            L += 1

        alpha = 0.0
        alpha = float((.5) * (np.log((1 - error) / error)))
        allAlpha.append(alpha)
        ##update weights
        for rowId in range(lengthTraining):
            if predictions[rowId] == train_data[rowId][-1]:
                weights[rowId] = weights[rowId] * np.exp(-1 * alpha)
            else:
                weights[rowId] = weights[rowId] * np.exp(1 * alpha)
        weights /= np.sum(weights)

    predicitonForDifferentLearners = []
    lengthOfTestData = len(test_data)
    numLearners = len(allLearners)
    for learner in allLearners:
        testingprediction = []
        for line in test_data:
            label = predictClass(learner, line)
            testingprediction.append(int(label))
        predicitonForDifferentLearners.append(testingprediction)

    finalPredictedLabels = []
    alphaSum = np.sum(allAlpha)

    for col in range(lengthOfTestData):
        prediction = 0.0
        for learner in range(numLearners):
            prediction += allAlpha[learner] * predicitonForDifferentLearners[learner][col]

        prediction = float(prediction / alphaSum)
        # print(prediction)
        if (prediction >= 0.5):
            finalPredictedLabels.append(1)
        else:
            finalPredictedLabels.append(0)

    truePositive = 0
    trueNegative = 0
    falseNegative = 0
    falsePositive = 0

    for row in range(lengthOfTestData):
        if (finalPredictedLabels[row] == test_data[row][-1]):
            if finalPredictedLabels[row] == 0:
                trueNegative += 1
            else:
                truePositive += 1
        else:
            if (finalPredictedLabels[row] == 0):
                falseNegative += 1
            else:
                falsePositive += 1
    Accuracy = float(truePositive + trueNegative) / (truePositive + trueNegative + falseNegative + falsePositive)
    Precision = float(truePositive / (truePositive + falsePositive))
    Recall = float(truePositive / (truePositive + falseNegative))
    FMeasure = float((2 * truePositive) / ((2 * truePositive) + falseNegative + falsePositive))

    Accuracy *= 100
    Precision *= 100
    Recall *= 100
    totalAccuracy.append(Accuracy)
    totalPrecision.append(Precision)
    totalRecall.append(Recall)
    totalFMeasure.append(FMeasure)
    print("Iteration", i+1)

print("averageAccuracy  : " + str(np.sum(totalAccuracy) / fold))
print("averagePrecision  : " + str(np.sum(totalPrecision) / fold))
print("averageRecall  : " + str(np.sum(totalRecall) / fold))
print("averageFMeasure  : " + str(np.sum(totalFMeasure) / fold))