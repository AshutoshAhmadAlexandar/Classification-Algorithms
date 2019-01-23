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

def bestGainSplit(data, m):
    bestGain, bestCol, bestVal = 0, 0, 0
    if len(data) == 0:
        return bestGain, bestCol, bestVal

    curGINI = GINI(data)
    features = randomCol(len(data[0])-1, m)
    for col in features:
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


def buildTree(data, m):
    gain, col, val = bestGainSplit(data, m)
    if gain == 0:
        return updateNode(data, None, None)
    leftClass, rightClass = partition(data, col, val)
    if isinstance(val, numbers.Number):
        val = float(val)
        col = int(col)
    node = TreeNode(col, val, None, None, -1, -1)
    node.left = buildTree(leftClass, m)
    node.right = buildTree(rightClass, m)
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

def baggingSample(Data,bagSize):
    bag = np.random.choice(Data, bagSize,replace=True)
    return bag

def randomCol(n, m):
    return np.random.choice(n, m, replace=False)

input_file = "../project3_dataset2.txt"
data = np.genfromtxt(input_file, dtype=None)
data = np.array(data)


bagSizeRatio = 1
fold = 10

t = int(input("Enter the number of bags = "))

m = int(round(0.2 * len(data[0])))

totalAccuracy = []
totalPrecision = []
totalRecall = []
totalFMeasure = []

for i in range(fold):
    train_data, test_data = split_data(data, i + 1, 10)
    lengthTraining = len(train_data)
    bagSize = np.ceil(lengthTraining * bagSizeRatio)
    bagSize = int(bagSize)

    trees=[]
    root = None
    index = 0
    while (index < t):
        sampleData = []
        sampleData = baggingSample(train_data, bagSize)
        root = buildTree(sampleData, m)
        trees.append(root)
        index+=1

    # Testing  Data validation
    prediction =[]
    for tdata in test_data:
        node_dict = {}
        for node in trees:
            res = predictClass(node, tdata)
            if node_dict.__contains__(res):
                node_dict[res] += 1
            else:
                node_dict[res] = 1

        max_class, max_val = None, 0
        for key in node_dict.keys():
            if node_dict[key] > max_val:
                max_val = node_dict[key]
                max_class = key
        prediction.append(max_class)

    truePositive = 0
    trueNegative = 0
    falseNegative = 0
    falsePositive = 0

    for row in range(len(test_data)):
        if (prediction[row] == test_data[row][-1]):
            if prediction[row] == 0:
                trueNegative += 1
            else:
                truePositive += 1
        else:
            if (prediction[row] == 0):
                falseNegative += 1
            else:
                falsePositive += 1
    try:
        Accuracy = float(truePositive + trueNegative) / (truePositive + trueNegative + falseNegative + falsePositive)
        Precision = float(truePositive / (truePositive + falsePositive))
        Recall = float(truePositive / (truePositive + falseNegative))
        FMeasure = float((2 * truePositive) / ((2 * truePositive) + falseNegative + falsePositive))
    except:
        print("error in division")
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