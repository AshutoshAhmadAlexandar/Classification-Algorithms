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
    bestGain , bestCol, bestVal = 0, 0, 0
    if len(data) == 0:
        return bestGain, bestCol, bestVal

    curGINI = GINI(data)
    features = len(data[0]) - 1

    for col in range(features):
        values = set([row[col] for row in data])
        for val in values:
            leftClass, rightClass = partition(data, col, val)
            if len(leftClass) == 0 or len(rightClass) == 0:
                continue
            total = (len(rightClass) + len(leftClass))
            probRight = float(len(rightClass) / total)
            probleft = float(len(leftClass) / total)
            gain = curGINI - (probRight * GINI(rightClass)) - (probleft * GINI(leftClass))

            if gain >= bestGain:
                bestGain = gain
                bestCol = col
                bestVal = val

    return bestGain, bestCol, bestVal

class TreeNode(object):
    def __init__(self,col,val, left, right, rightClass, leftClass):
        self.col = col
        self.val = val
        self.left = left
        self.right = right
        self.rightClass = rightClass
        self.leftClass=leftClass

def updateNode(data,col, val):
    rightClass=0
    leftClass=0
    for row in data:
        if int(row[-1])==1:
            rightClass +=1
        else:
            leftClass +=1
    return TreeNode(col, val, None, None,rightClass, leftClass)
    
def buildTree(data):
    gain, col, val = bestGainSplit(data)
    if gain == 0:
        return updateNode(data,None,None)
    leftClass, rightClass = partition(data, col, val)
    node = TreeNode(col,val, None, None,-1,-1)
    node.left = buildTree(leftClass)
    node.right = buildTree(rightClass)
    return node

def displayTree(root, tab):
    tab+="      "
    if (root.left == None) and (root.right == None):
        if root.rightClass >= root.leftClass:
            print(tab + "   ->(Class 1)" )
        else:
            print(tab + "   ->(Class 0)" )
        return
    print(tab  + "SPLIT" + ": " + str(root.val))
    print( tab + '   ->left:')
    displayTree(root.left, tab )
    print( tab + '   ->Right:')
    displayTree(root.right, tab )
data=[]
input_file = "../project3_dataset4.txt"
data = np.genfromtxt(input_file, dtype=None)
data = np.array(data)

root = None
root = buildTree(data)
displayTree(root, "")