import math
import data_progress as dp
import pandas as pd
from mlxtend.frequent_patterns import association_rules

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[tuple(trans)] = retDict.get(tuple(trans), 0) + 1
    # print(retDict)
    return retDict


def createTree(dataSet, minSup):
    headerTable = {}

    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    headerTable = {k: v for k, v in headerTable.items() if v >= minSup}
    # print('headerTable', headerTable)
    freqItemSet = set(headerTable.keys())

    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]

    retTree = treeNode('Null Set', 1, None)

    for tranSet, count in dataSet.items():

        localD = {}
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]

        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]

            updateTree(orderedItems, retTree, headerTable, count)

    return retTree, headerTable


def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:

        inTree.children[items[0]].inc(count)
    else:

        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)


def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode



def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[tuple(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]

    results = []
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        # print('finalFrequent Item: ', newFreqSet)

        freqItemList.append(newFreqSet)

        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        results.append([basePat, condPattBases])
        # print('condPattBases :', basePat, condPattBases)

        myCondTree, myHead = createTree(condPattBases, minSup)

        if myHead != None:
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
    return results


def loadData():
    skill_path = 'D:/Files/experiment/20220305/dkt/train_skills_all.txt'
    output_path = 'D:/Files/experiment/20220305/fpg/'
    dp.dp_skill4others(skill_path, output_path)
    mat_skill = dp.dp_tolist(output_path + 'skill_data.txt')
    for i in range(len(mat_skill)):
        if type(mat_skill[i]) == int:
            mat_skill[i] = [mat_skill[i]]
        if '' in mat_skill[i]:
            print('yes')
        mat_skill[i] = list(mat_skill[i])
    return mat_skill

def runFPGrowth(minSup):
    # data = loadSimpDat()
    data = loadData()
    ms = math.ceil(minSup * len(data))
    print("minSup:", ms)
    initSet = createInitSet(data)
    myFPtree, myHeaderTab = createTree(initSet, ms)
    myFPtree.disp()
    myFreqList = []
    results = mineTree(myFPtree, myHeaderTab, ms, set([]), myFreqList)
    return results


def calculate_support(results, t):
    ar = []
    for result in results:
        for k in result[1].keys():
            temp_rule = [result[0]] + list(k)
            ar.append([(result[1][k] / t) if (result[1][k] / t) < 1 else 1] + [temp_rule])
    ar = pd.DataFrame(ar, columns=['support', 'itemsets'])
    return ar

minSup = 0.02
results = runFPGrowth(minSup)
support_data = {}
result_list = []
for result in results:
    result_list.append(list(result[1].keys()))

ar = calculate_support(results, len(loadData()))
ar.sort_values(by='support', ascending=False, inplace=True)
association_rules = association_rules(ar, min_threshold=0.02, support_only=True, )
# association_rules.to_csv(path_or_buf='D:/Files/experiment/20220305/fpg/association_rules_FPG.csv')
print(association_rules)
