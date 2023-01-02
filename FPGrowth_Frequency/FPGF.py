import math
from collections import Counter
import data_progress as dp
import pandas as pd
from mlxtend.frequent_patterns import association_rules


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue  # 节点名字
        self.count = numOccur  # 节点计数值
        self.nodeLink = None  # 用于链接相似的元素项
        self.parent = parentNode  # 父节点
        self.children = {}  # 子节点

    def inc(self, numOccur):
        self.count += numOccur

    # 打印树
    def print_tree(self, ind=1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.print_tree(ind + 1)


def createInitSet(dataSet):
    data_temp = []
    retDict = {}

    for trans in dataSet:
        data_temp = data_temp + list(trans)
        retDict[tuple(trans)] = retDict.get(tuple(trans), 0) + 1
    data_sort = Counter(data_temp)
    return data_sort, retDict


def createTree(data_sort, dataSet, minSup):
    headerTable = {}
    # 计算item出现频数
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    headerTable = {k: v for k, v in headerTable.items() if v >= minSup}
    # print('headerTable', headerTable)
    freqItemSet = set(headerTable.keys())
    # print('freqItemSet:', freqItemSet)
    if len(freqItemSet) == 0:
        return None, None
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    # print('headerTable: ', headerTable)

    # 创建树
    retTree = treeNode('Null Set', 1, None)
    mapping = [v[0] for v in sorted(data_sort.items(), key=lambda p: p[1], reverse=True)]
    for tranSet, count in dataSet.items():
        sort_dit = {}
        result = dict(Counter(tranSet))
        # print('result:', result)
        if len(tranSet) > 0:
            for tran in mapping:
                if tran in result.keys() and tran in headerTable.keys():
                    sort_dit[tran] = result[tran]
        # print('sort_dit:', sort_dit)
        sort_key_list = list(sort_dit.keys())
        sort_values_list = list(sort_dit.values())

        updateTree(sort_key_list, sort_values_list, retTree, headerTable, count)
    # 返回树型结构和头指针表
    return retTree, headerTable


def updateTree(key, values, inTree, headerTable, count):
    if len(key) != 0:
        # 检查第一个元素项是否作为子节点存在
        if key[0] in inTree.children:
            # 存在，更新计数
            inTree.children[key[0]].inc(values[0] * count)
        else:
            # 不存在，创建一个新的treeNode,将其作为一个新的子节点加入其中
            inTree.children[key[0]] = treeNode(key[0], values[0] * count, inTree)
            if headerTable[key[0]][1] == None:  # 更新头指针表
                headerTable[key[0]][1] = inTree.children[key[0]]
            else:
                updateHeader(headerTable[key[0]][1], inTree.children[key[0]])
        # inTree.print_tree()
        if len(key) > 1:
            # 不断迭代调用自身，每次调用都会删掉列表中的第一个元素
            updateTree(key[1::], values[1::], inTree.children[key[0]], headerTable, count)


# 更新头指针表，确保节点链接指向树中该元素项的每一个实例
def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def ascendTree(leafNode, prefixPath):
    # 迭代上溯整棵树
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


def mineTree(data_sort, inTree, headerTable, minSup, preFix, freqItemList):
    # 1.排序头指针表
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    # print('bigL:', bigL)
    # 从头指针表的底端开始
    results = []
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        # print('finalFrequent Item: ', newFreqSet)
        # print('newFreqSet:', newFreqSet)
        # 添加的频繁项列表
        freqItemList.append(newFreqSet)
        # 条件模式基
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        results.append([basePat, condPattBases])
        # print('condPattBases :', basePat, condPattBases)
        # 2.从条件模式基创建条件FP树
        myCondTree, myHead = createTree(data_sort, condPattBases, minSup)
        # print('head from conditional tree: ', myHead)
        # 3.挖掘条件FP树
        if myHead != None:
            # print('conditional tree for: ', newFreqSet)
            # myCondTree.disp(1)
            mineTree(data_sort, myCondTree, myHead, minSup, newFreqSet, freqItemList)
    return results



def loadData():
    skill_path = 'D:/Files/experiment/20220305/dkt/train_skills_all.txt'
    output_path = 'D:/Files/experiment/20220305/fpgk/'
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
    data_sort, initSet = createInitSet(data)
    myFPtree, myHeaderTab = createTree(data_sort, initSet, ms)
    myFPtree.print_tree()
    myFreqList = []
    results = mineTree(data_sort, myFPtree, myHeaderTab, ms, set([]), myFreqList)
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
# association_rules.to_csv(path_or_buf='D:/Files/experiment/20220305/fpgk/association_rules_FPGK.csv')
print(association_rules)

