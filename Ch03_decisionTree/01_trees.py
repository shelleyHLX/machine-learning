'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def createDataSet_loan():
    # 数据集
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    # 分类属性
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    # 返回数据集和分类属性
    return dataSet, labels


def createDataSet_fish():
    """
    特征-- 不浮出水面是否可以生存，以及是否有脚蹼，
    类别--鱼类和非鱼类
    :return:数据集(特征+类别)，特征的意义
    """
    dataSet = [[1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0,0,'no']]
    labels = ['no surfacing','flippers']
    # change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    """
    计算此时数据集的香农熵
    :param dataSet:数据集，(fea1,fea2,...,label)
    :return:int，香农熵
    """
    numEntries = len(dataSet)  # 样本个数
    labelCounts = {}  # 计算每个类别的样本个数
    for featVec in dataSet:  # 每个样本
        currentLabel = featVec[-1]  # 该样本的类别
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries  # 计算该类别的信息
        shannonEnt -= prob * log(prob,2)  # log base 2，把每个类别的信息的期望相加
    return shannonEnt
    
def splitDataSet(dataSet, axis, value):
    """
    去掉下标为axis的特征，一整列
    :param dataSet:数据集，(fea,fea,...,label)
    :param axis: 某个特征的下标
    :param value: 该特征（下标为axis）的某个取值
    :return:  特征值=value的数据集，并且去除该特征
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:  # 对特征值=value的特征进行删除
            reducedFeatVec = featVec[:axis]  #
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    根据信息增益获得信息增益最大的特征的索引
    :param dataSet:(fea,fea,...,label)
    :return: 信息增益最大的特征的索引值
    """
    numFeatures = len(dataSet[0]) - 1   # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的香农熵
    bestInfoGain = 0.0  # 信息增益
    bestFeature = -1  # 最优特征的索引值
    for i in range(numFeatures):  # 遍历所有特征
        featList = [example[i] for example in dataSet]  # 获取dataSet的第i个特征
        uniqueVals = set(featList)  # 去除重复特征
        newEntropy = 0.0
        for value in uniqueVals:  # 遍历一个特征的所有取值
            # 特征值为value的子数据集，并且除去该特征
            subDataSet = splitDataSet(dataSet, i, value)  # 数据集，第几个特征，该特征的一个取值
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy  # 信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain  # 更新信息增益，找到最大的信息增益
            bestFeature = i  # 记录信息增益最大的特征的索引值
    return bestFeature  #

def majorityCnt(classList):
    """
    统计classList中出现次数最多的元素（类标签）
    服务于递归第两个终止条件
    :param classList: 类标签列表
    :return: 出现次数最多的元素（类标签）
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels, featLabels):
    """
    创建决策树（ID3算法）
    递归有两个终止条件：
    1、所有的类标签完全相同，直接返回类标签
    2、用完所有标签但是得不到唯一类别的分组，即特征不够用，挑选出现数量最多的类别作为返回
    :param dataSet:
    :param labels:  特征的意义
    :param featLabels: 存储选择的最优特征标签
    :return:
    """
    classList = [example[-1] for example in dataSet]  # 类别标签，是否是鱼类
    if classList.count(classList[0]) == len(classList): 
        return classList[0]  # 如果类别完全相同则停止继续划分
    if len(dataSet[0]) == 1: # 如果数据集没有别的特征，停止划分
        return majorityCnt(classList)  # 遍历完所有特征时返回出现次数最多的类标签
    bestFeat = chooseBestFeatureToSplit(dataSet)  # 选择最优特征
    bestFeatLabel = labels[bestFeat]  # 最优特征的名称，意义
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel:{}}  # 根据最优特征的名称生成树
    del(labels[bestFeat])  # 删除已经使用的特征标签
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 递归建立决策树
    for value in uniqueVals:  # 特征值为value的子数据集，并且除去该特征
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),
                                                  labels, featLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    # 获取决策树结点
    firstStr = next(iter(inputTree))
    # 下一个字典
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


def getNumLeafs(myTree):
    """
    获取决策树叶子结点的数目
    :param myTree: 决策树
    :return: None
    """
    # 初始化叶子
    numLeafs = 0
    # python3中myTree.keys()返回的是dict_keys,不是list,所以不能用
    # myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    # next() 返回迭代器的下一个项目 next(iterator[, default])
    firstStr = next(iter(myTree))
    # 获取下一组字典
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此节点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    """
    获取决策树的层数
    :param myTree: 决策树
    :return: 决策树的层数
    """
    # 初始化决策树深度
    maxDepth = 0
    # python3中myTree.keys()返回的是dict_keys,不是list,所以不能用
    # myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    # next() 返回迭代器的下一个项目 next(iterator[, default])
    firstStr = next(iter(myTree))
    # 获取下一个字典
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此节点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        # 更新最深层数
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    # 返回决策树的层数
    return maxDepth


def plotMidText(cntrPt, parentPt, txtString):
    """
    标注有向边属性值
    :param cntrPt: 用于计算标注位置
    :param parentPt: 用于计算标注位置
    :param txtString: 标注内容
    :return: None
    """
    # 计算标注位置（箭头起始位置的中点处）
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制结点
    :param nodeTxt: 结点名
    :param centerPt: 文本位置
    :param parentPt: 标注的箭头位置
    :param nodeType: 结点格式
    :return: None
    """
    # 定义箭头格式
    arrow_args = dict(arrowstyle="<-")
    # 设置中文字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    # 绘制结点createPlot.ax1创建绘图区
    # annotate是关于一个数据点的文本
    # nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType,
                            arrowprops=arrow_args, FontProperties=font)


def plotTree(myTree, parentPt, nodeTxt):
    """
    绘制决策树
    :param myTree: 决策树（字典）
    :param parentPt: 标注的内容
    :param nodeTxt: 结点名
    :return: None
    """
    # 设置结点格式boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    # 设置叶结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")
    # 获取决策树叶结点数目，决定了树的宽度
    numLeafs = getNumLeafs(myTree)
    # 获取决策树层数
    depth = getTreeDepth(myTree)
    # 下个字典
    firstStr = next(iter(myTree))
    # 中心位置
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yoff)
    # 标注有向边属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制结点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 下一个字典，也就是继续绘制结点
    secondDict = myTree[firstStr]
    # y偏移
    plotTree.yoff = plotTree.yoff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            # 不是叶结点，递归调用继续绘制
            plotTree(secondDict[key], cntrPt, str(key))
        # 如果是叶结点，绘制叶结点，并标注有向边属性值
        else:
            plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)
            plotMidText((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD


def createPlot(inTree):
    """
    创建绘图面板
    :param inTree: 决策树（字典）
    :return: None
    """
    # 创建fig
    fig = plt.figure(1, facecolor="white")
    # 清空fig
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # 去掉x、y轴
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 获取决策树叶结点数目
    plotTree.totalW = float(getNumLeafs(inTree))
    # 获取决策树层数
    plotTree.totalD = float(getTreeDepth(inTree))
    # x偏移
    plotTree.xoff = -0.5 / plotTree.totalW
    plotTree.yoff = 1.0
    # 绘制决策树
    plotTree(inTree, (0.5, 1.0), '')
    # 显示绘制结果
    plt.show()


def main():
    dataSet, features = createDataSet_fish()
    featLabels = []
    myTree = createTree(dataSet, features, featLabels)
    # 保存树
    storeTree(myTree, 'classifierStorage.txt')
    # 加载树
    myTree = grabTree('classifierStorage.txt')
    print(myTree)
    # 测试数据
    testVec = [1, 1]  # fish
    # testVec = [1, 1, 0, 0]  # loan
    result = classify(myTree, featLabels, testVec)
    if result == 'yes':
        print('是鱼类')
    if result == 'no':
        print('不是鱼类')
    # print(myTree)
    createPlot(myTree)
    # print(dataSet)
    # print(calcShannonEnt(dataSet))
    # print("最优特征索引值:" + str(chooseBestFeatureToSplit(dataSet)))

if __name__ == '__main__':
    main()