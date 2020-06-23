# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def loadDataSet(fileName):
    """
    加载数据
    :param fileName: 文件名
    :return: 数据矩阵(n_sampels,2)
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 转换为float类型
        # map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def plotDataSet(filename):
    """
    绘制数据集
    :param filename: 文件名
    :return:
    """
    dataMat = loadDataSet(filename)
    n = len(dataMat)
    xcord = []
    ycord = []
    # 样本点
    for i in range(n):
        xcord.append(dataMat[i][1])
        ycord.append(dataMat[i][2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制样本点
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)
    plt.title('DataSet')
    plt.xlabel('X')
    plt.show()
    



def binSplitDataSet(dataSet, feature, value):
    """
    根据特征切分数据集合
    :param dataSet:数据集合
    :param feature:带切分的特征
    :param value:该特征的值
    :return:
    mat0 - 切分的数据集合0
    mat1 - 切分的数据集合1
    """
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    """
    生成叶结点
    :param dataSet: 数据集合
    :return: 目标变量均值
    """
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    """
    误差估计函数
    :param dataSet: 数据集合
    :return: 目标变量的总方差
    """
    # var表示方差，即各项-均值的平方求和后再除以N
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]
# {'spInd': 0, 'spVal': 0.499171, 'left': 101.35815937735848, 'right': -2.6377193297872341}
# {'spInd': 0, 'spVal': 0.499171, 'left': {'spInd': 0, 'spVal': 0.729397, 'left': {'spInd': 0, 'spVal': 0.952833, 'left': {'spInd': 0, 'spVal': 0.958512, 'left': 105.24862350000001, 'right': 112.42895575000001}, 'right': {'spInd': 0, 'spVal': 0.759504, 'left': {'spInd': 0, 'spVal': 0.790312, 'left': {'spInd': 0, 'spVal': 0.833026, 'left': {'spInd': 0, 'spVal': 0.944221, 'left': 87.310387500000004, 'right': {'spInd': 0, 'spVal': 0.85497, 'left': {'spInd': 0, 'spVal': 0.910975, 'left': 96.452866999999998, 'right': {'spInd': 0, 'spVal': 0.892999, 'left': 104.82540899999999, 'right': {'spInd': 0, 'spVal': 0.872883, 'left': 95.181792999999999, 'right': 102.25234449999999}}}, 'right': 95.275843166666661}}, 'right': {'spInd': 0, 'spVal': 0.811602, 'left': 81.110151999999999, 'right': 88.784498800000009}}, 'right': 102.35780185714285}, 'right': 78.085643250000004}}, 'right': {'spInd': 0, 'spVal': 0.640515, 'left': {'spInd': 0, 'spVal': 0.666452, 'left': {'spInd': 0, 'spVal': 0.706961, 'left': 114.554706, 'right': {'spInd': 0, 'spVal': 0.698472, 'left': 104.82495374999999, 'right': 108.92921799999999}}, 'right': 114.15162428571431}, 'right': {'spInd': 0, 'spVal': 0.613004, 'left': 93.673449714285724, 'right': {'spInd': 0, 'spVal': 0.582311, 'left': 123.2101316, 'right': {'spInd': 0, 'spVal': 0.553797, 'left': 97.200180249999988, 'right': {'spInd': 0, 'spVal': 0.51915, 'left': {'spInd': 0, 'spVal': 0.543843, 'left': 109.38961049999999, 'right': 110.979946}, 'right': 101.73699325000001}}}}}}, 'right': {'spInd': 0, 'spVal': 0.457563, 'left': {'spInd': 0, 'spVal': 0.467383, 'left': 12.50675925, 'right': 3.4331330000000007}, 'right': {'spInd': 0, 'spVal': 0.126833, 'left': {'spInd': 0, 'spVal': 0.373501, 'left': {'spInd': 0, 'spVal': 0.437652, 'left': -12.558604833333334, 'right': {'spInd': 0, 'spVal': 0.412516, 'left': 14.38417875, 'right': {'spInd': 0, 'spVal': 0.385021, 'left': -0.89235549999999952, 'right': 3.6584772500000016}}}, 'right': {'spInd': 0, 'spVal': 0.335182, 'left': {'spInd': 0, 'spVal': 0.350725, 'left': -15.085111749999999, 'right': -22.693879600000002}, 'right': {'spInd': 0, 'spVal': 0.324274, 'left': 15.059290750000001, 'right': {'spInd': 0, 'spVal': 0.297107, 'left': -19.994155200000002, 'right': {'spInd': 0, 'spVal': 0.166765, 'left': {'spInd': 0, 'spVal': 0.202161, 'left': {'spInd': 0, 'spVal': 0.217214, 'left': {'spInd': 0, 'spVal': 0.228473, 'left': {'spInd': 0, 'spVal': 0.25807, 'left': 0.40377471428571476, 'right': -13.070501}, 'right': 6.770429}, 'right': -11.822278500000001}, 'right': 3.4496025000000001}, 'right': {'spInd': 0, 'spVal': 0.156067, 'left': -12.107972500000001, 'right': -6.2479000000000013}}}}}}, 'right': {'spInd': 0, 'spVal': 0.084661, 'left': 6.5098432857142843, 'right': {'spInd': 0, 'spVal': 0.044737, 'left': -2.5443927142857148, 'right': 4.0916259999999998}}}}}

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    找到数据的最佳二元切分方式函数
    :param dataSet: 数据集合
    :param leafType: 生成叶结点的函数
    :param errType: 误差估计函数
    :param ops: 用户定义的参数构成的元组
    :return:
    bestIndex - 最佳切分特征
    bestValue - 最佳特征值
    """
    # tolS：允许的误差下降值
    tolS = ops[0]
    # tolN：切分的最小样本数
    tolN = ops[1]
    # 如果当前所有值相等，则退出（根据set的特性只保留不重复的元素）
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    # m-n_samples, n-2
    m, n = np.shape(dataSet)
    # 默认最后一个特征为最佳切分特征，计算其误差估计
    S = errType(dataSet)
    # 分别为最佳误差，最佳特征切分的索引值，最佳特征值
    bestS = float('inf')
    bestIndex = 0
    bestValue = 0
    # 遍历所有特征
    for featIndex in range(n-1):  # 遍历所有特征值
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            # 根据特征和特征的值切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果数据少于tolN，则退出剪枝操作
            if(np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            # 计算误差估计,寻找newS的最小值
            newS = errType(mat0) + errType(mat1)
            # 如果误差估计更小，则更新特征索引值和特征值
            if newS < bestS:
                # 特征索引
                bestIndex = featIndex
                # 分割标准
                bestValue = splitVal
                # 更新目标函数的最小值
                bestS = newS
    # 如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    # 根据最佳的切分特征和特征值切分数据集合
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分出的数据集很小则退出
    if(np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    # 返回最佳切分特征和特征值
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    """
    树构建函数
    :param dataSet: 数据集合(n_sampels,2)
    :param leafType: 生成叶结点的函数
    :param errType: 误差估计函数
    :param ops: 用户定义的参数构成的元组
    :return: 构建的回归树
    """
    # 选择最佳切分特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 如果没有特征，则返回特征值
    if feat == None:
        return val
    # 回归树
    retTree = {}
    # 分割特征索引
    retTree['spInd'] = feat
    # 分割标准
    retTree['spVal'] = val
    # 分成左数据集和右数据集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 创建左子树和右子树 递归
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree




if __name__ == '__main__':
    filename = 'ex0.txt'
    plotDataSet(filename)
    dataMat = loadDataSet(filename)
    dataMat = np.mat(dataMat)
    tree = createTree(dataMat)
    print(tree)
"""
{'spInd': 1, 'spVal': 0.39435, '
left': {'spInd': 1, 'spVal': 0.582002, 
  'left': {'spInd': 1, 'spVal': 0.797583, 
    'left': 3.9871631999999999, 
    'right': 2.9836209534883724}, 
  'right': 1.980035071428571}, 
'right': {'spInd': 1, 'spVal': 0.197834, 
  'left': 1.0289583666666666, 
  'right': -0.023838155555555553}}

"""




