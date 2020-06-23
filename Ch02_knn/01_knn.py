# coding: utf-8
# Author: shelley
# 2020/5/916:05
from numpy import *
import numpy as np
import operator
from os import listdir


def classify0(inX, dataSet, labels, k):
    """
    knn分类算法
    :param inX: (1,3)
    :param dataSet: (n,3)
    :param labels: (n,3)
    :param k: int, k个相似度样本最大的样本，这些样本中相同标签最大的为测试样本的标签
    :return:测试样本的标签
    """
    dataSetSize = dataSet.shape[0]  # 样本个数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # 测试样本-训练样本
    sqDiffMat = diffMat ** 2  # 平方差
    sqDistances = sqDiffMat.sum(axis=1)  # 将每一行的3个特征进行相加 sum(0)列相加，sum(1)行相加
    distances = sqDistances ** 0.5  # 开方
    sortedDistIndicies = distances.argsort()  # argsort函数返回的是distances值从小到大的--索引值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename):
    """
    读取文件的数据
    :param filename: 文件名
    :return: 特征矩阵[fea1,fea2,fea3](n*3)，标签[label](n*1)
    """
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # 获得文件的所有文本行
    returnMat = zeros((numberOfLines, 3))  # 定义存储数据的矩阵
    classLabelVector = []  # 定义存储标签的矩阵
    fr = open(filename)
    index = 0  # 表示第几个样本
    for line in fr.readlines():
        line = line.strip()  # 去掉每一行首尾的空白符，例如'\n','\r','\t',' '
        listFromLine = line.split('\t')  # 每一行根据\t进行切片
        returnMat[index, :] = listFromLine[0:3]  # 前三列是特征
        classLabelVector.append(int(listFromLine[-1]))  # 最后一列是标签
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    数据归一化
    :param dataSet: 特征矩阵[fea1,fea2,fea3](n*3)
    :return:归一化的特征矩阵，特征取值范围(1,3)，特征最小值(1,3)
    """
    minVals = dataSet.min(0)  # (1,3) 每一列的最小值，即一个特征中所有样本的最小值
    maxVals = dataSet.max(0)  # (1,3) dataSet.min(1)，每一行的最小值
    ranges = maxVals - minVals  # 每个特征的取值范围
    m = dataSet.shape[0]  # 样本个数
    #  原始值减去最小值（x-xmin）
    normDataSet = dataSet - np.tile(minVals, (m, 1))  # (n,3)复制n个minVals，
    # 差值处以最大值和最小值的差值（x-xmin）/（xmax-xmin）
    normDataSet = normDataSet / np.tile(ranges, (m, 1))  # element wise divide
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    dating数据分类
    :return:
    """
    hoRatio = 0.1  # 保留10%作为测试集
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')  # 加载数据
    normMat, ranges, minVals = autoNorm(datingDataMat)  # 数据归一化，返回归一化数据结果，数据范围，最小值
    m = normMat.shape[0]  # 样本个数
    numTestVecs = int(m * hoRatio)  # 测试样本个数
    errorCount = 0.0
    for i in range(numTestVecs):  # 对每个样本进行测试
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print("knn获得的标签: %d, 真实标签: %d" % (classifierResult, datingLabels[i]))

        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print ("错误率: %f %%" % (errorCount / float(numTestVecs)*100))

    print(errorCount)


def img2vector(filename):
    """

    :param filename:  文件名，有32行，一行32个字符
    :return:  文件里的数据，(1,1024)
    """
    returnVect = zeros((1, 1024))  # 定义(1,1024)形状的变量，32*32=1024
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()  # 读取一行数据
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])  # 每行数据顺序存储在returnVect
    return returnVect


def handwritingClassTest():
    """
    手写数字分类
    :return:
    """
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # 获得trainingDigits下所有文件的路径
    m = len(trainingFileList)  # 文件个数
    trainingMat = zeros((m, 1024))  # 样本个数*特征维度
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 获得其中一个文件名testDigits\0_0.txt
        fileStr = fileNameStr.split('.')[0]  # 去掉 .txt
        classNumStr = int(fileStr.split('_')[0])  # 按_进行切分，第一个字符串是类别
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)  # 获得特征矩阵
    testFileList = listdir('testDigits')  # 读取测试数据
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类结果为: %d, 真实结果为: %d" % (classifierResult, classNumStr))

        if (classifierResult != classNumStr): errorCount += 1.0
    print("\n分类错误的个数: %d" % errorCount)

    print("\n错误率: %f" % (errorCount / float(mTest)))


if __name__ == '__main__':
    # dating数据分类
    # datingClassTest()
    # 手写数字分类
    handwritingClassTest()
