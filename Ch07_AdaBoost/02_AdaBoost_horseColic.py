# coding: utf-8
# Author: shelley
# 2020/5/1311:43
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 20:14:03 2018

@author: wzy
"""
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    """
    加载文件
    :param fileName: 文件名
    :return:
    dataMat - 数据矩阵
    labelMat - 数据标签
    """
    # 特征个数
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat




def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    单层决策树分类函数
    :param dataMatrix: 数据矩阵
    :param dimen: 第dimen列，也就是第几个特征
    :param threshVal: 阈值
    :param threshIneq: 标志
    :return: 分类结果
    """
    # 初始化retArray为全1列向量
    retArray = np.ones((np.shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        # 如果小于阈值则赋值为-1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        # 如果大于阈值则赋值为-1
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """
    找到数据集上最佳的单层决策树
    :param dataArr: 数据矩阵
    :param classLabels: 数据标签
    :param D: D - 样本权重,每个样本权重相等 1/n
    :return:
    bestStump - 最佳单层决策树信息
    minError - 最小误差
    bestClassEst - 最佳的分类结果
    """

    # 输入数据转为矩阵(5, 2)
    dataMatrix = np.mat(dataArr)
    # 将标签矩阵进行转置(5, 1)
    labelMat = np.mat(classLabels).T
    # m=5, n=2
    m, n = np.shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    # (5, 1)全零列矩阵
    bestClasEst = np.mat(np.zeros((m, 1)))
    # 最小误差初始化为正无穷大inf
    minError = float('inf')
    # 遍历所有特征
    for i in range(n):
        # 找到(每列)特征中的最小值和最大值
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        # 计算步长
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            # 大于和小于的情况均遍历，lt:Less than  gt:greater than
            for inequal in ['lt', 'gt']:
                # 计算阈值
                threshVal = (rangeMin + float(j) * stepSize)
                # 计算分类结果
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                # 初始化误差矩阵
                errArr = np.mat(np.ones((m, 1)))
                # 分类正确的，赋值为0
                errArr[predictedVals == labelMat] = 0
                # 计算误差
                weightedError = D.T * errArr
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (
                i, threshVal, inequal, weightedError))
                # 找到误差最小的分类方式
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr, classLabels, numIt=60):
    """
    使用AdaBoost进行优化
    :param dataArr: 数据矩阵
    :param classLabels: 数据标签
    :param numIt: 最大迭代次数
    :return:
    weakClassArr - 存储单层决策树的list
    aggClassEsc - 训练的label
    """
    weakClassArr = []
    # 获取数据集的行数
    m = np.shape(dataArr)[0]
    # 样本权重，每个样本权重相等，即1/n
    D = np.mat(np.ones((m, 1)) / m)
    # 初始化为全零列
    aggClassEst = np.mat(np.zeros((m, 1)))
    # 迭代
    for i in range(numIt):
        # 构建单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print("D:", D.T)
        # 计算弱学习算法权重alpha，使error不等于0，因为分母不能为0
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        # 存储弱学习算法权重
        bestStump['alpha'] = alpha
        # 存储单层决策树
        weakClassArr.append(bestStump)
        # 打印最佳分类结果
        # print("classEst: ", classEst.T)
        # 计算e的指数项
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        # 计算递推公式的分子
        D = np.multiply(D, np.exp(expon))
        # 根据样本权重公式，更新样本权重
        D = D / D.sum()
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        # 以下为错误率累计计算
        aggClassEst += alpha * classEst
        # print("aggClassEst: ", aggClassEst.T)
        # 计算误差
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        # print("total error:", errorRate)
        if errorRate == 0.0:
            # 误差为0退出循环
            break
    return weakClassArr, aggClassEst



def adaClassify(datToClass, classifierArr):
    """
    AdaBoost分类函数
    :param datToClass: 待分类样例
    :param classifierArr: 训练好的分类器
    :return: 分类结果
    """
    dataMatrix = np.mat(datToClass)
    m = np.shape(dataMatrix)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        # 遍历所有分类器进行分类
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        # print(aggClassEst)
    return np.sign(aggClassEst)


if __name__ == '__main__':
    dataArr, LabelArr = loadDataSet('horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr)
    testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
    print(weakClassArr)
    predictions = adaClassify(dataArr, weakClassArr)
    errArr = np.mat(np.ones((len(dataArr), 1)))
    print('训练集的错误率:%.3f%%' % float(errArr[predictions != np.mat(LabelArr).T].sum() / len(dataArr) * 100))
    predictions = adaClassify(testArr, weakClassArr)
    errArr = np.mat(np.ones((len(testArr), 1)))
    print('测试集的错误率:%.3f%%' % float(errArr[predictions != np.mat(testLabelArr).T].sum() / len(testArr) * 100))
# 训练集的错误率:18.729%
# 测试集的错误率:19.403%

