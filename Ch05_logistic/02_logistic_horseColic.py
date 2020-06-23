# coding: utf-8
# Author: shelley
# 2020/5/1214:55

import numpy as np
import random



def sigmoid(inX):
    """
    sigmoid函数
    :param inX: 数据
    :return:
    """
    return 1.0/(1+np.exp(-inX))
    # 优化
    # res = []
    # print(inX.shape)
    # inX =  np.array(inX)

    # for x in inX:
    #     x = x[0]
    #     if x >= 0:
    #         res.append( 1.0 / (1 + np.exp(-x)))
    #     else:
    #         res.append(np.exp(x) / (1 + np.exp(x)))
    # return np.array(res).reshape((len(inX),1))


def gradAscent(dataMath, classLabels):
    """
    梯度上升法
    :param dataMath: 数据集
    :param classLabels: 数据标签
    :return:
    weights.getA() - 求得的权重数组（最优参数）
    weights_array - 每次更新的回归系数
    """
    # 转换成numpy的mat(矩阵)
    dataMatrix = np.mat(dataMath)
    # 转换成numpy的mat(矩阵)并进行转置
    labelMat = np.mat(classLabels).transpose()
    # print(labelMat.shape) # (299,1)
    # 返回dataMatrix的大小，m为行数，n为列数
    m, n = np.shape(dataMatrix)
    # 移动步长，也就是学习效率，控制更新的幅度
    alpha = 0.01
    # 最大迭代次数
    maxCycles = 500
    weights = np.ones((n, 1))  # (3,1)
    for k in range(maxCycles):
        # 梯度上升矢量化公式
        # print(dataMatrix * weights)
        h = sigmoid(dataMatrix * weights)  # (n_samples,3)(3,1)=

        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    # 将矩阵转换为数组，返回权重数组
    # mat.getA()将自身矩阵变量转化为ndarray类型变量
    return weights.getA()


def stocGradAscent0(dataMatrix, classLabels, numIter=150):
    """
    改进的随机梯度上升法
    :param dataMatrix: 数据数组
    :param classLabels: 数据标签
    :param numIter: 迭代次数
    :return: 求得的回归系数数组（最优参数）
    """
    # 返回dataMatrix的大小，m为行数，n为列数
    m, n = np.shape(dataMatrix)
    # 参数初始化
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 每次都降低alpha的大小
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机选择样本
            randIndex = int(random.uniform(0, len(dataIndex)))
            # 随机选择一个样本计算h
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            # 计算误差
            error = classLabels[randIndex] - h
            # 更新回归系数
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 删除已使用的样本
            del (dataIndex[randIndex])
    # 返回
    return weights

def classifyVector(inX, weights):
    """
    分类函数
    :param inX: 特征向量
    :param weights: 回归系数
    :return: 分类结果
    """
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    """
    用python写的Logistic分类器做预测
    :return:
    """
    # 打开训练集
    frTrain = open('horseColicTraining.txt')
    # 打开测试集
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        # trainingLabels.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    # 使用改进的随机上升梯度训练
    # trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    # print(trainWeights.shape)
    # trainWeights = np.reshape(trainWeights, newshape=(21,1))
    # 使用上升梯度训练
    trainWeights = gradAscent(np.array(trainingSet), trainingLabels)
    print(trainWeights.shape)
    errorCount = 0
    numTestVect = 0.0
    for line in frTest.readlines():
        numTestVect += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        # if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
        pred_label = int(classifyVector(np.array(lineArr), trainWeights[:, 0]))
        if pred_label != int(currLine[-1]):
            errorCount += 1
    # 错误概率计算
    errorRate = (float(errorCount) / numTestVect) * 100
    print("测试集错误率为：%.5f%%" % errorRate)  # 28.35821%


if __name__ == '__main__':
    colicTest()
